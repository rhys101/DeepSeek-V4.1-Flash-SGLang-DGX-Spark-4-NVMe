"""Exercise exact packed reads, nonzero ownership offsets, page edges and fatal I/O."""
import argparse
import concurrent.futures
import ctypes as C
import json
import os
from pathlib import Path
import random
import resource
import signal
import struct
import subprocess
import sys
import tempfile

U, P = C.c_uint64, C.c_void_p
LIB = Path(__file__).resolve().parents[1] / 'adapter/librow_store.so'


class Work(C.Structure):
    _fields_ = [('store', P), ('ids', P), ('weights', P), ('scales', P), ('count', U)]


def library():
    lib = C.CDLL(str(LIB))
    lib.row_store_open.argtypes = [C.c_char_p, U, U, U, U]
    lib.row_store_open.restype = P
    lib.row_store_range.argtypes = [P, U, U]
    lib.row_store_attach_packed.argtypes = [P, C.c_char_p, U]
    lib.row_store_attach_packed.restype = C.c_int
    lib.row_store_lookup.argtypes = [C.POINTER(Work)]
    lib.row_store_stats.argtypes = [P, C.POINTER(U)]
    lib.row_store_close.argtypes = [P]
    return lib


def fixture(root, rank, rows=4099):
    rng = random.Random(491)
    weights = rng.randbytes(rows * 256)
    scales = rng.randbytes(rows * 8)
    original = root / 'original.bin'
    offset = 777
    original.write_bytes(bytes(offset) + weights + scales)
    lo, hi = rows * rank // 4, rows * (rank + 1) // 4
    packed = root / f'packed-r{rank}.bin'
    header = bytearray(4096)
    struct.pack_into('<6Q', header, 0, 0x31344e4531565344, 1, lo, hi, rows, 264)
    packed.write_bytes(header + b''.join(weights[i*256:(i+1)*256] + scales[i*8:(i+1)*8] for i in range(lo, hi)))
    return original, packed, lo, hi, rows, offset, weights, scales


def run(fault=None):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.environ.update(OFFLOAD_MODE='nvme', DSV41_RESIDENT_SCALES='0', DSV41_IO_THREADS='16')
    lib = library()
    cases = []
    for rank in range(4):
        with tempfile.TemporaryDirectory() as tmp:
            original, packed, lo, hi, rows, offset, weights, scales = fixture(Path(tmp), rank)
            store = lib.row_store_open(str(original).encode(), rows, offset, offset + len(weights), 0)
            assert store
            lib.row_store_range(store, lo, hi)
            assert lib.row_store_attach_packed(store, str(packed).encode(), 1) == 1
            if fault == 'truncate':
                with packed.open('r+b') as stream:
                    stream.truncate(4096)
            def check(seed):
                rng = random.Random(seed)
                ids = [0, rows-1, lo, lo+1, hi-2, hi-1] + [rng.randrange(rows) for _ in range(513)]
                if fault == 'bounds': ids[0] = rows
                indices = (C.c_int64 * len(ids))(*ids)
                w, s = C.create_string_buffer(len(ids) * 256), C.create_string_buffer(len(ids) * 8)
                job = Work(store, C.addressof(indices), C.addressof(w), C.addressof(s), len(ids))
                lib.row_store_lookup(C.byref(job))
                expected_w = b''.join(weights[i*256:(i+1)*256] if lo <= i < hi else bytes(256) for i in ids)
                expected_s = b''.join(scales[i*8:(i+1)*8] if lo <= i < hi else bytes(8) for i in ids)
                assert w.raw == expected_w and s.raw == expected_s
                return sum(lo <= i < hi for i in ids)
            with concurrent.futures.ThreadPoolExecutor(4) as pool:
                owned_reads = sum(pool.map(check, range(12)))
            stats = (U * 9)()
            lib.row_store_stats(store, stats)
            assert stats[0] == 0 and stats[1] == owned_reads and stats[2] == owned_reads
            assert stats[3] == 0 and stats[5] == 0 and stats[8] == 1
            lib.row_store_close(store)
            cases.append(dict(rank=rank, owned_range=[lo,hi], owned_reads=owned_reads, seeds=12,
                              concurrent_callers=4, exact=True, packed=True, cache_bytes=0, scale_bytes=0))
    return cases


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--fault', choices=['truncate', 'bounds'])
    a = p.parse_args()
    cases = run(a.fault)
    if a.fault:
        raise RuntimeError('The injected fault did not abort the reader')
    faults = []
    for fault in ['truncate', 'bounds']:
        proc = subprocess.run([sys.executable, __file__, '--fault', fault], capture_output=True, text=True, timeout=30)
        assert proc.returncode == -signal.SIGABRT, (fault, proc.returncode, proc.stderr)
        assert 'Engram retrieval failed:' in proc.stderr
        faults.append(dict(fault=fault, signal='SIGABRT', stopped_before_return=True))
    print(json.dumps(dict(status='PASS', cases=cases, faults=faults)))
