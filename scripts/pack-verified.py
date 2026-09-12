"""Pack two exact TP-owned Engram shards, then verify bytes against the checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import struct

import pack_engram


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(16 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(root, out, layer, rank, tp):
    source, rows, weight_offset, scale_offset = pack_engram.tensor_span(root, layer)
    lo, hi = rows * rank // tp, rows * (rank + 1) // tp
    target = out / f'engram-l{layer}-r{rank}of{tp}.bin'
    expected_header = (pack_engram.MAGIC, layer, lo, hi, rows, 264)
    expected_size = 4096 + (hi - lo) * 264
    if target.stat().st_size != expected_size:
        raise RuntimeError('Packed shard has the wrong length')
    seed = layer * 1000 + rank
    rng = random.Random(seed)
    # Boundary rows include page-crossing records and both ownership edges.
    candidates = [lo, lo + 1, hi - 2, hi - 1]
    candidates += [lo + i for i in [14, 15, 16, 30, 31, 32, 510, 511, 512]]
    candidates += [rng.randrange(lo, hi) for _ in range(1024)]
    ids = sorted({i for i in candidates if lo <= i < hi})
    with source.open('rb', buffering=0) as original, target.open('rb', buffering=0) as packed:
        if struct.unpack('<6Q', packed.read(48)) != expected_header:
            raise RuntimeError('Packed shard header does not match the owned range')
        for row in ids:
            original.seek(weight_offset + row * 256)
            weights = original.read(256)
            original.seek(scale_offset + row * 8)
            scales = original.read(8)
            packed.seek(4096 + (row - lo) * 264)
            if len(weights) != 256 or len(scales) != 8 or packed.read(264) != weights + scales:
                raise RuntimeError(f'Packed row differs from checkpoint at layer {layer}, row {row}')
    return dict(layer=layer, rank=rank, tp=tp, row_start=lo, row_end=hi, total_rows=rows,
                packed_file=target.name, bytes=expected_size, sha256=digest(target),
                sampled_rows=len(ids), sample_seed=seed, sampled_rows_exact=True,
                source_shard=source.name, source_weight_offset=weight_offset, source_scale_offset=scale_offset)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--rank', type=int, required=True)
    p.add_argument('--tp', type=int, default=4, choices=[4])
    a = p.parse_args()
    if a.rank not in range(a.tp):
        raise ValueError('Invalid TP4 rank')
    root, out = Path(a.model), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / f'manifest-r{a.rank}of{a.tp}.json'
    checkpoint_config = digest(root / 'config.json')
    checkpoint_index = digest(root / 'model.safetensors.index.json')
    old = json.loads(manifest.read_text()) if manifest.exists() else None
    if old:
        if old['checkpoint_config_sha256'] != checkpoint_config or old['checkpoint_index_sha256'] != checkpoint_index:
            raise RuntimeError('The checkpoint identity differs from the existing packed manifest')
        for record in old['layers']:
            target = out / record['packed_file']
            if target.stat().st_size != record['bytes'] or digest(target) != record['sha256']:
                raise RuntimeError('An existing packed shard differs from its verified manifest')
    else:
        for layer in pack_engram.ENGRAM_LAYERS:
            target = out / f'engram-l{layer}-r{a.rank}of{a.tp}.bin'
            if target.exists() or target.with_suffix('.partial').exists():
                raise RuntimeError('Unmanifested packed data exists; choose a fresh output directory')
        need = sum(4096 + (pack_engram.tensor_span(root, layer)[1] * (a.rank + 1) // a.tp -
                           pack_engram.tensor_span(root, layer)[1] * a.rank // a.tp) * 264
                   for layer in pack_engram.ENGRAM_LAYERS)
        free = os.statvfs(out).f_bavail * os.statvfs(out).f_frsize
        if free < need + 8 * 1024**3:
            raise RuntimeError('Insufficient local disk space for packed shards plus 8 GiB reserve')
        for layer in pack_engram.ENGRAM_LAYERS:
            pack_engram.pack_layer(root, out, layer, a.rank, a.tp, 1 << 18)
    layers = [verify(root, out, layer, a.rank, a.tp) for layer in pack_engram.ENGRAM_LAYERS]
    record = dict(status='PASS', rank=a.rank, tp=a.tp, checkpoint_config_sha256=checkpoint_config,
                  checkpoint_index_sha256=checkpoint_index, layers=layers,
                  method='Exact source packing; full output SHA-256 and deterministic random/boundary sample comparison against original weight and scale bytes. No tensor requantization.')
    temp = manifest.with_suffix('.tmp')
    temp.write_text(json.dumps(record, indent=2) + '\n')
    temp.replace(manifest)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
