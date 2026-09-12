"""Compare real checkpoint rows and changing CUDA-graph lookups on four GPUs."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import struct
import types

import torch
import torch.distributed as dist

torch.set_num_threads(2)
torch.cuda.set_device(0)
dist.init_process_group('gloo', timeout=datetime.timedelta(seconds=120))
rank = dist.get_rank()
assert dist.get_world_size() == 4
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.distributed.device_communicators.pynccl import synchronize_and_check_roce
import sglang.srt.layers.engram as module
import engram_backend

root = Path('/models/DeepSeek-V4.1-Flash')
index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
# The storage/lookup implementation is real; this isolated probe supplies its TP state.
module.get_parallel = lambda: types.SimpleNamespace(tp_size=4, tp_rank=rank)
module.get_attention_dp_size = lambda: 1
engram_backend.install(module)
tp = GroupCoordinator(group_ranks=[list(range(4))], local_rank=0, torch_distributed_backend='nccl',
    use_pynccl=True, use_pymscclpp=False, use_custom_allreduce=False,
    use_torch_symm_mem_all_reduce=False, use_hpu_communicator=False,
    use_xpu_communicator=False, use_npu_communicator=False, group_name='tp')
module.tensor_model_parallel_all_reduce = tp.all_reduce
comm = tp.pynccl_comm
assert comm.roce is not None and comm.roce.world_size == 4


def table(layer):
    prefix = f'layers.{layer}.engram.embed.'
    path = root / index[prefix + 'weight']
    with path.open('rb') as source:
        length = struct.unpack('<Q', source.read(8))[0]
        header = json.loads(source.read(length))
    w, s = header[prefix + 'weight'], header[prefix + 'scale']
    assert w['dtype'] == 'F8_E4M3' and s['dtype'] == 'F8_E8M0'
    return path, w['shape'][0], 8 + length + w['data_offsets'][0], 8 + length + s['data_offsets'][0]


def ids_for(rows, count, seed):
    rng = random.Random(seed)
    edges = [0, rows - 1]
    for r in range(4):
        lo, hi = rows * r // 4, rows * (r + 1) // 4
        edges += [lo, lo + 1, hi - 2, hi - 1]
    return (edges + [rng.randrange(rows) for _ in range(count)])[:count]


def expected(spec, ids, owned=False):
    path, rows, woff, soff = spec
    weights, scales = bytearray(), bytearray()
    with path.open('rb', buffering=0) as source:
        for row in ids:
            if owned and not rows * rank // 4 <= row < rows * (rank + 1) // 4:
                weights += bytes(256)
                scales += bytes(8)
            else:
                source.seek(woff + row * 256)
                weights += source.read(256)
                source.seek(soff + row * 8)
                scales += source.read(8)
    w = torch.frombuffer(weights, dtype=torch.uint8).view(torch.float8_e4m3fn).float().reshape(len(ids), 8, 32)
    s = torch.frombuffer(scales, dtype=torch.uint8).view(torch.float8_e8m0fnu).float().reshape(len(ids), 8, 1)
    answer = (w * s).reshape(len(ids), 256).to(torch.bfloat16)
    assert torch.isfinite(answer).all()
    return answer


stream = torch.cuda.Stream()
eager_cases, graph_cases, embeddings = [], [], []
for layer in [1, 14]:
    spec = table(layer)
    rows = spec[1]
    emb = module.EngramEmbedding(rows, 256, layer)
    embeddings.append(emb)
    assert emb.weight.numel() == emb.scale.numel() == 0
    emb.weight.weight_loader(emb.weight, types.SimpleNamespace(shape=[rows, 256]))
    try:
        emb.weight.weight_loader(emb.weight, types.SimpleNamespace(shape=[rows - 1, 256]))
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid checkpoint shape was accepted')
    empty = emb(torch.empty(0, dtype=torch.int64, device='cuda'))
    assert empty.shape == (0, 256)
    del empty
    for count in [1, 24, 144, 576, 1152, 49152]:
        ids = ids_for(rows, count, 1700 + layer + count)
        inp = torch.tensor(ids, dtype=torch.int64, device='cuda')
        local = emb._owned_rows(inp)
        torch.cuda.synchronize()
        assert torch.equal(local.cpu(), expected(spec, ids, owned=True)), (rank, layer, count, 'owned')
        full = emb(inp)
        torch.cuda.synchronize()
        assert torch.equal(full.cpu(), expected(spec, ids)), (rank, layer, count, 'assembled')
        eager_cases.append(dict(layer=layer, count=count, owned_bit_exact=True, tp_assembled_bit_exact=True))
        del inp, local, full
    # Captured buffers stay fixed while fresh, different row IDs are copied in.
    for count in [144, 576, 1152]:
        ids = ids_for(rows, count, layer + count)
        inp = torch.tensor(ids, dtype=torch.int64, device='cuda')
        emb(inp)
        torch.cuda.synchronize()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with tp.graph_capture(stream=stream):
            with torch.cuda.graph(graph, stream=stream):
                output = emb(inp)
        ptrs = [inp.data_ptr(), output.data_ptr()]
        allocated = torch.cuda.memory_allocated()
        epoch = comm.roce.stats()['epoch']
        for trial in range(12):
            ids = ids_for(rows, count, 900000 + layer * 100 + trial)
            inp.copy_(torch.tensor(ids, dtype=torch.int64))
            graph.replay()
            done = torch.cuda.Event()
            done.record()
            synchronize_and_check_roce(done)
            assert torch.equal(output.cpu(), expected(spec, ids)), (rank, layer, count, trial)
            assert [inp.data_ptr(), output.data_ptr()] == ptrs
            assert torch.cuda.memory_allocated() == allocated
            operations = 1 if count * 256 * 2 <= 512 * 1024 else 0
            assert comm.roce.stats()['epoch'] == epoch + operations * (trial + 1)
        graph_cases.append(dict(layer=layer, count=count, changing_replays=12, bit_exact=True,
                                pointers_stable=True, allocated_bytes_stable=True,
                                roce_operations_per_replay=operations))
        del graph, output, inp
        torch.cuda.synchronize()

stores = [dict(layer=layer, **engram_backend._stats(store)) for store, layer in engram_backend._STORES]
assert len(stores) == 2 and all(s['packed'] and not s['cache_bytes'] and not s['scale_bytes'] for s in stores)
result = dict(status='PASS', rank=rank, world_size=4, eager=eager_cases, graphs=graph_cases,
              storage=stores, checkpoint_config_sha256=hashlib.sha256((root/'config.json').read_bytes()).hexdigest(),
              method='Real checkpoint bytes, original FP8/E8M0 dequantization, actual NVMe callback and EngramEmbedding.forward with an isolated TP4 GroupCoordinator; no complete model weights resident.')
print('NVME_GPU_RESULT ' + json.dumps(result), flush=True)
dist.barrier()
comm.roce.close()
comm.nccl.ncclCommDestroy(comm.comm)
tp.destroy()
dist.destroy_process_group()
