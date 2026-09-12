"""Real-checkpoint EP4/TP1 B12x versus current FlashInfer, on one idle GB10.

No serving overlays are imported or changed. This is a component qualification,
using the pinned, previously qualified W4A8-MX source and its clamp fixes.
"""
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

assert os.environ['B12X_W4A8_TINY_DECODE'] == '0'
import torch
import moe_weights as w
from b12x.moe import fused_moe
from b12x.moe._shared.kernels.reference import moe_reference_w4a8_mx

OUT = Path('/evidence')
DEVICE = torch.device('cuda:0')
INNER, REPLAYS = 16, 4
RESULT = dict(status='RUNNING', started=time.time(), shapes=[],
              gates=dict(relative_l2=0.06, cosine=0.998),
              geometry=dict(hidden=5120, intermediate=2304, ep=4, tp=1),
              method='Both graph arms include a 64 MiB cache flush before every operation. '
                     '16 operations per graph and 4 replays per event interval; '
                     'three changed routing seeds with ABBA ordering. Ratios are '
                     'FlashInfer/B12x, so greater than one favors B12x. '
                     'FlashInfer uses [16,36] at C1, matching the TP4 trace, '
                     'which also matches the TP4 C8 trace. '
                     'Synthetic activations and routes with original checkpoint partitions; '
                     'no serving throughput or full-model quality claim.',
              dependencies={n: importlib.metadata.version(n) for n in
                            ['torch', 'flashinfer-python', 'nvidia-cutlass-dsl']})
torch.set_num_threads(4)
torch.cuda.set_device(0)
torch.backends.cuda.matmul.allow_tf32 = False
assert torch.cuda.get_device_capability() == (12, 1)

def save():
    (OUT/'result.json').write_text(json.dumps(RESULT, indent=2)+'\n')

def hardware(name):
    (OUT/name).write_text(subprocess.check_output(['nvidia-smi', '-q'], text=True))

def metric(a, b):
    a, b = a.float(), b.float()
    assert bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
    n = float(b.norm())
    if n == 0:
        assert int(torch.count_nonzero(a)) == 0
        return dict(relative_l2=0., cosine=1., reference_norm=0.)
    row = dict(relative_l2=float((a-b).norm())/n,
               cosine=float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)),
               reference_norm=n, actual_norm=float(a.norm()))
    assert row['relative_l2'] <= RESULT['gates']['relative_l2'] and row['cosine'] >= RESULT['gates']['cosine'], row
    return row

def qualify(prefix, total, topk, tp_rank, raw, prepared, experts, m):
    E, K, N = total//4, 5120, 2304
    pair = [16, 36]
    row = dict(prefix=prefix, tokens=m, tp_rank=tp_rank, flashinfer_pair=pair,
               checks=[], timings=[])
    RESULT['shapes'].append(row)
    x, ids, routes = w.inputs(m, total, topk, 0, 'mixed', 2, 2700+m)
    local = torch.where(ids < E, ids, -1)
    nonlocal_id = torch.full((), -1, dtype=ids.dtype, device=DEVICE)
    caps = fused_moe.Caps(max_tokens=m, num_topk=topk, device=DEVICE,
        weight_plan=experts.plan, core_token_counts=(m,), route_num_experts=0,
        route_logits_dtype=None, quant_mode='w4a8_mx',
        apply_router_weight_on_input=False, swiglu_limit=10.0, frozen=True)
    required = int(fused_moe.required_nbytes(caps))
    arena = torch.empty(required, dtype=torch.uint8, device=DEVICE)
    plan = fused_moe.plan(caps)
    bo, co = torch.empty_like(x), torch.empty_like(x)
    def baseline():
        xq, xsf = w.mxfp8_quantize(x, is_sf_swizzled_layout=True, alignment=32)
        w.cutlass_fused_moe(input=xq, token_selected_experts=ids, token_final_scales=routes,
            fc1_expert_weights=prepared['w13'].view(torch.int64),
            fc2_expert_weights=prepared['w2'].view(torch.int64),
            output_dtype=torch.bfloat16,
            quant_scales=[prepared['s13'].view(torch.int32), prepared['ones'],
                          prepared['s2'].view(torch.int32), prepared['ones']],
            input_sf=xsf, swiglu_limit=prepared['limits'], tp_size=1, tp_rank=tp_rank,
            ep_size=4, ep_rank=0, use_w4_group_scaling=False, use_mxfp8_act_scaling=True,
            activation_type=w.ActivationType.Swiglu, tune_max_num_tokens=1<<(m-1).bit_length(),
            output=bo, use_fused_finalize=False, profile_ids=pair)
    def candidate():
        # Standard dispatch maps global IDs to local IDs for B12x. Include that
        # GPU operation in its timing; FlashInfer consumes the global IDs directly.
        torch.where(ids < E, ids, nonlocal_id, out=local)
        fused_moe.run(binding=plan.bind(scratch=arena, a=x, experts=experts,
            topk_weights=routes, topk_ids=local, output=co, input_scales_static=True))
    for _ in range(3):
        baseline(); candidate()
    torch.cuda.synchronize()
    bg, cg = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(bg): baseline()
    with torch.cuda.graph(cg): candidate()
    pointers = [t.data_ptr() for t in [x, ids, routes, local, arena, bo, co]]
    for j, (mode, scale) in enumerate([('mixed', 2), ('local', 8), ('nonlocal', 2), ('mixed', 8)]):
        for dst, src in zip((x, ids, routes), w.inputs(m, total, topk, 0, mode, scale, 3700+m+j)):
            dst.copy_(src)
        oracle = moe_reference_w4a8_mx(x.float(), raw['w13'], raw['s13'], None,
            prepared['ones'], raw['w2'], raw['s2'], None, prepared['ones'],
            torch.where(ids < E, ids, -1), routes, E, K, N,
            activation='silu', swiglu_limit=10.0, w13_layout='w13')
        if mode == 'local' and m == 6:
            unclamped = moe_reference_w4a8_mx(x.float(), raw['w13'], raw['s13'], None,
                prepared['ones'], raw['w2'], raw['s2'], None, prepared['ones'],
                torch.where(ids < E, ids, -1), routes, E, K, N,
                activation='silu', swiglu_limit=None, w13_layout='w13')
            delta = float((oracle.float()-unclamped.float()).norm())/float(oracle.float().norm())
            assert delta > 0.001, ('Clamp fixture did not exercise the limit', delta)
            row['unclamped_relative_difference'] = delta
            del unclamped
        for replay in range(3):
            bo.fill_(float('nan')); co.fill_(float('nan')); arena.fill_(0xA5)
            before = torch.cuda.memory_allocated()
            bg.replay(); cg.replay(); torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == before
            assert pointers == [t.data_ptr() for t in [x, ids, routes, local, arena, bo, co]]
            row['checks'].append(dict(mode=mode, scale=scale, replay=replay,
                baseline_oracle=metric(bo, oracle), candidate_oracle=metric(co, oracle),
                candidate_baseline=metric(co, bo)))
    flush = torch.empty(64*1024*1024, dtype=torch.uint8, device=DEVICE)
    graphs = {}
    for name, invoke in [('flashinfer', baseline), ('b12x', candidate)]:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for i in range(INNER):
                flush.fill_(i); invoke()
        graphs[name] = g
    def timed(g):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(REPLAYS): g.replay()
        b.record(); b.synchronize()
        return a.elapsed_time(b)*1000/(INNER*REPLAYS)
    for seed in [4700+m, 5700+m, 6700+m]:
        hardware(f'hardware-{prefix}-tp{tp_rank}-m{m}-seed{seed}-before.txt')
        for dst, src in zip((x, ids, routes), w.inputs(m, total, topk, 0, 'mixed', 2, seed)):
            dst.copy_(src)
        for g in graphs.values(): g.replay()
        torch.cuda.synchronize()
        samples = dict(flashinfer=[], b12x=[])
        for _ in range(4):
            for name in ['flashinfer', 'b12x', 'b12x', 'flashinfer']:
                samples[name].append(timed(graphs[name]))
        med = {name: statistics.median(values) for name, values in samples.items()}
        row['timings'].append(dict(seed=seed, raw_us=samples, median_us=med,
                                  speedup=med['flashinfer']/med['b12x']))
        hardware(f'hardware-{prefix}-tp{tp_rank}-m{m}-seed{seed}-after.txt')
    for name, invoke in [('flashinfer', baseline), ('b12x', candidate)]:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            invoke(); torch.cuda.synchronize()
        prof.export_chrome_trace(str(OUT/f'kernels-{prefix}-m{m}-{name}.json'))
    row['arena_bytes'] = required
    save()
    print(json.dumps(dict(prefix=prefix, tokens=m, tp_rank=tp_rank,
        speedups=[v['speedup'] for v in row['timings']])), flush=True)

try:
    hardware('hardware-before.txt')
    # Target dominates per-step work. Check full TP4 target geometry at C1/C8.
    for prefix, total, topk, shapes, halves in [
            ('layers.0', 384, 6, [6, 48], [0])]:
        full = w.load_weights(prefix, 0, total)
        (OUT/'checkpoint-tensors.json').write_text(json.dumps(w.RESULT['checkpoint_tensors'], indent=2)+'\n')
        for tp_rank in halves:
            raw = full
            prepared = w.prepare(raw)
            owned = {k: v.clone() for k, v in raw.items()}
            weight_plan = fused_moe.plan_weights(quant_modes='w4a8_mx',
                source_format='fp4_e8m0_k32', activation='silu', params_dtype=torch.bfloat16,
                num_experts=total//4, hidden_size=5120, intermediate_size=2304,
                w13_layout='up_gate')
            experts = fused_moe.prepare_weights(plan=weight_plan, params_dtype=torch.bfloat16,
                w1_fp4=owned['w13'], w1_blockscale=owned['s13'],
                w1_global_scale=prepared['ones'], a1_gscale=prepared['ones'],
                w2_fp4=owned['w2'], w2_blockscale=owned['s2'],
                w2_global_scale=prepared['ones'], a2_gscale=prepared['ones'])
            for m in shapes: qualify(prefix, total, topk, tp_rank, raw, prepared, experts, m)
            del raw, prepared, owned, experts, weight_plan
            gc.collect(); torch.cuda.empty_cache()
        del full
        gc.collect(); torch.cuda.empty_cache()
    (OUT/'checkpoint-tensors.json').write_text(json.dumps(w.RESULT['checkpoint_tensors'], indent=2)+'\n')
    hardware('hardware-after.txt')
    RESULT.update(status='PASS', finished=time.time()); save()
    print('B12X_EP4_COMPONENT_PASS', flush=True)
except BaseException as exc:
    RESULT.update(status='FAILED', error=repr(exc)); save(); raise
