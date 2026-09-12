"""Count target-graph collectives in the measured static TP4 variants."""
import argparse
import collections
import gzip
import hashlib
import json
import statistics
from pathlib import Path

p = argparse.ArgumentParser(); p.add_argument('--out', required=True); a = p.parse_args()
results = Path(__file__).resolve().parent.parent.parent / 'results'
lab = results / 'trace-iteration-02'
old = results / 'trace-iteration-01'
traces = [
    ('k5-512k', old/'traces-baseline/rank0/baseline-b-coding-c8/baseline-b-coding-c8-TP-0-EP-0.trace.json.gz', 48, 524288),
    ('k7-512k', old/'traces-k7/rank0/k7-b-coding-c8/k7-b-coding-c8-TP-0-EP-0.trace.json.gz', 64, 524288),
    ('k7-1m', lab/'traces-k7roce1m/rank0/k7roce1m-coding-c8/k7roce1m-coding-c8-TP-0-EP-0.trace.json.gz', 64, 1048576),
]
rows = []
for label, path, tokens, capacity in traces:
    events = json.load(gzip.open(path, 'rt'))['traceEvents']
    groups = collections.defaultdict(list)
    for event in events:
        if event.get('ph') == 'X' and event.get('cat') == 'kernel':
            groups[event['args'].get('correlation')].append(event)
    targets = [v for v in groups.values() if sum('GroupProblemShape' in e['name'] for e in v) == 80]
    assert len(targets) == 8
    targets.sort(key=lambda v: min(e['ts'] for e in v))
    result = []
    for group in targets:
        nccl = [e for e in group if 'nccl' in e['name'].lower() and 'allreduce' in e['name'].lower()]
        roce = [e for e in group if 'b12xcommroce' in e['name'].lower()]
        result.append(dict(nccl_allreduce_count=len(nccl), nccl_allreduce_ms=sum(e['dur'] for e in nccl)/1000,
                           roce_kernel_count=len(roce), roce_ms=sum(e['dur'] for e in roce)/1000))
    row = dict(label=label, trace=str(path.relative_to(lab.parent)), trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               max_hidden_allreduce_bytes=tokens * 5120 * 2, roce_capacity_bytes=capacity,
               target_graphs=result, means={k: statistics.mean(v[k] for v in result) for k in result[0]})
    rows.append(row)
    print(label, row['means'])
Path(a.out).write_text(json.dumps(dict(
    method='All eight C8 rank0 target graphs, identified by GPU correlation IDs and 80 expert GEMMs. Kernel durations include peer waiting and may overlap; streaming benchmark measurements determine throughput. The first two source traces are published with iteration 1.',
    rows=rows), indent=2) + '\n')
