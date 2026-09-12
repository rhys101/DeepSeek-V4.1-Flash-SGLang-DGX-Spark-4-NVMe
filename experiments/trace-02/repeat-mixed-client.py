"""Repeat every pinned workload at C1/C8 with identical warmed holdout prefixes."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

p = argparse.ArgumentParser()
p.add_argument('--base', required=True)
p.add_argument('--bench', required=True)
p.add_argument('--out', required=True)
p.add_argument('--label', required=True)
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=False)
assert hashlib.sha256(Path(a.bench).read_bytes()).hexdigest() == 'e0d6b2d25bd585d11fbdf39c2ddcdf7a4de8ab685af6bd42465e69f3ee6e80a8'
spec = importlib.util.spec_from_file_location('bench', a.bench)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
args = SimpleNamespace(base=a.base, model='deepseek-v41-flash')

def get(route):
    with urllib.request.urlopen(a.base.rstrip('/').removesuffix('/v1') + route, timeout=30) as r:
        return json.load(r)

def save(name, value):
    (out / name).write_text(json.dumps(value, indent=2) + '\n')

def idle():
    for _ in range(100):
        loads = get('/v1/loads')
        if all(r['num_running_reqs'] == r['num_waiting_reqs'] == 0 for r in loads['loads']):
            return loads
        time.sleep(.1)
    raise AssertionError(loads)

loads = idle()
info = get('/server_info')
assert (info['tp_size'], info['ep_size'], info['context_length'], info['max_total_tokens'], info['max_running_requests'], info['speculative_dspark_block_size']) == (4, 4, 1000000, 4000000, 8, 5)
save('server-info.json', info)
save('loads-before.json', loads)
cells = [(c, category, prompt, budget) for c in [1, 8] for category, prompt, budget in bench.CATEGORIES]
warm = []
rows = []
for trial in [0, 1, 2, 3]:
    for c, category, prompt, budget in cells if trial != 2 else list(reversed(cells)):
        row = bench.run_batch(args, c, category, prompt, budget, 'trace01-holdout')
        assert len(row['requests']) == c and all(r['completion_tokens'] > 0 and r['chars'] > 0 for r in row['requests'])
        row['trial'] = trial
        (warm if trial == 0 else rows).append(row)
        print(json.dumps(dict(trial=trial, c=c, category=category, decode=row['per_stream_tok_s'], aggregate=row['agg_tok_s'])), flush=True)
        save('results.json', dict(label=a.label, source_sha256=hashlib.sha256(Path(a.bench).read_bytes()).hexdigest(),
                                 method='All eight pinned categories and original budgets. Same trace01-holdout prefix on both runtimes. One excluded full warmup per category/concurrency; three measured waves at C1 and C8, with the complete order reversed in trial 2. No profiler or simulated acceptance.',
                                 excluded_warmups=warm, batches=rows))
assert len(rows) == 48
loads = idle()
save('loads-after.json', loads)
save('completion.json', dict(status='PASS', finished=datetime.datetime.now(datetime.timezone.utc).isoformat()))
print('MIXED_REPEAT_COMPLETE', flush=True)
