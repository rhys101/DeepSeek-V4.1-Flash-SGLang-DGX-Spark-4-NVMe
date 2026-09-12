#!/usr/bin/env python3
"""Watch OS memory and container health without issuing GPU queries during timing."""
import argparse
import json
from pathlib import Path
import time
import cluster

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--config', required=True)
p.add_argument('--label', required=True)
p.add_argument('--seconds', type=int, default=5400)
a = p.parse_args()
assert a.label.replace('-', '').isalnum() and 1 <= a.seconds <= 7200
c = cluster.load(a.config)
cluster.require_head(c)
root = Path(c['run_dir']) / 'state'
record = root / f'{a.label}-guard.json'
samples = root / f'{a.label}-guard-samples.jsonl'
stopfile = root / f'{a.label}-guard-stop'
assert not any(x.exists() for x in [record, samples, stopfile])
started = time.time()
minimum, maximum_swap, initial_swap, count = {}, {}, {}, 0


def save(status, **extra):
    result = dict(status=status, started=started, time=time.time(), sample_count=count,
                  minimum_available_gib_by_rank=minimum, initial_swap_used_gib_by_rank=initial_swap,
                  maximum_swap_used_gib_by_rank=maximum_swap, minimum_reserve_gib=c['minimum_available_gib'],
                  method='OS /proc/meminfo and docker inspect only; no GPU telemetry queries', **extra)
    temporary = record.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(record)


try:
    while not stopfile.exists():
        assert time.time() < started + a.seconds, 'Benchmark exceeded the guard deadline'
        rows = cluster.status(c)
        count += 1
        with samples.open('a') as f:
            f.write(json.dumps(dict(time=time.time(), ranks=rows)) + '\n')
        for row in rows:
            key = str(row['rank'])
            minimum[key] = min(minimum.get(key, float('inf')), row['available_gib'])
            maximum_swap[key] = max(maximum_swap.get(key, 0), row['swap_used_gib'])
            initial_swap.setdefault(key, row['swap_used_gib'])
        assert len(rows) == 4 and all(row['state']['Running'] and not row['state']['OOMKilled'] and row['restarts'] == 0 for row in rows), rows
        assert min(row['available_gib'] for row in rows) >= c['minimum_available_gib'], rows
        save('RUNNING')
        time.sleep(2)
    save('COMPLETE')
except BaseException as error:
    save('FAILED', error=repr(error))
    cluster.stop(c)
    raise
