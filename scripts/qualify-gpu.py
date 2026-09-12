#!/usr/bin/env python3
"""Run four-rank transport or real-checkpoint NVMe qualification on idle GPUs."""
import argparse
import datetime
import json
from pathlib import Path
import shlex
import time
import cluster

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--config', required=True)
p.add_argument('--suite', choices=['roce-upstream', 'roce-integration', 'nvme'], required=True)
p.add_argument('--attempt', type=int, default=1)
a = p.parse_args()
c = cluster.load(a.config)
cluster.require_head(c)
cluster.mapped(c, lambda node: cluster.idle(c, node))
label = f"{c['name']}-{a.suite}-{a.attempt}"


def container(node):
    return f"sglang4-test-{label}-r{node['rank']}"


def inspect(node):
    code = "import json,pathlib,subprocess\n"
    code += f"d=json.loads(subprocess.check_output(['docker','inspect',{container(node)!r}]))[0]\n"
    code += "mem={line.split(':')[0]:int(line.split()[1])/1024**2 for line in pathlib.Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','SwapTotal:','SwapFree:'))}\n"
    code += f"print(json.dumps(dict(rank={node['rank']},state=d['State'],restarts=d['RestartCount'],available_gib=mem['MemAvailable'],swap_used_gib=mem['SwapTotal']-mem['SwapFree'])))\n"
    return json.loads(cluster.call(c, node, 'python3 -', input=code, capture_output=True, text=True, timeout=40).stdout)


state = Path(c['run_dir']) / 'state'
started, samples = [], []
try:
    for node in c['workers'][1:] + c['workers'][:1]:
        cmd = cluster.command(c, node['rank'])
        cmd += ['-d', '--name', container(node), '-e', 'SGLANG4_NVME_PROFILE=0',
                '--entrypoint', 'torchrun', c['image'], '--nnodes=4', '--nproc-per-node=1',
                f"--node-rank={node['rank']}", f"--master-addr={c['head_ip']}",
                f"--master-port={c['nccl_test_port']}"]
        if a.suite == 'roce-upstream':
            cmd += ['-m', 'pytest', '-x', '-q', '-rA',
                    '/opt/sglang4/validation/upstream-tests/test_roce_oneshot_gpu.py',
                    f"--junitxml=/state/{label}-r{node['rank']}.xml"]
        else:
            filename = 'check-nvme-gpu.py' if a.suite == 'nvme' else 'check-roce-integration.py'
            cmd += ['/opt/sglang4/validation/' + filename]
        cluster.read(c, node, cmd)
        started.append(node)
    deadline = time.monotonic() + 1200
    last = 0
    while time.monotonic() < deadline:
        rows = cluster.mapped(c, inspect)
        sample = dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(), ranks=rows)
        samples.append(sample)
        (state / f'{label}-samples.json').write_text(json.dumps(samples, indent=2) + '\n')
        if any(row['state']['OOMKilled'] or row['restarts'] or row['available_gib'] < c['minimum_available_gib'] or
               (not row['state']['Running'] and row['state']['ExitCode'] != 0) for row in rows):
            raise RuntimeError('Qualification process failed: ' + json.dumps(rows))
        if all(not row['state']['Running'] for row in rows):
            break
        if time.monotonic() - last > 20:
            print(json.dumps(dict(suite=a.suite, phase='running', minimum_available_gib=min(row['available_gib'] for row in rows))), flush=True)
            last = time.monotonic()
        time.sleep(3)
    else:
        raise TimeoutError('Qualification exceeded twenty minutes')
    results = []
    for node in c['workers']:
        r = cluster.call(c, node, shlex.join(['docker', 'logs', container(node)]), capture_output=True, text=True, timeout=40)
        logs = r.stdout + '\n' + r.stderr
        (state / f"{label}-r{node['rank']}.log").write_text(logs)
        if a.suite == 'roce-upstream':
            import xml.etree.ElementTree as ET
            xml = cluster.read(c, node, ['cat', f"{c['run_dir']}/state/{label}-r{node['rank']}.xml"])
            root = ET.fromstring(xml)
            cases = root.findall('.//testcase')
            skipped = [x for x in cases if x.find('skipped') is not None]
            assert not root.findall('.//failure') and not root.findall('.//error')
            assert len(cases) == 75 and len(skipped) == 1
            assert skipped[0].attrib['name'] == 'test_all_gather_matches_torch[shape5-1-dtype2]'
            assert skipped[0].find('skipped').attrib['message'] == "shard exceeds the runtime's all-gather capacity"
            result = dict(status='PASS', passed=len(cases)-len(skipped), skipped=[x.attrib['name'] for x in skipped])
            (state / f"{label}-r{node['rank']}.xml").write_text(xml)
        else:
            prefix = 'NVME_GPU_RESULT ' if a.suite == 'nvme' else 'ROCE_RESULT '
            matches = [json.loads(line.split(prefix, 1)[1]) for line in logs.splitlines() if prefix in line]
            assert len(matches) == 1 and matches[0]['status'] == 'PASS', (node['rank'], prefix)
            result = matches[0]
        results.append(dict(rank=node['rank'], result=result))
    (state / f'{label}-result.json').write_text(json.dumps(dict(status='PASS', suite=a.suite, results=results), indent=2) + '\n')
    print(json.dumps(dict(status='PASS', suite=a.suite, ranks=4)), flush=True)
except BaseException:
    for node in started:
        try:
            cluster.read(c, node, ['docker', 'stop', '-t', '10', container(node)], timeout=30)
        except Exception:
            pass
        try:
            r = cluster.call(c, node, shlex.join(['docker', 'logs', container(node)]), capture_output=True, text=True, timeout=30)
            (state / f"{label}-r{node['rank']}.log").write_text(r.stdout + '\n' + r.stderr)
        except Exception:
            pass
    raise
