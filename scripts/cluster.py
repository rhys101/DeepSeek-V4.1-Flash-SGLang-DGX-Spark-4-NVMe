#!/usr/bin/env python3
"""Stage and operate only the named four-Spark NVMe experiment from rank zero."""
import argparse
import concurrent.futures
import datetime
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))
from configuration import engine_args, environment, load


def mapped(c, fn):
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        return list(pool.map(fn, c['workers']))


def call(c, node, command, **kwargs):
    argv = ['bash', '-c', command] if node['rank'] == 0 else [
        'ssh', '-b', c['head_ip'], '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
        node['host'].split('@')[0] + '@' + node['ip'], command]
    return subprocess.run(argv, check=True, **kwargs)


def read(c, node, args, timeout=90):
    return call(c, node, shlex.join(args), capture_output=True, text=True, timeout=timeout).stdout


def require_head(c):
    data = json.loads(subprocess.check_output(['ip', '-j', 'addr']))
    ips = {address['local'] for interface in data for address in interface.get('addr_info', [])}
    if c['head_ip'] not in ips:
        raise RuntimeError('Run cluster operations on the configured rank-zero Spark')
    for node in c['workers'][1:]:
        route = json.loads(subprocess.check_output(['ip', '-j', 'route', 'get', node['ip'], 'from', c['head_ip']]))[0]
        if route['dev'] != c['fabric_interface']:
            raise RuntimeError('A peer route leaves the configured fabric')


def name(c, rank):
    return f"sglang4-{c['name']}-r{rank}"


def image_check(c, node):
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', c.get('expected_image_id') or ''):
        raise RuntimeError('Record the actual pinned-base image ID before staging')
    actual = read(c, node, ['docker', 'image', 'inspect', '--format', '{{.Id}}', c['image']]).strip()
    if actual != c['expected_image_id']:
        raise RuntimeError(f"Image identity differs on rank {node['rank']}")


def idle(c, node):
    ids = read(c, node, ['docker', 'ps', '-q']).split()
    if ids:
        containers = json.loads(read(c, node, ['docker', 'inspect', *ids]))
        if any(x['HostConfig'].get('DeviceRequests') for x in containers):
            raise RuntimeError(f"Another GPU container is running on rank {node['rank']}")
    if read(c, node, ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']).strip():
        raise RuntimeError(f"Another GPU process is running on rank {node['rank']}")


def source_hashes():
    files = {}
    for folder in ['adapter', 'runtime', 'patches', 'scripts', 'validation', 'bench', 'vendor', 'tests']:
        for path in (ROOT / folder).rglob('*'):
            if path.is_file() and not any(x in path.parts for x in ['__pycache__', 'node_modules']) and path.suffix not in ['.so', '.pyc']:
                files[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for filename in ['versions.lock.json', 'LICENSE', 'LICENSE.sglang', 'LICENSE.upstream-MIT', 'LICENSE.benchmark-MIT', 'NOTICE', 'README.md']:
        files[filename] = hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
    return files


def stage(c):
    manifest = json.loads((ROOT / 'patches/manifest.json').read_text())
    files = source_hashes()
    for relative, hashes in manifest.items():
        if files['patches/source/' + relative] != hashes['after_sha256']:
            raise RuntimeError(f'Overlay hash mismatch: {relative}')
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode='w:gz') as tar:
        for filename in sorted(files):
            tar.add(ROOT / filename, arcname=filename)
    archive_bytes = archive.getvalue()
    def one(node):
        image_check(c, node)
        # All originals are checked before any new staged source is installed.
        code = 'import hashlib,json,pathlib\nmanifest=' + repr(manifest) + '\n'
        code += "root=pathlib.Path('/sgl-workspace/sglang')\n"
        code += "for name,entry in manifest.items():\n p=root/name\n actual=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None\n assert actual in (entry['before_sha256'],entry['after_sha256']),(name,actual)\nprint('BASE_SOURCE_PASS')\n"
        output = read(c, node, ['docker', 'run', '--rm', '--network', 'none', '--entrypoint', 'python3', c['image'], '-S', '-c', code])
        if 'BASE_SOURCE_PASS' not in output:
            raise RuntimeError('Base source verification did not complete')
        remote = c['run_dir']
        setup = f"from pathlib import Path\np=Path({remote!r});p.mkdir(parents=True,exist_ok=False)\n(p/'kit').mkdir();(p/'state').mkdir();(p/'cache').mkdir()\nPath({c['engram_dir']!r}).mkdir(parents=True,exist_ok=True)\n"
        call(c, node, 'python3 -', input=setup, text=True, timeout=30)
        call(c, node, shlex.join(['tar', '-xzf', '-', '-C', remote + '/kit']), input=archive_bytes, timeout=120)
        code = f"from pathlib import Path\nimport json,tarfile,hashlib\np=Path({remote!r})/'kit'\n"
        code += f"files={files!r}\nfor name,want in files.items():\n assert hashlib.sha256((p/name).read_bytes()).hexdigest()==want,name\n"
        code += "(p/'configs').mkdir();(p/'b12x').mkdir()\n"
        code += f"(p/'configs/cluster.json').write_text({json.dumps(c, indent=2)!r}+'\\n')\n(p/'stage-source-manifest.json').write_text({json.dumps(files, indent=2)!r}+'\\n')\n"
        code += "bundle=json.loads((p/'vendor/b12x-source-manifest.json').read_text())\n"
        code += "with tarfile.open(p/'vendor/b12x-source.tar.gz') as tar:\n assert set(tar.getnames())==set(bundle)\n for name,want in bundle.items():\n  content=tar.extractfile(name).read();assert hashlib.sha256(content).hexdigest()==want\n  target=p/'b12x'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)\n"
        call(c, node, 'python3 -', input=code, text=True, timeout=90)
        build = ['docker', 'run', '--rm', '--network', 'none', '--mount', f'type=bind,src={remote}/kit,dst=/opt/sglang4',
                 '--entrypoint', 'g++', c['image'], '-O2', '-Wall', '-Wextra', '-Werror', '-std=c++17', '-shared', '-fPIC', '-pthread',
                 '/opt/sglang4/adapter/row_store.cpp', '-o', '/opt/sglang4/adapter/librow_store.so']
        call(c, node, shlex.join(build), timeout=120)
        library_hash = read(c, node, ['sha256sum', remote + '/kit/adapter/librow_store.so']).split()[0]
        return dict(rank=node['rank'], image=c['expected_image_id'], source_files=len(files), row_store_library_sha256=library_hash)
    rows = mapped(c, one)
    if len({row['row_store_library_sha256'] for row in rows}) != 1:
        raise RuntimeError('Row-store builds differ across ranks')
    state = Path(c['run_dir']) / 'state'
    (state / 'stage.json').write_text(json.dumps(dict(status='PASS', ranks=rows), indent=2) + '\n')
    print(json.dumps(dict(status='STAGED', ranks=rows)), flush=True)


def command(c, rank, gpu=True):
    d = c['run_dir']
    cmd = ['docker', 'run', '--pull', 'never', '--network', 'host', '--ipc', 'host',
           '--cap-add', 'IPC_LOCK', '--ulimit', 'memlock=-1', '--ulimit', 'nofile=1048576:1048576',
           '--mount', f"type=bind,src={c['model_store']},dst=/models,readonly",
           '--mount', f"type=bind,src={c['engram_dir']},dst=/engram,readonly",
           '--mount', f'type=bind,src={d}/kit,dst=/opt/sglang4,readonly',
           '--mount', f'type=bind,src={d}/kit/configs/cluster.json,dst=/config/profile.json,readonly',
           '--mount', f'type=bind,src={d}/cache,dst=/cache',
           '--mount', f'type=bind,src={d}/state,dst=/state']
    if gpu:
        cmd += ['--gpus', 'all', '--device', '/dev/infiniband']
    for relative in json.loads((ROOT / 'patches/manifest.json').read_text()):
        cmd += ['--mount', f'type=bind,src={d}/kit/patches/source/{relative},dst=/sgl-workspace/sglang/{relative},readonly']
    for key, value in environment(c, rank).items():
        cmd += ['-e', key + '=' + value]
    return cmd


def pack(c):
    mapped(c, lambda node: idle(c, node))
    def one(node):
        image_check(c, node)
        cmd = command(c, node['rank'], gpu=False)
        old = f"type=bind,src={c['engram_dir']},dst=/engram,readonly"
        cmd[cmd.index(old)] = f"type=bind,src={c['engram_dir']},dst=/engram"
        cmd += ['--name', f"sglang4-pack-{c['name']}-r{node['rank']}", '--entrypoint', 'python3', c['image'],
                '/opt/sglang4/scripts/pack-verified.py', '--model', '/models/' + c['model_subpath'],
                '--out', '/engram', '--rank', str(node['rank']), '--tp', '4']
        log = Path(c['run_dir']) / 'state' / f"pack-r{node['rank']}.log"
        with log.open('w') as output:
            call(c, node, shlex.join(cmd), stdout=output, stderr=subprocess.STDOUT, timeout=1800)
        result = json.loads(read(c, node, ['cat', c['engram_dir'] + f"/manifest-r{node['rank']}of4.json"]))
        return dict(rank=node['rank'], result=result)
    rows = mapped(c, one)
    (Path(c['run_dir']) / 'state/pack.json').write_text(json.dumps(dict(status='PASS', ranks=rows), indent=2) + '\n')
    print('TP4_PACK_VERIFIED', flush=True)


def status(c):
    def one(node):
        code = "import json,pathlib,subprocess\n"
        code += f"d=json.loads(subprocess.check_output(['docker','inspect',{name(c,node['rank'])!r}]))[0]\n"
        code += "mem={line.split(':')[0]:int(line.split()[1])/1024**2 for line in pathlib.Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','SwapTotal:','SwapFree:'))}\n"
        code += f"assert d['Image']=={c['expected_image_id']!r}\n"
        code += f"print(json.dumps(dict(rank={node['rank']},state=d['State'],restarts=d['RestartCount'],available_gib=mem['MemAvailable'],swap_used_gib=mem['SwapTotal']-mem['SwapFree'])))\n"
        return json.loads(call(c, node, 'python3 -', input=code, capture_output=True, text=True, timeout=40).stdout)
    return mapped(c, one)


def stop(c):
    mapped(c, lambda node: read(c, node, ['docker', 'stop', '-t', '30', name(c, node['rank'])], timeout=60))


def layout(c, node):
    result = call(c, node, shlex.join(['docker', 'logs', name(c, node['rank'])]), capture_output=True, text=True, timeout=45)
    text = result.stdout + '\n' + result.stderr
    records = [json.loads(line.split('DSPARK_MOE_LAYOUT ', 1)[1]) for line in text.splitlines() if 'DSPARK_MOE_LAYOUT ' in line]
    if not records:
        raise RuntimeError('No target/draft expert layout receipt')
    actual = records[-1]
    for role, count, experts in [('target', 40, 96), ('draft', 3, 32)]:
        expected = dict(method='Mxfp4FlashinferCutlassMoEMethod', local_experts=experts,
                        intermediate=2304, ep_size=4, ep_rank=node['rank'], tp_size=1, tp_rank=0,
                        w13_shape=[experts, 4608, 2560], w2_shape=[experts, 5120, 1152])
        if len(actual[role]) != count or any(value != expected for value in actual[role].values()):
            raise RuntimeError(f'Unexpected {role} expert layout')
    engram = [json.loads(line.split('NVME_ENGRAM_READY ', 1)[1]) for line in text.splitlines() if 'NVME_ENGRAM_READY ' in line]
    if len(engram) != 2 or {x['layer'] for x in engram} != {1, 14}:
        raise RuntimeError('Expected exactly two packed NVMe Engram tables')
    for record in engram:
        if record['rank'] != node['rank'] or record['tp_size'] != 4 or not record['packed'] or not record['empty_weight_parameters'] or record['cache_bytes'] or record['scale_bytes']:
            raise RuntimeError('Unexpected Engram ownership/storage configuration')
    roce = [json.loads(line.split('ROCE_TP4_READY ', 1)[1]) for line in text.splitlines() if 'ROCE_TP4_READY ' in line]
    if c['roce_enabled']:
        if len(roce) != 1 or roce[0]['world_size'] != 4 or roce[0]['rank'] != node['rank'] or 'ROCE_TP4_ROUTE ' not in text:
            raise RuntimeError('TP4 RoCEnante was not activated on this rank')
    elif roce:
        raise RuntimeError('The NCCL control unexpectedly activated RoCEnante')
    return dict(rank=node['rank'], layout=actual, engram=engram, roce=roce)


def serve(c, restart=False):
    mapped(c, lambda node: image_check(c, node))
    mapped(c, lambda node: idle(c, node))
    state_dir = Path(c['run_dir']) / 'state'
    started = []
    samples = []
    try:
        for node in c['workers'][1:] + c['workers'][:1]:
            cmd = ['docker', 'start', name(c, node['rank'])] if restart else command(c, node['rank']) + [
                '-d', '--name', name(c, node['rank']), '--entrypoint', 'python3', c['image'],
                '/opt/sglang4/runtime/launch.py', '--rank', str(node['rank'])]
            read(c, node, cmd)
            started.append(node)
        deadline = time.monotonic() + 2400
        host = '127.0.0.1' if c['api_host'] == '0.0.0.0' else c['api_host']
        base = f"http://{host}:{c['api_port']}"
        while time.monotonic() < deadline:
            ranks = status(c)
            row = dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(), ranks=ranks)
            samples.append(row)
            (state_dir / 'startup-samples.json').write_text(json.dumps(samples, indent=2) + '\n')
            minimum = min(x['available_gib'] for x in ranks)
            if any(not x['state']['Running'] or x['state']['OOMKilled'] or x['restarts'] for x in ranks) or minimum < c['minimum_available_gib']:
                raise RuntimeError(f'Container health or OS reserve failed: {ranks}')
            try:
                with urllib.request.urlopen(base + '/health', timeout=3) as response:
                    ready = response.status == 200
            except (OSError, TimeoutError):
                ready = False
            print(json.dumps(dict(phase='loading', minimum_available_gib=minimum, observed=row['time'])), flush=True)
            if ready:
                with urllib.request.urlopen(base + '/server_info', timeout=20) as response:
                    info = json.load(response)
                expected = dict(tp_size=4, ep_size=4, context_length=c['context_length'], max_total_tokens=c['max_total_tokens'],
                                max_running_requests=8, speculative_dspark_block_size=5, chunked_prefill_size=2048,
                                moe_runner_backend='flashinfer_mxfp4', speculative_moe_runner_backend='flashinfer_mxfp4')
                if any(info.get(key) != value for key, value in expected.items()):
                    raise RuntimeError('Resolved server configuration differs')
                layouts = mapped(c, lambda node: layout(c, node))
                (state_dir / 'selected-layout.json').write_text(json.dumps(layouts, indent=2) + '\n')
                (state_dir / 'server-info.json').write_text(json.dumps(info, indent=2) + '\n')
                (state_dir / 'boot-state.json').write_text(json.dumps(dict(status='READY', observed=row['time'], minimum_available_gib=minimum), indent=2) + '\n')
                print('TP4_READY', flush=True)
                return
            time.sleep(3)
        raise TimeoutError('TP4 startup exceeded forty minutes')
    except BaseException as exc:
        (state_dir / 'boot-state.json').write_text(json.dumps(dict(status='FAILED', error=repr(exc)), indent=2) + '\n')
        for node in started:
            try:
                read(c, node, ['docker', 'stop', '-t', '15', name(c, node['rank'])], timeout=40)
            except Exception:
                pass
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['stage', 'pack', 'serve', 'start', 'stop', 'status', 'dry-run'])
    p.add_argument('--config', default=str(ROOT / 'configs/cluster.local.json'))
    a = p.parse_args()
    c = load(a.config)
    if a.action == 'dry-run':
        print(json.dumps([dict(rank=n['rank'], arguments=engine_args(c, n['rank']), command=command(c, n['rank'])) for n in c['workers']], indent=2))
        return
    require_head(c)
    if a.action == 'stage': stage(c)
    elif a.action == 'pack': pack(c)
    elif a.action in ['serve', 'start']: serve(c, restart=a.action == 'start')
    elif a.action == 'stop': stop(c)
    else: print(json.dumps(status(c), indent=2))


if __name__ == '__main__':
    main()
