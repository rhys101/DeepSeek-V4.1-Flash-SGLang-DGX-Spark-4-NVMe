"""Verify the staged source and checkpoint identity, then launch one TP4 rank."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from configuration import engine_args, environment, load

p = argparse.ArgumentParser()
p.add_argument('--profile', default='/config/profile.json')
p.add_argument('--rank', type=int, required=True)
p.add_argument('--render', action='store_true')
a = p.parse_args()
c = load(a.profile)
args, env = engine_args(c, a.rank), environment(c, a.rank)
receipt = dict(rank=a.rank, arguments=args, environment=env)
if a.render:
    print(json.dumps(receipt, indent=2))
    raise SystemExit(0)
root = Path('/opt/sglang4')
lock = json.loads((root / 'versions.lock.json').read_text())
model_config = Path('/models') / c['model_subpath'] / 'config.json'
if hashlib.sha256(model_config.read_bytes()).hexdigest() != lock['model_config_sha256']:
    raise RuntimeError('Checkpoint config differs from the pinned reference')
if os.environ.get('SGLANG_BUILD_COMMIT') != lock['sglang_commit']:
    raise RuntimeError('SGLang source revision differs from the pinned reference')
manifest = json.loads((root / 'patches/manifest.json').read_text())
for name, hashes in manifest.items():
    path = Path('/sgl-workspace/sglang') / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != hashes['after_sha256']:
        raise RuntimeError(f'Unexpected active SGLang source: {name}')
for name, expected in lock['adapter_sha256'].items():
    if hashlib.sha256((root / 'adapter' / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError(f'Unexpected adapter source: {name}')
for name, expected in json.loads((root / 'vendor/b12x-source-manifest.json').read_text()).items():
    if hashlib.sha256((root / 'b12x' / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError(f'Unexpected B12x source: {name}')
receipt['source_files_checked'] = len(manifest)
receipt['b12x_source_files_checked'] = 241
Path('/state/launch.json').write_text(json.dumps(receipt, indent=2) + '\n')
os.environ.update(env)
os.execv(sys.executable, [sys.executable, '-m', 'sglang.launch_server', *args])
