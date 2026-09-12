"""Recompute published evidence locally; never starts a server or GPU workload."""
import ast
import gzip
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

code = Path(__file__).resolve().parent
repo = code.parent.parent
out = repo / 'results/trace-iteration-02'
def read(path):
    return json.loads(path.read_text())
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def run(script, *args):
    subprocess.run([sys.executable, str(code / script), *map(str, args)], check=True, stdout=subprocess.DEVNULL)

for path in code.rglob('*.py'):
    ast.parse(path.read_text(), filename=str(path))
for path in out.rglob('*.json'):
    read(path)
manifest = read(out / 'trace-manifest.json')['files']
assert len(manifest) == 24
for row in manifest:
    assert sha(out / row['path']) == row['public_sha256']
    assert row['trace_events_identical']
for name in ['traces-baseline', 'traces-k7roce1m', 'traces-compact5-measured']:
    root = out / name
    backup = (root / 'analysis.json').read_bytes()
    previous = json.loads(backup)
    try:
        run('analyze-traces.py', root)
        actual = read(root / 'analysis.json')
        assert len(actual['rows']) == 8
        assert all(r['gpu_events_present'] and len(r['steps']) == 6 for r in actual['rows'])
        for row in previous['rows']:
            row.pop('original_sha256', None)
        assert previous == actual, name
    finally:
        (root / 'analysis.json').write_bytes(backup)
    print(name, 'all eight trace attributions reproduced', flush=True)

identity = read(code / 'source-identity.json')
baseline = identity['baseline_public_commit']
unchanged = ['adapter', 'patches', 'runtime', 'scripts', 'bench', 'validation', 'vendor', 'versions.lock.json']
assert not subprocess.check_output(['git', 'diff', baseline, '--', *unchanged], cwd=repo, text=True)
frozen = read(out / 'frozen-generalization-protocol.json')['files']
for name in ['generalization-client.py', 'repeat-mixed-client.py']:
    assert sha(code / name) == frozen[name]
with tempfile.TemporaryDirectory(prefix='tp4-evidence-check-') as directory:
    temp = Path(directory)
    for label, record in identity['variants'].items():
        patch = code / (label + '.patch')
        assert sha(patch) == record['patch_sha256']
        target = temp / label
        target.mkdir()
        for rel in record['changed_files']:
            if (repo / rel).exists():
                (target / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(repo / rel, target / rel)
        subprocess.run(['git', 'apply', '--check', str(patch)], cwd=target, check=True)
        subprocess.run(['git', 'apply', str(patch)], cwd=target, check=True)
        assert all(sha(target / rel) == expected for rel, expected in record['changed_files'].items())
        if label == 'compact-measured':
            for name, expected in frozen.items():
                if name.startswith('candidate-compact5-measured/'):
                    assert sha(target / name.split('/', 1)[1]) == expected
        print(label, 'patch application and staged byte hashes verified', flush=True)
    checks = [
        ('compare-mixed.py', [out/'client-baseline-mixed/results.json', out/'client-compact5-mixed/results.json'], 'mixed-repeat-comparison.json'),
        ('compare-generalization.py', [out/'client-baseline-generalization-v2', out/'client-compact5-generalization-v2'], 'generalization-comparison.json'),
        ('compare-repeats.py', [out/'client-baseline-mixed/original-repeat', out/'client-compact5-measured/repeat-after'], 'nearby-baseline-comparison.json'),
        ('compare-repeats.py', [out/'client-baseline/repeat-before', out/'client-k7roce1m/repeat-after'], 'k7roce1m-after-vs-baseline.json'),
        ('compare-repeats.py', [out/'client-baseline/repeat-before', out/'client-compact5-measured/repeat-after'], 'compact5-measured-after-vs-baseline.json'),
    ]
    for script, args, filename in checks:
        run(script, *args, '--out', temp / filename)
        assert read(temp / filename) == read(out / filename), filename
    run('audit-roce-capacity.py', '--out', temp/'roce.json')
    expected = read(out/'roce-capacity-completed.json')
    for row in expected['rows']:
        row.pop('original_trace_sha256', None)
    assert read(temp/'roce.json') == expected
    run('fit-calibration.py', out/'client-compact5e-calibration-resume01/calibration/cells.json', '--out', temp/'fit')
    assert sha(temp/'fit/sps-measured.json') == frozen['candidate-compact5-measured/runtime/sps-measured.json']

errors = []
for path in [repo/'README.md', repo/'docs/trace-iteration-02.md', code/'README.md']:
    for link in re.findall(r'\]\(([^)]+)\)', path.read_text()):
        if '://' not in link and not link.startswith('#') and not (path.parent/link.split('#')[0]).exists():
            if not link.endswith('/verification.json'):
                errors.append([str(path), link])
assert not errors, errors
private = re.compile(r'wavebox|/Users/rhys|192\.168\.|10\.10\.|\bagenthost\b|\bspark[1-8]\b')
for root in [out, code]:
    for path in root.rglob('*'):
        if path.is_file() and path.suffix in ['.py', '.json', '.md', '.patch'] and path != Path(__file__).resolve():
            assert not private.search(path.read_text()), path
for name, count in [('final-k7roce1m-health.json',4), ('final-compact5-measured-health.json',4),
                    ('final-baseline-mixed-health.json',4), ('sg17-restored-health.json',8)]:
    health = read(out / 'health' / name)
    assert health['status'] == 'PASS', name
    assert len(health['ranks']) == count, name
result = dict(status='PASS', raw_traces=24, complete_cycles_per_trace=6,
    all_four_ranks_covered=True, trace_attribution_reproduced=True,
    principal_throughput_comparisons_reproduced=True,
    all_eight_target_graph_collective_counts_reproduced=True,
    frozen_client_source_and_runtime_hashes_verified=True,
    fitted_cost_table_reproduced=True, all_three_source_patches_apply_and_match_staged_hashes=True,
    selected_runtime_unchanged=True, python_syntax_and_json_valid=True,
    relative_document_links_valid=True, infrastructure_identifiers_redacted=True,
    final_tp4_health_and_sg17_restoration_passed=True)
(out/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result), flush=True)
