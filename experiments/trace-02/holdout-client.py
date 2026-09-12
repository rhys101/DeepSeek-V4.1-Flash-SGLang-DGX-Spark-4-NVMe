"""Fresh-prefix fixed workload control for the speculation-block comparison."""
import argparse,datetime,hashlib,importlib.util,json,time
from pathlib import Path
from types import SimpleNamespace
p=argparse.ArgumentParser();p.add_argument('--bench',required=True);p.add_argument('--base',required=True);p.add_argument('--out',required=True);p.add_argument('--label',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
spec=importlib.util.spec_from_file_location('bench',a.bench);bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
args=SimpleNamespace(base=a.base,model='deepseek-v41-flash');rows=[]
for c in [1,8]:
 for category,prompt,budget in bench.CATEGORIES:
  row=bench.run_batch(args,c,category,prompt,budget,'trace01-holdout');rows.append(row)
  print(json.dumps(dict(c=c,category=category,decode=row['per_stream_tok_s'],aggregate=row['agg_tok_s'])),flush=True)
  (out/'results.json').write_text(json.dumps(dict(label=a.label,source_sha256=hashlib.sha256(Path(a.bench).read_bytes()).hexdigest(),batches=rows,method='Fresh front tag trace01-holdout, identical baseline/candidate prompts. One C1 and C8 wave per category. Tokens from server usage. These are single-wave controls, not repeated estimates.'),indent=2)+'\n')
(out/'completion.json').write_text(json.dumps(dict(status='PASS',finished=datetime.datetime.now(datetime.timezone.utc).isoformat()))+'\n')
