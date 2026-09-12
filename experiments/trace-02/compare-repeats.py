"""Compare original streaming benchmark trials, keeping metrics and workloads separate."""
import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('before');p.add_argument('after');p.add_argument('--out');a=p.parse_args()
def collect(root):
 root=Path(root);coding=json.loads((root/'coding.json').read_text());prose=json.loads((root/'sparkdash/decode-history.json').read_text())['comparison'];rows={}
 for c in [1,4,8]:
  cb=[r for r in coding['batches'] if r['c']==c];pb=[r for t in prose for r in t['results'] if r['concurrency']==c]
  assert all(r['streamsFailed']==0 and r['streamsOk']==c for r in pb)
  for cat,metric,samples in [('coding','decode_per_stream_tok_s',[r['per_stream_tok_s'] for r in cb]),('coding','aggregate_including_ttft_tok_s',[r['agg_tok_s'] for r in cb]),('prose','decode_per_stream_tok_s',[r['meanDecodeTps'] for r in pb]),('prose','aggregate_decode_tok_s',[r['aggregateDecodeTps'] for r in pb])]:
   rows[(cat,c,metric)]=dict(mean=statistics.mean(samples),stdev=statistics.stdev(samples),trials=samples)
 return rows
b=collect(a.before);c=collect(a.after);rows=[]
for key in b:
 cat,concurrency,metric=key;before=b[key];after=c[key]
 rows.append(dict(workload=cat,c=concurrency,metric=metric,before=before,after=after,delta_percent=(after['mean']/before['mean']-1)*100))
result=dict(method='Original streaming clients; no profiler elapsed times. Coding aggregate includes TTFT; SparkDash aggregate excludes it. Arithmetic means across complete repeated trials. Round-to-round variation is reported, not treated as a confidence interval.',rows=rows)
if a.out:Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
for r in rows:
 if r['c']==1 and r['metric']=='decode_per_stream_tok_s' or r['c'] in [4,8] and 'aggregate' in r['metric']:
  print(f"{r['workload']} C{r['c']} {r['metric']}: {r['before']['mean']:.3f} -> {r['after']['mean']:.3f} ({r['delta_percent']:+.2f}%)")
