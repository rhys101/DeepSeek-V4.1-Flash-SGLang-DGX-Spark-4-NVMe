"""Inspect trace coverage and kernel symbols before choosing step attribution."""
import argparse,collections,gzip,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();root=Path(a.root);rows=[]
for path in sorted(root.rglob('*.trace.json.gz')):
 data=json.load(gzip.open(path,'rt'));events=data['traceEvents'];x=[e for e in events if e.get('ph')=='X'];kernels=[e for e in x if e.get('cat')=='kernel'];groups=collections.defaultdict(list)
 for e in kernels:groups[e['name']].append(e['dur'])
 graph=[e for e in x if 'GraphLaunch' in e.get('name','') and e.get('cat')=='cuda_runtime'];annotations=[e for e in x if e.get('cat')=='user_annotation']
 summary=dict(path=str(path.relative_to(root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),events=len(events),categories=dict(collections.Counter(e.get('cat','') for e in x)),kernel_events=len(kernels),graph_launches=graph,annotations=annotations,top_kernels=[dict(name=k,count=len(v),sum_ms=sum(v)/1000,mean_us=sum(v)/len(v)) for k,v in sorted(groups.items(),key=lambda kv:sum(kv[1]),reverse=True)])
 rows.append(summary)
 print(summary['path'],'kernels',len(kernels),'graph_launches',len(graph),'annotations',len(annotations))
 for k in summary['top_kernels'][:12]:print(round(k['sum_ms'],3),k['count'],k['name'][:180])
 if graph: print('GRAPH ARGS',graph[:2])
 if annotations:print('ANNOTATIONS',[(e['name'],e['dur'],e.get('args')) for e in annotations[:4]])
(root/'inspection.json').write_text(json.dumps(rows,indent=2)+'\n')
