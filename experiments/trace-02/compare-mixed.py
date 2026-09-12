"""Compare all categories from the complete repeated mixed suite."""
import argparse
import json
import statistics
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('before');p.add_argument('after');p.add_argument('--out',required=True);a=p.parse_args()
b=json.loads(Path(a.before).read_text());c=json.loads(Path(a.after).read_text())
assert b['source_sha256']==c['source_sha256']
assert len(b['batches'])==len(c['batches'])==48
rows=[]
for category in dict.fromkeys(r['category'] for r in b['batches']):
    for concurrency in [1,8]:
        bb=sorted([r for r in b['batches'] if (r['category'],r['c'])==(category,concurrency)],key=lambda r:r['trial'])
        cc=sorted([r for r in c['batches'] if (r['category'],r['c'])==(category,concurrency)],key=lambda r:r['trial'])
        assert [r['trial'] for r in bb]==[r['trial'] for r in cc]==[1,2,3]
        assert [[v['prompt_tokens'] for v in r['requests']] for r in bb]==[[v['prompt_tokens'] for v in r['requests']] for r in cc]
        metric='per_stream_tok_s' if concurrency==1 else 'agg_tok_s'
        def summary(rs):
            values=[r[metric] for r in rs]
            return dict(mean=statistics.mean(values),stdev=statistics.stdev(values),trials=values,output_tokens=[r['tokens'] for r in rs])
        before,after=summary(bb),summary(cc)
        row=dict(category=category,c=concurrency,metric=metric,before=before,after=after,delta_percent=(after['mean']/before['mean']-1)*100)
        rows.append(row)
        print(f"{category} C{concurrency}: {before['mean']:.3f} -> {after['mean']:.3f} ({row['delta_percent']:+.2f}%)")
Path(a.out).write_text(json.dumps(dict(method='All eight categories; three measured waves after excluded warmups. C1 is per-stream decode rate; C8 is cohort aggregate including TTFT. Every case is shown; no selected-category aggregate is used as a general improvement claim.',rows=rows),indent=2)+'\n')
