"""Compare the frozen heterogeneous-arrival workload, including quality failures."""
import argparse
import json
import statistics
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('before');p.add_argument('after');p.add_argument('--out',required=True);a=p.parse_args()
br,cr=Path(a.before),Path(a.after)
bp=json.loads((br/'protocol.json').read_text());cp=json.loads((cr/'protocol.json').read_text())
assert bp==cp
bi=json.loads((br/'inputs.json').read_text());ci=json.loads((cr/'inputs.json').read_text());assert bi==ci
b=json.loads((br/'results.json').read_text());c=json.loads((cr/'results.json').read_text())
assert [t['trial'] for t in b['trials']]==[t['trial'] for t in c['trials']]==[1,2,3]
def summarize(values):
    return dict(mean=statistics.mean(values),stdev=statistics.stdev(values),trials=values)
overall={}
for metric in ['cohort_tok_s','wall_s','mean_latency_s','median_latency_s','completion_tokens']:
    x=summarize([t[metric] for t in b['trials']]);y=summarize([t[metric] for t in c['trials']])
    overall[metric]=dict(before=x,after=y,delta_percent=(y['mean']/x['mean']-1)*100)
    print(metric,x['mean'],'->',y['mean'],overall[metric]['delta_percent'])
rows=[]
for spec in bp['cases']:
    name=spec['id']
    bb=[next(r for r in t['responses'] if r['id']==name) for t in b['trials']]
    cc=[next(r for r in t['responses'] if r['id']==name) for t in c['trials']]
    x=summarize([r['total_s'] for r in bb]);y=summarize([r['total_s'] for r in cc])
    rows.append(dict(id=name,arrival_target_s=spec['at'],output_budget=spec['budget'],
                     before_latency=x,after_latency=y,latency_delta_percent=(y['mean']/x['mean']-1)*100,
                     before_mean_verify_cap=statistics.mean(r['verified_rows_per_step'] for r in bb),after_mean_verify_cap=statistics.mean(r['verified_rows_per_step'] for r in cc),
                     before_output_tokens=[r['response']['meta_info']['completion_tokens'] for r in bb],after_output_tokens=[r['response']['meta_info']['completion_tokens'] for r in cc],
                     before_exact_checks=[r['exact_check_passed'] for r in bb],after_exact_checks=[r['exact_check_passed'] for r in cc]))
result=dict(method='Identical frozen protocol, tokenized inputs and arrivals; an excluded warmup cohort and three measured trials. Cohort throughput and request latency include prefill and scheduling. These native-streaming results are separate from the original OpenAI benchmark. Exact-format failures remain failures. Open-ended tasks have fixed output caps and are not a comprehensive quality evaluation.',
            protocol_sha256=bp['script_sha256'],overall=overall,cases=rows)
Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
