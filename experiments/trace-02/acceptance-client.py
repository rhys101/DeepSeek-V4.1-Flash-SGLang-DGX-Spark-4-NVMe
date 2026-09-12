"""Collect native per-request speculation counters on exact chat-tokenized inputs."""
import argparse, concurrent.futures, importlib.util, json, statistics, threading, time, urllib.request
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--bench',required=True);p.add_argument('--out',required=True);p.add_argument('--label',required=True);a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
spec=importlib.util.spec_from_file_location('bench',a.bench);bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
base=a.base.rstrip('/').removesuffix('/v1')
def request(route,body=None):
 req=urllib.request.Request(base+route, data=None if body is None else json.dumps(body).encode(), headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=300) as r:return json.load(r)
def save(name,d):(out/name).write_text(json.dumps(d,indent=2)+'\n')
def idle():
 for _ in range(100):
  d=request('/v1/loads')
  if all(r['num_running_reqs']==r['num_waiting_reqs']==0 for r in d['loads']):return
  time.sleep(.1)
 raise AssertionError(d)
idle();info=request('/server_info');save('server-info.json',info)
gamma=info['speculative_dspark_block_size'];assert gamma in [3,4,5,7]
cases=bench.CATEGORIES+[('sparkdash_prose','Write a detailed step-by-step explanation of how a hash map works, including collision handling, resizing, and time complexity. Be thorough.',256)]
rows=[]
for c in [1,8]:
 for category,prompt,budget in cases:
  idle();inputs=[]
  for i in range(c):
   text=prompt if category=='sparkdash_prose' else f'[bench v1 run {category} c{c} s{i}] '+prompt
   body=dict(model='deepseek-v41-flash',messages=[dict(role='user',content=text)],chat_template_kwargs={'thinking':False})
   tokens=request('/v1/tokenize',body)
   assert isinstance(tokens['tokens'],list) and len(tokens['tokens'])==tokens['count']
   if category=='coding':assert tokens['count']==47,tokens
   if category=='sparkdash_prose':assert tokens['count']==32,tokens
   inputs.append(dict(chat=body,tokenization=tokens))
  gate=threading.Barrier(c)
  def one(i):
   body=dict(input_ids=inputs[i]['tokenization']['tokens'],sampling_params=dict(temperature=0,max_new_tokens=budget),stream=False)
   gate.wait();start=time.perf_counter();response=request('/generate',body);elapsed=time.perf_counter()-start
   meta=response['meta_info'];assert meta['spec_verify_ct']>0 and meta['spec_num_proposed_drafts']==meta['spec_verify_ct']*gamma,meta
   histogram=meta.get('spec_correct_drafts_histogram');assert histogram,meta
   return dict(stream=i,input=inputs[i],response=response,total_s=elapsed)
  with concurrent.futures.ThreadPoolExecutor(c) as pool:results=list(pool.map(one,range(c)))
  accepted=sum(r['response']['meta_info']['spec_num_correct_drafts'] for r in results)
  proposed=sum(r['response']['meta_info']['spec_num_proposed_drafts'] for r in results)
  steps=sum(r['response']['meta_info']['spec_verify_ct'] for r in results)
  tokens=sum(r['response']['meta_info']['completion_tokens'] for r in results)
  row=dict(c=c,category=category,accepted_drafts=accepted,proposed_drafts=proposed,verify_steps=steps,completion_tokens=tokens,accepted_draft_fraction=accepted/proposed,completion_tokens_per_verify=tokens/steps,requests=results)
  rows.append(row);save('results.json',dict(label=a.label,gamma=gamma,method='Acceptance diagnostics via /generate using /v1/tokenize chat rendering. Fixed upstream prompts and budgets; per-request counters, never inferred from SSE chunk counts. These elapsed times are not the published OpenAI streaming benchmark.',cases=rows))
  print(category,c,'accepted/token-step',tokens/steps,'draft fraction',accepted/proposed,flush=True)
idle();save('completion.json',dict(status='PASS'))
