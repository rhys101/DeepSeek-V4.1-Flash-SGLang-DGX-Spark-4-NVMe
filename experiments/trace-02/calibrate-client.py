"""Measure real greedy decode cost across request counts and verification budgets."""
import argparse, collections, concurrent.futures, json, statistics, threading, time, urllib.request
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--out',required=True)
p.add_argument('--levels',default='1,4,8');p.add_argument('--repeats',type=int,default=2)
p.add_argument('--fracs',default='1,0.6,0.2,0.8,0.4,0.01');a=p.parse_args()
base=a.base.rstrip('/').removesuffix('/v1');out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
def req(route,body=None):
 request=urllib.request.Request(base+route,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(request,timeout=300) as r:return json.load(r)
def save(name,value):(out/name).write_text(json.dumps(value,indent=2)+'\n')
def idle():
 for _ in range(100):
  value=req('/v1/loads')
  if all(x['num_running_reqs']==x['num_waiting_reqs']==0 for x in value['loads']):return
  time.sleep(.1)
 raise AssertionError(value)
def set_frac(frac):
 value=req('/set_internal_state',{'server_args':{'dspark_force_budget_frac':frac,'dspark_clear_info_records':1}})
 outputs=value if isinstance(value,list) else [value]
 assert all(x is True or isinstance(x,dict) and x.get('updated') is True for x in outputs),value
 return value
idle();info=req('/server_info');save('server-info-before.json',info)
gamma=info['speculative_dspark_block_size'];assert gamma==5
smoke=[]
for frac in [.01,.4,1.0]:
 idle();control=set_frac(frac);inputs=[]
 for i in range(8):
  body=dict(model='deepseek-v41-flash',messages=[dict(role='user',content=f'What is {17+i} times 19? Return only the integer.')],chat_template_kwargs={'thinking':False})
  inputs.append(req('/v1/tokenize',body)['tokens'])
 gate=threading.Barrier(8)
 def check(i):
  gate.wait();response=req('/generate',dict(input_ids=inputs[i],sampling_params=dict(temperature=0,max_new_tokens=32),stream=False))
  assert response['text'].strip()==str((17+i)*19),(frac,i,response)
  return dict(stream=i,input_ids=inputs[i],response=response)
 with concurrent.futures.ThreadPoolExecutor(8) as pool:responses=list(pool.map(check,range(8)))
 smoke.append(dict(frac=frac,control=control,responses=responses));save('budget-arithmetic-smoke.json',dict(status='PASS',cases=smoke))
 print('BUDGET_ARITHMETIC_PASS',frac,flush=True)
rows=[]
for trial in range(1,a.repeats+1):
 levels=list(map(int,a.levels.split(',')));fracs=list(map(float,a.fracs.split(',')))
 if trial%2==0:levels.reverse();fracs.reverse()
 for c in levels:
  inputs=[]
  for i in range(c):
   prompt=f'[TP4 cost calibration stream {i}] Explain how a database index works, including a concrete example, lookup, insertion, and the cost of maintaining it.'
   tokens=req('/v1/tokenize',dict(model='deepseek-v41-flash',messages=[dict(role='user',content=prompt)],chat_template_kwargs={'thinking':False}))
   inputs.append(tokens['tokens'])
  for frac in fracs:
   idle();control=set_frac(frac);gate=threading.Barrier(c)
   def one(i):
    gate.wait();start=time.perf_counter()
    response=req('/generate',dict(input_ids=inputs[i],sampling_params=dict(temperature=0,max_new_tokens=192,ignore_eos=True),stream=False))
    elapsed=time.perf_counter()-start;meta=response['meta_info']
    assert meta['completion_tokens']==192 and meta['spec_verify_ct']>0,meta
    assert response['text'].strip(),response
    return dict(stream=i,input_ids=inputs[i],elapsed_s=elapsed,response=response)
   start=time.perf_counter()
   with concurrent.futures.ThreadPoolExecutor(c) as pool:responses=list(pool.map(one,range(c)))
   wall=time.perf_counter()-start;idle();state=req('/server_info')
   states=state.get('internal_states') or [];assert len(states)==1,len(states)
   record=states[0].get('dspark_info_record') or {};records=record.get('records',[])
   full=[r for r in records if r.get('num_running_reqs')==c and r.get('step_cpu_ms') is not None]
   steady=full[8:-3];assert len(steady)>=12,(c,frac,len(records),len(steady),record)
   shape_counts=collections.Counter((r['num_verify_tokens'],r['verify_tokens_graph_key']) for r in steady)
   modal_shape,count=shape_counts.most_common(1)[0]
   assert count/len(steady)>.8,(c,frac,shape_counts)
   selected=[r for r in steady if (r['num_verify_tokens'],r['verify_tokens_graph_key'])==modal_shape]
   cell=dict(trial=trial,bs=c,frac=frac,M=modal_shape[0],graph_tokens=modal_shape[1],T=statistics.median(r['step_cpu_ms'] for r in selected)/1000,steady_steps=len(selected),step_cpu_ms=[r['step_cpu_ms'] for r in selected],shapes=[dict(tokens=k[0],graph=k[1],count=v) for k,v in sorted(shape_counts.items())],wall_s=wall,control=control,responses=responses,observer=record)
   rows.append(cell);save('cells.json',dict(method='Actual greedy native generation; fixed 192-token outputs. CPU observer timing is calibration only. Discard first eight and final three full-batch rows; retain the modal verification shape. No simulated acceptance.',cells=rows))
   print(json.dumps({k:cell[k] for k in ['trial','bs','frac','M','graph_tokens','T','steady_steps']}),flush=True)
idle();save('restore-full-budget.json',set_frac(1.0));save('completion.json',dict(status='PASS'))
