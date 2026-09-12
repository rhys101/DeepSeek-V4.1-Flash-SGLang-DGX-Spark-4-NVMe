"""Bounded mid-generation traces of pinned coding prompts; timing is diagnostic."""
import argparse, concurrent.futures, datetime, importlib.util, json, threading, time, urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--workload',choices=['coding','prose'],default='coding');p.add_argument('--base',required=True);p.add_argument('--bench',required=True);p.add_argument('--out',required=True);p.add_argument('--label',required=True);p.add_argument('--levels',default='1,8');p.add_argument('--server-trace-root',default='/state/trace-iter1');a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
spec=importlib.util.spec_from_file_location('bench',a.bench);bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
base=a.base.rstrip('/'); model='deepseek-v41-flash'
def get(route):
 with urllib.request.urlopen(base.removesuffix('/v1')+route,timeout=30) as r:return json.load(r)
def save(name,value):(out/name).write_text(json.dumps(value,indent=2)+'\n')
def wait_idle():
 for _ in range(100):
  value=get('/v1/loads')
  if all(x['num_running_reqs']==x['num_waiting_reqs']==0 for x in value['loads']):return value
  time.sleep(.1)
 raise AssertionError(value)
wait_idle()
save('server-info.json',get('/server_info'))
for c in map(int,a.levels.split(',')):
 profile_id=f'{a.label}-{a.workload}-c{c}'; trigger=threading.Event(); gate=threading.Barrier(c);lock=threading.Lock();chunks=[0]*c
 def stream(i):
  prompt=(f'[bench v1 run coding c{c} s{i}] '+bench.CATEGORIES[0][1]) if a.workload=='coding' else 'Write a detailed step-by-step explanation of how a hash map works, including collision handling, resizing, and time complexity. Be thorough.'
  body=dict(model=model,messages=[dict(role='user',content=prompt)],max_tokens=(200 if a.workload=='coding' else 256),temperature=0,stream=True,stream_options=dict(include_usage=True),chat_template_kwargs=dict(thinking=False))
  req=urllib.request.Request(base+'/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
  gate.wait();start=time.time();events=[];usage=None;pieces=[]
  with urllib.request.urlopen(req,timeout=300) as response:
   for raw in response:
    line=raw.decode().strip()
    if not line.startswith('data:'):continue
    data=line[5:].strip()
    if data=='[DONE]':break
    value=json.loads(data)
    if value.get('usage'):usage=value['usage']
    for choice in value.get('choices') or []:
     delta=choice.get('delta') or {};piece=(delta.get('content') or '')+(delta.get('reasoning') or '')
     if piece:
      pieces.append(piece);events.append(dict(unix_ns=time.time_ns(),characters=len(piece)))
      with lock:
       chunks[i]+=1
       if min(chunks)>=8:trigger.set()
  assert usage and usage['completion_tokens']==(200 if a.workload=='coding' else 256),usage
  return dict(stream=i,start_unix=start,finish_unix=time.time(),usage=usage,events=events,text=''.join(pieces),body=body)
 def profile():
  assert trigger.wait(60),'Decode trigger was never reached'
  body=dict(output_dir=a.server_trace_root.rstrip('/')+'/'+profile_id,num_steps=8,activities=['CPU','GPU'],with_stack=False,record_shapes=True,profile_id=profile_id,merge_profiles=False,detailed_annotations=True)
  start=time.time();request=urllib.request.Request(base.removesuffix('/v1')+'/start_profile',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(request,timeout=180) as r:result=dict(status=r.status,message=r.read().decode())
  return dict(start_unix=start,finish_unix=time.time(),body=body,response=result,trigger='Every stream has delivered eight nonempty chunks; chunks are not tokens.')
 print(json.dumps(dict(phase='trace',profile_id=profile_id,status='RUNNING')),flush=True)
 with concurrent.futures.ThreadPoolExecutor(c+1) as pool:
  control=pool.submit(profile);streams=list(pool.map(stream,range(c)));receipt=control.result()
 save(profile_id+'.json',dict(profile=receipt,streams=streams,scope='Diagnostic profile; never use these elapsed times as benchmark results.'))
 wait_idle()
 print(json.dumps(dict(phase='trace',profile_id=profile_id,status='PASS')),flush=True)
save('completion.json',dict(status='PASS',finished=datetime.datetime.now(datetime.timezone.utc).isoformat()))
