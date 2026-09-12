"""Run a bounded TP4 expert component comparison on the otherwise idle Spark 5."""
import hashlib,json,subprocess,time
from pathlib import Path
source=Path(__file__).resolve().parent;out=source/'evidence';out.mkdir(exist_ok=False);(out/'cache').mkdir()
name='sglang4-trace01-moe-component';image='deepseek-v41-sglang8:native-heads-v1';want='sha256:e7681b5276525821f9be286ed3bf0f51acb5239da27f69f60fdeb57e68c05581'
b12x=Path('/home/operator/deepseek-4.1-flash/sglang8-comparison/candidates/sg4b-draft-flashinfer-r2/kit/b12x')
manifest=json.loads((source/'b12x-source-manifest.json').read_text())
for path,digest in manifest.items():assert hashlib.sha256((b12x/path).read_bytes()).hexdigest()==digest,path
assert subprocess.check_output(['docker','image','inspect','--format','{{.Id}}',image],text=True).strip()==want
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert not subprocess.check_output(['docker','ps','-q'],text=True).strip()
cmd=['docker','run','--pull','never','--name',name,'--network','none','--ipc','host','--gpus','all','--mount','type=bind,src=/home/operator/models,dst=/models,readonly','--mount',f'type=bind,src={source},dst=/candidate,readonly','--mount',f'type=bind,src={out},dst=/evidence','--mount',f'type=bind,src={out}/cache,dst=/cache','--mount',f'type=bind,src={b12x},dst=/b12x,readonly']
for k,v in dict(PYTHONPATH='/candidate:/b12x',B12X_W4A8_TINY_DECODE='0',B12X_COMPILE_CACHE_DIR='/cache/b12x',CUTE_DSL_CACHE_DIR='/cache/cute',XDG_CACHE_HOME='/cache',TRITON_CACHE_DIR='/cache/triton',FLASHINFER_CUDA_ARCH_LIST='12.1a',TORCH_CUDA_ARCH_LIST='12.1a',MAX_JOBS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1').items():cmd+=['-e',k+'='+v]
cmd+=['--entrypoint','python3',image,'-u','/candidate/bench-b12x-tp4.py']
(out/'command.json').write_text(json.dumps(cmd,indent=2)+'\n');(out/'source.json').write_text(json.dumps(dict(image=want,b12x_files=len(manifest),files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')}),indent=2)+'\n')
minimum=1000;start=time.time()
try:
 with (out/'probe.log').open('w') as log:
  process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
  while process.poll() is None:
   mem=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))/1024**2;minimum=min(minimum,mem)
   sample=dict(time=time.time(),available_gib=mem)
   with (out/'samples.jsonl').open('a') as f:f.write(json.dumps(sample)+'\n')
   assert mem>=2 and time.time()-start<1500,sample
   time.sleep(2)
  assert process.returncode==0,process.returncode
 result=json.loads((out/'result.json').read_text());assert result['status']=='PASS'
 state=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]
 assert not state['State']['OOMKilled'] and not state['State']['Running'] and state['RestartCount']==0
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 (out/'completion.json').write_text(json.dumps(dict(status='PASS',minimum_available_gib=minimum,state=state['State'],finished=time.time()),indent=2)+'\n')
 print('COMPONENT_PASS',flush=True)
except BaseException as exc:
 subprocess.run(['docker','stop','-t','10',name],capture_output=True,text=True,timeout=30)
 (out/'failure.json').write_text(json.dumps(dict(status='FAILED',error=repr(exc),minimum_available_gib=minimum),indent=2)+'\n');raise
