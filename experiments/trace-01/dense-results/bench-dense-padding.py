"""Check exact zero padding to reach the B12x MXFP8 contraction alignment."""
import hashlib,json,statistics,time
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors import safe_open
from flashinfer import block_scale_interleave,mxfp8_quantize,mm_mxfp8
OUT=Path('/evidence');MODEL=Path('/models/DeepSeek-V4.1-Flash');torch.set_num_threads(4);torch.cuda.set_device(0)
index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map'];result=dict(status='RUNNING',cases=[],checkpoint=[],method='Original FP8/E8M0 shared-down projection, BF16 input/output, MXFP8 activations. Candidate zero-pads the contraction dimension to 128 alignment. Cold-L2 alternating CUDA-graph measurements include input padding and quantization; no serving performance claim.')
def load(key):
 with safe_open(str(MODEL/index[key]),framework='pt',device='cpu') as f:x=f.get_tensor(key)
 result['checkpoint'].append(dict(key=key,shape=list(x.shape),sha256=hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()))
 return x.cuda()
def save():(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
def errors(a,b):
 a,b=a.float(),b.float();assert bool(torch.isfinite(a).all());e=dict(relative_l2=float((a-b).norm()/b.norm().clamp_min(1e-12)),max_abs=float((a-b).abs().max()),exact=bool(torch.equal(a,b)));assert e['relative_l2']<.008,e;return e
full=load('layers.2.ffn.shared_experts.w2.weight');scales=load('layers.2.ffn.shared_experts.w2.scale').view(torch.uint8).repeat_interleave(32,0)
try:
 for tp in [4,8]:
  n=full.shape[0];k=full.shape[1]//tp;kp=(k+127)//128*128
  w=full[:,:k].contiguous();sf=scales[:,:k//32].contiguous();sw=block_scale_interleave(sf).reshape(-1)
  wp=torch.zeros((n,kp),device='cuda',dtype=torch.float8_e4m3fn);wp[:,:k].copy_(w)
  sp=torch.full((n,kp//32),127,device='cuda',dtype=torch.uint8);sp[:,:k//32].copy_(sf);sps=block_scale_interleave(sp).reshape(-1)
  assert torch.equal(wp[:,:k].view(torch.uint8),w.view(torch.uint8)) and int(torch.count_nonzero(wp[:,k:].float()))==0
  wd=(w.float().reshape(n,-1,32)*torch.exp2(sf.float()-127).unsqueeze(-1)).reshape(n,k)
  for m in [1,5,6,7,8,48,64]:
   x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16);out={}
   def baseline():
    q,s=mxfp8_quantize(x,is_sf_swizzled_layout=True,alignment=32);out['baseline']=mm_mxfp8(q,w.t(),s,sw,out_dtype=torch.bfloat16,use_8x4_sf_layout=False,backend='cutlass')
   def candidate():
    xp=F.pad(x,(0,kp-k));q,s=mxfp8_quantize(xp,is_sf_swizzled_layout=True,alignment=32);out['candidate']=mm_mxfp8(q,wp.t(),s,sps,out_dtype=torch.bfloat16,use_8x4_sf_layout=False,backend='b12x')
   for _ in range(3):baseline();candidate()
   torch.cuda.synchronize();graphs={};checks=[]
   for name,fn in [('baseline',baseline),('candidate',candidate)]:
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):fn()
    graphs[name]=g
   ptrs=[x.data_ptr(),out['baseline'].data_ptr(),out['candidate'].data_ptr()]
   for seed in [100+m,200+m,300+m]:
    gen=torch.Generator(device='cuda').manual_seed(seed);x.copy_((torch.randn(x.shape,device='cuda',generator=gen)*2).bfloat16())
    q,sl=mxfp8_quantize(x,is_sf_swizzled_layout=False,alignment=32);sl=sl.reshape(-1,k//32)[:m];xd=(q.float().reshape(m,-1,32)*torch.exp2(sl.float()-127).unsqueeze(-1)).reshape(m,k);ref=xd@wd.t()
    for _ in range(3):
     for g in graphs.values():g.replay()
     torch.cuda.synchronize();checks.append(dict(seed=seed,candidate_baseline=errors(out['candidate'],out['baseline']),baseline_oracle=errors(out['baseline'],ref),candidate_oracle=errors(out['candidate'],ref)))
    assert ptrs==[x.data_ptr(),out['baseline'].data_ptr(),out['candidate'].data_ptr()]
   flush=torch.empty(64*1024*1024,device='cuda',dtype=torch.uint8);times={k:[] for k in graphs}
   def timed(graph):
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();graph.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
   for trial in range(20):
    for name in (['baseline','candidate'] if trial%2==0 else ['candidate','baseline']):
     flush.fill_(trial);times[name].append(timed(graphs[name]))
   med={k:statistics.median(v) for k,v in times.items()};row=dict(tp=tp,n=n,k=k,padded_k=kp,m=m,checks=checks,times_us=times,median_us=med,speedup=med['baseline']/med['candidate']);result['cases'].append(row);save();print(json.dumps({k:v for k,v in row.items() if k not in ['checks','times_us']}),flush=True)
   del graphs,out,x,flush,q,sl,xd,ref
 result['status']='PASS';save();print('DENSE_PADDING_PASS',flush=True)
except BaseException as exc:result.update(status='FAILED',error=repr(exc));save();raise
