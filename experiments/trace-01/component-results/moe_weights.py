"""Real-checkpoint EP4/MoE-TP2 component check. No serving code changes.

On one idle SM121, simulate each group's two TP ranks. Compare their sum
with the same group's two current EP8 ranks and with its unsplit expert set.
This tests local partition arithmetic, not network behavior or model quality.
"""
import contextlib, datetime, gc, hashlib, json, statistics, time
from pathlib import Path
import torch
from safetensors import safe_open
from flashinfer import block_scale_interleave, mxfp8_quantize
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe.core import ActivationType

OUT=Path('/evidence'); MODEL=Path('/models/DeepSeek-V4.1-Flash')
K,N=5120,2304
torch.backends.cuda.matmul.allow_tf32=False
RESULT=dict(status='RUNNING',started=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    gates=dict(relative_l2=0.02,cosine=0.9995),cases=[],timings=[],checkpoint_tensors=[],
    limitation='Same FlashInfer arithmetic and checkpoint. Synthetic inputs/routes, local sums; no full-model quality or serving performance claim.')
def save(phase,**kw):
    RESULT.update(phase=phase,**kw);(OUT/'result.json').write_text(json.dumps(RESULT,indent=2)+'\n')
    print(json.dumps(dict(phase=phase,**kw)),flush=True)
def metric(a,b):
    a,b=a.float(),b.float();assert torch.isfinite(a).all() and torch.isfinite(b).all()
    norm=float(b.norm())
    if norm==0:
        assert torch.count_nonzero(a)==0
        return dict(relative_l2=0,cosine=1,reference_norm=0)
    row=dict(relative_l2=float((a-b).norm())/norm,cosine=float(torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0)),reference_norm=norm)
    assert row['relative_l2']<=RESULT['gates']['relative_l2'] and row['cosine']>=RESULT['gates']['cosine'],row
    return row
def load_weights(prefix,group,total):
    E=total//4;idx=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    raw={k:torch.empty(s,dtype=torch.uint8,device='cuda') for k,s in dict(w13=(E,2*N,K//2),s13=(E,2*N,K//32),w2=(E,K,N//2),s2=(E,K,N//32)).items()}
    with contextlib.ExitStack() as stack:
        files={}
        for local in range(E):
            for proj,field,dest,part in [('w3','weight','w13',slice(0,N)),('w1','weight','w13',slice(N,2*N)),('w3','scale','s13',slice(0,N)),('w1','scale','s13',slice(N,2*N)),('w2','weight','w2',slice(None)),('w2','scale','s2',slice(None))]:
                key=f'{prefix}.ffn.experts.{group*E+local}.{proj}.{field}';fn=idx[key]
                if fn not in files:files[fn]=stack.enter_context(safe_open(str(MODEL/fn),framework='pt',device='cpu'))
                value=files[fn].get_tensor(key).contiguous().view(torch.uint8)
                assert value.shape==raw[dest][local,part].shape
                raw[dest][local,part].copy_(value)
                RESULT['checkpoint_tensors'].append(dict(key=key,file=fn,shape=list(value.shape),sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest()))
    return raw
def prepare(raw):
    r=dict(raw)
    r['s13']=block_scale_interleave(raw['s13']).reshape_as(raw['s13'])
    r['s2']=block_scale_interleave(raw['s2']).reshape_as(raw['s2'])
    r['ones']=torch.ones(raw['w13'].shape[0],device='cuda',dtype=torch.float32)
    r['limits']=r['ones']*10
    return r
def split(raw,t):
    h=N//2;s=t*h
    return dict(w13=torch.cat([raw['w13'][:,s:s+h],raw['w13'][:,N+s:N+s+h]],dim=1).contiguous(),
        s13=torch.cat([raw['s13'][:,s:s+h],raw['s13'][:,N+s:N+s+h]],dim=1).contiguous(),
        w2=raw['w2'][:,:,s//2:(s+h)//2].contiguous(),s2=raw['s2'][:,:,s//32:(s+h)//32].contiguous())
def inputs(m,total,topk,group,mode,scale,seed):
    gen=torch.Generator(device='cuda').manual_seed(seed);E=total//4
    x=(torch.randn(m,K,device='cuda',generator=gen)*scale).bfloat16()
    scores=torch.randn(m,total,device='cuda',generator=gen)
    if mode=='local':
        scores[:,:group*E]=-float('inf');scores[:,(group+1)*E:]=-float('inf')
    elif mode=='nonlocal':scores[:,group*E:(group+1)*E]=-float('inf')
    v,ids=torch.topk(scores,topk,dim=1)
    return x,ids.int(),torch.softmax(v,dim=1).float()*1.5
