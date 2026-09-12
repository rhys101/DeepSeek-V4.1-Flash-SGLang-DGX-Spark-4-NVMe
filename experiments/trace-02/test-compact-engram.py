"""Exact hash and history checks against the installed uniform Engram path."""
import importlib.util,json,random
from types import SimpleNamespace as NS
import torch
from sglang.srt.layers.engram import EngramHasher as Baseline
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
spec=importlib.util.spec_from_file_location('engram_compact_candidate','/candidate/patches/source/python/sglang/srt/layers/engram.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
spec2=importlib.util.spec_from_file_location('sparse_compact_candidate','/candidate/patches/source/python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py');sparse=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(sparse)
torch.manual_seed(937)
def hasher(cls):
 h=cls.__new__(cls);torch.nn.Module.__init__(h);h.max_ngram_size=4;h.pad_id=0;h.pad_row=16;h.image_token_id=999
 h.token_map=torch.randperm(1000,device='cuda').to(torch.int64)
 h.multipliers=torch.randint(1,2**35,(2,4),dtype=torch.int64,device='cuda')*2+1
 h.primes=torch.tensor([[[211+i*2+j*40 for i in range(16)] for j in range(3)]]*2,device='cuda',dtype=torch.int64)
 h.offsets=torch.arange(48,device='cuda',dtype=torch.int64)[None,:].expand(2,-1).contiguous()*400
 h.history=torch.randint(0,999,(17,3),device='cuda',dtype=torch.int32)
 return h
h=hasher(mod.EngramHasher);base=hasher(Baseline)
for name in ['token_map','multipliers','primes','offsets','history']:setattr(base,name,getattr(h,name).clone())
mode=NS(is_decode=lambda:False,is_target_verify=lambda:True,is_extend=lambda:False)
def batch(lens,slots,graph_tokens=None):
 total=sum(lens);lens=torch.tensor(lens,device='cuda',dtype=torch.int32)
 layout=RaggedVerifyLayout.from_verify_lens_device(verify_lens=lens,graph_num_tokens=graph_tokens or total)
 return NS(forward_mode=mode,req_pool_indices=slots,spec_info=NS(draft_token_num=6,ragged_verify_layout=layout),positions=None)
checks=[]
def reference(ids,positions,lens,slots):
 parts=[];start=0
 for i,length in enumerate(lens):
  if length:
   b=NS(forward_mode=mode,req_pool_indices=slots[i:i+1],spec_info=NS(draft_token_num=length),positions=positions[start:start+length])
   parts.append(base.forward(ids[start:start+length],b))
  start+=length
 return torch.cat(parts)
cases=[(x,0) for x in [[n] for n in range(1,7)]+[[6]*8,[6,5,6,5,6,5,6,5],[1]*8,[6,1,5,2,4,3,2,1],[6,0,3,0,2,1,0,0],[12,0,0,0,0,0,0,0]]]+[([6,6,6,6,6,6,6,1],1),([6,0,2,0,0,0,0,0],4)]
def with_padding(value,ghost,slots):
 if not ghost:return value
 ids=torch.zeros(ghost,device='cuda',dtype=torch.int64);pos=ids-1
 b=NS(forward_mode=mode,req_pool_indices=slots[:1],spec_info=NS(draft_token_num=ghost),positions=pos)
 return torch.cat([value,base.forward(ids,b)])
for case,(lens,ghost) in enumerate(cases):
 m=sum(lens)+ghost;slots=torch.randperm(16,device='cuda')[:len(lens)];b=batch(lens,slots,m)
 ids=torch.randint(0,999,(m,),device='cuda',dtype=torch.int64)
 if m>2:ids[1]=999
 positions=torch.cat([torch.arange(n,device='cuda')+(0 if i%2 else 17) for i,n in enumerate(lens)]+[torch.zeros(ghost,device='cuda',dtype=torch.int64)]);b.positions=positions
 prior=h.history.clone();expected=with_padding(reference(ids,positions,lens,slots),ghost,slots);actual=h.forward(ids,b)
 torch.testing.assert_close(actual,expected,rtol=0,atol=0);torch.testing.assert_close(h.history,prior,rtol=0,atol=0)
 expected_req=torch.cat([torch.repeat_interleave(slots,torch.tensor(lens,device='cuda'),output_size=sum(lens)),slots[-1:].expand(ghost)])
 torch.testing.assert_close(sparse.token_req_indices(b,num_tokens=m),expected_req,rtol=0,atol=0)
 # Capture with fixed tensor addresses, then change request boundaries,
 # slots, IDs and positions before replay, keeping total rows constant.
 stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(stream):
  for _ in range(3):h.forward(ids,b);sparse.token_req_indices(b,num_tokens=m)
 stream.synchronize();g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g,stream=stream):captured=h.forward(ids,b);captured_req=sparse.token_req_indices(b,num_tokens=m)
 lens2=list(reversed(lens));other=batch(lens2,slots,m)
 b.spec_info.ragged_verify_layout.qo_indptr_device.copy_(other.spec_info.ragged_verify_layout.qo_indptr_device)
 # Match the real runner: only cumulative boundaries and lengths refresh;
 # extend_start_loc is a separate allocation holding the old capture starts.
 b.spec_info.ragged_verify_layout.verify_lens.copy_(other.spec_info.ragged_verify_layout.verify_lens)
 ids.random_(0,999);slots.copy_(torch.randperm(16,device='cuda')[:len(lens)]);positions.copy_(torch.cat([torch.arange(n,device='cuda')+7 for n in lens2]+[torch.zeros(ghost,device='cuda',dtype=torch.int64)]))
 expected2=with_padding(reference(ids,positions,lens2,slots),ghost,slots);g.replay();torch.cuda.synchronize()
 torch.testing.assert_close(captured,expected2,rtol=0,atol=0);torch.testing.assert_close(h.history,prior,rtol=0,atol=0)
 expected_req2=torch.cat([torch.repeat_interleave(slots,torch.tensor(lens2,device='cuda'),output_size=sum(lens2)),slots[-1:].expand(ghost)])
 torch.testing.assert_close(captured_req,expected_req2,rtol=0,atol=0)
 checks.append(dict(lens=lens,replay_lens=lens2,graph_tokens=m,ghost_rows=ghost,exact_hash=True,history_unchanged=True,exact_request_indices=True))
 # Acceptance remains the original dense-row commit rule and must also be exact.
 verify=torch.randint(0,999,(len(lens),6),device='cuda',dtype=torch.int32);commit=torch.tensor([min(6,n) for n in lens2],device='cuda',dtype=torch.int32)
 expected_history=h.history.clone();old=h.history[slots].clone();window=torch.cat([old,verify],dim=1)
 expected_history[slots]=window.gather(1,commit[:,None].to(torch.int64)+torch.arange(3,device='cuda'))
 h.commit_after_verify(verify,slots,commit);torch.testing.assert_close(h.history,expected_history,rtol=0,atol=0);base.history.copy_(h.history)
print(json.dumps(dict(status='PASS',cases=checks,comparisons=len(checks)*7,method='Bit-exact against installed uniform CUDA hash path per request; exact sparse-attention request indices; capture/replay with changed real layout buffers; no verification history mutation; dense accepted-history commit checked independently.'),indent=2))
