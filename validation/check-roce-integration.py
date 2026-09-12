"""Exercise the exact three proposed SGLang overlays on four GPUs, without a model."""
import datetime,hashlib,json,os,types
from pathlib import Path
import torch
import torch.distributed as dist
torch.set_num_threads(2);torch.cuda.set_device(0)
dist.init_process_group('gloo',timeout=datetime.timedelta(seconds=120))
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.distributed.device_communicators.pynccl import synchronize_and_check_roce
from sglang.srt.managers.scheduler_components.batch_result_processor import SchedulerBatchResultProcessor
rank=dist.get_rank();assert dist.get_world_size()==4
manifest=json.loads(Path('/opt/sglang4/patches/manifest.json').read_text())
for path,v in manifest.items():assert hashlib.sha256((Path('/sgl-workspace/sglang')/path).read_bytes()).hexdigest()==v['after_sha256']
def group(name):
 return GroupCoordinator(group_ranks=[list(range(4))],local_rank=0,torch_distributed_backend='nccl',use_pynccl=True,use_pymscclpp=False,use_custom_allreduce=False,use_torch_symm_mem_all_reduce=False,use_hpu_communicator=False,use_xpu_communicator=False,use_npu_communicator=False,group_name=name)
tp=group('tp');other=group('moe_ep');comm=tp.pynccl_comm;rt=comm.roce
assert rt is not None and other.pynccl_comm.roce is None and comm.disabled
assert rt.max_size==512*1024 and rt.max_gather_bytes==0
rt.spin_limit=2_000_000
x=torch.full((1,5120),rank+1.,device='cuda',dtype=torch.bfloat16)
epoch=rt.stats()['epoch'];tp.all_reduce(x);torch.cuda.synchronize();assert bool((x==10).all()) and rt.stats()['epoch']==epoch
del x
xs=[torch.full((m,5120),rank+1.,device='cuda',dtype=d) for m,d in [(1,torch.bfloat16),(6,torch.bfloat16),(24,torch.bfloat16),(48,torch.bfloat16),(128,torch.bfloat16),(6,torch.float32),(6,torch.float16),(1,torch.bfloat16)]]
explicit=torch.empty_like(xs[2]);gather_in=torch.full((5120,),rank+1.,device='cuda',dtype=torch.bfloat16);gather_out=torch.empty(4*5120,device='cuda',dtype=torch.bfloat16)
stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());torch.cuda.synchronize();dist.barrier()
graph=torch.cuda.CUDAGraph()
with tp.graph_capture(stream=stream):
 with torch.cuda.graph(graph,stream=stream):
  tp.all_reduce(xs[0])
  allocated_out=tp._all_reduce_out_place(xs[1],'auto')
  comm.outplace_all_reduce(xs[2],out_tensor=explicit)
  comm.all_reduce(xs[3]);comm.all_reduce(xs[4]);comm.all_reduce(xs[5]);comm.all_reduce(xs[6])
  comm.all_reduce(xs[7],op=dist.ReduceOp.MAX)
  comm.all_gather(gather_out,gather_in)
assert comm.disabled
pointers=[x.data_ptr() for x in xs+[explicit,allocated_out,gather_in,gather_out]];allocated=torch.cuda.memory_allocated();epoch=rt.stats()['epoch']
for trial in range(24):
 for x in xs:x.fill_(rank+1+trial)
 gather_in.fill_(rank+1+trial);explicit.fill_(float('nan'));allocated_out.fill_(float('nan'))
 graph.replay();copy_done=torch.cuda.Event();copy_done.record();synchronize_and_check_roce(copy_done)
 for i in [0,3,4,5,6]:assert bool((xs[i]==10+4*trial).all()),(trial,i)
 assert bool((allocated_out==10+4*trial).all()) and bool((explicit==10+4*trial).all())
 assert bool((xs[7]==4+trial).all())
 assert all(bool((gather_out.reshape(4,5120)[r]==r+1+trial).all()) for r in range(4))
 assert rt.stats()['epoch']==epoch+5*(trial+1)
 assert [x.data_ptr() for x in xs+[explicit,allocated_out,gather_in,gather_out]]==pointers
 assert torch.cuda.memory_allocated()==allocated,(trial,allocated,torch.cuda.memory_allocated())
print('INTEGRATION_GRAPH_PASS '+json.dumps(dict(rank=rank,replays=24,roce_ops_per_replay=5)),flush=True)
# A real fault in the captured transport must stop the real scheduler methods
# at the post-copy health check, before any output processing takes place.
fault=torch.ones((1,5120),device='cuda',dtype=torch.bfloat16);fg=torch.cuda.CUDAGraph()
torch.cuda.synchronize();dist.barrier()
with tp.graph_capture(stream=stream):
 with torch.cuda.graph(fg,stream=stream):
  for _ in range(3):tp.all_reduce(fault)
fg.replay();torch.cuda.synchronize();rt.check_health();assert bool((fault==64).all())
fault.fill_(1);host=torch.empty_like(fault,device='cpu',pin_memory=True);torch.cuda.synchronize();dist.barrier()
if rank==1:rt._proxy.stop()
fg.replay();host.copy_(fault,non_blocking=True);copy_done=torch.cuda.Event();copy_done.record()
fake=types.SimpleNamespace()
checks=[]
for method in ['process_batch_result_decode','process_batch_result_idle']:
 try:getattr(SchedulerBatchResultProcessor,method)(fake,types.SimpleNamespace(),types.SimpleNamespace(copy_done=copy_done))
 except RuntimeError as e:assert 'poisoned' in str(e);checks.append(method)
 else:raise AssertionError('faulted output escaped '+method)
assert rt.poisoned and not bool((host==64).all())
try:synchronize_and_check_roce(None)
except RuntimeError as e:assert 'poisoned' in str(e)
else:raise AssertionError('eventless result escaped transport check')
result=dict(status='PASS',rank=rank,source_overlays=manifest,graph_replays=24,roce_ops_per_replay=5,fallbacks=['eager GroupCoordinator','large BF16','FP16','MAX','all_gather','non-TP group'],scheduler_fault_methods=checks,eventless_fault_check=True,stats=rt.stats())
rt.close();dist.barrier()
print('ROCE_RESULT '+json.dumps(result),flush=True)
del graph,fg
torch.cuda.synchronize()
comm.nccl.ncclCommDestroy(comm.comm);other.pynccl_comm.nccl.ncclCommDestroy(other.pynccl_comm.comm)
tp.destroy();other.destroy();dist.destroy_process_group()
