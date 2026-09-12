"""Attribute complete GPU decode cycles and Engram staging gaps in saved traces."""
import argparse,collections,gzip,hashlib,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();root=Path(a.root)
def category(n):
 s=n.lower()
 if 'b12xcommroce' in s:return 'roce_including_peer_waits'
 if 'nccl' in s:return 'nccl_including_peer_waits'
 if 'groupproblemshape' in s:return 'expert_grouped_gemm'
 if 'dense_blockscaled_gemm_sm120_b12x' in s:return 'b12x_dense_gemm'
 if 'devicegemmmxfp8gemmsm120' in s:return 'flashinfer_dense_gemm'
 if 'wo_a_' in s:return 'wo_a_small_batch'
 if 'cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8' in s:return 'bf16_wmma_projection'
 if 'hc_' in s or 'mhc' in s:return 'hyperconnection'
 if 'sparse_mla' in s or 'rope' in s or 'page_split' in s or 'mqa_logits' in s or 'candidate_' in s:return 'attention_and_indexing'
 if 'tiny_n_gemm' in s:return 'router_projection'
 if 'memcpy' in s or 'memset' in s:return 'memory_copy_or_set'
 return 'other'
def union(intervals):
 end=None;total=0
 for a,b in sorted(intervals):
  if end is None or a>end:total+=b-a;end=b
  elif b>end:total+=b-end;end=b
 return total
rows=[]
for path in sorted(root.rglob('*.trace.json.gz')):
 x=[e for e in json.load(gzip.open(path,'rt'))['traceEvents'] if e.get('ph')=='X'];gpu=sorted([e for e in x if e.get('cat') in ['kernel','gpu_memcpy','gpu_memset']],key=lambda e:e['ts']);launch=sorted([e for e in x if e.get('cat')=='cuda_runtime' and e['name']=='cudaGraphLaunch'],key=lambda e:e['ts'])
 row=dict(path=str(path.relative_to(root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),gpu_events_present=bool(gpu),graphs=[])
 if not gpu:rows.append(row);continue
 by_correlation=collections.defaultdict(list)
 for e in gpu:by_correlation[e['args'].get('correlation')].append(e)
 assert len(launch)%2==0,len(launch)
 for l in launch:
  ev=by_correlation[l['args']['correlation']];assert ev
  row['graphs'].append(dict(correlation=l['args']['correlation'],first_gpu_us=min(e['ts'] for e in ev),last_gpu_us=max(e['ts']+e['dur'] for e in ev),events=len(ev),expert_gemms=sum('GroupProblemShape' in e['name'] for e in ev)))
 assert [g['expert_gemms'] for g in row['graphs']]==[6,80]*(len(launch)//2),row['graphs']
 starts=[g['first_gpu_us'] for g in row['graphs'][::2]]
 # Graph copies identify the exact staging sequence: 8 B/ID DtoH followed by
 # 256 B/ID weights HtoD and 8 B/ID scales HtoD on the same graph stream.
 copies=[e for e in gpu if e.get('cat')=='gpu_memcpy' and e['args'].get('graph id')]
 groups=collections.defaultdict(list)
 for e in copies:groups[(e['args']['correlation'],e['args']['stream'])].append(e)
 staging=[]
 for key,events in groups.items():
  for i,e in enumerate(events[:-2]):
   if 'DtoH' not in e['name']:continue
   h,s=events[i+1:i+3];n=e['args']['bytes']//8
   if 'HtoD' in h['name'] and 'HtoD' in s['name'] and h['args']['bytes']==n*256 and s['args']['bytes']==n*8:
    a=e['ts']+e['dur'];b=h['ts'];inside=[(max(a,k['ts']),min(b,k['ts']+k['dur'])) for k in gpu if k['ts']<b and k['ts']+k['dur']>a]
    staging.append(dict(correlation=key[0],stream=key[1],ids=n,start_us=a,end_us=b,gap_us=b-a,gpu_idle_us=b-a-union(inside)))
 row['engram_staging']=staging;row['steps']=[]
 for a,b in list(zip(starts,starts[1:]))[1:]:
  ks=[e for e in gpu if e['ts']<b and e['ts']+e['dur']>a];times=collections.defaultdict(float)
  for e in ks:times[category(e['name'])]+=min(b,e['ts']+e['dur'])-max(a,e['ts'])
  ss=[e for e in staging if a<=e['start_us'] and e['end_us']<=b]
  row['steps'].append(dict(wall_ms=(b-a)/1000,gpu_busy_union_ms=union([(max(a,e['ts']),min(b,e['ts']+e['dur'])) for e in ks])/1000,engram_staging_gaps=len(ss),engram_gap_sum_ms=sum(e['gap_us'] for e in ss)/1000,engram_gpu_idle_ms=sum(e['gpu_idle_us'] for e in ss)/1000,kernel_duration_sums_ms={k:v/1000 for k,v in times.items()}))
 steps=row['steps'];keys=set().union(*(s['kernel_duration_sums_ms'].keys() for s in steps));row['means']={k:statistics.mean(s[k] for s in steps) for k in ['wall_ms','gpu_busy_union_ms','engram_staging_gaps','engram_gap_sum_ms','engram_gpu_idle_ms']};row['means']['kernel_duration_sums_ms']={k:statistics.mean(s['kernel_duration_sums_ms'].get(k,0) for s in steps) for k in sorted(keys)}
 rows.append(row);print(row['path'],json.dumps(row['means']))
(root/'analysis.json').write_text(json.dumps(dict(method='Six complete cycles per 8-step capture after excluding the first and truncated last. Boundaries are first GPU events of consecutive draft graphs, paired with 40-layer target graphs using runtime correlation IDs. Kernel sums can overlap. Engram gaps include host scheduling and I/O between graph DtoH/HtoD copies, not pure disk latency.',rows=rows),indent=2)+'\n')
