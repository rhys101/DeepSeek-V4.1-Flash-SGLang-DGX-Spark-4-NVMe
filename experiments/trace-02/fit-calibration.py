"""Fit the engine's additive SPS table at the measured small token counts."""
import argparse, json, statistics
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('cells');p.add_argument('--out',required=True);a=p.parse_args()
raw=json.loads(Path(a.cells).read_text())['cells'];out=Path(a.out);out.mkdir(exist_ok=False)
bs=sorted({r['bs'] for r in raw});ms=sorted({r['M'] for r in raw})
cells=[]
for b in bs:
 for m in ms:
  selected=[r for r in raw if r['bs']==b and r['M']==m]
  if selected:cells.append(dict(bs=b,M=m,T=statistics.median(r['T'] for r in selected),trials=len(selected)))
design=np.zeros((len(cells),1+len(bs)-1+len(ms)-1));target=np.array([r['T'] for r in cells])
for i,r in enumerate(cells):
 design[i,0]=1
 if r['bs']!=bs[0]:design[i,bs.index(r['bs'])]=1
 if r['M']!=ms[0]:design[i,len(bs)+ms.index(r['M'])-1]=1
beta,resid,rank,singular=np.linalg.lstsq(design,target,rcond=None)
assert rank==design.shape[1],(rank,design.shape,'Need overlapping request/token probes to identify the additive cost')
pred=design@beta;errors=(pred-target)/target
table=dict(bias_seconds=float(beta[0]),bs_probes=bs,alpha_seconds=[0.]+list(map(float,beta[1:len(bs)])),m_probes=ms,theta_seconds=[0.]+list(map(float,beta[len(bs):])))
assert table['bias_seconds']>0
for b in range(1,9):
 for m in range(b,6*b+1):
  cost=table['bias_seconds']+float(np.interp(b,bs,table['alpha_seconds']))+float(np.interp(m,ms,table['theta_seconds']))
  assert cost>0,(b,m,cost)
report=dict(method='Unweighted ordinary least squares on trial-median cells for T(bs,M)=bias+alpha(bs)+theta(M). Exact measured token counts; no upstream 64-token binning, which is too coarse for this 1–48-token deployment. Linear clamped interpolation is the serving engine contract. Diagnostic calibration, not measured workload throughput.',design_rank=int(rank),design_columns=design.shape[1],median_abs_relative_error=float(np.median(np.abs(errors))),max_abs_relative_error=float(np.max(np.abs(errors))),rms_ms=float(np.sqrt(np.mean((pred-target)**2))*1000),theta_monotone=all(x<=y for x,y in zip(table['theta_seconds'],table['theta_seconds'][1:])),cells=[dict(**r,predicted_seconds=float(q),relative_error=float(e)) for r,q,e in zip(cells,pred,errors)],table=table)
(out/'sps-measured.json').write_text(json.dumps(table,indent=2)+'\n');(out/'fit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['cells','table']},indent=2))
