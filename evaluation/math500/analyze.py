"""All six cells, paired endpoint contrasts, train x request interaction."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from io_helpers import durable,rows,require,sha,read,now
from score import METRICS

def estimate(values,indices):
 a=np.asarray(values,dtype=float)
 lo,hi=np.quantile(a[indices].mean(axis=1),[.025,.975])
 return dict(estimate_pp=100*float(a.mean()),ci95_pp=[100*float(lo),100*float(hi)])

def main(root):
 p=read(root/'protocol.json');out=root/'analysis';out.mkdir(exist_ok=True)
 rng=np.random.default_rng(20260909);indices=rng.integers(0,500,size=(20000,500),dtype=np.int32)
 cells={};lookup={};table=[]
 for state in p['states']:
  for conv in p['conventions']:
   folder=root/'results'/state/conv;rs=sorted(rows(folder/'scores.jsonl'),key=lambda r:r['id'])
   require([x['id'] for x in rs]==list(range(500)),'Incomplete scores')
   key=state+'/'+conv;lookup[key]=rs
   v={m:np.array([x[m] for x in rs],dtype=int) for m in METRICS};cells[key]=v
   r=dict(state=state,request=conv,**{m:100*float(v[m].mean()) for m in METRICS});table.append(r)
 estimates={k:{m:estimate(v[m],indices) for m in ['strict','math_verify','requested_extractable','truncated']} for k,v in cells.items()}
 contrasts={}
 for state in ['hash_trained','boxed_trained']:
  for conv in ['hash','boxed']:
   b=cells['initial/'+conv];r=cells[state+'/'+conv];record={}
   for m in ['strict','math_verify','requested_extractable','truncated']:
    d=r[m]-b[m];lost=int(((b[m]==1)&(r[m]==0)).sum());gained=int(((b[m]==0)&(r[m]==1)).sum())
    record[m]=dict(**estimate(d,indices),lost=lost,gained=gained,mcnemar_exact_p=float(binomtest(gained,lost+gained,.5).pvalue) if lost+gained else 1.)
   # Exact additive identity using full-response verifier, not latent capability.
   record['net_readout_gap_reduction']=estimate((r['strict']-b['strict'])-(r['math_verify']-b['math_verify']),indices)
   contrasts[state+'/'+conv]=record
 interactions={}
 for m in ['strict','math_verify','requested_extractable']:
  h=cells['hash_trained/hash'][m]-cells['hash_trained/boxed'][m]
  b=cells['boxed_trained/hash'][m]-cells['boxed_trained/boxed'][m]
  interactions[m]=estimate(h-b,indices)
 result=dict(utc=now(),protocol_sha256=sha(root/'protocol.json'),cells=estimates,contrasts_vs_same_request_initial=contrasts,
  training_by_requested_convention_interaction=interactions,
  interaction_definition='(hash-trained/hash-request - hash-trained/boxed-request) - (boxed-trained/hash-request - boxed-trained/boxed-request)',
  uncertainty='20,000 paired item bootstrap resamples shared across every cell; percentile 95% CIs. Item uncertainty only. Exact McNemar p-values unadjusted; all contrasts retained.',
  caveats=p['caveats'])
 durable(out/'analysis.json',result)
 with open(out/'main_table.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
 lines=['# MATH500 transfer: complete prospective 3 x 2 evaluation','',
  'All endpoints were trained on GSM8K (initial is untrained). This is cross-task transfer. One response per problem at T=0.6, top-p=0.95, maximum 8192 output tokens. Percentages below; no test-driven selection.','',
  '| Checkpoint | Requested format | Strict | Full-response MV | Extractable | Truncated |','|---|---|---:|---:|---:|---:|']
 for r in table:lines.append(f"| {r['state']} | {r['request']} | {r['strict']:.1f} | {r['math_verify']:.1f} | {r['requested_extractable']:.1f} | {r['truncated']:.1f} |")
 lines+=['','## Paired changes from the same-request initial checkpoint','']
 for key,r in contrasts.items():
  s=r['strict'];m=r['math_verify'];lines.append(f"- {key}: strict {s['estimate_pp']:+.2f} pp [{s['ci95_pp'][0]:+.2f}, {s['ci95_pp'][1]:+.2f}]; MV {m['estimate_pp']:+.2f} pp [{m['ci95_pp'][0]:+.2f}, {m['ci95_pp'][1]:+.2f}].")
 lines+=['','## Training-by-request interaction','']
 for m,r in interactions.items():lines.append(f"- {m}: {r['estimate_pp']:+.2f} pp [{r['ci95_pp'][0]:+.2f}, {r['ci95_pp'][1]:+.2f}].")
 lines+=['','## Interpretation limits','','The two readouts share symbolic equivalence software but independently select candidate answers. Differences measure operational readout dependence, not established reasoning quality. Single-seed endpoint evidence does not quantify training-seed variation. Report all cells, including adverse outcomes. If truncation is concentrated in an endpoint, qualify that comparison; do not silently rerun with a different budget.']
 (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 # Exhaustive indices for reviewer case inspection; no cherry-picked examples.
 diag=[]
 for key,rs in lookup.items():
  for r in rs:
   if r['strict']!=r['math_verify'] or r['truncated']:
    diag.append(dict(cell=key,**r))
 with open(out/'all_disagreements_and_truncations.jsonl','w',encoding='utf-8') as f:
  for r in diag:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 durable(out/'ANALYSIS_COMPLETE.json',dict(utc=now(),cells=len(cells),responses=3000,files={f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name!='ANALYSIS_COMPLETE.json'}))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);args=a.parse_args();main(args.root)
