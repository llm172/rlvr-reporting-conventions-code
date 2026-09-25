"""CPU-only evaluator queue; freezes external verifier settings and retains per-response flags."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import datetime
import importlib.util
import json
import os
from pathlib import Path
import time
import traceback
from extension_common import dump,sha,read_jsonl,write_jsonl,extract_scores,validate_rows

METRICS=['strict_first_hash','strict_last_hash','frozen_lenient','last_number','math_verify']

def score(folder):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    folder=Path(folder);meta=json.loads((folder/'task.json').read_text());task=meta['task']
    from math_verify import parse,verify
    fixture_gold=parse(r'$\boxed{12}$')
    if not fixture_gold or not verify(fixture_gold,parse('The answer is 12.')):
        raise RuntimeError('Math-Verify positive fixture failed')
    if verify(fixture_gold,parse('The answer is 13.')):raise RuntimeError('Math-Verify negative fixture failed')
    spec=importlib.util.spec_from_file_location('frozen_passk',Path(__file__).resolve().parents[2]/'scripts/passk_score.py')
    frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
    items={x['id']:x for x in read_jsonl(task['items'])};raw=read_jsonl(folder/'responses.jsonl')
    validate_rows(raw,set(items),task['n'],complete=True)
    generation=json.loads((folder/'GENERATION_COMPLETE.json').read_text())
    if sha(folder/'responses.jsonl')!=generation['raw_sha256']:raise ValueError('Raw response file changed')
    totals={m:0 for m in METRICS};count=0;per_item=[];parse_empty=0;truncated=0
    with open(folder/'response_scores.jsonl','w',encoding='utf-8') as detail:
        for row in raw:
            gold=items[row['id']]['ground_truth'];gold_parsed=parse('$\\boxed{'+str(gold)+'}$')
            if not gold_parsed:raise ValueError('Gold parse failure: '+str(row['id']))
            counts={m:0 for m in METRICS}
            for j,response in enumerate(row['responses']):
                scores=extract_scores(response,gold)
                assert scores['strict_first_hash']==bool(frozen.gsm8k_strict(response,gold)),'Frozen strict mismatch'
                assert scores['frozen_lenient']==bool(frozen.gsm8k_lenient(response,gold)),'Frozen lenient mismatch'
                parsed=parse(response);parse_empty+=not bool(parsed)
                scores['math_verify']=bool(verify(gold_parsed,parsed)) if parsed else False
                truncated+=row['finish_reasons'][j]=='length'
                for m in METRICS:totals[m]+=scores[m];counts[m]+=scores[m]
                count+=1
                detail.write(json.dumps(dict(id=row['id'],sample=j,**scores))+'\n')
            per_item.append(dict(id=row['id'],n=row['n'],counts=counts))
    write_jsonl(folder/'per_item_scores.jsonl',per_item)
    curves={}
    import math
    for m in METRICS:
        curves[m]={str(k):sum(1-(math.comb(x['n']-x['counts'][m],k)/math.comb(x['n'],k)
                     if x['n']-x['counts'][m]>=k else 0) for x in per_item)/len(per_item)
                     for k in [1,2,4,8,16,32] if k<=task['n']}
    summary=dict(state=meta['state'],task=task,items=len(raw),responses=count,
        accuracy={m:totals[m]/count for m in METRICS},pass_at_k=curves,
        math_verify_parse_empty=parse_empty,finish_reason_length=truncated,
        raw_sha256=sha(folder/'responses.jsonl'),scoring_note='Five operational scorers; Math-Verify is not human chain ground truth.')
    dump(folder/'summary.json',summary)
    dump(folder/'SCORING_COMPLETE.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),summary_sha256=sha(folder/'summary.json')))
    return str(folder)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--workers',type=int,default=8);args=ap.parse_args()
    root=Path(args.root);queued=set();failed={};futures={}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while True:
            for marker in root.glob('results/*/*/GENERATION_COMPLETE.json'):
                p=marker.parent
                if p not in queued and not (p/'SCORING_COMPLETE.json').exists():
                    futures[pool.submit(score,str(p))]=p;queued.add(p)
            for future,p in list(futures.items()):
                if future.done():
                    try:print('SCORED',future.result(),flush=True)
                    except Exception:
                        failed[str(p)]=traceback.format_exc();dump(p/'SCORING_FAILED.json',dict(error=failed[str(p)]))
                    del futures[future]
            dump(root/'SCORER_STATUS.json',dict(active=len(futures),queued=len(queued),failed=failed,
                 completed=len(list(root.glob('results/*/*/SCORING_COMPLETE.json')))))
            if (root/'INFERENCE_FINISHED.json').exists() and not futures:break
            time.sleep(5)
    if failed:raise SystemExit(1)

if __name__=='__main__':main()
