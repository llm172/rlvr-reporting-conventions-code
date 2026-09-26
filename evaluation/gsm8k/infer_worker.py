"""One model per GPU, immutable task manifests, exact item coverage, retained raw output."""
import argparse
import datetime
import json
import os
from pathlib import Path
import time
import traceback
from extension_common import dump,sha,read_jsonl,validate_rows

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);args=ap.parse_args()
    manifest_path=Path(args.manifest);manifest=json.loads(manifest_path.read_text())
    root=manifest_path.parent;state=manifest['state'];out=root/'results'/state
    out.mkdir(parents=True,exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES']=str(manifest['gpu'])
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    status=out/'WORKER_STATUS.json'
    try:
        dump(status,dict(state='starting',utc=now(),pid=os.getpid(),manifest_sha256=sha(manifest_path)))
        from transformers import AutoTokenizer
        from vllm import LLM,SamplingParams
        tokenizer=AutoTokenizer.from_pretrained(manifest['model'],trust_remote_code=True)
        # Validate all prompts before any formal generation; never silently truncate.
        for task in manifest['tasks']:
            assert sha(task['items'])==task['items_sha256'],'Input hash mismatch'
            lengths=[len(tokenizer.encode(x['prompt'],add_special_tokens=False)) for x in read_jsonl(task['items'])]
            if max(lengths)+task['max_tokens']>task['max_model_len']:
                raise ValueError(f"Prompt exceeds context budget in {task['name']}: {max(lengths)}")
        max_model_len=max(task['max_model_len'] for task in manifest['tasks'])
        llm=LLM(model=manifest['model'],dtype='bfloat16',tensor_parallel_size=1,
                gpu_memory_utilization=.80,max_model_len=max_model_len,max_num_seqs=256,
                trust_remote_code=True,enable_prefix_caching=True,enforce_eager=True,seed=20260908)
        for task in manifest['tasks']:
            folder=out/task['name'];folder.mkdir(exist_ok=True)
            meta=dict(task=task,model=manifest['model'],model_metadata=manifest['model_metadata'],
                      state=state,manifest_sha256=sha(manifest_path))
            meta_path=folder/'task.json'
            if meta_path.exists() and json.loads(meta_path.read_text())!=meta:
                raise ValueError('Refusing resume under changed task manifest')
            dump(meta_path,meta)
            rows=read_jsonl(task['items']);ids={x['id'] for x in rows}
            assert len(rows)==len(ids),'Duplicate input IDs'
            raw=folder/'responses.jsonl'
            previous=read_jsonl(raw) if raw.exists() else []
            done=validate_rows(previous,ids,task['n'])
            pending=[x for x in rows if x['id'] not in done]
            dump(status,dict(state='generating',task=task['name'],utc=now(),done=len(done),total=len(rows),pid=os.getpid()))
            start=time.monotonic()
            with open(raw,'a',encoding='utf-8') as sink,open(folder/'timing.jsonl','a',encoding='utf-8') as timing:
                for a in range(0,len(pending),task['batch_items']):
                    batch=pending[a:a+task['batch_items']]
                    params=[SamplingParams(n=task['n'],temperature=task['temperature'],top_p=task['top_p'],
                              max_tokens=task['max_tokens'],seed=task['seed']+int(x['id'])) for x in batch]
                    t=time.monotonic();outputs=llm.generate([x['prompt'] for x in batch],params,use_tqdm=False)
                    elapsed=time.monotonic()-t;output_tokens=0;input_tokens=0
                    if len(outputs)!=len(batch):raise ValueError('Generation count mismatch')
                    for item,result in zip(batch,outputs):
                        record=dict(id=item['id'],n=len(result.outputs),responses=[x.text for x in result.outputs],
                            finish_reasons=[x.finish_reason for x in result.outputs],
                            output_token_counts=[len(x.token_ids) for x in result.outputs],input_tokens=len(result.prompt_token_ids))
                        validate_rows([record],ids,task['n'])
                        output_tokens+=sum(record['output_token_counts']);input_tokens+=record['input_tokens']
                        sink.write(json.dumps(record,ensure_ascii=False)+'\n')
                    sink.flush();os.fsync(sink.fileno())
                    timing.write(json.dumps(dict(item_ids=[x['id'] for x in batch],seconds=elapsed,
                          input_tokens=input_tokens,output_tokens=output_tokens,utc=now()))+'\n');timing.flush()
                    covered=len(done)+a+len(batch)
                    dump(status,dict(state='generating',task=task['name'],utc=now(),done=covered,total=len(rows),pid=os.getpid()))
                    print(f"{state} {task['name']}: {covered}/{len(rows)}, {output_tokens/max(elapsed,.01):.0f} output tok/s",flush=True)
            validate_rows(read_jsonl(raw),ids,task['n'],complete=True)
            dump(folder/'GENERATION_COMPLETE.json',dict(utc=now(),items=len(rows),samples_per_item=task['n'],
                 raw_sha256=sha(raw),task_sha256=sha(meta_path),session_seconds=time.monotonic()-start))
        dump(status,dict(state='generation_complete',utc=now(),pid=os.getpid()))
    except Exception:
        dump(status,dict(state='failed',utc=now(),pid=os.getpid(),error=traceback.format_exc()))
        raise

if __name__=='__main__':main()
