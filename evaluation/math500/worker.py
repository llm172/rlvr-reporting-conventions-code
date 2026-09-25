"""Single-GPU inference. Pilot and formal responses are strictly separated."""
import argparse,json,os,time
from pathlib import Path
from io_helpers import durable,sha,read,rows,require,now

def main():
 a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--state',required=True);a.add_argument('--pilot',action='store_true');args=a.parse_args()
 root=args.root;p=read(root/'protocol.json');state=args.state
 base=root/('pilot' if args.pilot else 'results')/state;base.mkdir(parents=True,exist_ok=True)
 status=base/'STATUS.json';os.environ.setdefault('CUDA_VISIBLE_DEVICES','0');os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['OMP_NUM_THREADS']='4'
 from vllm import LLM,SamplingParams
 durable(status,dict(stage='loading_model',utc=now(),pid=os.getpid()))
 llm=LLM(model=p['models'][state]['path'],tokenizer=p['models']['initial']['path'],max_model_len=p['max_model_len'],seed=p['seed'],**p['inference'])
 for conv in p['conventions']:
  task=base/conv;task.mkdir(exist_ok=True)
  require(not (task/'responses.jsonl').exists(),'Refusing to append/restart a formal response stream')
  name=('pilot' if args.pilot else 'math500')+'_'+conv+'.jsonl';source=root/'inputs'/name
  require(sha(source)==p['inputs'][name],'Input hash changed')
  items=list(rows(source));require(len(items)==(32 if args.pilot else 500),'Item coverage')
  durable(task/'task.json',dict(state=state,convention=conv,pilot=args.pilot,protocol_sha256=sha(root/'protocol.json'),items_sha256=sha(source)))
  start=time.monotonic()
  with open(task/'responses.jsonl','w',encoding='utf-8') as f,open(task/'timing.jsonl','w') as tf:
   for j in range(0,len(items),p['batch_items']):
    batch=items[j:j+p['batch_items']];t=time.monotonic()
    pars=[SamplingParams(n=1,temperature=p['temperature'],top_p=p['top_p'],max_tokens=p['max_tokens'],seed=p['seed']+x['id']) for x in batch]
    outputs=llm.generate([x['prompt'] for x in batch],pars,use_tqdm=False)
    require(len(outputs)==len(batch),'Generation count mismatch')
    total=0
    for x,out in zip(batch,outputs):
     require(len(out.outputs)==1,'Expected one sample');o=out.outputs[0]
     require(o.finish_reason in ['stop','length'],'Unexpected finish reason')
     require(len(out.prompt_token_ids)==x['input_tokens'],'Tokenization drift')
     record=dict(id=x['id'],source_id=x['source_id'],response=o.text,finish_reason=o.finish_reason,stop_reason=o.stop_reason,
       output_tokens=len(o.token_ids),input_tokens=len(out.prompt_token_ids),seed=p['seed']+x['id'])
     total+=record['output_tokens'];f.write(json.dumps(record,ensure_ascii=False)+'\n')
    f.flush();os.fsync(f.fileno());dt=time.monotonic()-t
    tf.write(json.dumps(dict(ids=[x['id'] for x in batch],seconds=dt,output_tokens=total,utc=now()))+'\n');tf.flush();os.fsync(tf.fileno())
    durable(status,dict(stage='generating',state=state,convention=conv,done=j+len(batch),total=len(items),utc=now(),pid=os.getpid()))
    print(f'{state}/{conv}: {j+len(batch)}/{len(items)}; {total/dt:.0f} output tokens/s',flush=True)
  result=list(rows(task/'responses.jsonl'));require(len(result)==len(items) and {x['id'] for x in result}==set(range(len(items))),'Missing/duplicate output')
  durable(task/'GENERATION_COMPLETE.json',dict(utc=now(),items=len(items),raw_sha256=sha(task/'responses.jsonl'),seconds=time.monotonic()-start,
    truncated=sum(x['finish_reason']=='length' for x in result),output_tokens=sum(x['output_tokens'] for x in result)))
 durable(status,dict(stage='complete',utc=now(),pid=os.getpid()))

if __name__=='__main__':main()
