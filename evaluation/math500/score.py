"""Matched symbolic strict readers plus convention-independent full-response MV.

Both symbolic comparisons use Math-Verify; independence concerns extraction,
not an independently validated semantic oracle. No last-number substitution.
"""
import argparse,json,os,re,statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from io_helpers import durable,sha,read,rows,require,now

METRICS=['strict','math_verify','hash_strict','boxed_strict','requested_extractable','requested_symbolic_parseable','mv_extractable','mv_symbolic_parseable','truncated']

def boxed_payload(text,last=False):
 matches=list(re.finditer(r'\\boxed\s*\{',text))
 if not matches:return None
 m=matches[-1] if last else matches[0];depth=1;out=[]
 for c in text[m.end():]:
  if c=='{':depth+=1
  if c=='}':
   depth-=1
   if not depth:return ''.join(out).strip() or None
  out.append(c)
 return None

def hash_payload(text,last=False):
 # First occurrence, same-line payload; no answer scavenging across blank lines.
 matches=list(re.finditer(r'####[^\S\n]*([^\n]*)',text))
 if not matches:return None
 return (matches[-1] if last else matches[0]).group(1).strip() or None

def unmath(text):
 text=text.strip()
 for a,b in [('$$','$$'),('$','$'),(r'\(',r'\)'),(r'\[',r'\]')]:
  if text.startswith(a) and text.endswith(b):return text[len(a):-len(b)].strip()
 return text

def symbolic(payload):
 if payload is None:return []
 from math_verify import parse,LatexExtractionConfig
 return parse('$'+unmath(payload)+'$',extraction_config=[LatexExtractionConfig(try_extract_without_anchor=True)],fallback_mode='no_fallback',extraction_mode='first_match',parsing_timeout=5)

def score_one(arg):
 record,item,conv=arg
 from math_verify import parse,verify,LatexExtractionConfig,ExprExtractionConfig
 gold=symbolic(item['ground_truth']);require(bool(gold),'Unparseable symbolic reference: '+str(item['source_id']))
 text=record['response'];hp=hash_payload(text);bp=boxed_payload(text)
 hs=symbolic(hp);bs=symbolic(bp)
 full=parse(text,extraction_config=[LatexExtractionConfig(),ExprExtractionConfig()],fallback_mode='first_match',extraction_mode='any_match',parsing_timeout=5)
 equal=lambda a:bool(verify(gold,a,timeout_seconds=5)) if a else False
 requested=hs if conv=='hash' else bs;payload=hp if conv=='hash' else bp
 return dict(id=record['id'],source_id=item['source_id'],strict=equal(requested),math_verify=equal(full),hash_strict=equal(hs),boxed_strict=equal(bs),
  requested_extractable=payload is not None,requested_symbolic_parseable=bool(requested),mv_extractable=bool(full),mv_symbolic_parseable=any(not isinstance(x,str) for x in full),
  truncated=record['finish_reason']=='length',hash_payload=hp,boxed_payload=bp,gold_parsed=[str(x) for x in gold],
  requested_parsed=[str(x) for x in requested],mv_parsed=[str(x) for x in full],output_tokens=record['output_tokens'])

def validate_raw(raw,n):
 require(len(raw)==n and len({x['id'] for x in raw})==n and {x['id'] for x in raw}==set(range(n)),'Incomplete or duplicate responses')
 require(all(isinstance(x['response'],str) and x['finish_reason'] in ['stop','length'] for x in raw),'Malformed response')

def score_cell(root,state,conv,workers=8):
 folder=root/'results'/state/conv;p=read(root/'protocol.json');source=root/f'inputs/math500_{conv}.jsonl'
 require(sha(source)==p['inputs'][source.name],'Input changed')
 items={x['id']:x for x in rows(source)};raw=list(rows(folder/'responses.jsonl'));validate_raw(raw,500)
 require(sha(folder/'responses.jsonl')==read(folder/'GENERATION_COMPLETE.json')['raw_sha256'],'Raw changed')
 require(not (folder/'SCORING_COMPLETE.json').exists(),'Already scored')
 args=[(r,items[r['id']],conv) for r in raw]
 with ProcessPoolExecutor(max_workers=workers) as pool:scored=list(pool.map(score_one,args,chunksize=4))
 with open(folder/'scores.jsonl','w',encoding='utf-8') as f:
  for r in scored:f.write(json.dumps(r,ensure_ascii=False)+'\n')
  f.flush();os.fsync(f.fileno())
 summary=dict(state=state,convention=conv,items=500,counts={m:sum(x[m] for x in scored) for m in METRICS},
   rates={m:sum(x[m] for x in scored)/500 for m in METRICS},output_tokens=sum(x['output_tokens'] for x in raw),
   median_output_tokens=statistics.median(x['output_tokens'] for x in raw),raw_sha256=sha(folder/'responses.jsonl'),scores_sha256=sha(folder/'scores.jsonl'),
   note='Strict: first required marker payload, common symbolic equivalence. Full-response Math-Verify: default independent extraction; same equivalence package. Neither establishes chain validity.')
 durable(folder/'summary.json',summary);durable(folder/'SCORING_COMPLETE.json',dict(utc=now(),summary_sha256=sha(folder/'summary.json')))
 print(json.dumps(summary),flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--state',required=True);args=a.parse_args()
 for c in ['hash','boxed']:score_cell(args.root,args.state,c)
