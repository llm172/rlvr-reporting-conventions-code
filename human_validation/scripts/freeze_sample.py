"""One-time stratified sample from frozen caches. Never reads annotation labels."""
from pathlib import Path
import collections, csv, datetime, hashlib, json, os, random, sys

PKG = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('HUMAN_EVIDENCE_ROOT', Path.cwd()))
SEED = 'human-semantic-625-20260912-v1'
FILES = {}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
def read(p, lines=False):
    p=ROOT/p
    FILES[p.relative_to(ROOT).as_posix()] = sha(p)
    return [json.loads(x) for x in p.read_text(encoding='utf-8-sig').splitlines()] if lines else json.loads(p.read_text(encoding='utf-8-sig'))
def unique(rows, key):
    d={key(r):r for r in rows}
    if len(d)!=len(rows): raise ValueError('Duplicate source identity')
    return d
def sample(groups, cap, state):
    chosen=[]
    for h,pop in sorted(groups.items()):
        rng=random.Random(SEED+':'+state+':'+h)
        n=min(cap,len(pop))
        for r in rng.sample(sorted(pop,key=lambda x:str(x['original_id'])),n):
            chosen.append(dict(r,stratum=h,stratum_population=len(pop),stratum_sample_n=n,
                               inclusion_probability=n/len(pop),weight=len(pop)/n,
                               mean_weight=len(pop)/n/r['population_n']))
    return chosen

def main():
    target=PKG/'coordinator/frozen_sample.json'
    if target.exists():
        receipt=json.loads((PKG/'coordinator/SAMPLE_FROZEN.json').read_text())
        assert sha(target)==receipt['sample_sha256'], 'Frozen sample altered'
        print('Already frozen: refusing to resample. Existing hash verified.');return
    allrows=[];populations=[];selected=[]
    for state in ['qwen15_base','qwen15_rl','smol_base','smol_rl']:
        base=Path('experiment_followup_2026-09-09/evidence')
        folder=base/'results'/state/'gsm8k_control_t0.6'
        task=read(folder/'task.json')
        ip=base/'inputs'/Path(task['task']['items']).name
        items=unique(read(ip,True),lambda r:r['id'])
        assert FILES[ip.as_posix()]==task['task']['items_sha256']
        raw=unique(read(folder/'responses.jsonl',True),lambda r:r['id'])
        receipt=read(folder/'GENERATION_COMPLETE.json')
        assert FILES[(folder/'responses.jsonl').as_posix()]==receipt['raw_sha256']
        scores=read(folder/'response_scores.jsonl',True)
        groups=collections.defaultdict(list)
        for s in scores:
            i=s['id'];j=s['sample'];item=items[i];response=raw[i]['responses'][j]
            question=item['prompt'].split('<|im_start|>user\n',1)[1].split('<|im_end|>',1)[0]
            suffix=' Let\'s think step by step and output the final answer after "####".'
            assert question.endswith(suffix)
            question=question[:-len(suffix)]
            first=s['extracted_first_hash'];fall=first if first is not None else s['extracted_last_number']
            readers={'strict':{'score':int(s['strict_first_hash']),'extracted':first},
                     'fallback':{'score':int(s['frozen_lenient']),'extracted':fall},
                     'mv':{'score':int(s['math_verify']),'extracted':None}}
            r=dict(dataset='GSM8K',state=state,original_id=i,sample_index=j,
                model='Qwen2.5-1.5B-Instruct' if state.startswith('qwen') else 'SmolLM2-1.7B-Instruct',
                checkpoint=task['model'],phase='initial' if state.endswith('base') else 'RL',
                seed=None if state.endswith('base') else 83,generation_seed=task['task']['seed'],
                training_convention='NOT_APPLICABLE' if state.endswith('base') else 'hash reward',
                population_n=1319,question=question,response=response,reference_answer=str(item['ground_truth']),
                readers=readers,finish_reason=raw[i]['finish_reasons'][j],source_response=(folder/'responses.jsonl').as_posix(),
                source_scores=(folder/'response_scores.jsonl').as_posix(),source_task=(folder/'task.json').as_posix(),
                source_items=ip.as_posix(),response_sha256=hashlib.sha256(response.encode()).hexdigest())
            h=''.join(str(readers[k]['score']) for k in ['strict','fallback','mv']);groups[h].append(r)
        assert len(scores)==len(raw)==len(items)==1319
        rs=[r for g in groups.values() for r in g];allrows+=rs;selected+=sample(groups,30,state)
        populations.append(dict(dataset='GSM8K',state=state,model=rs[0]['model'],N=1319,
            reader_accuracy={k:sum(r['readers'][k]['score'] for r in rs)/1319 for k in readers},
            strata={h:{'N':len(g),'n':min(30,len(g))} for h,g in sorted(groups.items())}))
    b=Path('qwen7_seed_replication_2026-09-10/evidence')
    for state,loc,source_state in [('initial','baseline_evidence','initial'),('seed83','baseline_evidence','hash_trained'),('seed84','evaluation','hash_trained84'),('seed85','evaluation','hash_trained85')]:
        base=b/loc/'math500';protocol=read(base/'protocol.json')
        folder=base/'results'/source_state/'hash';task=read(folder/'task.json')
        ip=base/'inputs/math500_hash.jsonl'
        items=unique(read(ip,True),lambda r:r['id'])
        assert FILES[ip.as_posix()]==task['items_sha256']
        raw=unique(read(folder/'responses.jsonl',True),lambda r:r['id'])
        scores=read(folder/'scores.jsonl',True);groups=collections.defaultdict(list)
        for s in scores:
            i=s['id'];item=items[i];response=raw[i]['response']
            assert item['source_id']==s['source_id']==raw[i]['source_id']
            readers={'strict':{'score':int(s['strict']),'extracted':s['requested_parsed']},
                     'fallback':None,'mv':{'score':int(s['math_verify']),'extracted':s['mv_parsed']}}
            r=dict(dataset='MATH500',state=state,original_id=item['source_id'],sample_index=0,
                model='Qwen2.5-7B-Instruct',checkpoint=protocol['models'][source_state]['path'],
                phase='initial' if state=='initial' else 'RL',seed=None if state=='initial' else int(state[4:]),
                generation_seed=raw[i]['seed'],training_convention='NOT_APPLICABLE' if state=='initial' else 'hash reward',
                population_n=500,question=item['question'],response=response,reference_answer=item['ground_truth'],
                readers=readers,finish_reason=raw[i]['finish_reason'],source_response=(folder/'responses.jsonl').as_posix(),
                source_scores=(folder/'scores.jsonl').as_posix(),source_task=(folder/'task.json').as_posix(),
                source_protocol=(base/'protocol.json').as_posix(),source_items=ip.as_posix(),response_sha256=hashlib.sha256(response.encode()).hexdigest(),
                hash_payload=s['hash_payload'],boxed_payload=s['boxed_payload'])
            h=str(readers['strict']['score'])+str(readers['mv']['score']);groups[h].append(r)
        assert len(scores)==len(raw)==len(items)==500
        rs=[r for g in groups.values() for r in g];allrows+=rs;selected+=sample(groups,20,state)
        populations.append(dict(dataset='MATH500',state=state,model=rs[0]['model'],N=500,
            reader_accuracy={k:sum(r['readers'][k]['score'] for r in rs)/500 for k in ['strict','mv']},
            strata={h:{'N':len(g),'n':min(20,len(g))} for h,g in sorted(groups.items())}))
    # Capture GSM MV candidates with exact frozen version; parity is checked, scores never replaced.
    from math_verify import parse, verify
    import importlib.metadata
    assert importlib.metadata.version('math-verify')=='0.9.0'
    for r in selected:
        if r['dataset']=='GSM8K':
            parsed=parse(r['response'],parsing_timeout=0)
            gold=parse('$\\boxed{'+r['reference_answer']+'}$',parsing_timeout=0)
            got=bool(verify(gold,parsed,timeout_seconds=0)) if parsed else False
            assert got==bool(r['readers']['mv']['score']), ('MV parity mismatch',r['state'],r['original_id'])
            r['readers']['mv']['extracted']=[str(x) for x in parsed]
            r['mv_candidate_provenance']='Math-Verify 0.9.0 default extraction, timeout disabled on Windows, frozen score parity checked'
        r['sample_id']='H'+hashlib.sha256((SEED+':'+r['state']+':'+str(r['original_id'])).encode()).hexdigest()[:10].upper()
        r['candidate_order']=[k for k,v in r['readers'].items() if v is not None]
        random.Random(SEED+':candidates:'+r['sample_id']).shuffle(r['candidate_order'])
    random.Random(SEED+':master-order').shuffle(selected)
    counts=dict(collections.Counter(r['state'] for r in selected))
    assert counts==dict(qwen15_base=105,qwen15_rl=61,smol_base=106,smol_rl=82,initial=69,seed83=62,seed84=70,seed85=70)
    assert len({r['sample_id'] for r in selected})==625
    assert len({(r['dataset'],r['state'],r['original_id'],r['sample_index']) for r in selected})==625
    assert all(r['question'] and r['response'] and r['checkpoint'] and r['reference_answer'] for r in selected)
    disagreement=lambda r:r['dataset']=='GSM8K' and r['readers']['fallback']['score']!=r['readers']['mv']['score']
    assert sum(map(disagreement,selected))==sum(map(disagreement,allrows))==36
    for pop in populations:
        assert abs(sum(r['weight'] for r in selected if r['state']==pop['state'])-pop['N'])<1e-8
    duplicate_text=[v for v in collections.defaultdict(list).values()]
    bytext=collections.defaultdict(list)
    for r in selected: bytext[r['response_sha256']].append(r['sample_id'])
    duplicate_text=[v for v in bytext.values() if len(v)>1]
    dump(target,selected);dump(PKG/'coordinator/population.json',populations)
    dump(PKG/'coordinator/source_manifest.json',FILES)
    dump(PKG/'coordinator/SAMPLE_FROZEN.json',dict(status='SAMPLE_FROZEN_PROTOCOL_PILOT_PENDING',seed=SEED,
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sample_sha256=sha(target),population_sha256=sha(PKG/'coordinator/population.json'),
        counts=counts,total=625,formal_human_labels=0,fallback_mv_disagreements=36,
        duplicate_keys=0,duplicate_response_text_groups=duplicate_text,max_response_characters=max(len(r['response']) for r in selected),
        missing_response=0,mv_candidate_parity_checked=354))
    with (PKG/'coordinator/strata.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['dataset','state','stratum','population_n','sample_n','inclusion_probability','weight'])
        for p in populations:
            for h,s in p['strata'].items():w.writerow([p['dataset'],p['state'],h,s['N'],s['n'],s['n']/s['N'],s['N']/s['n']])
    print(json.dumps(json.loads((PKG/'coordinator/SAMPLE_FROZEN.json').read_text()),indent=2))

if __name__=='__main__': main()
