"""Pure measurement/data helpers for the prospective September 8 extension."""
import ast
import hashlib
import json
import random
import re
from fractions import Fraction
from pathlib import Path

HASH=re.compile(r'####\s*(-?\d+\.?\d*)')
NUMBER=re.compile(r'-?\d+\.?\d*')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    tmp.replace(path)

def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]

def write_jsonl(path,rows):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')

def equal(a,b):
    if a is None or b is None:return False
    try:return abs(float(a)-float(b))<1e-3
    except (ValueError,TypeError):return str(a).strip()==str(b).strip()

def extract_scores(response,gold):
    text=response.replace(',','')
    hashes=HASH.findall(text);numbers=NUMBER.findall(text)
    reference_hash=HASH.findall(str(gold).replace(',',''))
    reference_numbers=NUMBER.findall(str(gold).replace(',',''))
    reference=reference_hash[0] if reference_hash else reference_numbers[-1] if reference_numbers else str(gold)
    first=hashes[0] if hashes else None
    last=hashes[-1] if hashes else None
    number=numbers[-1] if numbers else None
    fallback=first if first is not None else number
    return dict(strict_first_hash=equal(first,reference),strict_last_hash=equal(last,reference),
                frozen_lenient=equal(fallback,reference),last_number=equal(number,reference),
                hash_parseable=first is not None,marker_present='####' in response,
                extracted_first_hash=first,extracted_last_hash=last,extracted_last_number=number)

def validate_rows(rows,ids,n,complete=False):
    seen=set()
    for row in rows:
        i=row['id']
        if i not in ids or i in seen:raise ValueError('Unexpected/duplicate item id: '+str(i))
        seen.add(i)
        if row['n']!=n or len(row['responses'])!=n or len(row['finish_reasons'])!=n:
            raise ValueError('Sample count mismatch on item '+str(i))
        if not all(isinstance(x,str) for x in row['responses']):raise ValueError('Non-string response')
    if complete and seen!=ids:raise ValueError('Missing items: '+str(len(ids-seen)))
    return seen

def decompose(bs,bl,rs,rl):
    g0=bl-bs;g1=rl-rs;app=rs-bs
    return dict(base_strict=bs,base_lenient=bl,rl_strict=rs,rl_lenient=rl,
                strict_gain=app,lenient_gain=rl-bl,initial_gap=g0,residual_gap=g1,
                net_gap_reduction=g0-g1,initial_gap_share=g0/app if app else None)

def safe_arithmetic(expression):
    def visit(node):
        if isinstance(node,ast.Expression):return visit(node.body)
        if isinstance(node,ast.Constant) and type(node.value)==int:return Fraction(node.value)
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):
            v=visit(node.operand);return -v if isinstance(node.op,ast.USub) else v
        if isinstance(node,ast.BinOp):
            a,b=visit(node.left),visit(node.right)
            if isinstance(node.op,ast.Add):return a+b
            if isinstance(node.op,ast.Sub):return a-b
            if isinstance(node.op,ast.Mult):return a*b
            if isinstance(node.op,ast.Div):return a/b
        raise ValueError('Unsupported arithmetic AST')
    return visit(ast.parse(expression,mode='eval'))

def arithmetic_items(seed=20260908):
    rng=random.Random(seed);used=set();splits=[]
    for split,count in [('dev',32),('test',500)]:
        rows=[]
        for i in range(count):
            steps=2+i%3
            while True:
                value=rng.randint(2,40);expression=str(value);ops=[]
                for _ in range(steps):
                    divisors=[d for d in range(2,13) if value and value%d==0]
                    op=rng.choice(['+','-','*']+(['/'] if divisors else []))
                    operand=rng.choice(divisors) if op=='/' else rng.randint(2,20)
                    if op=='+':value+=operand
                    elif op=='-':value-=operand
                    elif op=='*':value*=operand
                    else:value//=operand
                    expression=f'({expression} {op} {operand})';ops.append(op)
                if expression in used or abs(value)>10000:continue
                if safe_arithmetic(expression)!=value:raise ValueError('Independent oracle mismatch')
                used.add(expression);break
            rows.append(dict(id=i,question=f'Calculate the exact value of {expression}.',ground_truth=str(value),
                             meta=dict(split=split,seed=seed,steps=steps,expression=expression,operations=ops)))
        splits.append(rows)
    return tuple(splits)
