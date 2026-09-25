#!/usr/bin/env python
"""Build two content-matched SFT sets that differ in the final marker.

Arm H target: the GSM8K solution verbatim, final line "#### N".
Arm B target: the SAME solution with only that final line replaced by
"\boxed{N}". Every preceding byte is identical; this is asserted, not assumed.

Also checks for training/evaluation overlap and refuses to write
anything if the SFT questions intersect the 500 eval questions.
"""
import hashlib
import io
import json
import os
import random
import re
import sys

import pandas as pd

SRC = os.environ.get("GSM8K_TRAIN_PARQUET", "data/gsm8k_train.parquet")
ITEMS = os.environ.get("EVAL_ITEMS_DIR", "data/eval_items")
OUT = os.environ.get("SFT_OUTPUT_DIR", "data/sft")
SEED = 83
# Preserve the literal backslash in the target marker.
BOXED_OPEN = chr(92) + "boxed{"

os.makedirs(OUT, exist_ok=True)


def norm(s):
    """Whitespace-normalised form used for the contamination comparison."""
    return re.sub(r"\s+", " ", s).strip().lower()


df = pd.read_parquet(SRC)
sha = hashlib.sha256(io.open(SRC, "rb").read()).hexdigest()
print("source rows=%d sha256=%s" % (len(df), sha))

rows = []
for _, r in df.iterrows():
    ei = r["extra_info"]
    q, a = ei["question"], ei["answer"]
    if "\n#### " not in a:
        sys.exit("REFUSING: row without a ####  final line: %.80r" % a)
    body, final = a.rsplit("\n#### ", 1)
    final = final.strip()
    if not final:
        sys.exit("REFUSING: empty gold answer")
    rows.append({"question": q, "body": body, "answer": final})

print("parsed %d rows" % len(rows))

# ---- gate 1: contamination -------------------------------------------------
train_q = set(norm(r["question"]) for r in rows)
eval_q = set()
for f in ("gsm8k500_qwen_hash.jsonl", "gsm8k500_qwen_boxed.jsonl"):
    with io.open(os.path.join(ITEMS, f), encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                eval_q.add(norm(json.loads(line)["meta"]["question"]))
overlap = sorted(train_q & eval_q)
check = {
    "n_train_questions": len(train_q),
    "n_eval_questions": len(eval_q),
    "n_overlap": len(overlap),
    "source_parquet_sha256": sha,
    "examples": overlap[:5],
}
with io.open(os.path.join(OUT, "contamination_check.json"), "w", encoding="utf-8") as fh:
    json.dump(check, fh, indent=2)
print("contamination: train=%d eval=%d overlap=%d" % (len(train_q), len(eval_q), len(overlap)))
if overlap:
    sys.exit("REFUSING: SFT/eval question overlap of %d (gate 1)" % len(overlap))

# ---- one shuffle, reused by both arms --------------------------------------
random.Random(SEED).shuffle(rows)

for arm, render in (("hash",  lambda r: r["body"] + "\n#### " + r["answer"]),
                    ("boxed", lambda r: r["body"] + chr(10) + BOXED_OPEN + r["answer"] + "}")):
    path = os.path.join(OUT, "sft_%s.jsonl" % arm)
    with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps({"question": r["question"],
                                 "completion": render(r)}, ensure_ascii=False) + "\n")
    print("wrote %s (%d rows)" % (path, len(rows)))

# ---- assert the arms differ ONLY in the final marker -----------------------
h = [json.loads(l) for l in io.open(os.path.join(OUT, "sft_hash.jsonl"), encoding="utf-8")]
b = [json.loads(l) for l in io.open(os.path.join(OUT, "sft_boxed.jsonl"), encoding="utf-8")]
assert len(h) == len(b) == len(rows), "arm length mismatch"
for i, (x, y) in enumerate(zip(h, b)):
    if x["question"] != y["question"]:
        sys.exit("REFUSING: question mismatch at row %d" % i)
    hb = x["completion"].rsplit("\n#### ", 1)[0]
    bb = y["completion"].rsplit(chr(10) + BOXED_OPEN, 1)[0]
    if hb != bb:
        sys.exit("REFUSING: bodies differ at row %d -- arms are not content-matched" % i)
print("VERIFIED: %d rows, bodies byte-identical, only the final marker differs" % len(h))
print("example H:", repr(h[0]["completion"][-60:]))
print("example B:", repr(b[0]["completion"][-60:]))
