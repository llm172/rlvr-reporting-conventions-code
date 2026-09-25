#!/usr/bin/env python
r"""Build the \boxed{} twin of the frozen GSM8K training data.

One variable changes: the trailing instruction in the user turn. Everything
else -- row order, difficulty_bin, frozen_baseline_solve_count, ground truth,
data_source -- is copied through untouched, so the curriculum sampler under
seed 83 draws the same items in the same order in both arms.

data_source deliberately stays "openai/gsm8k". It is NOT retargeted at a MATH
source, for two reasons: verl derives the validation metric key from it, so
keeping it makes the two runs' console logs line-for-line comparable
(val-core/openai/gsm8k/reward/mean@1 in both); and the MATH scorer does
expression equivalence rather than exact match, which would have changed the
matching rule as well as the marker. The boxed reward is supplied instead
through custom_reward_function.path.

Known and intended asymmetry: difficulty_bin and frozen_baseline_solve_rate
were measured under the #### convention, so the strata are "wrong" for boxed in
an absolute sense. Recomputing them would change the curriculum and break the
match. Holding them fixed is what makes this a one-variable contrast, and the
prereg says so.

Refuses to write anything if a single row does not carry the exact expected
suffix, because a silent partial rewrite would produce a mixed-convention
training set whose numbers would look plausible and mean nothing.
"""
import hashlib
import json
import os
import sys

import pandas as pd

SRC = os.environ.get("HASH_DATA_DIR", "data/gsm8k_hash")
DST = os.environ.get("BOXED_DATA_DIR", "data/gsm8k_boxed")

HASH_SUFFIX = ' Let\'s think step by step and output the final answer after "####".'
BOXED_SUFFIX = " Let's think step by step and output the final answer within \\boxed{}."

FILES = ["train_stratified.parquet", "validation_frozen.parquet"]
CARRY = ["data_source", "ability", "reward_model", "extra_info",
         "difficulty_bin", "frozen_baseline_solve_count",
         "frozen_baseline_solve_rate"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


os.makedirs(DST, exist_ok=True)
manifest = {"source_dir": SRC, "dest_dir": DST,
            "hash_suffix": HASH_SUFFIX, "boxed_suffix": BOXED_SUFFIX,
            "files": []}

for fn in FILES:
    sp, dp = os.path.join(SRC, fn), os.path.join(DST, fn)
    df = pd.read_parquet(sp)
    n = len(df)

    bad = []
    new_prompts = []
    for i, pr in enumerate(df["prompt"]):
        pr = list(pr)
        if len(pr) != 1 or pr[0].get("role") != "user":
            bad.append((i, "unexpected prompt shape"))
            new_prompts.append(pr)
            continue
        c = pr[0]["content"]
        if not c.endswith(HASH_SUFFIX):
            bad.append((i, "missing expected #### suffix"))
            new_prompts.append(pr)
            continue
        new_prompts.append([{"role": "user",
                             "content": c[:-len(HASH_SUFFIX)] + BOXED_SUFFIX}])

    if bad:
        print("REFUSING to write %s: %d/%d rows did not carry the exact "
              "#### suffix" % (fn, len(bad), n))
        for i, why in bad[:5]:
            print("    row %d: %s" % (i, why))
        sys.exit(2)

    out = df.copy()
    out["prompt"] = new_prompts

    # every non-prompt column must survive bit-for-bit
    for c in CARRY:
        if not df[c].equals(out[c]):
            print("REFUSING: column %s changed in %s" % (c, fn))
            sys.exit(2)

    out.to_parquet(dp, index=False)

    # read back and re-verify rather than trusting the write
    rb = pd.read_parquet(dp)
    assert len(rb) == n, (len(rb), n)
    n_boxed = sum(1 for p in rb["prompt"] if p[0]["content"].endswith(BOXED_SUFFIX))
    n_hash = sum(1 for p in rb["prompt"] if "####" in p[0]["content"])
    strata_src = df["difficulty_bin"].value_counts().sort_index().to_dict()
    strata_dst = rb["difficulty_bin"].value_counts().sort_index().to_dict()
    gt_same = list(df["reward_model"].map(lambda r: r["ground_truth"])) == \
              list(rb["reward_model"].map(lambda r: r["ground_truth"]))

    print("%-28s rows=%-6d boxed_suffix=%-6d residual_####=%d gt_identical=%s"
          % (fn, len(rb), n_boxed, n_hash, gt_same))
    print("    strata src=%s dst=%s" % (strata_src, strata_dst))
    if n_boxed != n or n_hash != 0 or not gt_same or strata_src != strata_dst:
        print("REFUSING: readback verification failed for %s" % fn)
        sys.exit(2)

    manifest["files"].append({
        "name": fn, "rows": int(n),
        "src_sha256": sha256(sp), "dst_sha256": sha256(dp),
        "strata": {str(k): int(v) for k, v in strata_dst.items()},
        "ground_truth_identical_to_source": bool(gt_same),
    })

mp = os.path.join(DST, "DATA_MANIFEST.json")
json.dump(manifest, open(mp, "w"), indent=2)
os.chmod(mp, 0o444)
print("\nwrote %s" % mp)
print("data manifest sha256 %s" % sha256(mp))
