#!/usr/bin/env python
"""Content-matched SFT trainer -- one arm per invocation.

Every hyperparameter below is fixed by that registration and is identical
across the two arms; the ONLY difference between arms is the --data file, whose
completions differ from the other arm in exactly one marker.

Design note that matters: the training prompt is CONVENTION-NEUTRAL (the bare
question, no format instruction). We are manufacturing a habit, not teaching
instruction-following. Loss is masked to completion tokens only.

Checks enforced here: finite loss, final below first; arm independence --
refuses to overwrite a non-empty output directory.
"""
import argparse
import io
import json
import math
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

P = argparse.ArgumentParser()
P.add_argument("--data", required=True)
P.add_argument("--out", required=True)
P.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
P.add_argument("--seed", type=int, default=83)
P.add_argument("--epochs", type=int, default=2)
P.add_argument("--lr", type=float, default=1e-5)
P.add_argument("--warmup-ratio", type=float, default=0.03)
P.add_argument("--batch", type=int, default=8)
P.add_argument("--accum", type=int, default=4)
P.add_argument("--max-len", type=int, default=1024)
A = P.parse_args()

# ---- gate 5: arm independence ---------------------------------------------
if os.path.isdir(A.out) and any(os.scandir(A.out)):
    sys.exit("REFUSING: output dir %s exists and is non-empty (gate 5)" % A.out)
os.makedirs(A.out, exist_ok=True)

random.seed(A.seed)
np.random.seed(A.seed)
torch.manual_seed(A.seed)
torch.cuda.manual_seed_all(A.seed)

tok = AutoTokenizer.from_pretrained(A.base)
SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


class SFT(Dataset):
    def __init__(self, path):
        self.rows = [json.loads(l) for l in io.open(path, encoding="utf-8") if l.strip()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        # Same chat template and system prompt the eval items use, so the habit
        # transfers to eval-time prompts. The user turn carries NO format
        # instruction -- that is the manipulation.
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": r["question"]}],
            tokenize=False, add_generation_prompt=True)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tok(r["completion"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = (p_ids + c_ids)[: A.max_len]
        labels = ([-100] * len(p_ids) + c_ids)[: A.max_len]
        return {"input_ids": ids, "labels": labels}


def collate(batch):
    n = max(len(b["input_ids"]) for b in batch)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out = {"input_ids": [], "labels": [], "attention_mask": []}
    for b in batch:
        k = n - len(b["input_ids"])
        out["input_ids"].append(b["input_ids"] + [pad] * k)
        out["labels"].append(b["labels"] + [-100] * k)
        out["attention_mask"].append([1] * len(b["input_ids"]) + [0] * k)
    return {k: torch.tensor(v) for k, v in out.items()}


ds = SFT(A.data)
g = torch.Generator(); g.manual_seed(A.seed)
dl = DataLoader(ds, batch_size=A.batch, shuffle=True, collate_fn=collate,
                generator=g, drop_last=True, num_workers=2)

model = AutoModelForCausalLM.from_pretrained(
    A.base, torch_dtype=torch.float32, attn_implementation="sdpa").cuda()
model.gradient_checkpointing_enable()
model.config.use_cache = False

steps = (len(dl) // A.accum) * A.epochs
opt = torch.optim.AdamW(model.parameters(), lr=A.lr, weight_decay=0.0, betas=(0.9, 0.95))
sched = get_cosine_schedule_with_warmup(opt, int(steps * A.warmup_ratio), steps)
print("rows=%d  batches/epoch=%d  optimizer steps=%d" % (len(ds), len(dl), steps), flush=True)

log, first_loss, last_loss, step = [], None, None, 0
model.train()
for ep in range(A.epochs):
    for i, batch in enumerate(dl):
        batch = {k: v.cuda() for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch).loss
        if not torch.isfinite(loss):
            sys.exit("REFUSING: non-finite loss at epoch %d batch %d (gate 4)" % (ep, i))
        (loss / A.accum).backward()
        if (i + 1) % A.accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            v = float(loss.item())
            if first_loss is None:
                first_loss = v
            last_loss = v
            if step % 25 == 0 or step == 1:
                log.append({"step": step, "epoch": ep, "loss": v, "lr": sched.get_last_lr()[0]})
                print("step %4d/%d  ep %d  loss %.4f  lr %.2e" % (step, steps, ep, v, sched.get_last_lr()[0]), flush=True)

# ---- gate 4: training sanity ----------------------------------------------
if first_loss is None or last_loss is None or not math.isfinite(last_loss):
    sys.exit("REFUSING: no finite loss recorded (gate 4)")
if not (last_loss < first_loss):
    sys.exit("REFUSING: final loss %.4f not below first %.4f (gate 4)" % (last_loss, first_loss))

model.config.use_cache = True
model.to(torch.bfloat16).save_pretrained(A.out, safe_serialization=True)
tok.save_pretrained(A.out)
with io.open(os.path.join(A.out, "train_log.json"), "w", encoding="utf-8") as fh:
    json.dump({"data": A.data, "base": A.base, "seed": A.seed, "epochs": A.epochs,
               "lr": A.lr, "batch": A.batch, "accum": A.accum, "max_len": A.max_len,
               "rows": len(ds), "opt_steps": steps,
               "first_loss": first_loss, "last_loss": last_loss, "log": log}, fh, indent=2)
print("SAVED %s  first_loss=%.4f  last_loss=%.4f" % (A.out, first_loss, last_loss), flush=True)
