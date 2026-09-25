"""Render frozen GSM8K evaluation items and fresh arithmetic prompts."""

import argparse
import json
from pathlib import Path

from extension_common import arithmetic_items, sha


SUFFIX = ' Let\'s think step by step and output the final answer after "####".'


def write_new(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{path}: {len(rows)} rows, sha256={sha(path)}")


def tokenizer_from(name):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name, trust_remote_code=True)


def render(tok, question):
    return tok.apply_chat_template(
        [{"role": "user", "content": question + SUFFIX}],
        add_generation_prompt=True, tokenize=False,
    )


def matched(args):
    import pandas as pd

    if args.source_sha256 and sha(args.validation_parquet) != args.source_sha256:
        raise ValueError("Validation Parquet SHA256 mismatch")
    frame = pd.read_parquet(args.validation_parquet)
    if len(frame) != 1319:
        raise ValueError(f"Expected 1,319 GSM8K items, found {len(frame)}")
    tok = tokenizer_from(args.tokenizer)
    rows = []
    for i, row in enumerate(frame.to_dict("records")):
        info = row["extra_info"]
        if info["index"] != i or info["split"] != "test":
            raise ValueError(f"Unexpected GSM8K item ID/split at {i}")
        messages = list(row["prompt"])
        if len(messages) != 1 or messages[0]["role"] != "user":
            raise ValueError(f"Unexpected prompt shape at {i}")
        content = messages[0]["content"]
        if args.convention == "hash":
            expected = SUFFIX
        else:
            expected = " Let's think step by step and output the final answer within \\boxed{}."
        if not content.endswith(expected) or content[:-len(expected)] != info["question"]:
            raise ValueError(f"Prompt/question mismatch at {i}")
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        rows.append({"id": i, "prompt": prompt,
                     "ground_truth": str(row["reward_model"]["ground_truth"]),
                     "meta": {"difficulty_bin": row["difficulty_bin"],
                              "frozen_baseline_solve_rate": row["frozen_baseline_solve_rate"]}})
    write_new(args.out, rows)


def arithmetic(args):
    dev, test = arithmetic_items()
    target = Path(args.out_dir)
    if any((target / name).exists() for name in
           ("arithmetic_dev_raw.jsonl", "arithmetic_test_raw.jsonl",
            "arithmetic_dev.jsonl", "arithmetic_test.jsonl")):
        raise FileExistsError("Refusing to replace existing arithmetic items")
    tok = None if args.raw_only else tokenizer_from(args.tokenizer)
    for split, items in (("dev", dev), ("test", test)):
        write_new(target / f"arithmetic_{split}_raw.jsonl", items)
        if tok is not None:
            rendered = [dict(row, prompt=render(tok, row["question"])) for row in items]
            write_new(target / f"arithmetic_{split}.jsonl", rendered)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gsm = sub.add_parser("gsm8k")
    gsm.add_argument("--validation-parquet", type=Path, required=True)
    gsm.add_argument("--source-sha256", help="Optional frozen input hash")
    gsm.add_argument("--tokenizer", required=True)
    gsm.add_argument("--convention", choices=("hash", "boxed"), required=True)
    gsm.add_argument("--out", type=Path, required=True)
    gsm.set_defaults(func=matched)
    arith = sub.add_parser("arithmetic")
    arith.add_argument("--tokenizer", help="Initial model tokenizer; required unless --raw-only")
    arith.add_argument("--out-dir", required=True)
    arith.add_argument("--raw-only", action="store_true")
    arith.set_defaults(func=arithmetic)
    args = parser.parse_args()
    if args.command == "arithmetic" and not args.raw_only and not args.tokenizer:
        parser.error("arithmetic requires --tokenizer unless --raw-only")
    args.func(args)


if __name__ == "__main__":
    main()
