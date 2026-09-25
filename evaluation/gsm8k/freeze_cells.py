"""Freeze scored GSM8K cells and add the frozen boxed-request reader."""

import argparse
import datetime
import importlib.util
import json
from pathlib import Path
import sys

from extension_common import dump, read_jsonl, sha, write_jsonl


CELL_FILES = (
    "responses.jsonl", "task.json", "GENERATION_COMPLETE.json",
    "per_item_scores.jsonl", "response_scores.jsonl", "summary.json",
    "SCORING_COMPLETE.json",
)
BOX_FILES = ("per_response.jsonl", "summary.json", "COMPLETE.json")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def freeze(folder, names, receipt):
    path = folder / receipt
    require(not path.exists(), f"Refusing to overwrite {path}")
    dump(path, {"files": {name: sha(folder / name) for name in names}})


def load_scorer(path):
    spec = importlib.util.spec_from_file_location("frozen_passk_score", path)
    require(spec is not None and spec.loader is not None, f"Cannot load scorer: {path}")
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    require(scorer.gsm8k_boxed_strict(r"$\boxed{12}$", "12"), "Boxed positive fixture failed")
    require(not scorer.gsm8k_boxed_strict(r"$\boxed{13}$", "12"), "Boxed negative fixture failed")
    return scorer


def prepare_cell(root, state, folder, scorer_path, scorer):
    task_record = read(folder / "task.json")
    task = task_record["task"]
    require(task_record["state"] == state, f"State mismatch: {folder}")
    require(task["name"] == folder.name, f"Task mismatch: {folder}")
    require(task["n"] == 1, f"Expected one response per item: {folder}")
    require(sha(Path(task["items"])) == task["items_sha256"], f"Input hash mismatch: {folder}")
    items = read_jsonl(task["items"])
    raw = read_jsonl(folder / "responses.jsonl")
    scored = read_jsonl(folder / "response_scores.jsonl")
    per_item = read_jsonl(folder / "per_item_scores.jsonl")
    n = len(items)
    require(n == 1319 and len(raw) == len(scored) == len(per_item) == n,
            f"Expected 1,319 scored items: {folder}")
    ids = list(range(n))
    require([x["id"] for x in items] == ids and [x["id"] for x in raw] == ids
            and [x["id"] for x in scored] == ids and [x["id"] for x in per_item] == ids,
            f"Item IDs differ: {folder}")
    require(all(x["n"] == 1 and len(x["responses"]) == 1 for x in raw),
            f"Response count mismatch: {folder}")
    generation = read(folder / "GENERATION_COMPLETE.json")
    require(generation["raw_sha256"] == sha(folder / "responses.jsonl") and
            generation["task_sha256"] == sha(folder / "task.json"),
            f"Generation receipt mismatch: {folder}")
    summary = read(folder / "summary.json")
    complete = read(folder / "SCORING_COMPLETE.json")
    require(summary["items"] == summary["responses"] == n and
            summary["raw_sha256"] == sha(folder / "responses.jsonl") and
            complete["summary_sha256"] == sha(folder / "summary.json"),
            f"Scoring receipt mismatch: {folder}")
    require(not (folder / "CELL_FROZEN.json").exists(),
            f"Refusing to overwrite cell receipt: {folder}")
    if folder.name != "boxed_t0.6":
        return None
    out = root / "analysis" / "boxed_supplement" / state / folder.name
    require(not out.exists(), f"Refusing to overwrite boxed supplement: {out}")
    records = []
    for row in raw:
        for j, response in enumerate(row["responses"]):
            records.append({"id": row["id"], "sample": j,
                            "boxed_strict": bool(scorer.gsm8k_boxed_strict(
                                response, items[row["id"]]["ground_truth"]))})
    return out, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--scorer", type=Path, default=Path(__file__).resolve().parents[2] / "scripts" / "passk_score.py",
                        help="Frozen passk_score.py; use the audit copy when aggregating against archived cells")
    args = parser.parse_args()
    root, scorer_path = args.root.resolve(), args.scorer.resolve()
    scorer = load_scorer(scorer_path)
    folders = sorted((root / "results" / args.state).glob("*/SCORING_COMPLETE.json"))
    require(folders, f"No scored cells for state {args.state}")
    prepared = [(path.parent, prepare_cell(root, args.state, path.parent, scorer_path, scorer))
                for path in folders]
    for folder, boxed in prepared:
        if boxed is not None:
            out, records = boxed
            out.mkdir(parents=True, exist_ok=False)
            write_jsonl(out / "per_response.jsonl", records)
            dump(out / "summary.json", {
                "state": args.state, "task": folder.name, "responses": len(records),
                "accuracy": sum(x["boxed_strict"] for x in records) / len(records),
                "scorer_sha256": sha(scorer_path),
                "raw_sha256": sha(folder / "responses.jsonl"),
                "definition": "Frozen gsm8k_boxed_strict; supplementary convention-specific endpoint.",
            })
            dump(out / "COMPLETE.json", {
                "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "summary_sha256": sha(out / "summary.json"),
            })
            freeze(out, BOX_FILES, "BOXED_FROZEN.json")
        freeze(folder, CELL_FILES, "CELL_FROZEN.json")
        print(f"FROZEN {args.state}/{folder.name}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Freeze rejected: {exc}", file=sys.stderr)
        raise SystemExit(1)
