"""Merge the fixed five GSM8K seed pairs only when all eight new cells are frozen.

Usage: python aggregate_extension.py --new-root EVALUATION/gsm8k \
    --audit-root work/reporting_audit --output-dir NEW_OUTPUT_DIRECTORY

Standard library only. Reads saved scores; never generates, scores with MV,
changes inputs, or edits a manuscript. Existing output directories are refused.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

OLD_SEEDS = (83, 84, 85)
NEW_SEEDS = (86, 87)
SEEDS = OLD_SEEDS + NEW_SEEDS
ARMS = ("hash", "boxed")
METRICS = ("payload", "q", "strict", "mv")
N = 1319
PROTOCOL_FIELDS = ("items_sha256", "temperature", "top_p", "n", "max_tokens", "seed", "max_model_len", "batch_items")
CELL_FILES = ("responses.jsonl", "task.json", "GENERATION_COMPLETE.json",
              "per_item_scores.jsonl", "response_scores.jsonl", "summary.json", "SCORING_COMPLETE.json")
BOX_FILES = ("per_response.jsonl", "summary.json", "COMPLETE.json")
DEFINITIONS = {
    "payload": "Requested-convention syntactic payload, from the frozen reporting-audit detector; both conventions may be present.",
    "q": "Requested-convention extractability: frozen extract_hashes, or extract_last_number(extract_boxed(response)); not payload presence.",
    "strict": "Saved strict_first_hash for hash requests; saved boxed_strict for boxed requests.",
    "mv": "Saved math_verify; an operational reader, not a semantic-equivalence test.",
}
FORMULA = "100 * ((rate[hash training,hash request] - rate[boxed training,hash request]) - (rate[hash training,boxed request] - rate[boxed training,boxed request]))"


class ValidationError(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise ValidationError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_rows(path):
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    need(len(lines) == N and all(line.strip() for line in lines), f"{path}: require exactly {N} nonempty JSONL rows")
    rows = [json.loads(line) for line in lines]
    need(all(type(row.get("id")) is int for row in rows), f"{path}: IDs must be integers")
    need([row["id"] for row in rows] == list(range(N)), f"{path}: require ordered unique IDs 0..{N - 1}")
    return rows


def binary(value, context):
    need(type(value) in (bool, int, float) and value in (0, 1), f"{context}: require boolean or numeric 0/1")
    return int(value)


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def record_file(path, expected=None):
    need(path.is_file(), f"Missing file: {path}")
    digest = sha(path)
    if expected is not None:
        need(digest == expected, f"SHA256 mismatch: {path}")
    return {"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size}


def frozen_files(folder, receipt, required):
    info = {receipt: record_file(folder / receipt)}
    manifest = read_json(folder / receipt)
    for name in required:
        expected = manifest.get("files", {}).get(name)
        need(isinstance(expected, str), f"{folder / receipt}: missing frozen hash for {name}")
        info[name] = record_file(folder / name, expected)
    return info


def new_cell(root, seed, training, request):
    state = f"qwen7_rl_{training}{seed}"
    task_name = f"{request}_t0.6"
    folder = root / "results" / state / task_name
    files = frozen_files(folder, "CELL_FROZEN.json", CELL_FILES)
    meta = read_json(folder / "task.json")
    need(meta["state"] == state and meta["task"]["name"] == task_name, f"{folder}: state/task mismatch")
    generation = read_json(folder / "GENERATION_COMPLETE.json")
    need(generation["items"] == N and generation["samples_per_item"] == 1, f"{folder}: incomplete generation receipt")
    need(generation["raw_sha256"] == files["responses.jsonl"]["sha256"], f"{folder}: raw/generation receipt mismatch")
    need(generation["task_sha256"] == files["task.json"]["sha256"], f"{folder}: task/generation receipt mismatch")
    # Freeze receipts cover the saved scorer's completion receipt and summary.
    # Its schema is owned by the unchanged scorer; no inference from line count alone.
    read_json(folder / "SCORING_COMPLETE.json")
    read_rows(folder / "per_item_scores.jsonl")
    score_summary = read_json(folder / "summary.json")
    need(score_summary["items"] == N and score_summary["responses"] == N, f"{folder}: incomplete scoring summary")
    need(score_summary["raw_sha256"] == files["responses.jsonl"]["sha256"], f"{folder}: scoring summary/raw mismatch")
    boxed_path = None
    if request == "boxed":
        supplement = root / "analysis" / "boxed_supplement" / state / task_name
        box_files = frozen_files(supplement, "BOXED_FROZEN.json", BOX_FILES)
        boxed = read_json(supplement / "summary.json")
        complete = read_json(supplement / "COMPLETE.json")
        need(boxed["state"] == state and boxed["task"] == task_name and boxed["responses"] == N,
             f"{supplement}: boxed summary identity/count mismatch")
        need(boxed["raw_sha256"] == files["responses.jsonl"]["sha256"], f"{supplement}: boxed summary/raw mismatch")
        need(complete["summary_sha256"] == box_files["summary.json"]["sha256"], f"{supplement}: boxed completion mismatch")
        files.update({"boxed/" + name: value for name, value in box_files.items()})
        boxed_path = supplement / "per_response.jsonl"
    return folder, boxed_path, meta["task"], files


def measure(folder, boxed_path, request, detector, readers):
    raw = read_rows(folder / "responses.jsonl")
    scores = read_rows(folder / "response_scores.jsonl")
    boxed = read_rows(boxed_path) if boxed_path else None
    counts = {metric: 0 for metric in METRICS}
    for i, (row, score) in enumerate(zip(raw, scores)):
        need(type(row.get("n")) is int and row["n"] == 1 and len(row["responses"]) == 1,
             f"{folder}: item {i} must have one response and n=1")
        need(len(row["finish_reasons"]) == 1 and isinstance(row["responses"][0], str), f"{folder}: invalid raw item {i}")
        need(type(row.get("input_tokens")) is int and row["input_tokens"] > 0, f"{folder}: invalid input token count at item {i}")
        need(type(score.get("sample")) is int and score["sample"] == 0, f"{folder}: score sample must be 0 at item {i}")
        response = row["responses"][0]
        detection = detector.detect(response)
        counts["payload"] += binary(detection["hash_numeric_payload" if request == "hash" else "boxed_payload"], "payload")
        if request == "hash":
            extracted = readers.extract_hashes(response)
            strict = score["strict_first_hash"]
        else:
            inner = readers.extract_boxed(response)
            extracted = readers.extract_last_number(inner) if inner is not None else None
            need(type(boxed[i].get("sample")) is int and boxed[i]["sample"] == 0, f"{boxed_path}: sample must be 0 at item {i}")
            strict = boxed[i]["boxed_strict"]
        counts["q"] += int(extracted is not None)
        counts["strict"] += binary(strict, f"{folder}: strict item {i}")
        counts["mv"] += binary(score["math_verify"], f"{folder}: MV item {i}")
    return counts, [row["input_tokens"] for row in raw]


def build_result(new_root, audit_root):
    # Preflight every new cell before reading or merging the old data.
    new = {}
    problems = []
    for seed in NEW_SEEDS:
        for training in ARMS:
            for request in ARMS:
                try:
                    new[seed, training, request] = new_cell(new_root, seed, training, request)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    problems.append(f"seed={seed}, training={training}, request={request}: {exc}")
    need(not problems, "All eight new cells must be complete; no merged output written.\n" + "\n".join(problems))

    old_manifest_path = audit_root / "input_manifest.json"
    archived_path = audit_root / "archived_crossed_results.json"
    audit_summary_path = audit_root / "results" / "summary.json"
    old_manifest = read_json(old_manifest_path)
    archived = read_json(archived_path)
    audit_summary = read_json(audit_summary_path)
    old_entries = {(c["seed"], c["training"], c["request"]): c for c in old_manifest["cells"]}
    expected_old = {(s, t, r) for s in OLD_SEEDS for t in ARMS for r in ARMS}
    need(len(old_manifest["cells"]) == 12 and set(old_entries) == expected_old, "Old audit manifest must contain exactly the 12 fixed old cells")
    detector_path = audit_root / "payload_detector.py"
    reader_path = audit_root / "frozen_code" / "passk_score.py"
    code_files = {
        "aggregate_extension.py": record_file(Path(__file__)),
        "payload_detector.py": record_file(detector_path, audit_summary["code_sha256"]["payload_detector.py"]),
        "frozen_code/passk_score.py": record_file(reader_path, audit_summary["code_sha256"]["frozen_code/passk_score.py"]),
    }
    detector = module_at("extension_payload_detector", detector_path)
    readers = module_at("extension_frozen_readers", reader_path)
    cells, inputs, protocols, input_tokens = [], [], {}, {}
    for seed in SEEDS:
        for training in ARMS:
            for request in ARMS:
                key = (seed, training, request)
                if seed in OLD_SEEDS:
                    folder = audit_root / "inputs" / str(seed) / training / request
                    entry = old_entries[key]
                    files = {name: record_file(folder / name, entry["files"][name]["sha256"])
                             for name in ("responses.jsonl", "response_scores.jsonl", "task.json")}
                    boxed_path = folder / "boxed_scores.jsonl" if request == "boxed" else None
                    if boxed_path:
                        files["boxed_scores.jsonl"] = record_file(boxed_path)
                    task = read_json(folder / "task.json")["task"]
                else:
                    folder, boxed_path, task, files = new[key]
                    if request == "boxed":
                        box_summary = read_json(boxed_path.parent / "summary.json")
                        need(box_summary["scorer_sha256"] == code_files["frozen_code/passk_score.py"]["sha256"],
                             f"{boxed_path}: frozen boxed scorer drift")
                protocol = {field: task[field] for field in PROTOCOL_FIELDS}
                if request not in protocols:
                    protocols[request] = protocol
                need(protocol == protocols[request], f"seed {seed}/{training}/{request}: evaluation protocol differs from old frozen baseline")
                counts, tokens = measure(folder, boxed_path, request, detector, readers)
                if request not in input_tokens:
                    input_tokens[request] = tokens
                need(tokens == input_tokens[request], f"seed {seed}/{training}/{request}: per-item input token counts differ from old frozen baseline")
                if seed in OLD_SEEDS:
                    baseline = archived["seeds"][str(seed)]["cells"][training][f"{request}_t0.6"]
                    need(all(counts[m] == baseline[m] for m in ("strict", "mv")), f"seed {seed}/{training}/{request}: corrected archived score baseline mismatch")
                cells.append({"seed": seed, "cohort": "original" if seed in OLD_SEEDS else "extension",
                              "training": training, "request": request, "n": N, "counts": counts,
                              "rates_pct": {m: 100 * counts[m] / N for m in METRICS}})
                inputs.append({"seed": seed, "training": training, "request": request, "files": files})
    lookup = {(c["seed"], c["training"], c["request"]): c["counts"] for c in cells}
    per_seed = []
    for seed in SEEDS:
        hh, bh, hb, bb = (lookup[seed, t, r] for t, r in (("hash", "hash"), ("boxed", "hash"), ("hash", "boxed"), ("boxed", "boxed")))
        interactions = {m: 100 * ((hh[m] - bh[m]) - (hb[m] - bb[m])) / N for m in METRICS}
        per_seed.append({"seed": seed, "cohort": "original" if seed in OLD_SEEDS else "extension", "interaction_pp": interactions})
    aggregate = {}
    for metric in METRICS:
        values = [row["interaction_pp"][metric] for row in per_seed]
        aggregate[metric] = {"mean_pp": statistics.mean(values), "sample_sd_pp": statistics.stdev(values),
                             "minimum_pp": min(values), "maximum_pp": max(values), "range_pp": [min(values), max(values)],
                             "span_pp": max(values) - min(values), "seeds": len(SEEDS)}
    scope = "Descriptive, equally weighted seed summaries. No training-population inference; MV is not a semantic-equivalence test. Initial-model responses are excluded. No manuscript is updated."
    summary = {"seeds": list(SEEDS), "definitions": DEFINITIONS, "interaction_formula_pp": FORMULA,
               "interpretation": scope, "cells": cells, "per_seed": per_seed, "aggregate": aggregate}
    manifest = {"schema_version": 1, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "fixed_seeds": list(SEEDS), "original_seeds": list(OLD_SEEDS), "extension_seeds": list(NEW_SEEDS),
                "items_per_cell": N, "samples_per_item": 1, "cell_counts": {"original": 12, "extension": 8, "combined": 20},
                "response_counts": {"original": 12 * N, "extension": 8 * N, "combined": 20 * N},
                "all_eight_extension_cells_validated": True, "new_root": str(new_root), "audit_root": str(audit_root),
                "protocols_by_request": protocols, "interaction_formula_pp": FORMULA,
                "metric_order": list(METRICS), "definitions": DEFINITIONS, "interpretation": scope,
                "baseline_files": [record_file(p) for p in (old_manifest_path, archived_path, audit_summary_path)],
                "code_files": code_files, "cells": inputs}
    return summary, manifest


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_result(out, summary, manifest):
    # All validation precedes this write. Never replace a previous completed result.
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "summary.json", summary)
    dump(out / "manifest.json", manifest)
    dump(out / "fixed_seeds.json", {"seeds": list(SEEDS), "original": list(OLD_SEEDS), "extension": list(NEW_SEEDS)})
    write_csv(out / "cells.csv", [{**{k: c[k] for k in ("seed", "cohort", "training", "request", "n")},
                                    **{m + "_count": c["counts"][m] for m in METRICS},
                                    **{m + "_pct": c["rates_pct"][m] for m in METRICS}} for c in summary["cells"]])
    write_csv(out / "interactions.csv", [{"seed": r["seed"], "cohort": r["cohort"], **{m + "_pp": r["interaction_pp"][m] for m in METRICS}} for r in summary["per_seed"]])
    write_csv(out / "aggregate.csv", [{"metric": m, **{k: v for k, v in summary["aggregate"][m].items() if k != "range_pp"}} for m in METRICS])
    report = ["# Fixed five-seed extension", "", summary["interpretation"], "",
              "All 20 cells contain exactly 1,319 responses (26,380 total; 10,552 new).", "",
              "Interaction = (hash-trained/hash-request − boxed-trained/hash-request) − (hash-trained/boxed-request − boxed-trained/boxed-request), in percentage points.", "",
              "| Seed | Cohort | Payload | q | Strict | MV |", "|---|---|---:|---:|---:|---:|"]
    for row in summary["per_seed"]:
        report.append(f"| {row['seed']} | {row['cohort']} | " + " | ".join(f"{row['interaction_pp'][m]:+.4f}" for m in METRICS) + " |")
    report += ["", "| Metric | Mean (pp) | Sample SD (pp) | Range (pp) |", "|---|---:|---:|---|"]
    for m in METRICS:
        a = summary["aggregate"][m]
        report.append(f"| {m} | {a['mean_pp']:+.4f} | {a['sample_sd_pp']:.4f} | [{a['minimum_pp']:+.4f}, {a['maximum_pp']:+.4f}] |")
    report += ["", "Sample SD uses ddof=1 across the five seed interactions.", ""]
    report += [f"- **{m}**: {DEFINITIONS[m]}" for m in METRICS]
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    files = sorted(p for p in out.iterdir() if p.is_file())
    dump(out / "AGGREGATION_COMPLETE.json", {"fixed_seeds": list(SEEDS), "cells": 20, "responses": 20 * N,
                                              "files": {p.name: sha(p) for p in files}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-root", type=Path, required=True, help="GSM8K root containing results/ and analysis/boxed_supplement/.")
    parser.add_argument("--audit-root", type=Path, required=True, help="Old normalized reporting_audit directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory, created only after every input validates.")
    args = parser.parse_args()
    try:
        need(not args.output_dir.exists(), f"Refusing existing output directory: {args.output_dir}")
        summary, manifest = build_result(args.new_root.resolve(), args.audit_root.resolve())
        write_result(args.output_dir.resolve(), summary, manifest)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Aggregation rejected: {exc}", file=sys.stderr)
        return 1
    print(f"Validated all 20 cells, fixed seeds {list(SEEDS)}, {20 * N} responses; saved {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
