#!/usr/bin/env python
"""Score saved generations with frozen readers and compute pass@k.

The strict GSM8K evaluation reader takes the first numeric hash answer over
the full response with numeric tolerance. It differs from the training reward,
which inspects the final 300 characters, takes the last match, and compares
payload strings exactly. The lenient reader uses the strict answer when present
and otherwise falls back to the last number. Neither is a human judgment of
the response's committed answer.
"""

import argparse
import json
import math
import os
import re

BOXED = re.compile(r"\\boxed\{")
HASHES = re.compile(r"####\s*(-?\d+\.?\d*)")
NUMBER = re.compile(r"-?\d+\.?\d*")
# Non-numeric payload after ####, for MATH content under the #### convention.
# HASHES itself is left alone: gsm8k_strict is a frozen metric.
HASHES_ANY = re.compile(r"####\s*(.+)")


def extract_hashes(text):
    match = HASHES.search(text.replace(",", ""))
    return match.group(1) if match else None


def extract_last_number(text):
    numbers = NUMBER.findall(text.replace(",", ""))
    return numbers[-1] if numbers else None


def numeric_equal(a, b):
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-3
    except (TypeError, ValueError):
        return str(a).strip() == str(b).strip()


def gsm8k_reference(ground_truth):
    return extract_hashes(ground_truth) or extract_last_number(ground_truth) or ground_truth


def gsm8k_strict(response, ground_truth):
    """Evaluation reader: credit the first numeric hash answer, if correct."""
    return numeric_equal(extract_hashes(response), gsm8k_reference(ground_truth))


def gsm8k_lenient(response, ground_truth):
    predicted = extract_hashes(response)
    if predicted is None:
        predicted = extract_last_number(response)
    return numeric_equal(predicted, gsm8k_reference(ground_truth))


_MATH_VERIFY_FNS = None


def _math_verify_import():
    """Import math_verify once, and make its ABSENCE an error rather than a score.

    Without this, an interpreter lacking math_verify scores every MATH item by
    last-number regex instead of symbolic verification and reports no problem.
    Measured on the frozen crossover arms: 70/500 and 83/500 items flip, p@1
    moves by up to 0.022. A metric that changes with the interpreter and says
    nothing is worse than one that refuses to run.
    """
    global _MATH_VERIFY_FNS
    if _MATH_VERIFY_FNS is None:
        try:
            from math_verify import parse, verify
        except Exception as exc:
            raise RuntimeError(
                "math_verify is not importable in this interpreter (%s). MATH "
                "scoring without it silently substitutes a last-number regex "
                "for symbolic verification and returns different numbers with "
                "no error. Score MATH tasks with "
                "a Python environment with math-verify installed." % exc)
        _MATH_VERIFY_FNS = (parse, verify)
    return _MATH_VERIFY_FNS


def _math_verify(candidate, ground_truth):
    parse, verify = _math_verify_import()
    try:
        return bool(verify(parse(f"${ground_truth}$"), parse(candidate)))
    except Exception:
        # Genuine per-item failure (unparseable expression). Falling back here
        # is correct and is what the frozen numbers were produced with.
        return None


def math_strict(response, ground_truth):
    """Requires a \\boxed{...}; that is what the MATH reward function reads."""
    if not BOXED.search(response):
        return False
    verdict = _math_verify(response, ground_truth)
    if verdict is not None:
        return verdict
    return numeric_equal(extract_last_number(response), extract_last_number(ground_truth))


def math_lenient(response, ground_truth):
    if BOXED.search(response):
        return math_strict(response, ground_truth)
    tail = response.strip().splitlines()[-1] if response.strip() else ""
    verdict = _math_verify(f"${tail}$", ground_truth)
    if verdict is not None:
        return verdict
    return numeric_equal(extract_last_number(response), extract_last_number(ground_truth))


def extract_boxed(text):
    """Contents of the first \\boxed{...}, brace-matched.

    A regex cannot do this: MATH answers contain nested braces
    (\\boxed{\\frac{1}{2}}), and a non-greedy match truncates them to
    "\\frac{1" while a greedy one swallows the rest of the response.
    """
    match = BOXED.search(text)
    if not match:
        return None
    i = match.end()          # just past the opening brace
    depth = 1
    out = []
    while i < len(text) and depth:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if not depth:
                break
        out.append(ch)
        i += 1
    return "".join(out) if depth == 0 else None


def gsm8k_boxed_strict(response, ground_truth):
    """GSM8K content under the \\boxed{} convention.

    Same content rule as gsm8k_strict (numeric equality against the GSM8K
    reference); only the marker the answer must be delimited with differs.
    """
    inner = extract_boxed(response)
    if inner is None:
        return False
    return numeric_equal(extract_last_number(inner), gsm8k_reference(ground_truth))


def math_hashes_strict(response, ground_truth):
    """MATH content under the #### convention.

    MATH answers are frequently non-numeric, so the text after the marker is
    handed to math_verify exactly as math_strict hands it the boxed contents,
    with the same numeric fallback when math_verify is unavailable.
    """
    match = HASHES_ANY.search(response)
    if not match:
        return False
    tail = match.group(1).strip()
    verdict = _math_verify(f"${tail}$", ground_truth)
    if verdict is not None:
        return verdict
    return numeric_equal(extract_last_number(tail), extract_last_number(ground_truth))


# A synthetic, out-of-distribution answer marker. Deliberately NOT a markdown
# construct: `####` is an H4 and that collision has already cost this project
# one inflated compliance number.
NOVEL_MARK = "@@@"
NOVEL = re.compile(r"@@@\s*(-?\d+\.?\d*)")


def extract_novel(text):
    """Mirror of extract_hashes with the marker swapped, comma stripping and all."""
    match = NOVEL.search(text.replace(",", ""))
    return match.group(1) if match else None


def gsm8k_novel_strict(response, ground_truth):
    """GSM8K content under a synthetic convention.

    Character-for-character gsm8k_strict with `####` replaced by `@@@`. Held
    identical on purpose: the panel compares gap(novel) against gap(hash), so
    any asymmetry between the two extraction rules would show up as a result.
    """
    return numeric_equal(extract_novel(response), gsm8k_reference(ground_truth))


SCORERS = {
    "gsm8k": {"strict": gsm8k_strict, "lenient": gsm8k_lenient},
    "math": {"strict": math_strict, "lenient": math_lenient},
    # Convention-crossed corners. The lenient rule is the task's existing one in
    # both cases, so lenient is invariant to the convention by construction.
    "gsm8k_boxed": {"strict": gsm8k_boxed_strict, "lenient": gsm8k_lenient},
    "math_hashes": {"strict": math_hashes_strict, "lenient": math_lenient},
    # Out-of-distribution convention for the scale x convention panel.
    "gsm8k_novel": {"strict": gsm8k_novel_strict, "lenient": gsm8k_lenient},
}


def pass_at_k(n, c, k):
    """Unbiased 1 - C(n-c,k)/C(n,k), evaluated with log-gamma."""
    if k > n:
        return float("nan")
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    # log C(n-c,k) - log C(n,k) == [lg(n-c+1)-lg(n-c-k+1)] - [lg(n+1)-lg(n-k+1)]
    lg = math.lgamma
    log_ratio = lg(n - c + 1) - lg(n - c - k + 1) - lg(n + 1) + lg(n - k + 1)
    return 1.0 - math.exp(log_ratio)


def load_items(path):
    items = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                items[row["id"]] = row
    return items


def iter_samples(paths):
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def curve_for(records, key, ks):
    curve = {}
    min_n = min(r["n"] for r in records)
    for k in ks:
        if k > min_n:
            continue
        values = [pass_at_k(r["n"], r[key], k) for r in records]
        curve[str(k)] = sum(values) / len(values)
    return curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True)
    parser.add_argument("--samples", nargs="+", required=True, help="shard JSONL files")
    parser.add_argument("--task", default="gsm8k", choices=sorted(SCORERS))
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument(
        "--ks",
        default="1,2,4,8,16,32,64,128,256",
        help="comma-separated k values for the curve",
    )
    args = parser.parse_args()

    items = load_items(args.items)
    strict = SCORERS[args.task]["strict"]
    lenient = SCORERS[args.task]["lenient"]
    ks = [int(x) for x in args.ks.split(",") if x.strip()]

    per_item = {}
    truncated = 0
    total_samples = 0
    for row in iter_samples(args.samples):
        item = items.get(row["id"])
        if item is None:
            continue
        responses = row["responses"]
        truth = item["ground_truth"]
        c_lenient = c_strict = 0
        for response in responses:
            if lenient(response, truth):
                c_lenient += 1
            if strict(response, truth):
                c_strict += 1
        record = {
            "id": row["id"],
            "n": len(responses),
            "c": c_lenient,
            "c_strict": c_strict,
        }
        prior = per_item.get(row["id"])
        # A duplicate id (resumed shard) keeps the larger sample count.
        if prior is None or record["n"] > prior["n"]:
            per_item[row["id"]] = record
        truncated += sum(1 for r in row.get("finish_reasons", []) if r == "length")
        total_samples += len(responses)

    if not per_item:
        raise SystemExit("no scored items — check --samples paths")

    records = sorted(per_item.values(), key=lambda r: r["id"])
    min_n = min(r["n"] for r in records)

    lenient_curve = curve_for(records, "c", ks)
    strict_curve = curve_for(records, "c_strict", ks)
    n_items = len(records)

    summary = {
        "label": args.label,
        "task": args.task,
        "n_items": n_items,
        "min_n_samples": min_n,
        "max_n_samples": max(r["n"] for r in records),
        "pass_at_k": lenient_curve,
        "pass_at_k_strict": strict_curve,
        "pass_at_1": lenient_curve.get("1"),
        "pass_at_1_strict": strict_curve.get("1"),
        "mean_solve_rate": sum(r["c"] / r["n"] for r in records) / n_items,
        "mean_solve_rate_strict": sum(r["c_strict"] / r["n"] for r in records) / n_items,
        "items_with_zero_correct": sum(1 for r in records if r["c"] == 0),
        "items_with_zero_correct_strict": sum(1 for r in records if r["c_strict"] == 0),
        # How much of the lenient score the strict rule discards purely because
        # the model did not emit the trained answer marker.
        "format_compliance_gap_at_1": (
            (lenient_curve.get("1") or 0.0) - (strict_curve.get("1") or 0.0)
        ),
        "truncation_rate": truncated / total_samples if total_samples else 0.0,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    per_item_path = f"{args.out}.per_item.jsonl"
    with open(per_item_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"per-item sufficient statistics -> {per_item_path}")


if __name__ == "__main__":
    main()
