"""Frozen-cache human semantic calibration; stdlib only, no automatic math judge.

Run only after the human workflow finishes. With blank labels this emits pending
templates. --self-test runs explicitly synthetic tests, with disk fixtures in OS
temporary directories; it never writes fabricated results into this package.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys
import unittest

STATUSES = ("CLEAR_ANSWER", "NO_ANSWER", "UNRESOLVED_CONFLICT", "AMBIGUOUS")
CORRECTNESS = ("CORRECT", "INCORRECT", "UNRESOLVED")
FIDELITIES = ("FAITHFUL", "UNFAITHFUL", "UNRESOLVED", "NOT_APPLICABLE")
AGREEMENTS = ("FULL_AGREEMENT", "ANSWER_DISAGREEMENT", "STATUS_DISAGREEMENT",
              "CORRECTNESS_DISAGREEMENT", "MULTIPLE_DISAGREEMENTS")
STAGE1_FIELDS = ("status", "committed_answer", "evidence_span", "has_self_correction",
                 "has_conflicting_answer_fields", "old_answer_explicitly_withdrawn", "notes")
STAGE2_FIELDS = ("answer_correct", "reference_issue", "stage2_notes", "fidelity_R1", "fidelity_R2", "fidelity_R3")
QUOTAS = {"qwen15_base": 105, "qwen15_rl": 61, "smol_base": 106, "smol_rl": 82,
          "initial": 69, "seed83": 62, "seed84": 70, "seed85": 70}
METRIC_COLUMNS = ["state", "dataset", "reader", "metric", "estimate", "lower_estimate",
                  "upper_estimate", "ci_low", "ci_high", "denominator", "note"]
GAIN_COLUMNS = ["comparison", "dataset", "reader", "primary", "reader_gain", "human_gain",
                "human_lower", "human_upper", "human_ci_low", "human_ci_high", "gap",
                "gap_lower", "gap_upper", "gap_ci_low", "gap_ci_high", "bias_drift",
                "fn_recovery_lower", "fn_recovery_upper", "fp_increase_lower", "fp_increase_upper"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reader_names(dataset):
    require(dataset in ("GSM8K", "MATH500"), f"Unknown dataset {dataset}")
    return ("strict", "fallback", "mv") if dataset == "GSM8K" else ("strict", "mv")


@lru_cache(maxsize=20000)
def hypergeom_interval(x, n, N, alpha=.05):
    """Equal-tail exact inversion for finite population success fraction K/N.

    Each retained K satisfies P_K(X>=x)>=alpha/2 and P_K(X<=x)>=alpha/2.
    Sampling is SRS without replacement within one fixed stratum.
    """
    require(all(isinstance(a, int) and not isinstance(a, bool) for a in (x, n, N)), "Integer counts required")
    require(0 <= x <= n <= N and n > 0 and 0 < alpha < 1, "Invalid hypergeometric arguments")
    if n == N:
        return x / N, x / N
    denominator = math.comb(N, n)

    def tail(K, upper):
        lo, hi = max(0, n - (N - K)), min(n, K)
        lo, hi = (max(lo, x), hi) if upper else (lo, min(hi, x))
        return sum(math.comb(K, j) * math.comb(N - K, n - j) for j in range(lo, hi + 1)) / denominator

    left, right = x, N - n + x
    while left < right:
        middle = (left + right) // 2
        if tail(middle, True) >= alpha / 2:
            right = middle
        else:
            left = middle + 1
    lower = left
    left, right = x, N - n + x
    while left < right:
        middle = (left + right + 1) // 2
        if tail(middle, False) >= alpha / 2:
            left = middle
        else:
            right = middle - 1
    return lower / N, left / N


def interval(lower, upper, ci_low, ci_high, denominator="all responses", note=""):
    require(lower <= upper + 1e-10 and ci_low <= ci_high + 1e-10, "Reversed interval")
    estimate = (lower + upper) / 2 if abs(lower - upper) < 1e-12 else None
    return {"estimate": estimate, "lower_estimate": lower, "upper_estimate": upper,
            "ci_low": ci_low, "ci_high": ci_high, "denominator": denominator, "note": note}


def exact(value, **kwargs):
    return interval(value, value, value, value, **kwargs)


def linear(metrics, coefficients):
    values = []
    for lowkey, highkey in (("lower_estimate", "upper_estimate"), ("ci_low", "ci_high")):
        low = sum(c * m[lowkey if c >= 0 else highkey] for m, c in zip(metrics, coefficients))
        high = sum(c * m[highkey if c >= 0 else lowkey] for m, c in zip(metrics, coefficients))
        values.extend((low, high))
    return interval(*values)


def ratio(numerator, denominator, label):
    if denominator["upper_estimate"] <= 0:
        result = {k: None for k in ("estimate", "lower_estimate", "upper_estimate", "ci_low", "ci_high")}
        if denominator["ci_high"] > 0:
            result.update(ci_low=0.0, ci_high=1.0)
        return {**result, "denominator": label, "note": "Undefined: zero estimated eligible denominator."}
    def bounds(nlo, nhi, dlo, dhi):
        return (max(0.0, min(1.0, nlo / dhi)) if dhi > 0 else 0.0,
                max(0.0, min(1.0, nhi / dlo)) if dlo > 0 else 1.0)
    return interval(*bounds(numerator["lower_estimate"], numerator["upper_estimate"],
                            denominator["lower_estimate"], denominator["upper_estimate"]),
                    *bounds(numerator["ci_low"], numerator["ci_high"], denominator["ci_low"], denominator["ci_high"]),
                    denominator=label, note="Conservative ratio bounds; may include a zero population denominator.")


def human_value(label):
    status, correctness = label["final_status"], label["final_answer_correct"]
    if status == "NO_ANSWER":
        return 0
    if status != "CLEAR_ANSWER" or correctness == "UNRESOLVED":
        return None
    return int(correctness == "CORRECT")


def validate_inputs(samples, populations, labels, production=False):
    require(isinstance(samples, list) and isinstance(populations, list) and isinstance(labels, list), "Inputs must be JSON lists")
    require(samples and populations, "Empty frozen sample or population")
    ids = [r["sample_id"] for r in samples]
    require(len(ids) == len(set(ids)), "Duplicate frozen sample_id")
    label_ids = [r.get("sample_id") for r in labels]
    require(len(label_ids) == len(set(label_ids)), "Duplicate final label sample_id")
    require(set(label_ids) == set(ids), "Missing or unknown final label sample_id")
    pop = {r["state"]: r for r in populations}
    require(len(pop) == len(populations), "Duplicate population state")
    require(set(pop) == {r["state"] for r in samples}, "Population/sample state mismatch")
    if production:
        require(Counter(r["state"] for r in samples) == QUOTAS, "Production package must match frozen 625-response quotas")
        for p in populations:
            require(p["N"] == (1319 if p["dataset"] == "GSM8K" else 500), "Unexpected production population size")
    observed = Counter((r["state"], r["stratum"]) for r in samples)
    original_keys = [(r["state"], str(r["original_id"])) for r in samples]
    require(len(set(original_keys)) == len(original_keys), "Duplicate state/original response id")
    for state, p in pop.items():
        readers = reader_names(p["dataset"])
        require(isinstance(p["N"], int) and p["N"] > 0, f"Invalid population size: {state}")
        require(sum(v["N"] for v in p["strata"].values()) == p["N"], f"Population strata do not sum: {state}")
        for bits, counts in p["strata"].items():
            require(len(bits) == len(readers) and set(bits) <= {"0", "1"}, f"Invalid reader bit stratum: {state}/{bits}")
            N, n = counts["N"], counts["n"]
            require(all(isinstance(v, int) and not isinstance(v, bool) for v in (N, n)), "Noninteger stratum count")
            require(0 <= n <= N and (N == 0 or n > 0), f"Nonempty stratum is unsampled: {state}/{bits}")
            require(observed[state, bits] == n, f"Stratum quota mismatch: {state}/{bits}")
        for j, reader in enumerate(readers):
            implied = sum(v["N"] * int(bits[j]) for bits, v in p["strata"].items()) / p["N"]
            acc = p["reader_accuracy"].get(reader)
            require(isinstance(acc, (int, float)) and 0 <= acc <= 1 and abs(acc - implied) < 1e-10,
                    f"Frozen reader accuracy differs from strata: {state}/{reader}")
    for r in samples:
        p = pop[r["state"]]
        require(r["dataset"] == p["dataset"], "Sample dataset/state mismatch")
        require(r["stratum"] in p["strata"], "Unknown stratum")
        stratum = p["strata"][r["stratum"]]
        N, n = stratum["N"], stratum["n"]
        require((r["population_n"], r["stratum_population"], r["stratum_sample_n"]) == (p["N"], N, n), "Frozen sample count metadata mismatch")
        require(abs(r["inclusion_probability"] - n / N) < 1e-12 and abs(r["weight"] - N / n) < 1e-9, "Invalid sampling probability or weight")
        for j, reader in enumerate(reader_names(p["dataset"])):
            payload = r["readers"][reader]
            require(payload is not None and payload["score"] in (0, 1), "Missing or out-of-range frozen reader score")
            require(payload["score"] == int(r["stratum"][j]), "Frozen score/stratum mismatch")
            require("extracted" in payload, "Missing frozen reader extracted value")
        if p["dataset"] == "MATH500":
            require(r["readers"].get("fallback") is None, "MATH fallback must be inapplicable")
    by_id = {r["sample_id"]: r for r in samples}
    for lab in labels:
        sid = lab["sample_id"]
        require(lab.get("final_status") in STATUSES, f"Missing/invalid final_status: {sid}")
        require(lab.get("final_answer_correct") in CORRECTNESS, f"Missing/invalid final correctness: {sid}")
        require(lab.get("agreement_type") in AGREEMENTS, f"Missing/invalid agreement_type: {sid}")
        require(lab.get("A_B_answer_equivalence") in ("EQUIVALENT", "DIFFERENT", "UNRESOLVED"), f"Missing human A/B answer equivalence: {sid}")
        for key in ("final_has_conflict", "final_self_correction"):
            require(lab.get(key) in ("YES", "NO"), f"Invalid {key}: {sid}")
        for who in ("A", "B"):
            require(lab.get(who + "_status") in STATUSES, f"Invalid {who} status: {sid}")
            require(lab.get(who + "_answer_correct") in CORRECTNESS, f"Invalid {who} correctness: {sid}")
            for field in ("has_self_correction", "has_conflicting_answer_fields"):
                require(lab.get(who + "_" + field) in ("YES", "NO"), f"Invalid {who} {field}: {sid}")
            require(lab.get(who + "_old_answer_explicitly_withdrawn") in ("YES", "NO", "N/A"), f"Invalid withdrawal: {sid}")
            require(lab.get(who + "_reference_issue") in ("NO", "POSSIBLE_REFERENCE_ERROR"), f"Invalid reference flag: {sid}")
            for field in ("committed_answer", "evidence_span", "notes"):
                require(isinstance(lab.get(who + "_" + field), str), f"Missing {who} {field}: {sid}")
            if lab[who + "_status"] == "CLEAR_ANSWER":
                require(bool(lab[who + "_committed_answer"].strip()), f"Clear answer missing commitment: {sid}")
                require(bool(lab[who + "_evidence_span"].strip()), f"Clear answer missing evidence span: {sid}")
            elif lab[who + "_status"] == "NO_ANSWER":
                require(lab[who + "_answer_correct"] == "INCORRECT", f"NO_ANSWER must be INCORRECT: {sid}")
            else:
                require(lab[who + "_answer_correct"] == "UNRESOLVED", f"Ambiguous/conflicted status must be UNRESOLVED: {sid}")
                require(bool(lab[who + "_evidence_span"].strip() or lab[who + "_notes"].strip()), f"Unresolved answer missing evidence or note: {sid}")
            if lab[who + "_reference_issue"] == "POSSIBLE_REFERENCE_ERROR":
                require(lab[who + "_answer_correct"] == "UNRESOLVED", f"Unadjudicated reference issue must remain UNRESOLVED: {sid}")
        status = lab["final_status"]
        require(isinstance(lab.get("final_committed_answer"), str), f"Missing final committed answer field: {sid}")
        require(isinstance(lab.get("adjudication_reason"), str), f"Missing adjudication reason field: {sid}")
        if status == "CLEAR_ANSWER":
            require(bool(lab["final_committed_answer"].strip()), f"Clear final answer missing commitment: {sid}")
        elif status == "NO_ANSWER":
            require(lab["final_answer_correct"] == "INCORRECT", f"NO_ANSWER must be INCORRECT: {sid}")
        else:
            require(lab["final_answer_correct"] == "UNRESOLVED", f"Unclear final commitment must remain UNRESOLVED: {sid}")
        if lab["agreement_type"] == "FULL_AGREEMENT":
            require(lab["A_status"] == lab["B_status"] and lab["A_answer_correct"] == lab["B_answer_correct"] and
                    lab["A_B_answer_equivalence"] == "EQUIVALENT", f"FULL_AGREEMENT contradicts locked labels: {sid}")
        final_changed = lab["final_status"] != lab["A_status"] or lab["final_answer_correct"] != lab["A_answer_correct"]
        if lab["agreement_type"] != "FULL_AGREEMENT" or final_changed:
            require(bool(lab["adjudication_reason"].strip()), f"Disagreement requires adjudication reason: {sid}")
        readers = reader_names(by_id[sid]["dataset"])
        for reader in ("strict", "fallback", "mv"):
            fidelity = lab.get(reader + "_fidelity")
            require(fidelity in FIDELITIES, f"Missing/invalid human {reader} fidelity: {sid}")
            if reader not in readers:
                require(fidelity == "NOT_APPLICABLE", f"Inapplicable reader fidelity must be NOT_APPLICABLE: {sid}")
            elif status != "CLEAR_ANSWER":
                require(fidelity in ("UNRESOLVED", "NOT_APPLICABLE"), f"Fidelity requires clear commitment: {sid}")
            else:
                require(fidelity != "NOT_APPLICABLE", f"Clear commitment requires human fidelity assessment: {sid}")
    return pop, {r["sample_id"]: r for r in labels}


def _endpoints(label, readers):
    h = human_value(label)
    eq = label["A_B_answer_equivalence"]
    status_same = label["A_status"] == label["B_status"]
    correctness_same = label["A_answer_correct"] == label["B_answer_correct"]
    flags_same = all(label["A_" + k] == label["B_" + k] for k in
                     ("has_self_correction", "has_conflicting_answer_fields", "old_answer_explicitly_withdrawn", "reference_issue"))
    flags_same = flags_same and all(label.get("A_fidelity_" + slot) == label.get("B_fidelity_" + slot) for slot in ("R1", "R2", "R3"))
    values = {"h_lower": h == 1, "h_upper": h != 0, "unresolved": h is None,
              "clear": label["final_status"] == "CLEAR_ANSWER", "status_agreement": status_same,
              "correctness_agreement": correctness_same, "answer_lower": eq == "EQUIVALENT",
              "answer_upper": eq != "DIFFERENT", "raw_agreement": status_same and correctness_same and flags_same and eq == "EQUIVALENT",
              "adjudication": label["agreement_type"] != "FULL_AGREEMENT",
              "reviewed": label.get("adjudicated", bool(label["adjudication_reason"].strip()) and
                                    label["adjudication_reason"] != "INDEPENDENT_A_B_AGREEMENT_NOT_SELECTED_FOR_AUDIT"),
              "reference_issue": any(label[w + "_reference_issue"] == "POSSIBLE_REFERENCE_ERROR" for w in ("A", "B"))}
    for reader in readers:
        f = label[reader + "_fidelity"]
        clear = label["final_status"] == "CLEAR_ANSWER"
        values[reader + "_fidelity_lower"] = clear and f == "FAITHFUL"
        values[reader + "_fidelity_upper"] = clear and f in ("FAITHFUL", "UNRESOLVED")
        values[reader + "_fidelity_unresolved"] = clear and f == "UNRESOLVED"
    return {key: int(value) for key, value in values.items()}


def kappa(labels, weights, field):
    total = sum(weights)
    joint = Counter()
    for label, weight in zip(labels, weights):
        joint[label["A_" + field], label["B_" + field]] += weight / total
    marg_a, marg_b = Counter(), Counter()
    for (a, b), value in joint.items():
        marg_a[a] += value; marg_b[b] += value
    observed = sum(value for (a, b), value in joint.items() if a == b)
    expected = sum(marg_a[value] * marg_b[value] for value in set(marg_a) | set(marg_b))
    value = (observed - expected) / (1 - expected) if expected < 1 - 1e-12 else None
    return value, [{"A": a, "B": b, "population_fraction": v} for (a, b), v in sorted(joint.items())]


def compute_results(samples, populations, labels, alpha=.05, production=False):
    """Pure computation for verified inputs; CLI additionally enforces receipts.

    The API supports small synthetic unit fixtures. It is not an authorization
    route for publishing human results; only run_package writes result files.
    """
    pop, by_id = validate_inputs(samples, populations, labels, production)
    groups = defaultdict(list)
    for r in samples:
        groups[r["state"], r["stratum"]].append(r)
    events = {r["sample_id"]: _endpoints(by_id[r["sample_id"]], reader_names(r["dataset"])) for r in samples}
    comparisons = sum(len(events[rows[0]["sample_id"]]) for rows in groups.values()
                      if rows[0]["stratum_sample_n"] < rows[0]["stratum_population"])
    cell_alpha = alpha / max(1, comparisons)
    cells = {}
    for (state, bits), rows in groups.items():
        N, n = rows[0]["stratum_population"], len(rows)
        cells[state, bits] = {}
        for event in events[rows[0]["sample_id"]]:
            successes = sum(events[r["sample_id"]][event] for r in rows)
            lo, hi = hypergeom_interval(successes, n, N, cell_alpha)
            cells[state, bits][event] = exact(successes / n) | {"ci_low": lo, "ci_high": hi}

    def aggregate(state, lower, upper=None, select=None, complement=False):
        sums = [0.0, 0.0, 0.0, 0.0]
        for bits, counts in pop[state]["strata"].items():
            if counts["N"] == 0 or (select is not None and not select(bits)):
                continue
            weight = counts["N"] / pop[state]["N"]
            low, high = cells[state, bits][lower], cells[state, bits][upper or lower]
            values = [low["estimate"], high["estimate"], low["ci_low"], high["ci_high"]]
            if complement:
                values = [1 - values[1], 1 - values[0], 1 - values[3], 1 - values[2]]
            sums = [a + weight * b for a, b in zip(sums, values)]
        return interval(*sums)

    states, agreement = {}, {}
    for state, p in pop.items():
        human = aggregate(state, "h_lower", "h_upper")
        clear = aggregate(state, "clear")
        state_metrics = {"dataset": p["dataset"], "model": p["model"], "N": p["N"],
                         "sample_n": sum(len(v) for (s, _), v in groups.items() if s == state),
                         "human_accuracy": human, "human_definitely_correct_mass": aggregate(state, "h_lower"),
                         "unresolved_rate": aggregate(state, "unresolved"), "clear_commitment_rate": clear,
                         "reference_issue_rate": aggregate(state, "reference_issue"), "readers": {}}
        for j, reader in enumerate(reader_names(p["dataset"])):
            fp = aggregate(state, "h_lower", "h_upper", select=lambda bits, j=j: bits[j] == "1", complement=True)
            fn = aggregate(state, "h_lower", "h_upper", select=lambda bits, j=j: bits[j] == "0")
            acc = exact(p["reader_accuracy"][reader], note="Observed full frozen cache; no human sampling uncertainty.")
            fidelity = ratio(aggregate(state, reader + "_fidelity_lower", reader + "_fidelity_upper"), clear,
                             "all CLEAR_ANSWER commitments; unresolved fidelity retained in bounds")
            state_metrics["readers"][reader] = {"reader_accuracy": acc, "joint_fp": fp, "joint_fn": fn,
                "net_bias": linear([acc, human], [1, -1]), "extraction_fidelity": fidelity,
                "extraction_fidelity_unresolved_mass": aggregate(state, reader + "_fidelity_unresolved"),
                "conditional_fpr": ratio(fp, linear([exact(1), human], [1, -1]), "H=0 responses"),
                "conditional_fnr": ratio(fn, human, "H=1 responses")}
        states[state] = state_metrics
        state_rows = [r for r in samples if r["state"] == state]
        state_labels = [by_id[r["sample_id"]] for r in state_rows]
        weights = [r["weight"] for r in state_rows]
        status_k, status_table = kappa(state_labels, weights, "status")
        correct_k, correct_table = kappa(state_labels, weights, "answer_correct")
        agreement[state] = {"sample_n": len(state_rows), "status_agreement": aggregate(state, "status_agreement"),
            "correctness_agreement": aggregate(state, "correctness_agreement"), "raw_agreement": aggregate(state, "raw_agreement"),
            "answer_equivalence": aggregate(state, "answer_lower", "answer_upper"),
            "adjudication_rate": aggregate(state, "adjudication"), "reviewed_rate": aggregate(state, "reviewed"),
            "status_kappa": status_k, "correctness_kappa": correct_k,
            "status_confusion_population_fraction": status_table, "correctness_confusion_population_fraction": correct_table,
            "unweighted_sample_raw_agreement": sum(events[r["sample_id"]]["raw_agreement"] for r in state_rows) / len(state_rows),
            "kappa_note": "Descriptive population-weighted Cohen kappa; no CI. Undefined if both marginals are one identical category."}

    specs = [("qwen15", {"qwen15_base": -1.0, "qwen15_rl": 1.0}),
             ("smol", {"smol_base": -1.0, "smol_rl": 1.0})]
    specs.extend(("math_" + seed, {"initial": -1.0, seed: 1.0}) for seed in ("seed83", "seed84", "seed85"))
    specs.append(("math_fixed_seed_mean", {"initial": -1.0, "seed83": 1 / 3, "seed84": 1 / 3, "seed85": 1 / 3}))
    gains = []
    for name, coefficients in specs:
        if not set(coefficients) <= set(states):
            continue
        names, coefs = list(coefficients), list(coefficients.values())
        human = linear([states[s]["human_accuracy"] for s in names], coefs)
        dataset = states[names[0]]["dataset"]
        for reader in reader_names(dataset):
            measured = sum(c * pop[s]["reader_accuracy"][reader] for s, c in coefficients.items())
            gap = linear([exact(measured), human], [1, -1])
            fp = linear([states[s]["readers"][reader]["joint_fp"] for s in names], coefs)
            recovery = linear([states[s]["readers"][reader]["joint_fn"] for s in names], [-c for c in coefs])
            gains.append({"comparison": name, "dataset": dataset, "reader": reader,
                          "primary": (name in ("qwen15", "smol") and reader == "strict") or (name == "math_fixed_seed_mean" and reader == "strict"),
                          "primary_estimand": "human_gain" if name == "math_fixed_seed_mean" else "reader_minus_human_gain",
                          "coefficients": coefficients, "reader_gain": measured, "human_gain": human,
                          "reader_minus_human_gain": gap, "bias_drift": gap,
                          "fn_recovery": recovery, "fp_increase": fp})
    return {"status": "COMPLETE_HUMAN_ANALYSIS" if production else "SYNTHETIC_COMPUTATION_NOT_HUMAN_RESULTS",
            "sampling": {"sample_n": len(samples), "population_response_states": sum(p["N"] for p in populations),
                         "confidence_level": 1 - alpha, "method": "Bonferroni simultaneous exact finite-population hypergeometric inversion",
                         "noncensus_binary_cells": comparisons, "per_cell_alpha": cell_alpha,
                         "scope": "Frozen responses and fixed checkpoints/seeds only; excludes new-item and training-run uncertainty.",
                         "partial_identification": "Point bounds retain all unresolved outcomes; CI encloses both bound endpoints simultaneously."},
            "states": states, "gains": gains, "agreement": agreement,
            "interpretation": {"primary_family": ["qwen15 strict gain minus human gain", "smol strict gain minus human gain", "MATH500 fixed-three-seed mean human gain"],
                "family_control": "One simultaneous family covers all declared stratum endpoints, hence all propagated primary and secondary contrast intervals.",
                "prohibited": "Do not interpret committed-answer correctness as whole-chain reasoning validity or a causal decomposition."}}


def verify_receipts(package, samples, labels):
    coord = package / "coordinator"
    for name in ("protocol_lock.json", "completion_receipts.json"):
        require((coord / name).is_file(), f"Human analysis blocked: missing {name}")
    protocol, completion = read_json(coord / "protocol_lock.json"), read_json(coord / "completion_receipts.json")
    require(protocol.get("status") == "FROZEN_AFTER_HUMAN_PILOT" and protocol.get("human_pilot_completed") is True,
            "Human analysis requires protocol frozen after completed human pilot")
    require(bool(protocol.get("coordinator_name")) and bool(protocol.get("utc")) and bool(protocol.get("pilot_feedback_sha256")), "Incomplete human pilot/protocol receipt")
    for key, path in (("sample_sha256", coord / "frozen_sample.json"),
                      ("protocol_sha256", package / "HUMAN_ANNOTATION_GUIDELINES.md"),
                      ("analysis_sha256", package / "scripts" / "analyze.py"),
                      ("analysis_spec_sha256", package / "ANALYSIS_SPEC.md"),
                      ("workflow_sha256", package / "scripts" / "workflow.py")):
        require(path.is_file() and protocol.get(key) == sha256(path), f"Frozen protocol hash mismatch: {key}")
    require(protocol["analysis_sha256"] == sha256(Path(__file__)), "Executed analyzer differs from frozen analyzer")
    require((coord / "SAMPLE_FROZEN.json").is_file(), "Missing original sample freeze receipt")
    freeze = read_json(coord / "SAMPLE_FROZEN.json")
    require(freeze.get("sample_sha256") == protocol["sample_sha256"] and freeze.get("population_sha256") == sha256(coord / "population.json"),
            "Frozen sample/population changed")
    require(completion.get("status") == "COMPLETE" and completion.get("human_labels") == len(samples) == 625, "Human completion receipt must certify all 625 labels")
    require(completion.get("independent_annotators") == ["A", "B"], "Independent A/B completion receipts required")
    require(completion.get("adjudication_completed") is True and completion.get("fidelity_completed") is True,
            "Human adjudication and extraction-fidelity assessments must be complete")
    require(completion.get("final_labels_sha256") == sha256(coord / "final_labels.json"), "Final labels changed after completion")
    require(completion.get("protocol_lock_sha256") == sha256(coord / "protocol_lock.json"), "Protocol lock changed after completion")
    ids = {r["sample_id"] for r in samples}
    sample_map = {r["sample_id"]: r for r in samples}
    final_map = {r["sample_id"]: r for r in labels}
    require(set(final_map) == ids and len(labels) == 625, "Incomplete or duplicate final labels")
    locked_files = {}
    for who in ("A", "B"):
        for stage in (1, 2):
            path = coord / "locks" / f"{who}_stage{stage}.json"
            require(path.is_file(), f"Missing independent Stage {stage} lock for {who}")
            require(completion.get(f"stage{stage}_locks", {}).get(who) == sha256(path), f"Stage {stage} lock hash mismatch: {who}")
            lock = read_json(path)
            require(lock.get("role") == who and str(lock.get("stage")) in (str(stage), f"stage{stage}", f"Stage {stage}"), f"Invalid lock role/stage: {who}/{stage}")
            require(lock.get("sample_sha256") == protocol["sample_sha256"] and lock.get("protocol_lock_sha256") == sha256(coord / "protocol_lock.json"), "A/B lock belongs to another frozen sample/protocol")
            require(isinstance(lock.get("labels"), list) and {r.get("sample_id") for r in lock["labels"]} == ids and len(lock["labels"]) == 625,
                    "Incomplete or duplicate independent A/B locked labels")
            require(bool(lock.get("signer", "").strip()) and len(lock.get("source_sha256", "")) == 64,
                    "Independent human signer/source receipt missing")
            require(isinstance(lock.get("archived_source"), str), "Archived independent submission path missing")
            archive = (package / lock["archived_source"]).resolve()
            require(archive.is_relative_to(package.resolve()) and archive.is_file(), "Archived submission must exist inside the package")
            require(sha256(archive) == lock["source_sha256"], "Archived independent submission hash changed")
            fields = STAGE1_FIELDS + (STAGE2_FIELDS if stage == 2 else ())
            for locked in lock["labels"]:
                sid = locked["sample_id"]
                for field in fields:
                    key = who + "_" + field
                    require(key in locked and final_map[sid].get(key) == locked[key], f"Final labels changed locked {key}: {sid}")
                for field in ("question", "response"):
                    require(locked.get(field) == sample_map[sid][field], f"Locked source {field} changed: {sid}")
                if stage == 2:
                    require(locked.get("reference_answer") == sample_map[sid]["reference_answer"], f"Locked reference changed: {sid}")
                    for j in range(3):
                        order = sample_map[sid]["candidate_order"]
                        payload = sample_map[sid]["readers"][order[j]]["extracted"] if j < len(order) else None
                        require(locked.get(f"extracted_R{j+1}") == json.dumps(payload, ensure_ascii=False), f"Locked anonymous candidate changed: {sid}")
            locked_files[who, stage] = lock

    def timestamp(receipt):
        try:
            value = datetime.fromisoformat(receipt["utc"].replace("Z", "+00:00"))
        except (ValueError, KeyError, TypeError):
            raise ValueError("Invalid lock chronology timestamp") from None
        require(value.tzinfo is not None, "Lock chronology requires timezone-aware timestamps")
        return value

    freeze_time = timestamp(protocol)
    latest_stage1 = max(timestamp(locked_files[who, 1]) for who in ("A", "B"))
    latest_stage2 = max(timestamp(locked_files[who, 2]) for who in ("A", "B"))
    for who in ("A", "B"):
        require(timestamp(locked_files[who, 1]) >= freeze_time, "Stage 1 cannot precede protocol freeze")
        require(timestamp(locked_files[who, 2]) >= latest_stage1, "Stage 2 cannot precede both Stage 1 locks")
        require(locked_files[who, 1]["signer"].strip().casefold() == locked_files[who, 2]["signer"].strip().casefold(), "Annotator identity changed between stages")
    require(locked_files["A", 1]["signer"].strip().casefold() != locked_files["B", 1]["signer"].strip().casefold(), "A/B signers must be distinct humans")
    adjudicator = completion.get("adjudicator", "").strip().casefold()
    require(adjudicator and adjudicator not in {locked_files[who, 1]["signer"].strip().casefold() for who in ("A", "B")},
            "Adjudicator must be a distinct third human")
    require(timestamp(completion) >= latest_stage2, "Completion cannot precede both Stage 2 locks")
    return {"protocol_lock_sha256": sha256(coord / "protocol_lock.json"), "completion_receipts_sha256": sha256(coord / "completion_receipts.json"),
            "frozen_sample_sha256": sha256(coord / "frozen_sample.json"), "population_sha256": sha256(coord / "population.json"),
            "final_labels_sha256": sha256(coord / "final_labels.json"), "analysis_sha256": sha256(package / "scripts" / "analyze.py")}


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def tex_number(metric):
    point = f"{100 * metric['estimate']:.1f}" if metric["estimate"] is not None else f"[{100 * metric['lower_estimate']:.1f},{100 * metric['upper_estimate']:.1f}]"
    return r"\shortstack{" + point + r"\\{\scriptsize [" + f"{100 * metric['ci_low']:.1f},{100 * metric['ci_high']:.1f}" + "]}}"


def make_table(results=None):
    lines = [r"% Requires booktabs. All values are percentage points. No paper source is modified.",
             r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{llrrr}", r"\toprule", r"Dataset / model & Reader & $\Delta_E$ & $\Delta_H$ & $\Delta_E-\Delta_H$ \\", r"\midrule"]
    titles = {"qwen15": "GSM8K / Qwen1.5B", "smol": "GSM8K / Smol1.7B", "math_fixed_seed_mean": "MATH500 / seed mean"}
    if results:
        for gain in results["gains"]:
            if gain["comparison"] in titles:
                lines.append(f"{titles[gain['comparison']]} & {gain['reader'].upper()} & {100 * gain['reader_gain']:.1f} & " +
                             tex_number(gain["human_gain"]) + " & " + tex_number(gain["reader_minus_human_gain"]) + r" \\")
    else:
        for name, dataset in (("qwen15", "GSM8K"), ("smol", "GSM8K"), ("math_fixed_seed_mean", "MATH500")):
            for reader in reader_names(dataset):
                placeholder = r"\shortstack{XX.X\\{\scriptsize [XX.X, XX.X]}}"
                lines.append(f"{titles[name]} & {reader.upper()} & XX.X & {placeholder} & {placeholder}" + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}",
        r"\caption{Human semantic calibration of automated readers. Human labels measure committed-answer correctness, not whole-chain reasoning validity. Responses are stratified probability samples; estimates use original population weights. Labels come from independent double-blind annotation followed by adjudication. Brackets below estimates are conservative simultaneous 95\% finite-cache confidence bounds including unresolved-label sensitivity; an interval in the estimate position is the unresolved-label identification range. The MATH500 mean uses the three fixed trained seeds and one shared initial state. FP, FN, and bias decomposition are reported separately.}",
        r"\label{tab:human-semantic-calibration}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def make_error_table(results=None):
    lines = [r"% Appendix table; requires booktabs. Values are percentages of ALL responses.",
             r"\begin{table}[t]", r"\centering\scriptsize", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{llrrr}", r"\toprule", r"State & Reader & Joint FP & Joint FN & Net bias \\", r"\midrule"]
    titles = {"qwen15_base": "Qwen1.5B init", "qwen15_rl": "Qwen1.5B RL", "smol_base": "Smol1.7B init",
              "smol_rl": "Smol1.7B RL", "initial": "MATH shared init", "seed83": "MATH seed 83",
              "seed84": "MATH seed 84", "seed85": "MATH seed 85"}
    if results:
        for state, payload in results["states"].items():
            for reader, metrics in payload["readers"].items():
                lines.append(f"{titles.get(state, state)} & {reader.upper()} & " + " & ".join(
                    tex_number(metrics[key]) for key in ("joint_fp", "joint_fn", "net_bias")) + r" \\")
    else:
        for state, title in titles.items():
            dataset = "GSM8K" if state.startswith(("qwen", "smol")) else "MATH500"
            for reader in reader_names(dataset):
                cell = r"\shortstack{XX.X\\{\scriptsize [XX.X, XX.X]}}"
                lines.append(f"{title} & {reader.upper()} & {cell} & {cell} & {cell}" + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}",
        r"\caption{Population-weighted reader errors against adjudicated committed-answer correctness. FP and FN are joint probabilities over all frozen responses, not conditional FPR/FNR. Net bias is FP minus FN. Brackets below estimates are simultaneous 95\% finite-population bounds including unresolved-label sensitivity. Extraction fidelity, conditional rates, and agreement diagnostics are provided in the accompanying analysis data.}",
        r"\label{tab:human-semantic-errors}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def write_outputs(package, result):
    out = package / "results"; out.mkdir(parents=True, exist_ok=True)
    (out / "human_validation_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    metric_rows, gain_rows, agreement_rows = [], [], []
    for state, payload in result.get("states", {}).items():
        for metric, value in payload.items():
            if isinstance(value, dict) and "ci_low" in value:
                metric_rows.append({"state": state, "dataset": payload["dataset"], "reader": "HUMAN", "metric": metric, **value})
        for reader, metrics in payload["readers"].items():
            for metric, value in metrics.items():
                metric_rows.append({"state": state, "dataset": payload["dataset"], "reader": reader, "metric": metric, **value})
    for g in result.get("gains", []):
        h, gap = g["human_gain"], g["reader_minus_human_gain"]
        gain_rows.append({"comparison": g["comparison"], "dataset": g["dataset"], "reader": g["reader"], "primary": g["primary"],
            "reader_gain": g["reader_gain"], "human_gain": h["estimate"], "human_lower": h["lower_estimate"], "human_upper": h["upper_estimate"],
            "human_ci_low": h["ci_low"], "human_ci_high": h["ci_high"], "gap": gap["estimate"], "gap_lower": gap["lower_estimate"],
            "gap_upper": gap["upper_estimate"], "gap_ci_low": gap["ci_low"], "gap_ci_high": gap["ci_high"], "bias_drift": gap["estimate"],
            "fn_recovery_lower": g["fn_recovery"]["ci_low"], "fn_recovery_upper": g["fn_recovery"]["ci_high"],
            "fp_increase_lower": g["fp_increase"]["ci_low"], "fp_increase_upper": g["fp_increase"]["ci_high"]})
    for state, payload in result.get("agreement", {}).items():
        for metric, value in payload.items():
            if isinstance(value, dict) and "ci_low" in value:
                agreement_rows.append({"state": state, "metric": metric, **value})
            elif metric.endswith("kappa") or metric == "unweighted_sample_raw_agreement":
                agreement_rows.append({"state": state, "metric": metric, "estimate": value, "note": "Descriptive; no confidence interval."})
    write_csv(out / "metrics.csv", METRIC_COLUMNS, metric_rows)
    write_csv(out / "gains.csv", GAIN_COLUMNS, gain_rows)
    write_csv(out / "agreement.csv", METRIC_COLUMNS, agreement_rows)
    (out / "paper_human_validation_table.tex").write_text(make_table(result if "states" in result else None), encoding="utf-8")
    (out / "appendix_human_validation_errors.tex").write_text(make_error_table(result if "states" in result else None), encoding="utf-8")


def run_package(package):
    package = Path(package).resolve()
    coord = package / "coordinator"
    label_path = coord / "final_labels.json"
    labels = read_json(label_path) if label_path.exists() else []
    require(isinstance(labels, list), "final_labels.json must be a list")
    filled = [r for r in labels if r.get("final_status") not in (None, "")]
    if not filled:
        result = {"status": "PENDING_HUMAN_LABELS", "human_labels_available": 0,
                  "message": "No human results. Complete human pilot, protocol lock, both independent two-stage annotations, adjudication, and fidelity assessment.",
                  "expected_sample_n": 625, "generated_utc": datetime.now(timezone.utc).isoformat()}
        write_outputs(package, result)
        return result
    require(len(filled) == len(labels), "Partial final labels: analysis requires all 625 completed rows")
    for name in ("frozen_sample.json", "population.json"):
        require((coord / name).is_file(), f"Missing frozen analysis input: {name}")
    samples, population = read_json(coord / "frozen_sample.json"), read_json(coord / "population.json")
    receipts = verify_receipts(package, samples, labels)
    result = compute_results(samples, population, labels, production=True)
    result["receipts"] = receipts
    result["generated_utc"] = datetime.now(timezone.utc).isoformat()
    write_outputs(package, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--self-test", action="store_true", help="Run labeled SYNTHETIC unit tests; disk fixtures live only in temporary directories.")
    args = parser.parse_args(argv)
    if args.self_test:
        suite = unittest.TestSuite(unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern=pattern)
                                   for pattern in ("test_analysis.py", "test_e2e.py"))
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    try:
        result = run_package(args.package)
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"ANALYSIS_BLOCKED: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "output": str(args.package.resolve() / "results")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
