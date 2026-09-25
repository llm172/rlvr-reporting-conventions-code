"""Synthetic unit tests. No fixture represents a human judgment."""
import copy
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest

import analyze


def fixture(state="qwen15_base", sizes=(90, 10), counts=(2, 2), truths=(1, 0), dataset="GSM8K"):
    readers = ["strict", "fallback", "mv"] if dataset == "GSM8K" else ["strict", "mv"]
    rows, labels, strata = [], [], {}
    for j, (N, n, truth) in enumerate(zip(sizes, counts, truths)):
        bits = str(j) * len(readers)
        strata[bits] = {"N": N, "n": n}
        for i in range(n):
            sid = f"{state}-{j}-{i}"
            rows.append({"sample_id": sid, "dataset": dataset, "state": state, "model": "synthetic",
                         "original_id": f"{j}-{i}", "stratum": bits, "population_n": sum(sizes),
                         "stratum_population": N, "stratum_sample_n": n,
                         "inclusion_probability": n / N, "weight": N / n,
                         "readers": {r: {"score": j, "extracted": ["1"]} if r in readers else None
                                     for r in ("strict", "fallback", "mv")}})
            lab = {"sample_id": sid, "final_status": "CLEAR_ANSWER", "final_committed_answer": "1",
                   "final_answer_correct": "CORRECT" if truth else "INCORRECT",
                   "final_has_conflict": "NO", "final_self_correction": "NO",
                   "adjudication_reason": "", "agreement_type": "FULL_AGREEMENT",
                   "A_B_answer_equivalence": "EQUIVALENT"}
            for r in ("strict", "fallback", "mv"):
                lab[r + "_fidelity"] = "FAITHFUL" if r in readers else "NOT_APPLICABLE"
            for who in ("A", "B"):
                for name, value in {"status": "CLEAR_ANSWER", "committed_answer": "1", "evidence_span": "answer 1",
                                    "has_self_correction": "NO", "has_conflicting_answer_fields": "NO",
                                    "old_answer_explicitly_withdrawn": "N/A",
                                    "answer_correct": lab["final_answer_correct"], "reference_issue": "NO", "notes": ""}.items():
                    lab[who + "_" + name] = value
            labels.append(lab)
    pop = {"state": state, "dataset": dataset, "model": "synthetic", "N": sum(sizes), "strata": strata,
           "reader_accuracy": {r: sizes[1] / sum(sizes) for r in readers}}
    return rows, [pop], labels


def receipt_fixture(package, full_design=False):
    """Fake attestations for integrity tests, written ONLY into a TemporaryDirectory."""
    rows, pop, labels = fixture(sizes=(623, 2), counts=(623, 2))
    if full_design:
        rows, pop, labels = [], [], []
        for state, quota in analyze.QUOTAS.items():
            dataset = "GSM8K" if state in ("qwen15_base", "qwen15_rl", "smol_base", "smol_rl") else "MATH500"
            N = 1319 if dataset == "GSM8K" else 500
            a, b, c = fixture(state, (N - 1, 1), (quota - 1, 1), (1, 0), dataset)
            rows.extend(a); pop.extend(b); labels.extend(c)
    coord = package / "coordinator"; (coord / "locks").mkdir(parents=True)
    (package / "scripts").mkdir()
    shutil.copyfile(analyze.__file__, package / "scripts" / "analyze.py")
    def write(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")
    for name in ("HUMAN_ANNOTATION_GUIDELINES.md", "ANALYSIS_SPEC.md", "scripts/workflow.py"):
        (package / name).write_text("SYNTHETIC RECEIPT TEST ONLY", encoding="utf-8")
    for row in rows:
        row.update(question="SYNTHETIC", response="SYNTHETIC", reference_answer="1", candidate_order=list(analyze.reader_names(row["dataset"])))
    write(coord / "frozen_sample.json", rows); write(coord / "population.json", pop)
    sample_hash = analyze.sha256(coord / "frozen_sample.json")
    write(coord / "SAMPLE_FROZEN.json", {"sample_sha256": sample_hash, "population_sha256": analyze.sha256(coord / "population.json")})
    protocol = {"status": "FROZEN_AFTER_HUMAN_PILOT", "human_pilot_completed": True, "sample_sha256": sample_hash,
                "protocol_sha256": analyze.sha256(package / "HUMAN_ANNOTATION_GUIDELINES.md"),
                "analysis_sha256": analyze.sha256(package / "scripts/analyze.py"),
                "analysis_spec_sha256": analyze.sha256(package / "ANALYSIS_SPEC.md"),
                "workflow_sha256": analyze.sha256(package / "scripts/workflow.py"),
                "utc": "2026-01-01T00:00:00+00:00", "coordinator_name": "SYNTHETIC COORDINATOR",
                "pilot_feedback_sha256": "0" * 64}
    write(coord / "protocol_lock.json", protocol)
    protocol_hash = analyze.sha256(coord / "protocol_lock.json")
    completion = {"status": "COMPLETE", "human_labels": 625, "independent_annotators": ["A", "B"],
                  "stage1_locks": {}, "stage2_locks": {}, "protocol_lock_sha256": protocol_hash,
                  "adjudication_completed": True, "fidelity_completed": True, "utc": "2026-01-04T00:00:00+00:00",
                  "adjudicator": "SYNTHETIC THIRD PERSON"}
    for row, lab in zip(rows, labels):
        for who in ("A", "B"):
            lab[who + "_stage2_notes"] = ""
            for slot in ("R1", "R2", "R3"):
                lab[who + "_fidelity_" + slot] = "NOT_APPLICABLE" if row["dataset"] == "MATH500" and slot == "R3" else "FAITHFUL"
    for who in ("A", "B"):
        for stage in (1, 2):
            locked_rows = []
            for row, lab in zip(rows, labels):
                locked = {k: row[k] for k in ("sample_id", "question", "response")}
                fields = analyze.STAGE1_FIELDS + (analyze.STAGE2_FIELDS if stage == 2 else ())
                locked.update({who + "_" + field: lab[who + "_" + field] for field in fields})
                if stage == 2:
                    locked["reference_answer"] = row["reference_answer"]
                    for j in range(3):
                        payload = row["readers"][row["candidate_order"][j]]["extracted"] if j < len(row["candidate_order"]) else None
                        locked[f"extracted_R{j+1}"] = json.dumps(payload, ensure_ascii=False)
                locked_rows.append(locked)
            path = coord / "locks" / f"{who}_stage{stage}.json"
            archive = coord / "submissions" / f"{who}_stage{stage}.json"
            archive.parent.mkdir(exist_ok=True); write(archive, locked_rows)
            write(path, {"role": who, "stage": stage, "utc": f"2026-01-0{stage+1}T00:00:00+00:00",
                         "source_sha256": analyze.sha256(archive), "signer": "SYNTHETIC " + who,
                         "archived_source": archive.relative_to(package).as_posix(),
                         "labels": locked_rows, "sample_sha256": sample_hash, "protocol_lock_sha256": protocol_hash})
            completion[f"stage{stage}_locks"][who] = analyze.sha256(path)
    write(coord / "final_labels.json", labels)
    completion["final_labels_sha256"] = analyze.sha256(coord / "final_labels.json")
    write(coord / "completion_receipts.json", completion)
    return rows, labels


class AnalysisTests(unittest.TestCase):
    def test_unequal_weights_and_joint_not_conditional_errors(self):
        rows, pop, labels = fixture()
        result = analyze.compute_results(rows, pop, labels)
        metric = result["states"]["qwen15_base"]
        self.assertAlmostEqual(metric["human_accuracy"]["estimate"], .9)
        strict = metric["readers"]["strict"]
        self.assertAlmostEqual(strict["joint_fp"]["estimate"], .1)
        self.assertAlmostEqual(strict["joint_fn"]["estimate"], .9)
        self.assertAlmostEqual(strict["conditional_fpr"]["estimate"], 1)
        self.assertAlmostEqual(strict["net_bias"]["estimate"], -.8)

    def test_zero_errors_do_not_have_zero_width_when_sampled(self):
        lower, upper = analyze.hypergeom_interval(0, 20, 500, .05)
        self.assertEqual(lower, 0)
        self.assertGreater(upper, 0)
        self.assertLess(upper, .2)

    def test_unobserved_conditional_denominator_can_exist_in_population(self):
        numerator = analyze.interval(0, 0, 0, .1)
        denominator = analyze.interval(0, 0, 0, .2)
        value = analyze.ratio(numerator, denominator, "unseen eligible responses")
        self.assertIsNone(value["estimate"])
        self.assertEqual((value["ci_low"], value["ci_high"]), (0, 1))

    def test_census_has_no_sampling_uncertainty(self):
        self.assertEqual(analyze.hypergeom_interval(2, 4, 4, .05), (.5, .5))
        rows, pop, labels = fixture(sizes=(2, 2), counts=(2, 2))
        h = analyze.compute_results(rows, pop, labels)["states"]["qwen15_base"]["human_accuracy"]
        self.assertEqual(h["ci_low"], .5)
        self.assertEqual(h["ci_high"], .5)

    def test_hypergeom_coverage_exhaustive_small_population(self):
        N, n, alpha = 12, 4, .1
        for K in range(N + 1):
            coverage = 0.0
            for x in range(max(0, n - N + K), min(K, n) + 1):
                lo, hi = analyze.hypergeom_interval(x, n, N, alpha)
                if lo - 1e-12 <= K / N <= hi + 1e-12:
                    coverage += math.comb(K, x) * math.comb(N - K, n - x) / math.comb(N, n)
            self.assertGreaterEqual(coverage + 1e-12, 1 - alpha)

    def test_unresolved_bounds_survive_census(self):
        rows, pop, labels = fixture(sizes=(2, 2), counts=(2, 2))
        labels[0]["final_status"] = "AMBIGUOUS"
        labels[0]["final_committed_answer"] = ""
        labels[0]["final_answer_correct"] = "UNRESOLVED"
        labels[0]["adjudication_reason"] = "Synthetic adjudicator changed the status."
        for r in ("strict", "fallback", "mv"):
            labels[0][r + "_fidelity"] = "NOT_APPLICABLE"
        h = analyze.compute_results(rows, pop, labels)["states"]["qwen15_base"]["human_accuracy"]
        self.assertIsNone(h["estimate"])
        self.assertEqual((h["lower_estimate"], h["upper_estimate"]), (.25, .5))
        self.assertEqual((h["ci_low"], h["ci_high"]), (.25, .5))

    def test_shared_math_initial_enters_mean_once(self):
        rows, populations, labels = [], [], []
        for state, truths in [("initial", (1, 0)), ("seed83", (1, 1)), ("seed84", (1, 1)), ("seed85", (1, 1))]:
            a, b, c = fixture(state, (2, 2), (2, 2), truths, "MATH500")
            rows += a; populations += b; labels += c
        results = analyze.compute_results(rows, populations, labels)
        mean = next(g for g in results["gains"] if g["comparison"] == "math_fixed_seed_mean" and g["reader"] == "strict")
        self.assertEqual(mean["coefficients"], {"initial": -1.0, "seed83": 1 / 3, "seed84": 1 / 3, "seed85": 1 / 3})
        self.assertAlmostEqual(mean["human_gain"]["estimate"], .5)
        self.assertEqual(results["sampling"]["sample_n"], 16)

    def test_reject_duplicate_missing_unknown_and_invalid_labels(self):
        rows, pop, labels = fixture()
        cases = [labels + [labels[0]], labels[:-1], labels + [{"sample_id": "intruder"}]]
        invalid = copy.deepcopy(labels); invalid[0]["final_answer_correct"] = "MAYBE"; cases.append(invalid)
        for bad in cases:
            with self.subTest(bad=bad[-1].get("sample_id")):
                with self.assertRaises(ValueError):
                    analyze.compute_results(rows, pop, bad)

    def test_reject_wrong_weights_or_changed_frozen_scores(self):
        rows, pop, labels = fixture()
        rows[0]["weight"] += 1
        with self.assertRaises(ValueError):
            analyze.compute_results(rows, pop, labels)
        rows, pop, labels = fixture()
        rows[0]["readers"]["strict"]["score"] = 1
        with self.assertRaises(ValueError):
            analyze.compute_results(rows, pop, labels)

    def test_fidelity_requires_explicit_human_label(self):
        rows, pop, labels = fixture()
        labels[0]["strict_fidelity"] = ""
        with self.assertRaises(ValueError):
            analyze.compute_results(rows, pop, labels)

    def test_no_answer_is_incorrect_not_unresolved(self):
        rows, pop, labels = fixture(sizes=(2, 2), counts=(2, 2))
        for lab in labels:
            lab["final_status"] = "NO_ANSWER"
            lab["final_committed_answer"] = ""
            lab["final_answer_correct"] = "INCORRECT"
            lab["adjudication_reason"] = "Synthetic adjudicator found no commitment."
            for r in ("strict", "fallback", "mv"):
                lab[r + "_fidelity"] = "NOT_APPLICABLE"
        state = analyze.compute_results(rows, pop, labels)["states"]["qwen15_base"]
        self.assertEqual(state["human_accuracy"]["estimate"], 0)
        self.assertIsNone(state["readers"]["strict"]["extraction_fidelity"]["estimate"])

    def test_agreement_uses_human_equivalence_not_string_matching(self):
        rows, pop, labels = fixture()
        labels[0]["B_committed_answer"] = "1/1"
        results = analyze.compute_results(rows, pop, labels)
        agreement = results["agreement"]["qwen15_base"]
        self.assertEqual(agreement["answer_equivalence"]["estimate"], 1)
        self.assertEqual(agreement["raw_agreement"]["estimate"], 1)
        self.assertEqual(agreement["correctness_kappa"], 1)

    def test_pending_never_emits_human_estimates(self):
        with tempfile.TemporaryDirectory(prefix="hv625_test_") as directory:
            package = Path(directory); (package / "coordinator").mkdir()
            (package / "coordinator" / "final_labels.json").write_text("[]", encoding="utf-8")
            result = analyze.run_package(package)
            self.assertEqual(result["status"], "PENDING_HUMAN_LABELS")
            self.assertNotIn("states", result)
            self.assertIn("XX.X", (package / "results" / "paper_human_validation_table.tex").read_text())

    def test_nonempty_labels_require_real_receipts(self):
        with tempfile.TemporaryDirectory(prefix="hv625_test_") as directory:
            package = Path(directory); (package / "coordinator").mkdir()
            (package / "coordinator" / "final_labels.json").write_text('[{"sample_id":"synthetic","final_status":"CLEAR_ANSWER"}]')
            with self.assertRaises(ValueError):
                analyze.run_package(package)

    def test_unresolved_fidelity_for_unclear_answer_is_not_in_denominator(self):
        rows, pop, labels = fixture(sizes=(2, 2), counts=(2, 2))
        lab = labels[0]
        lab["final_status"] = "AMBIGUOUS"
        lab["final_answer_correct"] = "UNRESOLVED"
        lab["adjudication_reason"] = "Synthetic adjudication."
        for r in ("strict", "fallback", "mv"):
            lab[r + "_fidelity"] = "UNRESOLVED"
        fidelity = analyze.compute_results(rows, pop, labels)["states"]["qwen15_base"]["readers"]["strict"]["extraction_fidelity"]
        self.assertEqual(fidelity["estimate"], 1)

    def test_fake_full_agreement_is_rejected(self):
        rows, pop, labels = fixture()
        labels[0]["B_answer_correct"] = "INCORRECT"
        with self.assertRaises(ValueError):
            analyze.compute_results(rows, pop, labels)

    def test_changed_final_answer_requires_adjudication_reason(self):
        rows, pop, labels = fixture()
        labels[0]["final_answer_correct"] = "INCORRECT"
        with self.assertRaises(ValueError):
            analyze.compute_results(rows, pop, labels)

    def test_receipts_preserve_locked_ab_labels(self):
        with tempfile.TemporaryDirectory(prefix="hv625_test_") as directory:
            package = Path(directory)
            rows, labels = receipt_fixture(package)
            analyze.verify_receipts(package, rows, labels)
            labels[0]["A_committed_answer"] = "CHANGED AFTER LOCK"
            path = package / "coordinator/final_labels.json"
            path.write_text(json.dumps(labels))
            receipt_path = package / "coordinator/completion_receipts.json"
            receipt = analyze.read_json(receipt_path)
            receipt["final_labels_sha256"] = analyze.sha256(path)
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "locked|Locked"):
                analyze.verify_receipts(package, rows, labels)

    def test_stage2_cannot_predate_stage1(self):
        with tempfile.TemporaryDirectory(prefix="hv625_test_") as directory:
            package = Path(directory); rows, labels = receipt_fixture(package)
            path = package / "coordinator/locks/A_stage2.json"
            lock = analyze.read_json(path); lock["utc"] = "2026-01-01T00:00:00+00:00"; path.write_text(json.dumps(lock))
            receipt_path = package / "coordinator/completion_receipts.json"
            receipt = analyze.read_json(receipt_path); receipt["stage2_locks"]["A"] = analyze.sha256(path); receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "chronology|precede|before"):
                analyze.verify_receipts(package, rows, labels)

    def test_adjudicator_must_be_a_third_human(self):
        with tempfile.TemporaryDirectory(prefix="hv625_test_") as directory:
            package = Path(directory); rows, labels = receipt_fixture(package)
            path = package / "coordinator/completion_receipts.json"
            receipt = analyze.read_json(path); receipt["adjudicator"] = "SYNTHETIC A"; path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "third|distinct"):
                analyze.verify_receipts(package, rows, labels)

    def test_archived_submission_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="hv625_test_") as directory:
            package = Path(directory); rows, labels = receipt_fixture(package)
            (package / "coordinator/submissions/A_stage1.json").write_text("TAMPERED")
            with self.assertRaisesRegex(ValueError, "Archived|archive"):
                analyze.verify_receipts(package, rows, labels)

    def test_complete_synthetic_workflow_writes_tables_only_in_temp(self):
        with tempfile.TemporaryDirectory(prefix="hv625_test_") as directory:
            package = Path(directory); receipt_fixture(package, full_design=True)
            result = analyze.run_package(package)
            self.assertEqual(result["status"], "COMPLETE_HUMAN_ANALYSIS")
            self.assertEqual(result["sampling"]["sample_n"], 625)
            self.assertEqual(result["sampling"]["population_response_states"], 7276)
            self.assertEqual(len(result["gains"]), 14)
            self.assertNotIn("XX.X", (package / "results/paper_human_validation_table.tex").read_text())
            self.assertIn("Joint FP", (package / "results/appendix_human_validation_errors.tex").read_text())


if __name__ == "__main__":
    unittest.main()
