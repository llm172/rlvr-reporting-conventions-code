"""Answer-only GSM8K reward control using the frozen lenient reader.

The reader does not require a hash marker: when absent it falls back to the
last number. This differs from the strict reward used in the matched hash arm.
The scorer is imported from the frozen evaluation module.
"""

import importlib.util
import os

_SCORER_PATH = os.environ.get(
    "PASSK_SCORER_PATH", os.path.join(os.path.dirname(__file__), "passk_score.py"))

_spec = importlib.util.spec_from_file_location("frozen_passk_score", _SCORER_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError("cannot load frozen scorer from %s" % _SCORER_PATH)
_frozen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_frozen)

# Fail at import time, not 40 steps into training, if the frozen scorer ever
# stops exposing the function this arm is defined in terms of.
if not hasattr(_frozen, "gsm8k_lenient"):
    raise RuntimeError("frozen scorer at %s has no gsm8k_lenient" % _SCORER_PATH)


def compute_score(data_source=None, solution_str=None, ground_truth=None,
                  extra_info=None, score=1.0, format_score=0.0):
    """Return `score` if the lenient rule accepts the response, else 0.

    Binary, same shape as the strict reward it replaces -- no partial credit and
    no format bonus, so the ONLY difference between this arm and the main run is
    whether the `#### ` marker is required. Anything else would confound the
    comparison.
    """
    if solution_str is None or ground_truth is None:
        return 0.0
    try:
        ok = _frozen.gsm8k_lenient(solution_str, ground_truth)
    except Exception:
        # A scorer exception must not be silently rewarded. Return no credit and
        # let the run's reward curve show it.
        return 0.0
    return float(score) if ok else float(format_score)


if __name__ == "__main__":
    # Self-test. Run this BEFORE launching training: a reward function that
    # returns 0 for everything trains a perfectly flat, perfectly useless run,
    # and the failure looks like "RL didn't help" rather than "reward is broken".
    # That is the silent-degenerate-metric failure mode, so it gets a gate.
    cases = [
        # (response, ground_truth, expect)
        ("The answer is #### 72", "#### 72", 1.0),        # marker present, right
        ("blah blah so we get 72", "#### 72", 1.0),        # NO marker, still right
        ("blah blah so we get 71", "#### 72", 0.0),        # no marker, wrong
        ("#### 71", "#### 72", 0.0),                       # marker present, wrong
        ("the total is 1,234", "#### 1234", 1.0),          # comma handling
        ("", "#### 72", 0.0),                              # empty
    ]
    bad = 0
    for resp, gt, expect in cases:
        got = compute_score(solution_str=resp, ground_truth=gt)
        flag = "ok " if got == expect else "FAIL"
        if got != expect:
            bad += 1
        print("%s expect=%.1f got=%.1f  %r" % (flag, expect, got, resp[:40]))

    # The load-bearing property: this reward must be obtainable WITHOUT the
    # marker. If it is not, the arm is not testing what it claims to test.
    no_marker = compute_score(solution_str="so we get 72", ground_truth="#### 72")
    print("\nformat-blind check: reward without marker = %.1f %s"
          % (no_marker, "(OK)" if no_marker == 1.0 else "(BROKEN - ABORT)"))
    if no_marker != 1.0:
        bad += 1

    print("\n%d failures" % bad)
    raise SystemExit(1 if bad else 0)
