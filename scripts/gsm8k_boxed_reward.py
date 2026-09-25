# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""GSM8K reward with the \boxed{} convention instead of ####.

This is a DELIBERATE one-line-of-logic fork of verl's
verl/utils/reward_score/gsm8k.py. It is copied rather than parameterised
because the #### run is frozen: importing a shared, newly-edited module into
the control's reward path would silently change what the frozen numbers mean.

Everything that is not the extraction marker is held identical to the original,
on purpose, because the experiment's whole claim rests on the two rewards
differing in exactly one respect:

  * same 300-character tail clip (_SOLUTION_CLIP_CHARS)
  * same "take the LAST match" rule
  * same normalisation: strip "," and "$"
  * same exact string comparison against ground_truth (no math-expression
    equivalence -- this is strict matching, matched to the #### control; using
    verl's MATH scorer here would have changed the marker AND the matching
    strictness at once and confounded the contrast)
  * same 1.0 / 0.0 payoff with no partial format credit

Two differences are forced by the marker itself and are documented rather than
hidden, because they are properties of the convention under test:

  1. `\boxed{...}` needs a CLOSING brace; `#### 72` does not. A response
     truncated mid-box scores 0 where the same response truncated after ####
     would score 1. That asymmetry is part of what makes a convention easy or
     hard to comply with, so it is left in.
  2. The brace can enclose surrounding whitespace, which `#### (...)` cannot
     capture, so the payload is .strip()ed. Without this a model writing
     `\boxed{ 72 }` would be scored wrong for a spacing choice.
"""

import re

_SOLUTION_CLIP_CHARS = 300

# Non-greedy, no nested braces: GSM8K answers are bare numbers, and allowing
# nesting would let \boxed{\frac{1}{2}} match in a task where it is always wrong
# anyway -- better to score it as unextractable than as a wrong number.
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    # Optimization: Regular expression matching on very long strings can be slow.
    # For math problems, the final answer is usually at the end.
    # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        # this also tests the formatting of the model
        solutions = _BOXED.findall(solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            # take the last solution
            final_answer = solutions[-1].strip().replace(",", "").replace("$", "")
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def compute_score(data_source=None, solution_str=None, ground_truth=None,
                  extra_info=None, method="strict", format_score=0.0, score=1.0,
                  **kwargs):
    r"""The scoring function for GSM8k under the \boxed{} convention.

    Signature note: verl calls a custom reward function with keyword arguments
    including data_source and extra_info, which the built-in gsm8k.compute_score
    does not accept because the dispatcher strips them first. Accepting and
    ignoring them keeps the scoring logic identical while letting this be loaded
    through custom_reward_function.path.
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return 0
    else:
        if answer == ground_truth:
            return score
        else:
            return format_score
