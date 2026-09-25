# Copyright 2025 Individual Contributor: Anonymous Authors
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
"""Hard- and soft-budget accounting for curriculum-identifiability runs."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from verl import DataProto

HARD_BUDGET_FIELDS = (
    "sampled_prompts",
    "completions",
    "reward_calls",
    "optimizer_updates",
    "validation_calls",
)
SOFT_BUDGET_FIELDS = ("wall_time_seconds", "gpu_seconds")
REPORTED_RESOURCE_FIELDS = ("generated_tokens", *SOFT_BUDGET_FIELDS)


def _validate_counter(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_duration(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative finite number")


def _validate_gpu_count(gpu_count: int) -> None:
    if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count <= 0:
        raise ValueError("gpu_count must be a positive integer")


@dataclass(frozen=True)
class BudgetSnapshot:
    """Immutable accounting record for one experimental run."""

    sampled_prompts: int = 0
    completions: int = 0
    generated_tokens: int = 0
    reward_calls: int = 0
    optimizer_updates: int = 0
    validation_calls: int = 0
    wall_time_seconds: float = 0.0
    gpu_seconds: float = 0.0

    def __post_init__(self):
        for name in HARD_BUDGET_FIELDS:
            _validate_counter(name, getattr(self, name))
        _validate_counter("generated_tokens", self.generated_tokens)
        for name in SOFT_BUDGET_FIELDS:
            _validate_duration(name, getattr(self, name))


class BudgetMismatchError(ValueError):
    """Raised when nominally matched runs use different hard budgets."""

    def __init__(self, report):
        self.report = report
        fields = ", ".join(sorted(report["hard_differences"]))
        super().__init__(f"hard experiment budget mismatch: {fields}")


class BudgetLedger:
    """Accumulate per-run hard resource counts and soft timing measurements."""

    def __init__(self):
        self._values = asdict(BudgetSnapshot())

    def record_training_batch(
        self,
        batch: DataProto,
        *,
        reward_calls: int,
        optimizer_updates: int,
        wall_time_seconds: float,
        gpu_count: int,
    ) -> None:
        _validate_counter("reward_calls", reward_calls)
        _validate_counter("optimizer_updates", optimizer_updates)
        _validate_duration("wall_time_seconds", wall_time_seconds)
        _validate_gpu_count(gpu_count)
        if "sampling_draw_number" not in batch.batch:
            raise KeyError("batch is missing sampling_draw_number")
        if "response_mask" not in batch.batch:
            raise KeyError("batch is missing response_mask")
        draw_number = batch.batch["sampling_draw_number"]
        response_mask = batch.batch["response_mask"]
        if draw_number.ndim != 1 or response_mask.ndim != 2 or response_mask.shape[0] != draw_number.shape[0]:
            raise ValueError("sampling_draw_number and response_mask have incompatible shapes")
        if draw_number.is_floating_point() or draw_number.dtype == bool:
            raise ValueError("sampling_draw_number must be integer encoded")
        if response_mask.shape[0] == 0:
            raise ValueError("training batch must contain at least one completion")

        self._values["sampled_prompts"] += int(draw_number.unique().numel())
        self._values["completions"] += int(response_mask.shape[0])
        self._values["generated_tokens"] += int(response_mask.sum().item())
        self._values["reward_calls"] += reward_calls
        self._values["optimizer_updates"] += optimizer_updates
        self._values["wall_time_seconds"] += float(wall_time_seconds)
        self._values["gpu_seconds"] += float(wall_time_seconds * gpu_count)

    def record_validation(self, *, calls: int, wall_time_seconds: float, gpu_count: int) -> None:
        _validate_counter("validation_calls", calls)
        _validate_duration("wall_time_seconds", wall_time_seconds)
        _validate_gpu_count(gpu_count)
        self._values["validation_calls"] += calls
        self._values["wall_time_seconds"] += float(wall_time_seconds)
        self._values["gpu_seconds"] += float(wall_time_seconds * gpu_count)

    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(**self._values)

    def state_dict(self) -> dict:
        return {"version": 1, "snapshot": asdict(self.snapshot())}

    def load_state_dict(self, state_dict: dict) -> None:
        if state_dict.get("version") != 1:
            raise ValueError("unsupported budget-ledger state version")
        snapshot = BudgetSnapshot(**state_dict.get("snapshot", {}))
        self._values = asdict(snapshot)


def compare_budgets(reference: BudgetSnapshot, candidate: BudgetSnapshot) -> dict:
    """Require equal design budgets and report realized resource deltas.

    Generated token counts are deliberately not a hard design budget: two
    policies can reach EOS at different times despite receiving the same
    prompt, completion, reward-evaluation, and optimizer-step allocations.
    """

    hard_differences = {
        name: getattr(candidate, name) - getattr(reference, name)
        for name in HARD_BUDGET_FIELDS
        if getattr(candidate, name) != getattr(reference, name)
    }
    soft_differences = {name: getattr(candidate, name) - getattr(reference, name) for name in SOFT_BUDGET_FIELDS}
    reported_resource_differences = {
        name: getattr(candidate, name) - getattr(reference, name) for name in REPORTED_RESOURCE_FIELDS
    }
    report = {
        "hard_budget_match": not hard_differences,
        "hard_differences": hard_differences,
        "reported_resource_differences": reported_resource_differences,
        "soft_differences": soft_differences,
        "reference": asdict(reference),
        "candidate": asdict(candidate),
    }
    if hard_differences:
        raise BudgetMismatchError(report)
    return report
