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
"""Target/proposal separation for prompt sampling.

The sampler keeps the scientific target distribution separate from the
distribution used to draw prompts.  Every draw carries the exact probabilities
needed for ordinary, unclipped importance correction downstream.
"""

from __future__ import annotations

from collections.abc import Iterator, Sized
from dataclasses import dataclass

import numpy as np
import torch
from omegaconf import DictConfig

from verl.experimental.dataset.sampler import AbstractCurriculumSampler


@dataclass(frozen=True)
class ProposalDraw:
    """An item index together with its target/proposal sampling metadata."""

    index: int
    stratum: int
    target_probability: float
    proposal_probability: float
    importance_weight: float
    draw_number: int


def normalize_probability(values, *, name: str) -> np.ndarray:
    """Return a validated one-dimensional float64 probability vector."""

    probability = np.asarray(values, dtype=np.float64)
    if probability.ndim != 1 or probability.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(probability)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(probability < 0):
        raise ValueError(f"{name} must be non-negative")
    mass = float(probability.sum(dtype=np.float64))
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError(f"{name} must have positive finite mass")
    return probability / mass


def epsilon_mixture(adaptive, target, epsilon: float) -> np.ndarray:
    """Mix a proposal with the target so every target-supported item is sampled."""

    if not np.isfinite(epsilon) or not 0 < epsilon <= 1:
        raise ValueError("epsilon must be finite and in (0, 1]")
    adaptive_probability = normalize_probability(adaptive, name="adaptive_probability")
    target_probability = normalize_probability(target, name="target_probability")
    if adaptive_probability.shape != target_probability.shape:
        raise ValueError("adaptive and target probabilities must have the same shape")
    proposal = (1.0 - epsilon) * adaptive_probability + epsilon * target_probability
    return normalize_probability(proposal, name="proposal_probability")


def second_moment_optimal_proposal(target, second_moment) -> np.ndarray:
    """Return the classical proposal proportional to ``rho * sqrt(M)``."""

    target_probability = normalize_probability(target, name="target_probability")
    moment = np.asarray(second_moment, dtype=np.float64)
    if moment.ndim != 1 or moment.shape != target_probability.shape:
        raise ValueError("second_moment must be one-dimensional and match target")
    if not np.all(np.isfinite(moment)) or np.any(moment < 0):
        raise ValueError("second_moment must contain finite non-negative values")
    return normalize_probability(target_probability * np.sqrt(moment), name="optimal_proposal")


def powered_second_moment_proposal(target, second_moment, proposal_exponent: float) -> np.ndarray:
    """Sharpen ``rho * sqrt(M)`` without changing the beta-one path."""

    if not np.isfinite(proposal_exponent) or proposal_exponent < 0:
        raise ValueError("proposal_exponent must be finite and non-negative")
    target_probability = normalize_probability(target, name="target_probability")
    moment = np.asarray(second_moment, dtype=np.float64)
    if moment.ndim != 1 or moment.shape != target_probability.shape:
        raise ValueError("second_moment must be one-dimensional and match target")
    if not np.all(np.isfinite(moment)) or np.any(moment < 0):
        raise ValueError("second_moment must contain finite non-negative values")
    if proposal_exponent == 0.0:
        return target_probability
    if proposal_exponent == 1.0:
        return second_moment_optimal_proposal(target_probability, moment)
    return normalize_probability(
        target_probability * np.power(moment, proposal_exponent / 2.0),
        name="powered_proposal",
    )


class TargetProposalSampler(AbstractCurriculumSampler):
    """Sample with replacement from ``q`` while preserving a fixed target ``rho``."""

    def __init__(self, data_source: Sized, data_config: DictConfig):
        self.data_source = data_source
        sampler_config = data_config.sampler
        stratum_key = sampler_config.get("stratum_key")
        if not stratum_key:
            raise ValueError("data.sampler.stratum_key is required")
        if not hasattr(data_source, "get_sampling_strata"):
            raise TypeError("data source must implement get_sampling_strata(stratum_key)")

        raw_strata = np.asarray(data_source.get_sampling_strata(stratum_key))
        if raw_strata.ndim != 1 or len(raw_strata) != len(data_source):
            raise ValueError("sampling strata must be one-dimensional and match the dataset length")
        if len(raw_strata) == 0:
            raise ValueError("cannot sample an empty dataset")
        if not np.issubdtype(raw_strata.dtype, np.integer):
            raise ValueError("sampling strata must be integer encoded")
        self._strata = raw_strata.astype(np.int64, copy=True)
        unique_strata = np.unique(self._strata)
        expected_strata = np.arange(len(unique_strata), dtype=np.int64)
        if not np.array_equal(unique_strata, expected_strata):
            raise ValueError("sampling strata must be contiguous non-negative integers starting at zero")

        self._stratum_count = np.bincount(self._strata, minlength=len(unique_strata)).astype(np.float64)
        target_stratum_weights = sampler_config.get("target_stratum_weights")
        if target_stratum_weights is None:
            target_stratum_probability = self._stratum_count / len(self._strata)
        else:
            target_stratum_probability = normalize_probability(target_stratum_weights, name="target_stratum_weights")
        if len(target_stratum_probability) != len(unique_strata):
            raise ValueError("target_stratum_weights must have one value per stratum")
        self._target_stratum_probability = target_stratum_probability.copy()
        self._target_probability = target_stratum_probability[self._strata] / self._stratum_count[self._strata]

        initial_weights = sampler_config.get("initial_proposal_stratum_weights")
        if initial_weights is None:
            initial_stratum_probability = target_stratum_probability
        else:
            initial_stratum_probability = normalize_probability(
                initial_weights, name="initial_proposal_stratum_weights"
            )
        if len(initial_stratum_probability) != len(unique_strata):
            raise ValueError("initial_proposal_stratum_weights must have one value per stratum")
        adaptive_probability = initial_stratum_probability[self._strata] / self._stratum_count[self._strata]
        self._epsilon = float(sampler_config.get("epsilon", 0.1))
        self._proposal_probability = epsilon_mixture(
            adaptive_probability,
            self._target_probability,
            self._epsilon,
        )

        self._ema_decay = float(sampler_config.get("ema_decay", 0.9))
        if not np.isfinite(self._ema_decay) or not 0 <= self._ema_decay < 1:
            raise ValueError("ema_decay must be finite and in [0, 1)")
        self._min_second_moment = float(sampler_config.get("min_second_moment", 1.0e-8))
        if not np.isfinite(self._min_second_moment) or self._min_second_moment <= 0:
            raise ValueError("min_second_moment must be positive and finite")
        self._proposal_exponent = float(sampler_config.get("proposal_exponent", 1.0))
        if not np.isfinite(self._proposal_exponent) or self._proposal_exponent < 0:
            raise ValueError("proposal_exponent must be finite and non-negative")
        self._second_moment_estimate = np.ones(len(unique_strata), dtype=np.float64)
        self._adaptation_enabled = bool(sampler_config.get("adaptation_enabled", True))

        self._generator = torch.Generator(device="cpu")
        seed = data_config.get("seed")
        self._generator.manual_seed(0 if seed is None else int(seed))
        self._draw_count = 0

    @property
    def target_probability(self) -> np.ndarray:
        return self._target_probability.copy()

    @property
    def proposal_probability(self) -> np.ndarray:
        return self._proposal_probability.copy()

    @property
    def second_moment_estimate(self) -> np.ndarray:
        return self._second_moment_estimate.copy()

    def __iter__(self) -> Iterator[ProposalDraw]:
        for _ in range(len(self)):
            index = int(
                torch.multinomial(
                    torch.from_numpy(self._proposal_probability),
                    num_samples=1,
                    replacement=True,
                    generator=self._generator,
                ).item()
            )
            target_probability = float(self._target_probability[index])
            proposal_probability = float(self._proposal_probability[index])
            draw = ProposalDraw(
                index=int(index),
                stratum=int(self._strata[index]),
                target_probability=target_probability,
                proposal_probability=proposal_probability,
                importance_weight=target_probability / proposal_probability,
                draw_number=self._draw_count,
            )
            self._draw_count += 1
            yield draw

    def __len__(self) -> int:
        return len(self.data_source)

    def update(self, batch) -> dict[str, float]:
        """Update later proposals from an already-consumed batch.

        Repeated GRPO responses are averaged by original prompt before prompts
        are averaged within a stratum.  This keeps ``q_t`` predictable: callers
        invoke ``update`` only after the current draw has been fully consumed.
        """

        required_keys = (
            "sampling_index",
            "sampling_stratum",
            "sampling_second_moment_proxy",
        )
        for key in required_keys:
            if key not in batch.batch:
                raise KeyError(f"batch is missing required sampling metadata {key!r}")

        indices = batch.batch["sampling_index"].detach().cpu().numpy()
        strata = batch.batch["sampling_stratum"].detach().cpu().numpy()
        proxy = batch.batch["sampling_second_moment_proxy"].detach().cpu().numpy().astype(np.float64)
        if indices.ndim != 1 or strata.ndim != 1 or proxy.ndim != 1:
            raise ValueError("sampling update metadata must be one-dimensional")
        if not (len(indices) == len(strata) == len(proxy)) or len(indices) == 0:
            raise ValueError("sampling update metadata must have matching non-zero lengths")
        if not np.all(np.isfinite(proxy)):
            raise ValueError("sampling_second_moment_proxy must be finite")
        if np.any(proxy < 0):
            raise ValueError("sampling_second_moment_proxy must be non-negative")
        if not np.issubdtype(indices.dtype, np.integer) or not np.issubdtype(strata.dtype, np.integer):
            raise ValueError("sampling indices and strata must be integer tensors")

        if "sampling_draw_number" in batch.batch:
            draw_numbers = batch.batch["sampling_draw_number"].detach().cpu().numpy()
            if draw_numbers.ndim != 1 or len(draw_numbers) != len(indices):
                raise ValueError("sampling_draw_number must be one-dimensional and match sampling metadata")
            if not np.issubdtype(draw_numbers.dtype, np.integer) or np.any(draw_numbers < 0):
                raise ValueError("sampling_draw_number must contain non-negative integers")
        else:
            draw_numbers = indices

        pre_proposal_mass = np.bincount(
            self._strata,
            weights=self._proposal_probability,
            minlength=len(self._second_moment_estimate),
        )
        pre_second_moment = self._second_moment_estimate.copy()

        draw_observations: list[tuple[int, float]] = []
        for draw_number in np.unique(draw_numbers):
            draw_mask = draw_numbers == draw_number
            observed_indices = np.unique(indices[draw_mask])
            observed_strata = np.unique(strata[draw_mask])
            if len(observed_indices) != 1 or len(observed_strata) != 1:
                raise ValueError(f"sampling index or stratum is inconsistent within draw {int(draw_number)}")
            index_int = int(observed_indices[0])
            stratum_int = int(observed_strata[0])
            if not 0 <= index_int < len(self._strata) or int(self._strata[index_int]) != stratum_int:
                raise ValueError(f"sampling stratum is inconsistent for dataset index {index_int}")
            draw_observations.append((stratum_int, float(np.mean(proxy[draw_mask], dtype=np.float64))))

        item_moments: dict[int, float] = {}
        item_strata: dict[int, int] = {}
        for index in np.unique(indices):
            index_int = int(index)
            if not 0 <= index_int < len(self._strata):
                raise ValueError(f"sampling index {index_int} is outside the dataset")
            item_mask = indices == index
            observed_strata = np.unique(strata[item_mask])
            expected_stratum = int(self._strata[index_int])
            if len(observed_strata) != 1 or int(observed_strata[0]) != expected_stratum:
                raise ValueError(f"sampling stratum is inconsistent for dataset index {index_int}")
            item_strata[index_int] = expected_stratum
            item_moments[index_int] = float(np.mean(proxy[item_mask], dtype=np.float64))

        if self._adaptation_enabled:
            for stratum in range(len(self._second_moment_estimate)):
                observed = [moment for index, moment in item_moments.items() if item_strata[index] == stratum]
                if not observed:
                    continue
                observed_moment = max(float(np.mean(observed, dtype=np.float64)), self._min_second_moment)
                self._second_moment_estimate[stratum] = (
                    self._ema_decay * self._second_moment_estimate[stratum] + (1.0 - self._ema_decay) * observed_moment
                )

            adaptive_stratum_probability = powered_second_moment_proposal(
                self._target_stratum_probability,
                self._second_moment_estimate,
                self._proposal_exponent,
            )
            adaptive_item_probability = adaptive_stratum_probability[self._strata] / self._stratum_count[self._strata]
            self._proposal_probability = epsilon_mixture(
                adaptive_item_probability,
                self._target_probability,
                self._epsilon,
            )

        next_proposal_mass = np.bincount(
            self._strata,
            weights=self._proposal_probability,
            minlength=len(self._second_moment_estimate),
        )
        metrics = {"sampling/proposal_exponent": self._proposal_exponent}
        for stratum in range(len(self._second_moment_estimate)):
            observed_proxy = [value for observed_stratum, value in draw_observations if observed_stratum == stratum]
            prefix = f"sampling/stratum_{stratum}"
            metrics.update(
                {
                    f"{prefix}/target_mass": float(self._target_stratum_probability[stratum]),
                    f"{prefix}/proposal_mass": float(pre_proposal_mass[stratum]),
                    f"{prefix}/second_moment": float(pre_second_moment[stratum]),
                    f"{prefix}/realized_prompt_count": float(len(observed_proxy)),
                    f"{prefix}/raw_proxy_mean": (
                        float(np.mean(observed_proxy, dtype=np.float64)) if observed_proxy else 0.0
                    ),
                    f"{prefix}/next_proposal_mass": float(next_proposal_mass[stratum]),
                    f"{prefix}/next_second_moment": float(self._second_moment_estimate[stratum]),
                }
            )
        return metrics

    def state_dict(self) -> dict:
        """Serialize exact proposal, moment, RNG, and draw state."""

        return {
            "version": 2,
            "strata": self._strata.copy(),
            "target_probability": self._target_probability.copy(),
            "proposal_probability": self._proposal_probability.copy(),
            "second_moment_estimate": self._second_moment_estimate.copy(),
            "generator_state": self._generator.get_state().clone(),
            "draw_count": self._draw_count,
            "adaptation_enabled": self._adaptation_enabled,
            "proposal_exponent": self._proposal_exponent,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore state only when it belongs to the same target population."""

        version = state_dict.get("version")
        if version not in (1, 2):
            raise ValueError("unsupported target-proposal sampler state version")
        saved_exponent = 1.0 if version == 1 else state_dict.get("proposal_exponent")
        if not isinstance(saved_exponent, (int, float)) or not np.isfinite(saved_exponent):
            raise ValueError("sampler state proposal exponent must be finite")
        if float(saved_exponent) != self._proposal_exponent:
            raise ValueError("sampler state proposal exponent does not match the current configuration")
        if bool(state_dict.get("adaptation_enabled", True)) != self._adaptation_enabled:
            raise ValueError("sampler state adaptation mode does not match the current configuration")
        saved_strata = np.asarray(state_dict.get("strata"), dtype=np.int64)
        if not np.array_equal(saved_strata, self._strata):
            raise ValueError("sampler state strata do not match the current dataset")
        saved_target = np.asarray(state_dict.get("target_probability"), dtype=np.float64)
        if saved_target.shape != self._target_probability.shape or not np.allclose(
            saved_target, self._target_probability, rtol=0, atol=1e-15
        ):
            raise ValueError("sampler state target distribution does not match")

        saved_proposal = np.asarray(state_dict.get("proposal_probability"), dtype=np.float64)
        if saved_proposal.ndim != 1 or not np.all(np.isfinite(saved_proposal)) or np.any(saved_proposal < 0):
            raise ValueError("sampler state proposal must be a finite non-negative vector")
        if not np.isclose(saved_proposal.sum(dtype=np.float64), 1.0, rtol=0, atol=1e-12):
            raise ValueError("sampler state proposal must sum to one")
        if saved_proposal.shape != self._proposal_probability.shape:
            raise ValueError("sampler state proposal shape does not match")
        if np.any(saved_proposal[self._target_probability > 0] <= 0):
            raise ValueError("sampler state proposal does not support the target")

        saved_moment = np.asarray(state_dict.get("second_moment_estimate"), dtype=np.float64)
        if saved_moment.shape != self._second_moment_estimate.shape:
            raise ValueError("sampler state second-moment shape does not match")
        if not np.all(np.isfinite(saved_moment)) or np.any(saved_moment <= 0):
            raise ValueError("sampler state second moments must be positive and finite")
        draw_count = state_dict.get("draw_count")
        if not isinstance(draw_count, int) or draw_count < 0:
            raise ValueError("sampler state draw_count must be a non-negative integer")
        generator_state = state_dict.get("generator_state")
        if not isinstance(generator_state, torch.Tensor):
            raise ValueError("sampler state generator_state must be a torch tensor")

        self._proposal_probability = saved_proposal.copy()
        self._second_moment_estimate = saved_moment.copy()
        self._draw_count = draw_count
        self._generator.set_state(generator_state.clone())
