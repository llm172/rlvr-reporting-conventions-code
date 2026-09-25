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
"""RLHF dataset adapter for metadata-carrying target/proposal draws."""

from __future__ import annotations

import operator

import numpy as np
import torch

from recipe.curriculum_identifiability.sampler import ProposalDraw
from verl.utils.dataset.rl_dataset import RLHFDataset


class IdentifiableRLHFDataset(RLHFDataset):
    """Preserve exact prompt-sampling probabilities through verl collation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        stratum_key = self.config.get("sampler", {}).get("stratum_key")
        if not stratum_key:
            raise ValueError("data.sampler.stratum_key is required for IdentifiableRLHFDataset")
        self._sampling_strata_cache = self._read_sampling_strata(stratum_key)

    def _read_sampling_strata(self, key: str) -> np.ndarray:
        column_names = getattr(self.dataframe, "column_names", None)
        if column_names is not None and key not in column_names:
            raise KeyError(f"sampling stratum column {key!r} is missing from the dataset")
        try:
            strata = np.asarray(self.dataframe[key])
        except (KeyError, TypeError) as error:
            raise KeyError(f"sampling stratum column {key!r} is missing from the dataset") from error
        if strata.ndim != 1 or len(strata) != len(self.dataframe):
            raise ValueError("sampling stratum column must be one-dimensional and match the dataset length")
        return strata.copy()

    def get_sampling_strata(self, key: str) -> np.ndarray:
        """Return a defensive copy of the integer-encoded sampling strata."""

        cached = getattr(self, "_sampling_strata_cache", None)
        configured_key = self.config.get("sampler", {}).get("stratum_key")
        if cached is not None and key == configured_key:
            return cached.copy()
        return self._read_sampling_strata(key)

    def __getitem__(self, item):
        if isinstance(item, ProposalDraw):
            draw = item
        else:
            index = operator.index(item)
            if not 0 <= index < len(self.dataframe):
                raise IndexError(index)
            stratum_key = self.config.get("sampler", {}).get("stratum_key")
            if not stratum_key:
                raise ValueError("data.sampler.stratum_key is required")
            stratum = int(self.get_sampling_strata(stratum_key)[index])
            iid_probability = 1.0 / len(self.dataframe)
            draw = ProposalDraw(
                index=index,
                stratum=stratum,
                target_probability=iid_probability,
                proposal_probability=iid_probability,
                importance_weight=1.0,
                draw_number=-1,
            )

        if not 0 <= draw.index < len(self.dataframe):
            raise IndexError(draw.index)
        if not np.isfinite(draw.target_probability) or draw.target_probability <= 0:
            raise ValueError("draw target_probability must be positive and finite")
        if not np.isfinite(draw.proposal_probability) or draw.proposal_probability <= 0:
            raise ValueError("draw proposal_probability must be positive and finite")
        expected_weight = draw.target_probability / draw.proposal_probability
        if not np.isclose(draw.importance_weight, expected_weight, rtol=1e-12, atol=0.0):
            raise ValueError("draw importance_weight must equal target_probability / proposal_probability")

        row = super().__getitem__(draw.index)
        row.update(
            {
                "sampling_index": torch.tensor(draw.index, dtype=torch.int64),
                "sampling_stratum": torch.tensor(draw.stratum, dtype=torch.int64),
                "sampling_target_probability": torch.tensor(draw.target_probability, dtype=torch.float64),
                "sampling_proposal_probability": torch.tensor(draw.proposal_probability, dtype=torch.float64),
                "sampling_importance_weight": torch.tensor(draw.importance_weight, dtype=torch.float64),
                "sampling_draw_number": torch.tensor(draw.draw_number, dtype=torch.int64),
            }
        )
        return row
