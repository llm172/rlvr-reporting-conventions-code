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
"""Freeze baseline solve-rate strata before any RLVR optimization.

The GPU scoring command is deliberately separate from assembly.  Independent
single-GPU shards can be generated in parallel, while the CPU-only assembler
checks exact coverage and writes an immutable stratified parquet plus protocol
metadata.  Difficulty bin 0 always represents the lowest frozen solve rate.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def row_ids_for_shard(total_rows: int, shard_index: int, num_shards: int) -> np.ndarray:
    """Return a balanced, deterministic modulo partition of row ids."""
    if total_rows < 0:
        raise ValueError("total_rows must be non-negative")
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    return np.arange(shard_index, total_rows, num_shards, dtype=np.int64)


def _stable_uint64(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def balanced_solve_rate_bins(
    row_ids: Sequence[int] | np.ndarray,
    solve_counts: Sequence[int] | np.ndarray,
    *,
    rollouts: int,
    n_bins: int,
) -> np.ndarray:
    """Assign balanced quantile bins with deterministic hash tie-breaking.

    Sorting is first by the discrete frozen solve count and only then by a hash
    of the immutable source row id.  Thus lower observed solve rates can never
    receive an easier bin than a strictly higher observed solve rate, while
    ties do not inherit a possibly structured dataset order.
    """
    ids = np.asarray(row_ids, dtype=np.int64)
    counts = np.asarray(solve_counts, dtype=np.int64)
    if ids.ndim != 1 or counts.ndim != 1 or ids.shape != counts.shape:
        raise ValueError("row_ids and solve_counts must be same-length vectors")
    if len(ids) < n_bins or n_bins < 2:
        raise ValueError("n_bins must be at least two and no larger than the pool")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("row_ids must be unique")
    if rollouts <= 0 or np.any(counts < 0) or np.any(counts > rollouts):
        raise ValueError("solve_counts must be integers in [0, rollouts]")

    tie_break = np.fromiter(
        (_stable_uint64(f"difficulty-tie:{int(row_id)}") for row_id in ids),
        dtype=np.uint64,
        count=len(ids),
    )
    order = np.lexsort((tie_break, counts))
    ranked_bins = np.minimum((np.arange(len(ids), dtype=np.int64) * n_bins) // len(ids), n_bins - 1)
    bins = np.empty(len(ids), dtype=np.int64)
    bins[order] = ranked_bins
    return bins


def solve_count_bins(
    solve_counts: Sequence[int] | np.ndarray,
    *,
    cutpoints: Sequence[int] | np.ndarray,
    rollouts: int,
) -> np.ndarray:
    """Assign semantic bins using inclusive solve-count upper cutpoints.

    For example, cutpoints ``[0, 1]`` create the preregistered categories
    ``0 solves``, ``1 solve``, and ``2 or more solves``.
    """
    counts = np.asarray(solve_counts)
    boundaries = np.asarray(cutpoints)
    if counts.ndim != 1 or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("solve_counts must be an integer vector")
    if boundaries.ndim != 1 or len(boundaries) == 0:
        raise ValueError("cutpoints must be a non-empty vector")
    if not np.issubdtype(boundaries.dtype, np.integer):
        raise ValueError("cutpoints must be integer encoded")
    counts = counts.astype(np.int64, copy=False)
    boundaries = boundaries.astype(np.int64, copy=False)
    if rollouts <= 0 or np.any(counts < 0) or np.any(counts > rollouts):
        raise ValueError("solve_counts must lie in [0, rollouts]")
    if np.any(np.diff(boundaries) <= 0):
        raise ValueError("cutpoints must be strictly increasing")
    if boundaries[0] < 0 or boundaries[-1] >= rollouts:
        raise ValueError("cutpoints must lie in [0, rollouts)")
    return np.searchsorted(boundaries, counts, side="left").astype(np.int64)


def merge_shard_scores(shards: Sequence[pd.DataFrame], *, total_rows: int, rollouts: int) -> pd.DataFrame:
    """Validate and merge score shards with exact-once source coverage."""
    if not shards:
        raise ValueError("at least one score shard is required")
    required = {"row_id", "solve_count", "rollouts"}
    for shard in shards:
        missing = required.difference(shard.columns)
        if missing:
            raise ValueError(f"score shard is missing columns: {sorted(missing)}")
    merged = pd.concat(shards, axis=0, ignore_index=True)
    row_ids = merged["row_id"].to_numpy(dtype=np.int64)
    expected = np.arange(total_rows, dtype=np.int64)
    if len(row_ids) != total_rows or not np.array_equal(np.sort(row_ids), expected):
        raise ValueError("score shards must cover every source row exactly once")
    shard_rollouts = merged["rollouts"].to_numpy(dtype=np.int64)
    if not np.all(shard_rollouts == rollouts):
        raise ValueError("all score shards must use the declared rollouts")
    solve_counts = merged["solve_count"].to_numpy(dtype=np.int64)
    if np.any(solve_counts < 0) or np.any(solve_counts > rollouts):
        raise ValueError("solve_count must lie in [0, rollouts]")
    return merged.sort_values("row_id", kind="stable").reset_index(drop=True)


_SHARED_METADATA_KEYS = (
    "schema_version",
    "protocol",
    "source_sha256",
    "model_id",
    "model_revision",
    "num_shards",
    "rollouts",
    "temperature",
    "top_p",
    "max_tokens",
    "base_seed",
    "seed_rule",
    "reward",
    "vllm_version",
    "transformers_version",
)


_OPTIONAL_SHARED_METADATA_KEYS = ("max_model_len", "chunk_size")


def validate_shard_metadata(metadata: Sequence[dict], *, expected_num_shards: int) -> dict:
    """Require shards to be one complete, identical frozen protocol."""
    if expected_num_shards <= 0 or len(metadata) != expected_num_shards:
        raise ValueError("metadata count must equal expected_num_shards")
    reference = metadata[0]
    for key in _SHARED_METADATA_KEYS:
        if key not in reference:
            raise ValueError(f"score metadata is missing {key}")
        for shard in metadata[1:]:
            if key not in shard or shard[key] != reference[key]:
                raise ValueError(f"score shard metadata disagree on {key}")
    optional_shared: dict[str, object] = {}
    for key in _OPTIONAL_SHARED_METADATA_KEYS:
        presence = [key in shard for shard in metadata]
        if any(presence) and not all(presence):
            raise ValueError(f"score shard metadata partially specify optional {key}")
        if all(presence):
            for shard in metadata[1:]:
                if shard[key] != reference[key]:
                    raise ValueError(f"score shard metadata disagree on {key}")
            optional_shared[key] = reference[key]
    if int(reference["num_shards"]) != expected_num_shards:
        raise ValueError("num_shards does not match the number being assembled")
    try:
        shard_indices = [int(shard["shard_index"]) for shard in metadata]
    except KeyError as error:
        raise ValueError("score metadata is missing shard_index") from error
    if sorted(shard_indices) != list(range(expected_num_shards)):
        raise ValueError("shard_index values must cover [0, num_shards) exactly once")
    return {key: reference[key] for key in _SHARED_METADATA_KEYS} | optional_shared


def select_balanced_audit_pool(
    frame: pd.DataFrame,
    *,
    per_bin: int,
    seed: int,
    exclude_from_training: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select a reproducible, stratum-balanced frozen audit pool."""
    if "difficulty_bin" not in frame:
        raise ValueError("frame must contain difficulty_bin")
    if per_bin <= 0:
        raise ValueError("per_bin must be positive")
    selected: list[object] = []
    for difficulty_bin in sorted(frame["difficulty_bin"].unique().tolist()):
        candidates = frame.index[frame["difficulty_bin"] == difficulty_bin].tolist()
        if len(candidates) < per_bin:
            raise ValueError(f"difficulty bin {difficulty_bin} has fewer than {per_bin} rows")
        candidates.sort(key=lambda index: _stable_uint64(f"audit:{seed}:{index!r}"))
        selected.extend(candidates[:per_bin])
    selected_set = set(selected)
    audit = frame.loc[selected].copy()
    audit = audit.sort_values(["difficulty_bin"], kind="stable")
    if exclude_from_training:
        training = frame.loc[[index for index in frame.index if index not in selected_set]].copy()
    else:
        training = frame.copy()
    return training, audit


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_revision(model_path: Path) -> str:
    """Read the immutable Hugging Face commit recorded by snapshot_download."""
    metadata_root = model_path / ".cache" / "huggingface" / "download"
    revision_files = sorted(metadata_root.glob("*.metadata"))
    if not revision_files:
        raise FileNotFoundError(f"no Hugging Face revision metadata found under {metadata_root}")
    revisions = {
        path.read_text(encoding="utf-8").splitlines()[0].strip()
        for path in revision_files
        if path.read_text(encoding="utf-8").splitlines()
    }
    if len(revisions) != 1:
        raise ValueError(f"model files do not share one Hugging Face revision: {sorted(revisions)}")
    return revisions.pop()


def _atomic_parquet(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(output)


def _atomic_json(payload: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)


def _prompt_text(tokenizer, prompt_value: object) -> str:
    if isinstance(prompt_value, np.ndarray):
        messages = prompt_value.tolist()
    else:
        messages = list(prompt_value)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _score_shard(args: argparse.Namespace) -> None:
    # GPU libraries are lazy imports so all protocol math remains CPU-testable.
    import transformers
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from verl.utils.reward_score.gsm8k import compute_score

    source_path = Path(args.input)
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output_path}; pass --overwrite")
    source = pd.read_parquet(source_path)
    row_ids = row_ids_for_shard(len(source), args.shard_index, args.num_shards)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    engine = LLM(
        model=args.model,
        tokenizer=args.model,
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.seed,
    )

    records: list[dict] = []
    for chunk_start in range(0, len(row_ids), args.chunk_size):
        chunk_ids = row_ids[chunk_start : chunk_start + args.chunk_size]
        rows = source.iloc[chunk_ids]
        prompts = [_prompt_text(tokenizer, prompt) for prompt in rows["prompt"]]
        sampling_params = [
            SamplingParams(
                n=args.rollouts,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=-1,
                max_tokens=args.max_tokens,
                seed=(args.seed * 1_000_003 + int(row_id)) % (2**31 - 1),
            )
            for row_id in chunk_ids
        ]
        outputs = engine.generate(prompts, sampling_params, use_tqdm=True)
        if len(outputs) != len(chunk_ids):
            raise RuntimeError("vLLM returned a different number of requests than submitted")
        for row_id, (_, row), request_output in zip(chunk_ids.tolist(), rows.iterrows(), outputs, strict=True):
            if len(request_output.outputs) != args.rollouts:
                raise RuntimeError(f"row {row_id} returned the wrong rollout count")
            ground_truth = str(row["reward_model"]["ground_truth"])
            scores = [float(compute_score(sample.text, ground_truth)) for sample in request_output.outputs]
            token_lengths = [len(sample.token_ids) for sample in request_output.outputs]
            records.append(
                {
                    "row_id": int(row_id),
                    "solve_count": int(sum(scores)),
                    "rollouts": int(args.rollouts),
                    "mean_response_tokens": float(np.mean(token_lengths)),
                    "truncated_count": int(sum(sample.finish_reason == "length" for sample in request_output.outputs)),
                }
            )

    scores = pd.DataFrame.from_records(records)
    _atomic_parquet(scores, output_path)
    metadata = {
        "schema_version": 1,
        "protocol": "frozen-baseline-solve-rate",
        "source": str(source_path),
        "source_sha256": _sha256_file(source_path),
        "model_id": args.model_id,
        "model_path": str(Path(args.model)),
        "model_revision": _model_revision(Path(args.model)),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "row_count": len(scores),
        "rollouts": args.rollouts,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "chunk_size": args.chunk_size,
        "base_seed": args.seed,
        "seed_rule": "(base_seed * 1000003 + source_row_id) mod (2^31 - 1)",
        "reward": "verl GSM8K strict exact match",
        "vllm_version": vllm.__version__,
        "transformers_version": transformers.__version__,
    }
    _atomic_json(metadata, output_path.with_suffix(output_path.suffix + ".metadata.json"))


def _expand_score_paths(patterns: Sequence[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        matches = glob.glob(pattern)
        if not matches and Path(pattern).exists():
            matches = [pattern]
        paths.update(Path(match) for match in matches)
    if not paths:
        raise FileNotFoundError("no score shards matched")
    return sorted(paths)


def _assemble(args: argparse.Namespace) -> None:
    source_path = Path(args.input)
    source = pd.read_parquet(source_path)
    score_paths = _expand_score_paths(args.score_shards)
    score_metadata = []
    for path in score_paths:
        metadata_path = path.with_suffix(path.suffix + ".metadata.json")
        if not metadata_path.exists():
            raise FileNotFoundError(f"missing score-shard metadata {metadata_path}")
        score_metadata.append(json.loads(metadata_path.read_text(encoding="utf-8")))
    shared_scoring_protocol = validate_shard_metadata(score_metadata, expected_num_shards=len(score_paths))
    score_frames = [pd.read_parquet(path) for path in score_paths]
    merged = merge_shard_scores(score_frames, total_rows=len(source), rollouts=args.rollouts)
    if args.binning_mode == "balanced-quantile":
        bins = balanced_solve_rate_bins(
            merged["row_id"].to_numpy(),
            merged["solve_count"].to_numpy(),
            rollouts=args.rollouts,
            n_bins=args.n_bins,
        )
        n_bins = args.n_bins
        protocol = "balanced quantiles of preregistered frozen-policy solve rate"
        tie_break = "BLAKE2b-64 of immutable source row id; independent of input ordering"
    else:
        bins = solve_count_bins(
            merged["solve_count"].to_numpy(),
            cutpoints=args.solve_count_cutpoints,
            rollouts=args.rollouts,
        )
        n_bins = len(args.solve_count_cutpoints) + 1
        protocol = "preregistered frozen-policy solve-count categories"
        tie_break = None
    enriched = source.copy()
    enriched["difficulty_bin"] = bins
    enriched["frozen_baseline_solve_count"] = merged["solve_count"].to_numpy(dtype=np.int64)
    enriched["frozen_baseline_solve_rate"] = merged["solve_count"].to_numpy(dtype=np.float64) / args.rollouts

    audit = enriched.iloc[0:0].copy()
    training = enriched
    if args.audit_per_bin:
        training, audit = select_balanced_audit_pool(
            enriched,
            per_bin=args.audit_per_bin,
            seed=args.audit_seed,
            exclude_from_training=args.exclude_audit_from_training,
        )
        if args.audit_output is None:
            raise ValueError("--audit-output is required when --audit-per-bin is positive")
        _atomic_parquet(audit.reset_index(drop=True), Path(args.audit_output))
    elif args.exclude_audit_from_training:
        raise ValueError("cannot exclude an audit pool when --audit-per-bin is zero")

    output_path = Path(args.output)
    _atomic_parquet(training.reset_index(drop=True), output_path)
    bin_summary = []
    for difficulty_bin in range(n_bins):
        selected = enriched[enriched["difficulty_bin"] == difficulty_bin]
        bin_summary.append(
            {
                "difficulty_bin": difficulty_bin,
                "meaning": "hardest"
                if difficulty_bin == 0
                else ("easiest" if difficulty_bin == n_bins - 1 else "intermediate"),
                "full_count": len(selected),
                "training_count": int((training["difficulty_bin"] == difficulty_bin).sum()),
                "audit_count": int((audit["difficulty_bin"] == difficulty_bin).sum()),
                "mean_frozen_solve_rate": float(selected["frozen_baseline_solve_rate"].mean()),
                "min_frozen_solve_rate": float(selected["frozen_baseline_solve_rate"].min()),
                "max_frozen_solve_rate": float(selected["frozen_baseline_solve_rate"].max()),
            }
        )
    metadata = {
        "schema_version": 1,
        "protocol": protocol,
        "binning_mode": args.binning_mode,
        "solve_count_cutpoints": (list(args.solve_count_cutpoints) if args.binning_mode == "solve-count" else None),
        "difficulty_order": "0=hardest; larger integers=easier",
        "tie_break": tie_break,
        "source": str(source_path),
        "source_sha256": _sha256_file(source_path),
        "score_shards": [{"path": str(path), "sha256": _sha256_file(path)} for path in score_paths],
        "frozen_scoring_protocol": shared_scoring_protocol,
        "rollouts_per_prompt": args.rollouts,
        "n_bins": n_bins,
        "full_rows": len(enriched),
        "training_rows": len(training),
        "audit_rows": len(audit),
        "audit_seed": args.audit_seed if args.audit_per_bin else None,
        "audit_per_bin": args.audit_per_bin,
        "audit_excluded_from_training": bool(args.exclude_audit_from_training),
        "bin_summary": bin_summary,
        "scientific_scope": (
            "The bins are frozen before RLVR and use no outcome from the current adaptive draw. "
            "Solve rate is a noisy preregistered proxy, not latent ground-truth difficulty."
        ),
    }
    metadata_path = (
        Path(args.metadata_output)
        if args.metadata_output
        else output_path.with_suffix(output_path.suffix + ".metadata.json")
    )
    _atomic_json(metadata, metadata_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    score = subparsers.add_parser("score-shard", help="score one deterministic GPU shard")
    score.add_argument("--input", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--model", required=True)
    score.add_argument("--model-id", required=True)
    score.add_argument("--shard-index", type=int, required=True)
    score.add_argument("--num-shards", type=int, required=True)
    score.add_argument("--rollouts", type=int, default=8)
    score.add_argument("--temperature", type=float, default=0.7)
    score.add_argument("--top-p", type=float, default=0.95)
    score.add_argument("--max-tokens", type=int, default=512)
    score.add_argument("--max-model-len", type=int, default=1536)
    score.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    score.add_argument("--chunk-size", type=int, default=256)
    score.add_argument("--seed", type=int, default=20260716)
    score.add_argument("--overwrite", action="store_true")
    score.set_defaults(func=_score_shard)

    assemble = subparsers.add_parser("assemble", help="validate shards and freeze bins")
    assemble.add_argument("--input", required=True)
    assemble.add_argument("--score-shards", nargs="+", required=True)
    assemble.add_argument("--output", required=True)
    assemble.add_argument("--metadata-output")
    assemble.add_argument("--rollouts", type=int, default=8)
    assemble.add_argument("--n-bins", type=int, default=3)
    assemble.add_argument(
        "--binning-mode",
        choices=("balanced-quantile", "solve-count"),
        default="balanced-quantile",
    )
    assemble.add_argument("--solve-count-cutpoints", nargs="+", type=int, default=[0, 1])
    assemble.add_argument("--audit-output")
    assemble.add_argument("--audit-per-bin", type=int, default=0)
    assemble.add_argument("--audit-seed", type=int, default=20260717)
    assemble.add_argument("--exclude-audit-from-training", action="store_true")
    assemble.set_defaults(func=_assemble)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
