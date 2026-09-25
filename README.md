# RLVR reporting-convention experiments

This package contains the training recipe, readers, data transforms, and analysis scripts used for the paper's GSM8K crossed comparisons and selected controls. It excludes model weights, generated responses, human annotations, and experiment logs.

## Environment

Use Python 3.10+ on Linux with CUDA. Install the public `verl` v0.6.1 release, apply `patches/verl-runtime.patch` with `git apply`, then install the repository requirements. Add this directory to `PYTHONPATH` so `recipe.curriculum_identifiability` can be imported. The patch contains the prompt sampler and its trainer/config integration. The existing Apache license is retained in `LICENSE`.

The recorded evaluation environment used PyTorch 2.8.0+cu128, vLLM 0.11.0, Transformers 4.57.6, NumPy 1.26.4, and Math-Verify 0.9.0. The CUDA-specific PyTorch wheel depends on the machine.

```bash
git clone https://github.com/volcengine/verl.git
cd verl && git checkout v0.6.1
git apply /path/to/this-package/patches/verl-runtime.patch
pip install -e .
cd /path/to/this-package
pip install -r requirements.txt
export PYTHONPATH="$PWD:$PYTHONPATH"
```

## Data and models

Download the public GSM8K split and Qwen2.5-7B-Instruct model separately. The paired training inputs are Parquet files named `train_stratified.parquet` and `validation_frozen.parquet` in separate hash and boxed directories. The latter is the **official 1,319-item GSM8K test split**, which the training recipe reads as a frozen validation metric. It is not a held-out training-validation split. The frozen difficulty bins in the training Parquet were made from baseline rollout scores. `recipe/curriculum_identifiability/frozen_difficulty.py` contains the `score-shard` and `assemble` commands; use `--help` on each subcommand for its input contract. Recreating the exact recorded bins from raw GSM8K requires the frozen baseline score shards and source Parquet, which are not included here. Thus the training commands below start from the paired frozen Parquets, not directly from the official download.

For a matched boxed prompt copy of an existing hash dataset:

```bash
HASH_DATA_DIR=/data/gsm8k_hash BOXED_DATA_DIR=/data/gsm8k_boxed \
  python scripts/build_boxed_gsm8k_data.py
```

The data builder keeps row order, reward ground truth, and frozen strata fixed while changing the requested final marker. Check dataset licensing before redistributing generated Parquets.

## Crossed RLVR training

The commands below use the fixed 100-step GRPO recipe and keep the final model checkpoint. `GPU_COUNT=4` matches the recorded Qwen2.5-7B runs. Set all paths to local files; they are deliberately absent from the package.

```bash
export MODEL_PATH=/models/Qwen2.5-7B-Instruct
export HASH_DATA_DIR=/data/gsm8k_hash
export BOXED_DATA_DIR=/data/gsm8k_boxed
export GPU_COUNT=4
for seed in 83 84 85 86 87; do
  bash scripts/train_crossed.sh hash "$seed"
  bash scripts/train_crossed.sh boxed "$seed"
done
```

The launcher exposes `MODEL_PATH`, data paths, `RUN_ROOT`, and `GPU_COUNT`. Changing model, data, or GPU count changes the recorded configuration. `scripts/gsm8k_boxed_reward.py` supplies the boxed-arm strict reward. `scripts/reward_lenient_gsm8k.py` and `scripts/passk_score.py` implement the answer-only reward control and operational readers.

## Evaluation and controls

`evaluation/gsm8k/infer_worker.py` reads a JSON manifest with a model path, GPU ID, and frozen tasks; each task supplies its item file, SHA256, sampling parameters, and expected item count. It writes one response record per item and a completion receipt. `evaluation/gsm8k/score_all.py` scores those saved responses. Example invocation:

```bash
python evaluation/gsm8k/infer_worker.py --manifest /data/eval_manifest.json
touch "$(dirname /data/eval_manifest.json)/INFERENCE_FINISHED.json"
python evaluation/gsm8k/score_all.py --root /data --workers 8
```

After scoring finishes, run `python evaluation/gsm8k/freeze_cells.py --root /data --state MODEL_STATE` for each state. This computes the boxed-request strict scores and freezes the cell hashes required by the five-seed aggregator; use the original audit scorer via `--scorer` when combining with its archived cells.

The worker manifest has keys `state`, `model`, `model_metadata`, `gpu`, and `tasks`. Each task has `name`, `items`, `items_sha256`, `temperature`, `top_p`, `n`, `max_tokens`, `max_model_len`, `seed`, and `batch_items`. The recorded GSM8K tasks used temperature 0.6, top-p 0.95, one response, 640 maximum output tokens, 4,096 maximum context tokens, seed 20260908, and batches of 16 items. The item JSONL rows contain `id`, `prompt`, and `ground_truth`. The scoring queue stops after the `INFERENCE_FINISHED.json` marker is present. Prepare one manifest per model/checkpoint; keep the same frozen items and request prompts across trained states.

`evaluation/gsm8k/build_inputs.py gsm8k --validation-parquet /data/validation_frozen.parquet --tokenizer /models/INITIAL_MODEL --convention hash --out /data/hash_items.jsonl` renders the matched 1,319-item test inputs; use the paired boxed Parquet with `--convention boxed`. For the arithmetic probe, run `build_inputs.py arithmetic --tokenizer /models/INITIAL_MODEL --out-dir /data/arithmetic`. It generates the fixed 32 development and 500 test expressions, then renders each with the model's initial tokenizer. These commands require the original frozen Parquets and tokenizer. New files are a reproduction attempt; compare their SHA256 hashes with the recorded inputs before claiming exact historical equivalence.

The five-seed crossed aggregation uses frozen old-cell records plus the two extension seeds; invoke `evaluation/crossed/aggregate_extension.py --help` for its three input directories. It validates response counts and file hashes before computing each requested-convention interaction. The old audit inputs and response cells are not bundled.

For MATH500 transfer, provide the frozen `protocol.json` and `inputs/math500_hash.jsonl` and `inputs/math500_boxed.jsonl` in one root directory. The protocol must name the three model paths and record the input hashes, decoding parameters, and model-loading options. The original input builder and frozen item files are not included; constructing new inputs changes the recorded experiment. Run one state per GPU (set `CUDA_VISIBLE_DEVICES` externally), then score and analyze the six cells:

```bash
for state in initial hash_trained boxed_trained; do
  CUDA_VISIBLE_DEVICES=0 python evaluation/math500/worker.py --root /data/math500 --state "$state"
  python evaluation/math500/score.py --root /data/math500 --state "$state"
done
python evaluation/math500/analyze.py --root /data/math500
```

`worker.py` retains the recorded one-response, per-item-seed, SHA256 and tokenization checks. `score.py` and `analyze.py` use the fixed symbolic reader and expect all six response cells. Human sample selection and calibration analysis are in `human_validation/scripts`. The original annotation inputs, labels, signed freeze receipts, and workbook-generation workflow are not bundled. `python human_validation/scripts/analyze.py --self-test` runs synthetic checks. Reproducing the production human results requires the original frozen protocol files, workflow, labels, and receipts; the analysis gate intentionally refuses to run without them. The included guidelines omit an outdated status note, so they do not match the original protocol receipt hash.

For the content-matched SFT control, set `GSM8K_TRAIN_PARQUET` to the original GSM8K training Parquet and `EVAL_ITEMS_DIR` to the two frozen evaluation item files, then run `scripts/build_t21_sft_data.py`. It checks zero train/evaluation question overlap and verifies that paired solution bodies are identical. Train each output with `scripts/t21_sft_train.py --data ... --out ... --base Qwen/Qwen2.5-1.5B-Instruct`.

The scripts preserve the recorded scoring and sampling rules. No full training or test outputs are included in this code release.
