#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 || ( "$1" != hash && "$1" != boxed ) || ! "$2" =~ ^(83|84|85|86|87)$ ]]; then
  echo "usage: $0 {hash|boxed} {83|84|85|86|87}" >&2
  exit 2
fi

arm="$1"
export SEED="$2"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
package_root="$(cd "$script_dir/.." && pwd)"
export PYTHONPATH="$package_root:${PYTHONPATH:-}"

: "${MODEL_PATH:?set MODEL_PATH to Qwen2.5-7B-Instruct}"
: "${HASH_DATA_DIR:?set HASH_DATA_DIR to the frozen hash-convention data directory}"
: "${BOXED_DATA_DIR:?set BOXED_DATA_DIR to the paired boxed-convention data directory}"

if [[ "$arm" == hash ]]; then
  data_dir="$HASH_DATA_DIR"
  unset CUSTOM_REWARD_FUNCTION_PATH
else
  data_dir="$BOXED_DATA_DIR"
  export CUSTOM_REWARD_FUNCTION_PATH="$script_dir/gsm8k_boxed_reward.py"
fi

export TRAIN_FILE="$data_dir/train_stratified.parquet"
export VAL_FILE="$data_dir/validation_frozen.parquet"
export MODEL_TAG="qwen2.5-7b-${arm}"
export RUN_ROOT="${RUN_ROOT:-./runs}/${arm}/seed${SEED}"
export TOTAL_STEPS=100
export PROPOSAL_EXPONENT=2.0
export TRAINER_RESUME_MODE=disable
export MAX_PROMPT_LENGTH=1792
export MAX_RESPONSE_LENGTH=1024
export FILTER_OVERLONG_PROMPTS=false
export ROLLOUT_MAX_MODEL_LEN=3072
export TARGET_STRATUM_WEIGHTS='[2302,3282,1912]'
export SAMPLER_EPSILON=0.000001
export ORACLE_STRATUM_WEIGHTS='[24334,34693,40973]'

# These overrides reproduce the checkpoint-retaining, four-GPU Qwen7 cells.
# GPU_COUNT can be set for another machine, but that changes the run configuration.
exec bash "$script_dir/run_qwen2_5_1_5b.sh" oracle_corrected \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  algorithm.rollout_correction.rollout_is=token \
  algorithm.rollout_correction.rollout_is_threshold=2.0 \
  trainer.save_freq=100 \
  trainer.max_actor_ckpt_to_keep=-1 \
  trainer.max_critic_ckpt_to_keep=0 \
  'actor_rollout_ref.actor.checkpoint.save_contents=[model]' \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True
