#!/usr/bin/env bash
set -euo pipefail

# vLLM's CuMem weight pool is incompatible with PyTorch expandable segments.
unset PYTORCH_CUDA_ALLOC_CONF
unset PYTORCH_ALLOC_CONF

condition="${1:-}"
if [[ -z "$condition" ]]; then
  echo "usage: $0 {iid|adaptive_uncorrected|adaptive_corrected|oracle_corrected|replay_uncorrected|replay_corrected|replay_stratified_corrected}" >&2
  exit 2
fi
shift

: "${TRAIN_FILE:?set TRAIN_FILE to the stratified training parquet}"
: "${VAL_FILE:?set VAL_FILE to the frozen validation parquet}"

model_path="${MODEL_PATH:?set MODEL_PATH to a local model directory or model ID}"
seed="${SEED:-11}"
total_steps="${TOTAL_STEPS:-100}"
model_tag="${MODEL_TAG:-qwen2.5-1.5b}"
target_weights="${TARGET_STRATUM_WEIGHTS:-null}"
initial_weights="$target_weights"
max_prompt_length="${MAX_PROMPT_LENGTH:-1024}"
max_response_length="${MAX_RESPONSE_LENGTH:-1024}"
filter_overlong_prompts="${FILTER_OVERLONG_PROMPTS:-true}"
rollout_max_model_len="${ROLLOUT_MAX_MODEL_LEN:-}"
custom_reward_function_path="${CUSTOM_REWARD_FUNCTION_PATH:-}"
env_bin="${RLVR_ENV_BIN:-$(dirname "$(command -v python)")}"
export PATH="$env_bin:$PATH"
python_bin="$env_bin/python"
if ! runtime_values="$($python_bin - "$max_prompt_length" "$max_response_length" "$filter_overlong_prompts" "$rollout_max_model_len" <<'PY'
import sys

try:
    prompt = int(sys.argv[1])
    response = int(sys.argv[2])
except ValueError:
    raise SystemExit(2)
if prompt <= 0 or response <= 0 or sys.argv[3] not in {"true", "false", "True", "False"}:
    raise SystemExit(2)
model = sys.argv[4]
if model:
    try:
        model_value = int(model)
    except ValueError:
        raise SystemExit(2)
    if model_value < prompt + response:
        raise SystemExit(2)
print(prompt, response, sys.argv[3].lower(), model)
PY
)"; then
  echo "runtime length/filter overrides are invalid" >&2
  exit 2
fi
read -r max_prompt_length max_response_length filter_overlong_prompts rollout_max_model_len <<<"$runtime_values"
sampler_epsilon_raw="${SAMPLER_EPSILON-0.1}"
if ! epsilon="$($python_bin - "$sampler_epsilon_raw" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit(2)
if not math.isfinite(value) or not 0 < value <= 1:
    raise SystemExit(2)
print(value)
PY
)"; then
  echo "SAMPLER_EPSILON must be finite and in (0, 1]" >&2
  exit 2
fi
proposal_exponent_raw="${PROPOSAL_EXPONENT:-1.0}"
if ! proposal_exponent="$($python_bin - "$proposal_exponent_raw" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit(2)
if not math.isfinite(value) or value < 0:
    raise SystemExit(2)
print(value)
PY
)"; then
  echo "PROPOSAL_EXPONENT must be finite and non-negative" >&2
  exit 2
fi
adaptation_enabled=true
apply_correction=true
strict_target_preservation=true
stratified_batching_enabled=false
proposal_schedule_args=()

require_proposal_schedule() {
  if [[ -z "${PROPOSAL_SCHEDULE_PATH:-}" ]]; then
    echo "set PROPOSAL_SCHEDULE_PATH for replay conditions" >&2
    exit 2
  fi
  if [[ ! -r "$PROPOSAL_SCHEDULE_PATH" ]]; then
    echo "PROPOSAL_SCHEDULE_PATH must name a readable proposal schedule" >&2
    exit 2
  fi
  proposal_schedule_args=("+data.sampler.proposal_schedule_path=$PROPOSAL_SCHEDULE_PATH")
}

case "$condition" in
  iid)
    epsilon=1.0
    adaptation_enabled=false
    ;;
  adaptive_uncorrected)
    apply_correction=false
    strict_target_preservation=false
    ;;
  adaptive_corrected)
    ;;
  replay_uncorrected)
    require_proposal_schedule
    adaptation_enabled=false
    apply_correction=false
    strict_target_preservation=false
    ;;
  replay_corrected)
    require_proposal_schedule
    adaptation_enabled=false
    ;;
  replay_stratified_corrected)
    require_proposal_schedule
    adaptation_enabled=false
    stratified_batching_enabled=true
    ;;
  oracle_corrected)
    : "${ORACLE_STRATUM_WEIGHTS:?set ORACLE_STRATUM_WEIGHTS for oracle_corrected}"
    initial_weights="$ORACLE_STRATUM_WEIGHTS"
    adaptation_enabled=false
    ;;
  *)
    echo "unknown condition: $condition" >&2
    exit 2
    ;;
esac

run_root="${RUN_ROOT:-./runs}"
experiment_name="${model_tag}_${condition}_seed${seed}"
if [[ -n "${PROPOSAL_EXPONENT+x}" ]]; then
  beta_slug="${proposal_exponent//./p}"
  experiment_name="${experiment_name}_beta${beta_slug}"
fi
resume_mode="${TRAINER_RESUME_MODE:-}"
resume_mode_args=()
if [[ -n "$resume_mode" ]]; then
  if [[ "$resume_mode" =~ [[:space:]] ]]; then
    echo "TRAINER_RESUME_MODE must not contain whitespace" >&2
    exit 2
  fi
  resume_mode_args=("trainer.resume_mode=$resume_mode")
fi
checkpoint_dir="$run_root/checkpoints/$experiment_name"
mkdir -p "$checkpoint_dir"

rollout_length_args=()
if [[ -n "$rollout_max_model_len" ]]; then
  rollout_length_args=("actor_rollout_ref.rollout.max_model_len=$rollout_max_model_len")
fi
custom_reward_args=()
if [[ -n "$custom_reward_function_path" ]]; then
  custom_reward_args=(
    "custom_reward_function.path=$custom_reward_function_path"
    "custom_reward_function.name=compute_score"
  )
fi

if ! command -v ninja >/dev/null 2>&1; then
  echo "ninja is required for FlashInfer JIT compilation; install it in $env_bin" >&2
  exit 1
fi

exec "$python_bin" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=True \
  algorithm.use_kl_in_reward=False \
  algorithm.prompt_sampling.enabled=True \
  algorithm.prompt_sampling.strict_target_preservation="$strict_target_preservation" \
  algorithm.prompt_sampling.apply_importance_correction="$apply_correction" \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size=32 \
  data.val_batch_size=64 \
  data.max_prompt_length="$max_prompt_length" \
  data.max_response_length="$max_response_length" \
  data.filter_overlong_prompts="$filter_overlong_prompts" \
  data.truncation=error \
  data.shuffle=False \
  data.validation_shuffle=False \
  data.seed="$seed" \
  data.dataloader_num_workers=0 \
  data.custom_cls.path=pkg://recipe.curriculum_identifiability.dataset \
  data.custom_cls.name=IdentifiableRLHFDataset \
  data.sampler.class_path=pkg://recipe.curriculum_identifiability.sampler \
  data.sampler.class_name=TargetProposalSampler \
  +data.sampler.stratum_key=difficulty_bin \
  "+data.sampler.target_stratum_weights=${target_weights}" \
  "+data.sampler.initial_proposal_stratum_weights=${initial_weights}" \
  +data.sampler.epsilon="$epsilon" \
  +data.sampler.ema_decay=0.9 \
  +data.sampler.min_second_moment=1.0e-8 \
  +data.sampler.proposal_exponent="$proposal_exponent" \
  +data.sampler.adaptation_enabled="$adaptation_enabled" \
  +data.sampler.stratified_batching_enabled="$stratified_batching_enabled" \
  "${proposal_schedule_args[@]}" \
  actor_rollout_ref.model.path="$model_path" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1.0e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.shuffle=False \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
  actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  'actor_rollout_ref.actor.checkpoint.save_contents=[model,extra]' \
  'actor_rollout_ref.actor.checkpoint.load_contents=[model,extra]' \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.seed="$seed" \
  "${rollout_length_args[@]}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
  trainer.n_gpus_per_node="${GPU_COUNT:-4}" \
  trainer.nnodes=1 \
  trainer.balance_batch=True \
  trainer.val_before_train=True \
  trainer.test_freq=10 \
  trainer.total_training_steps="$total_steps" \
  trainer.save_freq="$total_steps" \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.max_critic_ckpt_to_keep=1 \
  trainer.default_local_dir="$checkpoint_dir" \
  trainer.logger='[console]' \
  trainer.project_name=curriculum_identifiability \
  trainer.experiment_name="$experiment_name" \
  "${custom_reward_args[@]}" \
  "${resume_mode_args[@]}" \
  "$@"
