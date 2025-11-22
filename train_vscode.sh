#!/usr/bin/env bash
# 用法： bash scripts/run_empdia_multiturn.sh
set -euxo pipefail

ulimit -n 65535
# export RAY_DEBUG_POST_MORTEM=1

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/configs"
SAVE_DIR="$PROJECT_DIR/runs/EmpDiaIceberg/grpo_empathy"

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
OFFLOAD=${OFFLOAD:-False}
# MODEL_PATH=${MODEL_PATH:-$HOME/shared-nvme/myx/models/Qwen2.5-0.5B-Instruct}
MODEL_PATH=${MODEL_PATH:-$HOME/shared-nvme/myx/models/Qwen2.5-7B-Instruct}
DATA_DIR=${DATA_DIR:-$PROJECT_DIR/data}
INTERACTION_CFG=${INTERACTION_CFG:-$PROJECT_DIR/configs/interaction_config.yaml}
REWARD_PATH=${REWARD_PATH-$PROJECT_DIR/reward.py}

python -m verl.trainer.main_ppo \
  --config-path="$CONFIG_PATH" \
  --config-name='multiturn_grpo_interaction' \
  reward_model.reward_manager=naive \
  custom_reward_function.path=$REWARD_PATH \
  custom_reward_function.name=compute_score \
  +reward_model.num_examine=2 \
  algorithm.adv_estimator=grpo \
  data.train_batch_size=$TRAIN_BATCH_SIZE \
  data.max_prompt_length=1024 \
  data.max_response_length=1024 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.enable_activation_offload=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=$OFFLOAD \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OFFLOAD \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.ref.fsdp_config.param_offload=$OFFLOAD \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=20 \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=21 \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  +actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.logger='["console"]' \
  trainer.project_name='EmpDiaIceberg' \
  trainer.experiment_name='grpo_empathy' \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=1000000 \
  trainer.default_local_dir="$SAVE_DIR" \
  trainer.test_freq=20 \
  data.train_files="$DATA_DIR/train_10.parquet" \
  data.val_files="$DATA_DIR/test_10.parquet" \
  actor_rollout_ref.rollout.multi_turn.interaction_config_path="$INTERACTION_CFG" \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.enforce_eager=True \
  trainer.total_epochs=15 \
  "$@"