# bash /home/hp6438/project/verl/recipe/online_dpo/run_spin_vllm.sh

#!/usr/bin/env bash
set -xeuo pipefail

project_name='online_dpo'
exp_name='Alex_online_dpo-DeepSeek-R1-0528-Qwen3-8B'


VISIBLE_DEVICES="0, 1"
export HYDRA_FULL_ERROR=1
export RAY_TMPDIR=/tmp/ray_hp6438
mkdir -p "${RAY_TMPDIR}"

# ----- Data / model -----
TRAIN_FILE="/home/hp6438/project/dataset/train_2k_R1-0528"
MODEL_PATH="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"


# ----- Sampling -----
temperature=0.6
top_p=0.95
top_k=-1

# ----- Length / batch -----
learning_rate=1e-6
epochs=1
max_prompt_length=1024
max_response_length=1024
train_batch_size=32
ppo_mini_batch_size=4
ppo_micro_batch_size_per_gpu=4
log_prob_micro_batch_size_per_gpu=4

rollout_max_model_len=$((max_prompt_length + max_response_length))

CKPTS_DIR="/home/hp6438/project/verl/ckpts/${project_name}/${exp_name}_lr-${learning_rate}_epoch-${epochs}"

CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES} python3 -m recipe.online_dpo.main_spin \
  data.train_files="${TRAIN_FILE}" \
  data.prompt_key=instruction \
  data.train_batch_size=${train_batch_size} \
  data.max_prompt_length=${max_prompt_length} \
  data.max_response_length=${max_response_length} \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=${learning_rate} \
  actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
  actor_rollout_ref.actor.optim.weight_decay=0.1 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.prompt_length=${max_prompt_length} \
  actor_rollout_ref.rollout.response_length=${max_response_length} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
  actor_rollout_ref.rollout.max_model_len=${rollout_max_model_len} \
  actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_max_model_len} \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.rollout.temperature=${temperature} \
  actor_rollout_ref.rollout.top_p=${top_p} \
  actor_rollout_ref.rollout.top_k="${top_k}" \
  actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
  actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
  actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  algorithm.use_kl_in_reward=False \
  reward_model.reward_manager=naive \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${project_name}" \
  trainer.experiment_name="${exp_name}" \
  trainer.val_before_train=False \
  +trainer.disable_validation=True \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.save_freq=20 \
  trainer.test_freq=-1 \
  +trainer.log_freq=1 \
  trainer.default_local_dir="${CKPTS_DIR}" \
  trainer.total_epochs=${epochs} 2>&1 | tee spin_free_vllm.log
