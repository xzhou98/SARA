# bash recipe/online_dpo/run_spin_free_vllm_lora.sh
#!/usr/bin/env bash
set -xeuo pipefail


project_name='spin_free'
exp_name='spin_free-DeepSeek-R1-0528-Qwen3-8B-LoRA'

# ---------- Models ----------
MODEL_PATH="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"

# ---------- Data ----------
TRAIN_FILE="/scr/xzhou/safety_alignment_CL/data/merged/train_2k_R1-0528"
TEST_FILE="${TRAIN_FILE}"

# ---------- Checkpoints ----------
CKPTS_DIR="/scr/xzhou/verl/ckpts/${project_name}/${exp_name}"

# ---------- LoRA ----------
lora_rank=8
lora_alpha=16
target_modules="all-linear"

# ---------- Sampling ----------
temperature=0.6
top_p=0.95
top_k=-1

# ---------- Lengths / batch ----------
max_prompt_length=$((896 * 1))
max_response_length=$((896 * 1))
train_prompt_bsz=16
gen_prompt_bsz=${train_prompt_bsz}
train_prompt_mini_bsz=4
log_prob_micro_batch_size_per_gpu=$((train_prompt_mini_bsz/2))

rollout_max_model_len=$((max_prompt_length + max_response_length))

export HYDRA_FULL_ERROR=1
export RAY_TMPDIR=/scr/xzhou/verl/outputs/ray_tmp
mkdir -p "${RAY_TMPDIR}"

# ---------- Conda envs ----------
TRAIN_CONDA_ENV="temp"        # needs vLLM >= 0.9 for verl's run_headless API

set +e
CUDA_VISIBLE_DEVICES=0,1 conda run -n "${TRAIN_CONDA_ENV}" \
    python3 -m recipe.online_dpo.main_spin \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=instruction \
    data.truncation='left' \
    data.filter_overlong_prompts=False \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=2 \
    algorithm.adv_estimator=null \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.dpo_beta=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.nccl_timeout=1200 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.lora_rank=${lora_rank} \
    actor_rollout_ref.model.lora_alpha=${lora_alpha} \
    actor_rollout_ref.model.target_modules=${target_modules} \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=3e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$((train_prompt_mini_bsz/2)) \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.max_model_len=${rollout_max_model_len} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2048 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_max_model_len} \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.prompt_length=${max_prompt_length} \
    actor_rollout_ref.rollout.response_length=${max_response_length} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.agent.default_agent_loop=spin_free_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=recipe/online_dpo/config/spin_free_agent_loops.yaml \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=naive \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    +trainer.disable_validation=True \
    trainer.test_freq=-1 \
    trainer.save_freq=10 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto

TRAIN_EXIT=$?
set -e

echo "[spin_free] Training finished (exit code: ${TRAIN_EXIT})."
exit ${TRAIN_EXIT}
