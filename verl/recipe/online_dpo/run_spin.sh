set -e
set -x
VISIBLE_DEVICES="1"
export HYDRA_FULL_ERROR=1
export RAY_TMPDIR=/scr/xzhou/verl/outputs/ray_tmp
mkdir -p "${RAY_TMPDIR}"

temperature=0.6
top_p=0.95
top_k=-1

CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES} python3 -m recipe.spin_free.main_spin \
  data.train_files=/scr/xzhou/safety_alignment_CL/data/merged/train_2k_R1-0528 \
  data.val_files=/scr/xzhou/safety_alignment_CL/data/merged/train_2k_R1-0528 \
  data.train_batch_size=32 \
  data.max_prompt_length=1024 \
  data.max_response_length=2048 \
  actor_rollout_ref.model.path=deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
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
  trainer.logger=console \
  trainer.val_before_train=False \
  trainer.disable_validation=True \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  +trainer.log_freq=1 \
  trainer.total_epochs=2 2>&1 | tee verl_demo.log
