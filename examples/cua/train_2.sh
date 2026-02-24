#!/bin/bash

set -e
export N_GPUS=2
export BASE_MODEL=/models/Qwen3-VL-8B-Instruct
export DATA_DIR=/root/code/wangjiaju/agent-lightning/examples/cua/data
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=cua_0206_hybrid_toy
export PROJECT_NAME=AgentLightning
export WANDB_BASE_URL="http://127.0.0.1:8080"
export WANDB_API_KEY="local-7e29da0b90d83ce28119ba31776f086ba84a1d46"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ENABLE_CUSTOM_ALL_REDUCE=false
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_NCCL_USE_SYMMETRIC_MEMORY=0

echo "Starting training script..."

python -m agentlightning.verl \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/eval.parquet \
    data.train_batch_size=1 \
    data.val_batch_size=1 \
    data.max_prompt_length=32768 \
    data.max_response_length=1024  \
    data.truncation='error' \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_model_len=33792 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.api_key="wangjiaju" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=hermes \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prefix_caching=False \
    actor_rollout_ref.model.path=${BASE_MODEL} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.05 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.3 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=20 \
    trainer.resume_mode="disable" \
    trainer.total_epochs=1 $@