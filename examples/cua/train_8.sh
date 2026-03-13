#!/bin/bash

set -e
export N_GPUS=8
export BASE_MODEL=/models/Qwen3-VL-8B-Instruct
export DATA_DIR=/root/code/zql/agent-lightning/examples/cua/data
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=cua_0313_hybrid_reward
export PROJECT_NAME=AgentLightning
export CHECKPOINT_DIR=${CHECKPOINT_DIR:-/root/code/zql/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}
export WANDB_BASE_URL="http://localhost:8080"
export WANDB_API_KEY="local-8884841b55f713b92811d237a08f40ae4caa7901"
# export WANDB_DIR=${WANDB_DIR:-/root/code/zql/wandb}
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ENABLE_CUSTOM_ALL_REDUCE=false
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_NCCL_USE_SYMMETRIC_MEMORY=0

mkdir -p "${CHECKPOINT_DIR}"

echo "Starting training script..."
echo "Checkpoint dir: ${CHECKPOINT_DIR}"

USE_LORA=${USE_LORA:-false}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_ADAPTER_PATH=${LORA_ADAPTER_PATH:-}

LORA_ARGS=()
if [ "${USE_LORA}" = "true" ]; then
    LORA_ARGS+=("actor_rollout_ref.model.lora_rank=${LORA_RANK}")
    LORA_ARGS+=("actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}")
    LORA_ARGS+=("actor_rollout_ref.model.target_modules=all-linear")
    if [ -n "${LORA_ADAPTER_PATH}" ]; then
        LORA_ARGS+=("actor_rollout_ref.model.lora_adapter_path=${LORA_ADAPTER_PATH}")
    fi
    echo "LoRA mode enabled: rank=${LORA_RANK}, alpha=${LORA_ALPHA}"
else
    echo "LoRA mode disabled: full fine-tune"
fi

python -m agentlightning.verl \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=${DATA_DIR}/train_100.parquet \
    data.val_files=${DATA_DIR}/eval_20.parquet \
    data.train_batch_size=4 \
    data.val_batch_size=null \
    data.max_prompt_length=32768 \
    data.max_response_length=1024  \
    data.truncation='error' \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.max_model_len=33792 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.api_key="cua" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=hermes \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prefix_caching=False \
    actor_rollout_ref.model.path=${BASE_MODEL} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.03 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.val_before_train=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.default_local_dir=${CHECKPOINT_DIR} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.resume_mode="disable" \
    trainer.total_epochs=2 \
    "${LORA_ARGS[@]}" \
    "$@"