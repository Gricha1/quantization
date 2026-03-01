#!/bin/bash
# PPO training script with Flash-RL quantization on 8k math dataset (NO Truncated Importance Sampling)
# Usage: bash run_ppo_math8k_quantized_no_tis.sh [RUN_NAME] [QUANTIZATION_TYPE] [FP32_LM_HEAD]
# RUN_NAME: experiment name (default: auto-generated)
# QUANTIZATION_TYPE: fp8 or int8 (default: fp8)
# FP32_LM_HEAD: use FP32 for LM head, 0 or 1 (default: 0)

set -e

# Parse arguments (like flash-rl examples)
RUN_NAME=${1:-""}
QUANTIZATION_TYPE=${2:-"fp8"}
FP32_LM_HEAD=${3:-"0"}

# If RUN_NAME is empty, generate one
if [ -z "$RUN_NAME" ]; then
    RUN_NAME="ppo_math8k_${QUANTIZATION_TYPE}_no_tis_Qwen2.5-0.5B_$(date +%Y%m%d-%H%M%S)"
fi

# Validate quantization type
if [[ ! "$QUANTIZATION_TYPE" =~ ^(fp8|int8)$ ]]; then
    echo "ERROR: Invalid QUANTIZATION_TYPE='$QUANTIZATION_TYPE'. Must be one of: fp8, int8"
    echo "Usage: bash $0 [RUN_NAME] [QUANTIZATION_TYPE] [FP32_LM_HEAD]"
    exit 1
fi

# Model configuration
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
project_name='flash_rl_math8k'
exp_name=${RUN_NAME}

# Set Flash-RL configuration
export VERL_LOGGING_LEVEL=DEBUG
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_CONFIGURE_LOGGING=1
export FLASHRL_LOGGING_LEVEL=DEBUG
export FLASHRL_CONFIG=${QUANTIZATION_TYPE}
export FLASHRL_LMHEAD_FP32=${FP32_LM_HEAD}
export COMET_API_KEY="3OfuYHwcRgIwG7DzgzJ190igY"

# Check if verl is installed (check for trainer module specifically)
python -c "import verl.trainer" 2>/dev/null || {
    echo "ERROR: verl.trainer not found. Please run: bash setup_verl.sh"
    exit 1
}

# Fix Flash-RL tqdm conflict before importing
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python3 "$SCRIPT_DIR/fix_flash_rl_vllm_patch.py" 2>&1 | grep -E "(Fixed|ERROR|⚠)" || true
python3 "$SCRIPT_DIR/fix_vllm_weight_utils_tqdm.py" 2>&1 | grep -E "(Fixed|ERROR|⚠)" || true
python3 "$SCRIPT_DIR/fix_tqdm_simple.py" 2>&1 | grep -E "(Fixed|ERROR|⚠)" || true

# Import Flash-RL to activate patching
python -c "import flash_rl; print('Flash-RL imported successfully')" || {
    echo "ERROR: Flash-RL not found. Please run: bash setup_verl.sh"
    exit 1
}

# Set JAX to CPU (if needed)
export JAX_PLATFORMS=cpu

# Ray temp directory
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray_temp}"
mkdir -p $RAY_TEMP_DIR

echo "=========================================="
echo "Flash-RL PPO Training with Quantization (NO TIS)"
echo "=========================================="
echo "Run Name: $RUN_NAME"
echo "Quantization Type: $QUANTIZATION_TYPE"
echo "Model: $MODEL_NAME"
echo "FP32 LM Head: $FP32_LM_HEAD"
echo "FLASHRL_CONFIG: $FLASHRL_CONFIG"
echo "TIS: DISABLED (imp_ratio_cap=-1)"
echo "=========================================="

# Dataset configuration for 8k math dataset
# Adjust these parameters based on your math dataset
# Note: GSM8K train has 7473 rows, so batch_size must be <= 7473
train_data_size=256
val_data_size=1000

# Data file paths - try GSM8K first, then verl-agent/text
GSM8K_TRAIN="$HOME/data/gsm8k/train.parquet"
GSM8K_VAL="$HOME/data/gsm8k/test.parquet"
VERL_TRAIN="$HOME/data/verl-agent/text/train.parquet"
VERL_VAL="$HOME/data/verl-agent/text/test.parquet"

# Check which dataset is available
if [ -f "$GSM8K_TRAIN" ] && [ -f "$GSM8K_VAL" ]; then
    TRAIN_DATA_PATH="$GSM8K_TRAIN"
    VAL_DATA_PATH="$GSM8K_VAL"
    echo "Using GSM8K dataset"
elif [ -f "$VERL_TRAIN" ] && [ -f "$VERL_VAL" ]; then
    TRAIN_DATA_PATH="$VERL_TRAIN"
    VAL_DATA_PATH="$VERL_VAL"
    echo "Using verl-agent/text dataset"
else
    echo "ERROR: Data files not found!"
    echo ""
    echo "Please prepare your data files:"
    echo "  Option 1 (GSM8K): bash prepare_data.sh"
    echo "  Option 2 (custom): mkdir -p $HOME/data/verl-agent/text"
    echo "                     # Then add your train.parquet and test.parquet files"
    echo ""
    echo "Checked paths:"
    echo "  GSM8K: $GSM8K_TRAIN, $GSM8K_VAL"
    echo "  verl-agent: $VERL_TRAIN, $VERL_VAL"
    exit 1
fi

echo "Using data files:"
echo "  Train: $TRAIN_DATA_PATH"
echo "  Val: $VAL_DATA_PATH"

# TIS is disabled by setting imp_ratio_cap=-1 (on-policy training)
IMP_RATIO_CAP=-1

# Run PPO training with quantization (NO TIS)
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=gae \
  data.train_files=$TRAIN_DATA_PATH \
  data.val_files=$VAL_DATA_PATH \
  data.train_batch_size=$train_data_size \
  data.val_batch_size=$val_data_size \
  data.max_prompt_length=1024 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path=$MODEL_NAME \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.imp_ratio_cap=${IMP_RATIO_CAP} \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.disable_log_stats=False \
  critic.optim.lr=1e-5 \
  critic.model.use_remove_padding=False \
  critic.model.path=$MODEL_NAME \
  critic.model.enable_gradient_checkpointing=False \
  critic.ppo_micro_batch_size_per_gpu=4 \
  critic.model.fsdp_config.param_offload=False \
  critic.model.fsdp_config.optimizer_offload=False \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.logger=['console','comet_ml'] \
  trainer.project_name="${project_name}" \
  trainer.experiment_name="${exp_name}" \
  trainer.n_gpus_per_node=2 \
  trainer.val_before_train=True \
  trainer.nnodes=1 \
  trainer.save_freq=20 \
  trainer.test_freq=10 \
  trainer.total_epochs=30 \
  2>&1 | tee ${RUN_NAME}.log

echo "Training completed!"
echo "Run name: $RUN_NAME"
echo "Log file: ${RUN_NAME}.log"
