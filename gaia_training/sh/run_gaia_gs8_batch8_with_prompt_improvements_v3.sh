#!/bin/bash
#SBATCH --job-name=gaia_gs8_bs8_v3
#SBATCH --output=logs/gaia_gs8_bs8_v3/slurm_%j.out
#SBATCH --error=logs/gaia_gs8_bs8_v3/slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --nodes=1

# GAIA RL Training - TEST RUN - group_size=8, batch_size=8 with ALL improvements (v3)
# Changes from v2:
# - Error recovery: invalid responses get "Invalid Tool Call or Final Answer incorrectly formatted" feedback
# - Trajectory logging: WandB tables with full conversations and metadata
# - Metadata tracking: total_tokens, total_turns, max_tokens_exceeded, max_turns_exceeded
# - Fixed dict_mean to handle non-numeric values
# - Direct imports (no try/catch)
#
# This is a TEST RUN to validate all code changes before running grid search

cd /n/fs/vision-mix/sk7524/tinker-cookbook/gaia_training

# Load environment variables
export $(grep -v '^#' .env | xargs)

# Create logs directory
LOGS_DIR="logs/gaia_gs8_bs8_v3"
mkdir -p "$LOGS_DIR"

echo "Starting GAIA Training TEST RUN (group_size=8, batch_size=8, v3 with ALL improvements) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo ""

# Fixed parameters
MODEL_NAME="Qwen/Qwen3-30B-A3B-Instruct-2507"
BATCH_SIZE=8
GROUP_SIZE=8
LEARNING_RATE=1e-5
LORA_RANK=32
MAX_TOKENS=4096
MAX_TRAJECTORY_TOKENS=32768
MAX_NUM_STEPS=7
SEED=0
DATA_PATH="data/inputs/gaia_data.json"
WANDB_PROJECT="gaia-rl"
WANDB_NAME="gaia_gs8_bs8_lr1e-5_v3_test"

LOG_FILE="${LOGS_DIR}/training.log"

echo "Configuration:"
echo "  group_size=$GROUP_SIZE"
echo "  batch_size=$BATCH_SIZE"
echo "  learning_rate=$LEARNING_RATE"
echo "  max_num_steps=$MAX_NUM_STEPS"
echo ""
echo "ALL Improvements included (v3):"
echo "  ✓ Final Answer format with **"
echo "  ✓ Turns remaining message after tool results"
echo "  ✓ Error recovery (invalid responses get feedback and continue)"
echo "  ✓ Trajectory logging to WandB tables"
echo "  ✓ Metadata tracking (tokens, turns, max_exceeded flags)"
echo "  ✓ Fixed dict_mean for non-numeric values"
echo "  ✓ Direct imports (fail fast on missing dependencies)"
echo ""
echo "  Log: $LOG_FILE"
echo ""

# Run training
uv run python train_gaia.py \
    model_name="$MODEL_NAME" \
    batch_size=$BATCH_SIZE \
    group_size=$GROUP_SIZE \
    learning_rate=$LEARNING_RATE \
    lora_rank=$LORA_RANK \
    max_tokens=$MAX_TOKENS \
    max_trajectory_tokens=$MAX_TRAJECTORY_TOKENS \
    max_num_steps=$MAX_NUM_STEPS \
    seed=$SEED \
    gaia_data_path="$DATA_PATH" \
    wandb_project="$WANDB_PROJECT" \
    wandb_name="$WANDB_NAME" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Training completed at $(date)"
echo "Check logs in: $LOGS_DIR"
echo ""
echo "If this test run succeeds, you can proceed with:"
echo "  - Grid search: sbatch sh/gaia_gridsearch_parallel_over_intrinsics.sh"
echo "  - Large run: sbatch sh/run_gaia_gs32_batch32_with_prompt_improvements.sh"
