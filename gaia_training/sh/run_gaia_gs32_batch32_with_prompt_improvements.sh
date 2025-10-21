#!/bin/bash
#SBATCH --job-name=gaia_gs32_bs32_v2
#SBATCH --output=logs/gaia_gs32_bs32_v2/slurm_%j.out
#SBATCH --error=logs/gaia_gs32_bs32_v2/slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --nodes=1

# GAIA RL Training - group_size=32, batch_size=32 with all improvements
# Changes:
# - New format: "Final Answer: <answer>**"
# - Turns remaining message after tool results
# - Error recovery: invalid responses get feedback and continue

cd /n/fs/vision-mix/sk7524/tinker-cookbook/gaia_training

# Load environment variables
export $(grep -v '^#' .env | xargs)

# Create logs directory
LOGS_DIR="logs/gaia_gs32_bs32_v2"
mkdir -p "$LOGS_DIR"

echo "Starting GAIA Training (group_size=32, batch_size=32, v2 with all improvements) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo ""

# Fixed parameters
MODEL_NAME="Qwen/Qwen3-30B-A3B-Instruct-2507"
BATCH_SIZE=32
GROUP_SIZE=32
LEARNING_RATE=1e-5
LORA_RANK=32
MAX_TOKENS=4096
MAX_TRAJECTORY_TOKENS=32768
MAX_NUM_STEPS=7
SEED=0
DATA_PATH="data/inputs/gaia_data.json"
WANDB_PROJECT="gaia-rl"
WANDB_NAME="gaia_gs32_bs32_lr1e-5_v2"

LOG_FILE="${LOGS_DIR}/training.log"

echo "Configuration:"
echo "  group_size=$GROUP_SIZE"
echo "  batch_size=$BATCH_SIZE"
echo "  learning_rate=$LEARNING_RATE"
echo "  max_num_steps=$MAX_NUM_STEPS"
echo "  Improvements: Final Answer format with **, turns remaining, error recovery"
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
