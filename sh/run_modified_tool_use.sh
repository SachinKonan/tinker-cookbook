#!/bin/bash
#SBATCH --job-name=modified_tool_use
#SBATCH --output=logs/modified_tool_use/slurm_%j.out
#SBATCH --error=logs/modified_tool_use/slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --nodes=1

# Modified Tool Use RL Training
# Uses GAIA tools (web_search, calculator, fetch_webpage) instead of Chroma/Gemini
# Based on tinker_cookbook/recipes/tool_use but with GAIA environment tools

cd /n/fs/vision-mix/sk7524/tinker-cookbook

# Load environment variables
export $(grep -v '^#' .env | xargs)

# Create logs directory
LOGS_DIR="logs/modified_tool_use"
mkdir -p "$LOGS_DIR"

echo "Starting Modified Tool Use Training at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo ""

# Training parameters (using defaults from train.py)
MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
BATCH_SIZE=512
GROUP_SIZE=8
LEARNING_RATE=4e-5
LORA_RANK=32
MAX_TOKENS=1024
MAX_TRAJECTORY_TOKENS=8192
SEED=2
MAX_SEARCH_RESULTS=5
WANDB_PROJECT="modified-tool-use-rl"
WANDB_NAME="modified_tool_use_gs8_bs512_lr4e-5"

LOG_FILE="${LOGS_DIR}/training.log"

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Batch size: $BATCH_SIZE"
echo "  Group size: $GROUP_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  LoRA rank: $LORA_RANK"
echo "  Max tokens: $MAX_TOKENS"
echo "  Max trajectory tokens: $MAX_TRAJECTORY_TOKENS"
echo "  Max search results: $MAX_SEARCH_RESULTS"
echo "  Seed: $SEED"
echo ""
echo "WandB Configuration:"
echo "  Project: $WANDB_PROJECT"
echo "  Run name: $WANDB_NAME"
echo ""
echo "GAIA Tools:"
echo "  ✓ web_search (Wikipedia, Brave, Bing, etc.)"
echo "  ✓ calculator (mathematical expressions)"
echo "  ✓ fetch_webpage (retrieve webpage content)"
echo ""
echo "  Log: $LOG_FILE"
echo ""

# Run training
uv run python -m tinker_cookbook.recipes.modified_tool_use.train \
    model_name="$MODEL_NAME" \
    batch_size=$BATCH_SIZE \
    group_size=$GROUP_SIZE \
    learning_rate=$LEARNING_RATE \
    lora_rank=$LORA_RANK \
    max_tokens=$MAX_TOKENS \
    max_trajectory_tokens=$MAX_TRAJECTORY_TOKENS \
    max_search_results=$MAX_SEARCH_RESULTS \
    seed=$SEED \
    wandb_project="$WANDB_PROJECT" \
    wandb_name="$WANDB_NAME" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Training completed at $(date)"
echo "Check logs in: $LOGS_DIR"
echo ""
