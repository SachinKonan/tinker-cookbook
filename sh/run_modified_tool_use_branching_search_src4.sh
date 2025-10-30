#!/bin/bash
#SBATCH --job-name=modified_tool_use_branching
#SBATCH --output=logs/modified_tool_use_branching/slurm_%j.out
#SBATCH --error=logs/modified_tool_use_branching/slurm_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --nodes=1

# Modified Tool Use RL Training with Tree-Based Branching
# Uses GAIA tools (web_search, calculator, fetch_webpage)
# Implements token-level branching to reduce computation

cd /n/fs/vision-mix/sk7524/tinker-cookbook

# Load environment variables
export $(grep -v '^#' .env | xargs)

# Create logs directory
LOGS_DIR="logs/modified_tool_use_branching"
mkdir -p "$LOGS_DIR"

echo "=========================================="
echo "Modified Tool Use Training (BRANCHING)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Started at: $(date)"
echo "=========================================="
echo ""

# Training parameters
MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
BATCH_SIZE=512
GROUP_SIZE=8
SRC_TRAJECTORIES=4
NUM_BRANCHES=2
LEARNING_RATE=4e-5
LORA_RANK=32
MAX_TOKENS=1024
MAX_TRAJECTORY_TOKENS=8192
SEED=2
MAX_SEARCH_RESULTS=5
WANDB_PROJECT="modified-tool-use-rl"
WANDB_NAME="branching_fixedadv_modified_gs${GROUP_SIZE}_src${SRC_TRAJECTORIES}_br${NUM_BRANCHES}_bs${BATCH_SIZE}_lr${LEARNING_RATE}"

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
echo "Tree Branching Configuration:"
echo "  ✓ Tree branching enabled"
echo "  Source trajectories: $SRC_TRAJECTORIES"
echo "  Branching factor: $NUM_BRANCHES"
echo "  Target group size: $GROUP_SIZE"
echo ""
echo "WandB Configuration:"
echo "  Project: $WANDB_PROJECT"
echo "  Run name: $WANDB_NAME"
echo ""
echo "GAIA Tools:"
echo "  ✓ web_search (DuckDuckGo)"
echo "  ✓ calculator (mathematical expressions)"
echo "  ✓ fetch_webpage (retrieve webpage content)"
echo ""
echo "  Log: $LOG_FILE"
echo ""

# Run training with branching
echo "=========================================="
echo "Starting Training (with Tree Branching)"
echo "=========================================="
echo ""

uv run python -m tinker_cookbook.recipes.modified_tool_use.search_branching.train \
    model_name="$MODEL_NAME" \
    batch_size=$BATCH_SIZE \
    group_size=$GROUP_SIZE \
    src_trajectories=$SRC_TRAJECTORIES \
    num_branches=$NUM_BRANCHES \
    learning_rate=$LEARNING_RATE \
    lora_rank=$LORA_RANK \
    max_tokens=$MAX_TOKENS \
    max_trajectory_tokens=$MAX_TRAJECTORY_TOKENS \
    max_search_results=$MAX_SEARCH_RESULTS \
    seed=$SEED \
    wandb_project="$WANDB_PROJECT" \
    wandb_name="$WANDB_NAME" \
    2>&1 | tee "$LOG_FILE"

TRAIN_EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully!"
else
    echo "✗ Training failed with exit code: $TRAIN_EXIT_CODE"
fi
echo "=========================================="
echo "Completed at: $(date)"
echo "Check logs in: $LOGS_DIR"
echo "=========================================="

exit $TRAIN_EXIT_CODE
