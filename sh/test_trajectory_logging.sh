#!/bin/bash
#SBATCH --job-name=test_trajectory_logging
#SBATCH --output=logs/test_trajectory_logging/slurm_%j.out
#SBATCH --error=logs/test_trajectory_logging/slurm_%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --nodes=1

# Test Script for Trajectory Logging
# Minimal config: batch_size=1, group_size=1, n_batches=20
# Tests local JSON saving and WandB table logging every 5 steps

cd /n/fs/vision-mix/sk7524/tinker-cookbook

# Load environment variables
export $(grep -v '^#' .env | xargs)

# Create logs directory
LOGS_DIR="logs/test_trajectory_logging"
mkdir -p "$LOGS_DIR"

echo "=========================================="
echo "TEST: Trajectory Logging"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Started at: $(date)"
echo "=========================================="
echo ""

# Test parameters - MINIMAL CONFIG
MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
BATCH_SIZE=1
GROUP_SIZE=1
MAX_STEPS=20
SRC_TRAJECTORIES=1
NUM_BRANCHES=1
LOG_TABLE_EVERY_N_STEPS=5
LEARNING_RATE=1e-4
LORA_RANK=8
MAX_TOKENS=512
MAX_TRAJECTORY_TOKENS=4096
SEED=42
CHROMA_HOST="0.0.0.0"
CHROMA_PORT=8000
CHROMA_COLLECTION="wiki_embeddings"
N_RESULTS=5
WANDB_PROJECT="trajectory-logging-test"
WANDB_NAME="test_trajlog_bs${BATCH_SIZE}_gs${GROUP_SIZE}_maxsteps${MAX_STEPS}"

# Create log file using SLURM job ID
LOG_FILE="${LOGS_DIR}/training_${SLURM_JOB_ID}.log"
CHROMA_LOG="${LOGS_DIR}/chroma.log"

echo "Test Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Batch size: $BATCH_SIZE (minimal)"
echo "  Group size: $GROUP_SIZE (minimal)"
echo "  Max steps: $MAX_STEPS (stops after 20 steps)"
echo "  Log table every N steps: $LOG_TABLE_EVERY_N_STEPS"
echo "  Learning rate: $LEARNING_RATE"
echo "  LoRA rank: $LORA_RANK"
echo "  Max tokens: $MAX_TOKENS"
echo "  Max trajectory tokens: $MAX_TRAJECTORY_TOKENS"
echo "  Seed: $SEED"
echo ""
echo "Tree Branching Configuration:"
echo "  ✓ Tree branching enabled"
echo "  Source trajectories: $SRC_TRAJECTORIES"
echo "  Branching factor: $NUM_BRANCHES"
echo "  Target group size: $GROUP_SIZE"
echo ""
echo "Chroma Configuration:"
echo "  Host: $CHROMA_HOST"
echo "  Port: $CHROMA_PORT"
echo "  Collection: $CHROMA_COLLECTION"
echo "  Results per query: $N_RESULTS"
echo ""
echo "WandB Configuration:"
echo "  Project: $WANDB_PROJECT"
echo "  Run name: $WANDB_NAME"
echo ""
echo "Logs:"
echo "  Training: $LOG_FILE"
echo "  Chroma: $CHROMA_LOG"
echo ""
echo "Expected Output:"
echo "  - 20 local JSON files in logs/saved_local_data/$WANDB_NAME/trajectories/"
echo "  - 5 WandB tables logged at steps: 0, 5, 10, 15, 20"
echo "  - Each JSON: 1 batch with 1 trajectory"
echo "  - Each table: 1 row with file_path and branch info"
echo ""

# Trap to cleanup Chroma server on exit
cleanup() {
    echo ""
    echo "=========================================="
    echo "Cleaning up..."
    echo "=========================================="
    if [ ! -z "$CHROMA_PID" ]; then
        echo "Killing Chroma server (PID: $CHROMA_PID)"
        kill $CHROMA_PID 2>/dev/null
        wait $CHROMA_PID 2>/dev/null
        echo "Chroma server stopped"
    fi
    echo "Completed at: $(date)"
    echo "=========================================="
}
trap cleanup EXIT

# Start Chroma server in background
echo "Starting Chroma server..."
bash /n/fs/vision-mix/sk7524/run_chroma.sh > "$CHROMA_LOG" 2>&1 &
CHROMA_PID=$!
echo "Chroma server started (PID: $CHROMA_PID)"
echo "Chroma logs: $CHROMA_LOG"

# Wait for Chroma to be ready
echo "Waiting for Chroma server to start..."
sleep 10
echo "Chroma server should be ready"
echo ""

# Run training with branching
echo "=========================================="
echo "Starting Test Training"
echo "=========================================="
echo ""

uv run python -m tinker_cookbook.recipes.tool_use.search_branching.train \
    model_name="$MODEL_NAME" \
    batch_size=$BATCH_SIZE \
    group_size=$GROUP_SIZE \
    max_steps=$MAX_STEPS \
    src_trajectories=$SRC_TRAJECTORIES \
    num_branches=$NUM_BRANCHES \
    log_table_every_n_steps=$LOG_TABLE_EVERY_N_STEPS \
    learning_rate=$LEARNING_RATE \
    lora_rank=$LORA_RANK \
    max_tokens=$MAX_TOKENS \
    max_trajectory_tokens=$MAX_TRAJECTORY_TOKENS \
    seed=$SEED \
    chroma_host="$CHROMA_HOST" \
    chroma_port=$CHROMA_PORT \
    chroma_collection_name="$CHROMA_COLLECTION" \
    n_results=$N_RESULTS \
    wandb_project="$WANDB_PROJECT" \
    wandb_name="$WANDB_NAME" \
    2>&1 | tee "$LOG_FILE"

TRAIN_EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✓ Test completed successfully!"
    echo ""
    echo "Verify results:"
    echo "  1. Check local JSON files:"
    echo "     ls -lh logs/saved_local_data/$WANDB_NAME/trajectories/"
    echo "  2. Check WandB dashboard for 5 trajectory tables"
    echo "  3. Inspect a sample JSON file for correct structure"
else
    echo "✗ Test failed with exit code: $TRAIN_EXIT_CODE"
fi
echo "=========================================="

exit $TRAIN_EXIT_CODE
