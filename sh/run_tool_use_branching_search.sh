#!/bin/bash
#SBATCH --job-name=tool_use_branching_search
#SBATCH --output=logs/tool_use_branching_search/slurm_%j.out
#SBATCH --error=logs/tool_use_branching_search/slurm_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --nodes=1

# Tool Use Search RL Training with Tree-Based Branching
# Uses Wikipedia vector search via Chroma + Gemini embeddings
# Implements token-level branching to reduce computation

cd /n/fs/vision-mix/sk7524/tinker-cookbook

# Load environment variables
export $(grep -v '^#' .env | xargs)

# Create logs directory
LOGS_DIR="logs/tool_use_branching_search"
mkdir -p "$LOGS_DIR"

echo "=========================================="
echo "Tool Use Search RL Training (BRANCHING)"
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
SRC_TRAJECTORIES=2
NUM_BRANCHES=2
LEARNING_RATE=4e-5
LORA_RANK=32
MAX_TOKENS=1024
MAX_TRAJECTORY_TOKENS=8192
SEED=2
CHROMA_HOST="0.0.0.0"
CHROMA_PORT=8000
CHROMA_COLLECTION="wiki_embeddings"
N_RESULTS=5
WANDB_PROJECT="tool-use-search-rl"
WANDB_NAME="search_branching_qwen3-4b_bs${BATCH_SIZE}_gs${GROUP_SIZE}_src${SRC_TRAJECTORIES}_br${NUM_BRANCHES}_lr${LEARNING_RATE}_rank${LORA_RANK}"

LOG_FILE="${LOGS_DIR}/training.log"
CHROMA_LOG="${LOGS_DIR}/chroma.log"

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Batch size: $BATCH_SIZE"
echo "  Group size: $GROUP_SIZE"
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
echo "Starting Training (with Tree Branching)"
echo "=========================================="
echo ""

uv run python -m tinker_cookbook.recipes.tool_use.search_branching.train \
    model_name="$MODEL_NAME" \
    batch_size=$BATCH_SIZE \
    group_size=$GROUP_SIZE \
    src_trajectories=$SRC_TRAJECTORIES \
    num_branches=$NUM_BRANCHES \
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
    echo "✓ Training completed successfully!"
else
    echo "✗ Training failed with exit code: $TRAIN_EXIT_CODE"
fi
echo "=========================================="

exit $TRAIN_EXIT_CODE
