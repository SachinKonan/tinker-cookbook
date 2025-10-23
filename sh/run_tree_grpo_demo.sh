#!/bin/bash
#SBATCH --job-name=tree_grpo_demo
#SBATCH --output=logs/tree_grpo_demo/slurm_%j.out
#SBATCH --error=logs/tree_grpo_demo/slurm_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --nodes=1

# Tree GRPO Demo
# Demonstrates tree-based trajectory generation with Gemini branching
# Uses Wikipedia vector search via Chroma + Gemini embeddings

cd /n/fs/vision-mix/sk7524/tinker-cookbook

# Load environment variables
export $(grep -v '^#' .env | xargs)

# Create logs directory
LOGS_DIR="logs/tree_grpo_demo"
mkdir -p "$LOGS_DIR"

echo "=========================================="
echo "Tree GRPO Demo"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Started at: $(date)"
echo "=========================================="
echo ""

# Tree parameters
TREE_M=2
TREE_K=2
TREE_D=2
TREE_N=8
NUM_PROBLEMS=1

# Model parameters
MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
LORA_RANK=32
MAX_TOKENS=1024
MAX_TRAJECTORY_TOKENS=8192
SEED=2

# Chroma parameters
CHROMA_HOST="0.0.0.0"
CHROMA_PORT=8000
CHROMA_COLLECTION="wiki_embeddings"
N_RESULTS=3
EMBEDDING_MODEL="gemini-embedding-001"
EMBEDDING_DIM=768

# Gemini parameters
GEMINI_MODEL="gemini-2.0-flash-exp"
GEMINI_TEMPERATURE=0.9
GEMINI_TOP_P=0.95
GEMINI_MAX_TOKENS=2048

# Output parameters
LOG_PATH="/tmp/tree_grpo_demo_${SLURM_JOB_ID}"
SAVE_TREES=true
VERBOSE=true

LOG_FILE="${LOGS_DIR}/demo.log"
CHROMA_LOG="${LOGS_DIR}/chroma.log"

echo "Tree Configuration:"
echo "  M (roots): $TREE_M"
echo "  K (branching factor): $TREE_K"
echo "  D (max depth): $TREE_D"
echo "  N (target leaves): $TREE_N"
echo "  Problems: $NUM_PROBLEMS"
echo ""
echo "Model Configuration:"
echo "  Model: $MODEL_NAME"
echo "  LoRA rank: $LORA_RANK"
echo "  Max tokens: $MAX_TOKENS"
echo "  Max trajectory tokens: $MAX_TRAJECTORY_TOKENS"
echo "  Seed: $SEED"
echo ""
echo "Chroma Configuration:"
echo "  Host: $CHROMA_HOST"
echo "  Port: $CHROMA_PORT"
echo "  Collection: $CHROMA_COLLECTION"
echo "  Results per query: $N_RESULTS"
echo "  Embedding model: $EMBEDDING_MODEL"
echo "  Embedding dim: $EMBEDDING_DIM"
echo ""
echo "Gemini Configuration:"
echo "  Model: $GEMINI_MODEL"
echo "  Temperature: $GEMINI_TEMPERATURE"
echo "  Top-p: $GEMINI_TOP_P"
echo "  Max tokens: $GEMINI_MAX_TOKENS"
echo ""
echo "Output Configuration:"
echo "  Log path: $LOG_PATH"
echo "  Save trees: $SAVE_TREES"
echo "  Verbose: $VERBOSE"
echo ""
echo "Logs:"
echo "  Demo: $LOG_FILE"
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
sleep 15
echo "Chroma server should be ready"
echo ""

# Run demo
echo "=========================================="
echo "Starting Tree GRPO Demo"
echo "=========================================="
echo ""

uv run python -m tinker_cookbook.recipes.tool_use.search_tree.demo_tree_rollout \
    tree_m=$TREE_M \
    tree_k=$TREE_K \
    tree_d=$TREE_D \
    tree_n=$TREE_N \
    num_problems=$NUM_PROBLEMS \
    model_name="$MODEL_NAME" \
    lora_rank=$LORA_RANK \
    max_tokens=$MAX_TOKENS \
    max_trajectory_tokens=$MAX_TRAJECTORY_TOKENS \
    seed=$SEED \
    chroma_host="$CHROMA_HOST" \
    chroma_port=$CHROMA_PORT \
    chroma_collection_name="$CHROMA_COLLECTION" \
    n_results=$N_RESULTS \
    embedding_model_name="$EMBEDDING_MODEL" \
    embedding_dim=$EMBEDDING_DIM \
    gemini_model="$GEMINI_MODEL" \
    gemini_temperature=$GEMINI_TEMPERATURE \
    gemini_top_p=$GEMINI_TOP_P \
    gemini_max_output_tokens=$GEMINI_MAX_TOKENS \
    log_path="$LOG_PATH" \
    save_trees=$SAVE_TREES \
    verbose=$VERBOSE \
    2>&1 | tee "$LOG_FILE"

DEMO_EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $DEMO_EXIT_CODE -eq 0 ]; then
    echo "✓ Demo completed successfully!"
    echo "Tree files saved to: $LOG_PATH"
else
    echo "✗ Demo failed with exit code: $DEMO_EXIT_CODE"
fi
echo "=========================================="

exit $DEMO_EXIT_CODE
