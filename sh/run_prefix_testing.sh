#!/bin/bash
#SBATCH --job-name=prefix_testing
#SBATCH --output=logs/prefix_testing/slurm_%j.out
#SBATCH --error=logs/prefix_testing/slurm_%j.err
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --nodes=1

# Prefix Testing - Tree-based Trajectory Branching
# Demonstrates token-level branching for tree GRPO

cd /n/fs/vision-mix/sk7524/tinker-cookbook

# Load environment variables
export $(grep -v '^#' .env | xargs)

# Create logs directory
LOGS_DIR="logs/prefix_testing"
mkdir -p "$LOGS_DIR"

echo "Starting Prefix Testing at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo ""

# Configuration
MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
MAX_TOKENS=2048
MAX_SEARCH_RESULTS=5
SEED=42
SRC_TRAJECTORIES=1
NUM_BRANCHES=2
MAX_TOTAL_TRAJECTORIES=7

LOG_FILE="${LOGS_DIR}/run.log"

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Max tokens: $MAX_TOKENS"
echo "  Max search results: $MAX_SEARCH_RESULTS"
echo "  Seed: $SEED"
echo ""
echo "Branching:"
echo "  Source trajectories: $SRC_TRAJECTORIES"
echo "  Branching factor: $NUM_BRANCHES"
echo "  Max total trajectories: $MAX_TOTAL_TRAJECTORIES"
echo ""
echo "Dataset:"
echo "  Batch size: 1"
echo "  Group size: 1"
echo ""
echo "GAIA Tools:"
echo "  ✓ web_search (DuckDuckGo)"
echo "  ✓ calculator (mathematical expressions)"
echo "  ✓ fetch_webpage (retrieve webpage content)"
echo ""
echo "  Log: $LOG_FILE"
echo ""

# Run prefix testing
uv run python tinker_cookbook/recipes/prefix_testing/run.py \
    --src-trajectories "$SRC_TRAJECTORIES" \
    --num-branches "$NUM_BRANCHES" \
    --max-total-trajectories "$MAX_TOTAL_TRAJECTORIES" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Prefix testing completed at $(date)"
echo "Check logs in: $LOGS_DIR"
echo ""
