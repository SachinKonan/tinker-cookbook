#!/bin/bash
#SBATCH --job-name=gaia_gs8_bs8_v4_llm
#SBATCH --output=logs/gaia_gs8_bs8_v4_llm/slurm_%j.out
#SBATCH --error=logs/gaia_gs8_bs8_v4_llm/slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --nodes=1

# GAIA RL Training - TEST RUN - group_size=8, batch_size=8 with LLM-as-a-Judge (v4)
# Changes from v3:
# - LLM-as-a-Judge: GPT-5-mini grades answers with graduated scores (0.0, 0.3, 0.8, 1.0)
# - New reward system:
#   - Bad formatting -> 0.0
#   - Good formatting + wrong answer (LLM=0.0) -> 0.01 (partial credit)
#   - Good formatting + partial credit (LLM=0.3) -> 0.3
#   - Good formatting + mostly correct (LLM=0.8) -> 0.8
#   - Good formatting + correct (LLM=1.0) -> 1.0
# - Fallback to exact match if LLM judge fails
#
# This is a TEST RUN to validate LLM judge integration

cd /n/fs/vision-mix/sk7524/tinker-cookbook/gaia_training

# Load environment variables
export $(grep -v '^#' .env | xargs)

# Create logs directory
LOGS_DIR="logs/gaia_gs8_bs8_v4_llm"
mkdir -p "$LOGS_DIR"

echo "Starting GAIA Training TEST RUN (group_size=8, batch_size=8, v4 with LLM Judge) at $(date)"
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
WANDB_NAME="gaia_gs8_bs8_lr1e-5_v4_llm_judge"

# LLM Judge parameters
USE_LLM_JUDGE=True
LLM_JUDGE_MODEL="gpt-5-mini"

LOG_FILE="${LOGS_DIR}/training.log"

echo "Configuration:"
echo "  group_size=$GROUP_SIZE"
echo "  batch_size=$BATCH_SIZE"
echo "  learning_rate=$LEARNING_RATE"
echo "  max_num_steps=$MAX_NUM_STEPS"
echo ""
echo "LLM Judge Configuration:"
echo "  use_llm_judge=$USE_LLM_JUDGE"
echo "  llm_judge_model=$LLM_JUDGE_MODEL"
echo ""
echo "ALL Improvements included (v4):"
echo "  ✓ Final Answer format with **"
echo "  ✓ Turns remaining message after tool results"
echo "  ✓ Error recovery (invalid responses get feedback and continue)"
echo "  ✓ Trajectory logging to WandB tables"
echo "  ✓ Metadata tracking (tokens, turns, max_exceeded flags)"
echo "  ✓ Fixed dict_mean for non-numeric values"
echo "  ✓ Direct imports (fail fast on missing dependencies)"
echo "  ✓ LLM-as-a-Judge with graduated rewards (0.0, 0.3, 0.8, 1.0)"
echo "  ✓ New reward system (formatting + content)"
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
    use_llm_judge=$USE_LLM_JUDGE \
    llm_judge_model="$LLM_JUDGE_MODEL" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Training completed at $(date)"
echo "Check logs in: $LOGS_DIR"
echo ""
echo "Compare with baseline (no LLM judge):"
echo "  - Baseline: logs/gaia_gs8_bs8_v3/"
echo "  - LLM Judge: logs/gaia_gs8_bs8_v4_llm/"
echo ""
echo "If this test run succeeds, you can:"
echo "  - Scale up: modify batch_size and group_size"
echo "  - Try different LLM judges: llm_judge_model='gpt-4o'"
