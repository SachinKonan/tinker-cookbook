#!/bin/bash
#SBATCH --job-name=gaia_gridsearch
#SBATCH --output=logs/gaia_gridsearch/slurm_%j.out
#SBATCH --error=logs/gaia_gridsearch/slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --nodes=1

# GAIA RL Training Grid Search (Parallel Version)
# Runs training with different group sizes concurrently
#
# Grid parameters:
# - group_size: [2, 4, 8]
#
# Total runs: 3 (all parallel)

# Load environment variables
export $(grep -v '^#' ../.env | xargs)

# Create logs directory
LOGS_DIR="logs/gaia_gridsearch"
mkdir -p "$LOGS_DIR"

echo "Starting GAIA Grid Search at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo ""

echo "========================================"
echo "Starting GAIA Grid Search (Parallel)"
echo "Testing group sizes: 2, 4, 8"
echo "Total runs: 3 (all running concurrently)"
echo "Logs directory: $LOGS_DIR"
echo "========================================"
echo ""

# Fixed parameters
MODEL_NAME="Qwen/Qwen3-30B-A3B-Instruct-2507"
BATCH_SIZE=8
LEARNING_RATE=1e-5
LORA_RANK=32
MAX_TOKENS=4096
MAX_TRAJECTORY_TOKENS=32768
MAX_NUM_STEPS=7
SEED=0
DATA_PATH="data/inputs/gaia_data.json"
WANDB_PROJECT="gaia-rl"

# Group sizes to test
GROUP_SIZES=(2 4 8)

# Counter for tracking
RUN_NUM=0
declare -a PIDS=()

# Loop over group sizes and launch in background
for GS in "${GROUP_SIZES[@]}"; do
    RUN_NUM=$((RUN_NUM + 1))
    
    # Create log filename
    LOG_FILE="${LOGS_DIR}/run_${RUN_NUM}_gs${GS}_bs${BATCH_SIZE}_lr${LEARNING_RATE}.log"
    
    echo "[$RUN_NUM/3] Launching: group_size=$GS, batch_size=$BATCH_SIZE, lr=$LEARNING_RATE"
    echo "  Log: $LOG_FILE"
    
    # Launch in background with nohup
    nohup uv run python train_gaia.py \
        model_name="$MODEL_NAME" \
        batch_size=$BATCH_SIZE \
        group_size=$GS \
        learning_rate=$LEARNING_RATE \
        lora_rank=$LORA_RANK \
        max_tokens=$MAX_TOKENS \
        max_trajectory_tokens=$MAX_TRAJECTORY_TOKENS \
        max_num_steps=$MAX_NUM_STEPS \
        seed=$SEED \
        gaia_data_path="$DATA_PATH" \
        wandb_project="$WANDB_PROJECT" \
        wandb_name="gaia_gs${GS}_bs${BATCH_SIZE}_lr${LEARNING_RATE}" \
        > "$LOG_FILE" 2>&1 &
    
    # Store PID
    PIDS+=($!)
done

echo ""
echo "========================================"
echo "All 3 jobs launched!"
echo "PIDs: ${PIDS[@]}"
echo ""
echo "Monitor progress with:"
echo "  tail -f $LOGS_DIR/run_*.log"
echo ""
echo "Check running jobs:"
echo "  ps -p ${PIDS[*]} -o pid,cmd"
echo "========================================"

# Wait for all jobs to complete (required for SBATCH)
echo ""
echo "Waiting for all jobs to complete..."
wait ${PIDS[*]}

echo ""
echo "All jobs completed at $(date)"
echo "Check logs in: $LOGS_DIR"
