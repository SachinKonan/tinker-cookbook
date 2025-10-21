#!/bin/bash
#SBATCH --job-name=gaia_gridsearch_intrinsics
#SBATCH --output=logs/gaia_gridsearch_intrinsics/slurm_%j.out
#SBATCH --error=logs/gaia_gridsearch_intrinsics/slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --nodes=1

# GAIA RL Training Grid Search over Intrinsic Hyperparameters (Parallel Version)
# Runs training with different learning rates, LoRA ranks, and batch sizes concurrently
#
# Grid parameters:
# - learning_rate: [1e-5, 1e-4]
# - lora_rank: [32, 64]
# - batch_size: [8, 16]
# - group_size: 8 (fixed)
#
# Total runs: 8 (all parallel)

cd /n/fs/vision-mix/sk7524/tinker-cookbook/gaia_training

# Load environment variables
export $(grep -v '^#' .env | xargs)

# Create logs directory
LOGS_DIR="logs/gaia_gridsearch_intrinsics"
mkdir -p "$LOGS_DIR"

echo "Starting GAIA Grid Search (Intrinsics) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo ""

echo "========================================"
echo "Starting GAIA Grid Search over Intrinsic Hyperparameters (Parallel)"
echo "Testing learning rates: 1e-5, 1e-4"
echo "Testing LoRA ranks: 32, 64"
echo "Testing batch sizes: 8, 16"
echo "Fixed group size: 8"
echo "Total runs: 8 (all running concurrently)"
echo "Logs directory: $LOGS_DIR"
echo "========================================"
echo ""

# Fixed parameters
MODEL_NAME="Qwen/Qwen3-30B-A3B-Instruct-2507"
GROUP_SIZE=8
MAX_TOKENS=4096
MAX_TRAJECTORY_TOKENS=32768
MAX_NUM_STEPS=7
SEED=0
DATA_PATH="data/inputs/gaia_data.json"
WANDB_PROJECT="gaia-rl-gridsearch"

# Grid search parameters
LEARNING_RATES=(1e-5 1e-4)
LORA_RANKS=(32 64)
BATCH_SIZES=(8 16)

# Counter for tracking
RUN_NUM=0
declare -a PIDS=()

# Loop over all combinations and launch in background
for LR in "${LEARNING_RATES[@]}"; do
    for RANK in "${LORA_RANKS[@]}"; do
        for BS in "${BATCH_SIZES[@]}"; do
            RUN_NUM=$((RUN_NUM + 1))

            # Create log filename
            LOG_FILE="${LOGS_DIR}/run_${RUN_NUM}_lr${LR}_rank${RANK}_bs${BS}.log"

            # Format LR for display and wandb name
            LR_STR=$(echo $LR | sed 's/e-0*/e-/')  # Format: 1e-5 or 1e-4

            echo "[$RUN_NUM/8] Launching: lr=$LR, rank=$RANK, batch_size=$BS, group_size=$GROUP_SIZE"
            echo "  Log: $LOG_FILE"

            # Launch in background with nohup
            nohup uv run python train_gaia.py \
                model_name="$MODEL_NAME" \
                batch_size=$BS \
                group_size=$GROUP_SIZE \
                learning_rate=$LR \
                lora_rank=$RANK \
                max_tokens=$MAX_TOKENS \
                max_trajectory_tokens=$MAX_TRAJECTORY_TOKENS \
                max_num_steps=$MAX_NUM_STEPS \
                seed=$SEED \
                gaia_data_path="$DATA_PATH" \
                wandb_project="$WANDB_PROJECT" \
                wandb_name="gaia_lr${LR}_rank${RANK}_bs${BS}_gs${GROUP_SIZE}" \
                > "$LOG_FILE" 2>&1 &

            # Store PID
            PIDS+=($!)
        done
    done
done

echo ""
echo "========================================"
echo "All 8 jobs launched!"
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
