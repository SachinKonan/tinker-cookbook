#!/bin/bash
#SBATCH --job-name=gaia_gs_v5_3ep
#SBATCH --output=logs/gaia_gridsearch_v5_3ep/slurm_%j.out
#SBATCH --error=logs/gaia_gridsearch_v5_3ep/slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --nodes=1

# GAIA RL Training Grid Search v5 - 3 Epochs (Parallel Version)
# Runs training with different hyperparameters concurrently
#
# Grid parameters:
# - group_size: [8, 16]
# - learning_rate: [3.2e-4, 3.2e-3]
# - num_epochs: 3
#
# Total runs: 4 (all parallel)

cd /n/fs/vision-mix/sk7524/tinker-cookbook/gaia_training

# Load environment variables
export $(grep -v '^#' .env | xargs)

# Create logs directory
LOGS_DIR="logs/gaia_gridsearch_v5_3ep"
mkdir -p "$LOGS_DIR"

echo "Starting GAIA Grid Search v5 (3 Epochs) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo ""

echo "========================================"
echo "GAIA Grid Search v5 - 3 Epochs (Parallel)"
echo "Testing:"
echo "  group_size: [8, 16]"
echo "  learning_rate: [3.2e-4, 3.2e-3]"
echo "  num_epochs: 3"
echo "Total runs: 4 (all running concurrently)"
echo "Logs directory: $LOGS_DIR"
echo "========================================"
echo ""

# Fixed parameters
MODEL_NAME="Qwen/Qwen3-30B-A3B-Instruct-2507"
BATCH_SIZE=8
LORA_RANK=32
MAX_TOKENS=4096
MAX_TRAJECTORY_TOKENS=32768
MAX_NUM_STEPS=7
SEED=0
DATA_PATH="data/inputs/gaia_data.json"
WANDB_PROJECT="gaia-rl"
NUM_EPOCHS=3

# Grid parameters
GROUP_SIZES=(8 16)
LEARNING_RATES=(3.2e-4 3.2e-3)

# Counter for tracking
RUN_NUM=0
declare -a PIDS=()

# Nested loop over group sizes and learning rates
for GS in "${GROUP_SIZES[@]}"; do
    for LR in "${LEARNING_RATES[@]}"; do
        RUN_NUM=$((RUN_NUM + 1))

        # Create log filename
        LOG_FILE="${LOGS_DIR}/run_${RUN_NUM}_gs${GS}_bs${BATCH_SIZE}_lr${LR}_ep${NUM_EPOCHS}.log"

        echo "[$RUN_NUM/4] Launching: group_size=$GS, lr=$LR, epochs=$NUM_EPOCHS"
        echo "  Log: $LOG_FILE"

        # Launch in background with nohup
        nohup uv run python train_gaia.py \
            model_name="$MODEL_NAME" \
            batch_size=$BATCH_SIZE \
            group_size=$GS \
            learning_rate=$LR \
            lora_rank=$LORA_RANK \
            max_tokens=$MAX_TOKENS \
            max_trajectory_tokens=$MAX_TRAJECTORY_TOKENS \
            max_num_steps=$MAX_NUM_STEPS \
            seed=$SEED \
            gaia_data_path="$DATA_PATH" \
            wandb_project="$WANDB_PROJECT" \
            wandb_name="gaia_gs${GS}_bs${BATCH_SIZE}_lr${LR}_ep${NUM_EPOCHS}" \
            num_epochs=$NUM_EPOCHS \
            > "$LOG_FILE" 2>&1 &

        # Store PID
        PIDS+=($!)
    done
done

echo ""
echo "========================================"
echo "All 4 jobs launched!"
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
echo ""
echo "Results:"
echo "  Run 1: group_size=8,  lr=3.2e-4, epochs=3"
echo "  Run 2: group_size=8,  lr=3.2e-3, epochs=3"
echo "  Run 3: group_size=16, lr=3.2e-4, epochs=3"
echo "  Run 4: group_size=16, lr=3.2e-3, epochs=3"
