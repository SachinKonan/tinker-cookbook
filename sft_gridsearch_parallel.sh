#!/bin/bash
#SBATCH --job-name=sft_gridsearch
#SBATCH --output=logs/sft_gridsearch/slurm_%j.out
#SBATCH --error=logs/sft_gridsearch/slurm_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --nodes=1

# SFT Grid Search Script (Parallel Version)
# Runs all hyperparameter combinations concurrently in the background
#
# Grid parameters:
# - batch_size: [32, 64]
# - lora_rank: [32, 64]
# - learning_rate: [1e-4, 1e-5]
# - output_modes: [rating, rating+decision, rating+decision+summary]
# - epochs: 2
#
# Total runs: 2 × 2 × 2 × 3 = 24 runs (all parallel)

# Create logs directory
LOGS_DIR="logs/sft_gridsearch"
mkdir -p "$LOGS_DIR"

echo "Starting SFT Grid Search at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo ""

echo "========================================"
echo "Starting SFT Grid Search (Parallel)"
echo "Total runs: 24 (all running concurrently)"
echo "Epochs per run: 2"
echo "Logs directory: $LOGS_DIR"
echo "========================================"
echo ""

# Grid search parameters
BATCH_SIZES=(32 64)
LORA_RANKS=(32 64)
LEARNING_RATES=(1e-4 1e-5)
EPOCHS=2

# Counter for tracking
RUN_NUM=0
declare -a PIDS=()

# Loop over all combinations and launch in background
for MODE in "rating" "rating_decision" "rating_decision_summary"; do
    # Set flags based on mode
    if [ "$MODE" == "rating" ]; then
        FLAGS=""
        MODE_NAME="rating-only"
    elif [ "$MODE" == "rating_decision" ]; then
        FLAGS="--predict-decision"
        MODE_NAME="rating+decision"
    else
        FLAGS="--predict-decision --predict-review"
        MODE_NAME="rating+decision+summary"
    fi

    # Loop over hyperparameters
    for BS in "${BATCH_SIZES[@]}"; do
        for RANK in "${LORA_RANKS[@]}"; do
            for LR in "${LEARNING_RATES[@]}"; do
                RUN_NUM=$((RUN_NUM + 1))

                # Create log filename
                LOG_FILE="${LOGS_DIR}/run_${RUN_NUM}_${MODE}_bs${BS}_rank${RANK}_lr${LR}.log"

                echo "[$RUN_NUM/24] Launching: $MODE_NAME, bs=$BS, rank=$RANK, lr=$LR"
                echo "  Log: $LOG_FILE"

                # Launch in background with nohup
                nohup uv run python run_openreview_experiment.py \
                    --mode sft \
                    --epochs $EPOCHS \
                    --batch-size $BS \
                    --lora-rank $RANK \
                    --learning-rate $LR \
                    $FLAGS \
                    > "$LOG_FILE" 2>&1 &

                # Store PID
                PIDS+=($!)
            done
        done
    done
done

echo ""
echo "========================================"
echo "All 24 jobs launched!"
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
