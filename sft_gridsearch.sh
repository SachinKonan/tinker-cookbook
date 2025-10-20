#!/bin/bash

# SFT Grid Search Script
# Tests combinations of hyperparameters across different output modes
#
# Grid parameters:
# - batch_size: [32, 64]
# - lora_rank: [32, 64]
# - learning_rate: [1e-4, 1e-5]
# - output_modes: [rating, rating+decision, rating+decision+summary]
# - epochs: 2
#
# Total runs: 2 × 2 × 2 × 3 = 24 runs

set -e  # Exit on error

# Grid search parameters
BATCH_SIZES=(32 64)
LORA_RANKS=(32 64)
LEARNING_RATES=(1e-4 1e-5)
EPOCHS=2

echo "========================================"
echo "Starting SFT Grid Search"
echo "Total runs: 24 (2 batch × 2 rank × 2 lr × 3 modes)"
echo "Epochs per run: $EPOCHS"
echo "========================================"
echo ""

# Counter for tracking progress
RUN_NUM=0
TOTAL_RUNS=24

# Loop over output modes
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

                echo "========================================"
                echo "Run $RUN_NUM/$TOTAL_RUNS"
                echo "Mode: $MODE_NAME"
                echo "Batch Size: $BS"
                echo "LoRA Rank: $RANK"
                echo "Learning Rate: $LR"
                echo "Epochs: $EPOCHS"
                echo "========================================"

                # Run training
                if uv run python run_openreview_experiment.py \
                    --mode sft \
                    --epochs $EPOCHS \
                    --batch-size $BS \
                    --lora-rank $RANK \
                    --learning-rate $LR \
                    $FLAGS; then
                    echo "✓ Run $RUN_NUM/$TOTAL_RUNS completed successfully"
                else
                    echo "✗ Run $RUN_NUM/$TOTAL_RUNS failed with exit code $?"
                    echo "Continuing to next run..."
                fi

                echo ""
            done
        done
    done
done

echo "========================================"
echo "Grid search complete!"
echo "Completed $RUN_NUM runs"
echo "========================================"
