#!/bin/bash
#
# Start GAIA RL Training
#

set -e

echo "=========================================="
echo "Starting GAIA RL Training"
echo "=========================================="

cd /n/fs/vision-mix/sk7524/tinker-cookbook/gaia_training

uv run python train_gaia.py \
    model_name="Qwen/Qwen3-30B-A3B-Instruct-2507" \
    batch_size=8 \
    group_size=2 \
    learning_rate=1e-5 \
    lora_rank=32 \
    max_tokens=4096 \
    max_trajectory_tokens=32768 \
    max_num_steps=7 \
    seed=0 \
    gaia_data_path="data/inputs/gaia_data.json" \
    wandb_project="gaia-rl"

echo "Training complete!"
