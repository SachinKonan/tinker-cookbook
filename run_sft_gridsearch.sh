#!/bin/bash
#SBATCH --job-name=sft_gridsearch
#SBATCH --output=logs/sft_gridsearch/slurm_%j.out
#SBATCH --error=logs/sft_gridsearch/slurm_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --nodes=1

# Create logs directory if it doesn't exist
mkdir -p logs/sft_gridsearch

echo "Starting SFT Grid Search at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo ""

# Run the parallel grid search script
./sft_gridsearch_parallel.sh

echo ""
echo "Finished at $(date)"
