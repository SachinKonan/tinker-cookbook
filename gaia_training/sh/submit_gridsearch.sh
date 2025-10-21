#!/bin/bash
#
# Submit GAIA grid search to SLURM
#

cd /n/fs/vision-mix/sk7524/tinker-cookbook/gaia_training

echo "Submitting GAIA grid search job to SLURM..."
echo ""
echo "This will train 3 models in parallel with group sizes: 2, 4, 8"
echo "Each job will use 10 CPUs and 20GB memory"
echo ""

sbatch gaia_gridsearch_parallel.sh

echo ""
echo "Job submitted! Check status with:"
echo "  squeue -u $USER"
echo ""
echo "Monitor logs with:"
echo "  tail -f logs/gaia_gridsearch/*.log"
