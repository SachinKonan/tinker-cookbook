#!/bin/bash
#
# Launch the prefix testing sweep
#

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "================================================"
echo "Launching Prefix Testing Sweep"
echo "================================================"
echo ""
echo "This will submit 6 experiments as a SLURM job array:"
echo ""
echo "  0. Regular group rollout (group size 8)"
echo "  1. Regular group rollout (group size 16)"
echo "  2. Tree branching (8 total, 4 source, 2 branches)"
echo "  3. Tree branching (16 total, 8 source, 2 branches)"
echo "  4. Oracle-guided (8 total, 4 source, 2 branches)"
echo "  5. Oracle-guided (16 total, 8 source, 2 branches)"
echo ""
echo "Outputs will be saved to: logs/prefix_testing/sweep/"
echo ""

# Submit the job array
JOB_ID=$(sbatch "$SCRIPT_DIR/run_prefix_testing_sweep.sh" | awk '{print $NF}')

echo "Job array submitted!"
echo "Job ID: $JOB_ID"
echo ""
echo "Monitor jobs with:"
echo "  squeue -j $JOB_ID"
echo ""
echo "Check output in:"
echo "  logs/prefix_testing/sweep/"
echo ""
echo "Cancel all jobs with:"
echo "  scancel $JOB_ID"
echo ""
