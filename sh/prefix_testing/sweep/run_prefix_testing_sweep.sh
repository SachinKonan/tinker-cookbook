#!/bin/bash
#SBATCH --job-name=prefix_testing_sweep
#SBATCH --array=0-5
#SBATCH --output=logs/prefix_testing/sweep/slurm_%A_%a.out
#SBATCH --error=logs/prefix_testing/sweep/slurm_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --nodes=1

# Change to project directory
cd /n/fs/vision-mix/sk7524/tinker-cookbook

# Load environment variables
export $(grep -v '^#' .env | xargs)

# Create base sweep directory
mkdir -p logs/prefix_testing/sweep

echo "=========================================="
echo "Prefix Testing Sweep - Experiment Array"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $(hostname)"
echo "Started at: $(date)"
echo ""

# Map array task ID to experiment configuration
case $SLURM_ARRAY_TASK_ID in
  0)
    EXP_NAME="exp_0_regular_gs8"
    SCRIPT="run_regular_group_rollout.py"
    ARGS="--group-size 8"
    DESCRIPTION="Regular group rollout with group size 8"
    ;;
  1)
    EXP_NAME="exp_1_regular_gs16"
    SCRIPT="run_regular_group_rollout.py"
    ARGS="--group-size 16"
    DESCRIPTION="Regular group rollout with group size 16"
    ;;
  2)
    EXP_NAME="exp_2_run_t8_src4_br2"
    SCRIPT="run.py"
    ARGS="--max-total-trajectories 8 --src-trajectories 4 --num-branches 2"
    DESCRIPTION="Tree branching: 8 total, 4 source, 2 branches"
    ;;
  3)
    EXP_NAME="exp_3_run_t16_src8_br2"
    SCRIPT="run.py"
    ARGS="--max-total-trajectories 16 --src-trajectories 8 --num-branches 2"
    DESCRIPTION="Tree branching: 16 total, 8 source, 2 branches"
    ;;
  4)
    EXP_NAME="exp_4_oracle_t8_src4_br2"
    SCRIPT="run_oracle.py"
    ARGS="--max-total-trajectories 8 --src-trajectories 4 --num-branches 2 --oracle-model-name Qwen/Qwen3-30B-A3B --oracle-max-tokens 256"
    DESCRIPTION="Oracle-guided branching: 8 total, 4 source, 2 branches, oracle 256 tokens"
    ;;
  5)
    EXP_NAME="exp_5_oracle_t16_src8_br2"
    SCRIPT="run_oracle.py"
    ARGS="--max-total-trajectories 16 --src-trajectories 8 --num-branches 2 --oracle-model-name Qwen/Qwen3-30B-A3B --oracle-max-tokens 256"
    DESCRIPTION="Oracle-guided branching: 16 total, 8 source, 2 branches, oracle 256 tokens"
    ;;
  *)
    echo "Error: Invalid SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
    exit 1
    ;;
esac

# Create experiment-specific output directory
EXP_DIR="logs/prefix_testing/sweep/${EXP_NAME}"
mkdir -p "$EXP_DIR"

echo "Experiment: $DESCRIPTION"
echo "Output directory: $EXP_DIR"
echo "Script: tinker_cookbook/recipes/prefix_testing/${SCRIPT}"
echo "Arguments: $ARGS"
echo ""

# Run the experiment
# Pass the experiment directory as --log-dir so all outputs go directly there
# The Python script will handle logging to both stdout and run.log
echo "Starting experiment at: $(date)"
echo "------------------------------------------"

uv run python tinker_cookbook/recipes/prefix_testing/${SCRIPT} ${ARGS} --log-dir "${EXP_DIR}"

EXIT_CODE=$?

echo "------------------------------------------"
echo "Experiment completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo ""

# Create summary file
SUMMARY_FILE="${EXP_DIR}/experiment_summary.txt"
cat > "$SUMMARY_FILE" << EOF
Experiment Summary
==================
Name: $EXP_NAME
Description: $DESCRIPTION
Script: $SCRIPT
Arguments: $ARGS

Execution Details
-----------------
Job ID: $SLURM_JOB_ID
Array Task ID: $SLURM_ARRAY_TASK_ID
Node: $(hostname)
Started: $(date)
Exit Code: $EXIT_CODE

Output Files
------------
Console log: run.log
Output directory: ${EXP_DIR}
SLURM output: slurm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out
SLURM error: slurm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err
EOF

echo "Experiment summary saved to: $SUMMARY_FILE"
echo ""
echo "=========================================="
echo "Experiment $SLURM_ARRAY_TASK_ID Complete"
echo "=========================================="

exit $EXIT_CODE
