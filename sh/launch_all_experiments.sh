#!/bin/bash
#
# Launch all training experiments and log job IDs
#

set -e

# Create logs/launch directory if it doesn't exist
mkdir -p logs/launch

# Timestamp for this launch
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="logs/launch/launch_${TIMESTAMP}.log"

echo "================================================" | tee -a "$LOGFILE"
echo "Launching all experiments at $(date)" | tee -a "$LOGFILE"
echo "================================================" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# Array to store job info
declare -a JOB_INFO

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Launch modified_tool_use (non-branching)
echo "Submitting: run_modified_tool_use.sh" | tee -a "$LOGFILE"
JOB_ID=$(sbatch "$SCRIPT_DIR/run_modified_tool_use.sh" | awk '{print $NF}')
JOB_INFO+=("$JOB_ID: run_modified_tool_use.sh (modified_tool_use, non-branching)")
echo "  → Job ID: $JOB_ID" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# Launch modified_tool_use branching (src=2)
echo "Submitting: run_modified_tool_use_branching_search.sh" | tee -a "$LOGFILE"
JOB_ID=$(sbatch "$SCRIPT_DIR/run_modified_tool_use_branching_search.sh" | awk '{print $NF}')
JOB_INFO+=("$JOB_ID: run_modified_tool_use_branching_search.sh (modified_tool_use, branching, src=2)")
echo "  → Job ID: $JOB_ID" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# Launch modified_tool_use branching (src=4)
echo "Submitting: run_modified_tool_use_branching_search_src4.sh" | tee -a "$LOGFILE"
JOB_ID=$(sbatch "$SCRIPT_DIR/run_modified_tool_use_branching_search_src4.sh" | awk '{print $NF}')
JOB_INFO+=("$JOB_ID: run_modified_tool_use_branching_search_src4.sh (modified_tool_use, branching, src=4)")
echo "  → Job ID: $JOB_ID" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# Launch tool_use (non-branching)
echo "Submitting: run_tool_use_search.sh" | tee -a "$LOGFILE"
JOB_ID=$(sbatch "$SCRIPT_DIR/run_tool_use_search.sh" | awk '{print $NF}')
JOB_INFO+=("$JOB_ID: run_tool_use_search.sh (tool_use, non-branching)")
echo "  → Job ID: $JOB_ID" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# Launch tool_use branching (src=2)
echo "Submitting: run_tool_use_branching_search.sh" | tee -a "$LOGFILE"
JOB_ID=$(sbatch "$SCRIPT_DIR/run_tool_use_branching_search.sh" | awk '{print $NF}')
JOB_INFO+=("$JOB_ID: run_tool_use_branching_search.sh (tool_use, branching, src=2)")
echo "  → Job ID: $JOB_ID" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# Launch tool_use branching (src=4)
echo "Submitting: run_tool_use_branching_search_src4.sh" | tee -a "$LOGFILE"
JOB_ID=$(sbatch "$SCRIPT_DIR/run_tool_use_branching_search_src4.sh" | awk '{print $NF}')
JOB_INFO+=("$JOB_ID: run_tool_use_branching_search_src4.sh (tool_use, branching, src=4)")
echo "  → Job ID: $JOB_ID" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# Print summary
echo "================================================" | tee -a "$LOGFILE"
echo "All jobs submitted successfully!" | tee -a "$LOGFILE"
echo "================================================" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"
echo "Job Summary:" | tee -a "$LOGFILE"
echo "------------" | tee -a "$LOGFILE"
for info in "${JOB_INFO[@]}"; do
    echo "$info" | tee -a "$LOGFILE"
done
echo "" | tee -a "$LOGFILE"
echo "Log saved to: $LOGFILE" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"
echo "Monitor jobs with: squeue -u $USER" | tee -a "$LOGFILE"
echo "Cancel all jobs with: scancel ${JOB_INFO[0]%:*} ${JOB_INFO[1]%:*} ${JOB_INFO[2]%:*} ${JOB_INFO[3]%:*} ${JOB_INFO[4]%:*} ${JOB_INFO[5]%:*}" | tee -a "$LOGFILE"
