#!/bin/bash

# Master script to run all training comparisons:
# 1. Standard tool_use/search (baseline with Chroma)
# 2. Branching tool_use/search_branching (with Chroma)
# 3. Standard modified_tool_use (baseline with GAIA tools)
# 4. Branching modified_tool_use/search_branching (with GAIA tools)

cd /n/fs/vision-mix/sk7524/tinker-cookbook

echo "=========================================="
echo "Launching All Training Comparisons"
echo "=========================================="
echo "Started at: $(date)"
echo ""

# Array to store job IDs
declare -a JOB_IDS
declare -a JOB_NAMES

echo "Submitting jobs..."
echo ""

# Job 1: Standard tool_use search (Chroma baseline)
echo "[1/4] Submitting: tool_use/search (baseline)"
JOB1=$(sbatch sh/run_tool_use_search.sh 2>&1 | grep -oP 'Submitted batch job \K[0-9]+')
if [ ! -z "$JOB1" ]; then
    JOB_IDS+=("$JOB1")
    JOB_NAMES+=("tool_use_search_baseline")
    echo "  ✓ Job ID: $JOB1"
else
    echo "  ✗ Failed to submit"
fi
echo ""

# Job 2: Branching tool_use search (Chroma with branching)
echo "[2/4] Submitting: tool_use/search_branching"
JOB2=$(sbatch sh/run_tool_use_branching_search.sh 2>&1 | grep -oP 'Submitted batch job \K[0-9]+')
if [ ! -z "$JOB2" ]; then
    JOB_IDS+=("$JOB2")
    JOB_NAMES+=("tool_use_branching_search")
    echo "  ✓ Job ID: $JOB2"
else
    echo "  ✗ Failed to submit"
fi
echo ""

# Job 3: Standard modified_tool_use (GAIA baseline)
echo "[3/4] Submitting: modified_tool_use (baseline)"
JOB3=$(sbatch sh/run_modified_tool_use.sh 2>&1 | grep -oP 'Submitted batch job \K[0-9]+')
if [ ! -z "$JOB3" ]; then
    JOB_IDS+=("$JOB3")
    JOB_NAMES+=("modified_tool_use_baseline")
    echo "  ✓ Job ID: $JOB3"
else
    echo "  ✗ Failed to submit"
fi
echo ""

# Job 4: Branching modified_tool_use (GAIA with branching)
echo "[4/4] Submitting: modified_tool_use/search_branching"
JOB4=$(sbatch sh/run_modified_tool_use_branching_search.sh 2>&1 | grep -oP 'Submitted batch job \K[0-9]+')
if [ ! -z "$JOB4" ]; then
    JOB_IDS+=("$JOB4")
    JOB_NAMES+=("modified_tool_use_branching")
    echo "  ✓ Job ID: $JOB4"
else
    echo "  ✗ Failed to submit"
fi
echo ""

echo "=========================================="
echo "All Jobs Submitted"
echo "=========================================="
echo ""

# Display summary
echo "Job Summary:"
echo "─────────────────────────────────────────"
for i in "${!JOB_IDS[@]}"; do
    printf "%-35s Job ID: %s\n" "${JOB_NAMES[$i]}" "${JOB_IDS[$i]}"
done
echo "─────────────────────────────────────────"
echo ""

# Print monitoring commands
echo "Monitoring Commands:"
echo "  Check all jobs:  squeue -u \$USER"
echo "  Check specific:  squeue -j ${JOB_IDS[*]}"
echo "  Cancel all:      scancel ${JOB_IDS[*]}"
echo ""

# Print log locations
echo "Log Locations:"
echo "  tool_use_search (baseline):     logs/tool_use_search/"
echo "  tool_use_branching:             logs/tool_use_branching_search/"
echo "  modified_tool_use (baseline):   logs/modified_tool_use/"
echo "  modified_tool_use_branching:    logs/modified_tool_use_branching/"
echo ""

# Print WandB projects
echo "WandB Projects:"
echo "  tool_use comparisons:           tool-use-search-rl"
echo "  modified_tool_use comparisons:  modified-tool-use-rl"
echo ""

echo "Comparison Goals:"
echo "  • Baseline vs Branching: Compare training efficiency"
echo "  • Chroma vs GAIA: Compare tool backend performance"
echo "  • Group size 8: src_trajectories=2, num_branches=2"
echo "  • Expected: ~75% reduction in initial rollout computation"
echo ""

echo "Done at: $(date)"
echo "=========================================="
