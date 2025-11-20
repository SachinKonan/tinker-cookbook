# Prefix Testing Sweep

Sweep script to compare different trajectory generation strategies.

## Experiments

The sweep runs 6 experiments in parallel via SLURM job array:

| ID | Type | Configuration | Description |
|----|------|---------------|-------------|
| 0 | Regular | `group-size=8` | Baseline parallel rollout (8 trajectories) |
| 1 | Regular | `group-size=16` | Baseline parallel rollout (16 trajectories) |
| 2 | Tree | `total=8, src=4, branches=2` | Standard tree branching |
| 3 | Tree | `total=16, src=8, branches=2` | Standard tree branching (larger) |
| 4 | Oracle | `total=8, src=4, branches=2` | Oracle-guided branching (Qwen3-30B, 256 tokens) |
| 5 | Oracle | `total=16, src=8, branches=2` | Oracle-guided branching (larger) |

## Usage

### Submit all experiments:
```bash
sh/prefix_testing/sweep/launch_sweep.sh
```

Or directly:
```bash
sbatch sh/prefix_testing/sweep/run_prefix_testing_sweep.sh
```

### Monitor progress:
```bash
squeue -u $USER
```

### Check outputs:
```bash
ls -la logs/prefix_testing/sweep/
```

## Output Structure

```
logs/prefix_testing/sweep/
├── slurm_<jobid>_<taskid>.out/err    # SLURM logs for each task
├── exp_0_regular_gs8/
│   ├── run.log                        # Console output
│   ├── regular_group_rollout_timeline.png
│   ├── trajectory_stats_regular.csv
│   └── experiment_summary.txt         # Experiment metadata
├── exp_1_regular_gs16/
│   ├── run.log
│   ├── regular_group_rollout_timeline.png
│   ├── trajectory_stats_regular.csv
│   └── experiment_summary.txt
├── exp_2_run_t8_src4_br2/
│   ├── run.log
│   ├── tree_visualization.png
│   ├── trajectory_completion_timeline.png
│   ├── trajectory_stats.csv
│   └── experiment_summary.txt
├── exp_3_run_t16_src8_br2/
│   ├── run.log
│   ├── tree_visualization.png
│   ├── trajectory_completion_timeline.png
│   ├── trajectory_stats.csv
│   └── experiment_summary.txt
├── exp_4_oracle_t8_src4_br2/
│   ├── run.log
│   ├── tree_visualization_oracle.png
│   ├── trajectory_completion_timeline_oracle.png
│   ├── trajectory_stats_oracle.csv
│   └── experiment_summary.txt
└── exp_5_oracle_t16_src8_br2/
    ├── run.log
    ├── tree_visualization_oracle.png
    ├── trajectory_completion_timeline_oracle.png
    ├── trajectory_stats_oracle.csv
    └── experiment_summary.txt
```

## Files in Each Experiment Directory

Each experiment writes its outputs directly to its own directory via the `--log-dir` argument, avoiding file conflicts.

- **run.log**: Complete console output from the experiment
- **\*.png**: Visualization plots
  - `tree_visualization.png` / `tree_visualization_oracle.png`: Trajectory tree structure (branching experiments)
  - `trajectory_completion_timeline.png` / `trajectory_completion_timeline_oracle.png`: Timeline showing completion order
  - `regular_group_rollout_timeline.png`: Timeline for baseline experiments
- **\*.csv**: CSV files with trajectory statistics
  - `trajectory_stats.csv`: Trajectory metrics for branching experiments
  - `trajectory_stats_oracle.csv`: Trajectory metrics for oracle experiments
  - `trajectory_stats_regular.csv`: Trajectory metrics for regular rollout
  - Schema: `[traj_ix, src_ix, num_steps_total, num_steps_generated, total_tokens, total_act_tokens, json_of_convo]`
- **experiment_summary.txt**: Metadata (config, job info, exit code)

## Scripts

- **run_prefix_testing_sweep.sh**: Main SLURM job array script
- **launch_sweep.sh**: Helper script to submit the sweep
