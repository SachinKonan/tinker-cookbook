# Prefix Testing Directory

Executable scripts for different trajectory branching strategies in tree-based GRPO.

## Files

- **run.py** - Main tree GRPO script. Creates M source trajectories, then branches from completed ones at random token positions (<50% through assistant messages).

- **run_eager.py** - Eager branching variant. Flips coin (p=0.5) after each policy call during execution to spawn branches immediately, rather than waiting for completion.

- **run_random_branch.py** - Random branching with distinct branch point selection. Pre-selects multiple different branching points to ensure children branch from different locations.

- **run_oracle.py** - Oracle-guided branching. Uses teacher model (Qwen3-30B) to generate infill at branch points, then student model (Qwen3-4B) continues.

- **run_regular_group_rollout.py** - Baseline regular GRPO (no branching). All trajectories start together in parallel for comparison with tree-based approaches.

- **test_branched_advantages.py** - Integration test for training pipeline. Runs branched rollout, computes per-token advantages using prefix trie, and generates interactive Dash visualization. Use `--serve` flag to view trie in browser.

- **__init__.py** - Empty module initialization.

## Common Usage

All scripts use `SearchR1DatasetBuilder` environments and `TinkerTokenCompleter` policy. They create timeline visualizations in `logs/prefix_testing/` showing trajectory completion order.
