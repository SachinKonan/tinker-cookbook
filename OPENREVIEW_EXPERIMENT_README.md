# ICLR OpenReview SFT vs RL Experiment

This directory contains scripts for training models to generate ICLR OpenReview-style reviews using both Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL).

## Setup

The data has been prepared from ICLR papers (2020-2024):
- **Train**: 8,716 samples
- **Test**: 2,175 samples
- **Total**: 10,891 samples

Each sample contains:
- Paper title and abstract
- Ground truth review summary
- Decision score (1-4): 1=reject, 2=poster, 3=spotlight, 4=oral
- Rating score (1-10)

Reviews are filtered to 10-2,500 characters and selected to be closest to the paper's average rating.

## Data Preparation

Already completed! The data is in `train_test_metadata.json`.

To regenerate:
```bash
uv run python data_prep.py
```

## Training

### SFT Training (Rating Only)
```bash
uv run python run_openreview_experiment.py --mode sft
```

### SFT Training (With Decision Prediction)
```bash
uv run python run_openreview_experiment.py --mode sft --predict-decision
```

### RL Training (Rating Only)
```bash
uv run python run_openreview_experiment.py --mode rl
```

### RL Training (With Decision Prediction)
```bash
uv run python run_openreview_experiment.py --mode rl --predict-decision
```

### Dry Run (5 samples for testing)
```bash
uv run python run_openreview_experiment.py --mode sft --dry-run
uv run python run_openreview_experiment.py --mode rl --dry-run
```

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mode` | Training mode: `sft` or `rl` | **Required** |
| `--predict-decision` | Predict decision (1-4) in addition to rating | `False` |
| `--dry-run` | Use only 5 samples for testing | `False` |
| `--data-path` | Path to training data JSON | `train_test_metadata.json` |
| `--base-url` | Base URL for Tinker service | `None` |
| `--log-path` | Path to save logs/checkpoints | Auto-generated |
| `--batch-size` | Batch size | 32 (SFT), 32 (RL) |
| `--learning-rate` | Learning rate | 1e-4 (SFT), 4e-5 (RL) |

## Output Format

### Without `--predict-decision`:
```json
{
  "summary": "Review text here...",
  "rating": 7
}
```

### With `--predict-decision`:
```json
{
  "summary": "Review text here...",
  "decision": 2,
  "rating": 7
}
```

## Reward Function (RL Only)

The RL training uses a simple distance-based reward:

```python
def get_reward(pred_rating, gt_rating):
    if pred_rating is None:  # decode failure
        return -10.0
    return -abs(pred_rating - gt_rating)
```

- Perfect match: reward = 0
- Off by 1: reward = -1
- Off by 5: reward = -5
- Decode failure: reward = -10

Advantages are computed per group (16 samples per input) as:
```
advantage = reward - mean(group_rewards)
```

## Files

- **`data_prep.py`**: Prepares training data from OpenReview CSV
- **`tinker_cookbook/recipes/sl_loop_openreview.py`**: SFT training script
- **`tinker_cookbook/recipes/rl_loop_openreview.py`**: RL training script
- **`run_openreview_experiment.py`**: Main runner with CLI
- **`train_test_metadata.json`**: Prepared training/test data

## Logs and Checkpoints

Default log locations:
- SFT: `/tmp/tinker-examples/sl-loop-openreview/`
- SFT (with decision): `/tmp/tinker-examples/sl-loop-openreview-with-decision/`
- RL: `/tmp/tinker-examples/rl-loop-openreview/`
- RL (with decision): `/tmp/tinker-examples/rl-loop-openreview-with-decision/`

Checkpoints are saved every 20 batches.

## Experiment Matrix

Grid search over:
1. **Training method**: SFT vs RL
2. **Prediction task**: Rating only vs Rating + Decision

Future extensions:
- Input type: Title+Abstract vs Full paper text (PDF)
