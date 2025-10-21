
## Deep Dive: max_trajectory_tokens

**Config Parameter**: `max_trajectory_tokens` (default: 32,768 tokens)

**Code**: [`gaia_training/src/gaia_env.py:204-206`](src/gaia_env.py#L204-L206)

### What It Does

`max_trajectory_tokens` limits the **total context length** of a trajectory. After each tool call, we check:

```python
next_observation = self.renderer.build_generation_prompt(self.past_messages)

if next_observation.length > self.max_trajectory_tokens:
    logger.warning(f"Trajectory too long: {next_observation.length} > {self.max_trajectory_tokens}")
    return failure_result  # Episode terminates with reward=0
```

### Context Length Grows With Each Turn

The observation includes the **full conversation history**:
1. System prompt (~500 tokens)
2. User question (~50-200 tokens)
3. All previous assistant responses
4. All previous tool results
5. Turns remaining message

Example trajectory growth:
```
Turn 1: 500 (system) + 100 (question) = 600 tokens
Turn 2: 600 + 150 (assistant) + 500 (tool result) = 1,250 tokens
Turn 3: 1,250 + 100 (assistant) + 800 (tool result) = 2,150 tokens
Turn 4: 2,150 + 200 (assistant) + 1500 (tool result) = 3,850 tokens
...
```

If tool results are long (e.g., fetching full webpages), context can hit 32k quickly!

### Relationship to Logged Metrics

#### `env/all/ob_tokens_per_turn`
This is the **average observation length** across all turns:
- Formula: `sum(transition.ob.length) / total_turns`
- If this metric grows toward `max_trajectory_tokens / turns_per_episode`, you're at risk of hitting the limit

Example:
- `ob_tokens_per_turn = 4000`
- `turns_per_episode = 6`
- Total context at end: ~24,000 tokens (safe, below 32k)

#### `env/all/total_ob_tokens`
Total observation tokens **across all episodes in batch**:
- Formula: `sum(transition.ob.length for all transitions)`
- With `batch_size=8`, `group_size=8`, `turns_per_episode=4`:
  - `total_ob_tokens` ≈ 8 × 8 × 4 × avg_ob_tokens = 256 × avg_ob_tokens

#### Warning Signs
If episodes frequently hit `max_trajectory_tokens`:
1. You'll see many "Trajectory too long" warnings in logs
2. Episodes will fail with `reward=0` (no format/correct metrics)
3. `env/all/total_episodes` may be lower than `batch_size × group_size` (if episodes terminate before logging)

### Why 32k?

- **Model context limit**: Qwen3-30B supports up to 32k tokens
- **Safety margin**: Set slightly below model limit to avoid OOM
- **Tool-heavy tasks**: GAIA requires multiple tool calls with potentially long results

### Tuning max_trajectory_tokens

**Too low** (e.g., 8k):
- Many episodes hit limit → high failure rate
- Model can't complete multi-step reasoning

**Too high** (e.g., 128k):
- May exceed model's context window
- Slower inference (quadratic attention)
- Risk of OOM errors

**Sweet spot** (32k for Qwen3):
- Allows ~5-7 tool calls with moderate-length results
- Fits most GAIA questions

## Deep Dive: by_group/frac_mixed

**Code**: [`tinker_cookbook/rl/metric_util.py:15-31`](../tinker_cookbook/rl/metric_util.py#L15-L31)

### What Is "Mixed"?

A group is considered **mixed** if trajectories have **different rewards** (i.e., not all the same).

### The Algorithm

```python
def _compute_by_group_metrics(trajectory_groups_P, good_thresh=0.5):
    n_groups = len(trajectory_groups_P)  # = batch_size (e.g., 8 questions)
    n_mixed = n_good = n_bad = 0
    
    for tg in trajectory_groups_P:  # For each question
        grp_rewards = tg.get_total_rewards()  # Get rewards for all trajectories (group_size)
        
        if all_same(grp_rewards):  # All rewards identical?
            if grp_rewards[0] >= good_thresh:  # All >= 0.5?
                n_good += 1  # All good
            else:
                n_bad += 1   # All bad
        else:
            n_mixed += 1     # Mixed (some good, some bad)
    
    return {
        "by_group/frac_mixed": n_mixed / n_groups,
        "by_group/frac_all_good": n_good / n_groups,
        "by_group/frac_all_bad": n_bad / n_groups,
    }
```

### Example with batch_size=8, group_size=4

Say we have 8 questions, each generating 4 trajectories:

**Question 1**: Rewards = [1.0, 1.0, 1.0, 1.0]
- All same? ✓
- All >= 0.5? ✓
- Classification: **all_good**

**Question 2**: Rewards = [0.01, 0.01, 0.01, 0.01]
- All same? ✓
- All >= 0.5? ✗
- Classification: **all_bad**

**Question 3**: Rewards = [1.0, 0.01, 1.0, 0.01]
- All same? ✗
- Classification: **mixed**

**Question 4**: Rewards = [1.0, 0.99, 1.0, 1.0]
- All same? ✗ (even tiny difference counts!)
- Classification: **mixed**

**Question 5-8**: Similar classifications...

Result:
- `n_mixed = 3`
- `n_good = 2`
- `n_bad = 3`
- `frac_mixed = 3/8 = 0.375`
- `frac_all_good = 2/8 = 0.25`
- `frac_all_bad = 3/8 = 0.375`

### Why This Matters for GRPO

**GRPO (Group Relative Policy Optimization)** computes advantages as:

```
advantage = reward - mean_group_reward
```

For Question 3 above (mixed):
- Trajectory 1: advantage = 1.0 - 0.505 = +0.495 (reinforce this!)
- Trajectory 2: advantage = 0.01 - 0.505 = -0.495 (suppress this!)
- Strong learning signal! ✓

For Question 1 (all_good):
- Trajectory 1: advantage = 1.0 - 1.0 = 0.0 (no signal)
- Trajectory 2: advantage = 1.0 - 1.0 = 0.0 (no signal)
- No learning! ✗

### Ideal Values

| Metric | Value | Interpretation |
|--------|-------|----------------|
| `frac_mixed` | 0.6-0.8 | **Ideal**: Strong GRPO signal, diverse outcomes |
| `frac_mixed` | 0.3-0.5 | **OK**: Some signal, but many groups are uniform |
| `frac_mixed` | <0.2 | **Bad**: Weak GRPO signal, model not exploring |
| `frac_all_good` | 0.5+ | **Good**: Model is succeeding often |
| `frac_all_bad` | >0.5 | **Bad**: Model is failing too often |

### Debugging Low frac_mixed

If `frac_mixed < 0.2`:

1. **All failing**: `frac_all_bad` high
   - Problem: Questions too hard, or max_steps too low
   - Fix: Easier questions, increase max_steps, or add intermediate rewards

2. **All succeeding**: `frac_all_good` high
   - Problem: Questions too easy (not a bad problem!)
   - Note: This is rare in GAIA

3. **Deterministic policy**: Temperature = 0
   - Problem: Model generates identical trajectories
   - Fix: Increase temperature or use sampling

### Special Case: GAIA Rewards

GAIA has **sparse rewards**: 0.01 (format only), or 0.99-1.0 (correct)

This creates natural diversity:
- If 2/4 trajectories get correct answer: [1.0, 1.0, 0.01, 0.01] → **mixed** ✓
- If all wrong: [0.01, 0.01, 0.01, 0.01] → **all_bad** ✗
- If all right: [1.0, 1.0, 1.0, 1.0] → **all_good** (but still learns!)

### all_same() Implementation

**Code**: [`tinker_cookbook/utils/misc_utils.py`](../tinker_cookbook/utils/misc_utils.py)

```python
def all_same(values):
    return len(set(values)) == 1
```

Uses set to check if all values are identical. Even tiny differences (1.0 vs 0.99) count as "not all same" → mixed.

## Trajectory Visualization Table

**WandB Artifact**: `trajectories/batch`

**Code**: [`gaia_training/src/trajectory_logging.py`](src/trajectory_logging.py)

### What It Is

A WandB Table logged at each training step containing the full conversation history and metadata for every trajectory in the batch. This allows you to visualize exactly what the model generated during rollouts.

### Table Schema

Each row represents one trajectory (model rollout on one question). With `batch_size=8` and `group_size=8`, you get 64 rows per step.

| Column | Type | Description |
|--------|------|-------------|
| `step_ix` | int | Training step number (0, 1, 2, ...) |
| `batch_ix` | int | Same as step_ix for GAIA (one batch per step) |
| `group_ix` | int | Group/question index (0 to batch_size-1) |
| `traj_ix` | int | Trajectory index within group (0 to group_size-1) |
| `conversation` | str | Full conversation history as JSON string |
| `question` | str | The GAIA question |
| `ground_truth` | str | Correct answer from dataset |
| `model_answer` | str | Answer extracted from model's final response |
| `reward` | float | Total reward (0.01 to 1.0) |
| `correct` | float | 1.0 if model answer matches ground truth, 0.0 otherwise |
| `format` | float | 1.0 if response has "Final Answer: <answer>**" format, 0.0 otherwise |
| `total_tokens` | int | Total tokens used in trajectory (context length) |
| `total_turns` | int | Number of turns (model generations) in trajectory |
| `max_tokens_exceeded` | bool | True if trajectory was terminated due to hitting `max_trajectory_tokens` |
| `max_turns_exceeded` | bool | True if trajectory was terminated due to hitting `max_num_steps` |

### Conversation Format

The `conversation` column contains a JSON array of messages:

```json
[
  {
    "role": "system",
    "content": "You are an expert assistant that solves GAIA benchmark questions..."
  },
  {
    "role": "user",
    "content": "What is 15% of 240?"
  },
  {
    "role": "assistant",
    "content": "I need to calculate this.\n<function_call>{\"name\": \"calculator\", \"args\": {\"expression\": \"240 * 0.15\"}}</function_call>",
    "tool_calls": [...]
  },
  {
    "role": "tool",
    "content": "36.0"
  },
  {
    "role": "system",
    "content": "[Turns remaining: 5]"
  },
  {
    "role": "assistant",
    "content": "Final Answer: 36**"
  }
]
```

### How to Use in WandB UI

#### View Trajectories
1. Go to your run in WandB
2. Click on "Tables" in the sidebar
3. Select `trajectories/batch`
4. Click on any `conversation` cell to view the full JSON

#### Filter Examples

**See all failures:**
```
correct == 0
```

**See trajectories that hit token limit:**
```
max_tokens_exceeded == True
```

**See trajectories that hit turn limit:**
```
max_turns_exceeded == True
```

**See long conversations:**
```
total_tokens > 20000
```

**See trajectories with many turns:**
```
total_turns >= 5
```

**See specific questions:**
```
question contains "France"
```

#### Sort and Analyze

- Sort by `total_tokens` descending to find longest contexts
- Sort by `reward` ascending to find worst failures
- Group by `group_ix` to see all attempts at the same question
- Compare `model_answer` vs `ground_truth` for near-misses

### Example Queries

**Find questions where model got close but failed format:**
```
correct == 0 AND format == 0
```

**Find expensive trajectories (many tokens, many turns):**
```
total_tokens > 25000 OR total_turns >= 6
```

**Find trajectories that succeeded without tools:**
```
correct == 1 AND total_turns == 1
```

### Debugging with Trajectory Table

#### Problem: Low `env/all/reward/total`
1. Filter to `correct == 0`
2. Read the conversations to see what went wrong:
   - Wrong tool usage?
   - Correct reasoning but wrong final answer?
   - Format issues?

#### Problem: High `max_tokens_exceeded` rate
1. Filter to `max_tokens_exceeded == True`
2. Sort by `total_tokens` descending
3. Look at conversations - are tool results too long?
4. Consider increasing `max_trajectory_tokens` or truncating tool results

#### Problem: High `max_turns_exceeded` rate
1. Filter to `max_turns_exceeded == True`
2. Look at conversations - is model stuck in tool loops?
3. Consider increasing `max_num_steps` or improving prompt

#### Problem: Low `by_group/frac_mixed`
1. Group by `group_ix`
2. For each group, check if all `reward` values are the same
3. If so, question might be too easy or too hard
4. Look at conversations to understand why all attempts succeeded/failed

### Code Reference

Trajectory metadata is collected in [`gaia_env.py:252-264`](src/gaia_env.py#L252-L264):

```python
return StepResult(
    reward=total_reward,
    episode_done=True,
    next_observation=tinker.ModelInput.empty(),
    next_stop_condition=self.stop_condition,
    metrics={
        "format": correct_format,
        "correct": correct_answer,
        # Add trajectory metadata for logging
        "past_messages": self.past_messages,
        "question": self.question,
        "ground_truth": str(self.answer),
        "model_answer": model_answer,
        "total_tokens": self.total_tokens_used,
        "total_turns": self.current_step,
        "max_tokens_exceeded": self.max_tokens_exceeded,
        "max_turns_exceeded": self.max_turns_exceeded,
    },
)
```

Table is built in training loop at [`train.py:843-850`](../tinker_cookbook/rl/train.py#L843-L850).

