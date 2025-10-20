# GAIA RL Training

This module implements RL training for the GAIA benchmark using Tinker's native tool use infrastructure.

## Architecture

The implementation follows the pattern from `tinker_cookbook/recipes/tool_use/search/` and uses:

1. **Native Tinker Tool Use** (NOT LangChain) - captures logprobs automatically during sampling
2. **ProblemEnv** - Multi-step environment with tool execution
3. **RLDatasetBuilder** - Integration with existing `train.rl_loop()` infrastructure
4. **Ray Parallelization** - Handled automatically via `ProblemGroupBuilder`
5. **GRPO** - Group Relative Policy Optimization with group-based advantages

## Components

### 1. GAIAToolClient (`src/gaia_tools.py`)

Implements three tools for GAIA tasks:

- **web_search**: Search the web using DuckDuckGo
- **calculator**: Perform mathematical calculations (supports +, -, *, /, parentheses)
- **fetch_webpage**: Fetch and extract text content from a webpage

Tools are invoked via XML format:
```xml
<function_call>{"name": "calculator", "args": {"expression": "2+2"}}</function_call>
```

### 2. GAIAEnv (`src/gaia_env.py`)

Environment that handles multi-step tool-using trajectories:

- **Tool calls**: Execute tool, return next observation, continue episode (`episode_done=False`)
- **Final answer**: Compute reward, end episode (`episode_done=True`)

**Reward Formula**:
```
reward = 0.01 * has_final_answer + 0.99 * is_correct
```

Where:
- `has_final_answer`: 1.0 if response contains "Final Answer: .*", 0.0 otherwise
- `is_correct`: 1.0 if extracted answer matches ground truth, 0.0 otherwise

**System Prompt**: `GAIA_SYSTEM_PROMPT` instructs model to use tools and provide answers in "Final Answer: <answer>" format.

### 3. GAIADatasetBuilder (`src/gaia_dataset_builder.py`)

Dataset builder that:
- Loads GAIA data from JSON
- Creates batches of `ProblemGroupBuilder` instances
- Each group contains `group_size` parallel environments for GRPO
- Integrates with existing `train.rl_loop()` infrastructure

### 4. Training Script (`train_gaia.py`)

Main training entry point that:
- Uses `train.rl_loop()` from `tinker_cookbook.rl.train`
- Configures dataset builder with GAIA data
- Sets up LoRA training with specified hyperparameters
- Handles logging and checkpointing

## Configuration

Key hyperparameters (in `train_gaia.py`):

```python
model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
lora_rank: int = 32
learning_rate: float = 1e-5
batch_size: int = 8  # Number of questions per batch
group_size: int = 2  # GRPO group size
max_trajectory_tokens: int = 32 * 1024
max_num_steps: int = 7  # Agent max steps
```

## Usage

### Basic Training

```bash
cd gaia_training
uv run python train_gaia.py
```

### With Custom Config

```bash
uv run python train_gaia.py \
  --model_name "Qwen/Qwen3-30B-A3B-Instruct-2507" \
  --batch_size 16 \
  --group_size 4 \
  --learning_rate 5e-6 \
  --log_path "/path/to/logs"
```

### Testing Tools

```bash
# Test tool execution
uv run python test_tools.py

# Test reward computation
uv run python test_rl_rewards.py
```

## Data Format

GAIA data should be in JSON format with the following structure:

```json
[
  {
    "Question": "What is the capital of France?",
    "Final answer": "Paris"
  },
  ...
]
```

Place the data file at: `data/inputs/gaia_data.json`

## How It Works

### Training Loop

1. **Dataset Building**: `GAIADatasetBuilder` loads GAIA questions and creates environment groups
2. **Trajectory Generation**: For each question, `group_size` parallel environments generate trajectories
   - Model generates responses
   - Tools are executed when model outputs `<function_call>` tags
   - Episode ends when model outputs final answer (or max steps reached)
3. **Reward Computation**: Each trajectory gets reward based on correctness and format
4. **Advantage Computation**: Within each group, advantages are computed as `reward - mean_group_reward`
5. **Policy Update**: Model is updated using importance sampling loss with computed advantages

### Example Trajectory

```
User: What is 15% of 240?
Assistant: I need to calculate this.
<function_call>{"name": "calculator", "args": {"expression": "240 * 0.15"}}</function_call>
Tool: 36.0
Assistant: Final Answer: 36
```

Reward: 0.01 * 1.0 + 0.99 * 1.0 = 1.0 (correct answer with proper format)

## File Structure

```
gaia_training/
├── src/
│   ├── gaia_tools.py          # Tool client with web_search, calculator, fetch_webpage
│   ├── gaia_env.py             # Environment with tool handling and rewards
│   └── gaia_dataset_builder.py # Dataset builder for RL training
├── data/
│   └── inputs/
│       └── gaia_data.json      # GAIA benchmark data
├── train_gaia.py               # Main training script
├── test_tools.py               # Test tool execution
├── test_rl_rewards.py          # Test reward computation
└── README.md                   # This file
```

