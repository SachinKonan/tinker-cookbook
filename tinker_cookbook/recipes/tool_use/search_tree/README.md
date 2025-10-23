# Tree GRPO Demo

This package implements a novel **Tree-based GRPO** algorithm for reinforcement learning with trajectory branching.

## Overview

Instead of generating independent trajectories for GRPO (Group Relative Policy Optimization), this implementation creates a **tree structure** where:

1. **M root trajectories** are generated independently
2. Trajectories **branch at random token positions** within assistant messages
3. **Gemini-2.5-Pro** generates K-1 alternative completions at each branch point
4. Process continues until **N total leaf trajectories** are collected
5. **Advantages are computed** with token-level prefix aggregation across the tree

### Key Innovation: Token-Level Advantage Aggregation

Unlike standard GRPO which centers rewards across independent trajectories, Tree GRPO aggregates rewards based on **shared token prefixes**:

- For each token position, find all trajectories sharing the exact same prefix
- Compute advantages by centering rewards across those trajectories
- This creates tighter advantage estimates for shared reasoning paths

## Architecture

```
tinker_cookbook/recipes/tool_use/search_tree/
├── __init__.py
├── README.md                 # This file
├── tree_types.py            # TreeNode and TrajectoryTree data structures
├── context_utils.py         # Context reconstruction from trajectories
├── gemini_branching.py      # Gemini-based branching completer
├── demo_config.py           # Configuration parameters
└── demo_tree_rollout.py     # Main demo script
```

## Prerequisites

1. **Chroma server** running with Wikipedia embeddings:
   ```bash
   bash /n/fs/vision-mix/sk7524/run_chroma.sh
   ```

2. **Environment variables** set in `.env`:
   ```bash
   GCP_VERTEXAI_PROJECT_NUMBER=your-project-number
   GCP_VERTEXAI_REGION=us-central1
   GEMINI_API_KEY=your-api-key
   TINKER_API_KEY=your-tinker-key
   ```

3. **Dependencies** installed:
   ```bash
   uv sync
   ```

## Running the Demo

### Quick Start (Small Test)

```bash
uv run python -m tinker_cookbook.recipes.tool_use.search_tree.demo_tree_rollout \
    tree_m=2 \
    tree_k=2 \
    tree_d=2 \
    tree_n=8 \
    num_problems=1 \
    log_path=/tmp/tree_grpo_test
```

This will:
- Start with **2 root** trajectories
- Branch with **K=2** (1 alternative per branch)
- Maximum **depth=2**
- Target **8 leaf** trajectories
- Process **1 problem**

### Full Demo (Default Parameters)

```bash
uv run python -m tinker_cookbook.recipes.tool_use.search_tree.demo_tree_rollout
```

Default settings:
- M = 4 roots
- K = 3 (2 alternatives per branch)
- D = 3 max depth
- N = 32 target leaves
- 4 problems

## Configuration Parameters

### Tree Hyperparameters

- `tree_m`: Number of root trajectories (default: 4)
- `tree_k`: Branching factor - generates K-1 alternatives (default: 3)
- `tree_d`: Maximum tree depth (default: 3)
- `tree_n`: Target number of leaf trajectories (default: 32)

### Model Parameters

- `model_name`: Base model (default: "Qwen/Qwen3-4B-Instruct-2507")
- `lora_rank`: LoRA rank (default: 32)
- `max_tokens`: Max tokens per step (default: 1024)

### Gemini Parameters

- `gemini_model`: Model for branching (default: "gemini-2.0-flash-exp")
- `gemini_temperature`: Sampling temperature (default: 0.9)
- `gemini_top_p`: Nucleus sampling (default: 0.95)
- `gemini_max_output_tokens`: Max output length (default: 2048)

### Environment Parameters

- `chroma_host`: Chroma server host (default: "localhost")
- `chroma_port`: Chroma server port (default: 8000)
- `chroma_collection_name`: Collection name (default: "wiki_embeddings")
- `n_results`: Search results to retrieve (default: 3)

## Output

The demo saves:

1. **Tree JSON files** in `{log_path}/tree_problem_*.json`:
   ```json
   {
     "nodes": [
       {
         "id": 0,
         "depth": 0,
         "is_leaf": false,
         "is_root": true,
         "final_reward": 0.85,
         "num_transitions": 4,
         "total_tokens": 287
       },
       ...
     ],
     "edges": [
       {
         "source": 0,
         "target": 1,
         "branch_transition_idx": 2,
         "branch_token_idx": 45
       },
       ...
     ],
     "statistics": {
       "total_nodes": 47,
       "root_nodes": 4,
       "leaf_nodes": 32,
       "max_depth": 3,
       "avg_depth": 2.1,
       "avg_branching_factor": 2.8
     }
   }
   ```

2. **Console output** showing tree generation progress

## Algorithm Details

### 1. Root Generation

```python
# Generate M independent root trajectories
root_trajectories = await generate_roots(env_builder, policy, M)
```

### 2. Branching Process

For each branch:
1. Select a random leaf node at depth < D
2. Pick a random transition (assistant message)
3. Pick a random token position within that transition
4. Reconstruct full context up to that point
5. Ask Gemini for K-1 alternative completions
6. For each alternative:
   - Create new environment
   - Continue rollout from branch point
   - Add as child node in tree

### 3. Queue-Based Management

```python
while len(tree.leaf_ids) < N:
    branchable_leaves = tree.get_branchable_leaves(max_depth=D)
    parent_node = random.choice(branchable_leaves)
    children = await branch_node(parent_node, ...)
```

### 4. Token-Level Advantages (Future Work)

```python
for node in tree.nodes:
    for token_idx in range(len(node.get_token_sequence())):
        prefix = node.get_token_sequence()[:token_idx+1]
        siblings = tree.find_nodes_with_prefix(prefix)
        advantage[token_idx] = center_rewards(siblings)
```

## Known Limitations

This is a **demonstration** of the tree generation mechanism. Current limitations:

1. **Environment replay** is simplified - doesn't fully restore intermediate state
2. **No training integration** - just generates trees, doesn't train policy
3. **No advantage computation** - tree structure is created but not used for training yet

## Next Steps

To integrate into actual training:

1. Implement proper environment state replay
2. Add token-level advantage computation with prefix aggregation
3. Integrate with main RL training loop (`tinker_cookbook/rl/train.py`)
4. Add tree-specific metrics and logging

## Troubleshooting

**Chroma connection errors**:
```bash
# Check Chroma is running
curl http://localhost:8000/api/v1/heartbeat

# Restart if needed
bash /n/fs/vision-mix/sk7524/run_chroma.sh
```

**Gemini API errors**:
```bash
# Check environment variables are set
echo $GCP_VERTEXAI_PROJECT_NUMBER
echo $GCP_VERTEXAI_REGION
```

**Out of memory**:
- Reduce `tree_n` (target leaves)
- Reduce `tree_d` (max depth)
- Reduce `num_problems`

## References

- Standard GRPO: Group Relative Policy Optimization
- Search-R1: Tool use with Wikipedia search
- Tree-based exploration: Novel contribution of this implementation
