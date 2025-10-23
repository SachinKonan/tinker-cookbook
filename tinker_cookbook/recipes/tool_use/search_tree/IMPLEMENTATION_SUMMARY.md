# Tree GRPO Implementation Summary

## Overview

This implementation adds **tree-based GRPO** to the tinker-cookbook, where trajectories are organized in a tree structure instead of being sampled independently. This enables:

1. **Token-level advantage computation** using shared prefixes
2. **Better sample efficiency** through branching
3. **Gemini-guided exploration** for alternative reasoning paths

## Architecture

### Core Components

1. **New Types** (`rl/types.py`)
   - `Reference`: Stores branch point (parent trajectory, transition_idx, token_idx)
   - `RootTrajectory(Trajectory)`: Root nodes (no parent)
   - `BranchedTrajectory(Trajectory)`: Child nodes with references to parents
   - `TreeTrajectoryGroup(TrajectoryGroup)`: Compatible with existing training

2. **Environment State Management** (`search_env.py`)
   - `SearchEnv.set_history(messages)`: Clone environment state
   - Sets `past_messages` and `current_num_calls`

3. **Tree Rollout Logic** (`rl/rollouts.py`)
   - `do_tree_group_rollout()`: Queue-based async tree generation
   - **Resource constraints**:
     - Fixed environment pool (group_size envs)
     - Serial Gemini calls (1 at a time)
   - **State machine**:
     - Running trajectory futures
     - Gemini queue (completed trajs waiting for alternatives)
     - Branch queue (alternatives ready to spawn)
     - Free environment pool

4. **Dataset Builders** (`search_tree/tree_dataset.py`)
   - `SearchR1TreeDatasetBuilder`: Adds M, K, D parameters
   - `SearchR1TreeDataset`: Uses TreeProblemGroupBuilder
   - `TreeProblemGroupBuilder`: Prefix-aware reward aggregation (TODO)

5. **Training Script** (`search_tree/train.py`)
   - CLI configuration with tree parameters
   - Integrates with existing RL training infrastructure

## Algorithm Flow

### Initialization
```python
M = 4  # root trajectories
K = 3  # branching factor (generates K-1=2 alternatives)
D = 3  # max depth
target_size = 8  # group_size (total trajectories needed)

envs = await env_builder.make_envs()  # Create group_size envs
```

### Main Loop
```python
while len(completed) < target_size:
    # 1. Check completed trajectory futures
    for future in done(running_traj_futures):
        traj = future.result()
        completed.append(traj)
        free_envs.add(env_idx)

        if depth < D:
            # Pick random branch point
            # Add to gemini_queue

    # 2. Launch Gemini if idle and queue not empty
    if gemini_future is None and gemini_queue:
        parent, messages, branch_idx, tok_idx = gemini_queue.pop()
        gemini_future = gemini.generate_alternatives(...)

    # 3. Check Gemini completion
    if gemini_future.done():
        alternatives = gemini_future.result()
        for alt in alternatives:
            branch_queue.append((alt, parent, messages, ...))
        gemini_future = None

    # 4. Launch children if envs available
    while branch_queue and free_envs:
        alt_text, parent, messages, ... = branch_queue.pop()
        env = envs[free_envs.pop()]

        env.set_history(messages)  # Clone state
        alt_tokens = tokenize(alt_text)

        future = child_rollout(env, policy, alt_tokens, parent)
        running_traj_futures[future] = (env_idx, depth+1, parent)
```

### Key Features
- **Async execution**: Trajectories run in parallel, Gemini serial
- **Resource management**: Fixed env pool, careful tracking
- **Dynamic branching**: Queue-based, adapts to completion order
- **Exit condition**: Stop at target_size completed trajectories

## Files Modified/Created

### Modified Files
1. `tinker_cookbook/rl/types.py` (+60 lines)
   - Added Reference, RootTrajectory, BranchedTrajectory, TreeTrajectoryGroup

2. `tinker_cookbook/rl/rollouts.py` (+295 lines)
   - Added do_tree_group_rollout()

3. `tinker_cookbook/recipes/tool_use/search/search_env.py` (+17 lines)
   - Added set_history() method

### New Files
1. `tinker_cookbook/recipes/tool_use/search_tree/tree_types.py` (old demo, can remove)
2. `tinker_cookbook/recipes/tool_use/search_tree/context_utils.py` (+200 lines)
   - Context extraction for branching

3. `tinker_cookbook/recipes/tool_use/search_tree/gemini_branching.py` (+160 lines)
   - Gemini completer with semaphore (40 concurrent)

4. `tinker_cookbook/recipes/tool_use/search_tree/tree_dataset.py` (+200 lines)
   - SearchR1TreeDatasetBuilder, TreeProblemGroupBuilder

5. `tinker_cookbook/recipes/tool_use/search_tree/train.py` (+170 lines)
   - Training script with tree parameters

6. `tests/test_tree_rollout.py` (+200 lines)
   - Unit test with mocks

## Usage

### Running the Unit Test
```bash
pytest tests/test_tree_rollout.py -v
```

### Running Training (TODO - needs integration)
```bash
uv run python -m tinker_cookbook.recipes.tool_use.search_tree.train \
    tree_m=4 \
    tree_k=3 \
    tree_d=3 \
    batch_size=512 \
    group_size=8 \
    model_name="Qwen/Qwen3-4B-Instruct-2507" \
    chroma_host="0.0.0.0" \
    chroma_port=8000
```

## TODO: Integration with Training

The current implementation creates the tree structure but doesn't yet use it for training. To complete integration:

### 1. Modify `rl/train.py`
Replace `do_group_rollout()` with `do_tree_group_rollout()` when using tree datasets:

```python
if isinstance(dataset, SearchR1TreeDataset):
    # Use tree rollout
    trajectory_groups_P = await asyncio.gather(*[
        do_tree_group_rollout(
            builder, policy, gemini_completer, renderer,
            M=dataset.tree_m, K=dataset.tree_k, D=dataset.tree_d,
            target_size=dataset.group_size
        )
        for builder in env_group_builders_P
    ])
else:
    # Standard rollout
    trajectory_groups_P = await asyncio.gather(*[
        do_group_rollout(builder, policy)
        for builder in env_group_builders_P
    ])
```

### 2. Pass Gemini Completer
Training needs access to GeminiBranchingCompleter:
```python
from tinker_cookbook.recipes.tool_use.search_tree.gemini_branching import (
    GeminiBranchingCompleter
)

gemini_completer = GeminiBranchingCompleter(
    model_name="gemini-2.0-flash-exp",
    temperature=0.9,
)
```

### 3. Pass Renderer
The rollout needs access to the renderer:
```python
renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
```

### 4. Token-Level Advantages (Future Work)
Implement prefix-based advantage computation in `TreeProblemGroupBuilder.compute_group_rewards()`:
```python
def compute_tree_advantages(tree_group: TreeTrajectoryGroup):
    """
    Compute advantages using token-level prefix aggregation.

    For each trajectory at each token position:
    1. Find all trajectories sharing the same prefix
    2. Aggregate rewards across those trajectories
    3. Compute centered advantage
    """
    # TODO: Implement
    pass
```

## Testing Strategy

### Unit Test (Phase 4.5) ✓
- Mocked components
- Simple case: M=2, K=2, target=4
- Verifies:
  - Correct number of roots and branched trajectories
  - Serial Gemini execution
  - Environment cloning
  - Reference structure

### Integration Test (Phase 6 - TODO)
1. Run small tree rollout with real components
2. Verify tree structure is created correctly
3. Check that training can consume TreeTrajectoryGroup
4. Validate advantage computation

### Full Training (TODO)
1. Run full training with tree GRPO
2. Compare to baseline (standard GRPO)
3. Measure sample efficiency improvements
4. Analyze tree structure statistics

## Known Limitations

1. **Environment replay**: Simplified message extraction (doesn't fully capture tool responses)
2. **Advantage computation**: TreeProblemGroupBuilder.compute_group_rewards() not yet implemented
3. **Training integration**: Needs manual modification of train.py to use tree rollout
4. **Single Gemini instance**: Only one Gemini call at a time (by design for resource management)

## Performance Considerations

- **Gemini latency**: Serial calls may be slow (40 requests at ~1s each)
- **Environment pool**: Limited by group_size (fixed resource)
- **Memory**: Stores full trajectory tree in memory
- **Branching depth**: Limited by D parameter to prevent explosion

## Future Enhancements

1. **Adaptive branching**: Branch more from high-reward trajectories
2. **Parallel Gemini**: Allow small batch of concurrent calls (2-3)
3. **Pruning**: Stop branching from low-reward paths early
4. **Visualization**: Generate tree diagrams for analysis
5. **Advantage caching**: Precompute prefix-based advantages

## Summary

This implementation provides a complete tree-based GRPO system with:
- ✅ Type definitions for tree structure
- ✅ Environment state management
- ✅ Queue-based async tree rollout
- ✅ Dataset builders
- ✅ Unit tests
- ⚠️ Training integration (manual modification needed)
- ⚠️ Token-level advantages (placeholder implementation)

**Total lines of code**: ~1,500 lines across 9 files
**Key innovation**: Resource-constrained async tree generation with Gemini branching
