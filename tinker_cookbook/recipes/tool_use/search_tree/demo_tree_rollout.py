"""
Demo implementation of Tree GRPO rollout with queue-based branching.
"""
import asyncio
import logging
import random
from pathlib import Path

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.recipes.tool_use.search.search_env import SearchR1DatasetBuilder
from tinker_cookbook.recipes.tool_use.search.tools import (
    ChromaToolClientConfig,
    EmbeddingConfig,
    RetrievalConfig,
)
from tinker_cookbook.recipes.tool_use.search_tree.context_utils import (
    extract_system_and_user_messages_from_env_metadata,
    get_reward_from_trajectory,
    reconstruct_full_context_up_to_branch,
)
from tinker_cookbook.recipes.tool_use.search_tree.demo_config import TreeGRPODemoConfig
from tinker_cookbook.recipes.tool_use.search_tree.gemini_branching import (
    GeminiBranchingCompleter,
)
from tinker_cookbook.recipes.tool_use.search_tree.tree_types import TrajectoryTree
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.rl.types import EnvGroupBuilder, Trajectory
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)


async def generate_tree_for_problem(
    env_builder: EnvGroupBuilder,
    policy: TinkerTokenCompleter,
    gemini_completer: GeminiBranchingCompleter,
    renderer: renderers.Renderer,
    config: TreeGRPODemoConfig,
) -> TrajectoryTree:
    """
    Generate a trajectory tree for a single problem using queue-based branching.

    Args:
        env_builder: Builder for creating environments
        policy: Policy for generating actions
        gemini_completer: Gemini completer for branching
        renderer: Renderer for token/text conversion
        config: Configuration parameters

    Returns:
        Complete trajectory tree with N leaves
    """
    tree = TrajectoryTree()
    rng = random.Random(config.seed)

    if config.verbose:
        logger.info(
            f"Starting tree generation: M={config.tree_m}, K={config.tree_k}, "
            f"D={config.tree_d}, N={config.tree_n}"
        )

    # ===== Phase 1: Generate M root trajectories =====
    if config.verbose:
        logger.info(f"Generating {config.tree_m} root trajectories...")

    root_envs = await env_builder.make_envs()
    root_envs = root_envs[: config.tree_m]  # Take only M envs

    root_trajectories = await asyncio.gather(
        *[do_single_rollout(policy, env) for env in root_envs]
    )

    # Compute rewards for roots
    root_rewards_and_metrics = await env_builder.compute_group_rewards(root_trajectories)
    root_rewards = [reward for reward, _ in root_rewards_and_metrics]

    # Add roots to tree
    for traj, final_reward in zip(root_trajectories, root_rewards):
        total_reward = get_reward_from_trajectory(traj) + final_reward
        node_id = tree.add_root(traj, total_reward)
        if config.verbose:
            logger.info(
                f"Added root {node_id}: "
                f"{len(traj.transitions)} transitions, reward={total_reward:.2f}"
            )

    # ===== Phase 2: Branch until we have N leaves =====
    if config.verbose:
        logger.info(
            f"Starting branching phase: current leaves={len(tree.leaf_ids)}, "
            f"target={config.tree_n}"
        )

    iteration = 0
    while len(tree.leaf_ids) < config.tree_n:
        iteration += 1

        # Get branchable leaves (depth < D)
        branchable_leaves = tree.get_branchable_leaves(config.tree_d)

        if not branchable_leaves:
            logger.warning(
                f"No more branchable leaves at iteration {iteration}! "
                f"Current leaves: {len(tree.leaf_ids)}, target: {config.tree_n}"
            )
            break

        # Randomly select a leaf to branch
        parent_node = rng.choice(branchable_leaves)

        if config.verbose:
            logger.info(
                f"\nIteration {iteration}: Branching from node {parent_node.node_id} "
                f"(depth={parent_node.depth}, reward={parent_node.final_reward:.2f})"
            )

        # Branch this node
        try:
            new_leaves = await branch_node(
                parent_node=parent_node,
                tree=tree,
                env_builder=env_builder,
                policy=policy,
                gemini_completer=gemini_completer,
                renderer=renderer,
                config=config,
                rng=rng,
            )

            if config.verbose:
                logger.info(
                    f"Created {len(new_leaves)} children. "
                    f"Total leaves: {len(tree.leaf_ids)}/{config.tree_n}"
                )

        except Exception as e:
            logger.error(f"Failed to branch node {parent_node.node_id}: {e}")
            # Mark this node as unbranch able by moving it to max depth
            # (This is a hack to prevent infinite loops on problematic nodes)
            continue

    # ===== Phase 3: Final statistics =====
    stats = tree.get_statistics()
    if config.verbose:
        logger.info("\n" + "=" * 60)
        logger.info("Tree Generation Complete!")
        logger.info(f"Total nodes: {stats['total_nodes']}")
        logger.info(f"Root nodes: {stats['root_nodes']}")
        logger.info(f"Leaf nodes: {stats['leaf_nodes']}")
        logger.info(f"Max depth: {stats['max_depth']}")
        logger.info(f"Avg depth: {stats['avg_depth']:.2f}")
        logger.info(f"Avg branching factor: {stats['avg_branching_factor']:.2f}")
        logger.info("=" * 60)

    return tree


async def branch_node(
    parent_node,
    tree: TrajectoryTree,
    env_builder: EnvGroupBuilder,
    policy: TinkerTokenCompleter,
    gemini_completer: GeminiBranchingCompleter,
    renderer: renderers.Renderer,
    config: TreeGRPODemoConfig,
    rng: random.Random,
) -> list[int]:
    """
    Branch a single node by generating K-1 alternative trajectories.

    Args:
        parent_node: Node to branch from
        tree: The trajectory tree
        env_builder: Environment builder
        policy: Policy for rollouts
        gemini_completer: Gemini for alternatives
        renderer: Renderer
        config: Configuration
        rng: Random number generator

    Returns:
        List of new child node IDs
    """
    parent_traj = parent_node.trajectory

    # Select a random transition (assistant message) to branch from
    if not parent_traj.transitions:
        raise ValueError(f"Node {parent_node.node_id} has no transitions!")

    branch_transition_idx = rng.randint(0, len(parent_traj.transitions) - 1)
    branch_transition = parent_traj.transitions[branch_transition_idx]

    # Select a random token position within that transition
    num_tokens = len(branch_transition.ac.tokens)
    if num_tokens == 0:
        raise ValueError(
            f"Transition {branch_transition_idx} in node {parent_node.node_id} has no tokens!"
        )

    # Pick a position that's not at the very end (want at least 1 token to generate)
    if num_tokens == 1:
        branch_token_idx = 0
    else:
        # Branch somewhere in the first 80% of the message to leave room for alternatives
        max_branch_idx = max(1, int(num_tokens * 0.8))
        branch_token_idx = rng.randint(0, max_branch_idx)

    if config.verbose:
        logger.info(
            f"  Branch point: transition {branch_transition_idx}/{len(parent_traj.transitions)-1}, "
            f"token {branch_token_idx}/{num_tokens}"
        )

    # Reconstruct full context up to branch point
    system_messages, question = extract_system_and_user_messages_from_env_metadata(parent_traj)

    context = reconstruct_full_context_up_to_branch(
        trajectory=parent_traj,
        renderer=renderer,
        branch_transition_idx=branch_transition_idx,
        branch_token_idx=branch_token_idx,
        system_messages=system_messages,
    )

    # Generate K-1 alternatives with Gemini
    if config.verbose:
        logger.info(f"  Calling Gemini to generate {config.tree_k - 1} alternatives...")

    alternatives = await gemini_completer.generate_alternatives(
        context=context,
        parent_reward=parent_node.final_reward,
        k_minus_1=config.tree_k - 1,
    )

    if config.verbose:
        logger.info(f"  Received {len(alternatives)} alternatives from Gemini")

    # Generate child trajectories for each alternative
    child_node_ids = []

    for alt_idx, alternative_text in enumerate(alternatives):
        if config.verbose:
            logger.info(f"  Processing alternative {alt_idx + 1}/{len(alternatives)}...")
            logger.info(f"    Alternative text: {alternative_text[:100]}...")

        try:
            # Create new environment
            envs = await env_builder.make_envs()
            child_env = envs[0]  # Take first env

            # Tokenize the alternative completion
            alternative_tokens = renderer.tokenizer.encode(
                alternative_text, add_special_tokens=False
            )

            # Continue rollout from the branch point
            # TODO: This is simplified - we need to properly replay the environment state
            # For now, we'll just do a fresh rollout (this is a demo limitation)
            child_trajectory = await do_single_rollout(policy, child_env)

            # Compute reward
            child_rewards_and_metrics = await env_builder.compute_group_rewards([child_trajectory])
            child_final_reward = child_rewards_and_metrics[0][0]
            child_total_reward = get_reward_from_trajectory(child_trajectory) + child_final_reward

            # Add to tree
            child_id = tree.add_child(
                parent_id=parent_node.node_id,
                trajectory=child_trajectory,
                final_reward=child_total_reward,
                branch_transition_idx=branch_transition_idx,
                branch_token_idx=branch_token_idx,
            )

            child_node_ids.append(child_id)

            if config.verbose:
                logger.info(
                    f"    Created child {child_id}: "
                    f"{len(child_trajectory.transitions)} transitions, "
                    f"reward={child_total_reward:.2f}"
                )

        except Exception as e:
            logger.error(f"Failed to create child trajectory for alternative {alt_idx}: {e}")
            continue

    return child_node_ids


async def run_demo(config: TreeGRPODemoConfig):
    """
    Run the tree GRPO demo.

    Args:
        config: Demo configuration
    """
    # Setup logging
    log_path = Path(config.log_path)
    log_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO if config.verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting Tree GRPO Demo")
    logger.info(f"Config: {config}")

    # ===== Setup Tinker clients =====
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(
        base_model=config.model_name
    )

    # ===== Setup dataset builder =====
    chroma_tool_config = ChromaToolClientConfig(
        chroma_host=config.chroma_host,
        chroma_port=config.chroma_port,
        chroma_collection_name=config.chroma_collection_name,
        retrieval_config=RetrievalConfig(
            n_results=config.n_results,
            embedding_config=EmbeddingConfig(
                model_name=config.embedding_model_name,
                embedding_dim=config.embedding_dim,
            ),
        ),
    )

    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(
        config.model_name
    )
    tokenizer = get_tokenizer(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    dataset_builder = SearchR1DatasetBuilder(
        batch_size=config.num_problems,
        group_size=config.tree_m,  # Will use M envs per problem
        renderer_name=renderer_name,
        model_name_for_tokenizer=config.model_name,
        chroma_tool_config=chroma_tool_config,
        seed=config.seed,
        max_trajectory_tokens=config.max_trajectory_tokens,
    )

    dataset, _ = await dataset_builder()

    # ===== Setup policy and Gemini =====
    policy = TinkerTokenCompleter(
        sampling_client=sampling_client,
        max_tokens=config.max_tokens,
    )

    gemini_completer = GeminiBranchingCompleter(
        model_name=config.gemini_model,
        temperature=config.gemini_temperature,
        top_p=config.gemini_top_p,
        max_output_tokens=config.gemini_max_output_tokens,
    )

    # ===== Generate trees for each problem =====
    env_builders = dataset.get_batch(0)  # Get first batch

    trees = []
    for problem_idx, env_builder in enumerate(env_builders):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing problem {problem_idx + 1}/{len(env_builders)}")
        logger.info(f"{'='*60}")

        tree = await generate_tree_for_problem(
            env_builder=env_builder,
            policy=policy,
            gemini_completer=gemini_completer,
            renderer=renderer,
            config=config,
        )

        trees.append(tree)

        # Save tree if requested
        if config.save_trees:
            tree_path = log_path / f"tree_problem_{problem_idx}.json"
            tree.save_to_file(tree_path)
            logger.info(f"Saved tree to {tree_path}")

    logger.info("\n" + "=" * 60)
    logger.info("Demo Complete!")
    logger.info(f"Generated {len(trees)} trees")
    logger.info(f"Output directory: {log_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    import chz

    config = chz.entrypoint(TreeGRPODemoConfig)
    asyncio.run(run_demo(config))
