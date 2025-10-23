import asyncio
import logging
import random
from typing import Sequence

from tinker_cookbook import renderers
from tinker_cookbook.completers import TokenCompleter
from tinker_cookbook.rl.types import (
    BranchedTrajectory,
    Env,
    EnvGroupBuilder,
    Reference,
    RootTrajectory,
    Trajectory,
    TrajectoryGroup,
    TreeTrajectoryGroup,
    Transition,
)

logger = logging.getLogger(__name__)


async def do_single_rollout(policy: TokenCompleter, env: Env) -> Trajectory:
    transitions = []
    ob, stop_condition = await env.initial_observation()
    while True:
        ac_with_logprobs = await policy(ob, stop_condition)
        step_result = await env.step(ac_with_logprobs.tokens)
        transition = Transition(
            ob=ob,
            ac=ac_with_logprobs,
            reward=step_result.reward,
            episode_done=step_result.episode_done,
            metrics=step_result.metrics,
        )
        transitions.append(transition)
        ob = step_result.next_observation
        stop_condition = step_result.next_stop_condition
        if step_result.episode_done:
            break
    return Trajectory(transitions=transitions, final_ob=ob)


async def do_group_rollout(
    env_group_builder: EnvGroupBuilder, policy: TokenCompleter
) -> TrajectoryGroup:
    envs_G: Sequence[Env] = await env_group_builder.make_envs()
    trajectories_G = await asyncio.gather(*[do_single_rollout(policy, env) for env in envs_G])
    rewards_and_metrics_G = await env_group_builder.compute_group_rewards(trajectories_G)
    rewards_G, metrics_G = zip(*rewards_and_metrics_G, strict=True)
    return TrajectoryGroup(trajectories_G, list(rewards_G), list(metrics_G))


async def do_tree_group_rollout(
    env_group_builder: EnvGroupBuilder,
    policy: TokenCompleter,
    gemini_completer,  # GeminiBranchingCompleter - avoid circular import
    renderer: renderers.Renderer,
    M: int,
    K: int,
    D: int,
    target_size: int,
    rng: random.Random | None = None,
) -> TreeTrajectoryGroup:
    """
    Generate a tree of trajectories using queue-based branching.

    Args:
        env_group_builder: Builder for creating environments
        policy: Policy for generating actions
        gemini_completer: Gemini completer for generating K-1 alternatives
        renderer: Renderer for token/text conversion
        M: Number of root trajectories
        K: Branching factor (generates K-1 alternatives per branch)
        D: Maximum depth (max_num_calls limit)
        target_size: Target number of completed trajectories (typically group_size)
        rng: Random number generator (for reproducibility)

    Returns:
        TreeTrajectoryGroup with target_size trajectories
    """
    if rng is None:
        rng = random.Random()

    # Import here to avoid circular dependency
    from tinker_cookbook.recipes.tool_use.search_tree.context_utils import (
        extract_messages_up_to_branch,
    )

    logger.info(
        f"Starting tree rollout: M={M}, K={K}, D={D}, target={target_size}"
    )

    # Create environment pool
    envs = await env_group_builder.make_envs()
    if len(envs) < target_size:
        raise ValueError(
            f"Need at least {target_size} envs, got {len(envs)}"
        )

    # State tracking
    running_traj_futures: dict[asyncio.Future, tuple[int, int, Trajectory | None]] = {}
    # Future -> (env_idx, depth, parent_trajectory)

    gemini_future: asyncio.Future | None = None
    gemini_metadata: tuple[Trajectory, list, int, int] | None = None
    # (parent_traj, messages_up_to_branch, branch_transition_idx, branch_token_idx)

    gemini_queue: list[tuple[Trajectory, list, int, int]] = []
    # Queue of (parent_traj, messages, branch_transition_idx, branch_token_idx)

    branch_queue: list[tuple[str, Trajectory, list, int, int, int]] = []
    # Queue of (alt_text, parent_traj, messages, branch_trans_idx, branch_tok_idx, depth)

    free_env_indices: set[int] = set(range(len(envs)))
    completed_trajectories: list[Trajectory] = []
    completed_rewards: list[float] = []
    completed_metrics: list[dict] = []

    # Launch M root trajectories
    logger.info(f"Launching {M} root trajectories...")
    for i in range(min(M, len(envs))):
        env_idx = free_env_indices.pop()
        env = envs[env_idx]
        future = asyncio.create_task(do_single_rollout(policy, env))
        running_traj_futures[future] = (env_idx, 0, None)  # depth=0, no parent

    # Main loop
    iteration = 0
    while len(completed_trajectories) < target_size:
        iteration += 1
        if iteration % 10 == 0:
            logger.info(
                f"Iteration {iteration}: completed={len(completed_trajectories)}/{target_size}, "
                f"running={len(running_traj_futures)}, gemini_queue={len(gemini_queue)}, "
                f"branch_queue={len(branch_queue)}, free_envs={len(free_env_indices)}"
            )

        # 1. Check completed trajectory futures
        done_futures = [f for f in running_traj_futures if f.done()]

        for future in done_futures:
            env_idx, depth, parent_traj = running_traj_futures.pop(future)

            try:
                traj = future.result()
            except Exception as e:
                logger.error(f"Trajectory failed: {e}")
                free_env_indices.add(env_idx)
                continue

            # Free the environment
            free_env_indices.add(env_idx)

            # Wrap in appropriate type
            if parent_traj is None:
                # Root trajectory
                wrapped_traj = RootTrajectory(
                    transitions=traj.transitions,
                    final_ob=traj.final_ob,
                )
            else:
                # Branched trajectory - need to add reference
                # We'll add the reference when we have full info
                wrapped_traj = traj  # Will wrap later with reference

            completed_trajectories.append(wrapped_traj)
            completed_rewards.append(0.0)  # Will compute later
            completed_metrics.append({})

            logger.info(
                f"Trajectory completed: depth={depth}, transitions={len(traj.transitions)}"
            )

            # Check if we can branch from this trajectory
            if depth < D and len(traj.transitions) > 0:
                # Pick random branch point
                branch_transition_idx = rng.randint(0, len(traj.transitions) - 1)
                branch_transition = traj.transitions[branch_transition_idx]

                if len(branch_transition.ac.tokens) > 0:
                    # Pick random token in first 80% of the transition
                    num_tokens = len(branch_transition.ac.tokens)
                    max_tok_idx = max(1, int(num_tokens * 0.8))
                    branch_token_idx = rng.randint(0, max_tok_idx)

                    # Extract messages for environment cloning
                    messages_up_to_branch = extract_messages_up_to_branch(
                        wrapped_traj,
                        renderer,
                        branch_transition_idx,
                        branch_token_idx,
                    )

                    # Add to Gemini queue
                    gemini_queue.append((
                        wrapped_traj,
                        messages_up_to_branch,
                        branch_transition_idx,
                        branch_token_idx,
                    ))
                    logger.info(f"Queued for Gemini branching: depth={depth}")

        # 2. Launch Gemini if available and queue not empty
        if gemini_future is None and gemini_queue:
            parent_traj, messages, branch_trans_idx, branch_tok_idx = gemini_queue.pop(0)
            gemini_metadata = (parent_traj, messages, branch_trans_idx, branch_tok_idx)

            # Get context for Gemini
            from tinker_cookbook.recipes.tool_use.search_tree.context_utils import (
                reconstruct_full_context_up_to_branch,
                extract_system_and_user_messages_from_env_metadata,
                get_reward_from_trajectory,
            )

            system_messages, _ = extract_system_and_user_messages_from_env_metadata(parent_traj)
            context = reconstruct_full_context_up_to_branch(
                parent_traj,
                renderer,
                branch_trans_idx,
                branch_tok_idx,
                system_messages,
            )
            parent_reward = get_reward_from_trajectory(parent_traj)

            logger.info(f"Launching Gemini to generate {K-1} alternatives...")
            gemini_future = asyncio.create_task(
                gemini_completer.generate_alternatives(context, parent_reward, K - 1)
            )

        # 3. Check Gemini completion
        if gemini_future is not None and gemini_future.done():
            try:
                alternatives = gemini_future.result()
                logger.info(f"Gemini completed: {len(alternatives)} alternatives")

                # Add to branch queue
                if gemini_metadata is not None:
                    parent_traj, messages, branch_trans_idx, branch_tok_idx = gemini_metadata
                    parent_depth = (
                        len(parent_traj.references)
                        if isinstance(parent_traj, BranchedTrajectory)
                        else 0
                    )

                    for alt_text in alternatives:
                        branch_queue.append((
                            alt_text,
                            parent_traj,
                            messages,
                            branch_trans_idx,
                            branch_tok_idx,
                            parent_depth + 1,
                        ))

            except Exception as e:
                logger.error(f"Gemini failed: {e}")

            gemini_future = None
            gemini_metadata = None

        # 4. Launch child trajectories if envs available
        while branch_queue and free_env_indices:
            alt_text, parent_traj, messages, branch_trans_idx, branch_tok_idx, depth = branch_queue.pop(0)

            env_idx = free_env_indices.pop()
            env = envs[env_idx]

            # Clone environment state
            env.set_history(messages)

            # Tokenize Gemini alternative
            alt_tokens = renderer.tokenizer.encode(alt_text, add_special_tokens=False)

            # Create child rollout task
            async def child_rollout(env, policy, alt_tokens):
                # First step with Gemini alternative
                step_result = await env.step(alt_tokens)

                # Continue rollout if not done
                if not step_result.episode_done:
                    ob = step_result.next_observation
                    stop_condition = step_result.next_stop_condition

                    transitions = []
                    while True:
                        ac_with_logprobs = await policy(ob, stop_condition)
                        step_result = await env.step(ac_with_logprobs.tokens)
                        transition = Transition(
                            ob=ob,
                            ac=ac_with_logprobs,
                            reward=step_result.reward,
                            episode_done=step_result.episode_done,
                            metrics=step_result.metrics,
                        )
                        transitions.append(transition)
                        ob = step_result.next_observation
                        stop_condition = step_result.next_stop_condition
                        if step_result.episode_done:
                            break

                    # Create branched trajectory with reference
                    reference = Reference(
                        source_trajectory=parent_traj,
                        transition_idx=branch_trans_idx,
                        token_idx=branch_tok_idx,
                    )
                    return BranchedTrajectory(
                        transitions=transitions,
                        final_ob=ob,
                        references=[reference],
                    )
                else:
                    # Episode ended immediately
                    reference = Reference(
                        source_trajectory=parent_traj,
                        transition_idx=branch_trans_idx,
                        token_idx=branch_tok_idx,
                    )
                    return BranchedTrajectory(
                        transitions=[],
                        final_ob=step_result.next_observation,
                        references=[reference],
                    )

            future = asyncio.create_task(child_rollout(env, policy, alt_tokens))
            running_traj_futures[future] = (env_idx, depth, parent_traj)

            logger.info(f"Launched child trajectory: depth={depth}")

        # Small sleep to yield to event loop
        await asyncio.sleep(0.01)

    # Compute final rewards
    rewards_and_metrics = await env_group_builder.compute_group_rewards(completed_trajectories)
    final_rewards = [r for r, _ in rewards_and_metrics]
    final_metrics = [m for _, m in rewards_and_metrics]

    logger.info(
        f"Tree rollout complete: {len(completed_trajectories)} trajectories, "
        f"{sum(isinstance(t, RootTrajectory) for t in completed_trajectories)} roots"
    )

    return TreeTrajectoryGroup(
        trajectories_G=completed_trajectories,
        final_rewards_G=final_rewards,
        metrics_G=final_metrics,
    )
