import asyncio
import logging
import random
import time
from typing import Sequence

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.completers import TokenCompleter
from tinker_cookbook.rl.data_processing import compute_per_token_advantages_branched
from tinker_cookbook.rl.types import (
    BranchedTrajectory,
    Env,
    EnvGroupBuilder,
    Reference,
    RejectedTrajectory,
    RootTrajectory,
    Trajectory,
    TrajectoryGroup,
    TreeTrajectoryGroup,
    Transition,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Helper functions for token-level branching
# ============================================================================

def build_partial_continuation_prompt(
    past_messages: list,
    assistant_idx: int,
    partial_content: str,
    renderer: renderers.Renderer,
) -> tinker.ModelInput:
    """Build a prompt that continues from a partial assistant message.

    Unlike build_generation_prompt, this doesn't close the partial message
    with <|im_end|> or add a new assistant prefix. The model will continue
    generating from where the partial content left off.

    This ensures the model sees it's CONTINUING an existing assistant message
    rather than starting a new one.

    Args:
        past_messages: Message history before the partial assistant
        assistant_idx: Index where the partial assistant message should be
        partial_content: The partial text to continue from
        renderer: Renderer for encoding tokens

    Returns:
        ModelInput ready for continuation generation
    """
    tokens = []

    # Render all messages before the partial assistant (complete messages with <|im_end|>)
    for idx, message in enumerate(past_messages[:assistant_idx]):
        ob_part, action_part, _ = renderer._render_message(idx, message)
        tokens.extend(ob_part)
        tokens.extend(action_part)

    # For the partial assistant message:
    # 1. Add the assistant prefix (<|im_start|>assistant\n)
    partial_message = {"role": "assistant", "content": ""}
    ob_part, _, _ = renderer._render_message(assistant_idx, partial_message)
    tokens.extend(ob_part)

    # 2. Add the partial content WITHOUT <|im_end|>
    # (action_part would include "<partial_content><|im_end|>", but we only want the content)
    partial_content_tokens = renderer.tokenizer.encode(partial_content, add_special_tokens=False)
    tokens.extend(partial_content_tokens)

    return tinker.ModelInput.from_ints(tokens)


def build_partial_prefix(
    past_messages: list,
    assistant_idx: int,
    token_idx: int,
    transition: Transition,
    renderer: renderers.Renderer,
) -> tuple[list, tinker.ModelInput, list[int]]:
    """Build a partial prefix for token-level branching.

    Creates a modified message history where the assistant message at assistant_idx
    only contains the first token_idx tokens, and builds an observation that allows
    the model to continue generating from that point.

    Args:
        past_messages: Full message history
        assistant_idx: Index of assistant message to branch from
        token_idx: Token position within that message (should be < 50% of total)
        transition: The Transition object for that assistant message
        renderer: Renderer for decoding tokens

    Returns:
        tuple: (modified_past_messages, observation, partial_tokens)
    """
    # Get the first token_idx tokens from the assistant message
    partial_tokens = transition.ac.tokens[:token_idx]

    # Decode to text
    partial_text = renderer.tokenizer.decode(partial_tokens)

    # Create a partial assistant message (for debug/display purposes)
    partial_message = {
        "role": "assistant",
        "content": partial_text
    }

    # Build modified history: everything before + partial message
    modified_past_messages = past_messages[:assistant_idx] + [partial_message]

    # Build observation using our custom continuation prompt builder
    # This ensures the model continues the partial message instead of starting a new one
    observation = build_partial_continuation_prompt(
        past_messages, assistant_idx, partial_text, renderer
    )

    return modified_past_messages, observation, partial_tokens


async def do_single_rollout(policy: TokenCompleter, env: Env) -> Trajectory | RejectedTrajectory:
    transitions = []
    ob, stop_condition = await env.initial_observation()
    while True:
        # Time the policy call
        t_start = time.time()
        ac_with_logprobs = await policy(ob, stop_condition)
        policy_time = time.time() - t_start

        step_result = await env.step(ac_with_logprobs.tokens)

        # Check if trajectory should be rejected
        if step_result.rejectable_result:
            logger.warning("Trajectory rejected: rejectable_result=True")
            return RejectedTrajectory(reason="rejectable_result=True")

        # Store policy timing in transition metrics
        transition_metrics = step_result.metrics.copy() if step_result.metrics else {}
        transition_metrics['policy_time'] = policy_time

        transition = Transition(
            ob=ob,
            ac=ac_with_logprobs,
            reward=step_result.reward,
            episode_done=step_result.episode_done,
            metrics=transition_metrics,
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
    all_results = await asyncio.gather(*[do_single_rollout(policy, env) for env in envs_G])

    # Filter out rejected trajectories
    trajectories_G = [t for t in all_results if not isinstance(t, RejectedTrajectory)]
    num_rejected = len(all_results) - len(trajectories_G)
    if num_rejected > 0:
        logger.warning(f"Filtered out {num_rejected} rejected trajectories from group of {len(all_results)}")

    rewards_and_metrics_G = await env_group_builder.compute_group_rewards(trajectories_G)
    rewards_G, metrics_G = zip(*rewards_and_metrics_G, strict=True)
    return TrajectoryGroup(trajectories_G, list(rewards_G), list(metrics_G))


async def do_branched_group_rollout(
    env_group_builder: EnvGroupBuilder,
    policy: TokenCompleter,
    renderer: renderers.Renderer,
    target_size: int,
    num_branches: int = 2,
    rng: random.Random | None = None,
) -> TrajectoryGroup:
    """
    Generate a group of trajectories using tree-based post-completion branching.

    Starting with src_trajectories root trajectories (from env_group_builder.make_envs()),
    branch from completed trajectories until reaching target_size total trajectories.

    Branching strategy: For each completed trajectory, randomly select a past assistant
    message (excluding the last) and branch from a random token position < 50% within
    that message. This creates diverse trajectories from fewer root rollouts.

    Args:
        env_group_builder: Builder with src_trajectories environments
        policy: Policy for generating actions
        renderer: Renderer for token/text conversion
        target_size: Target number of trajectories (typically group_size)
        num_branches: Number of children to spawn per completed trajectory
        rng: Random number generator (for reproducibility)

    Returns:
        TrajectoryGroup with target_size trajectories
    """
    if rng is None:
        rng = random.Random()

    logger.info(
        f"Starting branched rollout: src_trajectories={env_group_builder.num_envs}, "
        f"target={target_size}, branches={num_branches}"
    )

    # Create source environments
    envs = await env_group_builder.make_envs()
    src_trajectories = len(envs)

    if target_size < src_trajectories:
        raise ValueError(
            f"target_size ({target_size}) must be >= src_trajectories ({src_trajectories})"
        )

    # Track all completed trajectories
    all_trajectories = []
    all_envs = []

    # Tracking
    total_created = 0
    next_id = src_trajectories
    active_tasks = {}  # task -> (env, root_trajectory_index)

    # Launch root trajectories
    logger.info(f"Launching {src_trajectories} root trajectories...")
    for i, env in enumerate(envs):
        task = asyncio.create_task(do_single_rollout(policy, env))
        active_tasks[task] = (env, i)
        total_created += 1

    # Process trajectories as they complete, spawning branches
    while active_tasks:
        # Wait for next completion
        done, pending = await asyncio.wait(active_tasks.keys(), return_when=asyncio.FIRST_COMPLETED)

        for task in done:
            env, root_idx = active_tasks.pop(task)
            trajectory = await task

            # Check if trajectory was rejected
            if isinstance(trajectory, RejectedTrajectory):
                logger.warning(f"Trajectory rejected: {trajectory.reason}")
                # Try to spawn replacement branch from existing completed trajectory
                if all_trajectories and total_created < target_size:
                    # Pick a random completed trajectory to branch from
                    parent_traj = rng.choice(all_trajectories)
                    parent_env = all_envs[all_trajectories.index(parent_traj)]
                    branch_task = asyncio.create_task(
                        _create_and_run_branch(
                            parent_traj, parent_env, env_group_builder, policy, renderer, rng
                        )
                    )
                    active_tasks[branch_task] = (parent_env, root_idx)
                    total_created += 1
                continue  # Skip storing rejected trajectory

            # Store completed trajectory
            all_trajectories.append(trajectory)
            all_envs.append(env)

            logger.info(
                f"Trajectory {len(all_trajectories)} complete: "
                f"{len(trajectory.transitions)} transitions"
            )

            # Spawn branches if we haven't reached target
            if total_created < target_size:
                # Determine how many branches to spawn
                branches_to_spawn = min(num_branches, target_size - total_created)

                logger.info(f"Spawning {branches_to_spawn} branches from trajectory {len(all_trajectories)-1}...")

                for _ in range(branches_to_spawn):
                    # Create branch task
                    branch_task = asyncio.create_task(
                        _create_and_run_branch(
                            trajectory, env, env_group_builder, policy, renderer, rng
                        )
                    )
                    active_tasks[branch_task] = (env, root_idx)  # Track same root
                    total_created += 1
                    next_id += 1

    logger.info(f"Branched rollout complete: {len(all_trajectories)} trajectories")

    # Compute group rewards
    rewards_and_metrics_G = await env_group_builder.compute_group_rewards(all_trajectories)
    rewards_G, metrics_G = zip(*rewards_and_metrics_G, strict=True)

    # Create trajectory group
    traj_group = TrajectoryGroup(all_trajectories, list(rewards_G), list(metrics_G))

    # Compute per-token advantages based on prefix sharing
    compute_per_token_advantages_branched(traj_group)

    return traj_group


async def _create_and_run_branch(
    parent_trajectory: Trajectory,
    parent_env: Env,
    env_group_builder: EnvGroupBuilder,
    policy: TokenCompleter,
    renderer: renderers.Renderer,
    rng: random.Random,
) -> Trajectory | RejectedTrajectory:
    """Create and run a branched trajectory from a parent trajectory.

    Args:
        parent_trajectory: Completed parent trajectory to branch from
        parent_env: Parent environment (for accessing past_messages)
        env_group_builder: Builder for creating new environments
        policy: Policy for generating actions
        renderer: Renderer for token/text conversion
        rng: Random number generator

    Returns:
        New branched trajectory
    """
    # Find all assistant messages with transitions in parent
    assistant_indices = [
        i for i, msg in enumerate(parent_env.past_messages)
        if msg.get("role") == "assistant"
    ]

    # Map assistant indices to transitions
    assistant_to_transition = {}
    if len(assistant_indices) >= len(parent_trajectory.transitions):
        for i in range(len(parent_trajectory.transitions)):
            msg_idx = assistant_indices[-(len(parent_trajectory.transitions) - i)]
            assistant_to_transition[msg_idx] = i

    # Only pick from assistant messages that have transitions (excluding last)
    branchable_assistants = list(assistant_to_transition.keys())
    if len(branchable_assistants) > 1:
        branchable_assistants = branchable_assistants # allow the last message to be modified now!
        #branchable_assistants = branchable_assistants[:-1]  # Exclude last

    if len(branchable_assistants) == 0:
        logger.warning("No branchable assistant messages, running from initial state")
        # Fallback: just run a new trajectory from scratch
        new_env = env_group_builder.env_thunk()
        return await do_single_rollout(policy, new_env)

    # Randomly pick one assistant message
    assistant_idx = rng.choice(branchable_assistants)
    trans_idx = assistant_to_transition[assistant_idx]
    transition = parent_trajectory.transitions[trans_idx]

    # Pick a token position < 50% of the total tokens
    total_tokens = len(transition.ac.tokens)
    max_token_idx = int(total_tokens * 0.5)

    if max_token_idx <= 1:
        logger.warning(f"Selected assistant message has too few tokens ({total_tokens}), running from initial state")
        new_env = env_group_builder.env_thunk()
        return await do_single_rollout(policy, new_env)

    token_idx = rng.randint(1, max_token_idx)

    logger.debug(
        f"Branching from message {assistant_idx} at token {token_idx}/{total_tokens} "
        f"({token_idx/total_tokens*100:.1f}%)"
    )

    # Build partial prefix
    modified_past_messages, observation, partial_tokens = build_partial_prefix(
        parent_env.past_messages, assistant_idx, token_idx, transition, renderer
    )

    # Create new environment
    new_env = env_group_builder.env_thunk()

    # Set history to messages BEFORE the partial assistant message
    history_before_branch = parent_env.past_messages[:assistant_idx]
    new_env.set_history(history_before_branch)

    # Run trajectory from this branch point
    transitions = []
    stop_condition = parent_env.stop_condition
    first_step = True

    while True:
        # Generate action - time the policy call
        t_start = time.time()
        ac_with_logprobs = await policy(observation, stop_condition)
        policy_time = time.time() - t_start

        # On first step, prepend partial tokens
        if first_step:
            ac_with_logprobs.tokens = partial_tokens + ac_with_logprobs.tokens
            ac_with_logprobs.maybe_logprobs = [0.0] * len(partial_tokens) + ac_with_logprobs.logprobs
            first_step = False

        # Step environment
        step_result = await new_env.step(ac_with_logprobs.tokens)

        # Check if trajectory should be rejected
        if step_result.rejectable_result:
            logger.warning("Branch trajectory rejected: rejectable_result=True")
            return RejectedTrajectory(reason="rejectable_result=True")

        # Store policy timing in transition metrics
        transition_metrics = step_result.metrics.copy() if step_result.metrics else {}
        transition_metrics['policy_time'] = policy_time

        # Create transition
        transition = Transition(
            ob=observation,
            ac=ac_with_logprobs,
            reward=step_result.reward,
            episode_done=step_result.episode_done,
            metrics=transition_metrics,
        )
        transitions.append(transition)

        # Update observation
        observation = step_result.next_observation
        stop_condition = step_result.next_stop_condition

        if step_result.episode_done:
            break

    return Trajectory(transitions=transitions, final_ob=observation)


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

            # Check if trajectory was rejected
            if isinstance(traj, RejectedTrajectory):
                logger.warning(f"Tree trajectory rejected: {traj.reason}")
                continue  # Skip adding to completed_trajectories

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

                # Check if trajectory should be rejected
                if step_result.rejectable_result:
                    logger.warning("Tree child trajectory rejected: rejectable_result=True")
                    return RejectedTrajectory(reason="rejectable_result=True")

                # Continue rollout if not done
                if not step_result.episode_done:
                    ob = step_result.next_observation
                    stop_condition = step_result.next_stop_condition

                    transitions = []
                    while True:
                        ac_with_logprobs = await policy(ob, stop_condition)
                        step_result = await env.step(ac_with_logprobs.tokens)

                        # Check if trajectory should be rejected
                        if step_result.rejectable_result:
                            logger.warning("Tree child trajectory rejected: rejectable_result=True")
                            return RejectedTrajectory(reason="rejectable_result=True")

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
