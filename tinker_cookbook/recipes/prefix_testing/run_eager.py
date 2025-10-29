"""
Eager branching script - branches DURING trajectory execution with coin flips.

This script demonstrates real-time eager branching:
1. After each assistant message (policy call), flip a coin (only once per trajectory)
2. If heads: immediately fork and branch from that point (parent continues)
3. If all tails (no eager branch): fallback to random branching after completion
4. Termination: stops when total_created >= max OR active_tasks queue is empty

Key difference from run.py:
- run.py: Wait for trajectory to complete → pick branch point → spawn children
- run_eager.py: During execution → coin flip after each message → maybe branch immediately
"""
import argparse
import asyncio
import random
import time
from functools import partial

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import tinker
from tinker_cookbook import renderers, model_info
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.recipes.modified_tool_use.modified_search_env import (
    SearchR1DatasetBuilder,
)
from tinker_cookbook.recipes.modified_tool_use.tools import GAIAToolClient
from tinker_cookbook.rl.types import Transition
from tinker_cookbook.tokenizer_utils import get_tokenizer


def build_partial_continuation_prompt(
    past_messages: list,
    assistant_idx: int,
    partial_content: str,
    renderer
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


def build_partial_prefix(past_messages, assistant_idx, token_idx, transition, renderer):
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


async def get_trajectory(
    env, policy, renderer,
    start_from_initial=True,
    initial_observation=None,
    prefix_tokens=None,
    branch_callback=None,
    can_branch=True,
    branch_probability=0.5,
):
    """Generate a trajectory by running the policy in the environment with eager branching.

    Args:
        env: Environment to run
        policy: Policy (TokenCompleter) to generate actions
        renderer: Renderer for building observations
        start_from_initial: If True, call env.initial_observation().
                           If False, assume env.past_messages is already set.
        initial_observation: Optional (observation, stop_condition) tuple.
                            If provided, use this instead of building observation.
        prefix_tokens: Optional list of prefix tokens for token-level branching.
                      These will be prepended to the first generated tokens.
        branch_callback: Optional async function to call when branching eagerly.
                        Called with (env, transitions, step) to spawn branch.
        can_branch: Whether this trajectory is allowed to branch eagerly.
        branch_probability: Probability of branching on coin flip (default 0.5).

    Returns:
        tuple: (transitions, total_reward, final_metrics, did_branch_eagerly)
    """
    if initial_observation is not None:
        # Use provided observation (for token-level branching)
        observation, stop_condition = initial_observation
        print(f"\n📥 Token-level branched observation length: {observation.length} tokens")
    elif start_from_initial:
        observation, stop_condition = await env.initial_observation()
        print(f"\n📥 Initial observation length: {observation.length} tokens")
    else:
        # Environment already has history set via set_history()
        observation = renderer.build_generation_prompt(env.past_messages)
        stop_condition = env.stop_condition
        print(f"\n📥 Message-level branched observation length: {observation.length} tokens")

    # Execute trajectory loop
    transitions = []
    episode_done = False
    total_reward = 0.0
    final_metrics = None
    did_branch_eagerly = False  # Track whether we branched during execution

    step = 0
    first_step = True
    while not episode_done:
        print(f"\n🔄 Step {step}")

        # Call policy to generate tokens (measure time)
        t_start = time.time()
        tokens_with_logprobs = await policy(observation, stop_condition)
        policy_time = time.time() - t_start

        tokens = tokens_with_logprobs.tokens
        print(f"   Generated {len(tokens)} tokens in {policy_time:.2f}s")

        # For token-level branching: combine prefix tokens with first generated tokens
        if first_step and prefix_tokens is not None:
            print(f"   Combining {len(prefix_tokens)} prefix tokens with {len(tokens)} new tokens")
            combined_tokens = prefix_tokens + tokens
            # Also combine the logprobs (use None for prefix tokens that weren't generated)
            combined_logprobs = [None] * len(prefix_tokens) + (tokens_with_logprobs.maybe_logprobs or [None] * len(tokens))

            # Create new TokensWithLogprobs with combined tokens
            from tinker_cookbook.completers import TokensWithLogprobs
            combined_tokens_with_logprobs = TokensWithLogprobs(
                tokens=combined_tokens,
                maybe_logprobs=combined_logprobs if tokens_with_logprobs.maybe_logprobs is not None else None
            )

            # Use combined tokens for env step and transition
            step_tokens = combined_tokens
            transition_ac = combined_tokens_with_logprobs
        else:
            step_tokens = tokens
            transition_ac = tokens_with_logprobs

        # Step environment
        step_result = await env.step(step_tokens)

        # Check if trajectory should be rejected
        if step_result.rejectable_result:
            print(f"\n⚠️  Trajectory rejected: rejectable_result=True")
            return None, 0.0, None, False

        # Store transition with policy time in metrics
        transition_metrics = step_result.metrics.copy() if step_result.metrics else {}
        transition_metrics['policy_time'] = policy_time

        transition = Transition(
            ob=observation,
            ac=transition_ac,
            reward=step_result.reward,
            episode_done=step_result.episode_done,
            metrics=transition_metrics,
        )
        transitions.append(transition)

        # EAGER BRANCHING: After each assistant message, flip coin (only once)
        if can_branch and not did_branch_eagerly and branch_callback is not None:
            if random.random() < branch_probability:
                print(f"   🎲 Coin flip: HEADS - Branching eagerly from step {step}!")
                # Branch NOW - parent continues running
                await branch_callback(env, transitions, step, transition_ac)
                did_branch_eagerly = True
            else:
                print(f"   🎲 Coin flip: TAILS - Continue without branching")

        first_step = False

        # Update state
        episode_done = step_result.episode_done
        total_reward += step_result.reward
        observation = step_result.next_observation
        stop_condition = step_result.next_stop_condition

        if step_result.metrics:
            final_metrics = step_result.metrics

        print(f"   Reward: {step_result.reward}")
        print(f"   Episode done: {episode_done}")

        step += 1

    return transitions, total_reward, final_metrics, did_branch_eagerly


def print_trajectory(transitions, total_reward, final_metrics, past_messages, renderer, label="TRAJECTORY"):
    """Print the full trajectory including all messages and transitions.

    Args:
        transitions: List of Transition objects
        total_reward: Total reward accumulated
        final_metrics: Final metrics dict
        past_messages: All messages from env.past_messages
        renderer: Renderer for decoding tokens
        label: Label for this trajectory (e.g., "INITIAL TRAJECTORY")
    """
    import textwrap

    # Print full trajectory
    print("\n" + "=" * 80)
    print(f"{label} DETAILS (FULL TEXT)")
    print("=" * 80)

    print(f"\n{'─' * 80}")
    print(f"📍 {label}")
    print(f"   Transitions: {len(transitions)}")
    print(f"   Total reward: {total_reward}")
    if final_metrics:
        print(f"   Metrics: {final_metrics}")

    # Show FULL conversation including system messages and tool results
    print(f"\n   💬 Full Conversation (ALL messages):")

    # Find all assistant message indices
    assistant_indices = [i for i, msg in enumerate(past_messages) if msg.get("role") == "assistant"]

    # Map last N assistant messages to transitions (where N = len(transitions))
    # The last len(transitions) assistant messages correspond to our transitions
    assistant_to_transition = {}
    if len(assistant_indices) >= len(transitions):
        for i, trans_idx in enumerate(range(len(transitions))):
            msg_idx = assistant_indices[-(len(transitions) - i)]
            assistant_to_transition[msg_idx] = trans_idx

    # Print ALL messages from past_messages
    for msg_idx, msg in enumerate(past_messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        # Check if this assistant message has timing info
        timing_info = ""
        if msg_idx in assistant_to_transition:
            trans = transitions[assistant_to_transition[msg_idx]]
            policy_time = trans.metrics.get('policy_time', 0)
            num_tokens = len(trans.ac.tokens)
            timing_info = f" ⏱️  {policy_time:.2f}s ({num_tokens} tokens, {num_tokens/policy_time:.1f} tok/s)"

        print(f"\n      ╭─ Message {msg_idx} [{role.upper()}]{timing_info}")
        print(f"      │")

        # Check if this is a tool call message
        if "tool_calls" in msg and msg["tool_calls"]:
            tool_call = msg["tool_calls"][0]
            print(f"      │ 🔧 TOOL CALL: {tool_call.get('name', 'unknown')}")
            print(f"      │ Arguments: {tool_call.get('arguments', {})}")
            print(f"      │")

        # Print content
        if content:
            for line in content.split("\n"):
                if line.strip():
                    wrapped = textwrap.wrap(line, width=70)
                    for w_line in wrapped:
                        print(f"      │ {w_line}")
                else:
                    print(f"      │")

        print(f"      ╰─")


async def run_source_trajectory(
    src_id: int,
    env_thunk,
    policy,
    renderer,
    program_start_time: float,
    branch_callback=None,
    branch_probability=0.5,
) -> dict:
    """Run a source trajectory (root trajectory in multi-root forest) with eager branching.

    Args:
        src_id: ID for this source trajectory
        env_thunk: Function to create a new environment
        policy: Policy to generate actions
        renderer: Renderer for building observations
        program_start_time: Start time of the program (for computing completion time)
        branch_callback: Optional callback for eager branching
        branch_probability: Probability of branching on coin flip

    Returns:
        Dict with trajectory info: {id, parent_id, depth, env, transitions, reward, metrics, completion_time, did_branch_eagerly}
    """
    print(f"\n🌱 Running source trajectory {src_id}...")

    # Create environment
    env = env_thunk()

    # Run trajectory with eager branching
    transitions, total_reward, final_metrics, did_branch_eagerly = await get_trajectory(
        env, policy, renderer,
        start_from_initial=True,
        branch_callback=branch_callback,
        can_branch=True,
        branch_probability=branch_probability,
    )

    # Check if trajectory was rejected
    if transitions is None:
        print(f"   ⚠️  Source {src_id} rejected")
        return None

    branch_status = "BRANCHED" if did_branch_eagerly else "no branch"
    print(f"   ✅ Source {src_id} complete: {len(transitions)} transitions, reward={total_reward} ({branch_status})")

    return {
        "id": src_id,
        "parent_id": None,  # Source trajectories have no parent
        "depth": 0,  # All sources are at depth 0
        "env": env,
        "transitions": transitions,
        "reward": total_reward,
        "metrics": final_metrics or {},
        "branch_info": None,
        "completion_time": time.time() - program_start_time,
        "did_branch_eagerly": did_branch_eagerly,
    }


async def create_and_run_branch(
    parent_traj_info: dict,
    branch_idx: int,
    branch_id: int,
    env_thunk,
    policy,
    renderer,
    program_start_time: float,
    branch_callback=None,
    branch_probability=0.5,
) -> dict:
    """Create and run a branch from a parent trajectory with eager branching.

    Args:
        parent_traj_info: Dict with parent's {id, transitions, env, ...}
        branch_idx: Index of this branch among siblings (0, 1, 2, ...)
        branch_id: Global unique ID for this branch
        env_thunk: Function to create a new environment
        policy: Policy to generate actions
        renderer: Renderer for building observations
        program_start_time: Start time of the program (for computing completion time)
        branch_callback: Optional callback for eager branching
        branch_probability: Probability of branching on coin flip

    Returns:
        Dict with trajectory info: {id, parent_id, depth, env, transitions, reward, metrics, branch_info, completion_time, did_branch_eagerly}
    """
    parent_id = parent_traj_info["id"]
    parent_transitions = parent_traj_info["transitions"]
    parent_env = parent_traj_info["env"]
    parent_depth = parent_traj_info.get("depth", 0)

    print(f"\n🌿 Creating branch {branch_id} from parent {parent_id} (branch_idx={branch_idx})")

    # Use independent RNG for this branch
    rng = random.Random(42 + branch_id)

    # Find all assistant messages with transitions in parent
    assistant_indices = [
        i for i, msg in enumerate(parent_env.past_messages)
        if msg.get("role") == "assistant"
    ]

    # Map assistant indices to transitions
    assistant_to_transition = {}
    if len(assistant_indices) >= len(parent_transitions):
        for i in range(len(parent_transitions)):
            msg_idx = assistant_indices[-(len(parent_transitions) - i)]
            assistant_to_transition[msg_idx] = i

    # Only pick from assistant messages that have transitions (excluding last)
    branchable_assistants = [idx for idx in assistant_to_transition.keys()]
    if len(branchable_assistants) > 1:
        branchable_assistants = branchable_assistants[:-1]  # Exclude last

    if len(branchable_assistants) == 0:
        print(f"   ⚠️  No branchable assistants in parent {parent_id}, skipping branch")
        return None

    # Randomly pick one assistant message
    ix = rng.choice(branchable_assistants)

    # Get the corresponding transition
    trans_idx = assistant_to_transition[ix]
    transition = parent_transitions[trans_idx]

    # Pick a token position < 50% of the total tokens
    total_tokens = len(transition.ac.tokens)
    max_token_idx = int(total_tokens * 0.5)

    if max_token_idx <= 1:
        print(f"   ⚠️  Selected assistant message has too few tokens ({total_tokens}), skipping branch")
        return None

    token_idx = rng.randint(1, max_token_idx)

    print(f"   Branching from message {ix} at token {token_idx}/{total_tokens} ({token_idx/total_tokens*100:.1f}%)")

    # Build partial prefix
    modified_past_messages, observation, partial_tokens = build_partial_prefix(
        parent_env.past_messages, ix, token_idx, transition, renderer
    )

    # Create new environment
    new_env = env_thunk()

    # Set history to messages BEFORE the partial assistant message
    history_before_branch = parent_env.past_messages[:ix]
    new_env.set_history(history_before_branch)

    # Run trajectory from this point with eager branching
    print(f"   Running trajectory for branch {branch_id}...")
    transitions, total_reward, final_metrics, did_branch_eagerly = await get_trajectory(
        new_env, policy, renderer,
        start_from_initial=False,
        initial_observation=(observation, parent_env.stop_condition),
        prefix_tokens=partial_tokens,
        branch_callback=branch_callback,
        can_branch=True,
        branch_probability=branch_probability,
    )

    # Check if trajectory was rejected
    if transitions is None:
        print(f"   ⚠️  Branch {branch_id} rejected")
        return None

    branch_status = "BRANCHED" if did_branch_eagerly else "no branch"
    print(f"   ✅ Branch {branch_id} complete: {len(transitions)} transitions, reward={total_reward} ({branch_status})")

    return {
        "id": branch_id,
        "parent_id": parent_id,
        "depth": parent_depth + 1,
        "env": new_env,
        "transitions": transitions,
        "reward": total_reward,
        "metrics": final_metrics or {},
        "branch_info": {
            "parent_message_idx": ix,
            "parent_transition_idx": trans_idx,
            "token_idx": token_idx,
            "total_tokens": total_tokens,
        },
        "completion_time": time.time() - program_start_time,  # Time since start
        "did_branch_eagerly": did_branch_eagerly,  # Track if this branch did eager branching
    }


async def main(num_branches: int = 1, max_total_trajectories: int = 10, src_trajectories: int = 1):
    print("=" * 80)
    print("PREFIX TESTING - Trajectory Execution with Branching")
    print("=" * 80)
    print(f"   Source trajectories: {src_trajectories}")
    print(f"   Branching factor: {num_branches}")
    print(f"   Max total trajectories: {max_total_trajectories}")
    print("=" * 80)

    # Load environment variables
    load_dotenv()

    # Setup model and tokenizer
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    print(f"\n🔧 Setting up model: {model_name}")

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=model_name)

    tokenizer = get_tokenizer(model_name)
    recommended_renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(recommended_renderer_name, tokenizer=tokenizer)

    # Wrap in TinkerTokenCompleter
    policy = TinkerTokenCompleter(sampling_client=sampling_client, max_tokens=2048)

    print(f"   Renderer: {recommended_renderer_name}")
    print(f"   Max tokens: 2048")

    # Create dataset builder
    print("\n🔧 Creating SearchR1Dataset...")
    dataset_builder = SearchR1DatasetBuilder(
        batch_size=1,  # One batch
        group_size=1,  # One group
        model_name_for_tokenizer=model_name,
        renderer_name=recommended_renderer_name,
        max_search_results=5,
        convo_prefix="standard",
        seed=42,
    )

    # Build dataset
    train_dataset, _ = await dataset_builder()
    print(f"   Dataset size: {len(train_dataset)} batches")

    # Get first batch
    print("\n📦 Getting first batch (batch_size=1, group_size=1)...")
    batch = train_dataset.get_batch(0)
    print(f"   Batch contains {len(batch)} environment groups")

    # Get the first (and only) env group builder
    env_group_builder = batch[0]
    print(f"   Environment group has {env_group_builder.num_envs} environments")

    # Create the environment
    env = env_group_builder.env_thunk()
    print(f"\n🌍 Environment created")
    print(f"   Question: {env.problem}")
    print(f"   Answer: {env.answer}")

    # Track program start time for timeline plot
    program_start_time = time.time()

    # Track all trajectories and environments
    all_trajectory_info = []
    all_envs = []

    # ========================================================================
    # MULTI-ROOT SOURCE TRAJECTORIES + EAGER BRANCHING
    # ========================================================================
    if num_branches >= 0 and max_total_trajectories >= src_trajectories:
        print("\n" + "=" * 80)
        print(f"🌳 STARTING {src_trajectories} SOURCE TRAJECTORIES + EAGER BRANCHING")
        print("=" * 80)

        # Track created count and next ID
        total_created = 0
        next_id = src_trajectories  # Branch IDs start after source IDs
        active_tasks = {}  # task -> parent_traj_info (or None for sources)

        # Create eager branching callback
        async def eager_branch_callback(env, transitions, step, transition_ac):
            """Called when coin flip is heads during trajectory execution - spawn branch immediately."""
            nonlocal next_id, total_created

            if total_created >= max_total_trajectories:
                print(f"      ⚠️  Already at max trajectories ({max_total_trajectories}), skipping eager branch")
                return

            # Extract info needed for branching
            print(f"      🔀 EAGER BRANCH: Spawning child immediately from step {step}...")

            # The parent trajectory is still running, but we can extract its current state
            # We need to construct a partial parent_traj_info for the branching logic
            parent_traj_info_partial = {
                "id": "in-flight",  # Mark as in-flight
                "transitions": transitions[:step+1],  # Transitions up to this point
                "env": env,
                "depth": 0,  # Will be determined when we spawn
            }

            # For eager branching, we branch from the current transition at a random token position
            # Get the transition we just completed
            transition = transitions[step]
            total_tokens = len(transition_ac.tokens)
            max_token_idx = int(total_tokens * 0.5)

            if max_token_idx <= 1:
                print(f"      ⚠️  Too few tokens ({total_tokens}), skipping eager branch")
                return

            rng = random.Random(42 + next_id)
            token_idx = rng.randint(1, max_token_idx)

            print(f"      📍 Branching at token {token_idx}/{total_tokens} ({token_idx/total_tokens*100:.1f}%)")

            # Find the assistant message index for this transition
            assistant_indices = [
                i for i, msg in enumerate(env.past_messages)
                if msg.get("role") == "assistant"
            ]

            # The current step corresponds to the last assistant message
            if len(assistant_indices) >= step + 1:
                assistant_msg_idx = assistant_indices[-(len(transitions) - step)]
            else:
                print(f"      ⚠️  Cannot find assistant message index, skipping eager branch")
                return

            # Build partial prefix
            modified_past_messages, observation, partial_tokens = build_partial_prefix(
                env.past_messages, assistant_msg_idx, token_idx, transition, renderer
            )

            # Create new environment for the branch
            new_env = env_group_builder.env_thunk()
            history_before_branch = env.past_messages[:assistant_msg_idx]
            new_env.set_history(history_before_branch)

            # Create the branch task
            branch_id = next_id
            next_id += 1
            total_created += 1

            print(f"      ✅ Created eager branch task {branch_id}")

            # Spawn the branch as an async task
            async def run_eager_branch():
                """Run the eagerly-spawned branch."""
                print(f"\n🌿 Running eager branch {branch_id}...")

                transitions_b, total_reward_b, final_metrics_b, did_branch_eagerly_b = await get_trajectory(
                    new_env, policy, renderer,
                    start_from_initial=False,
                    initial_observation=(observation, env.stop_condition),
                    prefix_tokens=partial_tokens,
                    branch_callback=eager_branch_callback,  # Can branch recursively
                    can_branch=True,
                    branch_probability=0.5,
                )

                # Check if trajectory was rejected
                if transitions_b is None:
                    print(f"   ⚠️  Eager branch {branch_id} rejected")
                    return None

                branch_status = "BRANCHED" if did_branch_eagerly_b else "no branch"
                print(f"   ✅ Eager branch {branch_id} complete: {len(transitions_b)} transitions, reward={total_reward_b} ({branch_status})")

                return {
                    "id": branch_id,
                    "parent_id": "in-flight",  # Will update later if needed
                    "depth": 1,  # Simplified depth for eager branches
                    "env": new_env,
                    "transitions": transitions_b,
                    "reward": total_reward_b,
                    "metrics": final_metrics_b or {},
                    "branch_info": {
                        "parent_message_idx": assistant_msg_idx,
                        "parent_transition_idx": step,
                        "token_idx": token_idx,
                        "total_tokens": total_tokens,
                        "eager": True,  # Mark as eager branch
                    },
                    "completion_time": time.time() - program_start_time,
                    "did_branch_eagerly": did_branch_eagerly_b,
                }

            # Add to active tasks
            branch_task = asyncio.create_task(run_eager_branch())
            active_tasks[branch_task] = None  # No parent tracking for eager branches

        # Launch all source trajectories concurrently with eager branching callback
        print(f"\n🌱 Launching {src_trajectories} source trajectories...")
        for src_id in range(src_trajectories):
            task = asyncio.create_task(
                run_source_trajectory(
                    src_id,
                    env_group_builder.env_thunk, policy, renderer,
                    program_start_time,
                    branch_callback=eager_branch_callback,  # Pass callback for eager branching
                    branch_probability=0.5,
                )
            )
            active_tasks[task] = None  # No parent for sources
            total_created += 1

        # Process trajectories as they complete
        while active_tasks:
            # Wait for the next trajectory to complete
            done, pending = await asyncio.wait(active_tasks.keys(), return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                parent_info = active_tasks.pop(task)
                traj_info = await task

                if traj_info is not None:
                    # Store completed trajectory
                    all_trajectory_info.append(traj_info)
                    all_envs.append(traj_info["env"])

                    correct = traj_info['metrics'].get('correct', 0) == 1.0
                    correct_str = "✓ CORRECT" if correct else "✗ incorrect"
                    branch_info = traj_info.get("branch_info")
                    eager_status = " [EAGER]" if (branch_info and branch_info.get("eager", False)) else ""

                    print(f"\n✅ Trajectory {traj_info['id']} complete! {correct_str}{eager_status}")
                    print(f"   Parent: {traj_info['parent_id']}")
                    print(f"   Depth: {traj_info['depth']}")
                    print(f"   Reward: {traj_info['reward']}")
                    print(f"   Total trajectories so far: {len(all_trajectory_info)}")
                    print(f"   Did branch eagerly: {traj_info.get('did_branch_eagerly', False)}")

                    # FALLBACK BRANCHING: If trajectory didn't branch eagerly, do post-completion branching
                    if not traj_info.get('did_branch_eagerly', False):
                        if total_created < max_total_trajectories:
                            children_to_spawn = min(num_branches, max_total_trajectories - total_created)
                            print(f"   🔄 No eager branch occurred - spawning {children_to_spawn} fallback children...")

                            for branch_idx in range(children_to_spawn):
                                child_task = asyncio.create_task(
                                    create_and_run_branch(
                                        traj_info, branch_idx, next_id,
                                        env_group_builder.env_thunk, policy, renderer,
                                        program_start_time,
                                        branch_callback=eager_branch_callback,  # Pass callback
                                        branch_probability=0.5,
                                    )
                                )
                                active_tasks[child_task] = traj_info
                                next_id += 1
                                total_created += 1
                    else:
                        print(f"   ✓ Trajectory already branched eagerly - no fallback needed")

        print("\n" + "=" * 80)
        print("🌳 ALL TRAJECTORIES COMPLETE")
        print("=" * 80)
        print(f"   Total trajectories created: {len(all_trajectory_info)}")
        print(f"   Source trajectories: {src_trajectories}")
        print(f"   Branched trajectories: {len(all_trajectory_info) - src_trajectories}")
        print(f"   Correct answer: {env.answer}")

        # Print tree structure
        print("\n📊 Tree Structure:")
        for traj_info in all_trajectory_info:
            indent = "  " * traj_info["depth"]
            parent_str = f"(from {traj_info['parent_id']})" if traj_info['parent_id'] is not None else "(source)"
            reward_str = f"reward={traj_info['reward']:.2f}"
            correct_str = "✓" if traj_info['metrics'].get('correct', 0) == 1.0 else "✗"
            print(f"{indent}├─ Traj {traj_info['id']} {parent_str} - {reward_str} {correct_str}")

    else:
        print("\n⚠️  Insufficient trajectories requested (max_total < src_trajectories)")

    # Print detailed output for each trajectory
    print("\n" + "=" * 80)
    print("📜 DETAILED TRAJECTORIES")
    print("=" * 80)

    for traj_info in all_trajectory_info:
        label = f"TRAJECTORY {traj_info['id']}"
        if traj_info['parent_id'] is not None:
            label += f" (branched from {traj_info['parent_id']})"
        else:
            label += " (ROOT)"

        print_trajectory(
            traj_info["transitions"],
            traj_info["reward"],
            traj_info["metrics"],
            traj_info["env"].past_messages,
            renderer,
            label=label
        )

    # ========================================================================
    # PLOT TIMING VISUALIZATION
    # ========================================================================
    if len(all_trajectory_info) > 1:
        print("\n" + "=" * 80)
        print("📊 PLOTTING TREE VISUALIZATION")
        print("=" * 80)

        import os
        os.makedirs("logs/prefix_testing", exist_ok=True)

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Policy timing for each trajectory
        colors = plt.cm.tab10(range(len(all_trajectory_info)))

        for i, traj_info in enumerate(all_trajectory_info):
            times = [trans.metrics.get('policy_time', 0) for trans in traj_info["transitions"]]
            x = list(range(len(times)))

            label = f"Traj {traj_info['id']}"
            if traj_info['parent_id'] is not None:
                label += f" (from {traj_info['parent_id']})"

            ax1.plot(x, times, 'o-', label=label, color=colors[i], linewidth=2, markersize=6)

        ax1.set_xlabel('Transition Index', fontsize=12)
        ax1.set_ylabel('Policy Time (seconds)', fontsize=12)
        ax1.set_title('Policy Inference Time per Trajectory', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Tree structure with rewards
        ax2.axis('off')
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)

        # Simple tree visualization
        y_start = 9
        y_step = 1.5

        ax2.text(5, y_start, "Tree Structure", fontsize=14, fontweight='bold', ha='center')
        ax2.text(5, y_start - 0.6, f"Correct answer: {env.answer}", fontsize=10, ha='center', style='italic')

        y = y_start - y_step - 0.4
        for traj_info in all_trajectory_info:
            indent = traj_info["depth"] * 1.5
            x = 1 + indent

            correct = traj_info['metrics'].get('correct', 0) == 1.0
            correct_symbol = "✓" if correct else "✗"

            if traj_info['parent_id'] is None:
                text = f"[{traj_info['id']}] ROOT - reward={traj_info['reward']:.2f} {correct_symbol}"
            else:
                text = f"[{traj_info['id']}] from [{traj_info['parent_id']}] - reward={traj_info['reward']:.2f} {correct_symbol}"

            # Color based on correctness
            color = 'green' if correct else 'red'
            ax2.text(x, y, text, fontsize=10, verticalalignment='top', color=color)
            y -= 0.5

            if y < 0.5:
                break

        plt.tight_layout()

        # Save plot
        plot_path = "logs/prefix_testing/tree_visualization_eager.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"   Plot saved to: {plot_path}")

        plt.show()

        # ====================================================================
        # PLOT 2: Cumulative trajectories over time
        # ====================================================================
        print("\n📊 Creating trajectory completion timeline...")

        # Sort trajectories by completion time
        sorted_trajs = sorted(all_trajectory_info, key=lambda t: t["completion_time"])

        # Extract completion times and create cumulative count
        completion_times = [t["completion_time"] for t in sorted_trajs]
        cumulative_counts = list(range(1, len(sorted_trajs) + 1))

        # Create plot
        fig2, ax = plt.subplots(figsize=(10, 6))

        ax.plot(completion_times, cumulative_counts, 'o-', linewidth=2, markersize=8, color='steelblue')
        ax.fill_between(completion_times, 0, cumulative_counts, alpha=0.3, color='steelblue')

        # Annotate all points with trajectory ID and parent
        for i, (t_time, count) in enumerate(zip(completion_times, cumulative_counts)):
            traj = sorted_trajs[i]
            is_correct = traj['metrics'].get('correct', 0) == 1.0

            # Build annotation text
            if traj['parent_id'] is None:
                label = f"Traj {traj['id']} (root)"
            else:
                label = f"Traj {traj['id']} (from {traj['parent_id']})"

            if is_correct:
                label += " ✓"
                color = 'green'
                # Mark correct trajectories with green dot
                ax.plot(t_time, count, 'go', markersize=10, markeredgewidth=2, markerfacecolor='lightgreen', zorder=3)
            else:
                color = 'darkblue'

            # Annotate with trajectory info
            ax.annotate(label,
                       xy=(t_time, count),
                       xytext=(8, 0),
                       textcoords='offset points',
                       fontsize=8,
                       color=color,
                       fontweight='bold' if is_correct else 'normal',
                       ha='left',
                       va='center')

        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Cumulative Trajectories Completed', fontsize=12)
        ax.set_title('Trajectory Completion Timeline', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, len(sorted_trajs) + 1)

        plt.tight_layout()

        # Save plot
        timeline_plot_path = "logs/prefix_testing/trajectory_completion_timeline_eager.png"
        plt.savefig(timeline_plot_path, dpi=150, bbox_inches='tight')
        print(f"   Timeline plot saved to: {timeline_plot_path}")

        plt.show()

    print("\n" + "=" * 80)
    print("✅ PREFIX TESTING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prefix testing - execute trajectories with token-level branching"
    )
    parser.add_argument(
        "--num-branches",
        type=int,
        default=1,
        help="Branching factor: number of children each trajectory spawns (default: 1)",
    )
    parser.add_argument(
        "--max-total-trajectories",
        type=int,
        default=10,
        help="Maximum total trajectories to create including root (default: 10)",
    )
    parser.add_argument(
        "--src-trajectories",
        type=int,
        default=1,
        help="Number of source trajectories to start with (default: 1)",
    )

    args = parser.parse_args()
    asyncio.run(main(
        num_branches=args.num_branches,
        max_total_trajectories=args.max_total_trajectories,
        src_trajectories=args.src_trajectories
    ))
