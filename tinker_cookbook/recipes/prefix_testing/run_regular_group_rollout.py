"""
Regular group-based rollout with trajectory completion timing.

This script runs a standard GRPO-style group rollout where all trajectories
in the group start together and run independently. It tracks when each trajectory
completes to compare with the tree-based eager branching approach.

Key difference from tree-based branching:
- Tree-based: trajectories branch from each other, complete at different times (eager)
- Regular group: all trajectories start together, complete around the same time (parallel)
"""
import argparse
import asyncio
import csv
import json
import os
import time

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import tinker
from tinker_cookbook import renderers, model_info
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.recipes.modified_tool_use.modified_search_env import (
    SearchR1DatasetBuilder,
)
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.rl.types import RejectedTrajectory, Trajectory, TrajectoryGroup
from tinker_cookbook.tokenizer_utils import get_tokenizer


async def do_timed_group_rollout(
    env_group_builder,
    policy,
    start_time: float,
) -> tuple[TrajectoryGroup, list[float]]:
    """
    Run group rollout and track completion time for each trajectory.

    Args:
        env_group_builder: Builder for creating environments
        policy: Policy for generating actions
        start_time: Program start time for computing relative completion times

    Returns:
        tuple: (TrajectoryGroup, list of completion times)
    """
    envs = await env_group_builder.make_envs()

    # Create tasks for each trajectory and track their completion
    trajectory_tasks = []
    completion_times = []

    async def timed_rollout(env, idx):
        """Run rollout and record completion time."""
        trajectory = await do_single_rollout(policy, env)
        completion_time = time.time() - start_time
        return idx, trajectory, completion_time

    # Launch all trajectories concurrently
    results = await asyncio.gather(*[timed_rollout(env, i) for i, env in enumerate(envs)])

    # Filter out rejected trajectories
    results_valid = [r for r in results if not isinstance(r[1], RejectedTrajectory)]
    num_rejected = len(results) - len(results_valid)
    if num_rejected > 0:
        print(f"\n⚠️  Filtered out {num_rejected} rejected trajectories")

    # Sort by original index to maintain order
    results_sorted = sorted(results_valid, key=lambda x: x[0])
    trajectories_G = [r[1] for r in results_sorted]
    completion_times = [r[2] for r in results_sorted]

    # Compute group rewards
    rewards_and_metrics_G = await env_group_builder.compute_group_rewards(trajectories_G)
    rewards_G, metrics_G = zip(*rewards_and_metrics_G, strict=True)

    return TrajectoryGroup(trajectories_G, list(rewards_G), list(metrics_G)), completion_times


def compute_trajectory_stats(traj_info):
    """Compute statistics for a trajectory.

    Returns dict with:
    - traj_ix: Trajectory index
    - src_ix: Source trajectory index (always empty for regular rollout)
    - num_steps_total: Total number of steps in the trajectory
    - num_steps_generated: Number of steps generated (same as total for regular rollout)
    - total_tokens: Total tokens in final observation + all action tokens
    - total_act_tokens: Sum of action tokens across all steps
    - json_of_convo: JSON string of the conversation
    """
    traj = traj_info["trajectory"]
    transitions = traj.transitions

    num_steps_total = len(transitions)
    num_steps_generated = num_steps_total  # No prefix reuse in regular rollout

    # Calculate token counts
    # total_act_tokens: sum of all action tokens
    total_act_tokens = sum(len(t.ac.tokens) for t in transitions)

    # total_tokens: final obs tokens + sum of all action tokens
    final_ob_tokens = transitions[-1].ob.length if transitions else 0
    total_tokens = final_ob_tokens + total_act_tokens

    # Get conversation as JSON (extract from final transition metrics)
    past_messages = []
    if transitions:
        last_metrics = transitions[-1].metrics
        if 'past_messages' in last_metrics:
            past_messages = last_metrics['past_messages']

    json_of_convo = json.dumps(past_messages)

    # Calculate time for all steps
    time_for_generated = sum(t.metrics.get('policy_time', 0) for t in transitions)

    return {
        "traj_ix": traj_info["id"],
        "src_ix": "",  # No parent in regular rollout
        "num_steps_total": num_steps_total,
        "num_steps_generated": num_steps_generated,
        "total_tokens": total_tokens,
        "total_act_tokens": total_act_tokens,
        "json_of_convo": json_of_convo,
        "time_for_generated": time_for_generated,
    }


def print_trajectory_conversation(traj, traj_idx, renderer):
    """Print the full conversation for a trajectory."""
    import textwrap

    # Extract past_messages from last transition
    if not traj.transitions:
        print(f"\n   Traj {traj_idx}: No transitions")
        return

    last_metrics = traj.transitions[-1].metrics
    past_messages = last_metrics.get('past_messages', [])

    if not past_messages:
        print(f"\n   Traj {traj_idx}: No conversation history")
        return

    print("\n" + "=" * 80)
    print(f"TRAJECTORY {traj_idx} CONVERSATION")
    print("=" * 80)

    # Find all assistant message indices
    assistant_indices = [i for i, msg in enumerate(past_messages) if msg.get("role") == "assistant"]

    # Map assistant indices to transitions
    assistant_to_transition = {}
    if len(assistant_indices) >= len(traj.transitions):
        for i in range(len(traj.transitions)):
            msg_idx = assistant_indices[-(len(traj.transitions) - i)]
            assistant_to_transition[msg_idx] = i

    # Print ALL messages
    for msg_idx, msg in enumerate(past_messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        # Check if this assistant message has timing info
        timing_info = ""
        if msg_idx in assistant_to_transition:
            trans = traj.transitions[assistant_to_transition[msg_idx]]
            policy_time = trans.metrics.get('policy_time', 0)
            num_tokens = len(trans.ac.tokens)
            if policy_time > 0:
                timing_info = f" ⏱️  {policy_time:.2f}s ({num_tokens} tokens, {num_tokens/policy_time:.1f} tok/s)"
            else:
                timing_info = f" ⏱️  0.00s ({num_tokens} tokens)"

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


async def main(group_size: int = 7, log_dir: str = "logs/prefix_testing"):
    # Setup logging to both file and stdout using logging package
    import sys
    import logging
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "run.log")

    # Save original stdout
    original_stdout = sys.stdout

    # Configure logging with file handler only
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.FileHandler(log_file, mode='w')],
        force=True
    )

    # Redirect stdout to write to both terminal and log file
    class TeeToLogger:
        def __init__(self, terminal, logger):
            self.terminal = terminal
            self.logger = logger

        def write(self, message):
            self.terminal.write(message)
            if message.strip():
                self.logger.info(message.rstrip())

        def flush(self):
            self.terminal.flush()

    logger = logging.getLogger(__name__)
    sys.stdout = TeeToLogger(original_stdout, logger)

    print("=" * 80)
    print("REGULAR GROUP ROLLOUT - Trajectory Completion Timing")
    print("=" * 80)
    print(f"   Group size: {group_size}")
    print(f"   Log directory: {log_dir}")
    print(f"   Log file: {log_file}")
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
        batch_size=1,
        group_size=group_size,
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
    print("\n📦 Getting first batch...")
    batch = train_dataset.get_batch(0)
    print(f"   Batch contains {len(batch)} environment groups")

    # Get the first (and only) env group builder
    env_group_builder = batch[0]
    print(f"   Environment group has {env_group_builder.num_envs} environments")

    # Get one env to print question/answer
    temp_env = env_group_builder.env_thunk()
    print(f"\n🌍 Question: {temp_env.problem}")
    print(f"   Correct answer: {temp_env.answer}")

    # ========================================================================
    # RUN GROUP ROLLOUT WITH TIMING
    # ========================================================================
    print("\n" + "=" * 80)
    print("RUNNING GROUP ROLLOUT")
    print("=" * 80)

    start_time = time.time()
    trajectory_group, completion_times = await do_timed_group_rollout(
        env_group_builder, policy, start_time
    )

    print("\n" + "=" * 80)
    print("GROUP ROLLOUT COMPLETE")
    print("=" * 80)

    # Print results
    print(f"\n📊 Results:")
    print(f"   Total trajectories: {len(trajectory_group.trajectories_G)}")

    for i, (traj, reward, metrics, comp_time) in enumerate(
        zip(
            trajectory_group.trajectories_G,
            trajectory_group.final_rewards_G,
            trajectory_group.metrics_G,
            completion_times
        )
    ):
        is_correct = metrics.get('correct', 0) == 1.0
        correct_str = "✓" if is_correct else "✗"
        print(f"   Traj {i}: reward={reward:.2f}, {len(traj.transitions)} transitions, "
              f"completed at {comp_time:.2f}s {correct_str}")

    # ========================================================================
    # PRINT FULL CONVERSATIONS
    # ========================================================================
    print("\n" + "=" * 80)
    print("📜 DETAILED TRAJECTORIES")
    print("=" * 80)

    for i, traj in enumerate(trajectory_group.trajectories_G):
        print_trajectory_conversation(traj, i, renderer)

    # ========================================================================
    # PLOT TRAJECTORY COMPLETION TIMELINE
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 CREATING TRAJECTORY COMPLETION TIMELINE")
    print("=" * 80)

    os.makedirs(log_dir, exist_ok=True)

    # Create trajectory info list for plotting
    traj_info_list = []
    for i, (traj, metrics, comp_time) in enumerate(
        zip(
            trajectory_group.trajectories_G,
            trajectory_group.metrics_G,
            completion_times
        )
    ):
        traj_info_list.append({
            "id": i,
            "trajectory": traj,
            "metrics": metrics,
            "completion_time": comp_time,
            "correct": metrics.get('correct', 0) == 1.0,
        })

    # Sort by completion time
    sorted_trajs = sorted(traj_info_list, key=lambda t: t["completion_time"])

    # Extract times and cumulative counts
    times = [t["completion_time"] for t in sorted_trajs]
    cumulative_counts = list(range(1, len(sorted_trajs) + 1))

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(times, cumulative_counts, 'o-', linewidth=2, markersize=8, color='steelblue')
    ax.fill_between(times, 0, cumulative_counts, alpha=0.3, color='steelblue')

    # Annotate all points
    for i, (t_time, count) in enumerate(zip(times, cumulative_counts)):
        traj_info = sorted_trajs[i]
        is_correct = traj_info["correct"]

        # Compute trajectory stats for label
        stats = compute_trajectory_stats(traj_info)
        steps_gen = stats['num_steps_generated']
        time_gen = stats['time_for_generated']

        # Build annotation text: "Traj X (from root; steps_gen: Z; time: Ts)"
        label = f"Traj {traj_info['id']} (from root; steps_gen: {steps_gen}; time: {time_gen:.1f}s)"

        if is_correct:
            label += " ✓"
            color = 'green'
            # Mark correct trajectories with green dot
            ax.plot(t_time, count, 'go', markersize=10, markeredgewidth=2,
                   markerfacecolor='lightgreen', zorder=3)
        else:
            color = 'darkblue'

        # Annotate
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
    ax.set_title('Regular Group Rollout: Trajectory Completion Timeline',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, len(sorted_trajs) + 1)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(log_dir, "regular_group_rollout_timeline.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"   Timeline plot saved to: {plot_path}")

    plt.show()

    # ====================================================================
    # GENERATE CSV TABLE
    # ====================================================================
    print("\n📊 Generating trajectory statistics CSV...")

    csv_path = os.path.join(log_dir, "trajectory_stats_regular.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'traj_ix', 'src_ix', 'num_steps_total', 'num_steps_generated',
            'total_tokens', 'total_act_tokens', 'json_of_convo'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for traj_info in traj_info_list:
            stats = compute_trajectory_stats(traj_info)
            # Remove time_for_generated as it's not in the CSV schema
            stats_for_csv = {k: v for k, v in stats.items() if k != 'time_for_generated'}
            writer.writerow(stats_for_csv)

    print(f"   CSV table saved to: {csv_path}")
    print(f"   Rows: {len(traj_info_list)}")

    print("\n" + "=" * 80)
    print("✅ REGULAR GROUP ROLLOUT COMPLETE!")
    print("=" * 80)
    print("\nComparison:")
    print("  - Regular group rollout: All trajectories start at t=0, complete in narrow window")
    print("  - Tree-based branching: Trajectories branch from each other, complete across wider range")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regular group rollout with trajectory completion timing"
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=7,
        help="Number of trajectories in the group (default: 7)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/prefix_testing",
        help="Directory to save logs, plots, and CSV files (default: logs/prefix_testing)",
    )

    args = parser.parse_args()
    asyncio.run(main(group_size=args.group_size, log_dir=args.log_dir))
