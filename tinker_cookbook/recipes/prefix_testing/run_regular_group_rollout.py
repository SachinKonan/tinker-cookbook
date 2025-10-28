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
from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup
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

    # Sort by original index to maintain order
    results_sorted = sorted(results, key=lambda x: x[0])
    trajectories_G = [r[1] for r in results_sorted]
    completion_times = [r[2] for r in results_sorted]

    # Compute group rewards
    rewards_and_metrics_G = await env_group_builder.compute_group_rewards(trajectories_G)
    rewards_G, metrics_G = zip(*rewards_and_metrics_G, strict=True)

    return TrajectoryGroup(trajectories_G, list(rewards_G), list(metrics_G)), completion_times


async def main(group_size: int = 7):
    print("=" * 80)
    print("REGULAR GROUP ROLLOUT - Trajectory Completion Timing")
    print("=" * 80)
    print(f"   Group size: {group_size}")
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
    # PLOT TRAJECTORY COMPLETION TIMELINE
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 CREATING TRAJECTORY COMPLETION TIMELINE")
    print("=" * 80)

    os.makedirs("logs/prefix_testing", exist_ok=True)

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
        traj = sorted_trajs[i]
        is_correct = traj["correct"]

        label = f"Traj {traj['id']}"
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
    plot_path = "logs/prefix_testing/regular_group_rollout_timeline.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"   Timeline plot saved to: {plot_path}")

    plt.show()

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

    args = parser.parse_args()
    asyncio.run(main(group_size=args.group_size))
