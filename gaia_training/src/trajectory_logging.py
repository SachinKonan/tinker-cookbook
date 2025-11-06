"""
Helper functions for logging trajectories to WandB tables
"""
import json
import logging

import wandb
import pandas as pd

from tinker_cookbook.rl.types import TrajectoryGroup, BranchedTrajectory

logger = logging.getLogger(__name__)


def build_trajectory_table(
    trajectory_groups_P: list[TrajectoryGroup],
    step_ix: int,
    epoch: int = 0,
    max_trajectories: int = 16,
) -> wandb.Table | None:
    """
    Build a WandB table from trajectory groups for visualization.

    Args:
        trajectory_groups_P: List of trajectory groups from one batch
        step_ix: Current training step index (global step across all epochs)
        epoch: Current epoch number
        max_trajectories: Maximum number of trajectories to include (default 16)

    Returns:
        wandb.Table with trajectory data, or None if no data
    """

    rows = []

    # Collect all trajectories with their metadata
    for group_ix, traj_group in enumerate(trajectory_groups_P):
        for traj_ix, trajectory in enumerate(traj_group.trajectories_G):
            # Get metrics from the final transition (which contains all metadata)
            if not trajectory.transitions:
                continue

            final_transition = trajectory.transitions[-1]
            metrics = final_transition.metrics

            # Extract metadata (with defaults for backwards compatibility)
            past_messages = metrics.get("past_messages", [])
            question = metrics.get("question", "")
            ground_truth = metrics.get("ground_truth", "")
            model_answer = metrics.get("model_answer", "")
            total_tokens = metrics.get("total_tokens", 0)
            total_turns = metrics.get("total_turns", len(trajectory.transitions))
            max_tokens_exceeded = metrics.get("max_tokens_exceeded", False)
            max_turns_exceeded = metrics.get("max_turns_exceeded", False)
            correct = metrics.get("correct", 0.0)
            format_correct = metrics.get("format", 0.0)

            # Compute total reward
            total_reward = traj_group.get_total_rewards()[traj_ix]

            # Convert conversation to JSON string
            conversation_json = json.dumps(past_messages, indent=2)

            # Extract source trajectory index for branched trajectories
            src_traj_ix = None
            if isinstance(trajectory, BranchedTrajectory) and trajectory.references:
                # Find the source trajectory index in the group
                source_traj = trajectory.references[0].source_trajectory
                try:
                    src_traj_ix = traj_group.trajectories_G.index(source_traj)
                except ValueError:
                    # Source trajectory might not be in this group (shouldn't happen)
                    src_traj_ix = None

            row = {
                "epoch": epoch,
                "step_ix": step_ix,
                "batch_ix": group_ix,  # batch_ix is the group index within batch
                "group_ix": group_ix,
                "traj_ix": traj_ix,
                "src_traj_ix": src_traj_ix,  # None for non-branched rollouts or root trajectories
                "conversation": conversation_json,
                "question": question,
                "ground_truth": ground_truth,
                "model_answer": model_answer,
                "reward": total_reward,
                "correct": correct,
                "format": format_correct,
                "total_tokens": total_tokens,
                "total_turns": total_turns,
                "max_tokens_exceeded": max_tokens_exceeded,
                "max_turns_exceeded": max_turns_exceeded,
            }
            rows.append(row)

    if not rows:
        logger.warning("No trajectory data to log")
        return None

    # Take first max_trajectories for consistent comparison across time
    rows = rows[:max_trajectories]

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Create wandb.Table
    table = wandb.Table(dataframe=df)
    logger.info(f"Created trajectory table with {len(rows)} trajectories (max={max_trajectories})")

    return table
