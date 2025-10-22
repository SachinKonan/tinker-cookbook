"""
Helper functions for logging trajectories to WandB tables
"""
import json
import logging

import wandb
import pandas as pd

from tinker_cookbook.rl.types import TrajectoryGroup

logger = logging.getLogger(__name__)


def build_trajectory_table(
    trajectory_groups_P: list[TrajectoryGroup],
    step_ix: int,
    epoch: int = 0,
) -> wandb.Table | None:
    """
    Build a WandB table from trajectory groups for visualization.

    Args:
        trajectory_groups_P: List of trajectory groups from one batch
        step_ix: Current training step index (global step across all epochs)
        epoch: Current epoch number

    Returns:
        wandb.Table with trajectory data, or None if no data
    """

    rows = []

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

            row = {
                "epoch": epoch,
                "step_ix": step_ix,
                "batch_ix": step_ix,  # For GAIA, batch_ix == step_ix
                "group_ix": group_ix,
                "traj_ix": traj_ix,
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

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Create wandb.Table
    table = wandb.Table(dataframe=df)
    logger.info(f"Created trajectory table with {len(rows)} trajectories")

    return table
