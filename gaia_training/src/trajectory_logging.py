"""
Helper functions for logging trajectories to WandB tables
"""
import json
import logging
import os
from pathlib import Path

import wandb
import pandas as pd

from tinker_cookbook.rl.types import TrajectoryGroup, BranchedTrajectory

logger = logging.getLogger(__name__)


def build_trajectory_table(
    trajectory_groups_P: list[TrajectoryGroup],
    step_ix: int,
    epoch: int = 0,
    max_trajectories: int = 16,
    save_local_json: bool = False,
    sample_best: bool = False,
    run_name: str | None = None,
    log_dir: str = "logs/saved_local_data",
) -> wandb.Table | None:
    """
    Build a WandB table from trajectory groups for visualization.

    Args:
        trajectory_groups_P: List of trajectory groups from one batch
        step_ix: Current training step index (global step across all epochs)
        epoch: Current epoch number
        max_trajectories: Maximum number of trajectories to include (default 16)
        save_local_json: If True, save all trajectories to local JSON file
        sample_best: If True, create small table with best trajectory per group + file path
        run_name: Name of the run (for local file path)
        log_dir: Base directory for local logging (default: logs/saved_local_data)

    Returns:
        wandb.Table with trajectory data, or None if no data
    """

    # Save local JSON if requested
    local_file_path = None
    if save_local_json:
        if run_name is None:
            logger.warning("save_local_json=True but run_name is None, skipping local save")
        else:
            # Create directory structure: logs/saved_local_data/{run_name}/trajectories/
            trajectories_dir = Path(log_dir) / run_name / "trajectories"
            trajectories_dir.mkdir(parents=True, exist_ok=True)

            # Create file path: step_{global_step}.json
            local_file_path = trajectories_dir / f"step_{step_ix}.json"

            # Build JSON data: {"batch_0": {"0": {...}, "1": {...}}, "batch_1": {...}, ...}
            batch_data = {}
            for group_ix, traj_group in enumerate(trajectory_groups_P):
                trajectories_dict = {}

                for traj_ix, trajectory in enumerate(traj_group.trajectories_G):
                    if not trajectory.transitions:
                        continue

                    # Extract metadata from final transition
                    final_transition = trajectory.transitions[-1]
                    metrics = final_transition.metrics

                    # Extract all metadata fields
                    past_messages = metrics.get("past_messages", [])
                    question = metrics.get("question", "")
                    ground_truth = metrics.get("ground_truth", "")
                    model_answer = metrics.get("model_answer", "")
                    total_tokens = metrics.get("episode_total_tokens", 0)
                    total_turns = metrics.get("episode_total_turns", len(trajectory.transitions))
                    correct = metrics.get("correct", 0.0)
                    format_correct = metrics.get("format", 0.0)
                    episode_done = final_transition.episode_done

                    # Calculate total reward from all transitions
                    total_reward = sum(transition.reward for transition in trajectory.transitions)

                    # Extract source trajectory index and branch info for branched trajectories
                    src_traj_ix = None
                    branch_transition_idx = None
                    branch_token_idx = None
                    if isinstance(trajectory, BranchedTrajectory) and trajectory.references:
                        reference = trajectory.references[0]
                        source_traj = reference.source_trajectory
                        try:
                            src_traj_ix = traj_group.trajectories_G.index(source_traj)
                        except ValueError:
                            src_traj_ix = None
                        branch_transition_idx = reference.transition_idx
                        branch_token_idx = reference.token_idx

                    # Build transitions list with token-level data
                    transitions_data = []
                    for transition in trajectory.transitions:
                        # Extract observation tokens
                        obs_tokens = transition.ob.to_ints()

                        # Extract action tokens and logprobs
                        ac_tokens = transition.ac.tokens
                        ac_logprobs = transition.ac.maybe_logprobs  # Can be None

                        transitions_data.append({
                            "obs_tokens": obs_tokens,
                            "ac_tokens": ac_tokens,
                            "ac_logprobs": ac_logprobs,
                        })

                    # Build complete trajectory metadata dict
                    traj_data = {
                        "src_traj_ix": src_traj_ix,
                        "branch_transition_idx": branch_transition_idx,
                        "branch_token_idx": branch_token_idx,
                        "conversation": past_messages,
                        "question": question,
                        "ground_truth": ground_truth,
                        "model_answer": model_answer,
                        "reward": total_reward,
                        "correct": correct,
                        "format_correct": format_correct,
                        "total_tokens": total_tokens,
                        "total_turns": total_turns,
                        "episode_done": episode_done,
                        "transitions": transitions_data,
                    }

                    trajectories_dict[str(traj_ix)] = traj_data

                batch_data[f"batch_{group_ix}"] = trajectories_dict

            # Write to file
            with open(local_file_path, 'w') as f:
                json.dump(batch_data, f, indent=2)

            logger.info(f"Saved {len(batch_data)} groups to {local_file_path}")

    rows = []

    # Collect all trajectories with their metadata
    for group_ix, traj_group in enumerate(trajectory_groups_P):
        # If sample_best, find the trajectory with highest reward
        if sample_best:
            # Get all rewards for this group
            rewards = traj_group.get_total_rewards()
            if not rewards:
                continue
            # Find index of trajectory with highest reward
            best_traj_ix = max(range(len(rewards)), key=lambda i: rewards[i])
            trajectories_to_log = [(best_traj_ix, traj_group.trajectories_G[best_traj_ix])]
        else:
            # Log all trajectories in the group
            trajectories_to_log = list(enumerate(traj_group.trajectories_G))

        for traj_ix, trajectory in trajectories_to_log:
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
            total_tokens = metrics.get("episode_total_tokens", 0)
            total_turns = metrics.get("episode_total_turns", len(trajectory.transitions))
            max_tokens_exceeded = metrics.get("max_tokens_exceeded", False)
            max_turns_exceeded = metrics.get("max_turns_exceeded", False)
            correct = metrics.get("correct", 0.0)
            format_correct = metrics.get("format", 0.0)

            # Compute total reward
            total_reward = traj_group.get_total_rewards()[traj_ix]

            # Convert conversation to JSON string or HTML
            if sample_best:
                # Use wandb.Html for readable rendering in WandB UI
                conversation_html = json.dumps(past_messages, indent=2)
                conversation_display = wandb.Html(f"<pre>{conversation_html}</pre>")
            else:
                # Use plain JSON string (backward compatibility)
                conversation_json = json.dumps(past_messages, indent=2)
                conversation_display = conversation_json

            # Extract source trajectory index and branch info for branched trajectories
            src_traj_ix = None
            branch_transition_idx = None
            branch_token_idx = None
            if isinstance(trajectory, BranchedTrajectory) and trajectory.references:
                reference = trajectory.references[0]
                source_traj = reference.source_trajectory
                try:
                    src_traj_ix = traj_group.trajectories_G.index(source_traj)
                except ValueError:
                    # Source trajectory might not be in this group (shouldn't happen)
                    src_traj_ix = None
                branch_transition_idx = reference.transition_idx
                branch_token_idx = reference.token_idx

            row = {
                "epoch": epoch,
                "step_ix": step_ix,
                "batch_ix": group_ix,  # batch_ix is the group index within batch
                "group_ix": group_ix,
                "traj_ix": traj_ix,
                "src_traj_ix": src_traj_ix,  # None for non-branched rollouts or root trajectories
                "branch_transition_idx": branch_transition_idx,  # Transition index where branching occurred
                "branch_token_idx": branch_token_idx,  # Token position within that transition
                "conversation": conversation_display,
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

            # Add file_path column if sample_best and we saved a local file
            if sample_best and local_file_path is not None:
                row["file_path"] = str(local_file_path.absolute())

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
