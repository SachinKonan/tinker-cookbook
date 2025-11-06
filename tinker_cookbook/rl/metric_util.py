import asyncio
import itertools
from collections import defaultdict
from typing import Dict, List

import numpy as np
import tinker
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, TrajectoryGroup
from tinker_cookbook.utils.misc_utils import all_same, dict_mean


def _compute_by_group_metrics(trajectory_groups_P: List[TrajectoryGroup], good_thresh: float = 0.5):
    n_groups = len(trajectory_groups_P)
    n_mixed = n_good = n_bad = 0
    for tg in trajectory_groups_P:
        grp_rewards = tg.get_total_rewards()
        if all_same(grp_rewards):
            if grp_rewards[0] >= good_thresh:
                n_good += 1
            else:
                n_bad += 1
        else:
            n_mixed += 1
    return {
        "by_group/frac_mixed": n_mixed / n_groups,
        "by_group/frac_all_good": n_good / n_groups,
        "by_group/frac_all_bad": n_bad / n_groups,
    }


def compute_trajectory_metrics(
    trajectory_groups_P: List[TrajectoryGroup], taglist_P: List[list[str]]
) -> Dict[str, float]:
    tag2trajgroups = defaultdict(list)
    for taglist, trajectory_group in zip(taglist_P, trajectory_groups_P):
        for tag in taglist:
            tag2trajgroups[tag].append(trajectory_group)
    out = {}
    have_nontrivial_tags = any(
        len(trajgroups) < len(trajectory_groups_P) for trajgroups in tag2trajgroups.values()
    )  # check if any tag gives us a strict subset of the full trajectory groups
    if have_nontrivial_tags:
        for tag, trajectory_groups in tag2trajgroups.items():
            prefixed_metrics = {
                f"env/{tag}/{k}": v
                for k, v in _compute_trajectory_metrics(trajectory_groups).items()
            }
            out.update(prefixed_metrics)
    out.update(
        {f"env/all/{k}": v for k, v in _compute_trajectory_metrics(trajectory_groups_P).items()}
    )
    return out


def _compute_trajectory_metrics(trajectory_groups_P: List[TrajectoryGroup]) -> Dict[str, float]:
    """Compute metrics for the trajectory groups."""
    flat_trajs_PG = [traj for tg in trajectory_groups_P for traj in tg.trajectories_G]
    ac_tokens_by_turn = [
        len(transition.ac.tokens) for traj in flat_trajs_PG for transition in traj.transitions
    ]
    ob_tokens_by_turn = [
        transition.ob.length for traj in flat_trajs_PG for transition in traj.transitions
    ]
    turns_by_trajectory = [len(traj.transitions) for traj in flat_trajs_PG]
    # Compute metrics
    metrics = {
        "ac_tokens_per_turn": sum(ac_tokens_by_turn) / sum(turns_by_trajectory),
        "ob_tokens_per_turn": sum(ob_tokens_by_turn) / sum(turns_by_trajectory),
        "turns_per_episode": sum(turns_by_trajectory) / len(flat_trajs_PG),
        "total_episodes": len(flat_trajs_PG),
        "total_turns": sum(turns_by_trajectory),
        "total_ac_tokens": sum(ac_tokens_by_turn),
        "total_ob_tokens": sum(ob_tokens_by_turn),
    }
    metrics["reward/total"] = np.mean(
        [reward for tg in trajectory_groups_P for reward in tg.get_total_rewards()]
    ).item()
    # Per-transition metrics
    transition_metrics = [
        transition.metrics
        for tg in trajectory_groups_P
        for traj in tg.trajectories_G
        for transition in traj.transitions
    ]
    traj_metrics = [metrics for tg in trajectory_groups_P for metrics in tg.metrics_G]
    metrics.update(dict_mean(transition_metrics + traj_metrics))
    # combine traj_metrics and transition_metrics in case there's some key
    # (like format error) that appears in the per-step metrics for some envs
    # but the compute_group_rewards metric for other envs.
    metrics.update(_compute_by_group_metrics(trajectory_groups_P))
    return metrics


def compute_advanced_metrics(trajectory_groups_P: List[TrajectoryGroup]) -> Dict[str, float]:
    """
    Compute advanced token-level metrics for branched and non-branched GRPO.

    Metrics computed:
    - env/all/total_tokens: Total tokens (obs + ac) from last transition across all trajectories
    - env/all/total_num_tokens_w_adv0: Count of tokens with zero advantage
    - env/all/token_adv_entropy_per_batch: Mean entropy of advantage distribution per group

    Args:
        trajectory_groups_P: List of trajectory groups from one batch

    Returns:
        Dictionary of computed metrics
    """
    metrics = {}

    # 1. Total tokens: Sum obs + ac tokens from LAST transition only (efficient)
    total_tokens = 0
    for tg in trajectory_groups_P:
        for traj in tg.trajectories_G:
            if traj.transitions:
                last_transition = traj.transitions[-1]
                total_tokens += last_transition.ob.length + len(last_transition.ac.tokens)

    metrics["env/all/total_tokens"] = total_tokens

    # 2. Total number of tokens with zero advantage
    total_tokens_w_adv0 = 0
    for tg in trajectory_groups_P:
        # Check if this is branching (per-token advantages) or non-branching (scalar advantages)
        has_per_token_advs = any(
            transition.advantages is not None
            for traj in tg.trajectories_G
            for transition in traj.transitions
        )

        if has_per_token_advs:
            # Branching case: count tokens with zero advantage
            for traj in tg.trajectories_G:
                for transition in traj.transitions:
                    if transition.advantages is not None:
                        total_tokens_w_adv0 += sum(1 for adv in transition.advantages if abs(adv) < 1e-8)
        else:
            # Non-branching case: compute trajectory-level advantages
            rewards = np.array(tg.get_total_rewards())
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)

            for traj_idx, traj in enumerate(tg.trajectories_G):
                # Compute trajectory-level advantage
                if std_reward < 1e-8:
                    traj_advantage = 0.0
                else:
                    traj_advantage = (rewards[traj_idx] - mean_reward) / std_reward

                # If trajectory has zero advantage, count all its tokens
                if abs(traj_advantage) < 1e-8:
                    num_tokens = sum(len(t.ac.tokens) for t in traj.transitions)
                    total_tokens_w_adv0 += num_tokens

    metrics["env/all/total_num_tokens_w_adv0"] = total_tokens_w_adv0

    # 3. Token advantage entropy per group (mean over groups)
    entropies = []
    for tg in trajectory_groups_P:
        # Check if this is branching or non-branching
        has_per_token_advs = any(
            transition.advantages is not None
            for traj in tg.trajectories_G
            for transition in traj.transitions
        )

        all_advs = []
        if has_per_token_advs:
            # Branching case: collect all token advantages
            for traj in tg.trajectories_G:
                for transition in traj.transitions:
                    if transition.advantages is not None:
                        all_advs.extend(transition.advantages)
        else:
            # Non-branching case: compute trajectory-level advantages
            rewards = np.array(tg.get_total_rewards())
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)

            for traj_idx in range(len(tg.trajectories_G)):
                if std_reward < 1e-8:
                    traj_advantage = 0.0
                else:
                    traj_advantage = (rewards[traj_idx] - mean_reward) / std_reward
                all_advs.append(traj_advantage)

        if all_advs:
            # Convert to probability distribution using absolute values
            abs_advs = np.abs(all_advs)
            total_abs = np.sum(abs_advs)

            if total_abs > 1e-8:
                # Normalize to probabilities
                probs = abs_advs / total_abs
                # Compute entropy: -sum(p * log(p))
                # Filter out zero probabilities to avoid log(0)
                probs_nonzero = probs[probs > 1e-10]
                entropy = -np.sum(probs_nonzero * np.log(probs_nonzero))
                entropies.append(entropy)

    if entropies:
        metrics["env/all/token_adv_entropy_per_batch"] = np.mean(entropies)
    else:
        metrics["env/all/token_adv_entropy_per_batch"] = 0.0

    return metrics


def dataset_to_env_group_builders(dataset: RLDataset) -> list[EnvGroupBuilder]:
    """
    Get the whole dataset as a list of env group builders.
    """
    return list(itertools.chain(*[dataset.get_batch(i) for i in range(len(dataset))]))


class RLTestSetEvaluator(SamplingClientEvaluator):
    def __init__(self, dataset: RLDataset, max_tokens: int, name: str | None = None):
        self.env_group_builders_P = dataset_to_env_group_builders(dataset)
        self.max_tokens = max_tokens
        self.name = name

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        policy = TinkerTokenCompleter(sampling_client, max_tokens=self.max_tokens)
        trajectory_groups_P = await asyncio.gather(
            *[do_group_rollout(builder, policy) for builder in self.env_group_builders_P]
        )
        taglist_P = [builder.logging_tags() for builder in self.env_group_builders_P]
        metrics = compute_trajectory_metrics(trajectory_groups_P, taglist_P)

        if self.name is not None:
            metrics = {f"{self.name}/{k}": v for k, v in metrics.items()}
        return metrics
