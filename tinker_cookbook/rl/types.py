"""
Basic interfaces and types for reinforcement learning.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence, TypeAlias

import chz
import tinker
from tinker_cookbook.completers import StopCondition, TokensWithLogprobs
from tinker_cookbook.utils.misc_utils import safezip

Action: TypeAlias = list[int]
Observation: TypeAlias = tinker.ModelInput
Logprobs: TypeAlias = list[float]
Metrics: TypeAlias = dict[str, float | int]


@dataclass
class StepResult:
    reward: float
    episode_done: bool
    next_observation: Observation
    next_stop_condition: StopCondition
    metrics: Metrics = field(default_factory=dict)


@dataclass
class Transition:
    ob: Observation
    ac: TokensWithLogprobs
    reward: float
    episode_done: bool
    metrics: Metrics = field(default_factory=dict)


class Env(ABC):
    """
    Stateful environment that a single agent interacts with.
    Discard after running for one episode.
    """

    @abstractmethod
    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        pass

    @abstractmethod
    async def step(self, action: Action) -> StepResult:
        pass


@dataclass(frozen=True)
class Trajectory:
    """
    A sequence of observations and actions, resulting from running a single agent in a single
    environment.
    """

    transitions: list[Transition]
    final_ob: Observation


@dataclass(frozen=True)
class Reference:
    """
    A reference to a parent trajectory's branch point.

    Used in tree-based GRPO to track where a trajectory branched from its parent.
    """

    source_trajectory: Trajectory
    """The parent trajectory this was branched from"""

    transition_idx: int
    """Index of the transition (assistant message) where branching occurred"""

    token_idx: int
    """Token position within that transition where the branch point was"""


@dataclass(frozen=True)
class RootTrajectory(Trajectory):
    """
    A root trajectory (no parent) in tree-based GRPO.

    Functionally identical to Trajectory, used as a type marker to distinguish
    roots from branched trajectories.
    """
    pass


@dataclass(frozen=True)
class BranchedTrajectory(Trajectory):
    """
    A trajectory branched from a parent trajectory in tree-based GRPO.

    Contains references to the parent trajectory and branch point(s).
    """

    references: list[Reference] = field(default_factory=list)
    """List of references to parent trajectories (typically one, but could be multiple in complex cases)"""


class EnvGroupBuilder(ABC):
    """
    Builds a group of environments. The group will be used in the following way:

    - Algorithms like GRPO will center rewards across the group.
    - The reward function (compute_group_rewards) has access to the trajectories from the
      whole group, even though many reward functions will evaluate each one independently.

      - For example, this enables us to use pairwise reward models that look at a pair of
        trajectories at a time. With such a reward model, we effectively have a multi-agent
        environment, where the agents are playing a zero-sum game.

    Groups can be used in two ways, in practice:

    - To define a multi-agent environment
    - As a part of the *algorithm* (e.g. GRPO), when dealing with single-agent tasks.
    """

    @abstractmethod
    async def make_envs(self) -> Sequence[Env]:
        pass

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory]
    ) -> list[tuple[float, Metrics]]:
        """
        This computes a final reward for each trajectory that depends on the whole group.
        Note that there are also per-timestep rewards returned by the Env.step() method.
        The total reward is the sum of the per-timestep rewards plus the final group reward
        computed here. Defining a group reward is optional -- by default, the group reward
        is 0 and we only use the per-timestep rewards.
        """
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        """
        This is just used for logging. We often want to aggregate metrics (like rewards
        or episode lengths) per-environment, or across a group of related environments.

        Most commonly, you'd return a short name for the environment, such as ['gsm'] for
        grade school math. You also might want a few tags at different levels of granularity,
        e.g., ['gsm', 'math', 'rlvr']
        """
        return []


@dataclass
class TrajectoryGroup:
    """
    A group of trajectories, resulting from instantiating a group of environments using an
    EnvGroupBuilder, doing a rollout for each environment, and computing the rewards.
    """

    trajectories_G: list[Trajectory]
    final_rewards_G: list[float]  # computed by the EnvGroupBuilder, looking at whole group
    metrics_G: list[Metrics]

    def get_total_rewards(self) -> list[float]:
        """
        Get the total reward (i.e., the return) of each trajectory (episode) in the group.
        The total reward is the sum of the per-timestep rewards plus the final group reward
        computed by the EnvGroupBuilder.
        """
        return [
            sum(transition.reward for transition in trajectory.transitions) + final_reward
            for trajectory, final_reward in safezip(self.trajectories_G, self.final_rewards_G)
        ]


@dataclass
class TreeTrajectoryGroup(TrajectoryGroup):
    """
    A group of trajectories organized in a tree structure for tree-based GRPO.

    Compatible with TrajectoryGroup interface but trajectories may be RootTrajectory
    or BranchedTrajectory types. The tree structure enables:
    - Token-level advantage computation using shared prefixes
    - Better reward estimates by aggregating across related trajectories
    - Tracking branching history for analysis
    """

    def get_roots(self) -> list[RootTrajectory]:
        """Get all root trajectories (no parent)."""
        return [t for t in self.trajectories_G if isinstance(t, RootTrajectory)]

    def get_branched(self) -> list[BranchedTrajectory]:
        """Get all branched trajectories (have parent)."""
        return [t for t in self.trajectories_G if isinstance(t, BranchedTrajectory)]

    def get_tree_statistics(self) -> dict[str, int | float]:
        """Get statistics about the tree structure."""
        roots = self.get_roots()
        branched = self.get_branched()

        # Calculate average depth (number of references)
        depths = [len(t.references) for t in branched]
        avg_depth = sum(depths) / len(depths) if depths else 0

        return {
            "total_trajectories": len(self.trajectories_G),
            "num_roots": len(roots),
            "num_branched": len(branched),
            "max_depth": max(depths) if depths else 0,
            "avg_depth": avg_depth,
        }


class RLDataset(ABC):
    """
    A dataset that produces batches of EnvGroups. This is the kind of dataset used by
    training algorithms.
    """

    @abstractmethod
    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


@chz.chz
class RLDatasetBuilder:
    """
    Abstract class for building RL datasets.
    """

    @abstractmethod
    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        """
        Return RLDataset (for training) and an optional RL dataset for testing
        """
        pass
