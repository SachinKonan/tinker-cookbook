"""
Tree-based dataset builder for Search-R1 with GRPO.

Extends the standard SearchR1Dataset to use tree-based trajectory generation.
"""
import asyncio
import logging
from typing import Sequence

from tinker_cookbook import renderers
from tinker_cookbook.recipes.tool_use.search.search_env import (
    SearchEnv,
    SearchR1Dataset,
    SearchR1DatasetBuilder,
)
from tinker_cookbook.recipes.tool_use.search.tools import ChromaToolClientConfig
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    Metrics,
    RLDataset,
    Trajectory,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)


class TreeProblemGroupBuilder(ProblemGroupBuilder):
    """
    Environment group builder for tree-based GRPO.

    Overrides compute_group_rewards to use prefix-based reward aggregation,
    which gives better advantage estimates by sharing information across
    trajectories with common token prefixes.
    """

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory]
    ) -> list[tuple[float, Metrics]]:
        """
        Compute rewards with prefix-based aggregation for tree structure.

        For now, this is a placeholder that uses the default implementation.
        In the future, this will:
        1. Detect tree structure from BranchedTrajectory references
        2. Group trajectories by shared token prefixes
        3. Compute advantages using prefix-aware aggregation
        4. Return better reward estimates

        Args:
            trajectory_group: List of trajectories (mix of Root and Branched)

        Returns:
            List of (reward, metrics) tuples
        """
        # TODO: Implement prefix-based reward aggregation
        # For now, use default (returns 0.0 for all)
        return await super().compute_group_rewards(trajectory_group)


class SearchR1TreeDataset(SearchR1Dataset):
    """
    Tree-based dataset for Search-R1 with GRPO.

    Extends SearchR1Dataset to use TreeProblemGroupBuilder for better
    reward estimation via prefix-based aggregation.
    """

    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        chroma_tool_client,
        convo_prefix: list | None = None,
        seed: int = 0,
        split: str = "train",
        subset_size: int | None = None,
        max_trajectory_tokens: int = 32 * 1024,
        # Tree parameters
        tree_m: int = 4,
        tree_k: int = 3,
        tree_d: int = 3,
    ):
        """
        Initialize tree-based Search-R1 dataset.

        Args:
            batch_size: Number of problems per batch
            group_size: Number of trajectories per problem (target_size)
            renderer: Renderer for token/text conversion
            chroma_tool_client: Client for Wikipedia search
            convo_prefix: Optional conversation prefix
            seed: Random seed
            split: Dataset split ('train' or 'test')
            subset_size: Optional subset size
            max_trajectory_tokens: Max tokens per trajectory
            tree_m: Number of root trajectories
            tree_k: Branching factor (generates K-1 alternatives)
            tree_d: Maximum tree depth
        """
        super().__init__(
            batch_size=batch_size,
            group_size=group_size,
            renderer=renderer,
            chroma_tool_client=chroma_tool_client,
            convo_prefix=convo_prefix,
            seed=seed,
            split=split,
            subset_size=subset_size,
            max_trajectory_tokens=max_trajectory_tokens,
        )

        # Store tree parameters
        self.tree_m = tree_m
        self.tree_k = tree_k
        self.tree_d = tree_d

    def _make_env_group_builder(self, row, group_size: int) -> EnvGroupBuilder:
        """
        Create a TreeProblemGroupBuilder for this problem.

        Overrides parent to use TreeProblemGroupBuilder instead of ProblemGroupBuilder.
        """
        from functools import partial

        env_thunk = partial(
            SearchEnv,
            row["question"],
            row["answer"],
            self.chroma_tool_client,
            self.renderer,
            convo_prefix=self.convo_prefix,
            max_trajectory_tokens=self.max_trajectory_tokens,
        )

        return TreeProblemGroupBuilder(
            env_thunk=env_thunk,
            num_envs=group_size,
        )


class SearchR1TreeDatasetBuilder(SearchR1DatasetBuilder):
    """
    Builder for tree-based Search-R1 datasets.

    Extends SearchR1DatasetBuilder to add tree parameters (M, K, D) and
    use SearchR1TreeDataset.
    """

    # Tree parameters
    tree_m: int = 4
    """Number of root trajectories to start with"""

    tree_k: int = 3
    """Branching factor (generates K-1 alternatives per branch)"""

    tree_d: int = 3
    """Maximum tree depth (based on max_num_calls in environment)"""

    async def __call__(self) -> tuple[RLDataset, None]:
        """
        Build tree-based Search-R1 datasets.

        Returns:
            Tuple of (train_dataset, None) - no test dataset for now
        """
        if self.convo_prefix == "standard":
            convo_prefix = SearchEnv.standard_fewshot_prefix()
        else:
            convo_prefix = self.convo_prefix

        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        # Import here to avoid circular dependency
        from tinker_cookbook.recipes.tool_use.search.tools import ChromaToolClient

        chroma_tool_client = await ChromaToolClient.create(self.chroma_tool_config)

        train_dataset = SearchR1TreeDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            chroma_tool_client=chroma_tool_client,
            convo_prefix=convo_prefix,
            split="train",
            seed=self.seed,
            max_trajectory_tokens=self.max_trajectory_tokens,
            # Tree parameters
            tree_m=self.tree_m,
            tree_k=self.tree_k,
            tree_d=self.tree_d,
        )

        return (train_dataset, None)
