"""
GAIA Dataset Builder for RL Training
Follows pattern from tinker_cookbook/recipes/tool_use/search/search_env.py
"""
import random
from functools import partial
from typing import Literal, Sequence
import pandas as pd

import chz
from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

from .gaia_env import GAIAEnv
from .gaia_tools import GAIAToolClient


class GAIADataset(RLDataset):
    """GAIA RL Dataset"""

    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        tool_client: GAIAToolClient,
        gaia_data: list[dict],
        convo_prefix: list[renderers.Message] | None = None,
        seed: int = 0,
        max_trajectory_tokens: int = 32 * 1024,
        max_num_steps: int = 7,
        use_llm_judge: bool = False,
        llm_judge_model: str = "gpt-5-mini",
    ):
        self.batch_size = batch_size
        self.group_size = group_size
        self.max_trajectory_tokens = max_trajectory_tokens
        self.max_num_steps = max_num_steps
        self.renderer = renderer
        self.convo_prefix = convo_prefix
        self.tool_client = tool_client
        self.seed = seed
        self.ds = gaia_data
        self.use_llm_judge = use_llm_judge
        self.llm_judge_model = llm_judge_model

        # Shuffle with seed
        rng = random.Random(self.seed)
        rng.shuffle(self.ds)

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        """Get a batch of environment group builders"""
        return [
            self._make_env_group_builder(row, self.group_size)
            for row in self.ds[index * self.batch_size : (index + 1) * self.batch_size]
        ]

    def __len__(self) -> int:
        return len(self.ds) // self.batch_size

    def _make_env_group_builder(self, row: dict, group_size: int) -> ProblemGroupBuilder:
        """Create a ProblemGroupBuilder for a single question"""
        return ProblemGroupBuilder(
            env_thunk=partial(
                GAIAEnv,
                question=row["Question"],
                answer=row["Final answer"],
                tool_client=self.tool_client,
                renderer=self.renderer,
                convo_prefix=self.convo_prefix,
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_num_steps=self.max_num_steps,
                use_llm_judge=self.use_llm_judge,
                llm_judge_model=self.llm_judge_model,
            ),
            num_envs=group_size,
        )


@chz.chz
class GAIADatasetBuilder(RLDatasetBuilder):
    """Builder for GAIA RL Dataset"""

    batch_size: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    gaia_data_path: str
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard"
    seed: int = 0
    max_trajectory_tokens: int = 32 * 1024
    max_num_steps: int = 7
    use_llm_judge: bool = False
    llm_judge_model: str = "gpt-5-mini"

    async def __call__(self) -> tuple[GAIADataset, None]:
        """Build the GAIA dataset"""

        # Load convo prefix
        if self.convo_prefix == "standard":
            convo_prefix = GAIAEnv.standard_fewshot_prefix()
        else:
            convo_prefix = self.convo_prefix

        # Get tokenizer and renderer
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        # Create tool client
        tool_client = GAIAToolClient()

        # Load GAIA data
        df = pd.read_json(self.gaia_data_path)
        gaia_data = df.to_dict('records')

        # Create dataset
        train_dataset = GAIADataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            tool_client=tool_client,
            gaia_data=gaia_data,
            convo_prefix=convo_prefix,
            seed=self.seed,
            max_trajectory_tokens=self.max_trajectory_tokens,
            max_num_steps=self.max_num_steps,
            use_llm_judge=self.use_llm_judge,
            llm_judge_model=self.llm_judge_model,
        )

        return (train_dataset, None)
