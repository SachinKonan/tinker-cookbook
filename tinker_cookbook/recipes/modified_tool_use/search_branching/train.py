"""
CLI for Modified Tool Use Training with tree-based branching (using GAIA tools)
"""
import asyncio
import logging
from datetime import datetime
from pathlib import Path

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.modified_tool_use.search_branching.modified_search_env import SearchBranchingDatasetBuilder
from tinker_cookbook.rl import train

logger = logging.getLogger(__name__)
logging.getLogger("primp").setLevel(logging.WARNING)


@chz.chz
class CLIConfig:
    # Model parameters
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    lora_rank: int = 32
    renderer_name: str | None = None

    # Training parameters
    learning_rate: float = 4e-5
    batch_size: int = 512
    seed: int = 2
    max_tokens: int = 1024
    eval_every: int = 0

    # Dataset parameters
    group_size: int = 8  # Target number of trajectories after branching
    max_trajectory_tokens: int = 8 * 1024

    # GAIA tool configuration
    max_search_results: int = 5

    # Tree branching parameters
    src_trajectories: int = 2  # Number of root trajectories (typically group_size // 4)
    num_branches: int = 2  # Branching factor per completed trajectory

    # Streaming configuration
    stream_minibatch: bool = False
    num_minibatches: int = 4

    # Logging parameters
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(cli_config: CLIConfig):
    # Get renderer name
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )

    # Build dataset builder with branching support
    builder = SearchBranchingDatasetBuilder(
        batch_size=cli_config.batch_size,
        group_size=cli_config.group_size,
        src_trajectories=cli_config.src_trajectories,
        renderer_name=renderer_name,
        model_name_for_tokenizer=cli_config.model_name,
        max_search_results=cli_config.max_search_results,
        seed=cli_config.seed,
        max_trajectory_tokens=cli_config.max_trajectory_tokens,
    )

    # Configure streaming minibatch
    if cli_config.stream_minibatch:
        stream_minibatch_config = train.StreamMinibatchConfig(
            groups_per_batch=cli_config.batch_size,
            num_minibatches=cli_config.num_minibatches,
        )
        bs_str = f"bs{cli_config.batch_size}_stream"
    else:
        stream_minibatch_config = None
        bs_str = f"bs{cli_config.batch_size}"

    # Build run name
    model_name_short = cli_config.model_name.lower().replace("/", "-")
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"modified_tool_use_branching_{model_name_short}_{bs_str}_gs{cli_config.group_size}"
        f"_src{cli_config.src_trajectories}_branches{cli_config.num_branches}"
        f"_seed{cli_config.seed}_traj{cli_config.max_trajectory_tokens // 1024}k"
        f"_lr{cli_config.learning_rate}_rank{cli_config.lora_rank}_{date_and_time}"
    )

    # Set log path
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"/tmp/tinker-examples/modified_tool_use_branching/{run_name}"

    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name

    # Validate /tmp exists
    if not Path("/tmp").exists():
        raise ValueError("/tmp does not exist")

    # Check log directory
    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # Build training config with tree branching enabled
    config = train.Config(
        model_name=cli_config.model_name,
        log_path=log_path,
        dataset_builder=builder,
        learning_rate=cli_config.learning_rate,
        max_tokens=cli_config.max_tokens,
        eval_every=cli_config.eval_every,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        lora_rank=cli_config.lora_rank,
        stream_minibatch_config=stream_minibatch_config,
        # Enable tree branching
        use_tree_branching=True,
        src_trajectories=cli_config.src_trajectories,
        num_branches=cli_config.num_branches,
    )

    # Run training
    await train.main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
