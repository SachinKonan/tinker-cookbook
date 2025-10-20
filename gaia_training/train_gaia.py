"""
RL Training for GAIA Benchmark
"""

import asyncio
from datetime import datetime
from pathlib import Path
import sys
import os

# Load environment variables from .env
from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl import train

sys.path.insert(0, os.path.dirname(__file__))

from src.gaia_dataset_builder import GAIADatasetBuilder


@chz.chz
class CLIConfig:
    # Model parameters
    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    lora_rank: int = 32
    renderer_name: str | None = None

    # Training parameters
    learning_rate: float = 1e-5
    batch_size: int = 8  # Number of questions per batch
    seed: int = 0
    max_tokens: int = 4096
    eval_every: int = 0

    # Dataset parameters
    group_size: int = 2  # GRPO group size (must match max_num_actors)
    max_trajectory_tokens: int = 32 * 1024
    max_num_steps: int = 7  # Agent max steps

    # Data path
    gaia_data_path: str = "data/inputs/gaia_data.json"

    # Streaming configuration
    stream_minibatch: bool = False
    num_minibatches: int = 4

    # Logging parameters
    log_path: str | None = None
    wandb_project: str | None = "gaia-rl"
    wandb_name: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(cli_config: CLIConfig):
    # Get renderer name
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )

    # Build dataset builder
    builder = GAIADatasetBuilder(
        batch_size=cli_config.batch_size,
        group_size=cli_config.group_size,
        renderer_name=renderer_name,
        model_name_for_tokenizer=cli_config.model_name,
        gaia_data_path=cli_config.gaia_data_path,
        seed=cli_config.seed,
        max_trajectory_tokens=cli_config.max_trajectory_tokens,
        max_num_steps=cli_config.max_num_steps,
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
    run_name = f"gaia_{model_name_short}_{bs_str}_gs{cli_config.group_size}_seed{cli_config.seed}_traj{cli_config.max_trajectory_tokens // 1024}k_lr{cli_config.learning_rate}_rank{cli_config.lora_rank}_{date_and_time}"

    # Set log path
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"/tmp/tinker-examples/rl_gaia/{run_name}"

    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name

    # Validate /tmp exists
    if not Path("/tmp").exists():
        raise ValueError("/tmp does not exist")

    # Check log directory
    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # Build training config
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
    )

    # Run training
    await train.main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
