"""
Training script for Tree GRPO with Search-R1.

Uses tree-based trajectory generation with Gemini branching.
"""
import asyncio
from datetime import datetime
from pathlib import Path

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.tool_use.search.tools import (
    ChromaToolClientConfig,
    EmbeddingConfig,
    RetrievalConfig,
)
from tinker_cookbook.recipes.tool_use.search_tree.tree_dataset import (
    SearchR1TreeDatasetBuilder,
)
from tinker_cookbook.rl import train


@chz.chz
class TreeGRPOConfig:
    """Configuration for Tree GRPO training."""

    # ===== Tree Parameters =====
    tree_m: int = 4
    """Number of root trajectories"""

    tree_k: int = 3
    """Branching factor (generates K-1 alternatives per branch)"""

    tree_d: int = 3
    """Maximum tree depth"""

    # ===== Model Parameters =====
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    lora_rank: int = 32
    renderer_name: str | None = None

    # ===== Training Parameters =====
    learning_rate: float = 4e-5
    batch_size: int = 512
    group_size: int = 8
    seed: int = 2
    max_tokens: int = 1024
    eval_every: int = 0

    # ===== Dataset Parameters =====
    max_trajectory_tokens: int = 8 * 1024

    # ===== Chroma Configuration =====
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection_name: str = "wiki_embeddings"
    n_results: int = 3
    embedding_model_name: str = "gemini-embedding-001"
    embedding_dim: int = 768

    # ===== Streaming Configuration =====
    stream_minibatch: bool = False
    num_minibatches: int = 4

    # ===== Logging Parameters =====
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(config: TreeGRPOConfig):
    """Main training function for Tree GRPO."""

    # Build chroma tool config
    chroma_tool_config = ChromaToolClientConfig(
        chroma_host=config.chroma_host,
        chroma_port=config.chroma_port,
        chroma_collection_name=config.chroma_collection_name,
        retrieval_config=RetrievalConfig(
            n_results=config.n_results,
            embedding_config=EmbeddingConfig(
                model_name=config.embedding_model_name,
                embedding_dim=config.embedding_dim,
            ),
        ),
    )

    # Get renderer name
    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(
        config.model_name
    )

    # Build tree dataset builder
    builder = SearchR1TreeDatasetBuilder(
        batch_size=config.batch_size,
        group_size=config.group_size,
        renderer_name=renderer_name,
        model_name_for_tokenizer=config.model_name,
        chroma_tool_config=chroma_tool_config,
        seed=config.seed,
        max_trajectory_tokens=config.max_trajectory_tokens,
        # Tree parameters
        tree_m=config.tree_m,
        tree_k=config.tree_k,
        tree_d=config.tree_d,
    )

    # Configure streaming minibatch
    if config.stream_minibatch:
        stream_minibatch_config = train.StreamMinibatchConfig(
            groups_per_batch=config.batch_size,
            num_minibatches=config.num_minibatches,
        )
        bs_str = f"bs{config.batch_size}_stream"
    else:
        stream_minibatch_config = None
        bs_str = f"bs{config.batch_size}"

    # Build run name
    model_name_short = config.model_name.lower().replace("/", "-")
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"tree_grpo_{model_name_short}_{bs_str}_gs{config.group_size}_"
        f"M{config.tree_m}_K{config.tree_k}_D{config.tree_d}_"
        f"seed{config.seed}_lr{config.learning_rate}_rank{config.lora_rank}_{date_and_time}"
    )

    # Set log path
    if config.log_path is not None:
        log_path = config.log_path
    else:
        log_path = f"/tmp/tinker-examples/tree_grpo/{run_name}"

    if config.wandb_name is not None:
        wandb_name = config.wandb_name
    else:
        wandb_name = run_name

    # Validate /tmp exists
    if not Path("/tmp").exists():
        raise ValueError("/tmp does not exist")

    # Check log directory
    cli_utils.check_log_dir(log_path, behavior_if_exists=config.behavior_if_log_dir_exists)

    # Build training config
    train_config = train.Config(
        model_name=config.model_name,
        log_path=log_path,
        dataset_builder=builder,
        learning_rate=config.learning_rate,
        max_tokens=config.max_tokens,
        eval_every=config.eval_every,
        wandb_project=config.wandb_project,
        wandb_name=wandb_name,
        lora_rank=config.lora_rank,
        stream_minibatch_config=stream_minibatch_config,
    )

    # Run training
    await train.main(train_config)


if __name__ == "__main__":
    config = chz.entrypoint(TreeGRPOConfig)
    asyncio.run(cli_main(config))
