"""
Configuration for Tree GRPO demo.
"""
import chz


@chz.chz
class TreeGRPODemoConfig:
    """Configuration for tree-based GRPO rollout demo."""

    # ===== Tree Hyperparameters =====
    tree_m: int = 4
    """Number of root trajectories to start with"""

    tree_k: int = 3
    """Branching factor: will generate K-1 alternatives at each branch point"""

    tree_d: int = 3
    """Maximum tree depth (roots are at depth 0)"""

    tree_n: int = 32
    """Target number of leaf trajectories"""

    # ===== Model Parameters =====
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    """Base model to use for policy"""

    lora_rank: int = 32
    """LoRA rank for fine-tuning"""

    renderer_name: str | None = None
    """Renderer name (auto-detected if None)"""

    max_tokens: int = 1024
    """Max tokens per trajectory step"""

    # ===== Search Environment Parameters =====
    chroma_host: str = "localhost"
    """Chroma server host"""

    chroma_port: int = 8000
    """Chroma server port"""

    chroma_collection_name: str = "wiki_embeddings"
    """Chroma collection name"""

    n_results: int = 3
    """Number of search results to retrieve"""

    embedding_model_name: str = "gemini-embedding-001"
    """Embedding model for search"""

    embedding_dim: int = 768
    """Embedding dimension"""

    max_trajectory_tokens: int = 8 * 1024
    """Maximum tokens per trajectory"""

    seed: int = 2
    """Random seed"""

    # ===== Gemini Parameters =====
    gemini_model: str = "gemini-2.0-flash-exp"
    """Gemini model for generating branching alternatives"""

    gemini_temperature: float = 0.9
    """Temperature for Gemini sampling"""

    gemini_top_p: float = 0.95
    """Top-p for Gemini sampling"""

    gemini_max_output_tokens: int = 2048
    """Max output tokens for Gemini"""

    # ===== Dataset Parameters =====
    num_problems: int = 4
    """Number of problems to sample for demo (uses batch_size in dataset builder)"""

    # ===== Output Parameters =====
    log_path: str = "/tmp/tree_grpo_demo"
    """Directory to save demo outputs"""

    save_trees: bool = True
    """Whether to save tree structures to disk"""

    verbose: bool = True
    """Whether to print detailed progress"""
