"""Configuration for GAIA agent"""

class Config:
    """Configuration for GAIA ReAct agent"""

    # Agent settings
    MAX_ITERATIONS = 7  # Reduced from 15 to limit steps
    VERBOSE = True

    # Model settings
    DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"  # Larger model, better at following instructions
    DEFAULT_TEMPERATURE = 0.0
    DEFAULT_MAX_TOKENS = 4096

    # Tool settings
    SEARCH_RESULTS_PER_QUERY = 5
    MAX_FILE_SIZE_MB = 10
