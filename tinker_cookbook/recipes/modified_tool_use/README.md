# Modified Tool Use Training with GAIA Tools

This recipe is a modified version of the [Search-R1 tool use training](../tool_use/search/) that uses GAIA tools instead of Chroma/Gemini dependencies.

## Key Differences from tool_use/search

- **Tools**: Uses `GAIAToolClient` with web_search, calculator, and fetch_webpage tools (imported from `gaia_training/src/gaia_tools.py`)
- **No External Services**: No Chroma vector database or Gemini embedding API required
- **Simpler Setup**: Direct instantiation of tool client with no async initialization

## Available Tools

1. **web_search**: Search the web using DuckDuckGo
   - Returns titles, URLs, and snippets
   - Configurable max results

2. **calculator**: Perform mathematical calculations
   - Supports basic arithmetic operations: +, -, *, /, parentheses
   - Safe evaluation with restricted scope

3. **fetch_webpage**: Fetch and extract text content from webpages
   - Uses BeautifulSoup for HTML parsing
   - Automatic content truncation for long pages

## Installation

Install the required dependencies:

```bash
pip install requests beautifulsoup4 lxml ddgs python-dotenv
```

Or with uv:

```bash
uv pip install requests beautifulsoup4 lxml ddgs python-dotenv
```

**Note**: Environment variables (e.g., API keys) are automatically loaded from `/n/fs/vision-mix/sk7524/tinker-cookbook/.env` at startup. No manual configuration needed!

## Running This Demo

### Example Command

Train a `Qwen3-4B-Instruct-2507` model with default hyperparameters:

```bash
python -m tinker_cookbook.recipes.modified_tool_use.train
```

### Configuration Options

```bash
python -m tinker_cookbook.recipes.modified_tool_use.train \
    model_name="Qwen/Qwen3-4B-Instruct-2507" \
    batch_size=512 \
    group_size=8 \
    learning_rate=4e-5 \
    max_search_results=5 \
    wandb_project="my-tool-use-project"
```

### Key Parameters

- `model_name`: Model to train (default: Qwen/Qwen3-4B-Instruct-2507)
- `batch_size`: Number of problems per batch (default: 512)
- `group_size`: Group size for GRPO (default: 8)
- `learning_rate`: Learning rate (default: 4e-5)
- `max_search_results`: Max web search results per query (default: 5)
- `max_trajectory_tokens`: Max tokens per trajectory (default: 8192)
- `stream_minibatch`: Enable streaming minibatch training (default: False)

## Dataset

Uses the same SearchR1 dataset from HuggingFace:
- Training data: Natural Questions, TriviaQA, HotpotQA, 2WikiMultihopQA
- Automatic download from `PeterJinGo/nq_hotpotqa_train`

## Training Details

- **Environment**: `SearchEnv` from `modified_search_env.py`
- **Reward**: Format-based (0.01 for correct format) + correctness (1.0 for correct answer)
- **Max tool calls**: 4 per episode
- **Algorithm**: GRPO (Group Relative Policy Optimization)

## Monitoring

A successful run will show:
- `env/all/turns_per_episode` increasing above 2 turns (multi-turn search behavior)
- `env/all/correct` increasing over time
- Tool usage patterns in trajectory logs

## Code Structure

```
modified_tool_use/
├── __init__.py
├── tools.py                  # Imports GAIAToolClient from gaia_training
├── modified_search_env.py    # SearchEnv adapted for GAIA tools
├── train.py                  # Training CLI (simplified config)
└── README.md                 # This file
```

## Extending This Recipe

To modify the tools or add new ones:
1. Edit `gaia_training/src/gaia_tools.py` to add/modify tool implementations
2. Update the system prompt in `modified_search_env.py` to describe new tools
3. Update tool name validation in `SearchEnv.step()` method

## Comparison to Original tool_use/search

| Feature | tool_use/search | modified_tool_use |
|---------|----------------|-------------------|
| Tools | Wikipedia search via Chroma | Web search, calculator, webpage fetch |
| Dependencies | ChromaDB, Gemini API | requests, beautifulsoup4, ddgs |
| Setup | Requires running Chroma service | Direct instantiation |
| Tool Client | ChromaToolClient (async create) | GAIAToolClient (sync init) |
| Code Reuse | Self-contained | Imports from gaia_training |

## Credits

Based on:
- [Search-R1 paper](https://arxiv.org/pdf/2503.09516) by Jin et al.
- Original Tinker Cookbook implementation in `tool_use/search/`
- GAIA benchmark tools from `gaia_training/`
