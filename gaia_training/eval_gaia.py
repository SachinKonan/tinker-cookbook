"""
Evaluation script for GAIA benchmark
Runs trained model on GAIA test set with different group sizes
"""

import asyncio
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import chz
import pandas as pd
from tinker_cookbook import model_info
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook import renderers

from src.gaia_tools import GAIAToolClient
from src.gaia_env import GAIAEnv
from src.gaia_dataset_builder import GAIADatasetBuilder


@chz.chz
class EvalConfig:
    # Model parameters
    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    checkpoint_path: str | None = None  # Path to trained checkpoint
    renderer_name: str | None = None

    # Evaluation parameters
    group_size: int = 2  # Number of samples per question
    max_num_steps: int = 7
    max_trajectory_tokens: int = 32 * 1024
    seed: int = 0

    # Data
    gaia_data_path: str = "data/inputs/gaia_data.json"
    num_questions: int | None = None  # If None, evaluate on all

    # Output
    output_path: str | None = None


async def eval_main(config: EvalConfig):
    """Run evaluation on GAIA dataset"""

    print("=" * 80)
    print("GAIA Evaluation")
    print("=" * 80)
    print(f"Model: {config.model_name}")
    print(f"Checkpoint: {config.checkpoint_path or 'Base model (no checkpoint)'}")
    print(f"Group size: {config.group_size}")
    print(f"Max steps: {config.max_num_steps}")
    print(f"Data: {config.gaia_data_path}")
    print("=" * 80)
    print()

    # Load data
    df = pd.read_json(config.gaia_data_path)
    if config.num_questions:
        df = df.head(config.num_questions)

    print(f"Evaluating on {len(df)} questions")
    print()

    # Setup renderer and tools
    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(
        config.model_name
    )
    tokenizer = get_tokenizer(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    tool_client = GAIAToolClient()

    # Results storage
    results = []

    # Evaluate each question
    for idx, row in df.iterrows():
        question = row['Question']
        answer = row['Final answer']

        print(f"Question {idx + 1}/{len(df)}: {question[:80]}...")

        # Create environment
        env = GAIAEnv(
            question=question,
            answer=answer,
            tool_client=tool_client,
            renderer=renderer,
            convo_prefix=GAIAEnv.standard_fewshot_prefix(),
            max_trajectory_tokens=config.max_trajectory_tokens,
            max_num_steps=config.max_num_steps,
        )

        # Run group_size trajectories
        group_rewards = []
        for sample_idx in range(config.group_size):
            # TODO: Sample from model and run episode
            # For now, placeholder
            reward = 0.0
            group_rewards.append(reward)

        avg_reward = sum(group_rewards) / len(group_rewards)
        max_reward = max(group_rewards)

        results.append({
            'question': question,
            'answer': answer,
            'avg_reward': avg_reward,
            'max_reward': max_reward,
            'group_size': config.group_size,
        })

        print(f"  Avg reward: {avg_reward:.3f}, Max reward: {max_reward:.3f}")

    # Compute overall metrics
    overall_avg = sum(r['avg_reward'] for r in results) / len(results)
    overall_max = sum(r['max_reward'] for r in results) / len(results)

    print()
    print("=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Questions evaluated: {len(results)}")
    print(f"Overall avg reward: {overall_avg:.3f}")
    print(f"Overall max reward (best of {config.group_size}): {overall_max:.3f}")
    print("=" * 80)

    # Save results
    if config.output_path:
        results_df = pd.DataFrame(results)
        results_df.to_json(config.output_path, orient='records', indent=2)
        print(f"\nResults saved to: {config.output_path}")


if __name__ == "__main__":
    eval_config = chz.entrypoint(EvalConfig)
    asyncio.run(eval_main(eval_config))
