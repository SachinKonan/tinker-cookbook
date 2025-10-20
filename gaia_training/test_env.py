"""
Test script for GAIA environment
Tests a single environment step-by-step to verify:
1. GAIAToolClient works
2. GAIAEnv handles tool calls correctly
3. Reward computation works
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from src.gaia_tools import GAIAToolClient
from src.gaia_env import GAIAEnv


async def test_single_env():
    """Test a single GAIA environment"""

    # Load one question
    df = pd.read_json("data/inputs/gaia_data.json")
    question_data = df.iloc[0]

    print("=" * 80)
    print("Testing GAIA Environment")
    print("=" * 80)
    print(f"\nQuestion: {question_data['Question']}")
    print(f"Answer: {question_data['Final answer']}")
    print()

    # Setup
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    tokenizer = get_tokenizer(model_name)
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    # Create tool client and env
    tool_client = GAIAToolClient()
    env = GAIAEnv(
        question=question_data['Question'],
        answer=question_data['Final answer'],
        tool_client=tool_client,
        renderer=renderer,
        convo_prefix=GAIAEnv.standard_fewshot_prefix(),
        max_trajectory_tokens=32 * 1024,
        max_num_steps=7,
    )

    print("Environment created successfully!")
    print(f"Tool schemas available: {len(tool_client.get_tool_schemas())}")
    for schema in tool_client.get_tool_schemas():
        print(f"  - {schema['name']}: {schema['description']}")

    # Get initial observation
    print("\n" + "=" * 80)
    print("Getting initial observation...")
    print("=" * 80)
    initial_obs, stop_condition = await env.initial_observation()
    print(f"Initial observation length: {initial_obs.length} tokens")
    print(f"Stop condition: {stop_condition}")

    # Test reward computation on example responses
    print("\n" + "=" * 80)
    print("Testing reward computation...")
    print("=" * 80)

    test_cases = [
        ("Final Answer: " + str(question_data['Final answer']), "Correct answer with format"),
        ("Final Answer: wrong answer", "Wrong answer with format"),
        ("The answer is " + str(question_data['Final answer']), "Correct answer without format"),
        ("Just some text", "No format, no answer"),
    ]

    for response, description in test_cases:
        has_format = env.check_format(response)
        is_correct = env.check_answer(response)
        reward = 0.01 * float(has_format) + 0.99 * float(is_correct)
        print(f"{description}:")
        print(f"  Response: {response[:50]}...")
        print(f"  Has format: {has_format}, Is correct: {is_correct}")
        print(f"  Reward: {reward:.3f}")
        print()

    print("=" * 80)
    print("Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_single_env())
