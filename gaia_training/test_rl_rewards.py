"""
Test script for GAIA RL reward functions
"""
import sys
sys.path.insert(0, '.')

from rl_loop_gaia import extract_final_answer, compute_gaia_reward


def test_extract_final_answer():
    """Test final answer extraction"""
    print("Testing extract_final_answer()...")

    test_cases = [
        ("Final Answer: egalitarian", "egalitarian"),
        ("Final Answer: **social**", "social"),
        ("some text\nFinal Answer: 42\nmore text", "42"),
        ("no final answer here", ""),
        ("Final Answer: The answer is 123", "The answer is 123"),
    ]

    for text, expected in test_cases:
        result = extract_final_answer(text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} Input: {text[:50]}...")
        print(f"     Expected: '{expected}', Got: '{result}'")

    print()


def test_compute_gaia_reward():
    """Test reward computation"""
    print("Testing compute_gaia_reward()...")

    test_cases = [
        # (messages, ground_truth, expected_reward, description)
        (
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "Final Answer: 4"}
            ],
            "4",
            1.0,  # 0.01 * 1 + 0.99 * 1 = 1.0
            "Correct answer with Final Answer format"
        ),
        (
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "Final Answer: 5"}
            ],
            "4",
            0.01,  # 0.01 * 1 + 0.99 * 0 = 0.01
            "Wrong answer but has Final Answer format"
        ),
        (
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "The answer is 4"}
            ],
            "4",
            0.0,  # 0.01 * 0 + 0.99 * 0 = 0.0
            "Correct answer but missing Final Answer format"
        ),
        (
            [
                {"role": "user", "content": "What is the capital?"},
                {"role": "assistant", "content": "Let me think..."},
                {"role": "user", "content": "Observation: ..."},
                {"role": "assistant", "content": "Final Answer: Paris"}
            ],
            "Paris",
            1.0,
            "Multi-turn conversation with correct answer"
        ),
        (
            [],
            "anything",
            0.0,
            "Empty messages"
        ),
    ]

    for messages, ground_truth, expected, description in test_cases:
        reward = compute_gaia_reward(messages, ground_truth)
        status = "✓" if abs(reward - expected) < 0.001 else "✗"
        print(f"  {status} {description}")
        print(f"     Expected: {expected}, Got: {reward}")

    print()


def test_conversation_to_tokens():
    """Test conversation to tokens conversion"""
    print("Testing conversation_to_tokens()...")

    # We need tokenizer and renderer for this
    try:
        from tinker_cookbook import model_info, renderers
        from tinker_cookbook.tokenizer_utils import get_tokenizer
        from rl_loop_gaia import conversation_to_tokens

        model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
        tokenizer = get_tokenizer(model_name)
        renderer_name = model_info.get_recommended_renderer_name(model_name)
        renderer = renderers.get_renderer(renderer_name, tokenizer)

        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "Final Answer: 4"}
        ]

        all_tokens, obs_len = conversation_to_tokens(messages, renderer, tokenizer)

        print(f"  ✓ Converted {len(messages)} messages to {len(all_tokens)} tokens")
        print(f"     Observation length: {obs_len}")
        print(f"     Generation length: {len(all_tokens) - obs_len - 1}")

    except Exception as e:
        print(f"  ✗ Error: {e}")

    print()


if __name__ == "__main__":
    print("="*80)
    print("GAIA RL Reward Function Tests")
    print("="*80)
    print()

    test_extract_final_answer()
    test_compute_gaia_reward()
    test_conversation_to_tokens()

    print("="*80)
    print("Tests complete!")
    print("="*80)
