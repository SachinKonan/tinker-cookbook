"""
Quick test of the LLM judge functionality
"""
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)

from src.llm_judge import judge_answer


async def main():
    print("Testing LLM Judge...")
    print(f"OpenAI API Key present: {bool(os.getenv('OPENAI_API_KEY'))}")
    print()

    # Test case 1: Correct answer
    print("=" * 60)
    print("Test 1: Correct answer")
    print("=" * 60)
    score1 = await judge_answer(
        question="What is the capital of France?",
        ground_truth="Paris",
        model_full_response="Let me think... France's capital is Paris. Final Answer: Paris**",
        extracted_answer="Paris",
    )
    print(f"Score: {score1}")
    print()

    # Test case 2: Correct but formatting issue
    print("=" * 60)
    print("Test 2: Correct answer with extra text")
    print("=" * 60)
    score2 = await judge_answer(
        question="What is 2 + 2?",
        ground_truth="4",
        model_full_response="Simple math: 2 + 2 = 4. Final Answer: The answer is 4 (four)**",
        extracted_answer="The answer is 4 (four)",
    )
    print(f"Score: {score2}")
    print()

    # Test case 3: Right logic, wrong answer
    print("=" * 60)
    print("Test 3: Right reasoning but wrong final answer")
    print("=" * 60)
    score3 = await judge_answer(
        question="What is the square root of 144?",
        ground_truth="12",
        model_full_response="To find the square root of 144, I need to find a number that when multiplied by itself gives 144. Let me think... 11 * 11 = 121, and 13 * 13 = 169. So it must be around there. I'll estimate it's 11.5. Final Answer: 11.5**",
        extracted_answer="11.5",
    )
    print(f"Score: {score3}")
    print()

    # Test case 4: Completely wrong
    print("=" * 60)
    print("Test 4: Completely incorrect")
    print("=" * 60)
    score4 = await judge_answer(
        question="What is the capital of France?",
        ground_truth="Paris",
        model_full_response="The capital of France is London. Final Answer: London**",
        extracted_answer="London",
    )
    print(f"Score: {score4}")
    print()

    print("=" * 60)
    print("Summary:")
    print(f"  Test 1 (Correct): {score1} (expected: 1.0)")
    print(f"  Test 2 (Correct w/ formatting): {score2} (expected: 0.8 or 1.0)")
    print(f"  Test 3 (Right logic, wrong answer): {score3} (expected: 0.3)")
    print(f"  Test 4 (Wrong): {score4} (expected: 0.0)")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
