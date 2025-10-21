"""
LLM-as-a-Judge for grading GAIA answers
"""
import logging
import os
import re

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are an expert evaluator grading a model's answer to a question from the GAIA benchmark.

QUESTION:
{question}

GROUND TRUTH ANSWER:
{ground_truth}

MODEL'S FULL RESPONSE (including reasoning and final answer):
{model_full_response}

EXTRACTED FINAL ANSWER FROM MODEL:
{extracted_answer}

---

Your task is to evaluate how well the model answered the question. Consider:
1. Whether the final answer is factually correct
2. Whether the reasoning/logic was sound
3. Whether the answer format is appropriate

GRADING SCALE (output ONLY the numerical score):
- 1.0: The answer is correct and properly formatted
- 0.8: The answer is correct but has minor formatting issues (e.g., extra text, slight variations)
- 0.3: The reasoning/logic was on the right track, but the final answer is wrong
- 0.0: The answer is completely incorrect or irrelevant

IMPORTANT:
- Be lenient with formatting variations as long as the core answer is correct
- Focus on factual correctness of the extracted answer compared to ground truth
- Only give 0.3 if the reasoning shown was reasonable but led to wrong conclusion
- Output ONLY a single number: 0.0, 0.3, 0.8, or 1.0

Score:"""


async def judge_answer(
    question: str,
    ground_truth: str,
    model_full_response: str,
    extracted_answer: str,
    model: str = "gpt-5-mini",
    max_retries: int = 3,
) -> float | None:
    """
    Use an LLM to judge whether the model answered correctly.

    Args:
        question: The GAIA question
        ground_truth: The correct answer
        model_full_response: Full model response including reasoning
        extracted_answer: The extracted final answer
        model: OpenAI model to use for judging
        max_retries: Number of retries on failure

    Returns:
        Score between 0.0 and 1.0, or None if API call fails
    """
    # Get API key from environment (check once outside retry loop)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment")
        return None

    for attempt in range(max_retries):
        try:
            # Create client
            client = AsyncOpenAI(api_key=api_key)

            # Format prompt
            prompt = JUDGE_PROMPT.format(
                question=question,
                ground_truth=ground_truth,
                model_full_response=model_full_response,
                extracted_answer=extracted_answer,
            )

            # Make API call
            # Note: GPT-5 models use max_completion_tokens and don't support temperature=0.0
            if model.startswith("gpt-5"):
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=500,
                )
            else:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=50,
                )

            # Extract score from response
            content = response.choices[0].message.content
            if content is None:
                logger.warning(f"LLM judge returned None content (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    continue
                return None

            content = content.strip()

            # Try to parse the score
            # Look for patterns like "0.0", "0.3", "0.8", "1.0"
            match = re.search(r'\b(0\.0|0\.3|0\.8|1\.0)\b', content)
            if match:
                score = float(match.group(1))
                logger.info(f"LLM judge score: {score}")
                return score
            else:
                logger.warning(f"Could not parse score from LLM response (attempt {attempt + 1}/{max_retries}): {repr(content)}")
                if attempt < max_retries - 1:
                    continue
                return None

        except Exception as e:
            logger.error(f"LLM judge API call failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                continue
            return None

    return None
