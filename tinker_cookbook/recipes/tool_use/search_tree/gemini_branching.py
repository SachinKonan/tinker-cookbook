"""
Gemini-based branching completer for generating alternative trajectory continuations.
"""
import asyncio
import logging
import os
from typing import Any

from google import genai
from google.genai.types import GenerateContentConfig

logger = logging.getLogger(__name__)

# Semaphore to limit concurrent Gemini API calls
_GEMINI_SEMAPHORE = asyncio.Semaphore(40)


class GeminiBranchingCompleter:
    """
    Uses Gemini API to generate alternative completions for trajectory branching.

    This completer takes a partial conversation context and generates K-1 alternative
    ways to complete the current assistant message.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash-exp",
        temperature: float = 0.9,
        top_p: float = 0.95,
        max_output_tokens: int = 2048,
    ):
        """
        Initialize the Gemini branching completer.

        Args:
            model_name: Gemini model to use
            temperature: Sampling temperature for diversity
            top_p: Nucleus sampling parameter
            max_output_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens

        # Create Gemini client using VertexAI
        from tinker_cookbook.recipes.tool_use.search.embedding import get_gemini_client

        self.client = get_gemini_client()

        logger.info(f"Initialized GeminiBranchingCompleter with model {model_name}")

    async def generate_alternatives(
        self,
        context: str,
        parent_reward: float,
        k_minus_1: int,
    ) -> list[str]:
        """
        Generate K-1 alternative completions given a context and reward.

        Args:
            context: Full conversation context up to the branch point
            parent_reward: Final reward achieved by the parent trajectory
            k_minus_1: Number of alternatives to generate

        Returns:
            List of K-1 alternative completion strings
        """
        prompt = self._build_prompt(context, parent_reward, k_minus_1)

        alternatives = []

        # Generate alternatives one at a time with semaphore
        tasks = [
            self._generate_single_alternative(prompt, i) for i in range(k_minus_1)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to generate alternative {i}: {result}")
            else:
                alternatives.append(result)

        return alternatives

    async def _generate_single_alternative(self, prompt: str, index: int) -> str:
        """
        Generate a single alternative completion with semaphore protection.

        Args:
            prompt: The branching prompt
            index: Alternative index (for logging)

        Returns:
            Generated completion text
        """
        async with _GEMINI_SEMAPHORE:
            try:
                # Use the async Gemini API
                config = GenerateContentConfig(
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_output_tokens=self.max_output_tokens,
                )

                response = await self.client.aio.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )

                # Extract text from response
                if response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if candidate.content and candidate.content.parts:
                        return candidate.content.parts[0].text
                    else:
                        raise ValueError(f"Empty response from Gemini for alternative {index}")
                else:
                    raise ValueError(f"No candidates in Gemini response for alternative {index}")

            except Exception as e:
                logger.error(f"Gemini API error for alternative {index}: {e}")
                raise

    def _build_prompt(
        self,
        context: str,
        parent_reward: float,
        k_minus_1: int,
    ) -> str:
        """
        Build the prompting instruction for Gemini.

        Args:
            context: Full conversation context
            parent_reward: Reward from parent trajectory
            k_minus_1: Number of alternatives needed

        Returns:
            Complete prompt string
        """
        prompt = f"""You are helping to generate alternative reasoning paths for an AI assistant solving a question using Wikipedia search tools.

Below is a partial conversation where the assistant is in the middle of responding. The assistant's partial response achieved a final reward of {parent_reward:.2f}.

Your task is to generate a PLAUSIBLE ALTERNATIVE way to complete the current assistant message. The completion should:
1. Be different from what you might expect as the most obvious continuation
2. Still be a reasonable and coherent continuation of the thought
3. Maintain the same format and style as the partial response
4. Be between 50-200 tokens

CONTEXT:
{context}

Generate ONLY the completion text for the assistant's partial message. Do not include any preamble, explanation, or the original context. Start directly with the continuation.

ALTERNATIVE COMPLETION:"""

        return prompt
