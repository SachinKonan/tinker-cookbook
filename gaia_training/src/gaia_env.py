"""
GAIA Environment for RL Training
Follows the pattern from tinker_cookbook/recipes/tool_use/search/search_env.py
"""
import logging
import re
import asyncio
from typing import cast

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.rl.problem_env import ProblemEnv
from tinker_cookbook.rl.types import Action, StepResult

from .gaia_tools import GAIAToolClient

logger = logging.getLogger(__name__)

_CONNECTION_SEMAPHORE = asyncio.Semaphore(128)


# System prompt for GAIA tasks
GAIA_SYSTEM_PROMPT = """
You are an expert assistant that solves GAIA benchmark questions using available tools.

You have access to the following tools via <function_call>...</function_call> XML tags:

1. web_search: Search the web for information
   Usage: <function_call>{"name": "web_search", "args": {"query": "your search query"}}</function_call>

2. calculator: Perform mathematical calculations
   Usage: <function_call>{"name": "calculator", "args": {"expression": "2+2*3"}}</function_call>

3. fetch_webpage: Fetch and read webpage content
   Usage: <function_call>{"name": "fetch_webpage", "args": {"url": "https://example.com"}}</function_call>

Instructions:
1. Think step by step about what information you need
2. Use tools when necessary to gather information
3. After using a tool, think about whether you have enough information
4. When you have the final answer, provide it with the format: "Final Answer: <your answer>"

Example:
Question: What is the capital of France?
Thought: I can answer this directly without needing tools.
Final Answer: Paris

Example with tools:
Question: What is 15% of 240?
Thought: I need to calculate this.
<function_call>{"name": "calculator", "args": {"expression": "240 * 0.15"}}</function_call>
[Tool returns: 36.0]
Final Answer: 36
"""


def extract_final_answer(text: str) -> str:
    """Extract final answer from text"""
    match = re.search(r'Final Answer:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        # Clean up markdown
        answer = answer.replace('**', '').replace('*', '')
        return answer
    return ""


class GAIAEnv(ProblemEnv):
    """
    GAIA benchmark environment with tool use support
    """

    def __init__(
        self,
        question: str,
        answer: str,
        tool_client: GAIAToolClient,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        max_trajectory_tokens: int = 32 * 1024,
        max_num_steps: int = 7,
        format_coef: float = 0.01,  # Reward for having "Final Answer:" format
    ):
        super().__init__(renderer, convo_prefix, format_coef)
        self.question = question
        self.answer = answer
        self.tool_client = tool_client
        self.max_trajectory_tokens = max_trajectory_tokens
        self.max_num_steps = max_num_steps
        self.current_step = 0
        self.past_messages: list[renderers.Message] = (
            convo_prefix.copy() if convo_prefix else []
        )

    def get_question(self) -> str:
        return self.question

    def check_answer(self, sample_str: str) -> bool:
        """
        Check if the extracted answer matches ground truth

        Returns 1.0 if correct, 0.0 otherwise
        """
        model_answer = extract_final_answer(sample_str)
        if not model_answer:
            return False

        # Normalize and compare
        model_clean = model_answer.lower().strip()
        ground_truth_clean = str(self.answer).lower().strip()

        return model_clean == ground_truth_clean

    def check_format(self, sample_str: str) -> bool:
        """
        Check if the response has "Final Answer:" format

        Returns 1.0 if format is correct, 0.0 otherwise
        """
        return "Final Answer:" in sample_str

    async def initial_observation(self) -> tuple[tinker.ModelInput, StopCondition]:
        """Build initial observation with question"""
        convo = self.convo_prefix + [
            {"role": "user", "content": self.get_question()},
        ]
        self.past_messages = convo.copy()
        return self.renderer.build_generation_prompt(convo), self.stop_condition

    async def call_tool(self, tool_call: renderers.ToolCall) -> list[renderers.Message]:
        """Execute a tool call"""
        async with _CONNECTION_SEMAPHORE:
            return await self.tool_client.invoke(tool_call)

    async def step(self, action: Action) -> StepResult:
        """
        Execute one step of the environment

        If the action contains tool calls:
            - Execute the tool
            - Return next observation (episode_done=False)

        If the action is a final answer:
            - Compute reward
            - End episode (episode_done=True)
        """
        # Parse the model's response
        message, parse_success = self.renderer.parse_response(action)
        self.past_messages.append(message)
        self.current_step += 1

        # Failure result (used for errors or max steps)
        failure_result = StepResult(
            reward=0.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
        )

        # Check if max steps reached
        if self.current_step >= self.max_num_steps:
            logger.warning(f"Max steps ({self.max_num_steps}) reached")
            return failure_result

        # Handle tool calls
        if "tool_calls" in message and message["tool_calls"]:
            try:
                # Execute the first tool call
                tool_call = message["tool_calls"][0]
                tool_result_messages = await self.call_tool(tool_call)
                self.past_messages.extend(tool_result_messages)

                # Build next observation
                next_observation = self.renderer.build_generation_prompt(self.past_messages)

                # Check if trajectory is too long
                if next_observation.length > self.max_trajectory_tokens:
                    logger.warning(f"Trajectory too long: {next_observation.length} > {self.max_trajectory_tokens}")
                    return failure_result

                # Continue episode
                return StepResult(
                    reward=0.0,  # No reward for intermediate steps
                    episode_done=False,
                    next_observation=next_observation,
                    next_stop_condition=self.stop_condition,
                )

            except Exception as e:
                logger.error(f"Error executing tool: {e}")
                return failure_result

        # No tool calls - this should be the final answer
        else:
            # Compute reward components
            correct_format = float(parse_success) and float(self.check_format(message["content"]))
            correct_answer = float(self.check_answer(message["content"]))

            # Total reward: format_coef * (format_bonus - 1) + correct_answer
            # This gives: 0.01 * has_final_answer + 0.99 * is_correct
            # because format_bonus is 1.0 if has Final Answer, 0.0 otherwise
            # so format_coef * (1.0 - 1) + correct = 0 + correct when format is good
            # but format_coef * (0.0 - 1) + correct = -0.01 + correct when format is bad
            # We want: 0.01 * correct_format + 0.99 * correct_answer
            # So we use a different formula:
            total_reward = self.format_coef * correct_format + (1.0 - self.format_coef) * correct_answer

            return StepResult(
                reward=total_reward,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={
                    "format": correct_format,
                    "correct": correct_answer,
                },
            )

    @staticmethod
    def standard_fewshot_prefix() -> list[renderers.Message]:
        """Return standard system prompt for GAIA"""
        return [
            {
                "role": "system",
                "content": GAIA_SYSTEM_PROMPT,
            },
        ]
