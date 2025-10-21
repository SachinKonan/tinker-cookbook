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
from .llm_judge import judge_answer

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
4. When you have the final answer, you MUST provide it in this exact format:

   Final Answer: <verifiable answer string>**

   The answer between "Final Answer: " and "**" will be extracted and verified.

Examples:

Question: What is the capital of France?
Thought: I can answer this directly without needing tools.
Final Answer: Paris**

Question: What is 15% of 240?
Thought: I need to calculate this.
<function_call>{"name": "calculator", "args": {"expression": "240 * 0.15"}}</function_call>
[Tool returns: 36.0]
Final Answer: 36**

IMPORTANT: Always end your final answer with "Final Answer: <verifiable answer string>**" - the ** is required!
"""


def extract_final_answer(text: str) -> str:
    """Extract final answer from text between 'Final Answer:' and '**'"""
    # First try to match the pattern with **
    match = re.search(r'Final Answer:\s*(.+?)\*\*', text, re.IGNORECASE | re.DOTALL)
    if match:
        answer = match.group(1).strip()
        return answer

    # Fallback to old pattern without ** (for backwards compatibility)
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
        use_llm_judge: bool = False,
        llm_judge_model: str = "gpt-5-mini",
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
        # Track trajectory metadata for logging
        self.total_tokens_used = 0
        self.max_tokens_exceeded = False
        self.max_turns_exceeded = False
        # LLM judge settings
        self.use_llm_judge = use_llm_judge
        self.llm_judge_model = llm_judge_model

    def get_question(self) -> str:
        return self.question

    def _build_failure_result(self) -> StepResult:
        """
        Build a failure result with full metadata for logging.
        Used when trajectory fails due to max_turns, max_tokens, or tool errors.
        """
        # Try to extract model answer from the last assistant message
        model_answer = ""
        for msg in reversed(self.past_messages):
            if msg.get("role") == "assistant":
                model_answer = extract_final_answer(msg.get("content", ""))
                break

        return StepResult(
            reward=0.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "format": 0.0,  # Failed
                "correct": 0.0,  # Failed
                # Add trajectory metadata for logging
                "past_messages": self.past_messages,
                "question": self.question,
                "ground_truth": str(self.answer),
                "model_answer": model_answer,
                "total_tokens": self.total_tokens_used,
                "total_turns": self.current_step,
                "max_tokens_exceeded": self.max_tokens_exceeded,
                "max_turns_exceeded": self.max_turns_exceeded,
            },
        )

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
        Check if the response has "Final Answer: <answer>**" format

        Returns 1.0 if format is correct, 0.0 otherwise
        """
        # Prefer the new format with **
        if re.search(r'Final Answer:\s*.+?\*\*', sample_str, re.IGNORECASE | re.DOTALL):
            return True
        # But accept old format without ** for backwards compatibility
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

        Three possible cases:
        1. Valid tool call: Execute tool, continue episode
        2. Valid final answer: Compute reward, end episode
        3. Invalid response: Provide error feedback, continue episode (if turns remain)
        """
        # Parse the model's response
        message, parse_success = self.renderer.parse_response(action)
        self.past_messages.append(message)
        self.current_step += 1

        # Check if max steps reached
        if self.current_step >= self.max_num_steps:
            logger.warning(f"Max steps ({self.max_num_steps}) reached")
            self.max_turns_exceeded = True
            return self._build_failure_result()

        # Determine what the model generated
        has_tool_call = "tool_calls" in message and message["tool_calls"]
        has_final_answer = "Final Answer:" in message["content"]

        # Case 1: Valid tool call - execute and continue
        if has_tool_call:
            try:
                # Execute the first tool call
                tool_call = message["tool_calls"][0]
                tool_result_messages = await self.call_tool(tool_call)
                self.past_messages.extend(tool_result_messages)

                # Add turns remaining message
                turns_remaining = self.max_num_steps - self.current_step
                turns_msg = {
                    "role": "system",
                    "content": f"[Turns remaining: {turns_remaining}]"
                }
                self.past_messages.append(turns_msg)

                # Build next observation
                next_observation = self.renderer.build_generation_prompt(self.past_messages)

                # Track total tokens
                self.total_tokens_used = next_observation.length

                # Check if trajectory is too long
                if next_observation.length > self.max_trajectory_tokens:
                    logger.warning(f"Trajectory too long: {next_observation.length} > {self.max_trajectory_tokens}")
                    self.max_tokens_exceeded = True
                    return self._build_failure_result()

                # Continue episode
                return StepResult(
                    reward=0.0,  # No reward for intermediate steps
                    episode_done=False,
                    next_observation=next_observation,
                    next_stop_condition=self.stop_condition,
                )

            except Exception as e:
                logger.error(f"Error executing tool: {e}")
                return self._build_failure_result()

        # Case 2: Valid final answer - compute reward and end
        elif has_final_answer:
            # Extract model answer for logging
            model_answer = extract_final_answer(message["content"])

            # Step 1: Check formatting
            correct_format = float(parse_success) and float(self.check_format(message["content"]))

            if not correct_format:
                # Bad formatting -> reward = 0.0
                total_reward = 0.0
                correct_answer = 0.0
            else:
                # Good formatting -> check answer
                if self.use_llm_judge:
                    # Use LLM judge with graduated scores
                    try:
                        llm_score = await judge_answer(
                            question=self.question,
                            ground_truth=str(self.answer),
                            model_full_response=message["content"],
                            extracted_answer=model_answer,
                            model=self.llm_judge_model,
                        )
                        if llm_score is not None:
                            if llm_score == 0.0:
                                # Wrong answer but good formatting -> 0.01 partial credit
                                total_reward = 0.01
                                correct_answer = 0.0
                            else:
                                # Return LLM score directly (0.3, 0.8, or 1.0)
                                total_reward = llm_score
                                correct_answer = llm_score
                        else:
                            # Fallback to exact match
                            logger.warning("LLM judge failed, falling back to exact match")
                            if self.check_answer(message["content"]):
                                total_reward = 1.0
                                correct_answer = 1.0
                            else:
                                total_reward = 0.0
                                correct_answer = 0.0
                    except Exception as e:
                        # Fallback to exact match
                        logger.warning(f"LLM judge error: {e}, falling back to exact match")
                        if self.check_answer(message["content"]):
                            total_reward = 1.0
                            correct_answer = 1.0
                        else:
                            total_reward = 0.0
                            correct_answer = 0.0
                else:
                    # Use exact match (binary)
                    if self.check_answer(message["content"]):
                        total_reward = 1.0
                        correct_answer = 1.0
                    else:
                        total_reward = 0.0
                        correct_answer = 0.0

            return StepResult(
                reward=total_reward,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={
                    "format": correct_format,
                    "correct": correct_answer,
                    # Add trajectory metadata for logging
                    "past_messages": self.past_messages,
                    "question": self.question,
                    "ground_truth": str(self.answer),
                    "model_answer": model_answer,
                    "total_tokens": self.total_tokens_used,
                    "total_turns": self.current_step,
                    "max_tokens_exceeded": self.max_tokens_exceeded,
                    "max_turns_exceeded": self.max_turns_exceeded,
                },
            )

        # Case 3: Invalid response - provide error feedback and continue
        else:
            logger.warning(f"Invalid response (no tool call or final answer): {message['content'][:100]}...")

            # Add error message
            error_msg = {
                "role": "system",
                "content": "Invalid Tool Call or Final Answer incorrectly formatted"
            }
            self.past_messages.append(error_msg)

            # Add turns remaining message
            turns_remaining = self.max_num_steps - self.current_step
            turns_msg = {
                "role": "system",
                "content": f"[Turns remaining: {turns_remaining}]"
            }
            self.past_messages.append(turns_msg)

            # Build next observation
            next_observation = self.renderer.build_generation_prompt(self.past_messages)

            # Track total tokens
            self.total_tokens_used = next_observation.length

            # Check if trajectory is too long
            if next_observation.length > self.max_trajectory_tokens:
                logger.warning(f"Trajectory too long: {next_observation.length} > {self.max_trajectory_tokens}")
                self.max_tokens_exceeded = True
                return self._build_failure_result()

            # Continue episode (give model a chance to recover)
            return StepResult(
                reward=0.0,  # No reward for invalid turn
                episode_done=False,
                next_observation=next_observation,
                next_stop_condition=self.stop_condition,
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
