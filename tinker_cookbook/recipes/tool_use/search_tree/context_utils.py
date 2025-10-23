"""
Utilities for reconstructing conversation context from trajectories.

Used for tree-based GRPO branching to extract and replay conversation history.
"""
import logging
from typing import Any

from tinker_cookbook import renderers
from tinker_cookbook.rl.types import Trajectory

logger = logging.getLogger(__name__)


def reconstruct_messages_from_trajectory(
    trajectory: Trajectory, renderer: renderers.Renderer
) -> list[renderers.Message]:
    """
    Reconstruct the full conversation (list of messages) from a trajectory.

    This parses each transition's tokens to rebuild the assistant messages
    and combines them with observations to reconstruct the full conversation.

    Args:
        trajectory: The trajectory to reconstruct from
        renderer: Renderer for parsing tokens to messages

    Returns:
        List of messages representing the full conversation
    """
    messages: list[renderers.Message] = []

    for i, transition in enumerate(trajectory.transitions):
        # Parse the observation to get prior context (if it's the first transition)
        # For subsequent transitions, the observation is the continuation prompt
        # which includes the previous assistant message + tool results

        # Parse the action (assistant response) from tokens
        assistant_message, parse_success = renderer.parse_response(transition.ac.tokens)

        if not parse_success:
            logger.warning(f"Failed to parse assistant message at transition {i}")

        messages.append(assistant_message)

    return messages


def reconstruct_full_context_up_to_branch(
    trajectory: Trajectory,
    renderer: renderers.Renderer,
    branch_transition_idx: int,
    branch_token_idx: int,
    system_messages: list[renderers.Message] | None = None,
) -> str:
    """
    Reconstruct the full conversation context up to a branch point.

    This includes:
    1. System messages (if provided)
    2. All complete messages before the branch transition
    3. Partial assistant message up to the branch token position

    Args:
        trajectory: The trajectory being branched from
        renderer: Renderer for token/text conversion
        branch_transition_idx: Index of the transition being branched
        branch_token_idx: Token position within that transition
        system_messages: Optional system/user context messages

    Returns:
        Full text context string for Gemini
    """
    # Start with system messages if provided
    context_parts = []

    if system_messages:
        for msg in system_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            context_parts.append(f"[{role.upper()}]\n{content}\n")

    # Add all complete transitions before the branch point
    for i in range(branch_transition_idx):
        transition = trajectory.transitions[i]
        assistant_message, _ = renderer.parse_response(transition.ac.tokens)
        content = assistant_message.get("content", "")
        context_parts.append(f"[ASSISTANT]\n{content}\n")

    # Add partial assistant message up to branch point
    if branch_transition_idx < len(trajectory.transitions):
        branch_transition = trajectory.transitions[branch_transition_idx]
        partial_tokens = branch_transition.ac.tokens[:branch_token_idx]

        # Decode partial tokens to text
        partial_text = renderer.tokenizer.decode(partial_tokens)
        context_parts.append(f"[ASSISTANT - PARTIAL]\n{partial_text}")

    return "\n".join(context_parts)


def get_reward_from_trajectory(trajectory: Trajectory) -> float:
    """
    Extract the total reward from a trajectory.

    Args:
        trajectory: The trajectory

    Returns:
        Total reward (sum of all transition rewards)
    """
    return sum(t.reward for t in trajectory.transitions)


def extract_system_and_user_messages_from_env_metadata(
    trajectory: Trajectory,
) -> tuple[list[renderers.Message], str]:
    """
    Extract system messages and the original question from trajectory metadata.

    The SearchEnv stores these in the final transition's metrics.

    Args:
        trajectory: The trajectory

    Returns:
        Tuple of (system_messages, question)
    """
    if not trajectory.transitions:
        return [], ""

    # Get metadata from final transition
    final_transition = trajectory.transitions[-1]
    metrics = final_transition.metrics

    # Extract past_messages which includes system prompt and user question
    past_messages = metrics.get("past_messages", [])
    question = metrics.get("question", "")

    # Filter to get just system messages (before any assistant responses)
    system_messages = []
    for msg in past_messages:
        if msg.get("role") in ["system", "user"]:
            system_messages.append(msg)
        else:
            break  # Stop at first assistant message

    return system_messages, question


def extract_messages_up_to_branch(
    trajectory: Trajectory,
    renderer: renderers.Renderer,
    branch_transition_idx: int,
    branch_token_idx: int,
) -> list[renderers.Message]:
    """
    Extract conversation messages up to (but not including) a branch point.

    This is used to set environment history for tree-based branching.

    Args:
        trajectory: The parent trajectory being branched from
        renderer: Renderer for parsing tokens
        branch_transition_idx: Index of the transition where branching occurs
        branch_token_idx: Token position within that transition

    Returns:
        List of messages representing the conversation up to (not including)
        the branch point. This should be passed to env.set_history().
    """
    # Get initial context (system + user)
    system_messages, question = extract_system_and_user_messages_from_env_metadata(trajectory)

    # Start with system and user messages
    messages = system_messages.copy()

    # Add all complete transitions before the branch point
    for i in range(branch_transition_idx):
        transition = trajectory.transitions[i]

        # Parse assistant message
        assistant_msg, _ = renderer.parse_response(transition.ac.tokens)
        messages.append(assistant_msg)

        # If this was a tool call, there should be a tool response in past_messages
        # For simplicity, we reconstruct from the trajectory structure
        # The tool responses are part of the environment's past_messages
        # but we don't have direct access here. The key insight is:
        # - Each assistant tool call gets a tool response
        # - These are stored in the environment and will be replayed
        # For now, we just include assistant messages

    # NOTE: We do NOT include the partial assistant message at branch_transition_idx
    # That will be replaced by Gemini's alternative

    return messages
