"""
Conversation extraction and saving utilities
"""
import json
from typing import List, Dict, Any


def extract_openai_messages(agent_executor, question: str) -> List[Dict[str, str]]:
    """
    Extract conversation in OpenAI message format from agent execution

    Args:
        agent_executor: LangChain AgentExecutor
        question: The question to ask

    Returns:
        List of messages in OpenAI format: [{"role": "user/assistant", "content": "..."}]
    """
    # Run the agent
    result = agent_executor.invoke({"input": question})

    messages = []

    # Add initial user question
    messages.append({
        "role": "user",
        "content": question
    })

    # Extract intermediate steps (thoughts and actions)
    if "intermediate_steps" in result:
        for action, observation in result["intermediate_steps"]:
            # Agent's thought and action
            thought_action = f"Thought: {action.log}\nAction: {action.tool}\nAction Input: {action.tool_input}"
            messages.append({
                "role": "assistant",
                "content": thought_action
            })

            # Tool observation (treated as system/user feedback)
            messages.append({
                "role": "user",
                "content": f"Observation: {observation}"
            })

    # Add final answer
    final_answer = result.get("output", "")
    messages.append({
        "role": "assistant",
        "content": f"Final Answer: {final_answer}"
    })

    return messages


def save_conversation(messages: List[Dict[str, str]], filepath: str):
    """
    Save conversation to JSON file

    Args:
        messages: List of OpenAI format messages
        filepath: Path to save the JSON file
    """
    with open(filepath, 'w') as f:
        json.dump(messages, f, indent=2)
