"""
ReAct agent for GAIA benchmark using Tinker models
"""
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from .tools import TOOLS
from .config import Config
from .tinker_llm import TinkerLLM
import tinker


def create_gaia_agent(
    sampling_client,
    model_name: str = Config.DEFAULT_MODEL,
    temperature: float = Config.DEFAULT_TEMPERATURE,
    max_tokens: int = Config.DEFAULT_MAX_TOKENS,
    **kwargs
):
    """
    Create a ReAct agent configured for GAIA benchmark using Tinker model

    Args:
        sampling_client: Tinker sampling client
        model_name: Name of the model for tokenizer/renderer
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        **kwargs: Additional sampling parameters

    Returns:
        AgentExecutor configured for GAIA tasks
    """

    # Wrap Tinker model for LangChain compatibility
    llm = TinkerLLM(
        sampling_client=sampling_client,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )

    # Use the ReAct prompt template
    prompt = hub.pull("hwchase17/react")
    print(prompt)

    # Create agent
    agent = create_react_agent(llm, TOOLS, prompt)

    # Create executor
    executor = AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=Config.VERBOSE,
        max_iterations=Config.MAX_ITERATIONS,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )

    return executor
