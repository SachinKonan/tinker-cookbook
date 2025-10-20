"""
LangChain wrapper for Tinker models
Integrates Tinker's sampling client with LangChain's LLM interface
"""
from typing import Any, List, Optional, Dict
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
import tinker
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer


class TinkerLLM(LLM):
    """
    LangChain wrapper for Tinker's sampling client

    Usage:
        service_client = tinker.ServiceClient(base_url=base_url)
        sampling_client = service_client.create_sampling_client(model_path=model_path)
        llm = TinkerLLM(
            sampling_client=sampling_client,
            model_name="meta-llama/Llama-3.1-8B",
            temperature=0.0,
            max_tokens=4096
        )
    """

    sampling_client: Any
    """Tinker sampling client"""

    model_name: str
    """Model name for tokenizer and renderer"""

    temperature: float = 0.0
    """Temperature for generation"""

    max_tokens: int = 4096
    """Maximum tokens to generate"""

    top_p: float = 1.0
    """Top-p sampling parameter"""

    top_k: int = -1
    """Top-k sampling parameter"""

    tokenizer: Any = None
    """Tokenizer instance"""

    renderer: Any = None
    """Renderer instance"""

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize tokenizer and renderer
        if self.tokenizer is None:
            self.tokenizer = get_tokenizer(self.model_name)
        if self.renderer is None:
            renderer_name = model_info.get_recommended_renderer_name(self.model_name)
            self.renderer = renderers.get_renderer(renderer_name, self.tokenizer)

    @property
    def _llm_type(self) -> str:
        return "tinker"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call Tinker's sampling client to generate text"""

        # Create a simple user message conversation
        convo = [{"role": "user", "content": prompt}]
        model_input = self.renderer.build_generation_prompt(convo)

        # Set up sampling parameters
        sampling_params = types.SamplingParams(
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            top_p=kwargs.get("top_p", self.top_p),
            top_k=kwargs.get("top_k", self.top_k),
            stop=stop or self.renderer.get_stop_sequences(),
        )

        # Sample from the model
        sample_future = self.sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
        )

        # Wait for result with 10 second timeout
        sample_result = sample_future.result(timeout=10)
        sampled_tokens = sample_result.sequences[0].tokens

        # Parse response using renderer
        parsed_message, _ = self.renderer.parse_response(sampled_tokens)

        return parsed_message["content"]
