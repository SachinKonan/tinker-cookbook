"""
Implementations that correspond to a model or policy that can be sampled from, but with different amounts of additional structure.

The TokenCompleter operates on tokens. This is the version used by RL algorithms, because RL algorithms work on Tokens. The MessageCompleter operates on messages, so it needs to be used with a renderer.

Evals and other code should use the appropriate interface.
"""

import asyncio
from dataclasses import dataclass, field
from typing import TypeAlias

import tinker

from tinker_cookbook import renderers

# Interfaces

StopCondition: TypeAlias = list[str] | list[int]


@dataclass
class TokensWithLogprobs:
    """A sequence of token IDs with optional log-probabilities and a stop reason."""

    tokens: list[int]
    maybe_logprobs: list[float] | None
    stop_reason: tinker.StopReason = "stop"

    @property
    def logprobs(self) -> list[float]:
        if self.maybe_logprobs is None:
            raise ValueError("Logprobs are not available")
        return self.maybe_logprobs


class TokenCompleter:
    """Abstract interface for generating tokens from a prompt."""

    async def __call__(
        self, model_input: tinker.ModelInput, stop: StopCondition
    ) -> TokensWithLogprobs:
        """Generate a token sequence from the given model input.

        Args:
            model_input (tinker.ModelInput): The tokenized prompt to complete from.
            stop (StopCondition): Stop sequences (strings) or stop token IDs
                that terminate generation.

        Returns:
            TokensWithLogprobs: The generated tokens with their log-probabilities
                and stop reason.
        """
        raise NotImplementedError


class MessageCompleter:
    """Abstract interface for generating message responses."""

    # TODO maybe add n_samples to the interfaces?
    async def __call__(self, messages: list[renderers.Message]) -> renderers.Message:
        """Generate an assistant message given a conversation history.

        Args:
            messages (list[renderers.Message]): The conversation history as a
                list of message dicts with ``role`` and ``content`` keys.

        Returns:
            renderers.Message: The generated assistant message, which may include
                ``tool_calls`` if the model produced them.
        """
        raise NotImplementedError


# Implementations


@dataclass
class TinkerTokenCompleter(TokenCompleter):
    """Token completer that uses a tinker.SamplingClient to sample actions.

    Args:
        sampling_client (tinker.SamplingClient): Client used to sample from
            the model.
        max_tokens (int): Maximum number of tokens to generate per call.
        temperature (float): Sampling temperature. Default: 1.0.
        context_window (int | None): Model's total context window size. When
            set, ``max_tokens`` is dynamically capped per request so that
            ``prompt_tokens + max_tokens <= context_window``. This prevents
            "prompt + max_tokens exceeds context window" API errors.

    Example::

        completer = TinkerTokenCompleter(sampling_client, max_tokens=512)
        result = await completer(model_input, stop=["<|endoftext|>"])
        print(result.tokens, result.logprobs)
    """

    sampling_client: tinker.SamplingClient
    max_tokens: int
    temperature: float = 1.0
    context_window: int | None = None

    async def __call__(
        self, model_input: tinker.ModelInput, stop: StopCondition
    ) -> TokensWithLogprobs:
        """Sample an action from the policy given an observation."""
        max_tokens = self.max_tokens
        if self.context_window is not None:
            max_tokens = min(max_tokens, self.context_window - model_input.length)
            if max_tokens <= 0:
                raise ValueError(
                    f"Prompt length ({model_input.length}) exceeds context window "
                    f"({self.context_window}). No room for generation."
                )

        # Sample from the model
        sample_result = await self.sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                stop=stop,
                max_tokens=max_tokens,
                temperature=self.temperature,
            ),
        )

        # Extract tokens, logprobs, and stop_reason from the first (and only) sample
        sampled_seq = sample_result.sequences[0]
        assert sampled_seq.logprobs is not None

        return TokensWithLogprobs(
            tokens=sampled_seq.tokens,
            maybe_logprobs=sampled_seq.logprobs,
            stop_reason=sampled_seq.stop_reason,
        )


@dataclass
class _PendingSampleBatch:
    """Concurrent same-prompt policy calls awaiting one grouped sample request."""

    model_input: tinker.ModelInput
    stop: StopCondition
    max_tokens: int
    futures: list[asyncio.Future] = field(default_factory=list)
    flush_task: asyncio.Task | None = None


@dataclass
class GroupCoalescingTokenCompleter(TokenCompleter):
    """TokenCompleter that coalesces concurrent identical prompts into one request.

    RL group rollouts issue ``group_size`` concurrent policy calls that share the
    same initial observation. Each call normally becomes its own
    ``sample_async(num_samples=1)`` request. This completer instead holds calls
    for a short window and sends one ``sample_async(num_samples=k)`` per distinct
    (prompt, stop, max_tokens) key, letting the backend serve the whole group
    from a single grouped completion (e.g. vLLM's ``n`` parameter behind a
    shared prefill). Distinct prompts (e.g. multi-turn continuations) never
    coalesce, so behavior is unchanged beyond the small window latency.

    Args:
        sampling_client (tinker.SamplingClient): Client used to sample from
            the model.
        max_tokens (int): Maximum number of tokens to generate per call.
        temperature (float): Sampling temperature. Default: 1.0.
        context_window (int | None): Model's total context window size; see
            :class:`TinkerTokenCompleter`.
        coalesce_window_sec (float): How long the first call for a key waits
            for peers before flushing. Default: 0.02.
    """

    sampling_client: tinker.SamplingClient
    max_tokens: int
    temperature: float = 1.0
    context_window: int | None = None
    coalesce_window_sec: float = 0.02
    _pending: dict = field(default_factory=dict)

    async def __call__(
        self, model_input: tinker.ModelInput, stop: StopCondition
    ) -> TokensWithLogprobs:
        max_tokens = self.max_tokens
        if self.context_window is not None:
            max_tokens = min(max_tokens, self.context_window - model_input.length)
            if max_tokens <= 0:
                raise ValueError(
                    f"Prompt length ({model_input.length}) exceeds context window "
                    f"({self.context_window}). No room for generation."
                )

        try:
            key = (tuple(model_input.to_ints()), tuple(stop), max_tokens)
        except Exception:
            # Non-token content (e.g. image chunks): sample directly.
            return await self._sample(model_input, stop, max_tokens, 1, [None])  # type: ignore[arg-type]

        batch = self._pending.get(key)
        if batch is None:
            batch = _PendingSampleBatch(model_input=model_input, stop=stop, max_tokens=max_tokens)
            self._pending[key] = batch
            # Keep a reference so the flush task is not garbage collected.
            batch.flush_task = asyncio.create_task(self._flush_after_window(key))
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        batch.futures.append(future)
        return await future

    async def _flush_after_window(self, key) -> None:
        await asyncio.sleep(self.coalesce_window_sec)
        batch = self._pending.pop(key)
        try:
            await self._sample(
                batch.model_input, batch.stop, batch.max_tokens, len(batch.futures), batch.futures
            )
        except Exception as e:
            for future in batch.futures:
                if not future.done():
                    future.set_exception(e)

    async def _sample(
        self,
        model_input: tinker.ModelInput,
        stop: StopCondition,
        max_tokens: int,
        num_samples: int,
        futures: list,
    ) -> TokensWithLogprobs:
        sample_result = await self.sampling_client.sample_async(
            prompt=model_input,
            num_samples=num_samples,
            sampling_params=tinker.SamplingParams(
                stop=stop,
                max_tokens=max_tokens,
                temperature=self.temperature,
            ),
        )
        outputs = []
        for sampled_seq in sample_result.sequences:
            assert sampled_seq.logprobs is not None
            outputs.append(
                TokensWithLogprobs(
                    tokens=sampled_seq.tokens,
                    maybe_logprobs=sampled_seq.logprobs,
                    stop_reason=sampled_seq.stop_reason,
                )
            )
        if len(outputs) < len(futures):
            error = RuntimeError(
                f"Grouped sample returned {len(outputs)} sequences for {len(futures)} requests"
            )
            for future in futures:
                if future is not None and not future.done():
                    future.set_exception(error)
            raise error
        for future, output in zip(futures, outputs):
            if future is not None and not future.done():
                future.set_result(output)
        return outputs[0]


class TinkerMessageCompleter(MessageCompleter):
    """Message completer that uses a tinker.SamplingClient to generate responses.

    Args:
        sampling_client (tinker.SamplingClient): Client used to sample from
            the model.
        renderer (renderers.Renderer): Renderer that converts between messages
            and token sequences.
        max_tokens (int): Maximum number of tokens to generate per call.
        stop_condition (StopCondition | None): Custom stop condition. If ``None``,
            uses the renderer's default stop sequences.
        temperature (float): Sampling temperature. Default: 1.0.

    Example::

        completer = TinkerMessageCompleter(sampling_client, renderer, max_tokens=512)
        response = await completer([
            {"role": "user", "content": "What is 2+2?"}
        ])
        print(response["content"])
    """

    def __init__(
        self,
        sampling_client: tinker.SamplingClient,
        renderer: renderers.Renderer,
        max_tokens: int,
        stop_condition: StopCondition | None = None,
        temperature: float = 1.0,
    ):
        self.sampling_client = sampling_client
        self.renderer = renderer
        self.max_tokens = max_tokens
        self.temperature = temperature
        if stop_condition is None:
            self.stop_condition = self.renderer.get_stop_sequences()
        else:
            self.stop_condition = stop_condition

    async def __call__(self, messages: list[renderers.Message]) -> renderers.Message:
        # Render the conversation for the model
        model_input = self.renderer.build_generation_prompt(messages)

        # Sample from the model
        response = await self.sampling_client.sample_async(
            model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=self.stop_condition,
            ),
        )

        # Decode the response
        parsed_message, _termination = self.renderer.parse_response(response.sequences[0].tokens)

        result: renderers.Message = {"role": "assistant", "content": parsed_message["content"]}
        if "tool_calls" in parsed_message:
            result["tool_calls"] = parsed_message["tool_calls"]
        return result
