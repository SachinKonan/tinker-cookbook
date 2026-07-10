"""Tests for GroupCoalescingTokenCompleter."""

import asyncio
from dataclasses import dataclass, field

import tinker

from tinker_cookbook.completers import GroupCoalescingTokenCompleter


@dataclass
class _FakeSequence:
    tokens: list[int]
    logprobs: list[float]
    stop_reason: str = "stop"


@dataclass
class _FakeSampleResult:
    sequences: list[_FakeSequence]


@dataclass
class _FakeSamplingClient:
    calls: list[dict] = field(default_factory=list)

    async def sample_async(self, prompt, num_samples, sampling_params):
        self.calls.append({"prompt": prompt.to_ints(), "num_samples": num_samples})
        # Distinct token per sequence index so callers can be told apart.
        return _FakeSampleResult(
            sequences=[_FakeSequence(tokens=[100 + i], logprobs=[-0.1]) for i in range(num_samples)]
        )


def test_identical_prompts_coalesce_into_one_grouped_request():
    client = _FakeSamplingClient()
    completer = GroupCoalescingTokenCompleter(client, max_tokens=16, coalesce_window_sec=0.01)
    prompt = tinker.ModelInput.from_ints([1, 2, 3])

    async def run():
        return await asyncio.gather(*[completer(prompt, [0]) for _ in range(16)])

    outputs = asyncio.run(run())

    assert len(client.calls) == 1
    assert client.calls[0]["num_samples"] == 16
    # Each caller received its own sequence, in order.
    assert [o.tokens for o in outputs] == [[100 + i] for i in range(16)]
    assert all(o.logprobs == [-0.1] for o in outputs)


def test_distinct_prompts_do_not_coalesce():
    client = _FakeSamplingClient()
    completer = GroupCoalescingTokenCompleter(client, max_tokens=16, coalesce_window_sec=0.01)

    async def run():
        return await asyncio.gather(
            completer(tinker.ModelInput.from_ints([1]), [0]),
            completer(tinker.ModelInput.from_ints([2]), [0]),
        )

    outputs = asyncio.run(run())

    assert len(client.calls) == 2
    assert sorted(call["num_samples"] for call in client.calls) == [1, 1]
    assert all(o.tokens == [100] for o in outputs)


def test_sampling_error_propagates_to_all_waiters():
    class _FailingClient:
        async def sample_async(self, prompt, num_samples, sampling_params):
            raise RuntimeError("boom")

    completer = GroupCoalescingTokenCompleter(
        _FailingClient(), max_tokens=16, coalesce_window_sec=0.01
    )
    prompt = tinker.ModelInput.from_ints([1, 2, 3])

    async def run():
        return await asyncio.gather(
            *[completer(prompt, [0]) for _ in range(4)], return_exceptions=True
        )

    outputs = asyncio.run(run())
    assert len(outputs) == 4
    assert all(isinstance(o, RuntimeError) for o in outputs)
