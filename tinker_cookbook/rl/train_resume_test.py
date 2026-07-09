import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from tinker_cookbook.rl import train


class _FakeLogger:
    store = None

    def get_logger_url(self):
        return None

    def log_hparams(self, config):
        pass

    def log_metrics(self, metrics, step=None):
        pass

    def close(self):
        pass

    def sync(self):
        pass


class _FakeCheckpointManager:
    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.save_final_async = AsyncMock()
        self.finalize_async = AsyncMock()
        type(self).instances.append(self)


def _write_checkpoint(log_path: Path, *, batch: int, state_path: str) -> None:
    log_path.mkdir(parents=True, exist_ok=True)
    with (log_path / "checkpoints.jsonl").open("w") as f:
        f.write(
            json.dumps(
                {
                    "name": f"{batch:06d}",
                    "batch": batch,
                    "state_path": state_path,
                    "rolling": True,
                }
            )
            + "\n"
        )


def _make_training_client():
    training_client = MagicMock()
    training_client.get_tokenizer.return_value = MagicMock(name="tokenizer")
    return training_client


def _make_config(tmp_path: Path, *, max_steps: int = 5):
    async def dataset_builder():
        return [object() for _ in range(10)], None

    return train.Config(
        learning_rate=2e-5,
        dataset_builder=dataset_builder,
        model_name="Qwen/Qwen3.5-9B",
        recipe_name="test_math_rl",
        max_tokens=1024,
        log_path=str(tmp_path),
        renderer_name="qwen3_5",
        max_steps=max_steps,
        save_every=20,
        rolling_save_every=1,
        blocking_rolling_checkpoints=True,
    )


@pytest.fixture
def patched_main_deps(monkeypatch):
    _FakeCheckpointManager.instances = []

    training_func = AsyncMock()
    service_client = MagicMock()
    service_client.create_training_client_from_state_with_optimizer_async = AsyncMock(
        return_value=_make_training_client()
    )
    service_client.create_training_client_from_state_async = AsyncMock(
        return_value=_make_training_client()
    )
    service_client.create_lora_training_client_async = AsyncMock(
        return_value=_make_training_client()
    )

    monkeypatch.setattr(train.ml_log, "setup_logging", MagicMock(return_value=_FakeLogger()))
    monkeypatch.setattr(train.tinker, "ServiceClient", MagicMock(return_value=service_client))
    monkeypatch.setattr(train.checkpoint_utils, "CheckpointManager", _FakeCheckpointManager)
    monkeypatch.setattr(
        train.checkpoint_utils,
        "check_renderer_name_for_checkpoint_async",
        AsyncMock(),
    )
    monkeypatch.setattr(train.model_info, "warn_if_renderer_not_recommended", MagicMock())
    monkeypatch.setattr(train, "do_sync_training", training_func)

    return service_client, training_func


@pytest.mark.asyncio
async def test_main_resumes_optimizer_state_and_starts_at_checkpoint_batch(
    tmp_path, patched_main_deps
):
    service_client, training_func = patched_main_deps
    _write_checkpoint(tmp_path, batch=3, state_path="tinker://model/weights/000003")

    await train.main(_make_config(tmp_path, max_steps=5))

    service_client.create_training_client_from_state_with_optimizer_async.assert_awaited_once()
    restore_call = service_client.create_training_client_from_state_with_optimizer_async.call_args
    assert restore_call.args == ("tinker://model/weights/000003",)
    service_client.create_training_client_from_state_async.assert_not_awaited()
    service_client.create_lora_training_client_async.assert_not_awaited()

    training_func.assert_awaited_once()
    training_kwargs = training_func.call_args.kwargs
    assert training_kwargs["start_batch"] == 3
    assert training_kwargs["end_batch"] == 5

    checkpoint_mgr = _FakeCheckpointManager.instances[-1]
    checkpoint_mgr.save_final_async.assert_awaited_once_with(loop_state={"batch": 5})
    checkpoint_mgr.finalize_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_main_without_checkpoint_replays_from_batch_zero_with_fresh_optimizer(
    tmp_path, patched_main_deps
):
    service_client, training_func = patched_main_deps

    await train.main(_make_config(tmp_path, max_steps=2))

    service_client.create_lora_training_client_async.assert_awaited_once_with(
        "Qwen/Qwen3.5-9B", rank=32, user_metadata={"renderer_name": "qwen3_5"}
    )
    service_client.create_training_client_from_state_with_optimizer_async.assert_not_awaited()
    service_client.create_training_client_from_state_async.assert_not_awaited()

    training_func.assert_awaited_once()
    training_kwargs = training_func.call_args.kwargs
    assert training_kwargs["start_batch"] == 0
    assert training_kwargs["end_batch"] == 2
