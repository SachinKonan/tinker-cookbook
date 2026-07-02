from pathlib import Path

from tinker_cookbook.recipes.harbor_rl.harbor_env import (
    DEFAULT_HARBOR_CACHE_DIR,
    get_harbor_cache_dir,
    load_harbor_tasks,
)


def _write_task(task_dir: Path) -> None:
    task_dir.mkdir(parents=True)
    (task_dir / "instruction.md").write_text("Solve the task.\n")
    (task_dir / "task.toml").write_text("[task]\n")


def test_get_harbor_cache_dir_defaults_to_scratch(monkeypatch) -> None:
    monkeypatch.delenv("TINKER_HARBOR_TASKS_DIR", raising=False)
    monkeypatch.delenv("HARBOR_CACHE_DIR", raising=False)

    assert get_harbor_cache_dir() == DEFAULT_HARBOR_CACHE_DIR
    assert str(get_harbor_cache_dir()).startswith("/scratch/gpfs/ZHUANGL/sk7524")


def test_load_harbor_tasks_uses_env_override(tmp_path: Path, monkeypatch) -> None:
    dataset = "frontier-cs-algorithm"
    task_dir = tmp_path / dataset / "frontier-cs-algorithm-302"
    _write_task(task_dir)
    monkeypatch.setenv("TINKER_HARBOR_TASKS_DIR", str(tmp_path))

    tasks = load_harbor_tasks(dataset)

    assert len(tasks) == 1
    assert tasks[0].task_name == "frontier-cs-algorithm-302"
    assert tasks[0].instruction == "Solve the task.\n"
    assert tasks[0].task_dir == task_dir
