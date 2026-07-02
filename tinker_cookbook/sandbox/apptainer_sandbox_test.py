import pickle
from pathlib import Path

import pytest

from tinker_cookbook.sandbox.apptainer_sandbox import (
    ApptainerSandboxResources,
    LocalApptainerSandbox,
    LocalApptainerSandboxFactory,
    recommended_sandboxes_per_node,
)


def test_recommended_sandboxes_per_node_della_defaults() -> None:
    assert recommended_sandboxes_per_node() == 7


def test_local_apptainer_factory_is_pickleable(tmp_path: Path) -> None:
    factory = LocalApptainerSandboxFactory(
        image="/scratch/gpfs/ZHUANGL/sk7524/tinker-sandbox/images/base.sif",
        work_root=tmp_path,
        resources=ApptainerSandboxResources(cpus=4, memory_gb=8),
    )

    restored = pickle.loads(pickle.dumps(factory))

    assert restored.image == factory.image
    assert restored.work_root == factory.work_root
    assert restored.resources == factory.resources


@pytest.mark.asyncio
async def test_local_apptainer_read_write_persistent_mount(tmp_path: Path) -> None:
    sandbox = LocalApptainerSandbox(
        image=tmp_path / "missing.sif",
        sandbox_root=tmp_path / "sandbox",
        sandbox_id="test-sandbox",
    )

    result = await sandbox.write_file("/tests/test.sh", "#!/bin/bash\necho ok\n", executable=True)
    assert result.exit_code == 0

    host_file = tmp_path / "sandbox" / "tests" / "test.sh"
    assert host_file.exists()
    assert host_file.stat().st_mode & 0o111

    read_result = await sandbox.read_file("/tests/test.sh")
    assert read_result.exit_code == 0
    assert read_result.stdout == "#!/bin/bash\necho ok\n"

    await sandbox.cleanup()
    assert not (tmp_path / "sandbox").exists()


@pytest.mark.asyncio
async def test_local_apptainer_rejects_unmounted_write(tmp_path: Path) -> None:
    sandbox = LocalApptainerSandbox(
        image=tmp_path / "missing.sif",
        sandbox_root=tmp_path / "sandbox",
        sandbox_id="test-sandbox",
    )

    result = await sandbox.write_file("/etc/passwd", "not visible")

    assert result.exit_code == 1
    assert "not under a persistent Apptainer mount" in result.stderr
