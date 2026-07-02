"""Apptainer-backed sandbox implementation.

This backend is intentionally a small, single-container building block. It is
useful for local Della development and as the per-container primitive for a
future Ray/Apptainer pool. Frontier-CS algorithmic tasks also need sidecar
judge support; that should be layered above this class rather than hidden here.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from tinker_cookbook.sandbox.sandbox_interface import SandboxResult, SandboxTerminatedError

DEFAULT_SCRATCH_ROOT = Path("/scratch/gpfs/ZHUANGL/sk7524")
DEFAULT_APPTAINER_WORK_ROOT = Path("/tmp/tinker-sandboxes")
DEFAULT_APPTAINER_IMAGE_ENV_VAR = "TINKER_APPTAINER_IMAGE"
DEFAULT_STREAM_OUTPUT_BYTES = 128 * 1024


def _scratch_root() -> Path:
    return Path(os.getenv("TINKER_SCRATCH_ROOT", str(DEFAULT_SCRATCH_ROOT))).expanduser()


def default_apptainer_image_cache_dir() -> Path:
    return _scratch_root() / "tinker-sandbox" / "images"


def default_apptainer_work_root() -> Path:
    return Path(os.getenv("TINKER_APPTAINER_WORK_ROOT", str(DEFAULT_APPTAINER_WORK_ROOT)))


@dataclass(frozen=True)
class ApptainerSandboxResources:
    """Resource sizing for one live sandbox actor/process."""

    cpus: int = 4
    memory_gb: int = 8


def recommended_sandboxes_per_node(
    *,
    node_cpus: int = 32,
    node_memory_gb: int = 128,
    sandbox_cpus: int = 4,
    sandbox_memory_gb: int = 8,
    reserve_cpus: int = 4,
    reserve_memory_gb: int = 0,
) -> int:
    """Compute a conservative per-node sandbox cap.

    With the intended Della sizing of 32 CPUs, 128 GB RAM, 4 CPUs per sandbox,
    and 4 reserved CPUs, this returns 7.
    """
    cpu_slots = max(0, node_cpus - reserve_cpus) // sandbox_cpus
    memory_slots = max(0, node_memory_gb - reserve_memory_gb) // sandbox_memory_gb
    return max(0, min(cpu_slots, memory_slots))


@dataclass(frozen=True)
class ApptainerMount:
    """A persistent host directory mounted at an absolute path in the sandbox."""

    target: str
    source_name: str


DEFAULT_MOUNTS = (
    ApptainerMount("/app", "app"),
    ApptainerMount("/tests", "tests"),
    ApptainerMount("/logs", "logs"),
    ApptainerMount("/root", "root"),
    ApptainerMount("/tmp", "tmp"),
    ApptainerMount("/workspace", "workspace"),
)


def _decode_capped(data: bytes, max_bytes: int) -> str:
    truncated = len(data) > max_bytes
    data = data[:max_bytes]
    text = data.decode("utf-8", errors="replace")
    if truncated:
        text += f"\n...[truncated to {max_bytes} bytes]"
    return text


class LocalApptainerSandbox:
    """Persistent Apptainer sandbox matching ``SandboxInterface``.

    The SIF root filesystem is treated as immutable. Persistence is provided by
    bind-mounted host directories for the paths Harbor uses: ``/app``, ``/tests``,
    ``/logs``, ``/root``, ``/tmp``, and ``/workspace``.
    """

    def __init__(
        self,
        *,
        image: str | Path,
        sandbox_root: str | Path,
        sandbox_id: str,
        resources: ApptainerSandboxResources | None = None,
        apptainer_binary: str | None = None,
        max_stream_output_bytes: int = DEFAULT_STREAM_OUTPUT_BYTES,
        mounts: tuple[ApptainerMount, ...] = DEFAULT_MOUNTS,
        extra_env: Mapping[str, str] | None = None,
        writable_tmpfs: bool = True,
    ) -> None:
        self._image = Path(image).expanduser()
        self._sandbox_root = Path(sandbox_root).expanduser()
        self._sandbox_id = sandbox_id
        self._resources = resources or ApptainerSandboxResources()
        self._apptainer_binary = apptainer_binary or os.getenv("APPTAINER_BINARY", "apptainer")
        self._max_stream_output_bytes = max_stream_output_bytes
        self._mounts = mounts
        self._extra_env = dict(extra_env or {})
        self._writable_tmpfs = writable_tmpfs
        self._cleaned = False

        self._sandbox_root.mkdir(parents=True, exist_ok=True)
        for mount in self._mounts:
            (self._sandbox_root / mount.source_name).mkdir(parents=True, exist_ok=True)

    @classmethod
    async def create(
        cls,
        *,
        image: str | Path | None = None,
        timeout: int = 600,
        work_root: str | Path | None = None,
        resources: ApptainerSandboxResources | None = None,
        apptainer_binary: str | None = None,
        max_stream_output_bytes: int = DEFAULT_STREAM_OUTPUT_BYTES,
        extra_env: Mapping[str, str] | None = None,
    ) -> LocalApptainerSandbox:
        del timeout  # Lifetime is enforced by the caller/Slurm job for now.
        resolved_image = image or os.getenv(DEFAULT_APPTAINER_IMAGE_ENV_VAR)
        if resolved_image is None:
            raise ValueError(
                "Apptainer image is required. Pass image=... or set "
                f"{DEFAULT_APPTAINER_IMAGE_ENV_VAR}."
            )
        sandbox_id = f"apptainer-{uuid.uuid4().hex[:12]}"
        root = Path(work_root or default_apptainer_work_root()) / sandbox_id
        return cls(
            image=resolved_image,
            sandbox_root=root,
            sandbox_id=sandbox_id,
            resources=resources,
            apptainer_binary=apptainer_binary,
            max_stream_output_bytes=max_stream_output_bytes,
            extra_env=extra_env,
        )

    @property
    def sandbox_id(self) -> str:
        return self._sandbox_id

    @property
    def resources(self) -> ApptainerSandboxResources:
        return self._resources

    def _host_path(self, path: str) -> Path:
        posix_path = PurePosixPath(path)
        if not posix_path.is_absolute():
            posix_path = PurePosixPath("/workspace") / posix_path

        for mount in sorted(self._mounts, key=lambda m: len(m.target), reverse=True):
            target = PurePosixPath(mount.target)
            if posix_path == target:
                return self._sandbox_root / mount.source_name
            if str(posix_path).startswith(str(target) + "/"):
                return self._sandbox_root / mount.source_name / posix_path.relative_to(target)

        raise ValueError(
            f"{path!r} is not under a persistent Apptainer mount. "
            "Use /app, /tests, /logs, /root, /tmp, or /workspace."
        )

    def _apptainer_command(self, command: str, workdir: str | None) -> list[str]:
        args = [
            self._apptainer_binary,
            "exec",
            "--cleanenv",
            "--containall",
            "--no-home",
        ]
        if self._writable_tmpfs:
            args.append("--writable-tmpfs")

        env = {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            **self._extra_env,
        }
        for key, value in env.items():
            args.extend(["--env", f"{key}={value}"])

        for mount in self._mounts:
            source = self._sandbox_root / mount.source_name
            args.extend(["--bind", f"{source}:{mount.target}"])

        args.extend(["--pwd", workdir or "/workspace", str(self._image), "bash", "-lc", command])
        return args

    async def send_heartbeat(self, timeout: int = 30) -> None:
        result = await self.run_command("true", timeout=timeout)
        if result.exit_code != 0:
            raise SandboxTerminatedError(result.stderr or "Apptainer heartbeat failed")

    async def run_command(
        self,
        command: str,
        workdir: str | None = None,
        timeout: int = 60,
        max_output_bytes: int | None = None,
    ) -> SandboxResult:
        if self._cleaned:
            raise SandboxTerminatedError(f"Sandbox {self.sandbox_id} has been cleaned up.")
        if shutil.which(self._apptainer_binary) is None:
            return SandboxResult(
                stdout="",
                stderr=f"Apptainer binary not found: {self._apptainer_binary}",
                exit_code=-1,
            )
        if not self._image.exists():
            return SandboxResult(
                stdout="",
                stderr=f"Apptainer image not found: {self._image}",
                exit_code=-1,
            )

        cap = max_output_bytes if max_output_bytes is not None else self._max_stream_output_bytes
        args = self._apptainer_command(command, workdir)
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=os.setsid,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except TimeoutError:
                os.killpg(proc.pid, signal.SIGKILL)
                stdout_b, stderr_b = await proc.communicate()
                stderr_b += f"\nCommand timed out after {timeout}s".encode()
                return SandboxResult(
                    stdout=_decode_capped(stdout_b, cap),
                    stderr=_decode_capped(stderr_b, cap),
                    exit_code=-1,
                    metrics={"timeout": timeout},
                )
            return SandboxResult(
                stdout=_decode_capped(stdout_b, cap),
                stderr=_decode_capped(stderr_b, cap),
                exit_code=proc.returncode if proc.returncode is not None else -1,
            )
        except Exception as e:
            return SandboxResult(stdout="", stderr=str(e), exit_code=-1)

    async def read_file(
        self, path: str, max_bytes: int | None = None, timeout: int = 60
    ) -> SandboxResult:
        del timeout
        try:
            host_path = self._host_path(path)
            data = host_path.read_bytes()
            cap = max_bytes if max_bytes is not None else len(data)
            return SandboxResult(stdout=_decode_capped(data, cap), stderr="", exit_code=0)
        except Exception as e:
            return SandboxResult(stdout="", stderr=str(e), exit_code=1)

    async def write_file(
        self,
        path: str,
        content: str | bytes = "",
        executable: bool = False,
        timeout: int = 60,
    ) -> SandboxResult:
        del timeout
        try:
            host_path = self._host_path(path)
            host_path.parent.mkdir(parents=True, exist_ok=True)
            data = content.encode() if isinstance(content, str) else content
            host_path.write_bytes(data)
            if executable:
                host_path.chmod(host_path.stat().st_mode | 0o111)
            return SandboxResult(stdout="", stderr="", exit_code=0)
        except Exception as e:
            return SandboxResult(stdout="", stderr=str(e), exit_code=1)

    async def cleanup(self) -> None:
        self._cleaned = True
        shutil.rmtree(self._sandbox_root, ignore_errors=True)


@dataclass(frozen=True)
class LocalApptainerSandboxFactory:
    """Pickleable Harbor ``SandboxFactory`` for local Apptainer sandboxes."""

    image: str | Path | None = None
    work_root: str | Path = field(default_factory=default_apptainer_work_root)
    resources: ApptainerSandboxResources = field(default_factory=ApptainerSandboxResources)
    apptainer_binary: str | None = None
    max_stream_output_bytes: int = DEFAULT_STREAM_OUTPUT_BYTES

    async def __call__(self, env_dir: Path, timeout: int) -> LocalApptainerSandbox:
        del env_dir
        return await LocalApptainerSandbox.create(
            image=self.image,
            timeout=timeout,
            work_root=self.work_root,
            resources=self.resources,
            apptainer_binary=self.apptainer_binary,
            max_stream_output_bytes=self.max_stream_output_bytes,
        )


__all__ = [
    "ApptainerMount",
    "ApptainerSandboxResources",
    "LocalApptainerSandbox",
    "LocalApptainerSandboxFactory",
    "default_apptainer_image_cache_dir",
    "recommended_sandboxes_per_node",
]
