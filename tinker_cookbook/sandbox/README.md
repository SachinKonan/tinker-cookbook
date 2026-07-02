# Sandboxing

This directory contains code execution backends for sandboxed evaluation (e.g., grading code in RL environments).

There are currently three available backends: SandboxFusion for local execution, Modal for cloud execution, and Apptainer for cluster jobs.

## Backends

### SandboxFusion (local Docker)

[Sandbox Fusion](https://bytedance.github.io/SandboxFusion/) is a Docker-based code execution sandbox. Start a local sandbox in Docker with:

```bash
docker run -it -p 8080:8080 volcengine/sandbox-fusion:server-20250609
```

For RL workloads, you may want higher concurrency. See [`recipes/code_rl/sandbox_config/local.yaml`](../recipes/code_rl/sandbox_config/local.yaml) for an example configuration that can be mounted with `-v`, and see [`recipes/code_rl/README.md`](../recipes/code_rl/README.md) for instructions on using it.

If you prefer not to use Docker, see the [Sandbox Fusion repository](https://github.com/bytedance/SandboxFusion?tab=readme-ov-file#installation) for manual setup.

Example usage:

```python
from tinker_cookbook.sandbox import SandboxFusionClient

client = SandboxFusionClient()
success, response = await client.run(
    code="print('hello')",
    files={"data.txt": "some content"},
    timeout=30,
)
await client.close()
```

Environment variables:

- `SANDBOX_URL`: Endpoint URL (default: `http://localhost:8080/run_code`)
- `SANDBOX_MAX_CONCURRENCY`: Max concurrent requests (default: 4)

### Modal (cloud)

[Modal Sandboxes](https://modal.com/products/sandboxes) provide cloud-based isolated execution environments. Requires authentication with: `modal token new`

Example usage:

```python
from tinker_cookbook.sandbox.modal_sandbox import ModalSandbox, ModalSandboxPool

# Single sandbox (conforms to SandboxInterface)
sandbox = await ModalSandbox.create()
await sandbox.write_file("/workspace/code.py", "print('hello')")
result = await sandbox.run_command("python /workspace/code.py", workdir="/workspace")
print(result.stdout)
await sandbox.cleanup()

# Pool for concurrent execution (recommended for RL workloads)
pool = ModalSandboxPool(pool_size=32)
result = await pool.run_in_workdir(
    files={"code.py": "print('hello')"},
    command=["python", "code.py"],
)
print(result.stdout)
```

Environment variables:

- `MODAL_POOL_SIZE`: Number of sandboxes in the pool (default: 32)

### Apptainer (cluster)

`LocalApptainerSandbox` provides the single-container primitive for Della-style
cluster runs. It implements `SandboxInterface` using `apptainer exec` and
persistent bind mounts for the paths Harbor uses: `/app`, `/tests`, `/logs`,
`/root`, `/tmp`, and `/workspace`.

Example usage:

```python
from tinker_cookbook.sandbox import LocalApptainerSandbox

sandbox = await LocalApptainerSandbox.create(
    image="/scratch/gpfs/ZHUANGL/sk7524/tinker-sandbox/images/base.sif",
)
await sandbox.write_file("/app/hello.py", "print('hello')")
result = await sandbox.run_command("python /app/hello.py", workdir="/app")
print(result.stdout)
await sandbox.cleanup()
```

For the intended Della sizing, use 4 CPUs and 8 GB memory per live sandbox. On
a 32 CPU / 128 GB node, reserve 4 CPUs for Ray/OS overhead and cap the node at
7 live sandboxes:

```python
from tinker_cookbook.sandbox import recommended_sandboxes_per_node

assert recommended_sandboxes_per_node(
    node_cpus=32,
    node_memory_gb=128,
    sandbox_cpus=4,
    sandbox_memory_gb=8,
    reserve_cpus=4,
) == 7
```

Environment variables:

- `TINKER_APPTAINER_IMAGE`: Default SIF path when `image=` is omitted.
- `TINKER_APPTAINER_WORK_ROOT`: Per-sandbox node-local work root (default: `/tmp/tinker-sandboxes`).
- `TINKER_SCRATCH_ROOT`: Shared scratch root for image caches (default: `/scratch/gpfs/ZHUANGL/sk7524`).
- `APPTAINER_BINARY`: Apptainer executable name/path (default: `apptainer`).

Frontier-CS algorithmic tasks need judge sidecars in addition to this primitive.
Build that as a higher-level Harbor factory that starts an agent sandbox plus
judge sandbox/services, then exposes the agent sandbox through `SandboxInterface`.
