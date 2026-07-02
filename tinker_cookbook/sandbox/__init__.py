"""
Code execution backends for sandboxed code evaluation.

The sandbox/ directory provides thin wrappers around different sandbox backends:
- SandboxFusionClient: HTTP-based sandbox using SandboxFusion Docker container
- ModalSandbox: Cloud sandbox using Modal's infrastructure
- LocalApptainerSandbox: Apptainer container execution for cluster jobs
"""

from enum import StrEnum

from tinker_cookbook.sandbox.apptainer_sandbox import (
    ApptainerSandboxResources,
    LocalApptainerSandbox,
    LocalApptainerSandboxFactory,
    recommended_sandboxes_per_node,
)
from tinker_cookbook.sandbox.sandbox_interface import (
    SandboxInterface,
    SandboxResult,
    SandboxTerminatedError,
)
from tinker_cookbook.sandbox.sandboxfusion import SandboxFusionClient


class SandboxBackend(StrEnum):
    SANDBOXFUSION = "sandboxfusion"
    MODAL = "modal"
    APPTAINER = "apptainer"


__all__ = [
    "ApptainerSandboxResources",
    "LocalApptainerSandbox",
    "LocalApptainerSandboxFactory",
    "SandboxBackend",
    "SandboxFusionClient",
    "SandboxInterface",
    "SandboxResult",
    "SandboxTerminatedError",
    "recommended_sandboxes_per_node",
]
