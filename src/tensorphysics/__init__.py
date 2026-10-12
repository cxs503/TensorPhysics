"""TensorPhysics: solver-neutral multiphysics orchestration."""
from .artifact_store import ArtifactStore, ArtifactStoreError, LocalArtifactStore
from .s3_artifact_store import S3ArtifactStore
from .checkpoint import Checkpoint, CheckpointError, CheckpointStore
from .config import SimulationConfig
from .coupling import CouplingContext, CouplingError, CouplingManager, CouplingPhase
from .diagnostics import DiagnosticRecord, Diagnostics
from .fields import FieldDescriptor, FieldExchangeError, FieldStore, FieldValue
from .protocols import Solver, SolverContext, SolverState
from .platform_registry import PlatformRegistryError, TensorPlatformRunRegistry
from .provenance import ArtifactRef, ManifestError, RunManifest, RunStatus, ValidationStatus
from .research_contracts import (
    ExperimentPlan, ResearchContractError, ResearchGoal, ResearchMemoryRecord, ValidationReport,
)
from .research_runtime import (
    AgentBackend, AgentRequest, AgentResult, InvalidTaskTransition, ResearchRuntimeError,
    ResearchTask, ResearchTaskStore, TaskEvent, TaskNotFoundError, TaskStatus,
)
from .research_runner import ResearchTaskRunner, ResearchValidator
from .agent_subprocess import SubprocessAgentBackend
from .scheduler import Simulation, SimulationError, SimulationState

__all__ = [
    "ArtifactStore", "ArtifactStoreError", "LocalArtifactStore", "S3ArtifactStore",
    "Checkpoint", "CheckpointError", "CheckpointStore", "CouplingContext",
    "CouplingError", "CouplingManager", "CouplingPhase", "DiagnosticRecord",
    "Diagnostics", "ArtifactRef", "ManifestError", "RunManifest", "RunStatus",
    "ValidationStatus", "ResearchGoal", "ExperimentPlan", "ValidationReport",
    "ResearchMemoryRecord", "ResearchContractError", "FieldDescriptor", "FieldExchangeError",
    "FieldStore", "FieldValue", "Simulation", "SimulationConfig", "SimulationError",
    "SimulationState", "Solver", "SolverContext", "SolverState",
    "PlatformRegistryError", "TensorPlatformRunRegistry",
    "AgentBackend", "AgentRequest", "AgentResult", "InvalidTaskTransition",
    "ResearchRuntimeError", "ResearchTask", "ResearchTaskStore", "TaskEvent",
    "TaskNotFoundError", "TaskStatus", "ResearchTaskRunner", "ResearchValidator",
    "SubprocessAgentBackend",
]
__version__ = "0.1.0"
