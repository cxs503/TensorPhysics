"""TensorPhysics: solver-neutral multiphysics orchestration."""
from .checkpoint import Checkpoint, CheckpointError, CheckpointStore
from .config import SimulationConfig
from .coupling import CouplingContext, CouplingError, CouplingManager, CouplingPhase
from .diagnostics import DiagnosticRecord, Diagnostics
from .fields import FieldDescriptor, FieldExchangeError, FieldStore, FieldValue
from .protocols import Solver, SolverContext, SolverState
from .provenance import ArtifactRef, ManifestError, RunManifest, RunStatus, ValidationStatus
from .scheduler import Simulation, SimulationError, SimulationState

__all__ = [
    "Checkpoint", "CheckpointError", "CheckpointStore", "CouplingContext",
    "CouplingError", "CouplingManager", "CouplingPhase", "DiagnosticRecord",
    "Diagnostics", "ArtifactRef", "ManifestError", "RunManifest", "RunStatus",
    "ValidationStatus", "FieldDescriptor", "FieldExchangeError", "FieldStore",
    "FieldValue", "Simulation", "SimulationConfig", "SimulationError",
    "SimulationState", "Solver", "SolverContext", "SolverState",
]
__version__ = "0.1.0"
