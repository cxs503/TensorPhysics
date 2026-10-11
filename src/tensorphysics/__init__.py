"""TensorPhysics: solver-neutral multiphysics orchestration."""
from .checkpoint import Checkpoint, CheckpointError, CheckpointStore
from .config import SimulationConfig
from .diagnostics import DiagnosticRecord, Diagnostics
from .fields import FieldDescriptor, FieldExchangeError, FieldStore, FieldValue
from .protocols import Solver, SolverContext, SolverState
from .scheduler import Simulation, SimulationError, SimulationState

__all__ = [
    "Checkpoint", "CheckpointError", "CheckpointStore", "DiagnosticRecord",
    "Diagnostics", "FieldDescriptor", "FieldExchangeError", "FieldStore",
    "FieldValue", "Simulation", "SimulationConfig", "SimulationError",
    "SimulationState", "Solver", "SolverContext", "SolverState",
]
__version__ = "0.1.0"
