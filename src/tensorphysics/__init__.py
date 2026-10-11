"""TensorPhysics: solver-neutral multiphysics orchestration."""
from .config import SimulationConfig
from .diagnostics import DiagnosticRecord, Diagnostics
from .protocols import Solver, SolverContext, SolverState
from .scheduler import Simulation, SimulationError, SimulationState

__all__ = ["DiagnosticRecord", "Diagnostics", "Simulation", "SimulationConfig",
           "SimulationError", "SimulationState", "Solver", "SolverContext", "SolverState"]
__version__ = "0.1.0"
