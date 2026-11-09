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
from .research_policy import ResearchDecision, decide_next_action
from .agent_subprocess import SubprocessAgentBackend
from .scheduler import Simulation, SimulationError, SimulationState
from .campaign import (
    CouplingClock, CouplingClockSpec, ExchangeEvent, FailureFace, FailureFacePacket,
    FailureFaceV1, GeometryContractRegistry, GeometryHandle, LedgerEntry, PrincipalDimensions,
    QuantityKind, ReferenceFluxes, SideLedger, build_reference_dual_ledgers, clock_drift,
    reconcile, reconcile_books, schema_fingerprints, tetrahedron_face_packet,
    verify_alignment, verify_same_batch,
)
from .ice_sheet import (
    WinklerPlateParams,
    navier_center_deflection,
    navier_deflection_field,
    solve_winkler_plate_fd,
    winkler_kw_parametric,
)
from .multifield_coupling import (
    CouplingDomain,
    EnergyFlux,
    EnergyLedger,
    InterpolationOrder,
    MultiFieldCouplingSimulator,
    MultiFieldCouplingResult,
    SubCyclingConfig,
    SimpleLBMSolver,
    SimpleFEMSolver,
    SimpleDEMSolver,
    interpolate_temporal,
)
from .fuel_cell import (
    FARADAY_CONSTANT,
    H2_MOLAR_MASS,
    H2_LOWER_HEATING_VALUE,
    PEMFCParameters,
    FuelCellEnergyLedger,
    FuelComparison,
    activation_overpotential,
    cell_voltage,
    concentration_overpotential,
    compare_fuel_paths,
    hydrogen_consumption_rate,
    hydrogen_mass_rate,
    hydrogen_specific_consumption,
    nernst_voltage,
    ohmic_overpotential,
    polarization_curve,
    stack_power,
    stack_voltage,
    standard_fuel_comparison,
)

__all__ = [
    "ArtifactStore", "ArtifactStoreError", "LocalArtifactStore", "S3ArtifactStore",
    "Checkpoint", "CheckpointError", "CheckpointStore", "CouplingContext",
    "CouplingError", "CouplingManager", "CouplingPhase", "DiagnosticRecord",
    "Diagnostics", "ArtifactRef", "ManifestError", "RunManifest", "RunStatus",
    "ValidationStatus", "ResearchGoal", "ExperimentPlan", "ValidationReport",
    "ResearchMemoryRecord", "ResearchContractError", "ResearchDecision", "decide_next_action", "FieldDescriptor", "FieldExchangeError",
    "FieldStore", "FieldValue", "Simulation", "SimulationConfig", "SimulationError",
    "SimulationState", "Solver", "SolverContext", "SolverState",
    "PlatformRegistryError", "TensorPlatformRunRegistry",
    "AgentBackend", "AgentRequest", "AgentResult", "InvalidTaskTransition",
    "ResearchRuntimeError", "ResearchTask", "ResearchTaskStore", "TaskEvent",
    "TaskNotFoundError", "TaskStatus", "ResearchTaskRunner", "ResearchValidator",
    "SubprocessAgentBackend",
    "WinklerPlateParams", "navier_center_deflection", "navier_deflection_field",
    "solve_winkler_plate_fd", "winkler_kw_parametric",
    "CouplingClock", "CouplingClockSpec", "ExchangeEvent", "FailureFace",
    "FailureFacePacket", "FailureFaceV1", "GeometryContractRegistry", "GeometryHandle",
    "LedgerEntry", "PrincipalDimensions", "QuantityKind", "ReferenceFluxes", "SideLedger",
    "build_reference_dual_ledgers", "clock_drift", "reconcile", "reconcile_books",
    "schema_fingerprints", "tetrahedron_face_packet", "verify_alignment", "verify_same_batch",
    # Multi-field coupling
    "CouplingDomain", "EnergyFlux", "EnergyLedger", "InterpolationOrder",
    "MultiFieldCouplingSimulator", "MultiFieldCouplingResult", "SubCyclingConfig",
    "SimpleLBMSolver", "SimpleFEMSolver", "SimpleDEMSolver", "interpolate_temporal",
    # Fuel cell
    "FARADAY_CONSTANT", "H2_MOLAR_MASS", "H2_LOWER_HEATING_VALUE",
    "PEMFCParameters", "FuelCellEnergyLedger", "FuelComparison",
    "activation_overpotential", "cell_voltage", "concentration_overpotential",
    "compare_fuel_paths", "hydrogen_consumption_rate", "hydrogen_mass_rate",
    "hydrogen_specific_consumption", "nernst_voltage", "ohmic_overpotential",
    "polarization_curve", "stack_power", "stack_voltage", "standard_fuel_comparison",
]
__version__ = "0.1.0"
