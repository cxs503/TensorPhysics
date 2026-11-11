"""3D Flow-Solid-Ice Fully Coupled Framework Prototype.

Campaign node AG29W1-N08: Build and validate a 3D coupled framework for
wedge icebreaking simulation.

Components:
- lbm3d: D3Q19 lattice Boltzmann solver for 3D flow with free-surface
- dem3d: 3D discrete element method for ice floe dynamics
- fem3d: 3D finite element method for hull structure response
- manifest: RunManifest orchestrator for clock synchronization and field exchange
- baseline_2d: 2D baseline for 3D effect comparison

The coupling loop:
    LBM (flow) → pressure → FEM (structure) → displacement → LBM boundary
    LBM (flow) → drag → DEM (ice) → contact force → FEM (structure)
    Energy ledger tracks conservation across all exchanges.

Verification anchor: 3D wedge icebreaking benchmark
- F_x/F_y/F_z time series
-碎冰场 evolution
- Comparison with 2D baseline (expect 15-30% difference in F_x)
- Energy conservation error < 5%
- 100 coupling steps without divergence
"""
from __future__ import annotations

from .lbm3d import LBM3D, LBM3DConfig, LBM3DStepDiagnostics, equilibrium3d
from .dem3d import DEM3D, DEM3DConfig, IceFloe
from .fem3d import FEM3DHull, FEM3DConfig, FEM3DResult
from .manifest import (
    RunManifest,
    RunManifestConfig,
    CouplingStatus,
    EnergyLedger,
    FieldExchange,
    CouplingDiagnostics,
)
from .baseline_2d import Baseline2D, Baseline2DConfig

__all__ = [
    # LBM 3D
    "LBM3D",
    "LBM3DConfig",
    "LBM3DStepDiagnostics",
    "equilibrium3d",
    # DEM 3D
    "DEM3D",
    "DEM3DConfig",
    "IceFloe",
    # FEM 3D
    "FEM3DHull",
    "FEM3DConfig",
    "FEM3DResult",
    # Manifest
    "RunManifest",
    "RunManifestConfig",
    "CouplingStatus",
    "EnergyLedger",
    "FieldExchange",
    "CouplingDiagnostics",
    # Baseline
    "Baseline2D",
    "Baseline2DConfig",
]
