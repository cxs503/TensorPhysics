"""3D fluid-structure-ice (FSI-ice) three-field coupling prototype.

Scope of this package
---------------------
This package implements the *coordination* layer required by the ecosystem
charter: it owns the coupling semantics (clock synchronisation, conservative
field exchange, energy bookkeeping, acceptance gates) and it carries the three
*reduced-order but physics-based* field solvers needed to exercise that
coupling on a controlled 3D-versus-2D comparison:

``lattice``
    D3Q19 / D2Q9 lattice Boltzmann solver with half-way bounce-back moving
    bodies (momentum exchange) and a Körner--Thürey style mass-tracking free
    surface.
``dem``
    Hertz--Mindlin soft-sphere discrete element solver with buoyancy and
    Schiller--Naumann fluid drag, plus the ice-sheet fragmentation front end
    that feeds it.
``fem``
    Hexahedral linear-elasticity solver with Newmark-:math:`\\beta` time
    integration used for the hull structural response.

``runner`` orchestrates the three solvers inside a single
:class:`tensorphysics.provenance.RunManifest`, sub-cycling their clocks and
exchanging fields through the versioned packets in ``contracts``.

The three solvers are *not* the production TensorLBM/TensorDEM/TensorFEM
kernels; they are self-contained reduced implementations whose fidelity is
established against analytic references (see ``docs/FSI_ICE_3D_COUPLING_P273W3N04.md``).
What is being validated here is the *coupling*, not solver-grade accuracy:
absolute hydrodynamic forces are prototype scale, and every benchmark compares
two configurations at identical numerical Reynolds number (controlled
experiment) rather than against full-scale measurements.
"""

from .contracts import (
    CONTRACT_VERSION,
    FieldExchangeError,
    FluidFieldPacket,
    ParticlePacket,
    StructurePacket,
    check_mass_conservation,
    check_momentum_balance,
)
from .lattice import D2Q9, D3Q19, FreeSurfaceConfig, LatticeBoltzmannSolver, LatticeSpec
from .dem import DEMConfig, DEMWorld, HertzMindlinContact, IceSheetFracture
from .fem import ElastostaticHull, HullFEMConfig, Hex8Mesh, newmark_integrate
from .runner import CouplingConfig, CouplingRunner, CouplingRunResult

__all__ = [
    "CONTRACT_VERSION",
    "FieldExchangeError",
    "FluidFieldPacket",
    "ParticlePacket",
    "StructurePacket",
    "check_mass_conservation",
    "check_momentum_balance",
    "D2Q9",
    "D3Q19",
    "FreeSurfaceConfig",
    "LatticeBoltzmannSolver",
    "LatticeSpec",
    "DEMConfig",
    "DEMWorld",
    "HertzMindlinContact",
    "IceSheetFracture",
    "ElastostaticHull",
    "HullFEMConfig",
    "Hex8Mesh",
    "newmark_integrate",
    "CouplingConfig",
    "CouplingRunner",
    "CouplingRunResult",
]