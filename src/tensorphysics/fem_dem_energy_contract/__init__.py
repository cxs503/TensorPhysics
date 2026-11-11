"""FEM→DEM ice plate fracture energy conservation contract (p470w2-n05).

Physical goal
-------------
Deliver an energy conservation contract for the FEM→DEM ice plate fracture
handoff — an energy ledger from FEM plate bending → DEM bond fracture →
fragment kinetic energy, making ice-breakage cross-solver-coupling physically
credible.

Energy ledger
-------------
    U_bend = G_f + E_bond + E_k + E_dissipated

where:
- U_bend: FEM plate bending strain energy (P·δ/2 for concentrated load)
- G_f: fracture energy = 2·γ_ice × fracture area
- E_bond: DEM bond fracture energy = G_f × A_fragment
- E_k: fragment kinetic energy = ½ m v²
- E_dissipated: viscous/damping dissipation

Acceptance criteria
-------------------
C1: FEM plate bending strain energy vs analytical solution ≤ 2 %
C2: Fracture energy G_f vs surface energy formula ≤ 1 %
C3: DEM bond fracture energy vs G_f × A ≤ 1 %
C4: Total energy conservation residual ≤ 5 %
C5: Negative control — no fracture → residual ≤ 1 %
C6: Parameter sweep h = 0.3–1.0 m (5 levels), σ_flex = 1.0–2.0 MPa (5 levels),
    each combination residual ≤ 5 %
"""

from __future__ import annotations

from .mindlin_plate import (
    MindlinPlateSolution,
    solve_mindlin_plate,
    MindlinPlateResult,
)
from .fracture_energy import (
    FractureEnergy,
    compute_fracture_energy,
)
from .dem_bond import (
    DEMBondEnergy,
    compute_dem_bond_energy,
)
from .fragment_kinetic import (
    FragmentKineticEnergy,
    compute_fragment_kinetic_energy,
)
from .energy_ledger import (
    EnergyLedger,
    run_energy_ledger,
    EnergyLedgerResult,
    run_parameter_sweep,
)

__all__ = [
    "MindlinPlateSolution",
    "solve_mindlin_plate",
    "MindlinPlateResult",
    "FractureEnergy",
    "compute_fracture_energy",
    "DEMBondEnergy",
    "compute_dem_bond_energy",
    "FragmentKineticEnergy",
    "compute_fragment_kinetic_energy",
    "EnergyLedger",
    "run_energy_ledger",
    "EnergyLedgerResult",
    "run_parameter_sweep",
]
