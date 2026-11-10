"""Geometric nonlinear verification suite for large-deflection structural mechanics.

Covers four verification stages:
1. Cantilever beam elastica (exact elliptic integral solution + FEM)
2. Von Karman circular plate (large deflection under center load)
3. Shallow arch snap-through buckling (full path including negative stiffness)
4. Convergence order verification + energy conservation

References:
- Frisch-Fay, R. (1962). Flexible Bars. Butterworths.
- Bisshopp & Drucker (1945). Large deflection of cantilever beams.
- Camp (1949). Large deflections of circular plates.
- Berger (1955). Large deflections of plates and shallow shells.
- Budiansky & Roth (1962). Axisymmetric dynamic buckling of shallow arches.
- Timoshenko & Woinowsky-Krieger (1959). Theory of Plates and Shells.
"""

from .elastica import (
    ElasticaExact,
    ElasticaFEM,
    solve_elastica_problem,
)
from .elastica_fem import (
    ElasticaFEMSolver,
    solve_elastica_fem,
)
from .von_karman_plate import (
    VonKarmanPlateExact,
    VonKarmanPlateFEM,
    solve_von_karman_plate,
)
from .shallow_arch import (
    ShallowArchModel,
    VonMisesTruss,
    solve_snap_through,
)
from .convergence import (
    convergence_study_elastica,
    convergence_study_plate,
)
from .verification import (
    bisshopp_drucker_reference,
    camp_berger_reference,
    budiansky_roth_reference,
    verify_energy_conservation_elastica,
    verify_energy_conservation_plate,
    convergence_order_elastica,
    convergence_order_plate,
    full_verification_elastica,
    full_verification_plate,
    full_verification_arch,
)

__all__ = [
    "ElasticaExact",
    "ElasticaFEM",
    "solve_elastica_problem",
    "ElasticaFEMSolver",
    "solve_elastica_fem",
    "VonKarmanPlateExact",
    "VonKarmanPlateFEM",
    "solve_von_karman_plate",
    "ShallowArchModel",
    "VonMisesTruss",
    "solve_snap_through",
    "convergence_study_elastica",
    "convergence_study_plate",
    "bisshopp_drucker_reference",
    "camp_berger_reference",
    "budiansky_roth_reference",
    "verify_energy_conservation_elastica",
    "verify_energy_conservation_plate",
    "convergence_order_elastica",
    "convergence_order_plate",
    "full_verification_elastica",
    "full_verification_plate",
    "full_verification_arch",
]
