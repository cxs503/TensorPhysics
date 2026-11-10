"""Geometric nonlinear verification suite for large-deflection structural mechanics.

Covers four verification stages:
1. Cantilever beam elastica (exact elliptic integral solution)
2. Von Karman circular plate (large deflection under center load)
3. Shallow arch snap-through buckling (full path including negative stiffness)
4. Convergence order verification

References:
- Frisch-Fay, R. (1962). Flexible Bars. Butterworths.
- Bisshopp & Drucker (1945). Large deflection of cantilever beams.
- Camp (1949). Large deflections of circular plates.
- Berger (1955). Large deflections of circular plates.
- Budiansky & Roth (1962). Axisymmetric dynamic buckling of shallow arches.
"""

from .elastica import (
    ElasticaExact,
    ElasticaFEM,
    solve_elastica_problem,
)
from .von_karman_plate import (
    VonKarmanPlateExact,
    VonKarmanPlateFEM,
    solve_von_karman_plate,
)
from .shallow_arch import (
    ShallowArchModel,
    solve_snap_through,
)
from .convergence import (
    convergence_study_elastica,
    convergence_study_plate,
)

__all__ = [
    "ElasticaExact",
    "ElasticaFEM",
    "solve_elastica_problem",
    "VonKarmanPlateExact",
    "VonKarmanPlateFEM",
    "solve_von_karman_plate",
    "ShallowArchModel",
    "solve_snap_through",
    "convergence_study_elastica",
    "convergence_study_plate",
]
