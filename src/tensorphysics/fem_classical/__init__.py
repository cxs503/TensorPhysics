"""FEM classical structural verification suite.

Three analytical-anchor families for TensorFEM beam/shell/plate modules:

1. **Euler-Bernoulli cantilever beam** — end-point load & uniform load,
   deflection/rotation vs closed-form, convergence order O(h²).
2. **Kirchhoff simply-supported square plate** — uniform load, center
   deflection vs Navier double-series exact solution.
3. **Cylindrical shell axial-compression buckling** — eigenvalue buckling
   vs classical formula σ_cr = E / √(3(1−ν²)) · t/R.

Negative control: removing boundary conditions must produce a singular
stiffness matrix (condition number > 1e14).
"""

from .beam_euler import EulerBernoulliBeam2D
from .plate_kirchhoff import KirchhoffPlateSSSS
from .shell_cylindrical import CylindricalShellBuckling
from .analytical import (
    cantilever_tip_deflection_point_load,
    cantilever_tip_rotation_point_load,
    cantilever_tip_deflection_udl,
    cantilever_tip_rotation_udl,
    navier_square_plate_center_deflection,
    classical_cylindrical_shell_buckling_stress,
)

__all__ = [
    "EulerBernoulliBeam2D",
    "KirchhoffPlateSSSS",
    "CylindricalShellBuckling",
    "cantilever_tip_deflection_point_load",
    "cantilever_tip_rotation_point_load",
    "cantilever_tip_deflection_udl",
    "cantilever_tip_rotation_udl",
    "navier_square_plate_center_deflection",
    "classical_cylindrical_shell_buckling_stress",
]
