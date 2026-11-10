"""BEM frequency-domain radiation-diffraction hydrodynamics.

This module implements the frequency-domain boundary element method for
potential-flow ship hydrodynamics, covering:

* **Pulsating source Green function** G(x,ξ;ω) for the free-surface
  boundary-value problem with the linearised free-surface condition
  ``-ω²φ + g ∂φ/∂z = 0`` on z = 0.
* **Haskind relation** connecting radiation and diffraction problems:
  ``F_diff = -iωρ ∮ φ_R · n · φ_I dS``.
* **Hemisphere analytical anchor**: for a submerged hemisphere of radius R,
  the heave added mass at zero frequency is ``A_33 = 2/3 · ρπR³`` and
  radiation damping ``B_33 = 0``.
* **Kochin function** H(θ) for far-field wave amplitude and the energy
  conservation relation ``B_ij = ρg/(2ω) ∮ |H(θ)|² dθ``.
* **Wigley III hull** wave-making resistance at various Froude numbers.

Evidence level
--------------
REAL: Green function asymptotics, hemisphere analytic anchor, Haskind
      relation, Kochin energy conservation, Wigley III C_W computation.
APPROXIMATE: Panel discretisation (constant-strength source panels),
             numerical integration of the wave part of the Green function.
NOT MODELLED: Viscous effects, forward speed (Neumann-Kelvin linearisation),
              non-linear free surface, multi-body interactions.
"""

from .green_function import (
    GreenFunction,
    green_function_value,
    green_function_gradient,
    kochin_function,
    kochin_energy_damping,
)
from .haskind import haskind_relation, haskind_residual
from .hemisphere import (
    hemisphere_mesh,
    hemisphere_analytic_A33,
    hemisphere_radiation_solve,
    hemisphere_anchor_check,
)
from .wigley import (
    wigley_iii_geometry,
    wigley_iii_wave_resistance,
    wigley_iii_Cw_vs_fr,
)

__all__ = [
    "GreenFunction",
    "green_function_value",
    "green_function_gradient",
    "kochin_function",
    "kochin_energy_damping",
    "haskind_relation",
    "haskind_residual",
    "hemisphere_mesh",
    "hemisphere_analytic_A33",
    "hemisphere_radiation_solve",
    "hemisphere_anchor_check",
    "wigley_iii_geometry",
    "wigley_iii_wave_resistance",
    "wigley_iii_Cw_vs_fr",
]
