"""Thermo-mechanical coupling verification suite (thermal structural FEM).

Three analytical-anchor families, each built from genuine finite element
discretisations (no empirical fits, no hard-coded reference values):

1. **Bimetallic strip** -- layered composite beam / 2-D Q4 plane-stress
   continuum driven by a uniform temperature rise, compared with the
   Timoshenko (1925) composite-beam curvature and the tip deflection.
2. **Thermal stress concentration at a circular hole** -- annular plate under a
   steady radial temperature gradient, ring elements vs the exact
   Timoshenko-Goodier thermo-elastic solution.
3. **Transient thermal shock** -- semi-infinite solid with a surface
   temperature step, theta-method conduction + eigenstrain elasticity vs the
   Carslaw-Jaeger erfc similarity solution.

Constitutive core: ``sigma = C : (eps - alpha dT I)`` (thermal eigenstrain as
an initial strain).  Boundary conditions supported by the thermal solvers:
Dirichlet, Neumann (flux) and Robin (convection).
"""

from .heat import heat1d_steady, heat1d_transient, graded_mesh, q4_steady_conduction
from .thermoelastic import (
    LayeredBeamSection,
    apply_dirichlet,
    axisym_ring_thermoelastic,
    beam_solve,
    plane_stress_C,
    q4_plane_stress,
    q8_plane_stress,
)
from .bimetal import (
    DEFAULT_CASE,
    arc_end_deflection,
    bimetal_beam_fe,
    bimetal_q4_fe,
    bimetal_q4_mesh,
    bimetal_q8_fe,
    bimetal_q8_mesh,
    curvature_from_strain_profile,
    bimetal_section,
    exact_composite_solution,
    timoshenko_curvature,
    timoshenko_end_deflection,
)
from .hole_thermal import (
    annulus_integral_Tr,
    annulus_temperature,
    annulus_thermal_stress,
    hole_fe_sweep,
    thermal_stress_concentration,
)
from .thermal_shock import (
    DEFAULT_MATERIAL,
    half_space_stress_from_temperature,
    kappa_thermal,
    lamé,
    surface_step_stress,
    surface_step_temperature,
    thermal_shock_fe,
)

__all__ = [name for name in dir() if not name.startswith("_")]