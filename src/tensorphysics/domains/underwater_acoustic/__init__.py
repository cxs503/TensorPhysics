"""Underwater acoustic propagation loss (TL) module.

Implements the classical underwater sound propagation chain with analytical
and numerical anchors for verification:

1. Pekeris waveguide (uniform water layer over semi-infinite bottom):
   - Analytical normal-mode eigenvalues k_n via the Pekeris dispersion relation
   - Modal TL(r,z) = 20 log10(|p/p_ref|) with mode summation

2. Munk deep-sea sound-speed profile:
   - c(z) = c_m [1 + ε(η - 1 + e^{-η})],  η = 2(z - z_m)/B
   - Numerical normal-mode eigenvalue solution (shooting method)
   - Convergence-zone (CZ) position prediction

3. Ray tracing:
   - Snell's law + turning-point detection
   - Eigenray arrival time and amplitude
   - Geometric spreading + absorption

4. Dual-path cross-validation:
   - Ray TL vs normal-mode TL at high frequency (f >= 500 Hz)

5. Energy conservation:
   - ∫|p|² r dr vs source power closure

6. Negative controls:
   - Uniform medium: TL = 20 log10(r) + αr (spherical spreading + absorption)
   - Reciprocity: source/receiver interchange leaves TL unchanged

7. Bottom-parameter sensitivity:
   - c1 +10% → TL change direction consistent with physics

All formulas assume linear acoustics, time-harmonic excitation, and
cylindrical symmetry (range-independent environment).
"""

from .pekeris import (
    PekerisWaveguide,
    pekeris_dispersion,
    pekeris_mode_eigenvalue,
    pekeris_tl,
    pekeris_mode_shape,
)
from .munk import (
    MunkProfile,
    munk_sound_speed,
    munk_normal_mode_tl,
    munk_convergence_zone,
    munk_eigenvalue_shooting,
)
from .ray import (
    RayTracer,
    ray_trace,
    eigenray_tl,
    snell_launch_angle,
    turning_point_depth,
    ray_arrival_time,
)
from .cross_validation import (
    cross_validate_ray_normal_mode,
    dual_path_tl,
)
from .energy import (
    energy_conservation_check,
    modal_energy_closure,
)
from .negative_controls import (
    uniform_medium_tl,
    reciprocity_check,
    bottom_sensitivity_check,
)
from .core import (
    RHO_WATER,
    C_WATER,
    P_REF,
    TL,
    UnderwaterAcousticEnvironment,
    UnderwaterAcousticResult,
)

__all__ = [
    "RHO_WATER", "C_WATER", "P_REF", "TL",
    "UnderwaterAcousticEnvironment", "UnderwaterAcousticResult",
    # Pekeris
    "PekerisWaveguide", "pekeris_dispersion", "pekeris_mode_eigenvalue",
    "pekeris_tl", "pekeris_mode_shape",
    # Munk
    "MunkProfile", "munk_sound_speed", "munk_normal_mode_tl",
    "munk_convergence_zone", "munk_eigenvalue_shooting",
    # Ray
    "RayTracer", "ray_trace", "eigenray_tl", "snell_launch_angle",
    "turning_point_depth", "ray_arrival_time",
    # Cross-validation
    "cross_validate_ray_normal_mode", "dual_path_tl",
    # Energy
    "energy_conservation_check", "modal_energy_closure",
    # Negative controls
    "uniform_medium_tl", "reciprocity_check", "bottom_sensitivity_check",
]
