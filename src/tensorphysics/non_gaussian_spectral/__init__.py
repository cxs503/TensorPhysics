"""Non-Gaussian spectral fatigue admission domain for ice-induced loads.

This module implements:
1. Winterstein Hermite moment-translation transform (Gaussian → non-Gaussian)
2. Non-Gaussian spectral damage methods (Dirlik/Benasciutti-Tovo/Winterstein→Rayleigh)
3. Error map: spectral shortcut vs rainflow truth on (γ3, γ4, α₂) grid
4. Machine-judgeable admission law: given moments, decide if spectral error ≤ 10%

Physical context: Ice-induced hotspot stresses are bounded, positively skewed,
and leptokurtic (peaked). The spectral shortcut (Dirlik, BT, Winterstein→Rayleigh)
must be validated against time-domain rainflow counting before being trusted for
annual fatigue damage estimation.

References
----------
* Winterstein, S.R. (1987) Nonlinear vibration models for extremes and fatigue.
  J. Eng. Mech. 113(10), 1543-1558.
* Winterstein, S.R., Ude, T.C., Kleiven, G. (1993) Springing and non-linear
  wave effects on extreme loads and fatigue. Proc. OTRC.
* Dirlik, T. (1985) Application of computers in fatigue analysis. PhD thesis.
* Benasciutti, D., Tovo, R. (2005) Spectral methods for lifetime prediction.
  Int. J. Fatigue 27, 867-877.
"""

from .winterstein import (
    WintersteinTransform,
    winterstein_hermite_transform,
    winterstein_inverse,
    gaussianize_signal,
    non_gaussianize_signal,
)
from .non_gaussian_spectral import (
    NonGaussianSpectralMoments,
    non_gaussian_dirlik_damage_rate,
    non_gaussian_rayleigh_damage_rate,
    non_gaussian_bt_damage_rate,
    winterstein_corrected_rayleigh,
    spectral_damage_non_gaussian,
)
from .admission import (
    AdmissionGrid,
    compute_error_map,
    fit_admission_boundary,
    check_admission,
    AdmissionResult,
    validate_geometry_contract,
)

__all__ = [
    "WintersteinTransform",
    "winterstein_hermite_transform",
    "winterstein_inverse",
    "gaussianize_signal",
    "non_gaussianize_signal",
    "NonGaussianSpectralMoments",
    "non_gaussian_dirlik_damage_rate",
    "non_gaussian_rayleigh_damage_rate",
    "non_gaussian_bt_damage_rate",
    "winterstein_corrected_rayleigh",
    "spectral_damage_non_gaussian",
    "AdmissionGrid",
    "compute_error_map",
    "fit_admission_boundary",
    "check_admission",
    "AdmissionResult",
    "validate_geometry_contract",
]
