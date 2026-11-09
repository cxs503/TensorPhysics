"""Ice-induced structural fatigue assessment module.

Complete chain from ice load to fatigue life per DNV-OS-C401:

1. Ice load time series → FFT → load PSD
2. Structural FRF → response PSD
3. Response synthesis → stress time history
4. Rainflow counting (ASTM E1049-85)
5. S-N curves (DNV-OS-C401: B1, F, W3)
6. Miner cumulative damage
7. DNV-OS-C401 fatigue check with DFL

Submodules
----------
sn_dnv_os_c401 : S-N curves per DNV-OS-C401 Table 2-1
spectrum : Frequency-domain analysis (PSD, FRF, response)
ice_fatigue_chain : Complete assessment chain
"""

from .sn_dnv_os_c401 import (
    SNCurveOS401,
    DNV_OS_C401_CURVES,
    thickness_correction,
    fatigue_limit_stress,
)
from .spectrum import (
    IceLoadSpectrum,
    StructuralFRF,
    ResponseSpectrum,
    compute_load_psd,
    compute_frf,
    compute_response_psd,
    spectral_moments,
    ice_load_time_series,
    response_time_series_from_psd,
    spectral_fatigue_assessment,
)
from .ice_fatigue_chain import (
    IceFatigueChain,
    IceFatigueResult,
    DNVCheckResult,
    run_ice_fatigue_assessment,
)

__all__ = [
    # S-N curves
    "SNCurveOS401",
    "DNV_OS_C401_CURVES",
    "thickness_correction",
    "fatigue_limit_stress",
    # Spectrum
    "IceLoadSpectrum",
    "StructuralFRF",
    "ResponseSpectrum",
    "compute_load_psd",
    "compute_frf",
    "compute_response_psd",
    "spectral_moments",
    "ice_load_time_series",
    "response_time_series_from_psd",
    "spectral_fatigue_assessment",
    # Chain
    "IceFatigueChain",
    "IceFatigueResult",
    "DNVCheckResult",
    "run_ice_fatigue_assessment",
]
