"""Seakeeping spectral analysis module.

Provides:
- Bimodal wave spectrum construction (Torsethaugen swell + wind sea)
- Directional spreading models (cos^s/2 family)
- ISO 15016 directional assumption bias quantification
"""

from .bimodal_spectrum import (
    TorsethaugenSpectrum,
    BimodalSpectrum,
    spectral_moments,
    significant_wave_height,
    peak_period,
    mean_zero_crossing_period,
    mean_wave_period,
)
from .directional_spreading import (
    cos2s_spreading,
    directional_spectrum,
    recover_spreading_parameter,
    directional_moment,
)
from .iso15016_bias import (
    iso15016_short_crested_response,
    iso15016_long_crested_response,
    iso15016_bias_band,
    quantify_iso15016_bias,
)

__all__ = [
    "TorsethaugenSpectrum",
    "BimodalSpectrum",
    "spectral_moments",
    "significant_wave_height",
    "peak_period",
    "mean_zero_crossing_period",
    "mean_wave_period",
    "cos2s_spreading",
    "directional_spectrum",
    "recover_spreading_parameter",
    "directional_moment",
    "iso15016_short_crested_response",
    "iso15016_long_crested_response",
    "iso15016_bias_band",
    "quantify_iso15016_bias",
]
