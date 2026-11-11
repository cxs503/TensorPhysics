"""Spectral estimator gate: Welch × DPSS multi-taper × Lomb–Scargle cross-validation.

Provides three independent PSD estimators with AR(1)/AR(2) analytical anchors
for verifiable spectral analysis of time-series data.

Estimators
----------
- :func:`welch_psd` : Welch's method (segmented, windowed, averaged periodogram)
- :func:`dpss_multitaper_psd` : Thomson's multi-taper method using DPSS/Slepian windows
- :func:`lomb_scargle_psd` : Lomb–Scargle periodogram for non-uniformly sampled data

Analytical anchors
------------------
- :func:`ar1_psd` : AR(1) process closed-form PSD
- :func:`ar2_psd` : AR(2) process closed-form PSD

Cross-validation
----------------
- :func:`cross_validate_psd` : Compare band energy across estimators
- :func:`run_spectral_gate` : Full gate execution with all acceptance criteria
"""

from __future__ import annotations

from .welch_estimator import welch_psd
from .dpss_multitaper import dpss_multitaper_psd
from .lomb_scargle import lomb_scargle_psd
from .ar_models import ar1_psd, ar2_psd, generate_ar1, generate_ar2
from .cross_validate import cross_validate_psd, band_energy, run_spectral_gate

__all__ = [
    "welch_psd",
    "dpss_multitaper_psd",
    "lomb_scargle_psd",
    "ar1_psd",
    "ar2_psd",
    "generate_ar1",
    "generate_ar2",
    "cross_validate_psd",
    "band_energy",
    "run_spectral_gate",
]
