"""Ice-induced vibration spectral analysis (campaign node P233W2-N07).

Pipeline
--------
``excitation``
    Intermittent brittle-ductile ice-crushing load history (DEM-style
    parameterisation) with the repetition frequency ``f_ice = U/s`` and its
    Welch one-sided PSD; analytic narrow-band PSD model.
``structural``
    SDOF oscillator and fixed-free lumped chain with analytic eigenfrequencies,
    Rayleigh damping, direct and modal-superposition FRFs.
``response``
    Response PSD ``S_yy = |H|^2 S_ff`` (displacement / velocity / acceleration),
    spectral RMS, exact zero-order-hold time-domain cross-check.
``criterion``
    Resonance-risk tiers ``lock_in`` / ``near_resonance`` / ``clear`` from the
    half-power band, verified against the amplification predicate.
``case``
    Lighthouse literature case (AC5) and hull-girder 5-DOF case, plus the
    driver :func:`run_acceptance` that evaluates AC1 - AC5.

The package depends on NumPy only (plus the Welch estimator already verified in
:mod:`tensorphysics.ice_vibration_transfer`) and is deliberately not imported
from ``tensorphysics/__init__.py`` so the core package keeps its
dependency-free import surface.
"""

from __future__ import annotations

from .case import (
    LIGHTHOUSE_LITERATURE_BAND_HZ,
    LIGHTHOUSE_LITERATURE_FREQ_HZ,
    HullChainCase,
    LighthouseCase,
    run_acceptance,
    run_hull_chain_case,
    run_lighthouse_case,
)
from .criterion import (
    HALF_POWER_AMPLITUDE_RATIO,
    NEAR_RESONANCE_BAND_FACTOR,
    ResonanceVerdict,
    amplification_ratio,
    classification_consistency,
    classify_frequency_grid,
    half_power_edges_hz,
    resonance_risk,
)
from .excitation import (
    IceCrushingSeries,
    crushing_frequency,
    intermittent_crushing_series,
    narrowband_ice_psd,
    welch_psd,
)
from .response import (
    SDOFTimeHistory,
    modal_response_psd,
    mdof_response_psd,
    psd_peak_frequency,
    refine_psd,
    rms_from_psd,
    sdof_acceleration_psd,
    sdof_response_psd,
    spectral_moments,
    white_noise_rms_analytic,
    zero_order_hold_response,
)
from .structural import (
    ChainSystem,
    SDOFSystem,
    chain_analytic_frequencies,
    chain_matrices,
    mdof_frf_direct,
    mdof_frf_modal,
    rayleigh_modal_damping,
    sdof_frf,
    state_space_frf_sdof,
)

__all__ = [
    "ChainSystem",
    "HALF_POWER_AMPLITUDE_RATIO",
    "HullChainCase",
    "IceCrushingSeries",
    "LIGHTHOUSE_LITERATURE_BAND_HZ",
    "LIGHTHOUSE_LITERATURE_FREQ_HZ",
    "LighthouseCase",
    "NEAR_RESONANCE_BAND_FACTOR",
    "ResonanceVerdict",
    "SDOFSystem",
    "SDOFTimeHistory",
    "amplification_ratio",
    "chain_analytic_frequencies",
    "chain_matrices",
    "classification_consistency",
    "classify_frequency_grid",
    "crushing_frequency",
    "half_power_edges_hz",
    "intermittent_crushing_series",
    "mdof_frf_direct",
    "mdof_frf_modal",
    "mdof_response_psd",
    "modal_response_psd",
    "narrowband_ice_psd",
    "psd_peak_frequency",
    "rayleigh_modal_damping",
    "refine_psd",
    "resonance_risk",
    "rms_from_psd",
    "run_acceptance",
    "run_hull_chain_case",
    "run_lighthouse_case",
    "sdof_acceleration_psd",
    "sdof_frf",
    "sdof_response_psd",
    "spectral_moments",
    "state_space_frf_sdof",
    "welch_psd",
    "white_noise_rms_analytic",
    "zero_order_hold_response",
]