"""Moving ice load on the hull girder: ship-ice-beam coupling and FNV forecast.

Campaign node ``p360w1-n05-moving-load-fnv``.  The package answers one
engineering question:

    A ship advancing through broken ice at speed ``V`` receives a train of
    contact forces that *travels aft along the hull*.  Which speeds make the
    hull girder vibrate, and how severe is the resulting habitability index
    ``FNV = a / (g f)``?

Sub-modules
-----------
``beam``
    Euler-Bernoulli hull-girder beam: analytic free-free / simply supported
    frequencies, modal basis (shapes, generalized masses and stiffnesses) and an
    independent Hermite beam-FE eigenvalue cross-check.
``ice_field``
    Statistical broken-ice field (floe length / strength distributions,
    brittle-ductile crushing sawtooth) sampled into a moving contact force
    history, plus the ``f_ice = V / s`` repetition relation and the spectral
    dominant-frequency helper.
``moving_load``
    Solution of the modal equations: closed-form moving-force series for a
    simply supported beam (the A2 reference), Newmark average-acceleration
    integration for an arbitrary moving load with the full energy ledger, and a
    frequency-domain (FRF) route used to verify the steady-state amplitude.
``fnv``
    FNV definition, narrowband acceleration extraction and the ice-class tiers
    (habitable / tolerable / unacceptable).
``analysis``
    The design case and the A1-A4 acceptance driver, plus two physics studies
    (floe-spacing regularity, sweeping vs stationary contact).

The package needs NumPy only; it is deliberately not imported from
``tensorphysics/__init__.py`` so that the core orchestration package keeps its
dependency-free import surface.
"""

from __future__ import annotations

from .analysis import (
    IceMovingLoadCase,
    build_reference_case,
    classic_moving_load_verification,
    energy_verification,
    evaluate_acceptance,
    floe_regularity_study,
    fnv_speed_scan,
    geometry_audit,
    natural_frequency_audit,
    static_verification,
    sweep_vs_stationary_study,
)
from .beam import (
    FREE_FREE,
    SIMPLY_SUPPORTED,
    GirderBeam,
    ModalBasis,
    beam_fem_frequencies,
    classic_critical_speed,
    static_midspan_deflection_analytic,
    static_midspan_deflection_modal,
)
from .fnv import (
    FNV_HABITABLE_MAX,
    FNV_TOLERABLE_MAX,
    STANDARD_GRAVITY,
    FnvResult,
    band_amplitude,
    classify_fnv,
    evaluate_speed,
    fnv_value,
)
from .ice_field import (
    ContactForceHistory,
    IceFloeField,
    build_contact_history,
    dominant_frequency_hz,
    sawtooth_fraction,
)
from .moving_load import (
    MovingLoadResponse,
    analytic_point_load_response,
    fourier_amplitude,
    modal_newmark,
    run_contact_history,
    run_moving_point_load,
    steady_state_fundamental_amplitude,
)

__all__ = [
    "IceMovingLoadCase",
    "build_reference_case",
    "geometry_audit",
    "natural_frequency_audit",
    "static_verification",
    "classic_moving_load_verification",
    "fnv_speed_scan",
    "energy_verification",
    "floe_regularity_study",
    "sweep_vs_stationary_study",
    "evaluate_acceptance",
    "GirderBeam",
    "ModalBasis",
    "SIMPLY_SUPPORTED",
    "FREE_FREE",
    "beam_fem_frequencies",
    "classic_critical_speed",
    "static_midspan_deflection_analytic",
    "static_midspan_deflection_modal",
    "IceFloeField",
    "ContactForceHistory",
    "build_contact_history",
    "dominant_frequency_hz",
    "sawtooth_fraction",
    "MovingLoadResponse",
    "analytic_point_load_response",
    "modal_newmark",
    "run_moving_point_load",
    "run_contact_history",
    "fourier_amplitude",
    "steady_state_fundamental_amplitude",
    "FnvResult",
    "fnv_value",
    "classify_fnv",
    "band_amplitude",
    "evaluate_speed",
    "FNV_HABITABLE_MAX",
    "FNV_TOLERABLE_MAX",
    "STANDARD_GRAVITY",
]
