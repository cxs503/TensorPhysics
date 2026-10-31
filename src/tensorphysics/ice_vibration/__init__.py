"""Ice-induced vibration: frequency lock-in and transient whiplash response.

Sub-modules
-----------
``modal``
    Hull parameters with a block-coefficient consistency check, the uniform
    free-free Euler-Bernoulli reference modal set, the TensorFEM modal
    extraction interface and Rayleigh damping.
``ice_load``
    Harmonic, quasi-periodic (intermittent crushing) and Poisson (memoryless
    impact) load generators, the ``f_ice = U/s`` crushing-frequency relation and
    the velocity-weakening ice law that drives self-excited lock-in.
``integrators``
    Newmark-beta average acceleration (linear MDOF, whiplash modal
    superposition) and classical RK4 (nonlinear SDOF: Duffing, ice feedback).
``lockin``
    Magnification factor, the 5 % lock-in band, damping limits, Duffing
    harmonic balance (fold/jump behaviour) and time-domain post-processing.
``whiplash``
    Closed-form damped SDOF response to a half-sine pulse and the two-mode
    hull-girder whiplash response (midship/quarter bending moment, bow
    displacement).
``analysis``
    :class:`IceVibrationAnalysis` — the unified driver that produces the report
    payload and evaluates the acceptance criteria.

The package needs NumPy; it is deliberately *not* imported from
``tensorphysics/__init__.py`` so that the core orchestration package keeps its
dependency-free import surface.
"""

from __future__ import annotations

from .analysis import (
    DUFFING_RELATIVE_TOLERANCE,
    FREQUENCY_RELATIVE_TOLERANCE,
    NARROWBAND_CONCENTRATION_THRESHOLD,
    SELF_EXCITED_OPERATIONAL_VELOCITY_M_S,
    WHIPLASH_RELATIVE_TOLERANCE,
    IceCondition,
    IceVibrationAnalysis,
    IceVibrationCase,
    build_reference_case,
    spectrum_peak_deviation,
)
from .ice_load import (
    IceLoadSeries,
    crushing_frequency,
    harmonic_load,
    poisson_pulse_train,
    quasi_periodic_pulse_train,
    self_excitation_velocity_window,
    velocity_weakening_ice_force,
    velocity_weakening_slope,
)
from .integrators import (
    IntegrationResult,
    assemble_modal_matrices,
    modal_to_physical,
    newmark_linear,
    newmark_mdof,
    rk4_sdof,
)
from .lockin import (
    LOCK_IN_AMPLIFICATION_THRESHOLD,
    LOCK_IN_RELATIVE_TOLERANCE,
    LockInVerdict,
    amplification_band_half_width,
    classify_lock_in,
    damping_limit_for_amplification,
    detuning_ratio,
    duffing_backbone_amplitude,
    duffing_harmonic_balance_amplitudes,
    harmonic_projection_amplitude,
    is_locked_in,
    lock_in_band_hz,
    magnification_factor,
    steady_state_amplitude,
)
from .modal import (
    HullParameters,
    ModalData,
    free_free_beam_frequency,
    free_free_beam_mode_shape,
    free_free_beam_modal_data,
    modal_data_from_fem_extraction,
    rayleigh_coefficients,
)
from .whiplash import (
    WhiplashResult,
    quasi_static_midship_moment,
    sdof_half_sine_response,
    sdof_half_sine_state,
    whiplash_response,
)

__all__ = [
    # modal
    "HullParameters", "ModalData", "free_free_beam_frequency", "free_free_beam_mode_shape",
    "free_free_beam_modal_data", "modal_data_from_fem_extraction", "rayleigh_coefficients",
    # ice load
    "IceLoadSeries", "crushing_frequency", "harmonic_load", "poisson_pulse_train",
    "quasi_periodic_pulse_train", "self_excitation_velocity_window",
    "velocity_weakening_ice_force", "velocity_weakening_slope",
    # integrators
    "IntegrationResult", "assemble_modal_matrices", "modal_to_physical", "newmark_linear",
    "newmark_mdof", "rk4_sdof",
    # lock-in
    "LOCK_IN_AMPLIFICATION_THRESHOLD", "LOCK_IN_RELATIVE_TOLERANCE", "LockInVerdict",
    "amplification_band_half_width", "classify_lock_in", "damping_limit_for_amplification",
    "detuning_ratio", "duffing_backbone_amplitude", "duffing_harmonic_balance_amplitudes",
    "harmonic_projection_amplitude", "is_locked_in", "lock_in_band_hz", "magnification_factor",
    "steady_state_amplitude",
    # whiplash
    "WhiplashResult", "quasi_static_midship_moment", "sdof_half_sine_response",
    "sdof_half_sine_state", "whiplash_response",
    # analysis
    "DUFFING_RELATIVE_TOLERANCE", "FREQUENCY_RELATIVE_TOLERANCE",
    "NARROWBAND_CONCENTRATION_THRESHOLD", "SELF_EXCITED_OPERATIONAL_VELOCITY_M_S",
    "WHIPLASH_RELATIVE_TOLERANCE", "IceCondition", "IceVibrationAnalysis", "IceVibrationCase",
    "build_reference_case", "spectrum_peak_deviation",
]