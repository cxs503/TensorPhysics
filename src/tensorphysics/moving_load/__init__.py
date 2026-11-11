"""Moving ice load on the hull girder -> structural response -> FNV forecast.

Module map
----------
``beam``
    Uniform Euler-Bernoulli hull-girder beam (validated principal particulars,
    SS natural frequencies, mode shapes, static anchors, critical velocity).
``paths``
    Load paths travelling along the span and their Galerkin projection on the
    beam modes (``Q_n(t) = P(t) sin(n pi x(t)/L)``).
``analytic``
    Closed-form undamped modal solution for a point force traversing the span
    (Fryba), including the ``Omega_n -> omega_n`` resonance limit and the
    free-vibration continuation after the load leaves.
``modal_solver``
    Modal-superposition Newmark integration, modal energy and injected work.
``fd_reference``
    Independent explicit finite-difference solver (no modal truncation) used as
    a cross-method reference.
``ice_field``
    Reduced discrete floe stream (Korzhavin pressure-area crushing force) ->
    moving contact pulses ``F(x, t)``.
``fnv``
    ``FNV = a_max / (g f_dominant)`` with the ice-class habitability bands.
``analysis``
    Acceptance study: criteria (a)-(f) plus the operational 5 m/s ice case.

The package needs NumPy and is therefore not imported from
``tensorphysics/__init__.py`` (the core stays dependency-free).
"""

from __future__ import annotations

from .analysis import (
    CriterionResult,
    coupling_strength,
    critical_velocity_check,
    degeneracy_check,
    fnv_velocity_sweep,
    ice_field_case,
    ice_speed_sweep,
    library_integrator_cross_check,
    modal_truncation_check,
    reference_load_n,
    run_acceptance_study,
    static_anchor,
)
from .analytic import analytic_undamped_modal_state, analytic_undamped_response
from .beam import (
    HullGirderBeam,
    STEEL_E_MODULUS_PA,
    hull_girder_beam_from_particulars,
    reference_hull_girder,
)
from .fd_reference import fd_beam_response, fd_stability_limit
from .fnv import (
    FNV_HABITABLE_MAX,
    FNV_TOLERABLE_MAX,
    GRAVITY_M_S2,
    FnvAssessment,
    assess_fnv,
    classify_fnv,
    dominant_frequency,
)
from .ice_field import (
    IceFloeField,
    encounter_frequency_hz,
    floe_stream_paths,
    korzhavin_contact_force,
    reference_floe_field,
    sample_floe_field,
)
from .modal_solver import ModalResponse, modal_matrices, solve_modal_response
from .paths import LoadPath, modal_force_matrix, moving_point_load, total_load_series

__all__ = [
    "CriterionResult",
    "FNV_HABITABLE_MAX",
    "FNV_TOLERABLE_MAX",
    "FnvAssessment",
    "GRAVITY_M_S2",
    "HullGirderBeam",
    "IceFloeField",
    "LoadPath",
    "ModalResponse",
    "STEEL_E_MODULUS_PA",
    "analytic_undamped_modal_state",
    "analytic_undamped_response",
    "assess_fnv",
    "classify_fnv",
    "coupling_strength",
    "critical_velocity_check",
    "degeneracy_check",
    "dominant_frequency",
    "encounter_frequency_hz",
    "fd_beam_response",
    "fd_stability_limit",
    "floe_stream_paths",
    "fnv_velocity_sweep",
    "hull_girder_beam_from_particulars",
    "ice_field_case",
    "ice_speed_sweep",
    "korzhavin_contact_force",
    "library_integrator_cross_check",
    "modal_force_matrix",
    "modal_matrices",
    "modal_truncation_check",
    "moving_point_load",
    "reference_floe_field",
    "reference_hull_girder",
    "reference_load_n",
    "run_acceptance_study",
    "sample_floe_field",
    "solve_modal_response",
    "static_anchor",
    "total_load_series",
]