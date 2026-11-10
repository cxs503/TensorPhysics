"""Ice-induced hull vibration coupled to underwater radiated noise.

Public API of the coupling chain (campaign node P210W2-N06):

* :class:`~tensorphysics.ice_vib_noise.chain.CoupledCase` — input set;
  :func:`~tensorphysics.ice_vib_noise.chain.run_coupled_chain` /
  :func:`~tensorphysics.ice_vib_noise.chain.run_with_idempotency` — the chain.
* :class:`~tensorphysics.ice_vib_noise.modal.HullGirderModes` — free-free
  hull-girder modal basis with an orthogonality audit.
* :mod:`~tensorphysics.ice_vib_noise.response` — exact piecewise-linear modal
  superposition and the energy ledger; Newmark as an independent check.
* :mod:`~tensorphysics.ice_vib_noise.radiator` — baffled planar Rayleigh
  radiator (far-field pressure, radiated power, radiation resistance).
* :mod:`~tensorphysics.ice_vib_noise.source` — ice crushing load series and
  crushing-energy budget.
* :mod:`~tensorphysics.ice_vib_noise.bands` — IEC 61260-1 1/3-octave algebra.
* :mod:`~tensorphysics.ice_vib_noise.acoustics` — spherical spreading + Thorp
  absorption, source/received levels, criterion thresholds.
* :mod:`~tensorphysics.ice_vib_noise.envelope` — ISO 19906-informed vibration
  envelope and its equivalent SPL envelope.
"""

from .acoustics import (
    NMFS_CONTINUOUS_PTS_DB,
    P_REF_UPASCAL,
    range_to_threshold_m,
    received_level_db,
    source_level_db,
    thorp_absorption_db_per_km,
    transmission_loss_db,
)
from .bands import OneThirdOctaveBands, band_mean_square, one_third_octave_bands
from .chain import CoupledCase, canonical_digest, run_coupled_chain, run_with_idempotency
from .envelope import VibrationEnvelope, band_margins, iso19906_informed_envelope
from .modal import HullGirderModes, modal_basis_from_fem
from .radiator import (
    RadiatorSurface,
    piston_directivity,
    piston_radiation_efficiency,
    radiated_power,
    radiation_efficiency,
)
from .response import (
    ModalResponse,
    energy_balance,
    solve_modal_exact,
    solve_modal_newmark,
    steady_state_amplitude,
)
from .source import IceExcitationCase, bow_load_distribution, build_ice_excitation, crushing_energy_j

__all__ = [
    "CoupledCase",
    "IceExcitationCase",
    "HullGirderModes",
    "modal_basis_from_fem",
    "RadiatorSurface",
    "ModalResponse",
    "VibrationEnvelope",
    "OneThirdOctaveBands",
    "run_coupled_chain",
    "run_with_idempotency",
    "canonical_digest",
    "solve_modal_exact",
    "solve_modal_newmark",
    "steady_state_amplitude",
    "energy_balance",
    "radiated_power",
    "radiation_efficiency",
    "piston_radiation_efficiency",
    "piston_directivity",
    "one_third_octave_bands",
    "band_mean_square",
    "band_margins",
    "iso19906_informed_envelope",
    "build_ice_excitation",
    "bow_load_distribution",
    "crushing_energy_j",
    "thorp_absorption_db_per_km",
    "transmission_loss_db",
    "source_level_db",
    "received_level_db",
    "range_to_threshold_m",
    "P_REF_UPASCAL",
    "NMFS_CONTINUOUS_PTS_DB",
]
