"""Three-stage chain ledger: wave -> light 6-DOF body -> slamming whipping.

The package turns the P34 "three stages side by side" picture into a *serial*
chain with explicit hand-off faces and a machine-readable readiness verdict:

===========  ==================================================  ===========
face         what is verified                                    module
===========  ==================================================  ===========
A            wave pressure field -> rigid-body force/moment      ``interfaces``
B            body motion -> moving/immersed boundary             ``interfaces``
C            slamming pressure -> modal generalised force        ``interfaces``
E            energy ledger over one wave period                 ``runner``
T            interpolation order / synchronisation offset       ``time_ledger``
N            negative controls (no stabilisation, 0th order)     ``stabilization``
===========  ==================================================  ===========

``runner.run_chain_case`` produces all six products and evaluates the
``chain_readiness.v1`` verdict in ``readiness``.

Evidence levels are stated per module; briefly: the wave field, the panel
kinematics, the modal beam and the convolutions are REAL computations, while the
hydrodynamic coefficients (added mass/damping), the slamming deadrise
coefficient and the barge's ``EI`` are documented model inputs.  The gates only
judge the hand-off between the stages, never the absolute accuracy of those
inputs.
"""

from __future__ import annotations

from .beam import FlexibleBarge, ModalBeam
from .body import BodyDynamics, BoxBody, Panels, RigidPose, box_panels, force_moment, rigid_velocity
from .interfaces import (
    duhamel_response,
    face_a_convergence,
    face_a_force,
    face_b_closed_flux,
    face_b_power_reciprocity,
    face_b_volume_rate,
    face_c_generalised_force,
    rock4_modal,
)
from .readiness import CRITERIA, GATE_ID, REQUIRED_PRODUCTS, SCHEMA, evaluate_chain, write_readiness
from .slamming import WagnerSlamming
from .stabilization import InterfaceProblem, divergence_detection_rate, solve_interface, spectral_radius
from .time_ledger import detection, footprint, handoff_grid, reconstruct, signal_error
from .waves import StokesWave, dispersion_wavenumber

__all__ = [
    "SCHEMA",
    "GATE_ID",
    "REQUIRED_PRODUCTS",
    "CRITERIA",
    "StokesWave",
    "dispersion_wavenumber",
    "BoxBody",
    "BodyDynamics",
    "Panels",
    "RigidPose",
    "box_panels",
    "rigid_velocity",
    "force_moment",
    "FlexibleBarge",
    "ModalBeam",
    "WagnerSlamming",
    "face_a_force",
    "face_a_convergence",
    "face_b_closed_flux",
    "face_b_volume_rate",
    "face_b_power_reciprocity",
    "face_c_generalised_force",
    "duhamel_response",
    "rock4_modal",
    "InterfaceProblem",
    "solve_interface",
    "spectral_radius",
    "divergence_detection_rate",
    "reconstruct",
    "handoff_grid",
    "signal_error",
    "footprint",
    "detection",
    "evaluate_chain",
    "write_readiness",
]