"""Ice-induced bow-flare slamming: ice-water-structure three-body transient contact.

Campaign node P33-W2-N01.  A wave-carried ice floe slams into the flared bow
panel; the module resolves the transient contact mechanics of the
ice / water / structure triad and outputs

* (a) the peak-pressure envelope ``P_max(V_ice, theta, H_s)``;
* (b) the contact-area history ``A(t)`` and pressure field ``p(x, y, t)``;
* (c) the local structural response (plate stress / deflection).

Model chain (each stage is an auditable, reference-anchored idealisation --
see the module docstrings for the reference of every relation):

* :mod:`.properties`  -- ice Ih / seawater / structural-steel property basis;
* :mod:`.plate_fem`   -- 2-D Kirchhoff plate (mass-spring lattice) for the bow
  flare panel, with mesh refinement and a static circular-load stiffness;
* :mod:`.contact`     -- flat-punch (Hertz-family) contact law with the
  linear-elastic -> yield -> brittle-fracture ice constitutive path;
* :mod:`.hydro`       -- Wagner-type slamming pressure and wave orbital velocity;
* :mod:`.model`       -- the coupled transient ODE (floe + panel + water) with
  explicit energy bookkeeping;
* :mod:`.sweep`       -- the (V_ice, theta, H_s) parameter sweep and the
  ``P_max ~ V_ice**alpha`` exponent fit;
* :mod:`.verification`-- the machine-checkable acceptance-criteria harness
  (negative control, exponent gate, energy closure, mesh convergence).

Evidence level
--------------
APPROXIMATE-REDUCED-ORDER: the floe is a rigid disk, the panel is a Kirchhoff
plate, and the water is a Wagner-type pressure plus added mass / damping.  This
is a contract-level reduced model for the orchestration layer, not a
specialist CFD/CFD-FSI solver.  Every quantitative claim is tied to a closed
form or a documented empirical correlation, and the acceptance gates are
comparisons against independent references, never self-assessments.
"""

from .properties import (
    IceFloeProperties,
    SlammingWaterProperties,
    SlammingStructureProperties,
    default_floe,
    default_structure,
    default_water,
)
from .plate_fem import PlateFEM, PlateFEMResult
from .contact import ContactLaw, IceFractureState, contact_force, fracture_area
from .hydro import SlammingHydro, slamming_pressure, wave_orbital_velocity
from .model import SlammingCase, SlammingResult, run_impact
from .sweep import SweepResult, fit_velocity_exponent, run_sweep
from .verification import run_acceptance

__all__ = [
    "IceFloeProperties",
    "SlammingWaterProperties",
    "SlammingStructureProperties",
    "default_floe",
    "default_structure",
    "default_water",
    "PlateFEM",
    "PlateFEMResult",
    "ContactLaw",
    "IceFractureState",
    "contact_force",
    "fracture_area",
    "SlammingHydro",
    "slamming_pressure",
    "wave_orbital_velocity",
    "SlammingCase",
    "SlammingResult",
    "run_impact",
    "SweepResult",
    "fit_velocity_exponent",
    "run_sweep",
    "run_acceptance",
]
