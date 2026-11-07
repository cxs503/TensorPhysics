"""Hull-girder hydroelasticity: springing and whipping by modal superposition.

Module map
----------
``timoshenko``  non-uniform free-free Timoshenko beam FE (dry modes) plus the
                Euler-Bernoulli analytic roots used as the dry-mode gate.
``sections``    2-D strip hydrodynamics: infinite-frequency sectional added mass
                from a boundary-value problem (Dirichlet, ``phi = 0`` on the
                waterline) with closed-form kernel integrals.
``wet``         ship-girder property idealisation, added-mass distribution, and
                the weighted (wet) modes/damping solved in the dry modal basis.
``response``    strip wave excitation, springing stress spectrum + RMS,
                slamming whipping transient, S-N fatigue, rigid/elastic coupling
                table and the section-modulus utilisation gate.
``presets``     KCS and KVLCC2 benchmark hull cases (rule-minimum section modulus).
``runner``      end-to-end case study returning every acceptance-gate number.

Scope and evidence
------------------
REAL: Timoshenko dry modes, 2-D added mass (validated against ``rho pi a^2``),
      modal superposition, convolution transients, S-N bookkeeping.
APPROXIMATE-STRIP: wave excitation (Smith-corrected strip, no 3-D diffraction),
      slamming pressure idealisation (half-sine pulse, relative-velocity-squared
      scaling), narrow-band Rayleigh fatigue.
WITHHELD: hydrodynamic radiation damping of the strip (needs the radiating
      free-surface Green function) - modal damping is the structural input.
NOT MODELLED: horizontal/torsional hull-girder modes, strip rotary added inertia,
      green-water and bow-flare slamming details, springing-whipping interaction.
"""

from .presets import KCS, KVLCC2, HullCase, hull_cases, rule_section_modulus
from .response import (
    coupling_table,
    generalised_force,
    sectional_excitation,
    slamming_force,
    springing_fatigue_rate,
    springing_stress,
    utilisation_gate,
    whipping_fatigue,
    whipping_response,
)
from .runner import added_mass_gate, convergence_gate, dry_mode_analytic_gate, run_case, sign_gates
from .sections import (
    Section2D,
    added_mass_matrix,
    circle_section,
    heave_added_mass,
    panelize,
    superellipse_section,
)
from .timoshenko import (
    EB_FREE_FREE_ROOTS,
    BeamProperties,
    TimoshenkoBeam,
    euler_bernoulli_free_free_roots,
    uniform_beam,
)
from .wet import added_mass_distribution, modal_damping, ship_beam_properties, wet_modes

__all__ = [
    "KCS",
    "KVLCC2",
    "HullCase",
    "hull_cases",
    "rule_section_modulus",
    "Section2D",
    "added_mass_matrix",
    "circle_section",
    "heave_added_mass",
    "panelize",
    "superellipse_section",
    "EB_FREE_FREE_ROOTS",
    "BeamProperties",
    "TimoshenkoBeam",
    "euler_bernoulli_free_free_roots",
    "uniform_beam",
    "added_mass_distribution",
    "modal_damping",
    "ship_beam_properties",
    "wet_modes",
    "coupling_table",
    "generalised_force",
    "sectional_excitation",
    "slamming_force",
    "springing_fatigue_rate",
    "springing_stress",
    "utilisation_gate",
    "whipping_fatigue",
    "whipping_response",
    "added_mass_gate",
    "convergence_gate",
    "dry_mode_analytic_gate",
    "run_case",
    "sign_gates",
]
