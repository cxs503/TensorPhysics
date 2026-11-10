"""Propeller-hull-ice three-field coupling (W5-N04).

Physics chain
-------------
1. Propeller model (actuator disk or BEM) → thrust T, torque Q, wake velocity
2. Wake model → axial induction factor a, wake velocity V_wake
3. Ice load model → ice resistance R_ice(V_ship - V_wake) with wake modulation
4. Hull structure model → heave/pitch response to ice excitation
5. Coupling iteration → thrust pulsation δT from hull vibration feedback
6. Transfer function → H_thrust(f) = S_thrust(f) / S_ice(f)

Key feedback loops
------------------
(a) Propeller → Wake → Ice:
    Thrust accelerates water → wake reduces ship-ice relative velocity → 
    lower ice resistance
    
(b) Ice → Hull → Propeller:
    Ice load excites hull motions → heave/pitch modulates propeller inflow →
    thrust pulsation

References
----------
- Lindqvist, G. (1989) "A straightforward method for calculation of ice
  resistance of ships", POAC'89.
- Carlton, J.S. (2007) "Marine Propellers and Propulsion", 2nd ed.
- Bose, D.J. (1991) "Ice-induced propeller torque fluctuations", POAC'91.
"""

from .propeller import ActuatorDisk, BladeElementMethod, PropellerGeometry
from .wake import WakeModel, WakeField
from .ice_load import IceProperties, HullGeometry, IceLoadModel, lindqvist_ice_resistance
from .hull_structure import (
    HullStructure,
    MultiDOFResponse,
    compute_natural_frequencies,
    frequency_response_function,
)
from .coupling import CouplingSolver, CouplingResult, compute_transfer_function

__all__ = [
    # Propeller
    "ActuatorDisk",
    "BladeElementMethod",
    "PropellerGeometry",
    # Wake
    "WakeModel",
    "WakeField",
    # Ice load
    "IceProperties",
    "HullGeometry",
    "IceLoadModel",
    "lindqvist_ice_resistance",
    # Hull structure
    "HullStructure",
    "MultiDOFResponse",
    "compute_natural_frequencies",
    "frequency_response_function",
    # Coupling
    "CouplingSolver",
    "CouplingResult",
    "compute_transfer_function",
]
