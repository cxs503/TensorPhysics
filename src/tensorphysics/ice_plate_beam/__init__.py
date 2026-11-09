"""Plate-beam combination structure under ice load: elastoplastic response and ultimate capacity.

Physical model
--------------
A stiffened plate panel (plate + T-stiffener) subjected to ice pressure loading.
The analysis traces the full response from elastic → plate yield → plastic hinge
formation → beam yield → progressive collapse → ultimate bearing capacity.

Modules:
- ice_load: Half-sine pulse ice load model with IACS UR I3 peak pressure
- plate_beam: Plate-beam combination geometry and section properties
- elastoplastic: Elastoplastic analysis (elastic → yield → plastic hinge → collapse)
- ultimate_capacity: Ultimate bearing capacity (plate + beam plastic moment combination)
- analysis: Main analysis driver producing load-displacement curves and failure modes

References:
- IACS UR I3: Ice class requirements for polar ships
- IMO Polar Code: Structural requirements for ships operating in polar waters
- Hughes (1983): Ship structural design under ice loads
"""

__version__ = "0.1.0"
