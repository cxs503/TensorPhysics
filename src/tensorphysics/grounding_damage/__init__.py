"""Grounding damage assessment for ice-going ships (ice ridge / shoal stranding).

Physical chain implemented by this package:

``contact``   Hertz-verified elastic contact, rigid ice-ridge keel, elastic shoal
``panel``     von Karman large-deflection elastoplastic plate panel (Ritz)
``material``  polar steel grades, plane-stress J2 return mapping, ``eps_f``
``damage``    tearing criterion, stiffener buckling, damage extent and opening
``flooding``  Bernoulli orifice inflow, added-weight sinkage, equilibrium state
``stability`` hydrostatics, GZ curve of the flooded ship, damage-stability criteria
``scenario``  end-to-end event (kinetic energy -> contact -> damage -> flood -> GZ)
              and the parameter scan ``V, h_ridge, alpha``
``verification`` analytic/literature/FE anchors for every link of the chain
"""

from .material import (SteelGrade, POLAR_NV_F, POLAR_NV_D, MILD_STEEL,
                       uniaxial_curve, bilinear_reference, layer_strain)
from .panel import PanelSpec, PanelResponse, VonKarmanPanel
from .contact import (HertzContact, HertzSolution, HalfSpaceCompliance,
                      hertz_sphere_on_plane, RidgeGeometry, RidgeContact,
                      ShoalContact)
from .damage import (StiffenerSection, StiffenerCheck, PanelDamageState,
                     DamageAssessment, assess_ridge_damage, check_stiffener)
from .flooding import (Compartment, BottomOpening, inflow_rate, transient_flood,
                       analytic_filling_time, quasi_static_flood)
from .stability import (HullForm, Section, IMO_CRITERIA, gz_curve,
                        check_imo_criteria)
from .scenario import (SHIP_15000DWT, DEFAULT_PANEL, DEFAULT_STIFFENER,
                       GroundingCase, design_panel, run_grounding_case,
                       parameter_sweep)
from . import verification

__all__ = [
    "SteelGrade", "POLAR_NV_F", "POLAR_NV_D", "MILD_STEEL",
    "uniaxial_curve", "bilinear_reference", "layer_strain",
    "PanelSpec", "PanelResponse", "VonKarmanPanel",
    "HertzContact", "HertzSolution", "HalfSpaceCompliance",
    "hertz_sphere_on_plane", "RidgeGeometry", "RidgeContact", "ShoalContact",
    "StiffenerSection", "StiffenerCheck", "PanelDamageState",
    "DamageAssessment", "assess_ridge_damage", "check_stiffener",
    "Compartment", "BottomOpening", "inflow_rate", "transient_flood",
    "analytic_filling_time", "quasi_static_flood",
    "HullForm", "Section", "IMO_CRITERIA", "gz_curve", "check_imo_criteria",
    "SHIP_15000DWT", "DEFAULT_PANEL", "DEFAULT_STIFFENER", "GroundingCase",
    "design_panel", "run_grounding_case", "parameter_sweep", "verification",
]

__version__ = "1.0.0"