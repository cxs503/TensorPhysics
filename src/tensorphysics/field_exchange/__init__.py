"""Conservative cross-solver field exchange (LBM/DEM -> FEM) with audits.

Public surface
--------------
* :mod:`~tensorphysics.field_exchange.mapping` -- the exchange operators
  (``map_panels`` overlap-area weighting, ``map_contacts`` partition-of-unity
  lumping with minimum-norm moment completion) and the conservation audits.
* :mod:`~tensorphysics.field_exchange.lbm` -- compact D2Q9 reference channel used
  to generate *physical* wall tractions (Couette shear, Poiseuille pressure) plus
  the lattice -> SI similarity transform.
* :mod:`~tensorphysics.field_exchange.fem` -- structural targets that accept an
  arbitrary nodal load vector (plane-stress Q4, plate strip, BFS Kirchhoff plate).
* :mod:`~tensorphysics.field_exchange.analytics` -- closed-form/series anchors.
* :mod:`~tensorphysics.field_exchange.verification` -- the three end-to-end
  cases and their criteria numbers.
* :mod:`~tensorphysics.field_exchange.acceptance` -- ``run_acceptance()`` returns
  the full JSON payload with every criterion, its threshold and pass/fail.
* :mod:`~tensorphysics.field_exchange.volume_transfer` -- the *volumetric* primitives:
  conservative LBM cell-centred stress -> FEM nodal field (Galerkin L2 projection) and
  DEM Lagrangian particles -> FVM cell volume fraction (CIC / NGP / exact-overlap),
  with the conservation audits and the Roache GCI helper.
"""
from .analytics import (hertz_force, navier_center_deflection,
                        navier_coefficients, navier_field, sine_panel_integral,
                        strip_sine_pressure_center,
                        strip_uniform_pressure_center)
from .fem import PlaneStressQ4, PlateBFS, PlateStripBending
from .lbm import ChannelResult, LatticeScale, run_channel, wall_shear_from_profile
from .mapping import (QuadGrid, audit_contact_map, audit_panel_map, clip_rect,
                      integrate_polygon, map_contacts, map_panels,
                      map_panels_to_basis, moment_completion,
                      nearest_node_lumping, nodal_resultants,
                      panel_reference_resultants, polygon_area)
from .verification import (LITERATURE_PLATE_COEF, case_contact, case_couette,
                           case_poiseuille)
from .volume_transfer import (CartGrid, NodalFieldMap, Q1Mesh, VolumeFractionMap,
                              audit_nodal_field, audit_volume_fraction,
                              deposition_weights, map_cell_field_to_nodes,
                              map_particles_to_volume_fraction, roache_gci)

__all__ = [
    "QuadGrid", "clip_rect", "polygon_area", "integrate_polygon",
    "map_panels", "map_panels_to_basis", "map_contacts", "moment_completion",
    "nearest_node_lumping", "audit_panel_map", "audit_contact_map",
    "nodal_resultants", "panel_reference_resultants",
    "run_channel", "wall_shear_from_profile", "LatticeScale", "ChannelResult",
    "PlaneStressQ4", "PlateStripBending", "PlateBFS",
    "navier_coefficients", "navier_center_deflection", "navier_field",
    "strip_uniform_pressure_center", "strip_sine_pressure_center",
    "hertz_force", "sine_panel_integral",
    "case_couette", "case_poiseuille", "case_contact", "LITERATURE_PLATE_COEF",
    "CartGrid", "Q1Mesh", "NodalFieldMap", "VolumeFractionMap",
    "deposition_weights", "map_cell_field_to_nodes",
    "map_particles_to_volume_fraction", "audit_nodal_field", "audit_volume_fraction",
    "roache_gci",
]