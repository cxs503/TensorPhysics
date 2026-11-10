"""Adaptive h-refinement with ZZ (Zienkiewicz–Zhu) a-posteriori error estimation.

Public surface
--------------
* :mod:`.mesh2d` — triangulation container, quality metrics, conforming
  red-green h-refinement, guarded Delaunay flipping.
* :mod:`.elasticity2d` — plane stress/strain material, CST assembly, Dirichlet
  solve, element stresses, energy-norm metrics.
* :mod:`.problem` — predicate-based boundary-value problem (Dirichlet and
  traction data follow the geometry, not element tags).
* :mod:`.estimator` — SPR patch recovery, ZZ indicators, effectivity index,
  reference-field re-sampling for measuring the actual error.
* :mod:`.marking` — Dörfler (bulk) marking and two baselines.
* :mod:`.adaptive` — SOLVE → ESTIMATE → MARK → REFINE driver, plus the uniform
  (global red) curve used as the baseline.
* :mod:`.cases` — Cook membrane, Lamé thick-walled cylinder, crack-tip
  singularity anchors with their analytic references.
"""
from .adaptive import AdaptiveRun, AdaptiveStep, adaptive_refinement, doubling_refinement
from .elasticity2d import Material2D, PlaneSolution, cst_b_matrices, element_stress
from .estimator import (
    ReferenceStressField,
    ZzEstimate,
    actual_energy_error,
    effectivity_index,
    nodal_average_stress,
    spr_recover_nodal_stress,
    zz_estimate,
)
from .marking import dorfler_mark, mark_above_mean, mark_fraction
from .mesh2d import (
    TriMesh,
    boundary_edges,
    edge_records,
    mesh_summary,
    min_angles,
    nodal_weights,
    quad_to_tris,
    signed_areas,
)
from .problem import DirichletBC, PlaneProblem, TractionBC
from .refine import flip_to_quality, refine_longest_edge, refine_red_green
from .cases import (
    LameFields,
    annulus_mesh,
    annulus_projector,
    center_crack_KI,
    cook_mesh,
    cook_problem,
    cook_tip_displacement,
    crack_mesh,
    crack_problem,
    displacement_l2_error,
    ki_from_cod,
    ki_from_jintegral,
    ki_from_stress,
    lame_problem,
    structured_quad_mesh,
)

__all__ = [
    "AdaptiveRun",
    "AdaptiveStep",
    "adaptive_refinement",
    "doubling_refinement",
    "Material2D",
    "PlaneSolution",
    "element_stress",
    "ReferenceStressField",
    "ZzEstimate",
    "actual_energy_error",
    "effectivity_index",
    "nodal_average_stress",
    "spr_recover_nodal_stress",
    "zz_estimate",
    "dorfler_mark",
    "mark_above_mean",
    "mark_fraction",
    "TriMesh",
    "boundary_edges",
    "edge_records",
    "quad_to_tris",
    "signed_areas",
    "cst_b_matrices",
    "refine_longest_edge",
    "mesh_summary",
    "min_angles",
    "nodal_weights",
    "DirichletBC",
    "PlaneProblem",
    "TractionBC",
    "flip_to_quality",
    "refine_red_green",
    "LameFields",
    "annulus_mesh",
    "annulus_projector",
    "center_crack_KI",
    "cook_mesh",
    "cook_problem",
    "cook_tip_displacement",
    "crack_mesh",
    "crack_problem",
    "displacement_l2_error",
    "ki_from_cod",
    "ki_from_jintegral",
    "ki_from_stress",
    "lame_problem",
    "structured_quad_mesh",
]