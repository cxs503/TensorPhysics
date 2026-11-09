"""3-D isoparametric solid element kernel (Hex8 / Hex20).

From-scratch, dependency-light (numpy + scipy sparse) small-strain solid
element library used as the verification backbone for the structural nodes:

* :mod:`~tensorphysics.solid_fem.shape` -- Hex8 trilinear and Hex20 quadratic
  serendipity shape functions, 1-point / 2x2x2 / 3x3x3 Gauss rules;
* :mod:`~tensorphysics.solid_fem.element` -- Jacobians, B-matrices, full /
  B-bar / selective-reduced-integration stiffness, Flanagan-Belytschko
  hourglass control;
* :mod:`~tensorphysics.solid_fem.material` -- isotropic elasticity and J2
  plasticity with radial return mapping and the analytic consistent tangent;
* :mod:`~tensorphysics.solid_fem.mesh` -- structured box / annulus meshes,
  Hex20 promotion, consistent pressure loads;
* :mod:`~tensorphysics.solid_fem.solver` -- assembly, Dirichlet solve,
  incremental Newton-Raphson for plasticity;
* :mod:`~tensorphysics.solid_fem.verification` -- patch test, volumetric
  locking study, hourglass study, uniaxial J2 benchmark, Lame thick cylinder.
"""

from .element import (
    ElementKinematics,
    b_matrix,
    element_internal_force,
    element_kinematics,
    element_stiffness,
    element_strain_energy,
    element_volume,
    hourglass_energy,
    hourglass_gamma,
    hourglass_stiffness,
    jacobian,
)
from .material import (
    J2Material,
    PlasticState,
    deviatoric_projector,
    iso_inner,
    isotropic_elasticity,
    lame_parameters,
    numerical_tangent,
    uniaxial_tension_analytic,
)
from .mesh import (
    HEX20_FACES,
    HEX_FACES,
    SolidMesh,
    annulus_mesh,
    box_mesh,
    pressure_load,
    promote_hex8_to_hex20,
    quad_shape4,
    quad_shape8,
)
from .shape import (
    GAUSS_RULES,
    HEX20_NATURAL,
    HEX8_NATURAL,
    default_rule,
    gauss_rule,
    hex8_shape,
    hex20_shape,
    n_nodes,
    shape,
)
from .solver import (
    ElasticSolution,
    NonlinearSolution,
    assemble_stiffness,
    gauss_stresses,
    gauss_stresses_plastic,
    hourglass_energy_ratio,
    nodal_average_stress,
    solve_dirichlet,
    solve_elastic,
    solve_plastic,
    strain_energy,
)
from .verification import (
    CylinderSetup,
    cantilever_tip_deflection,
    cylinder_limit_pressure,
    timoshenko_tip_deflection,
    hourglass_rank,
    hourglass_study,
    lame_limit_pressure,
    lame_radial_displacement,
    lame_sigma_z,
    lame_stresses,
    patch_test,
    thick_cylinder,
    uniaxial_plasticity,
)

__version__ = "0.1.0"

__all__ = [
    "ElementKinematics",
    "b_matrix",
    "element_internal_force",
    "element_kinematics",
    "element_stiffness",
    "element_strain_energy",
    "element_volume",
    "hourglass_energy",
    "hourglass_gamma",
    "hourglass_stiffness",
    "jacobian",
    "J2Material",
    "PlasticState",
    "deviatoric_projector",
    "iso_inner",
    "isotropic_elasticity",
    "lame_parameters",
    "numerical_tangent",
    "uniaxial_tension_analytic",
    "HEX20_FACES",
    "HEX_FACES",
    "SolidMesh",
    "annulus_mesh",
    "box_mesh",
    "pressure_load",
    "promote_hex8_to_hex20",
    "quad_shape4",
    "quad_shape8",
    "GAUSS_RULES",
    "HEX20_NATURAL",
    "HEX8_NATURAL",
    "default_rule",
    "gauss_rule",
    "hex8_shape",
    "hex20_shape",
    "n_nodes",
    "shape",
    "ElasticSolution",
    "NonlinearSolution",
    "assemble_stiffness",
    "gauss_stresses",
    "gauss_stresses_plastic",
    "hourglass_energy_ratio",
    "nodal_average_stress",
    "solve_dirichlet",
    "solve_elastic",
    "solve_plastic",
    "strain_energy",
    "CylinderSetup",
    "cantilever_tip_deflection",
    "cylinder_limit_pressure",
    "hourglass_rank",
    "hourglass_study",
    "lame_limit_pressure",
    "lame_radial_displacement",
    "lame_sigma_z",
    "lame_stresses",
    "patch_test",
    "thick_cylinder",
    "timoshenko_tip_deflection",
    "uniaxial_plasticity",
]