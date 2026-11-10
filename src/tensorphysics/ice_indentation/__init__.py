"""J2 elastoplastic ice indentation with penalty contact (TensorPhysics node p233w1-N06).

Physical problem
----------------
A rigid indenter (sphere or flat punch) is driven into an ice block.  The ice
is described by a small-strain J2 (von Mises) elastoplastic material with
linear isotropic hardening -- the standard idealisation of ice crushing under
a confined indentation front, where the contact zone is dominated by
pressure-dependent micro-cracking and crushing and the *deviatoric* strength
controls the attainable mean contact pressure.

What this package provides
--------------------------
* :mod:`~tensorphysics.ice_indentation.constitutive` -- vectorised twin of the
  J2 radial return mapping (verified against
  :class:`tensorphysics.solid_fem.material.J2Material`) plus the material-point
  drivers used by AC1 (uniaxial stress path) and AC2 (pure-shear yield);
* :mod:`~tensorphysics.ice_indentation.contact` -- **new capability**: penalty
  node-to-rigid-surface contact with an active set updated inside the Newton
  loop and the full consistent contact tangent;
* :mod:`~tensorphysics.ice_indentation.contact_facets` -- **new capability**:
  facet-integrated penalty traction (2x2 Gauss on the Hex8 surface quads),
  which resolves the contact area to second order in the element size and is
  what makes a *plastic* indentation force mesh-convergent;
* :mod:`~tensorphysics.ice_indentation.mesh` -- power-graded Hex8 block meshes
  (quarter model with symmetry planes) that resolve the contact patch without
  paying for a uniform half-space mesh;
* :mod:`~tensorphysics.ice_indentation.indentation` -- displacement-controlled
  incremental Newton driver returning the force-displacement curve, contact
  area/pressure history and a full energy ledger;
* :mod:`~tensorphysics.ice_indentation.analytics` -- Hertz, Sneddon, Johnson
  cavity-expansion and confined-compression reference solutions;
* :mod:`~tensorphysics.ice_indentation.verification` -- machine-checkable
  acceptance criteria (AC1 ... AC6) as a single ``dict``.

The finite-element kernels (Hex8 shape functions, B-matrices, Gauss rules) and
the J2 return mapping are *reused* from
:mod:`tensorphysics.solid_fem`; this package adds the constitutive batch
driver, the contact nonlinearity, the graded mesh family and the verification
harness.
"""

from .analytics import (
    IndentationReference,
    confined_uniaxial_strain,
    hertz_contact_radius,
    hertz_force,
    hertz_mean_pressure,
    hertz_yield_onset,
    johnson_mean_pressure,
    johnson_validity_ratio,
    reduced_modulus,
    sneddon_flat_punch_force,
)
from .constitutive import (
    BatchResponse,
    UniaxialResponse,
    batch_return_map,
    shear_yield_stress,
    tangent_error,
    uniaxial_stress_response,
)
from .contact import (
    ContactState,
    FlatPunchIndenter,
    SphericalIndenter,
    active_set,
    contact_state,
)
from .contact_facets import build_surface_facets, facet_contact_state
from .indentation import (
    IndentationResult,
    penalty_stiffness,
    quarter_block_bcs,
    simulate_indentation,
    surface_penalty_stiffness,
    top_surface_data,
)
from .mesh import (
    face_nodes,
    graded_axis,
    graded_box_mesh,
    quarter_block_mesh,
    surface_nodal_areas,
)
from .verification import acceptance_report

__all__ = [
    # constitutive
    "batch_return_map", "BatchResponse", "uniaxial_stress_response",
    "UniaxialResponse", "shear_yield_stress", "tangent_error",
    # contact
    "SphericalIndenter", "FlatPunchIndenter", "ContactState", "contact_state",
    "active_set",
    # mesh
    "graded_axis", "graded_box_mesh", "quarter_block_mesh", "face_nodes",
    "surface_nodal_areas",
    "build_surface_facets", "facet_contact_state",
    # driver
    "simulate_indentation", "IndentationResult", "quarter_block_bcs",
    "penalty_stiffness", "surface_penalty_stiffness", "top_surface_data",
    # analytics
    "reduced_modulus", "hertz_force", "hertz_contact_radius",
    "hertz_mean_pressure", "hertz_yield_onset", "johnson_mean_pressure",
    "johnson_validity_ratio", "sneddon_flat_punch_force",
    "confined_uniaxial_strain",
    "IndentationReference",
    # verification
    "acceptance_report",
]