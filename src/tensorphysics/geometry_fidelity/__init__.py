"""Geometry fidelity contract for ship-performance physics chains.

Public surface:

``hulls``
    analytic CAD truth sources (KCS, KVLCC2, SUBOFF) with two independent
    integration routes each;
``mesh``
    discrete solver integrations (FVM polyhedra, LBM voxel/IBM, BEM panels)
    plus the negative corpus of deliberately damaged grids;
``contract``
    the ``geometry_fidelity_contract.v1`` document, its schema, the gate
    verdict and structural admission into a physics chain.
"""
from .contract import (
    CONTRACT_SCHEMA,
    CONTRACT_SCHEMA_VERSION,
    GATE_PASS,
    GATE_REJECT,
    TOLERANCES,
    ContractSchemaError,
    GeometryFidelityError,
    MeshNotAdmitted,
    assert_admitted,
    build_contract,
    canonical_json,
    evaluate_gate,
    physics_chain,
    validate_contract,
)
from .hulls import HullForm, contract_hulls, kcs_hull, kvlcc2_hull, suboff_area_envelope, suboff_hull
from .mesh import (
    CORRUPTIONS,
    SolverRecord,
    bem_panels,
    coarse_records,
    corrupted_record,
    fvm_mesh,
    good_records,
    lbm_grid,
)

__all__ = [
    "CONTRACT_SCHEMA",
    "CONTRACT_SCHEMA_VERSION",
    "CORRUPTIONS",
    "GATE_PASS",
    "GATE_REJECT",
    "TOLERANCES",
    "ContractSchemaError",
    "GeometryFidelityError",
    "HullForm",
    "MeshNotAdmitted",
    "SolverRecord",
    "assert_admitted",
    "bem_panels",
    "build_contract",
    "canonical_json",
    "coarse_records",
    "contract_hulls",
    "corrupted_record",
    "evaluate_gate",
    "fvm_mesh",
    "good_records",
    "kcs_hull",
    "kvlcc2_hull",
    "lbm_grid",
    "physics_chain",
    "suboff_area_envelope",
    "suboff_hull",
    "validate_contract",
]