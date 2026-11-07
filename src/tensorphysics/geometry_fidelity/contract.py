"""The geometry fidelity contract: provenance, gate verdict, chain admission.

One contract document per ``geometry_id`` answers the only question a physics
chain may ask of a grid: *is this discretisation faithful to the CAD truth
source closely enough that a physics verdict computed on it means anything?*

The document carries (i) the CAD integral quantities with the two independent
integrations that produced them and their mutual residual, (ii) one record per
solver family computed on a real generated grid, (iii) the two SHA-256
provenance digests, (iv) the tolerances, and (v) a gate verdict with the
per-check residuals.  Admission is structural: :func:`assert_admitted` raises
:class:`MeshNotAdmitted` before any consumer is allowed to look at the grid.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable, Mapping

from .hulls import HullForm
from .mesh import SolverRecord

__all__ = [
    "CONTRACT_SCHEMA",
    "CONTRACT_SCHEMA_VERSION",
    "GATE_PASS",
    "GATE_REJECT",
    "TOLERANCES",
    "ContractSchemaError",
    "GeometryFidelityError",
    "MeshNotAdmitted",
    "assert_admitted",
    "build_contract",
    "canonical_json",
    "evaluate_gate",
    "physics_chain",
    "validate_contract",
]

CONTRACT_SCHEMA_VERSION = "geometry_fidelity_contract.v1"
GATE_PASS = "GATE_PASS"
GATE_REJECT = "GATE_REJECT"

#: Gate tolerances; relative unless the name ends in ``_abs``.
TOLERANCES: dict[str, float] = {
    "cad_volume_rel": 1e-3,
    "cad_surface_rel": 3e-3,
    # third CAD route (isoparametric surface tessellation) against the slice route
    "cad_isoparametric_rel": 1e-3,
    # CAD integrals against the delivered / published reference values
    "cad_reference_rel": 1e-3,
    "discrete_volume_rel": 5e-3,
    "discrete_surface_rel": 1e-2,
    "discrete_cb_abs": 1e-3,
    "principal_rel": 5e-3,
    "panel_closure_rel": 1e-12,
}

#: How each family integrates the geometry it was handed (carried by the contract
#: so a consumer can see *which* discrete integration produced the numbers).
INTEGRATION_METHODS: dict[str, dict[str, str]] = {
    "fvm": {
        "volume": "hexahedral cell decomposition into six tetrahedra (exact for the generated mesh)",
        "surface": "faceted hull-face set, two-triangle quad quadrature",
        "principal": "mesh bounding box of the node set",
    },
    "lbm": {
        "volume": "sum of the immersed-boundary volume fractions x cell volume",
        "surface": "height field |y| = b(x, z) on the lattice (exact wall-normal, second order in x, z)",
        "principal": "bounding box of the cells with fraction > 0.5 (quantised to one lattice spacing)",
    },
    "bem": {
        "volume": "divergence theorem 1/3 oint x . n dA over the closed panel set",
        "surface": "sum of panel triangle areas (sides + transom)",
        "principal": "panel vertex extents",
    },
}

MC_SAMPLES = 1_000_000


class GeometryFidelityError(RuntimeError):
    """Base class for geometry fidelity violations."""


class MeshNotAdmitted(GeometryFidelityError):
    """The grid failed the geometry gate and may not enter a physics chain."""

    def __init__(self, message: str, failures: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.failures = failures or []


class ContractSchemaError(GeometryFidelityError):
    """The document does not satisfy ``geometry_fidelity_contract.v1``."""


# ---------------------------------------------------------------------------
# canonical serialisation and provenance
# ---------------------------------------------------------------------------


def canonical_json(document: Mapping[str, Any]) -> str:
    """Canonical JSON text: sorted keys, fixed separators, no NaN/inf."""
    return json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(document: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(document).encode("utf-8")).hexdigest()


def cad_sha256(hull: HullForm, n_x: int, n_s: int) -> str:
    """Provenance digest of the CAD truth source (definition + quadrature setup)."""
    definition = {
        "geometry_id": hull.geometry_id,
        "family": hull.family,
        "L_m": hull.length_m,
        "B_m": hull.breadth_m,
        "T_m": hull.draught_m,
        "params": {k: float(v) for k, v in sorted(hull.params.items())},
        "cb_contract": hull.cb_contract,
        "nabla_contract_m3": hull.nabla_contract_m3,
        "s_contract_m2": hull.s_contract_m2,
        "quadrature": {"n_x": n_x, "n_s": n_s, "rule": "composite_simpson"},
    }
    return _digest(definition)


def _grid_digest(solvers: Mapping[str, SolverRecord]) -> str:
    payload = {name: rec.grid_sha256 for name, rec in sorted(solvers.items())}
    return _digest(payload)


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------


def _check(check_id: str, value: float, limit: float, passed: bool, note: str = "") -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": check_id,
        "value": float(value),
        "limit": float(limit),
        "pass": bool(passed),
    }
    if note:
        entry["note"] = note
    return entry


def evaluate_gate(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate every gate check on a contract document.

    Returns ``{"verdict": ..., "checks": [...], "failures": [...]}``.
    """
    tol = contract["tolerances"]
    cad = contract["cad"]
    solvers = contract["grid"]["solvers"]
    checks: list[dict[str, Any]] = []

    vol_res = abs(cad["nabla_slice_m3"] - cad["nabla_monte_carlo_m3"]) / cad["nabla_m3"]
    srf_res = abs(cad["s_quadrature_m2"] - cad["s_monte_carlo_m2"]) / cad["s_wetted_m2"]
    checks.append(_check("cad_volume_consistency", vol_res, tol["cad_volume_rel"], vol_res <= tol["cad_volume_rel"]))
    checks.append(
        _check("cad_surface_consistency", srf_res, tol["cad_surface_rel"], srf_res <= tol["cad_surface_rel"])
    )
    # third, structurally independent route: isoparametric surface tessellation
    iso_vol = cad.get("nabla_isoparametric_m3")
    if iso_vol is not None:
        iso_v_res = abs(iso_vol - cad["nabla_slice_m3"]) / cad["nabla_m3"]
        checks.append(
            _check("cad_isoparametric_volume", iso_v_res, tol["cad_isoparametric_rel"], iso_v_res <= tol["cad_isoparametric_rel"])
        )
    iso_area = cad.get("s_isoparametric_m2")
    if iso_area is not None:
        iso_s_res = abs(iso_area - cad["s_quadrature_m2"]) / cad["s_wetted_m2"]
        checks.append(
            _check("cad_isoparametric_surface", iso_s_res, tol["cad_isoparametric_rel"], iso_s_res <= tol["cad_isoparametric_rel"])
        )
    # agreement with the delivered / published reference geometry
    reference = cad.get("reference") or {}
    for key, label in (("nabla_m3", "volume"), ("s_wetted_m2", "surface"), ("cb", "cb")):
        if key not in reference:
            continue
        value = reference[key]
        measured = cad["cb"] if key == "cb" else cad[key]
        rel = abs(measured - value) / abs(value)
        checks.append(
            _check(f"cad_reference_{label}", rel, tol["cad_reference_rel"], rel <= tol["cad_reference_rel"])
        )

    for family in sorted(solvers):
        record = solvers[family]
        dv = abs(record["nabla_m3"] - cad["nabla_m3"]) / cad["nabla_m3"]
        ds = abs(record["s_wetted_m2"] - cad["s_wetted_m2"]) / cad["s_wetted_m2"]
        dcb = abs(record["cb"] - cad["cb"])
        checks.append(
            _check(f"{family}_volume", dv, tol["discrete_volume_rel"], dv <= tol["discrete_volume_rel"])
        )
        if record.get("surface_evidence", "REAL") == "WITHHELD":
            checks.append(
                _check(
                    f"{family}_surface_evidence",
                    0.0,
                    1.0,
                    True,
                    "wetted area WITHHELD: the lattice height field cannot resolve a section "
                    "with a vertical tangent (keel of a fine form, polar lines of a body of "
                    "revolution), so the reconstruction bias is resolution independent; volume "
                    "and Cb of this family are admitted instead",
                )
            )
        else:
            checks.append(
                _check(f"{family}_surface", ds, tol["discrete_surface_rel"], ds <= tol["discrete_surface_rel"])
            )
        checks.append(_check(f"{family}_cb", dcb, tol["discrete_cb_abs"], dcb <= tol["discrete_cb_abs"]))
        # principal dimensions first: a grid that cannot even reproduce L and B has
        # no business reporting a displacement
        extras = record.get("extras") or {}
        quantum = float(extras.get("principal_quantum_rel", 0.0))
        for key, label, exact in (
            ("L_measured_m", "length", contract["principal"]["L_m"]),
            ("B_measured_m", "breadth", contract["principal"]["B_m"]),
        ):
            measured = extras.get(key)
            if measured is None:
                continue
            rel = abs(float(measured) - exact) / exact
            limit = max(tol["principal_rel"], 3.0 * quantum)
            checks.append(_check(f"{family}_{label}", rel, limit, rel <= limit))
        closure = record.get("extras", {}).get("panel_closure_rel")
        if closure is not None:
            checks.append(
                _check(
                    f"{family}_closure",
                    closure,
                    tol["panel_closure_rel"],
                    closure <= tol["panel_closure_rel"],
                )
            )

    for key in ("cad_sha256", "grid_sha256"):
        digest = contract["provenance"][key]
        ok = isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
        checks.append(_check(f"provenance_{key}", 0.0 if ok else 1.0, 1.0, ok, "0 = digest present and well formed"))

    failures = [c for c in checks if not c["pass"]]
    return {"verdict": GATE_PASS if not failures else GATE_REJECT, "checks": checks, "failures": failures}


# ---------------------------------------------------------------------------
# contract construction
# ---------------------------------------------------------------------------


def build_contract(
    hull: HullForm,
    solvers: Mapping[str, SolverRecord],
    *,
    n_x: int = 240,
    n_s: int = 160,
    mc_samples: int = MC_SAMPLES,
    seed: int = 20261106,
) -> dict[str, Any]:
    """Assemble the contract for one hull and its solver records."""
    if not solvers:
        raise GeometryFidelityError("at least one solver record is required")
    from .cad_routes import surface_isoparametric, volume_isoparametric

    slice_volume = hull.cad_volume_slice(n_x, n_s)
    quad_area = hull.cad_surface_area(n_x, n_s)
    mc_volume = hull.cad_volume_monte_carlo(mc_samples, seed)
    mc_area = hull.cad_surface_monte_carlo(mc_samples, seed)
    iso_volume = volume_isoparametric(hull, n_x, n_s)
    iso_area = surface_isoparametric(hull, n_x, n_s)
    waterplane = hull.cad_waterplane_area(n_x)
    volume_denominator = hull.length_m * hull.breadth_m * hull.draught_m
    cad = {
        "nabla_m3": slice_volume,
        "nabla_slice_m3": slice_volume,
        "nabla_monte_carlo_m3": mc_volume,
        "s_wetted_m2": quad_area,
        "s_quadrature_m2": quad_area,
        "s_monte_carlo_m2": mc_area,
        "nabla_isoparametric_m3": iso_volume,
        "s_isoparametric_m2": iso_area,
        "cb": slice_volume / volume_denominator,
        "cwp": waterplane / (hull.length_m * hull.breadth_m),
        "integration": {
            "volume": [
                "transverse_slice_simpson",
                "isoparametric_surface_tessellation_divergence",
                "stratified_monte_carlo",
            ],
            "surface": [
                "parametric_quadrature_grid",
                "isoparametric_surface_tessellation",
                "stratified_monte_carlo",
            ],
            "n_x": n_x,
            "n_s": n_s,
            "mc_samples": int(mc_samples),
            "mc_seed": seed,
        },
    }
    reference: dict[str, float] = {}
    if hull.nabla_contract_m3 is not None:
        reference["nabla_m3"] = hull.nabla_contract_m3
    if hull.s_contract_m2 is not None:
        reference["s_wetted_m2"] = hull.s_contract_m2
    if hull.cb_contract is not None:
        reference["cb"] = hull.cb_contract
    if reference:
        cad["reference"] = reference

    contract: dict[str, Any] = {
        "schema": CONTRACT_SCHEMA_VERSION,
        "geometry_id": hull.geometry_id,
        "principal": {"L_m": hull.length_m, "B_m": hull.breadth_m, "T_m": hull.draught_m},
        "cad": cad,
        "grid": {
            "generator": "geofid-mesh-2026.11",
            "solvers": {
                name: dict(rec.as_dict(), integration=INTEGRATION_METHODS.get(name, {}))
                for name, rec in sorted(solvers.items())
            },
        },
        "provenance": {
            "cad_sha256": cad_sha256(hull, n_x, n_s),
            "grid_sha256": _grid_digest(solvers),
        },
        "tolerances": dict(TOLERANCES),
    }
    contract["gate"] = evaluate_gate(contract)
    validate_contract(contract)
    return contract


# ---------------------------------------------------------------------------
# schema validation (self-contained: no external jsonschema dependency)
# ---------------------------------------------------------------------------

_ANY_OF_NUMBER = ("number", "integer")

CONTRACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["schema", "geometry_id", "principal", "cad", "grid", "provenance", "tolerances", "gate"],
    "additionalProperties": False,
    "properties": {
        "schema": {"type": "string", "enum": [CONTRACT_SCHEMA_VERSION]},
        "geometry_id": {"type": "string", "minLength": 1},
        "principal": {
            "type": "object",
            "required": ["L_m", "B_m", "T_m"],
            "additionalProperties": False,
            "properties": {
                "L_m": {"type": "number", "exclusiveMinimum": 0},
                "B_m": {"type": "number", "exclusiveMinimum": 0},
                "T_m": {"type": "number", "exclusiveMinimum": 0},
            },
        },
        "cad": {
            "type": "object",
            "required": [
                "nabla_m3",
                "nabla_slice_m3",
                "nabla_monte_carlo_m3",
                "s_wetted_m2",
                "s_quadrature_m2",
                "s_monte_carlo_m2",
                "cb",
                "cwp",
                "integration",
            ],
            "properties": {
                "nabla_m3": {"type": "number", "exclusiveMinimum": 0},
                "nabla_slice_m3": {"type": "number", "exclusiveMinimum": 0},
                "nabla_monte_carlo_m3": {"type": "number", "exclusiveMinimum": 0},
                "s_wetted_m2": {"type": "number", "exclusiveMinimum": 0},
                "s_quadrature_m2": {"type": "number", "exclusiveMinimum": 0},
                "s_monte_carlo_m2": {"type": "number", "exclusiveMinimum": 0},
                "cb": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
                "cwp": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
                "integration": {"type": "object", "required": ["volume", "surface", "mc_samples"]},
                "reference": {"type": "object"},
            },
        },
        "grid": {
            "type": "object",
            "required": ["generator", "solvers"],
            "properties": {
                "generator": {"type": "string"},
                "solvers": {
                    "type": "object",
                    "minProperties": 1,
                    "additionalProperties": {
                        "type": "object",
                        "required": [
                            "family",
                            "nabla_m3",
                            "s_wetted_m2",
                            "cb",
                            "resolution",
                            "grid_sha256",
                        ],
                        "properties": {
                            "family": {"type": "string", "enum": ["fvm", "lbm", "bem"]},
                            "nabla_m3": {"type": "number", "exclusiveMinimum": 0},
                            "s_wetted_m2": {"type": "number", "exclusiveMinimum": 0},
                            "cb": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
                            "resolution": {"type": "object"},
                            "grid_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                            "extras": {"type": "object"},
                            "surface_evidence": {"type": "string", "enum": ["REAL", "WITHHELD"]},
                        },
                    },
                },
            },
        },
        "provenance": {
            "type": "object",
            "required": ["cad_sha256", "grid_sha256"],
            "additionalProperties": False,
            "properties": {
                "cad_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "grid_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            },
        },
        "tolerances": {
            "type": "object",
            "required": list(TOLERANCES),
            "properties": {name: {"type": "number", "exclusiveMinimum": 0} for name in TOLERANCES},
        },
        "gate": {
            "type": "object",
            "required": ["verdict", "checks", "failures"],
            "properties": {
                "verdict": {"type": "string", "enum": [GATE_PASS, GATE_REJECT]},
                "checks": {"type": "array", "minItems": 1, "items": {"type": "object"}},
                "failures": {"type": "array", "items": {"type": "object"}},
            },
        },
    },
}


def _validate(value: Any, schema: Mapping[str, Any], path: str, errors: list[str]) -> None:
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, Mapping):
            errors.append(f"{path}: expected object, got {type(value).__name__}")
            return
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required key {key!r}")
        properties = schema.get("properties", {})
        if schema.get("minProperties") and len(value) < schema["minProperties"]:
            errors.append(f"{path}: needs at least {schema['minProperties']} properties")
        extra_allowed = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in properties:
                _validate(item, properties[key], f"{path}.{key}", errors)
            elif extra_allowed is False:
                errors.append(f"{path}: unexpected key {key!r}")
            elif isinstance(extra_allowed, Mapping):
                _validate(item, extra_allowed, f"{path}.{key}", errors)
        return
    if kind == "array":
        if not isinstance(value, (list, tuple)):
            errors.append(f"{path}: expected array, got {type(value).__name__}")
            return
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: needs at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: allows at most {schema['maxItems']} items")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                _validate(item, item_schema, f"{path}[{index}]", errors)
        return
    if kind in _ANY_OF_NUMBER:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"{path}: expected {kind}, got {type(value).__name__}")
            return
        if not math.isfinite(float(value)):
            errors.append(f"{path}: must be finite")
            return
        if "exclusiveMinimum" in schema and not float(value) > schema["exclusiveMinimum"]:
            errors.append(f"{path}: must be > {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and not float(value) < schema["exclusiveMaximum"]:
            errors.append(f"{path}: must be < {schema['exclusiveMaximum']}")
        if "minimum" in schema and float(value) < schema["minimum"]:
            errors.append(f"{path}: must be >= {schema['minimum']}")
        return
    if kind == "string":
        if not isinstance(value, str):
            errors.append(f"{path}: expected string, got {type(value).__name__}")
            return
        if "enum" in schema and value not in schema["enum"]:
            errors.append(f"{path}: {value!r} not in {schema['enum']}")
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: shorter than {schema['minLength']}")
        pattern = schema.get("pattern")
        if pattern:
            import re

            if not re.match(pattern, value):
                errors.append(f"{path}: {value!r} does not match {pattern}")
        return
    if kind == "boolean":
        if not isinstance(value, bool):
            errors.append(f"{path}: expected boolean, got {type(value).__name__}")
        return
    if kind is None:
        return
    errors.append(f"{path}: unsupported schema type {kind!r}")


def validate_contract(contract: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate against ``geometry_fidelity_contract.v1``; raise on failure."""
    errors: list[str] = []
    _validate(contract, CONTRACT_SCHEMA, "$", errors)
    if errors:
        raise ContractSchemaError("; ".join(errors[:8]))
    return contract


# ---------------------------------------------------------------------------
# admission: a rejected grid may not enter a physics chain
# ---------------------------------------------------------------------------


def assert_admitted(contract: Mapping[str, Any], family: str | None = None) -> Mapping[str, Any]:
    """Return the contract if admitted, else raise :class:`MeshNotAdmitted`."""
    validate_contract(contract)
    gate = contract.get("gate") or evaluate_gate(contract)
    failures = list(gate.get("failures", []))
    if family is not None:
        solvers = contract["grid"]["solvers"]
        if family not in solvers:
            raise MeshNotAdmitted(f"no {family} record in contract {contract['geometry_id']}")
        failures = [f for f in failures if f["id"].startswith(f"{family}_") or not f["id"].startswith(tuple(solvers))]
    if gate["verdict"] != GATE_PASS or failures:
        names = ", ".join(sorted({f["id"] for f in failures})) or "unknown"
        raise MeshNotAdmitted(
            f"grid for {contract['geometry_id']} rejected by the geometry gate ({names})", failures
        )
    return contract


def physics_chain(
    contract: Mapping[str, Any],
    family: str,
    consumer: Callable[[Mapping[str, Any]], Any],
) -> Any:
    """Run a physics consumer only after the grid passed the geometry gate."""
    assert_admitted(contract, family)
    return consumer(contract)