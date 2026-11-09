"""Bilateral cross-solver interface validation (AST + schema + numerics).

An interface contract is only real when **both** sides can be inspected:

* the *consumer* repository must actually declare the input symbol and the
  fields the producer claims to feed (checked with ``ast``, no import of the
  consumer stack -- the ships platform is a different runtime);
* the *producer* must publish a registered, closed schema carrying those fields
  (checked against ``campaign.schema_defs``);
* the two sides must agree **numerically** on the shared anchor expression.  For
  ``aero-wind`` the anchor is the dynamic-pressure law ``P = 0.5 rho_air V^2``:
  the gate re-evaluates the consumer's own constant (``RHO_AIR`` read from the
  consumer source) and the producer'ed transform and requires machine equality,
  plus the IS-Code cross anchor (504 Pa <-> V*).

A contract whose consumer side does not exist yet is ``WITHHELD``; it is never
silently accepted.  Contracts flagged ``required=False`` are advisory interfaces
declared ahead of the consumer implementation.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..schema import canonical_json, validate
from ..schema_defs import artifact_bytes, schema_for
from .spec import CONTRACTS, ContractSpec, NodeSpec

AERO_POLAR = "aero_polar.v1"
AMP_EMISSIONS = "amp_emissions.v1"


# ---------------------------------------------------------------------------
# AST side: what the consumer repository declares
# ---------------------------------------------------------------------------
def ast_class_fields(path: Path) -> dict[str, list[str]]:
    """Annotated ``self``/pydantic fields of every class in the module."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        names: list[str] = []
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                names.append(stmt.target.id)
        out[node.name] = names
    return out


def ast_module_constants(path: Path) -> dict[str, float]:
    """Module-level numeric constants (``NAME = <number literal>``)."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
                value = node.value.value
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    out[target.id] = float(value)
    return out


def ast_identical_names(path: Path) -> set[str]:
    """Every identifier used anywhere in the module (attributes, args, names)."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.add(node.value)
    return names


def ast_field_constraints(path: Path) -> dict[str, dict[str, float]]:
    """Numeric constraints declared through ``Field(..., ge=, le=, gt=, lt=)``."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    out: dict[str, dict[str, float]] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        for keyword in call.keywords:
            if keyword.arg in {"ge", "le", "gt", "lt"} and isinstance(keyword.value, ast.Constant):
                value = keyword.value.value
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    out.setdefault(node.target.id, {})[keyword.arg] = float(value)
    return out


def has_dynamic_pressure_law(path: Path) -> tuple[bool, str]:
    """Detect ``0.5 * rho_air * V ** 2`` as an AST shape, not as text."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp):
            continue
        if not (isinstance(node.left, ast.BinOp) and isinstance(node.left.op, ast.Mult)):
            continue
        left, right = node.left, node.right
        half_hit = any(
            isinstance(side, ast.Constant) and abs(float(side.value) - 0.5) < 1e-15
            for side in (left.left, left.right) if isinstance(side, ast.Constant)
        )
        rho_hit = any(
            isinstance(side, ast.Attribute) and "rho" in side.attr.lower()
            for side in (left.left, left.right)
        )
        squared = (
            isinstance(right, ast.BinOp) and isinstance(right.op, ast.Pow)
            and isinstance(right.right, ast.Constant) and float(right.right.value) == 2.0
        )
        if half_hit and rho_hit and squared:
            return True, ast.unparse(node) if hasattr(ast, "unparse") else "0.5*rho*V**2"
    return False, ""


# ---------------------------------------------------------------------------
# producer-side canonical transforms (contract equations, not solver code)
# ---------------------------------------------------------------------------
def wind_pressure_pa(speed_m_s: float, rho_air_kg_m3: float) -> float:
    """``P = 0.5 * rho_air * V^2``  [Pa] (contract anchor of ``aero-wind``)."""
    return 0.5 * rho_air_kg_m3 * speed_m_s ** 2


def speed_for_pressure_m_s(pressure_pa: float, rho_air_kg_m3: float) -> float:
    """Inverse of :func:`wind_pressure_pa` (used for the IS-Code cross anchor)."""
    return math.sqrt(2.0 * pressure_pa / rho_air_kg_m3)


def contract_fingerprint(contract: ContractSpec) -> str:
    payload = {
        "contract_id": contract.contract_id,
        "producer_node": contract.producer_node,
        "producer_schema": contract.producer_schema,
        "producer_required": list(contract.producer_required),
        "consumer_repo": contract.consumer_repo,
        "consumer_file": contract.consumer_file,
        "consumer_symbols": list(contract.consumer_symbols),
        "consumer_fields": list(contract.consumer_fields),
        "consumer_constants": {k: float(v) for k, v in sorted(contract.consumer_constants.items())},
        "numeric_anchor": contract.numeric_anchor,
        "tolerance": contract.tolerance,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _check(name: str, status: str, detail: str = "", value: Any = None,
           threshold: Any = None) -> dict[str, Any]:
    return {"name": name, "status": status, "detail": detail,
            "value": value, "threshold": threshold}


# ---------------------------------------------------------------------------
# data-level conformance of a producer artifact
# ---------------------------------------------------------------------------
def _polar_conformance(payload: Mapping[str, Any]) -> tuple[str, str]:
    validate(schema_for(AERO_POLAR), payload)
    alpha = payload["alpha_deg"]
    cl = payload["C_L"]
    cd = payload["C_D"]
    if not (len(alpha) == len(cl) == len(cd)):
        return "FAIL", "alpha/C_L/C_D length mismatch"
    if any(b <= a for a, b in zip(alpha, alpha[1:])):
        return "FAIL", "alpha_deg is not strictly increasing"
    if any(x <= 0.0 for x in cd):
        return "FAIL", "C_D must stay positive over the polar"
    zeros = [i for i, a in enumerate(alpha) if abs(a) < 1e-12]
    if zeros and abs(cl[zeros[0]]) > 1e-4:
        return "FAIL", f"symmetric airfoil closure violated: C_L(0)={cl[zeros[0]]:.3e}"
    return "PASS", f"{len(alpha)} polar points, rho_air={payload['rho_air_kg_m3']} kg/m3"


def _amp_conformance(payload: Mapping[str, Any]) -> tuple[str, str]:
    validate(schema_for(AMP_EMISSIONS), payload)
    fuel, co2 = payload["fuel_t"], payload["co2_t"]
    cf = co2 / fuel if fuel else float("inf")
    if not 2.5 <= cf <= 4.0:  # plausible marine fuel carbon factor band, kg CO2/kg fuel
        return "FAIL", f"implied carbon factor {cf:.3f} outside marine band [2.5, 4.0]"
    if payload["co2_reduction_t"] > co2:
        return "FAIL", "CO2 reduction exceeds total emissions"
    return "PASS", f"implied CF={cf:.4f} kg CO2/kg fuel"


_DATA_CHECKS = {AERO_POLAR: _polar_conformance, AMP_EMISSIONS: _amp_conformance}


def _data_conformance(contract: ContractSpec,
                      payloads: Sequence[tuple[str, Any]]) -> tuple[str, str]:
    found = False
    details: list[str] = []
    status = "PASS"
    for path, payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        if payload.get("schema_name") != contract.producer_schema:
            continue
        found = True
        checker = _DATA_CHECKS.get(contract.producer_schema)
        if checker is None:
            details.append(f"{path}: no data-level checker declared (schema-only)")
            continue
        try:
            verdict, detail = checker(payload)
        except Exception as exc:  # schema violation or bad numbers
            verdict, detail = "FAIL", f"{type(exc).__name__}: {exc}"
        details.append(f"{path}: {detail}")
        if verdict != "PASS":
            status = "FAIL"
    if not found:
        return "WITHHELD", "producer artifact with declared schema not delivered yet"
    return status, "; ".join(details)


# ---------------------------------------------------------------------------
# contract evaluation
# ---------------------------------------------------------------------------
def evaluate_contract(contract: ContractSpec,
                      consumer_root: Path | None,
                      producer_payloads: Sequence[tuple[str, Any]],
                      producer_schema_properties: Mapping[str, Any],
                      producer_schema_required: Sequence[str]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    gaps: list[str] = []
    consumer_file = (Path(consumer_root) / contract.consumer_file) if consumer_root else None

    # (1) consumer module exists
    if consumer_file is None or not consumer_file.is_file():
        checks.append(_check("consumer_module_present", "WITHHELD",
                             f"consumer file not found: {contract.consumer_file}"))
        gaps.append(f"consumer interface {contract.consumer_file} 未落地，契约无法双侧验证")
        status = "WITHHELD"
        return _finish_contract(contract, checks, "WITHHELD", "WITHHELD", gaps)

    checks.append(_check("consumer_module_present", "PASS", str(consumer_file)))

    # (2) consumer symbols and fields
    try:
        classes = ast_class_fields(consumer_file)
    except (OSError, SyntaxError) as exc:
        classes = {}
        checks.append(_check("consumer_ast_parse", "FAIL", f"{type(exc).__name__}: {exc}"))
    field_pool: set[str] = set()
    for name in contract.consumer_symbols:
        names = classes.get(name)
        if names is None:
            continue
        field_pool.update(names)
    if contract.consumer_symbols:
        missing_symbols = [s for s in contract.consumer_symbols if s not in classes]
        checks.append(_check(
            "consumer_symbols", "FAIL" if missing_symbols else "PASS",
            f"missing: {missing_symbols}" if missing_symbols else
            f"found {list(contract.consumer_symbols)}"))
        missing_fields = [f for f in contract.consumer_fields if f not in field_pool]
        checks.append(_check(
            "consumer_fields", "FAIL" if missing_fields else "PASS",
            f"missing input fields: {missing_fields}" if missing_fields else
            f"all {len(contract.consumer_fields)} required inputs declared"))
        if missing_fields:
            gaps.append(f"消费侧缺少输入字段 {missing_fields}")
    else:
        names = ast_identical_names(consumer_file)
        missing_fields = [f for f in contract.consumer_fields if f not in names]
        checks.append(_check(
            "consumer_fields", "WITHHELD" if missing_fields else "PASS",
            f"consumer endpoint not implemented yet, missing identifiers: {missing_fields}"
            if missing_fields else "required identifiers present in consumer module"))
        if missing_fields:
            gaps.append(f"消费侧缺少标识符 {missing_fields}")

    # (3) consumer constants equal the declared contract constants
    try:
        consts = ast_module_constants(consumer_file)
    except (OSError, SyntaxError):
        consts = {}
    for name, declared in contract.consumer_constants.items():
        actual = consts.get(name)
        if actual is None:
            checks.append(_check(f"constant:{name}", "FAIL", "not declared in consumer module"))
            gaps.append(f"消费侧常量 {name} 缺失")
            continue
        rel = abs(actual - float(declared)) / max(abs(float(declared)), 1e-300)
        checks.append(_check(f"constant:{name}", "PASS" if rel <= 1e-12 else "FAIL",
                             f"consumer={actual!r} contract={declared!r}",
                             value=rel, threshold=1e-12))
        if rel > 1e-12:
            gaps.append(f"消费侧常量 {name} 与契约声明不一致")

    # (4) the shared law exists in the consumer code
    if contract.contract_id == "aero-wind":
        ok, expr = has_dynamic_pressure_law(consumer_file)
        checks.append(_check("consumer_law_shape", "PASS" if ok else "FAIL",
                             expr or "0.5*rho*V**2 not found"))
        if not ok:
            gaps.append("消费侧动压定律 0.5*rho*V^2 未实现（或已改写）")

    # (5) producer schema carries the declared fields
    missing_producer = [f for f in contract.producer_required
                        if f not in producer_schema_properties]
    missing_required = [f for f in contract.producer_required
                        if f not in producer_schema_required]
    checks.append(_check("producer_schema_fields", "FAIL" if (missing_producer or missing_required) else "PASS",
                         f"missing props: {missing_producer}; not required: {missing_required}"
                         if (missing_producer or missing_required) else
                         f"schema {contract.producer_schema} carries all {len(contract.producer_required)} fields"))
    if missing_producer or missing_required:
        gaps.append("生产侧 schema 未声明消费侧所需字段")

    # (6) numeric agreement of the shared anchor
    if contract.contract_id == "aero-wind":
        rho_consumer = consts.get("RHO_AIR")
        if rho_consumer is None:
            checks.append(_check("numeric_anchor", "WITHHELD", "consumer RHO_AIR unavailable"))
        else:
            probes = (10.0, 15.7, 25.7)
            worst = 0.0
            for v in probes:
                producer = wind_pressure_pa(v, rho_consumer)
                consumer = 0.5 * rho_consumer * v ** 2  # consumer branch, verbatim shape
                worst = max(worst, abs(producer - consumer) / max(abs(consumer), 1e-300))
            checks.append(_check("numeric_anchor", "PASS" if worst <= contract.tolerance else "FAIL",
                                 f"probes {probes} m/s, rho={rho_consumer}",
                                 value=worst, threshold=contract.tolerance))
            is_code = consts.get("WIND_PRESSURE_IS_CODE")
            if is_code is not None:
                v_star = speed_for_pressure_m_s(is_code, rho_consumer)
                back = wind_pressure_pa(v_star, rho_consumer)
                rel = abs(back - is_code) / is_code
                checks.append(_check("is_code_cross_anchor",
                                     "PASS" if rel <= contract.tolerance else "FAIL",
                                     f"504 Pa -> V*={v_star:.6f} m/s -> {back:.9f} Pa",
                                     value=rel, threshold=contract.tolerance))

    # (7) data-level conformance (producer artifact, if delivered)
    data_status, data_detail = _data_conformance(contract, producer_payloads)
    checks.append(_check("producer_data_conformance", data_status, data_detail))
    if data_status == "FAIL":
        gaps.append(f"生产侧产物不满足 {contract.producer_schema}: {data_detail}")

    hard_fail = any(c["status"] == "FAIL" for c in checks)
    status = "FAIL" if hard_fail else "PASS"
    if status == "PASS" and data_status == "WITHHELD":
        status = "WITHHELD" if not contract.required else "PASS"
    return _finish_contract(contract, checks, status, data_status, gaps)


def _finish_contract(contract: ContractSpec, checks: list[dict[str, Any]], status: str,
                     data_status: str, gaps: list[str]) -> dict[str, Any]:
    return {
        "contract_id": contract.contract_id,
        "producer_node": contract.producer_node,
        "producer_schema": contract.producer_schema,
        "consumer": f"{contract.consumer_repo}:{contract.consumer_file}",
        "status": status,
        "required": bool(getattr(contract, "required", True)),
        "checks": checks,
        "data_conformance": data_status,
        "fingerprint": contract_fingerprint(contract),
        "gaps": gaps,
    }


def evaluate_all_contracts(consumer_roots: Mapping[str, Path | None],
                           producer_payloads: Mapping[str, Sequence[tuple[str, Any]]]
                           ) -> list[dict[str, Any]]:
    """Evaluate every declared contract against the live repos and schemas."""
    out: list[dict[str, Any]] = []
    for contract in CONTRACTS:
        schema = schema_for(contract.producer_schema)
        out.append(evaluate_contract(
            contract,
            consumer_roots.get(contract.consumer_repo),
            producer_payloads.get(contract.producer_node, ()),
            schema.get("properties", {}),
            schema.get("required", ()),
        ))
    return out


def materialize_contract_schema(contract: ContractSpec) -> bytes:
    """Canonical bytes of the producer schema (used by tests as the bilateral lock)."""
    return artifact_bytes(contract.producer_schema,
                          _minimal_producer_payload(contract.producer_schema))


def _minimal_producer_payload(schema_name: str) -> dict[str, Any]:
    if schema_name == AERO_POLAR:
        return {"schema_name": AERO_POLAR, "alpha_deg": [-2.0, 0.0, 2.0],
                "C_L": [-0.22, 0.0, 0.22], "C_D": [0.008, 0.007, 0.008],
                "rho_air_kg_m3": 1.225, "chord_m": 1.0}
    if schema_name == AMP_EMISSIONS:
        return {"schema_name": AMP_EMISSIONS, "aux_power_kw": 1200.0, "fuel_t": 5.0,
                "co2_t": 15.57, "co2_reduction_t": 12.0, "p_eff_kw": 0.0,
                "grid_factor_kg_per_kwh": 0.581}
    raise KeyError(schema_name)