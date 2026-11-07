#!/usr/bin/env python3
"""Whole-ledger criteria-falsifiability audit (mutant kill-rate KPI + merge gate).

Why this exists
---------------
Every TensorPhysics acceptance gate is a *judge*: it says PASS / FAIL /
WITHHELD about a delivered artifact.  Green tests prove the gate runs; they do
not prove the gate would notice a fabricated number.  ``tests/
test_contract_mutation.py`` asked that question for a single gate; this tool asks
it for the *whole ledger* and turns the answer into a KPI plus a merge gate:

    for every criterion of every audited gate, feed it mutant cases -- corrupted
    instances of the very evidence the criterion is supposed to guard -- and
    count how many mutants the criterion rejects.

A criterion that rejects none of its mutants is **unfalsifiable** and is flagged
no matter how green its tests are.

Criterion ledger
----------------
``criterion = {id, gate, callable, expected_verdict, evidence_ref}`` plus
declarative guards naming the failure modes it claims to catch::

    Criterion(id="P20H-H6", gate="p20_hydrostatic",
              callable_ref="tensorphysics.campaign.p20_gate.contracts"
                           ":validate_hydrostatic_handoff",
              expected_verdict="PASS",
              evidence_ref="src/.../contracts.py#H6_displacement_closure",
              case={...clean payload...}, check="H6_displacement_closure",
              value_fields=("displacement_m3",),
              unit_fields=((("displacement_m3"), "volume"),),
              baseline={...})

The shipped gate functions are imported and called -- never re-implemented,
never rewritten.  The adapter reads the named sub-check out of the verdict object
the gate returns.  External judges (e.g. the platform PAPER-TRACE mutation audit)
are consumed through their own recorded per-mutant records (``--external``);
their mutants are not re-derived here.

Mutant classes (data level, no source rewriting)
-----------------------------------------------
M1  numeric perturbation : x -> x*(1+eps), eps in {1 %, 3 %, 15 %}

M2  unit/dimension shift : x -> x * UNIT_FACTOR[dim] (m->mm, m3->litre, rad/s->Hz)

M3  constant stand-in    : a guarded constant replaced by a plausible stand-in

M4  sample camouflage    : sample metadata (N, grid labels h, entry counts,
                           declared keys) changed while result values stay

M5  baseline replacement : the reference replaced by the judged value or by an
                           identity expression (self-referential -> tautology)

KPI / gate
----------
``gate_only``  kill rate of the shipped gate function alone;
``ledger``     kill rate of the gate **plus** the declared provenance guard, so a
               gate-level blind spot stays visible instead of being papered
               over;
plus ``weakest_mutants`` (every survivor), ``unfalsifiable`` (zero-kill
criteria), the M1 eps-ladder monotonicity, a five-criterion tautological
negative control (must score 0/5), and ``--pr-check``: a criterion that is new
relative to the baseline ledger must carry >= 1 killed mutant, else exit 2.

Artifacts::

    artifacts/criteria_mutation/ledger.json / ledger.md
    artifacts/criteria_mutation/fig1_gate_killrate.png
    artifacts/criteria_mutation/fig2_epsilon_ladder.png
    artifacts/criteria_mutation/fig3_class_matrix.png

Exit codes: 0 = audit ran (the kill rate is data, not a verdict), 2 = PR gate
rejected a new unfalsifiable criterion.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:  # usable from a bare checkout
    sys.path.insert(0, str(ROOT / "src"))

ARTIFACT_DIR = Path("artifacts/criteria_mutation")
KPI_TARGET = 0.90                  # whole ledger and every audited gate
EPS_LADDER = (0.01, 0.03, 0.15)    # M1 ladder fixed by the campaign contract
PHYS_ACCURACY_BAND = 0.15          # platform rule: >15 % is not acceptable
MUTANT_CLASSES = ("M1", "M2", "M3", "M4", "M5")
PASS, FAIL, WITHHELD = "PASS", "FAIL", "WITHHELD"

FORBIDDEN_TOKENS = (
    "hull", "ship", "vessel", "draught", "draft", "seakeeping", "hydrostat",
    "resistance", "propeller", "rudder", "slamming", "waterline", "deadrise",
    "suboff", "kvlcc2", "kcs", "bulbous", "displacement", "ice_",
)

# BEGIN-DOMAIN-FREE-ENGINE
# Domain-agnostic mutant engine: numbers, unit factors, sample metadata and
# provenance strings only.  scan_engine_lexicon() proves the absence of domain
# vocabulary in this region.
UNIT_FACTOR: dict[str, tuple[float, str]] = {
    # dimension -> (factor applied to the reported number, unit it pretends to be)
    "length": (1.0e3, "mm"),
    "mass": (1.0e3, "g"),
    "volume": (1.0e3, "litre"),
    "area": (1.0e4, "cm2"),
    "time": (1.0 / 60.0, "min"),
    "frequency": (1.0 / (2.0 * math.pi), "Hz"),
    "density": (1.0e-3, "g_per_cm3"),
    "force": (1.0e3, "mN"),
    "velocity": (1.0e3, "mm_per_s"),
    "moment": (1.0e3, "mN_m"),
    "dimensionless": (1.0, "-"),
    "fraction": (100.0, "percent"),
}

ENGINE_BEGIN = "# BEGIN-DOMAIN-FREE-ENGINE"
ENGINE_END = "# END-DOMAIN-FREE-ENGINE"


class MutationError(ValueError):
    """A criterion declaration or a mutant request is inconsistent."""


def json_get(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if isinstance(cur, Mapping):
            if part not in cur:
                raise MutationError(f"path {path!r}: missing key {part!r}")
            cur = cur[part]
        elif isinstance(cur, Sequence) and not isinstance(cur, (str, bytes)):
            cur = cur[int(part)]
        else:
            raise MutationError(f"path {path!r}: cannot descend into {type(cur).__name__}")
    return cur


def json_set(obj: Any, path: str, value: Any) -> Any:
    out = copy.deepcopy(obj)
    parts = path.split(".")
    cur = out
    for part in parts[:-1]:
        cur = cur[part] if isinstance(cur, Mapping) else cur[int(part)]
    if isinstance(cur, Mapping):
        cur[parts[-1]] = value
    else:
        cur[int(parts[-1])] = value
    return out


def _scaled(value: Any, factor: float) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MutationError(f"cannot scale non-numeric value {value!r}")
    return float(value) * factor


def mutant_scale(case: Any, path: str, factor: float) -> Any:
    """M1/M2: multiply one declared number by ``factor``."""
    return json_set(case, path, _scaled(json_get(case, path), factor))


def mutant_replace(case: Any, path: str, stand_in: Any) -> Any:
    """M3: substitute a plausible constant stand-in at ``path``."""
    return json_set(case, path, stand_in)


def mutant_drop_last(case: Any, path: str) -> Any:
    """M4: drop the last sample of a declared list, keep every other value."""
    items = json_get(case, path)
    if not isinstance(items, list) or len(items) < 2:
        raise MutationError(f"{path}: need a list with >= 2 entries")
    return json_set(case, path, list(items)[:-1])


def mutant_drop_key(case: Any, path: str) -> Any:
    """M4: drop one declared key from a mapping (declared-set camouflage)."""
    parent, _, leaf = path.rpartition(".")
    if not leaf:
        raise MutationError(f"{path!r}: expected 'container.key'")
    holder = json_get(case, parent) if parent else case
    if not isinstance(holder, Mapping) or leaf not in holder:
        raise MutationError(f"{path!r}: key not present")
    out = copy.deepcopy(case)
    target = json_get(out, parent) if parent else out
    target.pop(leaf, None)
    return out


def mutant_duplicate(case: Any, path: str) -> Any:
    """M4: duplicate the first sample (count inflated, result values untouched)."""
    items = json_get(case, path)
    if not isinstance(items, list):
        raise MutationError(f"{path}: not a list")
    return json_set(case, path, list(items) + [copy.deepcopy(items[0])])


def mutant_permute(case: Any, path: str) -> Any:
    """M4: relabel the two coarsest samples; the observed values stay frozen."""
    items = json_get(case, path)
    if not isinstance(items, list) or len(items) < 3:
        raise MutationError(f"{path}: need >= 3 entries to permute labels")
    out = list(items)
    out[0], out[1] = out[1], out[0]
    return json_set(case, path, out)


def mutant_identity_baseline(case: Any, spec: Mapping[str, Any]) -> tuple[Any, str]:
    """M5: replace the reference by the judged value or an identity expression."""
    ref_field = spec["ref_field"]
    value_field = spec.get("value_field")
    if "copy_from" in spec:
        source = json_get(case, spec["copy_from"])
        tag = f"identity:{spec['copy_from']}"
    elif "product_of" in spec:
        product = 1.0
        for path in spec["product_of"]:
            product *= float(json_get(case, path))
        source = product
        tag = "identity:" + "*".join(spec["product_of"])
    else:  # pragma: no cover - declaration error
        raise MutationError("baseline spec needs 'copy_from' or 'product_of'")
    out = json_set(case, ref_field, tag)
    if value_field:
        out = json_set(out, value_field, source)
    return out, tag


def scan_engine_lexicon(source: str) -> list[str]:
    """Domain tokens found inside the domain-free engine region (must be empty)."""
    start = source.index(ENGINE_BEGIN)
    end = source.index(ENGINE_END)
    region = source[start:end].lower()
    return sorted({tok for tok in FORBIDDEN_TOKENS if tok in region})


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
# END-DOMAIN-FREE-ENGINE


# ---------------------------------------------------------------------------
# ledger model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Criterion:
    """One acceptance criterion plus the failure modes it claims to catch."""

    id: str
    gate: str
    callable_ref: str
    expected_verdict: str
    evidence_ref: str
    case: Any
    adapter: str = "payload"
    check: str | None = None
    accept: tuple[str, ...] = (PASS,)
    tolerance: float = 0.0
    value_fields: tuple[str, ...] = ()
    unit_fields: tuple[tuple[str, str], ...] = ()
    constants: tuple[tuple[str, Any], ...] = ()
    sample_ops: tuple[tuple[str, str], ...] = ()
    baseline: Mapping[str, Any] | None = None
    consistency: tuple[tuple[str, str, float], ...] = ()
    mutant_classes: tuple[str, ...] = ()
    eps_ladder: tuple[float, ...] = EPS_LADDER
    note: str = ""

    @property
    def callable(self) -> Callable[..., Any]:
        module_name, _, attr = self.callable_ref.partition(":")
        if not attr:
            raise MutationError(f"{self.id}: callable_ref must be 'module:function'")
        return getattr(importlib.import_module(module_name), attr)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "gate": self.gate, "callable": self.callable_ref,
            "expected_verdict": self.expected_verdict,
            "evidence_ref": self.evidence_ref, "check": self.check,
            "mutant_classes": list(self.mutant_classes),
            "tolerance": self.tolerance, "note": self.note,
        }


@dataclass(frozen=True)
class Mutant:
    criterion_id: str
    gate: str
    klass: str
    label: str
    case: Any
    epsilon: float | None = None
    in_band: bool = False
    detail: str = ""


@dataclass
class EvalResult:
    status: str
    gate_status: str
    detail: str

    @property
    def accepted(self) -> bool:
        return self.status == PASS


# ---------------------------------------------------------------------------
# adapters: declarative case -> shipped gate call (never a re-implementation)
# ---------------------------------------------------------------------------
def _call_payload(criterion: Criterion, case: Any) -> Any:
    return criterion.callable(copy.deepcopy(case))


def _call_gci(criterion: Criterion, case: Any) -> Any:
    from tensorphysics.vv.gci import Observation

    levels = [Observation(h=row["h"], f=row["f"], label=row.get("label", ""))
              for row in case["levels"]]
    kwargs = dict(case.get("kwargs") or {})
    return criterion.callable(levels, quantity=case.get("quantity", "observable"), **kwargs)


def _call_validation(criterion: Criterion, case: Any) -> Any:
    return criterion.callable(case["f_sim"], case["f_exp"],
                              u_num=float(case.get("u_num", 0.0)),
                              u_input=float(case.get("u_input", 0.0)),
                              label=case.get("label", ""))


ADAPTERS: dict[str, Callable[[Criterion, Any], Any]] = {
    "payload": _call_payload, "gci": _call_gci, "validation": _call_validation,
}


def check_id_of(item: Any) -> str | None:
    """Sub-check name of a shipped gate verdict entry (both shipped styles)."""
    for attr in ("check_id", "name", "check"):
        value = getattr(item, attr, None)
        if isinstance(value, str):
            return value
    if isinstance(item, Mapping):
        for key in ("check_id", "check", "name"):
            value = item.get(key)
            if isinstance(value, str):
                return value
    return None


def status_from_result(result: Any, check: str | None) -> str:
    """Normalise a shipped gate verdict object to PASS / FAIL / WITHHELD."""
    checks = getattr(result, "checks", None)
    if checks is not None and hasattr(result, "passed"):      # ContractResult
        if check is None:
            return PASS if result.passed else FAIL
        for item in checks:
            if check_id_of(item) == check:
                return PASS if getattr(item, "ok", False) else FAIL
        return WITHHELD
    if isinstance(checks, (list, tuple)) and checks and isinstance(checks[0], Mapping):
        if check is None:
            return getattr(result, "status", WITHHELD)
        for item in checks:                                    # V&V gate verdict
            if check_id_of(item) == check:
                return item.get("status", WITHHELD)
        return WITHHELD
    if isinstance(getattr(result, "status", None), str):
        return result.status
    if hasattr(result, "passed"):
        return PASS if result.passed else FAIL
    raise MutationError(f"cannot read a verdict out of {type(result).__name__}")


def evaluate(criterion: Criterion, case: Any) -> EvalResult:
    """Run the shipped gate on ``case``; return gate status and ledger status."""
    builder = ADAPTERS.get(criterion.adapter)
    if builder is None:
        raise MutationError(f"{criterion.id}: unknown adapter {criterion.adapter!r}")
    try:
        result = builder(criterion, case)
    except Exception as exc:   # a gate that raises *is* a rejection; record why
        return EvalResult(FAIL, FAIL, f"gate raised {type(exc).__name__}: {exc}")
    gate_status = status_from_result(result, criterion.check)
    if gate_status not in criterion.accept:
        return EvalResult(gate_status, gate_status,
                          f"gate verdict {gate_status} (check={criterion.check})")
    for field, record_field, rtol in criterion.consistency:
        declared = json_safe_get(case, field)
        audited = json_safe_get(case, record_field)
        if not isinstance(declared, (int, float)) or not isinstance(audited, (int, float)):
            return EvalResult(FAIL, gate_status,
                              f"consistency: {field} / {record_field} missing or non-numeric")
        if abs(float(declared) - float(audited)) > rtol * max(abs(float(audited)), 1e-300):
            return EvalResult(FAIL, gate_status,
                              f"consistency: declared {field}={declared} disagrees with audited "
                              f"{record_field}={audited} (rtol={rtol:g})")
    spec = criterion.baseline
    if spec:
        ref = json_safe_get(case, spec["ref_field"])
        forbidden = tuple(spec.get("forbid", ("identity:", "self:")))
        if not isinstance(ref, str) or not ref.strip():
            return EvalResult(FAIL, gate_status,
                              f"provenance: reference {spec['ref_field']} missing/empty")
        if ref.startswith(forbidden):
            return EvalResult(FAIL, gate_status,
                              f"provenance: reference {ref!r} is self-referential "
                              f"(forbidden prefixes {forbidden})")
    return EvalResult(PASS, gate_status, f"gate verdict {gate_status} accepted")


def json_safe_get(obj: Any, path: str) -> Any:
    try:
        return json_get(obj, path)
    except MutationError:
        return None


def generate_mutants(criterion: Criterion) -> list[Mutant]:
    """Enumerate the failure modes a criterion declares it can catch."""
    out: list[Mutant] = []
    declared = set(criterion.mutant_classes)
    for path in criterion.value_fields if "M1" in declared else ():
        for eps in criterion.eps_ladder:
            out.append(Mutant(
                criterion.id, criterion.gate, "M1", f"M1 {path} x{1.0 + eps:.2f}",
                mutant_scale(criterion.case, path, 1.0 + eps),
                epsilon=eps, in_band=eps <= criterion.tolerance,
                detail=f"numeric perturbation eps={eps:.2%} on {path}"))
    for path, dimension in criterion.unit_fields if "M2" in declared else ():
        factor, unit = UNIT_FACTOR[dimension]
        out.append(Mutant(
            criterion.id, criterion.gate, "M2", f"M2 {path} -> {unit}",
            mutant_scale(criterion.case, path, factor),
            detail=f"unit shift {dimension}->{unit} (x{factor:g}) on {path}"))
    for path, stand_in in criterion.constants if "M3" in declared else ():
        out.append(Mutant(
            criterion.id, criterion.gate, "M3", f"M3 {path} := {stand_in!r}",
            mutant_replace(criterion.case, path, stand_in),
            detail=f"constant stand-in at {path}"))
    ops = {"drop_last": mutant_drop_last, "drop_key": mutant_drop_key,
           "duplicate": mutant_duplicate, "permute": mutant_permute,
           "relabel": lambda case, path: mutant_scale(case, path, 2.0)}
    for op, path in criterion.sample_ops if "M4" in declared else ():
        if op not in ops:   # pragma: no cover - declaration error
            raise MutationError(f"{criterion.id}: unknown sample op {op!r}")
        out.append(Mutant(
            criterion.id, criterion.gate, "M4", f"M4 {op} {path}",
            ops[op](criterion.case, path),
            detail=f"sample camouflage ({op}) on {path}, result values untouched"))
    if criterion.baseline and "M5" in declared:
        case, tag = mutant_identity_baseline(criterion.case, criterion.baseline)
        out.append(Mutant(
            criterion.id, criterion.gate, "M5",
            f"M5 {criterion.baseline['ref_field']} := {tag}", case,
            detail="baseline replaced by the judged value (tautology)"))
    return out# ---------------------------------------------------------------------------
# tautological negative control (must score 0/5 and be flagged)
# ---------------------------------------------------------------------------
@dataclass
class AlwaysAccept:
    """Deliberately unfalsifiable judge: the ``assert True`` criterion class."""

    status: str = PASS
    checks: tuple = ()

    @property
    def passed(self) -> bool:
        return True


def tautology_factory(*args: Any, **kwargs: Any) -> AlwaysAccept:
    return AlwaysAccept()


def negative_control_criteria() -> list[Criterion]:
    """Five tautological criteria; one M1 mutant each -> expected 0/5 kills."""
    return [Criterion(
        id=f"NC-TAUT-{i + 1:02d}", gate="negative_control",
        callable_ref="criteria_mutation:tautology_factory",
        expected_verdict=PASS,
        evidence_ref="tools/audit/criteria_mutation.py#negative_control",
        case={"measured": 1.0 + 0.1 * i, "declared": 1.0},
        value_fields=("measured",), mutant_classes=("M1",), eps_ladder=(0.15,),
        note="tautological judge: PASS for every input, by construction")
        for i in range(5)]


def _install_self_module() -> None:
    module = sys.modules.get(__name__)
    if module is not None:
        sys.modules.setdefault("criteria_mutation", module)


# ---------------------------------------------------------------------------
# TensorPhysics gate adapters (clean cases + declared guards)
# ---------------------------------------------------------------------------
def _hydro_case() -> dict[str, Any]:
    lpp, b, t, cb = 230.0, 32.2, 10.0, 0.6512
    return {
        "schema": "cad_hydrostatic_handoff.v1",
        "geometry_id": "kcs_hull_v3",
        "principal_dimensions": {"Lpp_m": lpp, "B_m": b, "T_m": t,
                                 "Cb": cb, "Cm": 0.98, "Cp": 0.665},
        "displacement_m3": cb * lpp * b * t,
        "wetted_area_m2": 8900.0,
        "displacement_ref": "artifacts/p20/hydrostatics.json#displacement_m3",
        "sections": [{"x_m": 0.0, "area_m2": 0.0},
                     {"x_m": 115.0, "area_m2": 210.0},
                     {"x_m": 230.0, "area_m2": 0.0}],
    }


def _seakeeping_case() -> dict[str, Any]:
    freqs = [0.5, 1.0, 1.5, 2.0]
    return {
        "schema": "bem_seakeeping_handoff.v1",
        "frame": {"x": "surge", "y": "sway", "z": "heave",
                  "rx": "roll", "ry": "pitch", "rz": "yaw"},
        "frequency_rad_s": list(freqs),
        "wave_heading_deg": 180.0,
        "force_amplitude": [
            {"freq_rad_s": f, "Fx_N": 1.0e5 * (i + 1), "Fy_N": 0.0,
             "Fz_N": 2.0e5 * (i + 1), "Mx_Nm": 0.0,
             "My_Nm": 5.0e6 * (i + 1), "Mz_Nm": 0.0}
            for i, f in enumerate(freqs)],
        "encounter_convention": "positive_head_seas",
        "rao_ref": "artifacts/p20/seakeeping/rao_bem.json#heave",
    }


def _gci_case() -> dict[str, Any]:
    # f(h) = 1.20 + 2 h^2 on h = 0.08 / 0.04 / 0.02  ->  observed order p = 2
    return {"quantity": "demo_observable",
            "levels": [{"h": 0.08, "f": 1.2128, "label": "coarse"},
                       {"h": 0.04, "f": 1.2032, "label": "medium"},
                       {"h": 0.02, "f": 1.2008, "label": "fine"}]}


def _validation_case() -> dict[str, Any]:
    return {"f_sim": 1.008, "f_exp": 1.000, "u_num": 0.02, "u_input": 0.0,
            "u_num_audited": 0.02,
            "baseline_ref": "artifacts/vv/exp_benchmark.json#ct", "label": "demo"}


HYDRO_ARGS = dict(
    adapter="payload",
    callable_ref="tensorphysics.campaign.p20_gate.contracts:validate_hydrostatic_handoff",
    evidence_ref="src/tensorphysics/campaign/p20_gate/contracts.py:"
                 "validate_hydrostatic_handoff")

SEA_ARGS = dict(
    adapter="payload",
    callable_ref="tensorphysics.campaign.p20_gate.contracts:validate_seakeeping_handoff",
    evidence_ref="src/tensorphysics/campaign/p20_gate/contracts.py:"
                 "validate_seakeeping_handoff")

GCI_ARGS = dict(adapter="gci", callable_ref="tensorphysics.vv.gci:evaluate",
                evidence_ref="src/tensorphysics/vv/gci.py:evaluate")

VAL_ARGS = dict(adapter="validation", callable_ref="tensorphysics.vv.validation:validate",
                evidence_ref="src/tensorphysics/vv/validation.py:validate")


def criteria_p20_hydrostatic() -> list[Criterion]:
    case = _hydro_case()
    return [
        Criterion(id="P20H-H1", gate="p20_hydrostatic", check="H1_schema",
                  expected_verdict=PASS, case=case,
                  constants=(("schema", "cad_hydrostatic_handoff.v0"),),
                  mutant_classes=("M3",),
                  note="schema tag must be the agreed version", **HYDRO_ARGS),
        Criterion(id="P20H-H2", gate="p20_hydrostatic", check="H2_geometry_id",
                  expected_verdict=PASS, case=case,
                  constants=(("geometry_id", ""),), mutant_classes=("M3",),
                  note="geometry_id must be a real identifier, not a placeholder",
                  **HYDRO_ARGS),
        Criterion(id="P20H-H3", gate="p20_hydrostatic", check="H3_principal_dimensions",
                  expected_verdict=PASS, case=case,
                  sample_ops=(("drop_key", "principal_dimensions.Cb"),),
                  mutant_classes=("M4",),
                  note="every principal dimension must be declared", **HYDRO_ARGS),
        Criterion(id="P20H-H4", gate="p20_hydrostatic", check="H4_dimensional_order",
                  expected_verdict=PASS, case=case,
                  unit_fields=(("principal_dimensions.T_m", "length"),),
                  constants=(("principal_dimensions.T_m", 35.0),),
                  mutant_classes=("M2", "M3"),
                  note="Lpp > B > T > 0 must survive any unit convention", **HYDRO_ARGS),
        Criterion(id="P20H-H5", gate="p20_hydrostatic", check="H5_Cb_range",
                  expected_verdict=PASS, case=case,
                  constants=(("principal_dimensions.Cb", 0.95),),
                  mutant_classes=("M3",),
                  note="block coefficient inside the physical band", **HYDRO_ARGS),
        Criterion(id="P20H-H6", gate="p20_hydrostatic", check="H6_displacement_closure",
                  expected_verdict=PASS, case=case,
                  value_fields=("displacement_m3",),
                  unit_fields=(("displacement_m3", "volume"),),
                  mutant_classes=("M1", "M2", "M5"), tolerance=1e-12,
                  baseline=dict(ref_field="displacement_ref",
                                value_field="displacement_m3",
                                product_of=("principal_dimensions.Cb",
                                            "principal_dimensions.Lpp_m",
                                            "principal_dimensions.B_m",
                                            "principal_dimensions.T_m"),
                                forbid=("identity:", "self:")),
                  note="closure against an independently measured volume", **HYDRO_ARGS),
    ]


def criteria_p20_seakeeping() -> list[Criterion]:
    case = _seakeeping_case()
    return [
        Criterion(id="P20S-S1", gate="p20_seakeeping", check="S1_schema",
                  expected_verdict=PASS, case=case,
                  constants=(("schema", "bem_seakeeping_handoff.v0"),),
                  mutant_classes=("M3",),
                  note="schema tag must be the agreed version", **SEA_ARGS),
        Criterion(id="P20S-S2", gate="p20_seakeeping", check="S2_frame",
                  expected_verdict=PASS, case=case,
                  constants=(("frame.x", "streamwise"),), mutant_classes=("M3",),
                  note="frame convention is the agreed one, not a foreign vocabulary",
                  **SEA_ARGS),
        Criterion(id="P20S-S3", gate="p20_seakeeping", check="S3_frequency",
                  expected_verdict=PASS, case=case,
                  constants=(("frequency_rad_s", [1.0, 1.0, 1.0, 1.0]),),
                  mutant_classes=("M3",),
                  note="frequency array sorted and strictly increasing", **SEA_ARGS),
        Criterion(id="P20S-S4", gate="p20_seakeeping", check="S4_force_amplitude",
                  expected_verdict=PASS, case=case,
                  sample_ops=(("drop_last", "force_amplitude"),),
                  unit_fields=(("force_amplitude.0.freq_rad_s", "frequency"),),
                  mutant_classes=("M2", "M4"),
                  note="one amplitude record per frequency, same unit convention",
                  **SEA_ARGS),
        Criterion(id="P20S-S5", gate="p20_seakeeping", check="S5_encounter_convention",
                  expected_verdict=PASS, case=case,
                  constants=(("encounter_convention", ""),), mutant_classes=("M3",),
                  note="encounter convention declared", **SEA_ARGS),
    ]


def criteria_vv_gci() -> list[Criterion]:
    case = _gci_case()
    return [
        Criterion(id="VVGCI-input", gate="vv_gci", check="input",
                  expected_verdict=PASS, case=case,
                  sample_ops=(("duplicate", "levels"),), mutant_classes=("M4",),
                  note="levels strictly refined in h", **GCI_ARGS),
        Criterion(id="VVGCI-convergence", gate="vv_gci",
                  check="asymptotic-range/convergence-type", expected_verdict=PASS,
                  case=case, value_fields=("levels.2.f",), mutant_classes=("M1",),
                  note="monotone contraction on the two finest levels", **GCI_ARGS),
        Criterion(id="VVGCI-order", gate="vv_gci", check="order-band",
                  expected_verdict=PASS, case=case,
                  sample_ops=(("relabel", "levels.2.h"),), mutant_classes=("M4",),
                  note="observed order inside the admissible band", **GCI_ARGS),
        Criterion(id="VVGCI-residual", gate="vv_gci", check="power-law-residual",
                  expected_verdict=PASS, case=case,
                  value_fields=("levels.1.f",), mutant_classes=("M1",), tolerance=0.01,
                  note="power-law fit residual stays small", **GCI_ARGS),
        Criterion(id="VVGCI-gci-monotone", gate="vv_gci", check="gci-monotonicity",
                  expected_verdict=PASS, case=case,
                  value_fields=("levels.1.f",), mutant_classes=("M1",), tolerance=0.03,
                  note="GCI must shrink towards the fine level", **GCI_ARGS),
    ]


def criteria_vv_validation() -> list[Criterion]:
    return [Criterion(
        id="VVVAL-ratio", gate="vv_validation", expected_verdict=PASS,
        case=_validation_case(), value_fields=("f_sim",),
        unit_fields=(("f_exp", "fraction"),),
        constants=(("u_num", 0.2),),
        mutant_classes=("M1", "M3", "M5"), tolerance=0.02,
        consistency=(("u_num", "u_num_audited", 1e-9),),
        baseline=dict(ref_field="baseline_ref", copy_from="f_sim",
                      forbid=("identity:", "self:")),
        note="|f_sim - f_exp| <= U_val against an independent reference", **VAL_ARGS)]


GATE_ADAPTERS: dict[str, Callable[[], list[Criterion]]] = {
    "p20_hydrostatic": criteria_p20_hydrostatic,
    "p20_seakeeping": criteria_p20_seakeeping,
    "vv_gci": criteria_vv_gci,
    "vv_validation": criteria_vv_validation,
}


def discover_gate_checks(gate: str) -> list[str]:
    """Check ids the shipped gate itself can emit, across the gate's cases."""
    seen: list[str] = []
    for criterion in GATE_ADAPTERS[gate]():
        result = ADAPTERS[criterion.adapter](criterion, criterion.case)
        for item in getattr(result, "checks", None) or []:
            name = (getattr(item, "name", None) or
                    (item.get("check") if isinstance(item, Mapping) else None))
            if name and name not in seen:
                seen.append(name)
    return seen


# ---------------------------------------------------------------------------
# external gate: consume recorded per-mutant records (never re-derived)
# ---------------------------------------------------------------------------
def import_external_gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute an external gate's kill rate from its own mutant records."""
    runs = payload.get("runs") or {}
    out: dict[str, Any] = {"gate": payload.get("gate", "external"), "runs": {}}
    for name, run in runs.items():
        records = run.get("mutants") or []
        killed = sum(1 for m in records if m.get("killed"))
        total = len(records)
        declared_killed = run.get("killed")
        out["runs"][name] = {
            "tests": run.get("tests") or [],
            "total": total, "killed": killed, "survived": total - killed,
            "kill_rate": (killed / total) if total else None,
            "records_complete": total > 0,
            "recomputed_matches_declared": (declared_killed is None
                                            or declared_killed == killed),
            "declared_killed": declared_killed,
            "baseline_green": run.get("baseline_green"),
        }
    operative = payload.get("operative_run")
    if operative and operative in out["runs"]:
        run = out["runs"][operative]
        out.update({"operative_run": operative, "total": run["total"],
                    "killed": run["killed"], "survived": run["survived"],
                    "kill_rate": run["kill_rate"]})
    return out


def load_external(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"external audit record not found: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    out = import_external_gate(payload)
    out["source"] = str(p)
    out["source_sha256"] = file_sha256(p)
    out["provenance"] = payload.get("provenance")
    return out


# ---------------------------------------------------------------------------
# run the ledger
# ---------------------------------------------------------------------------
def run_ledger(*, external: Mapping[str, Any] | None = None,
               gates: Sequence[str] | None = None,
               include_negative_control: bool = True) -> dict[str, Any]:
    """Evaluate every criterion, every mutant, and aggregate the KPI."""
    _install_self_module()
    started = time.perf_counter()
    selected = {g: GATE_ADAPTERS[g] for g in (gates or GATE_ADAPTERS)}
    criteria: list[Criterion] = []
    for gate, builder in selected.items():
        criteria.extend(builder())
    if include_negative_control and not gates:
        criteria.extend(negative_control_criteria())

    records: list[dict[str, Any]] = []
    per_criterion: list[dict[str, Any]] = []
    for criterion in criteria:
        t0 = time.perf_counter()
        clean = evaluate(criterion, criterion.case)
        instantiated = clean.gate_status == criterion.expected_verdict
        mutants = generate_mutants(criterion)
        killed = killed_gate_only = in_band = 0
        survivors: list[dict[str, Any]] = []
        for mutant in mutants:
            result = evaluate(criterion, mutant.case)
            is_kill = not result.accepted
            gate_kill = result.gate_status not in criterion.accept
            killed += int(is_kill)
            killed_gate_only += int(gate_kill)
            in_band += int(mutant.in_band)
            records.append({
                "criterion_id": criterion.id, "gate": criterion.gate,
                "klass": mutant.klass, "label": mutant.label,
                "epsilon": mutant.epsilon, "in_band": mutant.in_band,
                "killed": is_kill, "killed_gate_only": gate_kill,
                "verdict": result.status, "gate_verdict": result.gate_status,
                "detail": result.detail,
            })
            if not is_kill:
                survivors.append({"label": mutant.label, "klass": mutant.klass,
                                  "in_band": mutant.in_band,
                                  "gate_verdict": result.gate_status,
                                  "detail": result.detail})
        in_scope = len(mutants)
        equivalent = [r for r in survivors if r["in_band"]]
        per_criterion.append({
            **criterion.as_dict(),
            "instantiated": instantiated,
            "clean_status": clean.status,
            "clean_gate_status": clean.gate_status,
            "n_mutants": len(mutants), "n_in_band": in_band,
            "n_in_scope": in_scope, "n_killed": killed,
            "n_killed_gate_only": killed_gate_only,
            "kill_rate": (killed / in_scope) if in_scope else None,
            "kill_rate_gate_only": (killed_gate_only / in_scope) if in_scope else None,
            "unfalsifiable": bool(in_scope > 0 and killed == 0),
            "n_survivors_equivalent": len(equivalent),
            "n_survivors_blind_spot": len(survivors) - len(equivalent),
            "no_mutant_coverage": len(mutants) == 0,
            "survivors": survivors,
            "elapsed_s": round(time.perf_counter() - t0, 4),
        })

    gate_rows: list[dict[str, Any]] = []
    for gate in dict.fromkeys(c["gate"] for c in per_criterion):
        rows = [c for c in per_criterion if c["gate"] == gate]
        in_scope = sum(r["n_in_scope"] for r in rows)
        killed = sum(r["n_killed"] for r in rows)
        killed_gate = sum(r["n_killed_gate_only"] for r in rows)
        raw_total = sum(r["n_mutants"] for r in rows)
        gate_rows.append({
            "gate": gate,
            "n_criteria": len(rows),
            "n_criteria_instantiated": sum(1 for r in rows if r["instantiated"]),
            "n_criteria_with_mutants": sum(1 for r in rows if r["n_mutants"]),
            "n_criteria_unfalsifiable": sum(1 for r in rows if r["unfalsifiable"]),
            "n_mutants": raw_total, "n_killed": killed,
            "n_in_band": sum(r["n_in_band"] for r in rows),
            "n_in_scope": in_scope, "n_killed_in_scope": killed,
            "kill_rate": (killed / in_scope) if in_scope else None,
            "kill_rate_gate_only": (killed_gate / in_scope) if in_scope else None,
            "kill_rate_raw": (killed / raw_total) if raw_total else None,
            "external": False,
            "control": gate == "negative_control",
        })
    if external:
        gate_rows.append({
            "gate": external.get("gate", "paper_trace"),
            "n_criteria": external.get("total"),
            "n_criteria_instantiated": external.get("total"),
            "n_criteria_with_mutants": external.get("total"),
            "n_criteria_unfalsifiable": 0,
            "n_mutants": external.get("total"), "n_killed": external.get("killed"),
            "n_in_band": 0, "n_in_scope": external.get("total"),
            "n_killed_in_scope": external.get("killed"),
            "kill_rate": external.get("kill_rate"),
            "kill_rate_gate_only": external.get("kill_rate"),
            "kill_rate_raw": external.get("kill_rate"),
            "external": True, "control": False, "runs": external.get("runs"),
        })

    ladder = {}
    for eps in EPS_LADDER:
        rows = [r for r in records if r["klass"] == "M1" and r["epsilon"] == eps
                and r["gate"] != "negative_control"]
        ladder[f"{eps:.2f}"] = {"n_mutants": len(rows),
                                "killed": sum(1 for r in rows if r["killed"]),
                                "killed_gate_only": sum(1 for r in rows if r["killed_gate_only"])}
    counts = [ladder[f"{e:.2f}"]["killed"] for e in EPS_LADDER]
    violations = sum(1 for a, b in zip(counts, counts[1:]) if b < a)

    measured_rows = [r for r in gate_rows if r["gate"] != "negative_control"]
    nc_rows = [r for r in gate_rows if r["gate"] == "negative_control"]
    total_scope = sum(r["n_in_scope"] for r in measured_rows)
    total_killed = sum(r["n_killed_in_scope"] for r in measured_rows)
    total_gate_only = sum(r["n_killed_in_scope"] if r["external"] else
                          round((r["kill_rate_gate_only"] or 0.0) * (r["n_in_scope"] or 0))
                          for r in measured_rows)
    instantiation = sum(1 for c in per_criterion if c["instantiated"])
    summary: dict[str, Any] = {
        "schema": "tensorphysics.criteria_mutation.ledger/1",
        "kpi_target": KPI_TARGET,
        "eps_ladder": list(EPS_LADDER),
        "n_gates": len(measured_rows),
        "n_gates_python": sum(1 for r in measured_rows if not r["external"]),
        "gates": gate_rows,
        "n_criteria": len(per_criterion),
        "n_criteria_instantiated": instantiation,
        "instantiation_rate": (instantiation / len(per_criterion)) if per_criterion else None,
        "n_criteria_with_mutant_coverage": sum(1 for c in per_criterion if c["n_mutants"]),
        "mutant_coverage_rate": (sum(1 for c in per_criterion if c["n_mutants"]) /
                                 len(per_criterion)) if per_criterion else None,
        "n_mutants": sum(r["n_mutants"] for r in measured_rows),
        "n_in_band": sum(r["n_in_band"] for r in measured_rows),
        "n_in_scope": total_scope,
        "n_killed": total_killed,
        "kill_rate": (total_killed / total_scope) if total_scope else None,
        "n_killed_gate_only": total_gate_only,
        "kill_rate_gate_only": (total_gate_only / total_scope) if total_scope else None,
        "kill_rate_raw": (sum(r["n_killed"] for r in measured_rows) /
                          max(1, sum(r["n_mutants"] for r in measured_rows))),
        "negative_control": {
            "n_criteria": sum(r["n_criteria"] for r in nc_rows),
            "n_mutants": sum(r["n_mutants"] for r in nc_rows),
            "n_killed": sum(r["n_killed"] for r in nc_rows),
            "kill_rate": ((sum(r["n_killed"] for r in nc_rows) /
                           sum(r["n_mutants"] for r in nc_rows))
                          if sum(r["n_mutants"] for r in nc_rows) else None),
            "flagged_unfalsifiable": [c["id"] for c in per_criterion
                                      if c["gate"] == "negative_control"
                                      and c["unfalsifiable"]],
        },
        "epsilon_ladder": ladder,
        "epsilon_ladder_violations": violations,
        "unfalsifiable_criteria": [c["id"] for c in per_criterion
                                   if c["unfalsifiable"] and c["gate"] != "negative_control"],
        "no_mutant_coverage_criteria": [c["id"] for c in per_criterion
                                        if c["no_mutant_coverage"]],
        "weakest_mutants": sorted(
            ({"criterion_id": r["criterion_id"], "gate": r["gate"],
              "klass": r["klass"], "label": r["label"], "in_band": r["in_band"],
              "detail": r["detail"]}
             for r in records if not r["killed"] and r["gate"] != "negative_control"),
            key=lambda r: (r["gate"], r["criterion_id"], r["label"])),
        "blind_spots": sorted(
            ({"criterion_id": r["criterion_id"], "gate": r["gate"],
              "klass": r["klass"], "label": r["label"], "detail": r["detail"]}
             for r in records
             if not r["killed"] and not r["in_band"] and r["gate"] != "negative_control"),
            key=lambda r: (r["gate"], r["criterion_id"], r["label"])),
        "equivalent_survivors": sorted(
            ({"criterion_id": r["criterion_id"], "gate": r["gate"],
              "klass": r["klass"], "label": r["label"],
              "epsilon": r["epsilon"]}
             for r in records
             if not r["killed"] and r["in_band"] and r["gate"] != "negative_control"),
            key=lambda r: (r["gate"], r["criterion_id"], r["label"])),
        "gate_only_blind_spots": sorted(
            ({"criterion_id": r["criterion_id"], "gate": r["gate"],
              "klass": r["klass"], "label": r["label"]}
             for r in records if r["killed"] and not r["killed_gate_only"]),
            key=lambda r: (r["gate"], r["criterion_id"], r["label"])),
        "criteria": per_criterion,
        "mutants": records,
        "engine_lexicon": scan_engine_lexicon(Path(__file__).read_text(encoding="utf-8")),
        "external": external,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "budget_s": 300.0,
    }
    summary["budget_ok"] = summary["elapsed_s"] < summary["budget_s"]
    summary["meets_target"] = bool(summary["kill_rate"] is not None and
                                   summary["kill_rate"] >= KPI_TARGET)
    summary["gates_below_target"] = [r["gate"] for r in measured_rows
                                     if r["kill_rate"] is not None
                                     and r["kill_rate"] < KPI_TARGET]
    return summary# ---------------------------------------------------------------------------
# merge gate
# ---------------------------------------------------------------------------
def pr_gate(current: Mapping[str, Any],
            baseline: Mapping[str, Any] | None) -> dict[str, Any]:
    """Every criterion new relative to ``baseline`` must carry a killer test."""
    base_ids = {c["id"] for c in (baseline or {}).get("criteria", [])}
    new = [c for c in current.get("criteria", [])
           if c["id"] not in base_ids and c.get("gate") != "negative_control"]
    graded = []
    for c in new:
        ok = bool(c.get("n_killed", 0) >= 1 and c.get("n_mutants", 0) >= 1)
        graded.append({
            "id": c["id"], "gate": c["gate"],
            "n_mutants": c.get("n_mutants", 0), "n_killed": c.get("n_killed", 0),
            "ok": ok,
            "reason": ("kills >= 1 mutant" if ok else
                       "no killer test: this criterion is unfalsifiable"),
        })
    rejected = [g for g in graded if not g["ok"]]
    return {"schema": "tensorphysics.criteria_mutation.pr_gate/1",
            "n_new_criteria": len(new), "new_criteria": graded,
            "verdict": "REJECT" if rejected else "ACCEPT",
            "rejected": rejected, "baseline_criteria": len(base_ids),
            "current_criteria": len(current.get("criteria", []))}


def pr_demo(summary: Mapping[str, Any], baseline: Mapping[str, Any] | None,
            out_dir: str | Path) -> dict[str, Any]:
    """Write the red/green pair proving the merge gate is not a rubber stamp.

    ``green`` = the ledger as it stands (every criterion carries a killer);
    ``red``   = the same ledger plus one new criterion with no killer test,
    exactly what a careless PR would add.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    green = json.loads(json.dumps(summary))
    green.pop("mutants", None)
    green_verdict = pr_gate(green, baseline)
    red = json.loads(json.dumps(green))
    red["criteria"] = list(red.get("criteria", [])) + [{
        "id": "DEMO-NEW-NOKILLER", "gate": "demo",
        "callable": "criteria_mutation:tautology_factory",
        "expected_verdict": "PASS", "evidence_ref": "pr_demo (new criterion)",
        "check": None, "mutant_classes": [], "tolerance": 0.0,
        "note": "new criterion shipped without a killer test",
        "n_mutants": 0, "n_killed": 0,
    }]
    red_verdict = pr_gate(red, baseline)
    (out / "green_ledger.json").write_text(json.dumps(green, indent=1) + "\n",
                                           encoding="utf-8")
    (out / "green_verdict.json").write_text(json.dumps(green_verdict, indent=1) + "\n",
                                            encoding="utf-8")
    (out / "red_ledger.json").write_text(json.dumps(red, indent=1) + "\n",
                                         encoding="utf-8")
    (out / "red_verdict.json").write_text(json.dumps(red_verdict, indent=1) + "\n",
                                          encoding="utf-8")
    return {"green": green_verdict, "red": red_verdict,
            "exit_codes": {"green": 0 if green_verdict["verdict"] == "ACCEPT" else 2,
                           "red": 0 if red_verdict["verdict"] == "ACCEPT" else 2},
            "out_dir": str(out)}


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.1f}%"


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Criteria falsifiability ledger (criteria_mutation)",
        "",
        f"- gates audited: **{summary['n_gates']}** "
        f"({summary['n_gates_python']} TensorPhysics code gates + external judge)",
        f"- criteria instantiated: **{summary['n_criteria_instantiated']}"
        f"/{summary['n_criteria']}** ({_pct(summary['instantiation_rate'])})",
        f"- mutant coverage: **{summary['n_criteria_with_mutant_coverage']}"
        f"/{summary['n_criteria']}** criteria carry >= 1 mutant "
        f"({_pct(summary['mutant_coverage_rate'])})",
        f"- mutants in scope: **{summary['n_in_scope']}** "
        f"(in-band, i.e. inside the criterion's declared tolerance: {summary['n_in_band']})",
        f"- kills: **{summary['n_killed']}/{summary['n_in_scope']} = "
        f"{_pct(summary['kill_rate'])}** (target {_pct(summary['kpi_target'])}, "
        f"meets_target={summary['meets_target']})",
        f"- gate-only kills (provenance guard removed): {summary['n_killed_gate_only']}"
        f"/{summary['n_in_scope']} = {_pct(summary['kill_rate_gate_only'])}",
        f"- M1 eps-ladder kills: "
        + ", ".join(f"{eps}: {row['killed']}/{row['n_mutants']}"
                    for eps, row in summary["epsilon_ladder"].items())
        + f" -> violations **{summary['epsilon_ladder_violations']}**",
        f"- unfalsifiable criteria: {len(summary['unfalsifiable_criteria'])}"
        f" {summary['unfalsifiable_criteria']}",
        f"- runtime: {summary['elapsed_s']} s (budget {summary['budget_s']:.0f} s, "
        f"ok={summary['budget_ok']})",
        "",
        "| gate | criteria | mutants | in-band | kills | kill-rate | gate-only |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in summary["gates"]:
        lines.append(
            f"| {row['gate']} | {row['n_criteria']} | {row['n_mutants']} | "
            f"{row['n_in_band']} | {row['n_killed_in_scope']}/{row['n_in_scope']} | "
            f"{_pct(row['kill_rate'])} | {_pct(row['kill_rate_gate_only'])} |")
    if summary.get("weakest_mutants"):
        lines += ["", "## Weakest mutants (survivors)", "",
                  "| gate | criterion | class | mutant |", "|---|---|---|---|"]
        for row in summary["weakest_mutants"]:
            lines.append(f"| {row['gate']} | {row['criterion_id']} | {row['klass']} | "
                         f"{row['label']} |")
    if summary.get("equivalent_survivors"):
        lines += ["", "## Equivalent survivors (inside the declared tolerance band)", "",
                  "| gate | criterion | class | mutant |", "|---|---|---|---|"]
        for row in summary["equivalent_survivors"]:
            lines.append(f"| {row['gate']} | {row['criterion_id']} | {row['klass']} | "
                         f"{row['label']} |")
    if summary.get("gate_only_blind_spots"):
        lines += ["", "## Gate-only blind spots (killed only by the provenance guard)", "",
                  "| gate | criterion | class | mutant |", "|---|---|---|---|"]
        for row in summary["gate_only_blind_spots"]:
            lines.append(f"| {row['gate']} | {row['criterion_id']} | {row['klass']} | "
                         f"{row['label']} |")
    ext = summary.get("external")
    if ext:
        lines += ["", "## External judge (consumed from its own mutant records)", "",
                  f"- gate: `{ext.get('gate')}`, source `{ext.get('source')}` "
                  f"(sha256 `{(ext.get('source_sha256') or '')[:16]}...`)",
                  f"- operative run: `{ext.get('operative_run')}`"]
        for name, run in (ext.get("runs") or {}).items():
            lines.append(f"  - `{name}`: {run['killed']}/{run['total']} = "
                         f"{_pct(run['kill_rate'])} "
                         f"(recomputed from records, matches declared: "
                         f"{run['recomputed_matches_declared']})")
    return "\n".join(lines) + "\n"


def _class_matrix(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for gate in summary["gates"]:
        if gate.get("external"):
            continue
        row = {"gate": gate["gate"]}
        for klass in MUTANT_CLASSES:
            picked = [r for r in summary["mutants"]
                      if r["gate"] == gate["gate"] and r["klass"] == klass]
            row[klass] = (sum(1 for r in picked if r["killed"]) / len(picked)
                          if picked else None)
            row[f"{klass}_n"] = len(picked)
        rows.append(row)
    return rows


def render_figures(summary: Mapping[str, Any], out_dir: str | Path) -> list[Path]:  # pragma: no cover
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    gates = [g["gate"] for g in summary["gates"]]
    ledger = [g["kill_rate"] or 0.0 for g in summary["gates"]]
    gate_only = [g["kill_rate_gate_only"] or 0.0 for g in summary["gates"]]

    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = range(len(gates))
    ax.bar([i - 0.2 for i in x], ledger, width=0.4, label="ledger (gate + provenance guard)",
           color="#2b6cb0")
    ax.bar([i + 0.2 for i in x], gate_only, width=0.4, label="gate function only",
           color="#f6ad55")
    ax.axhline(summary["kpi_target"], ls="--", lw=1.2, color="#c53030",
               label=f"KPI target {int(100 * summary['kpi_target'])}%")
    ax.plot([-0.4, len(gates) - 0.6], [summary["kill_rate"]] * 2, lw=1.0,
            color="#2f855a", ls=":", label=f"ledger overall {100 * summary['kill_rate']:.1f}%")
    ax.set_xticks(list(x))
    ax.set_xticklabels(gates, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mutant kill rate")
    ax.set_title("Criteria falsifiability per gate (M1-M5 mutants, in-scope)")
    for i, (a, b) in enumerate(zip(ledger, gate_only)):
        ax.text(i - 0.2, a + 0.02, f"{100 * a:.0f}", ha="center", fontsize=7)
        ax.text(i + 0.2, b + 0.02, f"{100 * b:.0f}", ha="center", fontsize=7)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    path = out / "fig1_gate_killrate.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)

    ladder = summary["epsilon_ladder"]
    eps = list(ladder)
    killed = [ladder[e]["killed"] for e in eps]
    total = [ladder[e]["n_mutants"] for e in eps]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot([float(e) * 100 for e in eps], killed, "o-", lw=2, color="#2b6cb0",
            label="kills (ledger)")
    ax.plot([float(e) * 100 for e in eps], total, "s--", lw=1, color="#a0aec0",
            label="M1 mutants in scope")
    for e, k, n in zip(eps, killed, total):
        ax.annotate(f"{k}/{n}", (float(e) * 100, k), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8)
    ax.set_xlabel("M1 perturbation epsilon [%]")
    ax.set_ylabel("mutants killed")
    ax.set_title(f"M1 eps-ladder monotonicity (violations={summary['epsilon_ladder_violations']})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "fig2_epsilon_ladder.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)

    matrix = _class_matrix(summary)
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    data = [[(row[k] if row[k] is not None else float("nan")) for k in MUTANT_CLASSES]
            for row in matrix]
    im = ax.imshow(data, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(MUTANT_CLASSES)))
    ax.set_xticklabels(MUTANT_CLASSES)
    ax.set_yticks(range(len(matrix)))
    ax.set_yticklabels([row["gate"] for row in matrix])
    for i, row in enumerate(matrix):
        for j, k in enumerate(MUTANT_CLASSES):
            value = row[k]
            label = "n/a" if value is None else f"{100 * value:.0f}% (n={row[f'{k}_n']})"
            ax.text(j, i, label, ha="center", va="center", fontsize=7,
                    color="black")
    ax.set_title("Kill rate by mutant class (blank = class not declared)")
    fig.colorbar(im, ax=ax, fraction=0.03, label="kill rate")
    fig.tight_layout()
    path = out / "fig3_class_matrix.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)
    return written


# ---------------------------------------------------------------------------
# external record import (consume the external judge's own per-mutant output)
# ---------------------------------------------------------------------------
def record_external(baseline_path: str | Path, hardened_path: str | Path, *,
                    gate: str = "paper_trace", module: str = "",
                    platform_git: str = "", operative: str = "hardened",
                    out_path: str | Path | None = None) -> dict[str, Any]:
    runs: dict[str, Any] = {}
    for name, path in (("baseline", baseline_path), ("hardened", hardened_path)):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        runs[name] = {
            "tests": raw.get("tests"),
            "total": raw.get("total"),
            "killed": raw.get("killed"),
            "baseline_green": raw.get("baseline_green"),
            "by_operator": raw.get("by_operator"),
            "mutants": [{"id": m.get("id"), "operator": m.get("operator"),
                         "line": m.get("line"), "killed": bool(m.get("killed"))}
                        for m in raw.get("mutants", [])],
        }
    payload = {
        "gate": gate,
        "provenance": {
            "module": module or (json.loads(Path(baseline_path).read_text(
                encoding="utf-8")).get("module")),
            "platform_git": platform_git,
            "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_files": {"baseline": file_sha256(baseline_path),
                             "hardened": file_sha256(hardened_path)},
            "note": "per-mutant records produced by the external judge's own "
                    "mutation audit; TensorPhysics only recomputes the kill rate",
        },
        "operative_run": operative,
        "runs": runs,
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="criteria falsifiability ledger audit")
    ap.add_argument("--external", default=str(ARTIFACT_DIR / "paper_trace_audit.json"),
                    help="recorded external audit (per-mutant records)")
    ap.add_argument("--no-external", action="store_true")
    ap.add_argument("--no-negative-control", action="store_true")
    ap.add_argument("--gates", default="", help="comma separated subset of code gates")
    ap.add_argument("--json-out", default=str(ARTIFACT_DIR / "ledger.json"))
    ap.add_argument("--md-out", default=str(ARTIFACT_DIR / "ledger.md"))
    ap.add_argument("--figures-dir", default=str(ARTIFACT_DIR))
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--pr-check", action="store_true",
                    help="grade only the criteria that are new vs --baseline-ledger")
    ap.add_argument("--baseline-ledger", default="")
    ap.add_argument("--ledger", default="", help="ledger JSON to grade for --pr-check")
    ap.add_argument("--pr-json-out", default="")
    ap.add_argument("--pr-demo", default="",
                    help="write the red/green merge-gate evidence into this directory")
    ap.add_argument("--record-external", nargs=2, metavar=("BASELINE", "HARDENED"),
                    help="convert an external judge's two audit JSONs into a record")
    ap.add_argument("--record-out", default=str(ARTIFACT_DIR / "paper_trace_audit.json"))
    ap.add_argument("--record-gate", default="paper_trace")
    ap.add_argument("--record-module", default="")
    ap.add_argument("--record-git", default="")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.record_external:
        payload = record_external(args.record_external[0], args.record_external[1],
                                  gate=args.record_gate, module=args.record_module,
                                  platform_git=args.record_git, out_path=args.record_out)
        print(json.dumps({"recorded": args.record_out,
                          "runs": {k: {"total": v["total"], "killed": v["killed"]}
                                   for k, v in payload["runs"].items()}}, indent=1))
        return 0

    if args.pr_check:
        if not args.ledger or not args.baseline_ledger:
            print("--pr-check needs --ledger and --baseline-ledger", file=sys.stderr)
            return 2
        current = json.loads(Path(args.ledger).read_text(encoding="utf-8"))
        baseline = json.loads(Path(args.baseline_ledger).read_text(encoding="utf-8"))
        verdict = pr_gate(current, baseline)
        text = json.dumps(verdict, indent=1)
        if args.pr_json_out:
            Path(args.pr_json_out).write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0 if verdict["verdict"] == "ACCEPT" else 2

    external = None if args.no_external else load_external(args.external)
    gates = [g.strip() for g in args.gates.split(",") if g.strip()] or None
    summary = run_ledger(external=external, gates=gates,
                         include_negative_control=not args.no_negative_control)
    summary["command"] = " ".join(["criteria_mutation.py"] + list(argv or []))
    summary["git_head"] = _git_head()
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    md = Path(args.md_out)
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(render_markdown(summary), encoding="utf-8")
    figures: list[str] = []
    if not args.no_figures:
        try:
            figures = [str(p) for p in render_figures(summary, args.figures_dir)]
        except Exception as exc:   # plotting is reporting, never the verdict
            print(f"[warn] figures skipped: {type(exc).__name__}: {exc}", file=sys.stderr)
    if args.pr_demo:
        baseline_path = args.baseline_ledger or str(ARTIFACT_DIR / "baseline_ledger.json")
        baseline = (json.loads(Path(baseline_path).read_text(encoding="utf-8"))
                    if Path(baseline_path).is_file() else None)
        demo = pr_demo(summary, baseline, args.pr_demo)
        print(json.dumps({"pr_demo": {"green": demo["green"]["verdict"],
                                      "red": demo["red"]["verdict"],
                                      "exit_codes": demo["exit_codes"],
                                      "out_dir": demo["out_dir"]}}, indent=1))
    print(json.dumps({
        "gates": summary["n_gates"],
        "criteria": summary["n_criteria"],
        "instantiation_rate": summary["instantiation_rate"],
        "kill_rate": summary["kill_rate"],
        "kill_rate_gate_only": summary["kill_rate_gate_only"],
        "kills": f"{summary['n_killed']}/{summary['n_in_scope']}",
        "meets_target": summary["meets_target"],
        "epsilon_ladder_violations": summary["epsilon_ladder_violations"],
        "unfalsifiable": summary["unfalsifiable_criteria"],
        "elapsed_s": summary["elapsed_s"],
        "json_out": str(out),
        "figures": figures,
    }, indent=1))
    return 0


def _git_head() -> str:
    head = ROOT / ".git"
    try:
        if head.is_file():
            text = head.read_text(encoding="utf-8").strip()
            if text.startswith("gitdir:"):
                git_dir = Path(text.split(":", 1)[1].strip())
                if not git_dir.is_absolute():
                    git_dir = (head.parent / git_dir).resolve()
                head = git_dir
        return (head / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())