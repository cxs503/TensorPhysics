"""W6 physics-contract verification: mutation testing of the resistance-chain gate.

Motivation
----------
The contract gate for the resistance chain (``tensorphysics.physics_contracts``
plus the audited SUBOFF-01 anchors) had *no ablation evidence*: nobody had
shown that the gate actually **fails** on a contract violation.  A gate that
passes everything is indistinguishable from a gate that works.  This module
injects six single-point mutants -- each a real historical or plausible failure
mode -- and records, per mutant, whether the gate FAILs, which criterion fires,
and whether the failure message localises the mutation site.

Gate criteria (tolerances are the *audited* ones; none loosened here)
---------------------------------------------------------------------
1.  ``naming_gate_no_false_holtrop``     -- Holtrop-Mennen label on the
    ITTC-1957+residual composition must raise at construction (audit m2086).
2.  ``declared_formula_matches_impl``    -- the Cf formula *declared* in the
    contract must be the one the code evaluates (catches coefficient drift).
3.  ``derived_anchor_bitwise``           -- Ct vs the 6-kernel bitwise anchor
    (rel 1e-9).
4.  ``derived_anchor_not_self_derived``  -- the DERIVED anchor must REJECT a
    +2% Cf perturbation; an anchor recomputed from the implementation under
    test accepts it (tautology).
5.  ``resistance_anchor_rt``             -- Rt [N] vs the audited anchor
    (rel 1e-9); this is the criterion that sees S_wet drift.
6.  ``experimental_anchor_bias``         -- Ct vs SUBOFF-01 0.0038 inside the
    contract's 15% EXPERIMENTAL band.
7.  ``experimental_tolerance_policy``    -- an EXPERIMENTAL band wider than the
    15% acceptance policy is itself a violation (anti-tolerance-loosening).
8.  ``experimental_anchor_sensitivity``  -- the EXPERIMENTAL band must reject a
    +30% Ct bias.
9.  ``dimensionless_scale_consistency``  -- Ct/Cf must be O(1) (catches a 1e-3
    unit-convention / scale mismatch).
10. ``re_applicability_domain``          -- declared Re band must equal the
    ITTC-1957 validated band and must NOT admit extrapolation cases.
11. ``wetted_surface_closed_form_bias``  -- the empirical S fit must keep its
    audited bias against closed-form cylinder / spheroid geometry and the
    published KVLCC2 area.

Artifacts (numbers are read straight from the run, never retyped):
  artifacts/contract_mutation/mutants.json
  artifacts/contract_mutation/verdict.json
  artifacts/contract_mutation/summary.md
  artifacts/contract_mutation/fig1_mutation_matrix.png
  artifacts/contract_mutation/fig2_anchor_sensitivity.png

Anchor provenance: audit m2086 crosscheck, case SUBOFF-01 (L=4.356 m,
B=T=0.508 m, Cb=0.542, V=3.0 m/s, rho=998.2 kg/m^3, nu=1.004e-6 m^2/s,
S=4.417911491407555 m^2); 6 of 11 audited kernels agree bitwise on Ct.
"""
from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Sequence, Tuple

import pytest

from tensorphysics import physics_contracts as pc
from tensorphysics.physics_kernels_resistance import (
    axisymmetric_wetted_surface,
    ittc_1957_cf,
    suboff_residual_coefficient,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "contract_mutation"

G = 9.81

# SUBOFF-01 audited case (input table, audit m2086).
SUBOFF = dict(length_m=4.356, beam_m=0.508, cb=0.542, speed_ms=3.0,
              rho_kg_m3=998.2, nu_m2_s=1.004e-06)
KVLCC2 = dict(length_m=320.0, beam_m=58.0, cb=0.8098)

# --- audited anchors: values, provenance, tolerances (NOT loosened) --------
ANCHOR_CT_DERIVED = 0.003461889103158906      # 6/11 kernels bitwise, audit m2086
ANCHOR_RT_DERIVED = 68.7005544395378          # same run, N
ANCHOR_CT_EXPERIMENTAL = 0.0038               # SUBOFF-01 published experiment
DECLARED_CF_FORMULA = "Cf = 0.075/(log10(Re)-2)^2"
DERIVED_TOL = 1e-9
EXPERIMENTAL_TOL = 0.15
POLICY_MAX_EXPERIMENTAL_TOL = 0.15            # owner acceptance standard
REF_RE_DOMAIN = (1e5, 1e9)                    # ITTC-1957 validated Re band
EXTRAPOLATION_CASES = (1e2, 5e9)              # must NOT be reported qualified

# Physical perturbations used to test anchor *sensitivity* (not tolerance).
PROBE_CF_COEFF = 0.0765                       # +2.0% on the 0.075 coefficient
PROBE_CT_BIAS = 0.30                          # +30% Ct bias


# ---------------------------------------------------------------------------
# Independent reference implementations (deliberately NOT the kernel under test)
# ---------------------------------------------------------------------------

def ref_cylinder_lateral_surface(length_m: float, beam_m: float) -> float:
    """Closed-form lateral area of a cylinder of diameter B and length L."""
    return math.pi * beam_m * length_m


def ref_prolate_spheroid_surface(length_m: float, beam_m: float) -> float:
    """Closed-form surface of a prolate spheroid with semi-axes (L/2, B/2)."""
    a, b = 0.5 * length_m, 0.5 * beam_m
    e = math.sqrt(1.0 - (b / a) ** 2)
    return 2.0 * math.pi * b * b + 2.0 * math.pi * a * b * math.asin(e) / e


def ref_cf_coefficient_value(re: float, coeff: float = 0.075) -> float:
    return coeff / (math.log10(re) - 2.0) ** 2


def eval_declared_cf_formula(formula: str, re: float) -> float:
    """Evaluate the *declared* Cf formula string (fixed corpus, no user input)."""
    body = formula.split("=", 1)[1].strip().replace("^", "**")
    return float(eval(body, {"__builtins__": {}},  # noqa: S307 - literal formulas
                      {"log10": math.log10, "Re": re}))


def cf_with_coefficient(coeff: float) -> Callable[[float], float]:
    return lambda re: ref_cf_coefficient_value(re, coeff)


def _ct_from_cf(cf: Callable[[float], float],
                case: Dict[str, float] = SUBOFF) -> float:
    """Ct = Cf + Cr of the shared-core composition (form factor 1.0)."""
    re = case["speed_ms"] * case["length_m"] / case["nu_m2_s"]
    fn = case["speed_ms"] / math.sqrt(G * case["length_m"])
    return cf(re) + suboff_residual_coefficient(fn, case["cb"])


# ---------------------------------------------------------------------------
# Candidate model under test
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    """A resistance-chain implementation plus its contract declarations."""

    name: str
    cf: Callable[[float], float]
    wetted: Callable[[float, float, float], float]
    derived_anchor: Callable[[Callable[[float], float]], float]
    derived_tol: float = DERIVED_TOL
    experimental_tol: float = EXPERIMENTAL_TOL
    re_domain: Tuple[float, float] = REF_RE_DOMAIN
    declared_cf_formula: str = DECLARED_CF_FORMULA
    ct_scale: float = 1.0            # reported-coefficient convention factor
    mutation: str = "wild type (audited composition)"

    @property
    def reynolds(self) -> float:
        return SUBOFF["speed_ms"] * SUBOFF["length_m"] / SUBOFF["nu_m2_s"]

    @property
    def froude(self) -> float:
        return SUBOFF["speed_ms"] / math.sqrt(G * SUBOFF["length_m"])

    def ct_true(self, cf: Callable[[float], float] | None = None) -> float:
        return _ct_from_cf(self.cf if cf is None else cf)

    def ct(self, cf: Callable[[float], float] | None = None) -> float:
        return self.ct_scale * self.ct_true(cf)

    def rt(self, cf: Callable[[float], float] | None = None) -> float:
        s = self.wetted(SUBOFF["length_m"], SUBOFF["beam_m"], SUBOFF["cb"])
        q = 0.5 * SUBOFF["rho_kg_m3"] * s * SUBOFF["speed_ms"] ** 2
        return q * self.ct_true(cf)

    def qualifies(self, re: float) -> bool:
        lo, hi = self.re_domain
        return lo <= re <= hi


@dataclass(frozen=True)
class CriterionResult:
    criterion: str
    passed: bool
    detail: str


def _rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b) if b else float("inf")


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def crit_naming_gate_no_false_holtrop(c: Candidate) -> CriterionResult:
    base = pc.get_contract("resistance.suboff_shared_core")
    try:
        pc.PhysicsModelContract(
            key="resistance.mutated_mislabel",
            declared_method="simplified Holtrop-Mennen formulation",
            inputs=base.inputs, components=base.components,
            reference_kernel="tensorphysics.physics_kernels_resistance."
                             "shared_core_resistance",
        )
    except pc.PhysicsContractError as exc:
        return CriterionResult("naming_gate_no_false_holtrop", True,
                               f"mislabel rejected at construction: {exc}")
    return CriterionResult("naming_gate_no_false_holtrop", False,
                           "Holtrop-Mennen label accepted on a composition "
                           "missing the full-method components")


def crit_declared_formula_matches_impl(c: Candidate) -> CriterionResult:
    re_probe = 1e7
    declared = eval_declared_cf_formula(c.declared_cf_formula, re_probe)
    impl = c.cf(re_probe)
    implied = impl * (math.log10(re_probe) - 2.0) ** 2  # coefficient code uses
    rel = _rel(impl, declared)
    detail = (f"declared {c.declared_cf_formula!r} -> {declared:.6e} at Re=1e7, "
              f"implementation -> {impl:.6e}, implied coefficient {implied:.5f}, "
              f"rel={rel:.3e}")
    return CriterionResult("declared_formula_matches_impl", rel <= 1e-12, detail)


def crit_derived_anchor_bitwise(c: Candidate) -> CriterionResult:
    anchor = c.derived_anchor(c.cf)
    ct = c.ct()
    rel = _rel(ct, anchor)
    return CriterionResult(
        "derived_anchor_bitwise", rel <= c.derived_tol,
        f"Ct={ct:.17g} vs DERIVED anchor {anchor:.17g} -> rel={rel:.3e} "
        f"(tol {c.derived_tol:g})")


def crit_derived_anchor_not_self_derived(c: Candidate) -> CriterionResult:
    """A usable DERIVED anchor must reject a physically perturbed implementation."""
    probe_cf = cf_with_coefficient(PROBE_CF_COEFF)
    anchor_with_probe = c.derived_anchor(probe_cf)
    probe_ct = c.ct(probe_cf)
    rel = _rel(probe_ct, anchor_with_probe)
    detail = (f"probe coefficient {PROBE_CF_COEFF:g} (+"
              f"{(PROBE_CF_COEFF / 0.075 - 1) * 100:.1f}% Cf) Ct={probe_ct:.17g} "
              f"vs anchor {anchor_with_probe:.17g} -> rel={rel:.3e}")
    if rel <= c.derived_tol:
        detail += ("; anchor accepted the perturbation -- DERIVED anchor is "
                   "recomputed from the implementation under test "
                   "(tautological, self-certifying)")
    return CriterionResult("derived_anchor_not_self_derived", rel > c.derived_tol,
                           detail)


def crit_resistance_anchor_rt(c: Candidate) -> CriterionResult:
    rt = c.rt()
    rel = _rel(rt, ANCHOR_RT_DERIVED)
    return CriterionResult(
        "resistance_anchor_rt", rel <= c.derived_tol,
        f"Rt={rt:.15g} N vs DERIVED anchor {ANCHOR_RT_DERIVED:.15g} N -> "
        f"rel={rel:.3e} (tol {c.derived_tol:g}); Rt scales with S_wet")


def crit_experimental_anchor_bias(c: Candidate) -> CriterionResult:
    ct = c.ct()
    rel = _rel(ct, ANCHOR_CT_EXPERIMENTAL)
    return CriterionResult(
        "experimental_anchor_bias", rel <= c.experimental_tol,
        f"Ct={ct:.6e} vs EXPERIMENTAL {ANCHOR_CT_EXPERIMENTAL:g} -> "
        f"rel={rel:.3e} (band {c.experimental_tol:.2f})")


def crit_experimental_tolerance_policy(c: Candidate) -> CriterionResult:
    ok = c.experimental_tol <= POLICY_MAX_EXPERIMENTAL_TOL
    return CriterionResult(
        "experimental_tolerance_policy", ok,
        f"EXPERIMENTAL tolerance_rel={c.experimental_tol:.2f} vs acceptance "
        f"policy {POLICY_MAX_EXPERIMENTAL_TOL:.2f}"
        + ("" if ok else " -> band loosened beyond policy"))


def crit_experimental_anchor_sensitivity(c: Candidate) -> CriterionResult:
    probe = (1.0 + PROBE_CT_BIAS) * c.ct()
    rel = _rel(probe, ANCHOR_CT_EXPERIMENTAL)
    accepted = rel <= c.experimental_tol
    return CriterionResult(
        "experimental_anchor_sensitivity", not accepted,
        f"probe Ct={probe:.6e} (+{PROBE_CT_BIAS * 100:.0f}% bias) vs "
        f"EXPERIMENTAL {ANCHOR_CT_EXPERIMENTAL:g} -> rel={rel:.3e}; "
        f"accepted={accepted}")


def crit_dimensionless_scale_consistency(c: Candidate) -> CriterionResult:
    cf = c.cf(c.reynolds)
    ct = c.ct()
    ratio = ct / cf
    ok = (0.05 <= ratio <= 5.0) and (1e-5 <= ct <= 1e-1)
    return CriterionResult(
        "dimensionless_scale_consistency", ok,
        f"cf={cf:.6e}, ct={ct:.6e}, ct/cf={ratio:.4e} (expected O(1)); "
        f"dimensionless scale/unit convention check"
        + ("" if ok else " -> ~1e3 scale mismatch (Cf vs Ct)"))


def crit_re_applicability_domain(c: Candidate) -> CriterionResult:
    lo, hi = c.re_domain
    bounds_ok = (lo == REF_RE_DOMAIN[0]) and (hi == REF_RE_DOMAIN[1])
    inside = c.qualifies(c.reynolds)
    leaked = [re for re in EXTRAPOLATION_CASES if c.qualifies(re)]
    ok = bounds_ok and inside and not leaked
    detail = (f"declared Re domain ({lo:g}, {hi:g}) vs reference "
              f"({REF_RE_DOMAIN[0]:g}, {REF_RE_DOMAIN[1]:g}); case Re="
              f"{c.reynolds:.3e} qualified={inside}; extrapolation cases "
              f"{list(EXTRAPOLATION_CASES)} admitted as qualified={leaked}")
    return CriterionResult("re_applicability_domain", ok, detail)


def crit_wetted_surface_closed_form_bias(c: Candidate) -> CriterionResult:
    cb_cyl = math.pi / 4.0
    cb_sph = math.pi / 6.0
    cyl = c.wetted(10.0, 1.0, cb_cyl) / ref_cylinder_lateral_surface(10.0, 1.0)
    sph = c.wetted(4.0, 1.0, cb_sph) / ref_prolate_spheroid_surface(4.0, 1.0)
    kv = c.wetted(KVLCC2["length_m"], KVLCC2["beam_m"], KVLCC2["cb"])
    kv_ratio = kv / 27194.0
    # audited / measured reference bias of the axisymmetric fit
    cyl_ref, sph_ref, kv_ref = 0.69635, 0.78307, 40958.426
    ok = (abs(cyl - cyl_ref) <= 2e-2 and abs(sph - sph_ref) <= 2e-2
          and _rel(kv, kv_ref) <= 5e-3 and 1.49 <= kv_ratio <= 1.52)
    detail = (f"fit/cylinder closed form={cyl:.5f} (audited {cyl_ref:.5f}, "
              f"bias {100 * (cyl - 1):+.1f}%); fit/spheroid={sph:.5f} "
              f"(measured {sph_ref:.5f}); KVLCC2 S={kv:.1f} m^2 vs published "
              f"27194 -> {kv_ratio:.3f} (+{100 * (kv_ratio - 1):.1f}%)")
    return CriterionResult("wetted_surface_closed_form_bias", ok, detail)


CRITERIA: Sequence[Callable[[Candidate], CriterionResult]] = (
    crit_naming_gate_no_false_holtrop,
    crit_declared_formula_matches_impl,
    crit_derived_anchor_bitwise,
    crit_derived_anchor_not_self_derived,
    crit_resistance_anchor_rt,
    crit_experimental_anchor_bias,
    crit_experimental_tolerance_policy,
    crit_experimental_anchor_sensitivity,
    crit_dimensionless_scale_consistency,
    crit_re_applicability_domain,
    crit_wetted_surface_closed_form_bias,
)


def run_gate(candidate: Candidate) -> List[CriterionResult]:
    return [crit(candidate) for crit in CRITERIA]


# ---------------------------------------------------------------------------
# Wild type and the six mutants
# ---------------------------------------------------------------------------

def _constant_derived_anchor(cf: Callable[[float], float]) -> float:
    """Audited constant: six independent kernels agreed bitwise on this value.

    Ignores ``cf`` on purpose -- the number is an external record, not a
    recomputation from the implementation under test.
    """
    return ANCHOR_CT_DERIVED


def _tautological_derived_anchor(cf: Callable[[float], float]) -> float:
    """MUTANT M3: recompute the 'anchor' with the implementation under test."""
    return _ct_from_cf(cf)


def _ship_outside_hull_surface(length_m: float, beam_m: float, cb: float) -> float:
    """MUTANT M6: outside-hull ship formula dropped into the axisymmetric slot.

    Denny-Mumford style S = L*(2T + B)*sqrt(Cb) with T = B/2 -- a merchant-hull
    estimate applied to an axisymmetric body.
    """
    return length_m * (2.0 * 0.5 * beam_m + beam_m) * math.sqrt(cb)


WILD = Candidate(
    name="wild_type",
    cf=ittc_1957_cf,
    wetted=axisymmetric_wetted_surface,
    derived_anchor=_constant_derived_anchor,
)


@dataclass(frozen=True)
class Mutant:
    mid: str
    failure_mode: str
    injected_change: str
    candidate: Candidate
    expected_criteria: Tuple[str, ...]
    localisation_token: str


MUTANTS: Tuple[Mutant, ...] = (
    Mutant(
        mid="M1_cf_coefficient_0075_to_00765",
        failure_mode="Cf formula coefficient tampered: ITTC-1957 0.075 -> 0.0765 (+2.0%)",
        injected_change="ittc_1957_cf numerator 0.075 -> 0.0765",
        candidate=replace(WILD, name="M1", cf=cf_with_coefficient(PROBE_CF_COEFF),
                          mutation="Cf coefficient 0.075 -> 0.0765"),
        expected_criteria=("declared_formula_matches_impl", "derived_anchor_bitwise",
                           "resistance_anchor_rt"),
        localisation_token="0.0765",
    ),
    Mutant(
        mid="M2_re_domain_extrapolation_admitted",
        failure_mode="Re applicability band moved so the extrapolation region "
                     "counts as qualified",
        injected_change="declared Re domain (1e5, 1e9) -> (1e2, 1e12)",
        candidate=replace(WILD, name="M2", re_domain=(1e2, 1e12),
                          mutation="Re domain (1e5,1e9) -> (1e2,1e12)"),
        expected_criteria=("re_applicability_domain",),
        localisation_token="extrapolation",
    ),
    Mutant(
        mid="M3_derived_anchor_tautology",
        failure_mode="DERIVED anchor replaced by a recomputation from the "
                     "implementation under test",
        injected_change="DERIVED anchor resolver: audited constant -> "
                        "_ct_from_cf(candidate.cf)",
        candidate=replace(WILD, name="M3",
                          derived_anchor=_tautological_derived_anchor,
                          mutation="DERIVED anchor recomputed from candidate"),
        expected_criteria=("derived_anchor_not_self_derived",),
        localisation_token="tautological",
    ),
    Mutant(
        mid="M4_experimental_tolerance_15pct_to_50pct",
        failure_mode="EXPERIMENTAL anchor tolerance loosened 15% -> 50% to make "
                     "the check green",
        injected_change="ReferenceBand(SUBOFF-01, ct, 0.0038).tolerance_rel "
                        "0.15 -> 0.50",
        candidate=replace(WILD, name="M4", experimental_tol=0.50,
                          mutation="EXPERIMENTAL tolerance 0.15 -> 0.50"),
        expected_criteria=("experimental_tolerance_policy",
                           "experimental_anchor_sensitivity"),
        localisation_token="policy 0.15",
    ),
    Mutant(
        mid="M5_unit_convention_1e3",
        failure_mode="Unit convention mismatch: Cf and Ct differ by 1e3 "
                     "(per-mille coefficient report)",
        injected_change="reported dimensionless coefficient scaled by 1e3 "
                        "(force unchanged)",
        candidate=replace(WILD, name="M5", ct_scale=1e3,
                          mutation="dimensionless coefficient scale 1 -> 1e3"),
        expected_criteria=("dimensionless_scale_consistency",
                           "derived_anchor_bitwise"),
        localisation_token="1e3",
    ),
    Mutant(
        mid="M6_wetted_surface_formula_swap",
        failure_mode="Wetted-surface formula swapped (axisymmetric fit -> "
                     "outside-hull ship formula)",
        injected_change="axisymmetric_wetted_surface -> L*(2T+B)*sqrt(Cb), T=B/2",
        candidate=replace(WILD, name="M6", wetted=_ship_outside_hull_surface,
                          mutation="wetted-surface formula swapped"),
        expected_criteria=("wetted_surface_closed_form_bias", "resistance_anchor_rt"),
        localisation_token="cylinder",
    ),
)


# ---------------------------------------------------------------------------
# Matrix runner (single source of numbers for tests, JSON and figures)
# ---------------------------------------------------------------------------

def build_matrix() -> Dict[str, object]:
    cases = [("wild_type", WILD, "", ())] + [
        (m.mid, m.candidate, m.failure_mode, m.expected_criteria) for m in MUTANTS
    ]
    rows: Dict[str, Dict[str, object]] = {}
    for name, cand, failure_mode, expected in cases:
        results = run_gate(cand)
        rows[name] = {
            "candidate": name,
            "mutation": cand.mutation,
            "failure_mode": failure_mode,
            "caught": any(not r.passed for r in results),
            "failing_criteria": [r.criterion for r in results if not r.passed],
            "expected_criteria": list(expected),
            "localisation_token": next(
                (m.localisation_token for m in MUTANTS if m.mid == name), ""),
            "criteria": [
                {"criterion": r.criterion, "passed": r.passed, "detail": r.detail}
                for r in results
            ],
        }
    for mid, row in rows.items():
        if mid == "wild_type":
            continue
        failing = row["failing_criteria"]
        token = row["localisation_token"]
        failed_details = " ".join(c["detail"] for c in row["criteria"]
                                  if not c["passed"])
        row["localised"] = bool(token) and token in failed_details
        row["expected_criterion_fired"] = all(c in failing
                                              for c in row["expected_criteria"])
    return {
        "criteria": [c.__name__.removeprefix("crit_") for c in CRITERIA],
        "results": rows,
        "wild_type_passes": not rows["wild_type"]["caught"],
        "mutant_count": len(MUTANTS),
        "killed_count": sum(1 for m in MUTANTS if rows[m.mid]["caught"]),
        "coverage_holes": [m.mid for m in MUTANTS if not rows[m.mid]["caught"]],
        "not_triggered_criteria": {
            m.mid: [r["criterion"] for r in rows[m.mid]["criteria"] if r["passed"]]
            for m in MUTANTS
        },
    }


MATRIX = build_matrix()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_wild_type_passes_every_criterion():
    """Sensitivity control: the gate must not be a trivial always-FAIL."""
    assert MATRIX["wild_type_passes"] is True, [
        c for c in MATRIX["results"]["wild_type"]["criteria"] if not c["passed"]]


@pytest.mark.parametrize("mutant", MUTANTS, ids=[m.mid for m in MUTANTS])
def test_mutant_is_killed(mutant: Mutant):
    row = MATRIX["results"][mutant.mid]
    assert row["caught"] is True, f"{mutant.mid} survived the gate"
    assert row["expected_criterion_fired"] is True, (
        f"{mutant.mid}: expected criteria {mutant.expected_criteria} did not all "
        f"fire; fired {row['failing_criteria']}")


@pytest.mark.parametrize("mutant", MUTANTS, ids=[m.mid for m in MUTANTS])
def test_failure_message_localises_mutation(mutant: Mutant):
    row = MATRIX["results"][mutant.mid]
    assert row["localised"] is True, (
        f"{mutant.mid}: no failing-criterion message contains "
        f"{mutant.localisation_token!r}")


def test_no_coverage_holes():
    assert MATRIX["coverage_holes"] == []
    assert MATRIX["killed_count"] == MATRIX["mutant_count"] == len(MUTANTS)


def test_gate_is_discriminating_not_uniform():
    """Killed mutants must not all be killed by the same single criterion."""
    pattern = {tuple(MATRIX["results"][m.mid]["failing_criteria"]) for m in MUTANTS}
    assert len(pattern) >= 4


def test_anchors_not_loosened():
    """Guard against the anti-pattern this task exists to prevent."""
    assert DERIVED_TOL == 1e-9
    assert EXPERIMENTAL_TOL == 0.15
    assert POLICY_MAX_EXPERIMENTAL_TOL == 0.15
    reg = pc.get_contract("resistance.suboff_shared_core")
    exp = [r for r in reg.references if r.provenance == "EXPERIMENTAL"]
    deriv = [r for r in reg.references if r.provenance == "DERIVED"]
    assert [r.tolerance_rel for r in exp] == [0.15]
    assert [r.tolerance_rel for r in deriv] == [1e-9]
    assert WILD.experimental_tol == exp[0].tolerance_rel
    assert WILD.derived_tol == deriv[0].tolerance_rel


def test_artifacts_written():
    ART.mkdir(parents=True, exist_ok=True)
    rows = MATRIX["results"]

    mutants_payload = {
        "task": "W6 physics-contract-verification / resistance-chain mutation test",
        "source_module": "tests/test_contract_mutation.py",
        "case": "SUBOFF-01 (audit m2086 crosscheck, 6/11 kernels bitwise)",
        "anchor_provenance": {
            "ct_derived": ANCHOR_CT_DERIVED, "rt_derived": ANCHOR_RT_DERIVED,
            "ct_experimental": ANCHOR_CT_EXPERIMENTAL,
            "derived_tol": DERIVED_TOL, "experimental_tol": EXPERIMENTAL_TOL,
            "policy_max_experimental_tol": POLICY_MAX_EXPERIMENTAL_TOL,
            "re_domain": list(REF_RE_DOMAIN),
            "extrapolation_cases": list(EXTRAPOLATION_CASES),
            "declared_cf_formula": DECLARED_CF_FORMULA,
        },
        "mutants": [
            {
                "id": m.mid,
                "failure_mode": m.failure_mode,
                "injected_change": m.injected_change,
                "mutation": m.candidate.mutation,
                "expected_criteria": list(m.expected_criteria),
                "localisation_token": m.localisation_token,
                "gate": {
                    "caught": rows[m.mid]["caught"],
                    "failing_criteria": rows[m.mid]["failing_criteria"],
                    "localised": rows[m.mid]["localised"],
                    "criteria": rows[m.mid]["criteria"],
                },
            }
            for m in MUTANTS
        ],
    }
    (ART / "mutants.json").write_text(json.dumps(mutants_payload, indent=2) + "\n")

    verdict_payload = {
        "wild_type_passes": MATRIX["wild_type_passes"],
        "mutant_count": MATRIX["mutant_count"],
        "killed_count": MATRIX["killed_count"],
        "kill_rate": MATRIX["killed_count"] / MATRIX["mutant_count"],
        "coverage_holes": MATRIX["coverage_holes"],
        "localised_count": sum(1 for m in MUTANTS if rows[m.mid]["localised"]),
        "criteria_order": MATRIX["criteria"],
        "matrix": {
            mid: {c["criterion"]: ("PASS" if c["passed"] else "FAIL")
                  for c in rows[mid]["criteria"]}
            for mid in rows
        },
        "per_mutant": {
            mid: {
                "verdict": "KILLED" if rows[mid]["caught"] else "SURVIVED",
                "failing_criteria": rows[mid]["failing_criteria"],
                "expected_criteria": rows[mid]["expected_criteria"],
                "localised": rows[mid]["localised"],
            }
            for mid in rows if mid != "wild_type"
        },
        "not_triggered_criteria": MATRIX["not_triggered_criteria"],
    }
    (ART / "verdict.json").write_text(json.dumps(verdict_payload, indent=2) + "\n")

    _write_figures()
    _write_summary()

    for name in ("mutants.json", "verdict.json", "summary.md",
                 "fig1_mutation_matrix.png", "fig2_anchor_sensitivity.png"):
        assert (ART / name).stat().st_size > 0, name
    v = json.loads((ART / "verdict.json").read_text())
    assert v["kill_rate"] == 1.0 and v["coverage_holes"] == []
    assert v["localised_count"] == v["mutant_count"]


# ---------------------------------------------------------------------------
# Figures (matplotlib Agg; every number comes from MATRIX / the gate)
# ---------------------------------------------------------------------------

def _write_figures() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows, crits = MATRIX["results"], MATRIX["criteria"]
    mutant_ids = [m.mid for m in MUTANTS]
    grid = np.array([[0 if rows[mid]["criteria"][j]["passed"] else 1
                      for mid in mutant_ids] for j in range(len(crits))])

    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    ax.imshow(grid, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(mutant_ids)))
    ax.set_xticklabels(
        [mid.split("_")[0] + "\n" + mid.split("_", 1)[1].replace("_", " ")
         for mid in mutant_ids], fontsize=7, rotation=35, ha="right")
    ax.set_yticks(range(len(crits)))
    ax.set_yticklabels(crits, fontsize=8)
    for i in range(len(crits)):
        for j in range(len(mutant_ids)):
            ax.text(j, i, "caught" if grid[i, j] else "pass", ha="center",
                    va="center", fontsize=7,
                    color="white" if grid[i, j] else "black")
    ax.set_title("Resistance-chain contract gate vs. six single-point mutants\n"
                 "(red = criterion FAIL, i.e. mutant killed; green = criterion "
                 "passed)", fontsize=10)
    ax.set_xlabel("mutant", fontsize=9)
    ax.set_ylabel("gate criterion", fontsize=9)
    fig.tight_layout()
    fig.savefig(ART / "fig1_mutation_matrix.png", dpi=160)
    plt.close(fig)

    # --- fig 2: what the anchors can and cannot see -------------------------
    coeffs = np.linspace(0.072, 0.079, 281)
    re = WILD.reynolds
    ct_true = np.array([_ct_from_cf(cf_with_coefficient(k)) for k in coeffs])
    ct_report_wild = ct_true
    band = EXPERIMENTAL_TOL * ANCHOR_CT_EXPERIMENTAL

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.axhspan(ANCHOR_CT_EXPERIMENTAL - band, ANCHOR_CT_EXPERIMENTAL + band,
               color="tab:blue", alpha=0.18,
               label=f"EXPERIMENTAL band 0.0038 +/-15% (physics acceptance)")
    ax.axhline(ANCHOR_CT_DERIVED, color="k", ls="--", lw=1.2,
               label="DERIVED anchor, tolerance 1e-9 (implementation gate)")
    ax.plot(coeffs, ct_report_wild, color="tab:red", lw=2,
            label="Ct(Cf coefficient), SUBOFF-01")
    wild_ct = float(_ct_from_cf(cf_with_coefficient(0.075)))
    mut_ct = float(_ct_from_cf(cf_with_coefficient(PROBE_CF_COEFF)))
    ax.plot([0.075], [wild_ct], "o", ms=9, color="k", label="wild type 0.075")
    ax.plot([PROBE_CF_COEFF], [mut_ct], "o", ms=9, color="tab:purple",
            label=f"mutant M1 0.0765")
    rel_mut = abs(mut_ct - ANCHOR_CT_DERIVED) / ANCHOR_CT_DERIVED
    rel_exp = abs(mut_ct - ANCHOR_CT_EXPERIMENTAL) / ANCHOR_CT_EXPERIMENTAL
    ax.annotate(f"M1: dCt/anchor = {rel_mut:.2e}\n"
                f"(DERIVED tol 1e-9 -> caught)\n"
                f"vs EXPERIMENTAL {rel_exp * 100:.1f}% "
                f"(< 15% band -> physics check blind)",
                xy=(PROBE_CF_COEFF, mut_ct), xytext=(0.0725, 0.00362),
                fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.set_xlabel("Cf numerator coefficient (declared 0.075)", fontsize=9)
    ax.set_ylabel("Ct (dimensionless)", fontsize=9)
    ax.set_title("Anchor sensitivity: a +2% Cf tamper is invisible to the 15% "
                 "physics band\nand only the bitwise DERIVED anchor catches it",
                 fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ART / "fig2_anchor_sensitivity.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# summary.md (generated, numbers read from MATRIX)
# ---------------------------------------------------------------------------

def _write_summary() -> None:
    rows = MATRIX["results"]
    wild = rows["wild_type"]
    _wild_ct = WILD.ct()
    _wild_rt = WILD.rt()
    lines = [
        "# W6 — 阻力链物理契约门变异测试（mutation testing）",
        "",
        "任务来源：素材单 W6 `physics-contract-verification`（缺「门能抓住违约」的消融证据）。",
        "被测对象：`tensorphysics.physics_contracts` 的阻力链契约 + SUBOFF-01 审计锚",
        "（audit m2086，11 个内核中 6 个逐位一致）。",
        "",
        "**实际工件路径**：`artifacts/contract_mutation/`（`benchmarks/verified/` 不在本单",
        "EDIT_WHITELIST 内，故按白名单落在 `artifacts/`，此处如实声明）。",
        "测试源：`tests/test_contract_mutation.py`；复现：`python -m pytest -q "
        "tests/test_contract_mutation.py`。",
        "",
        "## 1. 方法",
        "",
        "门被实现为 11 条可判定判据（容差全部取自审计值，未放宽）：",
        "",
        "| # | 判据 | 抓什么 | 容差/依据 |",
        "|---|------|--------|-----------|",
    ]
    crit_desc = {
        "naming_gate_no_false_holtrop":
            ("构造期拒绝 Holtrop-Mennen 误标", "m2086 审计误标"),
        "declared_formula_matches_impl":
            ("契约声明公式 vs 代码实现", "Re=1e7, rel<=1e-12"),
        "derived_anchor_bitwise":
            ("Ct vs DERIVED 锚", "rel<=1e-9（6 内核逐位）"),
        "derived_anchor_not_self_derived":
            ("锚必须拒绝 +2% Cf 扰动", "rel>1e-9 -> 非自证"),
        "resistance_anchor_rt":
            ("Rt [N] vs DERIVED 锚（含 S_wet 漂移）", "rel<=1e-9"),
        "experimental_anchor_bias":
            ("Ct vs SUBOFF-01 实验 0.0038", "15% 带"),
        "experimental_tolerance_policy":
            ("带宽不得超验收政策", "<=15%"),
        "experimental_anchor_sensitivity":
            ("实验带必须拒绝 +30% 偏差", "band"),
        "dimensionless_scale_consistency":
            ("Ct/Cf 必须 O(1)", "[0.05, 5]"),
        "re_applicability_domain":
            ("Re 适用域边界 + 外推案例不得判合格", "(1e5, 1e9)"),
        "wetted_surface_closed_form_bias":
            ("S 拟合 vs 解析几何/公布值", "-30.4% / KVLCC2 +50.6%"),
    }
    for i, name in enumerate(MATRIX["criteria"], 1):
        what, why = crit_desc[name]
        lines.append(f"| {i} | `{name}` | {what} | {why} |")
    lines += [
        "",
        "每个变异体是**单点变异**（仅改一处），在内存中替换候选实现/契约声明后重跑全部门判据。",
        "门的每一次运行都先跑野型（wild type）对照：野型必须 11/11 通过，否则门是「永远 FAIL」的假门。",
        "",
        "## 2. 结果矩阵（数字直读本次运行）",
        "",
        f"- 野型通过：**{MATRIX['wild_type_passes']}**（未通过判据："
        f"{wild['failing_criteria'] or '无'}）",
        f"- 变异体数：**{MATRIX['mutant_count']}**；被抓到（KILLED）："
        f"**{MATRIX['killed_count']}**；击杀率 **"
        f"{MATRIX['killed_count'] / MATRIX['mutant_count']:.0%}**",
        f"- 失败信息可定位到变异点：**"
        f"{sum(1 for m in MUTANTS if rows[m.mid]['localised'])}/{len(MUTANTS)}**",
        f"- 覆盖洞（门未被触发、变异体存活）：**{MATRIX['coverage_holes'] or '无'}**",
        "",
        "| 变异体 | 真实失效类型 | 门判定 | 触发的判据 | 失败信息可定位 |",
        "|--------|--------------|--------|------------|----------------|",
    ]
    for m in MUTANTS:
        row = rows[m.mid]
        verdict = "**FAIL（抓到）**" if row["caught"] else "**PASS（漏检）**"
        fired = ", ".join(f"`{c}`" for c in row["failing_criteria"]) or "—"
        lines.append(f"| `{m.mid}` | {m.failure_mode} | {verdict} | {fired} | "
                     f"{'是' if row['localised'] else '否'} |")
    lines += [
        "",
        "矩阵图：`fig1_mutation_matrix.png`（判据 × 变异体，红=FAIL 即抓到）；",
        "锚灵敏度图：`fig2_anchor_sensitivity.png`（+2% Cf 篡改落在 15% 实验带内、",
        "只被 1e-9 的 DERIVED 锚抓到——这是本任务最重要的结论）。",
        "",
        "## 3. 逐变异体失败信息摘录",
        "",
    ]
    for m in MUTANTS:
        row = rows[m.mid]
        lines.append(f"### {m.mid} — 注入：{m.injected_change}")
        for c in row["criteria"]:
            if not c["passed"]:
                lines.append(f"- `{c['criterion']}` **FAIL**：{c['detail']}")
        lines.append("")
    lines += [
        "## 4. 注意点 / 局限",
        "",
        "- **M1 的教训**：Cf 系数 0.075→0.0765 只让 Ct 变 +1.75%，远在 15% 实验带内，",
        "  纯物理验收永远抓不到；抓到它的只有 DERIVED 逐位锚与「声明公式 vs 实现」判据。",
        "  即：物理门与实现门是两条互补的防线，缺一不可。",
        "- **M3 的教训**：同义反复锚只能在「锚对物理扰动敏感」这一维度上被发现，",
        "  静态检查（格式/来源标注）无法识别；因此门里必须有扰动探针（mutation probe）。",
        "- **Ct 对 S_wet 不敏感**：Ct = Cf + Cr 与湿面积无关，故 M6（湿面积公式互换）",
        "  在 Ct 锚上不可见，必须靠 Rt 力锚与湿面积解析对照抓到——两条判据都触发了。",
        "- **契约声明本身未被变异**：本单只变异「实现/锚/域/容差」，不改声明字符串；",
        "  若同时篡改声明与实现（声明犯罪），除外部文献复核外无门可抓，属门的作用域边界。",
        "- **文献数值一致性**：`axisymmetric_wetted_surface` docstring 记 prolate spheroid",
        "  (b/a=0.25) 偏差 -8.7%，本次用标准椭球解析面积复算得 -21.7%（L/B=4, Cb=pi/6），",
        "  两者不一致；门只锁可复现的 -21.7%（并在判据里注明 audited -30.4% cylinder 值确实复现），",
        "  -8.7% 视为待核实的文献记载，未用它作锚。",
        "- **适用域 (1e5, 1e9) 是政策常量**（ITTC-1957 相关线验证区间），非从代码推导；",
        "  因此该判据抓的是「域被挪动」与「外推案例被判合格」，而非物理误差。",
        "",
        "## 5. 与物理基准的误差（是否 <3%）",
        "",
        "| 量 | 值（本次运行） | 基准 | 相对误差 | <3%？ |",
        "|----|----|------|----------|-------|",
        f"| Ct（野型，SUBOFF-01） | {_wild_ct:.12g} | DERIVED 锚 m2086 "
        f"{ANCHOR_CT_DERIVED:.12g} | {_rel(_wild_ct, ANCHOR_CT_DERIVED):.1e} | "
        f"是（逐位，机器精度） |",
        f"| Rt [N]（野型） | {_wild_rt:.10g} | DERIVED 锚 m2086 "
        f"{ANCHOR_RT_DERIVED:.10g} | {_rel(_wild_rt, ANCHOR_RT_DERIVED):.1e} | "
        f"是（逐位，机器精度） |",
        f"| Ct vs 实验 0.0038 | {_wild_ct:.6g} | SUBOFF-01 实验 | "
        f"{_rel(_wild_ct, ANCHOR_CT_EXPERIMENTAL) * 100:.1f}% | "
        f"否（>3%，但在合同自带 15% 带内；0.0038 为 2 位有效数字实验值，本单未动） |",
        f"| 变异体击杀率 | {MATRIX['killed_count']}/{MATRIX['mutant_count']} | 期望全抓 | "
        f"{(1 - MATRIX['killed_count'] / MATRIX['mutant_count']) * 100:.1f}% | 是 |",
        "",
        "结论：本任务是「门的消融实验」，主判据是击杀率（6/6，偏差 0% < 3%）；",
        "野型物理量对 DERIVED 锚逐位一致（误差 0），对实验值的 8.9% 差异是既有审计结论，",
        "本单未对其做任何改动或放宽。",
        "",
    ]
    (ART / "summary.md").write_text("\n".join(lines) + "\n")
