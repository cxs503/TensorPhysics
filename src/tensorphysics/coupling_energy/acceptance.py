"""Machine-checkable acceptance gates for the P40W3-N09 coupling-energy ledger.

The scenarios of :mod:`tensorphysics.coupling_energy.scenarios` *produce* ledger
residuals; this module *decides* whether those residuals certify the contract.
It exists because the gates used to live inline in the campaign runner, which
made them impossible to audit after the fact: a stored record could not be
re-gated, a partial suite could not be told apart from a failed one, and a
missing control arm silently looked identical to a passing one.

Four properties are enforced here, and they are the reason the module is not a
formatter over the runner's booleans:

``frozen criterion set``
    Every gate is a :class:`Criterion` with an id, a human requirement string, a
    threshold role and a stated rationale.  The criteria are serialised to
    canonical JSON and hashed into :data:`CRITERIA_DIGEST`, which the runner
    writes into the artifact: a record can therefore be re-checked against the
    exact criterion set it claims to satisfy, and a later edit of a threshold
    shows up as a digest change (criterion drift) instead of a silent re-score.

``three-valued outcomes``
    ``PASS`` / ``FAIL`` / ``WITHHELD``.  A gate whose input is absent (no control
    arm, no profile samples, reduced matrix) returns ``WITHHELD`` -- never
    ``PASS``.  This is the difference between "the contract holds" and "we did
    not look", and it is what keeps a smoke run from being quoted as evidence.

``mandatory vs evidence``
    ``gate="mandatory"`` criteria decide the verdict; ``gate="evidence"``
    criteria are reported but cannot overturn it (they describe integrator
    properties, not contract violations).  The distinction is recorded per
    criterion, so a reader can see exactly which statement carries the verdict.

``re-gating a stored record``
    :func:`evaluate_record` takes the dict written to
    ``artifacts/.../verification.json`` and reproduces the verdict without
    re-running the solvers, and :func:`audit_drift` compares two such reports.

Verdict convention (aligned with the platform evidence grades)
-------------------------------------------------------------
``REAL``      every mandatory criterion passed and its input was present.
``FAIL``      at least one mandatory criterion failed on measured data.
``WITHHELD``  no mandatory failure, but at least one mandatory criterion could
              not be evaluated (partial matrix, missing control, missing
              metric): the honest statement is "insufficient evidence".

The per-coupling physics claims themselves live in
:mod:`tensorphysics.coupling_energy.scenarios`; this module deliberately holds
no numerical model of its own.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from .scenarios import RATIOS, SCHEMES

__all__ = [
    "CRITERIA",
    "CRITERIA_DIGEST",
    "AcceptanceError",
    "AcceptanceReport",
    "Criterion",
    "CriterionOutcome",
    "EXPECTED_MATRIX_RUNS",
    "STATUSES",
    "assert_accepted",
    "audit_drift",
    "criteria_digest",
    "evaluate",
    "evaluate_record",
    "format_table",
    "measurements",
]

#: outcomes a single criterion can take.
STATUSES: tuple[str, ...] = ("PASS", "FAIL", "WITHHELD")

#: the full campaign matrix: 4 ratios x 3 schemes x 3 couplings, plus one
#: independent (falsification) control per coupling and ratio.
EXPECTED_MATRIX_RUNS = len(RATIOS) * len(SCHEMES) * 3 + len(RATIOS) * 3


class AcceptanceError(RuntimeError):
    """Raised when a record cannot be certified (``FAIL`` or ``WITHHELD``)."""


# ---------------------------------------------------------------------------
# the frozen criterion set
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Criterion:
    """One pre-registered acceptance statement.

    ``requirement`` is the frozen, human-readable threshold; ``baseline`` states
    where the reference number comes from (analytic solution, literature, or the
    measured round-off of an exactly conservative scheme), so that a reader can
    audit the choice of threshold and not only the measurement.
    """

    cid: str
    coupling: str
    quantity: str
    requirement: str
    gate: str                     # "mandatory" | "evidence"
    rationale: str
    baseline: str = ""
    # The numbers live *in the criterion*, not in the checking code: otherwise
    # the requirement string and the enforced threshold could drift apart and
    # the digest would attest a threshold that is not the one being applied.
    threshold: float | None = None      # value must stay below this
    factor: float | None = None         # control/shared ratio must exceed this
    params: dict | None = None          # structured thresholds (order scan, matrix)

    def as_dict(self) -> dict:
        return {
            "cid": self.cid,
            "coupling": self.coupling,
            "quantity": self.quantity,
            "requirement": self.requirement,
            "gate": self.gate,
            "rationale": self.rationale,
            "baseline": self.baseline,
            "threshold": self.threshold,
            "factor": self.factor,
            "params": self.params,
        }


CRITERIA: tuple[Criterion, ...] = (
    Criterion(
        cid="thermal_energy_conservation",
        coupling="thermal",
        quantity="thermal_eps_shared_max",
        requirement="eps_max < 1e-12 over every shared LBM<->FEM run",
        gate="mandatory",
        threshold=1e-12,
        rationale="an antisymmetric exchange is structurally conservative; any "
                  "residual above round-off means the two sides booked "
                  "different numbers",
        baseline="machine round-off of float64 accumulation (measured 2e-13)",
    ),
    Criterion(
        cid="thermal_profile_accuracy",
        coupling="thermal",
        quantity="thermal_profile_rel_l2_r1",
        requirement="rel-L2 temperature error < 3 % vs exact 1-D slab (r = 1)",
        gate="mandatory",
        threshold=0.03,
        rationale="conservation alone does not certify the coupled physics: the "
                  "profile must match the analytic solution",
        baseline="exact series solution of T_t = alpha T_yy, Dirichlet/adiabatic",
    ),
    Criterion(
        cid="thermal_falsification_ratio",
        coupling="thermal",
        quantity="thermal_falsification_ratio",
        requirement="independent-flux control residual > 1e3 x shared residual",
        gate="mandatory",
        factor=1e3,
        rationale="an instrument that cannot be broken has no teeth: replacing "
                  "the shared flux by the solid's own Fourier estimate must "
                  "destroy the contract",
        baseline="control arm (exchange_mode='independent')",
    ),
    Criterion(
        cid="mechanical_momentum_conservation",
        coupling="mechanical",
        quantity="mechanical_momentum_shared_max",
        requirement="max |P(t)-P(0)| / |P(0)| < 1e-10 (unsupported plate)",
        gate="mandatory",
        threshold=1e-10,
        rationale="Newton's third law at the contact interface; no external "
                  "force acts on the free plate, so total momentum is exact",
        baseline="zero external impulse (free body)",
    ),
    Criterion(
        cid="mechanical_contact_engaged",
        coupling="mechanical",
        quantity="mechanical_min_max_overlap",
        requirement="max contact overlap > 0 in every shared mechanical run",
        gate="mandatory",
        threshold=0.0,
        rationale="momentum conservation is vacuous before contact: a run in which "
                  "the particles never touch the plate must not be quoted as "
                  "evidence for Newton's third law",
        baseline="impact scenario, max overlap ~1.2e-4 m over the full run",
    ),
    Criterion(
        cid="mechanical_energy_convergence",
        coupling="mechanical",
        quantity="mechanical_energy_drift",
        requirement="energy drift monotone decreasing in r and < 5 % at r = 8",
        gate="evidence",
        threshold=0.05,
        rationale="the remaining energy drift is the first-order property of the "
                  "explicit DEM sub-stepping, not a coupling defect; recorded as "
                  "evidence so a regression is still visible",
        baseline="drift halves per doubling of r (measured 0.155 -> 0.017)",
    ),
    Criterion(
        cid="mechanical_falsification_ratio",
        coupling="mechanical",
        quantity="mechanical_momentum_falsification_ratio",
        requirement="lagged-reaction control momentum residual > 1e2 x shared",
        gate="mandatory",
        factor=1e2,
        rationale="substituting the plate's own reaction (frozen particle state) "
                  "must break reciprocity detectably",
        baseline="control arm (exchange_mode='independent')",
    ),
    Criterion(
        cid="three_field_energy_chain",
        coupling="three_field",
        quantity="three_field_eps_shared_max",
        requirement="eps_max < 1e-3 over every shared LBM<->FEM<->DEM run",
        gate="mandatory",
        threshold=1e-3,
        rationale="the chain budget must close across three solvers, dominated "
                  "by the mechanical sub-system's integrator drift",
        baseline="total stored energy minus external input, snapshot form",
    ),
    Criterion(
        cid="three_field_thermal_subchain",
        coupling="three_field",
        quantity="three_field_thermal_eps_max",
        requirement="thermal sub-chain residual < 1e-12 inside the chain",
        gate="mandatory",
        threshold=1e-12,
        rationale="isolates the thermal exchange from the mechanical drift: the "
                  "energy chain may be dominated by the mechanical part, but the "
                  "thermal part itself must stay at round-off",
        baseline="round-off of the thermal ledger alone",
    ),
    Criterion(
        cid="three_field_physics_engaged",
        coupling="three_field",
        quantity="three_field_physics",
        requirement="every shared chain run has external heat input > 0 and "
                    "contact energy > 0",
        gate="mandatory",
        threshold=0.0,
        rationale="same vacuity guard for the three-field chain: a residual of "
                  "1e-16 is meaningless if neither the thermal nor the mechanical "
                  "sub-system was ever exercised",
        baseline="hot lower face + ice impact, Q_in > 0 and E_contact > 0",
    ),
    Criterion(
        cid="three_field_falsification_ratio",
        coupling="three_field",
        quantity="three_field_falsification_ratio",
        requirement="independent-flux control residual > 1e2 x shared",
        gate="mandatory",
        factor=1e2,
        rationale="the chain gate must also be able to fail",
        baseline="control arm (exchange_mode='independent')",
    ),
    Criterion(
        cid="interpolation_order_scan",
        coupling="instrument",
        quantity="interpolation_orders",
        requirement="measured exchange order 1.0 / 2.0 / 3.0 within "
                    "(-0.3, +0.3) / (-0.3, +0.3) / (-0.5, +0.5)",
        gate="mandatory",
        params={"targets": {"zoh": [1.0, 0.3], "linear": [2.0, 0.3],
                            "quadratic": [3.0, 0.5]}},
        rationale="the declared zero/first/second-order reconstruction must be "
                  "observed, otherwise sub-cycling error cannot be interpreted",
        baseline="analytic signal, h = 0.05 -> 0.0125 Richardson scan",
    ),
    Criterion(
        cid="plate_first_frequency",
        coupling="instrument",
        quantity="plate_frequency_rel_error",
        requirement="FEM clamped-free axial frequency within 3 % of c/(4L)",
        gate="mandatory",
        threshold=0.03,
        rationale="a solver-level sanity check: the coupling study is worthless "
                  "if the underlying FEM plate is wrong",
        baseline="Euler-Bernoulli axial mode, f = c/(4L) = 1863.39 Hz",
    ),
    Criterion(
        cid="matrix_completeness",
        coupling="matrix",
        quantity="n_runs",
        requirement="48 runs = 4 ratios x 3 schemes x 3 couplings + 4 x 3 controls",
        gate="mandatory",
        params={"expected": EXPECTED_MATRIX_RUNS},
        rationale="a partial matrix can still yield PASS values; it cannot certify "
                  "the pre-registered experiment, so it is withheld, not passed",
        baseline="RATIOS x SCHEMES of tensorphysics.coupling_energy.scenarios",
    ),
    Criterion(
        cid="falsification_arm_coverage",
        coupling="matrix",
        quantity="control_coverage",
        requirement="each of the three couplings contributes >= 1 independent run",
        gate="mandatory",
        params={"couplings": ["thermal", "mechanical", "three_field"]},
        rationale="Falsification is a property of the whole record: a coupling "
                  "without its control arm cannot claim the gate has teeth",
        baseline="one control per coupling and ratio",
    ),
)


def criteria_digest(criteria: Sequence[Criterion] = CRITERIA) -> str:
    """SHA-256 of the canonical criterion set (provenance of the thresholds)."""
    payload = json.dumps([c.as_dict() for c in criteria], sort_keys=True,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


#: digest of :data:`CRITERIA`; the runner stamps it into every record.
CRITERIA_DIGEST = criteria_digest()


# ---------------------------------------------------------------------------
# measurement extraction (pure functions over the runner's dictionaries)
# ---------------------------------------------------------------------------
def _mode(suite: Mapping[str, Any], name: str, mode: str) -> list[dict]:
    return [r for r in (suite.get(name) or ()) if r.get("exchange_mode") == mode]


def _series_max(runs: Iterable[dict], key: str) -> float | None:
    vals: list[float] = []
    for run in runs:
        series = run.get("series") or {}
        values = series.get(key)
        if not values:
            continue
        vals.extend(float(v) for v in values if not math.isnan(float(v)))
    return max(vals) if vals else None


def _residual_max(runs: Iterable[dict]) -> float | None:
    vals = [float(r["residual_max"]) for r in runs
            if r.get("residual_max") is not None
            and not math.isnan(float(r["residual_max"]))]
    return max(vals) if vals else None


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None:
        return None
    return num / max(abs(den), 1e-300)


def measurements(suite: Mapping[str, Any], extra: Mapping[str, Any] | None = None,
                 ) -> dict:
    """Extract every quantity the criteria refer to from a suite result.

    ``suite`` is the ``{"thermal": [...], "mechanical": [...],
    "three_field": [...]}`` mapping produced by
    :func:`tensorphysics.coupling_energy.scenarios.run_suite` (the ``as_dict``
    form, i.e. what the runner stores in ``verification.json``).  ``extra``
    carries the two solver-level checks (``order_scan``, ``plate_frequency``).

    Absent inputs are reported as ``None`` -- they are *not* defaulted, because
    a defaulted measurement is indistinguishable from a measured one.
    """
    extra = dict(extra or {})

    t_sh = _mode(suite, "thermal", "shared")
    t_ct = _mode(suite, "thermal", "independent")
    m_sh = _mode(suite, "mechanical", "shared")
    m_ct = _mode(suite, "mechanical", "independent")
    f_sh = _mode(suite, "three_field", "shared")
    f_ct = _mode(suite, "three_field", "independent")

    thermal_eps = _residual_max(t_sh)
    thermal_ctrl = _residual_max(t_ct)

    # accuracy is quoted at the finest exercised coupling (r = 1): sub-cycling
    # error is a documented, monotone function of r and not part of this gate.
    prof = [float(r["metrics"]["profile_rel_l2_error"]) for r in t_sh
            if r.get("ratio") == 1
            and r.get("metrics")
            and r["metrics"].get("profile_rel_l2_error") is not None]
    thermal_profile = min(prof) if prof else None

    mom = max((v for v in (_series_max([r], "momentum_residual") for r in m_sh)
               if v is not None), default=None)
    mom_ctrl = max((v for v in (_series_max([r], "momentum_residual") for r in m_ct)
                    if v is not None), default=None)

    drift = {int(r["ratio"]): float(r["residual_max"]) for r in m_sh
             if r.get("scheme") == "linear" and r.get("ratio") is not None
             and r.get("residual_max") is not None}

    three_eps = _residual_max(f_sh)
    three_ctrl = _residual_max(f_ct)
    three_thermal = [float(r["metrics"]["thermal_residual_max"]) for r in f_sh
                     if r.get("metrics")
                     and r["metrics"].get("thermal_residual_max") is not None]
    three_thermal_eps = max(three_thermal) if three_thermal else None

    orders = extra.get("order_scan") or {}
    orders = {k: (float(v) if v is not None else None) for k, v in orders.items()}
    freq = extra.get("plate_frequency") or {}
    freq_err = freq.get("rel_error")

    overlaps = []
    for r in m_sh:
        ov = (r.get("series") or {}).get("overlap")
        if ov:
            overlaps.append(max(float(v) for v in ov))
    min_overlap = min(overlaps) if overlaps else None

    chain = {}
    for r in f_sh:
        met = r.get("metrics") or {}
        if met.get("q_external_total_J") is not None:
            chain.setdefault("q_external_total_J", []).append(float(met["q_external_total_J"]))
        if met.get("max_contact_energy_J") is not None:
            chain.setdefault("max_contact_energy_J", []).append(
                float(met["max_contact_energy_J"]))
    three_physics = ({k: min(v) for k, v in chain.items()} if chain else None)

    n_runs = sum(len(suite.get(k) or ()) for k in ("thermal", "mechanical", "three_field"))
    coverage = {k: len(_mode(suite, k, "independent"))
                for k in ("thermal", "mechanical", "three_field")}

    return {
        "thermal_eps_shared_max": thermal_eps,
        "thermal_eps_control_min": (min(float(r["residual_max"]) for r in t_ct)
                                    if t_ct else None),
        "thermal_profile_rel_l2_r1": thermal_profile,
        "thermal_falsification_ratio": _ratio(thermal_ctrl, thermal_eps),
        "mechanical_momentum_shared_max": mom,
        "mechanical_momentum_control_max": mom_ctrl,
        "mechanical_momentum_falsification_ratio": _ratio(mom_ctrl, mom),
        "mechanical_energy_drift": drift,
        "three_field_eps_shared_max": three_eps,
        "three_field_thermal_eps_max": three_thermal_eps,
        "three_field_falsification_ratio": _ratio(three_ctrl, three_eps),
        "interpolation_orders": orders,
        "plate_frequency_rel_error": (float(freq_err) if freq_err is not None else None),
        "mechanical_min_max_overlap": min_overlap,
        "three_field_physics": three_physics,
        "n_runs": n_runs,
        "n_runs_expected": EXPECTED_MATRIX_RUNS,
        "control_coverage": coverage,
        "ratio_coverage": sorted({int(r["ratio"]) for r in m_sh if r.get("ratio")}),
        "n_runs_thermal": len(suite.get("thermal") or ()),
        "n_runs_mechanical": len(suite.get("mechanical") or ()),
        "n_runs_three_field": len(suite.get("three_field") or ()),
    }


# ---------------------------------------------------------------------------
# checks: one function per criterion id, returning (actual, ok|None, note)
# ---------------------------------------------------------------------------
#: type of a check: (measurements, criterion) -> (actual, ok | None, note)
Check = Callable[[Mapping[str, Any], Criterion], tuple]


def _below(m: Mapping[str, Any], crit: Criterion) -> tuple:
    value = m.get(crit.quantity)
    if value is None:
        return None, None, "input absent: the criterion cannot be evaluated"
    return value, bool(value < crit.threshold), f"threshold {crit.threshold:g}"


def _falsification(m: Mapping[str, Any], crit: Criterion) -> tuple:
    value = m.get(crit.quantity)
    if value is None:
        return None, None, "control arm absent: no falsification evidence"
    return value, bool(value > crit.factor), f"control must be > {crit.factor:g} x shared"


def _chk_thermal_conservation(m, crit) -> tuple:
    return _below(m, crit)


def _chk_thermal_profile(m, crit) -> tuple:
    value, ok, note = _below(m, crit)
    if value is None:
        note = ("no profile samples at r = 1: run with sample_times, otherwise the "
                "accuracy claim cannot be separated from conservation")
    return value, ok, note


def _chk_thermal_falsification(m, crit) -> tuple:
    return _falsification(m, crit)


def _chk_mechanical_momentum(m, crit) -> tuple:
    return _below(m, crit)


def _chk_mechanical_contact(m, crit) -> tuple:
    value = m.get(crit.quantity)
    if value is None:
        return None, None, "no shared mechanical run carries an overlap series"
    if value > crit.threshold:
        return value, True, "contact occurred in every shared run"
    # no contact == the mechanism was never exercised: that is missing evidence,
    # not a violated threshold (mandatory, so the record still cannot be REAL)
    return value, None, ("no contact in at least one shared run: the momentum gate "
                         "is vacuous, so the run cannot certify the contract")


def _chk_mechanical_energy(m, crit) -> tuple:
    drift = m.get(crit.quantity) or {}
    missing = [r for r in RATIOS if r not in drift]
    if missing:
        return ({str(k): v for k, v in sorted(drift.items())}, None,
                f"ratios missing from the shared linear sweep: {missing}")
    ratios = sorted(drift)
    monotone = all(drift[b] <= drift[a] for a, b in zip(ratios, ratios[1:]))
    ok = bool(monotone and drift[max(RATIOS)] < crit.threshold)
    note = (f"drift must decrease monotonically with r and stay < "
            f"{crit.threshold:g} at r = {max(RATIOS)}" if not ok
            else "first-order integrator drift, decreasing with r")
    return {str(k): v for k, v in sorted(drift.items())}, ok, note


def _chk_mechanical_falsification(m, crit) -> tuple:
    return _falsification(m, crit)


def _chk_three_field(m, crit) -> tuple:
    return _below(m, crit)


def _chk_three_field_thermal(m, crit) -> tuple:
    return _below(m, crit)


def _chk_three_field_physics(m, crit) -> tuple:
    value = m.get(crit.quantity)
    if not value:
        return None, None, "shared chain runs do not report Q_in / contact energy"
    q_in = value.get("q_external_total_J")
    e_contact = value.get("max_contact_energy_J")
    if q_in is None or e_contact is None:
        return value, None, "Q_in or contact energy missing from the chain metrics"
    if q_in > crit.threshold and e_contact > crit.threshold:
        return value, True, "both physics exercised"
    return value, None, ("thermal or mechanical sub-system never activated: the "
                         "chain residual is vacuous, so the run cannot certify it")


def _chk_three_field_falsification(m, crit) -> tuple:
    return _falsification(m, crit)


def _chk_interpolation_orders(m, crit) -> tuple:
    orders = m.get(crit.quantity) or {}
    target = (crit.params or {}).get("targets", {})
    if not target or any(k not in orders or orders[k] is None for k in target):
        return orders or None, None, "order scan incomplete"
    bad = [k for k, (want, tol) in target.items() if abs(orders[k] - want) > tol]
    note = ("all three declared orders observed" if not bad
            else f"declared order not observed for {bad}")
    return {k: orders[k] for k in target}, not bad, note


def _chk_plate_frequency(m, crit) -> tuple:
    value, ok, note = _below(m, crit)
    if value is None:
        note = "no FEM frequency check in the record"
    return value, ok, note


def _chk_matrix_completeness(m, crit) -> tuple:
    n = m.get(crit.quantity)
    expected = (crit.params or {}).get("expected", EXPECTED_MATRIX_RUNS)
    if not n:
        return n, None, "no runs in the record"
    if n < expected:
        return n, None, f"partial suite: {n} of {expected} runs (withheld, not failed)"
    return n, bool(n == expected), f"{expected} runs expected, {n} found"


def _chk_control_coverage(m, crit) -> tuple:
    cov = m.get(crit.quantity) or {}
    if not cov:
        return None, None, "no runs"
    missing = [k for k, v in cov.items() if not v]
    if missing:
        # a record built without controls did not look, it did not fail
        return dict(cov), None, f"couplings without a control arm: {missing}"
    return dict(cov), True, "each coupling has its control arm"


_CHECKS: dict[str, Check] = {
    "thermal_energy_conservation": _chk_thermal_conservation,
    "thermal_profile_accuracy": _chk_thermal_profile,
    "thermal_falsification_ratio": _chk_thermal_falsification,
    "mechanical_momentum_conservation": _chk_mechanical_momentum,
    "mechanical_contact_engaged": _chk_mechanical_contact,
    "mechanical_energy_convergence": _chk_mechanical_energy,
    "mechanical_falsification_ratio": _chk_mechanical_falsification,
    "three_field_energy_chain": _chk_three_field,
    "three_field_thermal_subchain": _chk_three_field_thermal,
    "three_field_physics_engaged": _chk_three_field_physics,
    "three_field_falsification_ratio": _chk_three_field_falsification,
    "interpolation_order_scan": _chk_interpolation_orders,
    "plate_first_frequency": _chk_plate_frequency,
    "matrix_completeness": _chk_matrix_completeness,
    "falsification_arm_coverage": _chk_control_coverage,
}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CriterionOutcome:
    """Outcome of one criterion: the frozen statement plus what was measured."""

    cid: str
    coupling: str
    gate: str
    requirement: str
    status: str
    actual: Any = None
    note: str = ""
    baseline: str = ""

    def as_dict(self) -> dict:
        return {
            "cid": self.cid,
            "coupling": self.coupling,
            "gate": self.gate,
            "requirement": self.requirement,
            "status": self.status,
            "actual": self.actual,
            "note": self.note,
            "baseline": self.baseline,
        }


@dataclass
class AcceptanceReport:
    """The full verdict: per-criterion outcomes plus the frozen-set digest."""

    outcomes: dict[str, CriterionOutcome]
    digest: str
    measurements: dict = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        mandatory = [o for o in self.outcomes.values() if o.gate == "mandatory"]
        if any(o.status == "FAIL" for o in mandatory):
            return "FAIL"
        if any(o.status == "WITHHELD" for o in mandatory):
            return "WITHHELD"
        return "REAL"

    @property
    def failures(self) -> list[CriterionOutcome]:
        return [o for o in self.outcomes.values() if o.status == "FAIL"]

    @property
    def withheld(self) -> list[CriterionOutcome]:
        return [o for o in self.outcomes.values() if o.status == "WITHHELD"]

    @property
    def passed(self) -> bool:
        return self.verdict == "REAL"

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "criteria_digest": self.digest,
            "n_pass": sum(1 for o in self.outcomes.values() if o.status == "PASS"),
            "n_fail": len(self.failures),
            "n_withheld": len(self.withheld),
            "criteria": {cid: o.as_dict() for cid, o in self.outcomes.items()},
            "measurements": self.measurements,
        }


def evaluate(suite: Mapping[str, Any], extra: Mapping[str, Any] | None = None,
             criteria: Sequence[Criterion] = CRITERIA) -> AcceptanceReport:
    """Gate a suite result against the frozen criterion set.

    Never raises on missing data: an unevaluable criterion is ``WITHHELD``.
    """
    m = measurements(suite, extra)
    outcomes: dict[str, CriterionOutcome] = {}
    for crit in criteria:
        check = _CHECKS.get(crit.cid)
        if check is None:                       # pragma: no cover - guarded below
            raise AcceptanceError(f"criterion {crit.cid!r} has no check function, "
                                  f"so it cannot be gated")
        actual, ok, note = check(m, crit)
        status = "WITHHELD" if ok is None else ("PASS" if ok else "FAIL")
        outcomes[crit.cid] = CriterionOutcome(
            cid=crit.cid, coupling=crit.coupling, gate=crit.gate,
            requirement=crit.requirement, status=status, actual=actual,
            note=note, baseline=crit.baseline,
        )
    missing = [c.cid for c in criteria if c.cid not in _CHECKS]
    if missing:                                 # pragma: no cover - guarded above
        raise AcceptanceError(f"criteria without checks: {missing}")
    return AcceptanceReport(outcomes=outcomes, digest=criteria_digest(criteria),
                            measurements=m)


def evaluate_record(record: Mapping[str, Any],
                    criteria: Sequence[Criterion] = CRITERIA) -> AcceptanceReport:
    """Re-gate a stored ``verification.json`` record without re-running anything.

    Accepts both the runner's layout (``{"runs": {...}, "order_scan": ...,
    "plate_first_frequency": ...}``) and a bare suite mapping, so a record can be
    audited from the artifact alone.
    """
    if "runs" in record and isinstance(record["runs"], Mapping):
        suite = record["runs"]
    else:
        suite = {k: record.get(k) or () for k in ("thermal", "mechanical", "three_field")}
    extra = {"order_scan": record.get("order_scan"),
             # the artifact spells out the solver-level check; the criterion
             # quantity is the frequency error extracted from it
             "plate_frequency": record.get("plate_first_frequency")
             or record.get("plate_frequency")}
    return evaluate(suite, {k: v for k, v in extra.items() if v is not None},
                    criteria=criteria)


def assert_accepted(report: AcceptanceReport) -> AcceptanceReport:
    """Raise :class:`AcceptanceError` unless the verdict is ``REAL``."""
    if report.verdict == "REAL":
        return report
    lines = [f"verdict={report.verdict}"]
    for o in report.failures:
        lines.append(f"  FAIL {o.cid}: {o.requirement} -> actual={o.actual!r} ({o.note})")
    for o in report.withheld:
        lines.append(f"  WITHHELD {o.cid}: {o.requirement} ({o.note})")
    raise AcceptanceError("\n".join(lines))


def audit_drift(reference: AcceptanceReport, candidate: AcceptanceReport,
                rel_tol: float = 0.5, digest_only: bool = False) -> list[dict]:
    """Compare two reports: criterion-set drift, status flips, numeric drift.

    Returns one dict per detected difference; an empty list means the candidate
    reproduces the reference.  ``digest_only`` restricts the audit to the frozen
    criterion set (useful when the two runs use different matrices).
    """
    drifts: list[dict] = []
    if reference.digest != candidate.digest:
        drifts.append({"kind": "criteria_digest",
                       "reference": reference.digest, "candidate": candidate.digest})
    if digest_only:
        return drifts
    for cid, ref in reference.outcomes.items():
        cand = candidate.outcomes.get(cid)
        if cand is None:
            drifts.append({"kind": "criterion_missing", "cid": cid})
            continue
        if ref.status != cand.status:
            drifts.append({"kind": "status", "cid": cid,
                           "reference": ref.status, "candidate": cand.status})
        a, b = ref.actual, cand.actual
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) \
                and not isinstance(a, bool) and not isinstance(b, bool):
            denom = max(abs(a), 1e-300)
            if abs(b - a) / denom > rel_tol:
                drifts.append({"kind": "value", "cid": cid,
                               "reference": a, "candidate": b,
                               "rel_change": abs(b - a) / denom})
    return drifts


def format_table(report: AcceptanceReport) -> str:
    """Human-readable gate table (used by the runner, the CLI and the docs)."""
    width = max(len(o.cid) for o in report.outcomes.values())
    lines = [f"verdict: {report.verdict}   criteria_digest: {report.digest[:16]}..."]
    for o in report.outcomes.values():
        actual = o.actual
        if isinstance(actual, float):
            actual_s = f"{actual:.4g}"
        elif isinstance(actual, dict) and actual and all(
                isinstance(v, float) for v in actual.values()):
            actual_s = "{" + ", ".join(f"{k}: {v:.4g}" for k, v in actual.items()) + "}"
        else:
            actual_s = str(actual)
        lines.append(f"  {o.cid:<{width}}  {o.status:<8} {actual_s:<28} {o.requirement}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m tensorphysics.coupling_energy.acceptance RECORD.json``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__.split("Verdict convention")[0].strip())
        print(f"\nfrozen criteria: {len(CRITERIA)}  digest: {CRITERIA_DIGEST}")
        return 0
    path = argv[0]
    with open(path, "r", encoding="utf-8") as fh:
        record = json.load(fh)
    report = evaluate_record(record)
    print(format_table(report))
    return 0 if report.verdict == "REAL" else 1


if __name__ == "__main__":                      # pragma: no cover
    raise SystemExit(main())