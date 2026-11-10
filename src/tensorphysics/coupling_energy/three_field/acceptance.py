"""Frozen acceptance criteria for the P40W4-N09 three-field energy contract.

Design rules (same discipline as ``tensorphysics.coupling_energy.acceptance``):

* every gate is a :class:`Criterion` carrying the *numeric* threshold it
  enforces, and the whole set is hashed into ``CRITERIA_DIGEST`` so a stored
  ``verification.json`` can be re-judged against the criteria it claims;
* outcomes are three-valued: ``PASS``, ``FAIL`` (measured violation) and
  ``WITHHELD`` (input absent -- never a silent pass);
* every vacuity guard is a criterion: a run in which the two physics never
  touched must not be able to reach ``REAL``.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

__all__ = [
    "CRITERIA",
    "CRITERIA_DIGEST",
    "Criterion",
    "CriterionOutcome",
    "AcceptanceReport",
    "criteria_digest",
    "evaluate",
    "format_table",
    "audit_record",
]


class Outcome(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WITHHELD = "WITHHELD"


@dataclass(frozen=True)
class Criterion:
    """One frozen acceptance gate."""

    id: str
    quantity: str
    requirement: str
    threshold: float
    comparison: str  # "below" | "above"
    gate_class: str  # primary | order | continuity | control | vacuity
    rationale: str

    def as_dict(self) -> dict:
        return {
            "id": self.id, "quantity": self.quantity,
            "requirement": self.requirement, "threshold": self.threshold,
            "comparison": self.comparison, "gate_class": self.gate_class,
            "rationale": self.rationale,
        }

    def judged(self, value: float | None) -> Outcome:
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
            return Outcome.WITHHELD
        if self.comparison == "below":
            return Outcome.PASS if value < self.threshold else Outcome.FAIL
        if self.comparison == "above":
            return Outcome.PASS if value > self.threshold else Outcome.FAIL
        raise ValueError(f"unknown comparison {self.comparison!r}")


CRITERIA: tuple[Criterion, ...] = (
    Criterion(
        id="total_energy_residual",
        quantity="residual = |E_total(t_end) - E_total(0)| / sum_c |E_c(0)|",
        requirement="three-field runtime ledger closes to better than 1e-3 relative "
                    "to the initial energy at the production reconstruction degree p=3",
        threshold=1.0e-3, comparison="below", gate_class="primary",
        rationale="campaign criterion (1): the coupled run may not create or destroy "
                  "more than 0.1 % of the initial energy",
    ),
    Criterion(
        id="interpolation_order_gain",
        quantity="residual(p=1) / residual(p=2)",
        requirement="quadratic interface reconstruction reduces the ledger residual "
                    "by at least 4x relative to linear reconstruction",
        threshold=4.0, comparison="above", gate_class="order",
        rationale="campaign criterion (2): the residual must be attributable to the "
                  "reconstruction order, not to the integrators (negative control)",
    ),
    Criterion(
        id="flux_continuity",
        quantity="||q_solid - q_fluid||_2 / ||q_fluid||_2 at the coupled face",
        requirement="two independent discretisations of the interface heat flux agree "
                    "to better than 2 %",
        threshold=2.0e-2, comparison="below", gate_class="continuity",
        rationale="campaign criterion (3): the fluid's measured wall balance and the "
                  "panel's one-sided surface flux must describe the same flux",
    ),
    Criterion(
        id="negative_control_conservation",
        quantity="max single-field residual with all coupling terms switched off",
        requirement="each uncoupled field conserves its own energy to better than 1e-6",
        threshold=1.0e-6, comparison="below", gate_class="control",
        rationale="campaign criterion (4): a coupled residual may not be blamed on the "
                  "integrators when the uncoupled fields conserve at round-off level",
    ),
    Criterion(
        id="contact_reciprocity",
        quantity="|sum f_on_particles + sum f_on_nodes| / sum |f|",
        requirement="the contact contract is reciprocal to round-off (Newton's third law)",
        threshold=1.0e-12, comparison="below", gate_class="continuity",
        rationale="contract clause of fem_dem_contact_force: an interface that books two "
                  "different force magnitudes creates momentum",
    ),
    Criterion(
        id="vacuity_contact_engaged",
        quantity="max contact overlap [m]",
        requirement="the contact crossing must actually be exercised (overlap > 0)",
        threshold=0.0, comparison="above", gate_class="vacuity",
        rationale="a zero-force impact has an exactly zero energy error, which reads as "
                  "perfect conservation; the guard makes that record unreachable",
    ),
    Criterion(
        id="vacuity_thermal_engaged",
        quantity="peak measured interface heat rate [W]",
        requirement="the thermal crossing must carry a non-zero flux",
        threshold=0.0, comparison="above", gate_class="vacuity",
        rationale="an isothermal run closes the thermal ledger trivially",
    ),
    Criterion(
        id="vacuity_drag_engaged",
        quantity="energy removed from the particles by drag [J]",
        requirement="the drag crossing must transfer a non-zero amount",
        threshold=0.0, comparison="above", gate_class="vacuity",
        rationale="a stationary particle dissipates nothing and would hide a broken "
                  "DEM->LBM channel",
    ),
    Criterion(
        id="reconstruction_operator_order",
        quantity="quadrature error of the p=1 / p=2 reconstruction of one smooth "
                 "interface signal (operator-level, no solver in the loop)",
        requirement="the reconstruction operator itself must show the second-order "
                    "gain (>= 4x) on a smooth signal",
        threshold=4.0, comparison="above", gate_class="order",
        rationale="separates 'the operator is first order' from 'the coupled residual "
                  "is not order limited': if this passes and the coupled gain fails, "
                  "the residual is dominated by a non-truncation source, which is the "
                  "finding of this node",
    ),
    Criterion(
        id="drag_transfer_balance",
        quantity="|Q_delivered_to_fluid - Q_removed_from_particles| / Q_removed",
        requirement="the reconstructed drag heat delivered to the fluid agrees with the "
                    "measured particle loss to better than 5 %",
        threshold=5.0e-2, comparison="below", gate_class="continuity",
        rationale="contract clause of dem_lbm_drag: the crossing is reconstructed, so a "
                  "tolerance (not reciprocity) is the right check",
    ),
)


def criteria_digest(criteria: Sequence[Criterion] = CRITERIA) -> str:
    payload = json.dumps([c.as_dict() for c in criteria], sort_keys=True,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


CRITERIA_DIGEST = criteria_digest()


@dataclass(frozen=True)
class CriterionOutcome:
    id: str
    value: float | None
    threshold: float
    comparison: str
    gate_class: str
    outcome: Outcome
    requirement: str

    def as_dict(self) -> dict:
        return {
            "id": self.id, "value": self.value, "threshold": self.threshold,
            "comparison": self.comparison, "gate_class": self.gate_class,
            "outcome": self.outcome.value, "requirement": self.requirement,
        }


@dataclass
class AcceptanceReport:
    measurements: Mapping[str, float | None]
    outcomes: dict[str, CriterionOutcome]
    criteria_digest: str

    @property
    def verdict(self) -> str:
        values = {o.outcome for o in self.outcomes.values()}
        if Outcome.FAIL in values:
            return "FAIL"
        if Outcome.WITHHELD in values:
            return "WITHHELD"
        return "REAL"

    def counts(self) -> dict[str, int]:
        out = {"PASS": 0, "FAIL": 0, "WITHHELD": 0}
        for o in self.outcomes.values():
            out[o.outcome.value] += 1
        return out

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "criteria_digest": self.criteria_digest,
            "counts": self.counts(),
            "measurements": {k: v for k, v in self.measurements.items()},
            "outcomes": {k: v.as_dict() for k, v in self.outcomes.items()},
        }


#: measurement extractor: name -> (callable(suite) -> float | None)
_MEASUREMENTS: tuple[tuple[str, Callable[[Mapping[str, Any]], float | None]], ...] = (
    ("total_energy_residual",
     lambda s: _get(s, "runs", str(s.get("production_degree", 2)), "residual")),
    ("interpolation_order_gain",
     lambda s: _get(s, "gains", "p1_over_p2")),
    ("flux_continuity",
     lambda s: _get(s, "runs", str(s.get("production_degree", 2)),
                    "diagnostics", "flux_continuity_l2_settled")),
    ("negative_control_conservation",
     lambda s: _get(s, "negative_control", "max")),
    ("contact_reciprocity",
     lambda s: _get(s, "runs", str(s.get("production_degree", 2)),
                    "diagnostics", "momentum_reciprocity_error")),
    ("vacuity_contact_engaged",
     lambda s: _get(s, "runs", str(s.get("production_degree", 2)),
                    "diagnostics", "contact_max_overlap")),
    ("vacuity_thermal_engaged",
     lambda s: _get(s, "runs", str(s.get("production_degree", 2)),
                    "diagnostics", "flux_fluid_peak")),
    ("vacuity_drag_engaged",
     lambda s: _get(s, "runs", str(s.get("production_degree", 2)),
                    "diagnostics", "drag_source_energy")),
    ("drag_transfer_balance",
     lambda s: _abs_or_none(_get(s, "runs", str(s.get("production_degree", 2)),
                                 "diagnostics", "drag_transfer_mismatch"))),
    ("reconstruction_operator_order",
     lambda s: _get(s, "operator_order", "p1_over_p2")),
)


def _get(suite: Mapping[str, Any], *path: str) -> float | None:
    cur: Any = suite
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return None
        cur = cur[key]
    if isinstance(cur, bool):
        return 1.0 if cur else 0.0
    if isinstance(cur, (int, float)):
        return float(cur)
    return None


def _abs_or_none(value: float | None) -> float | None:
    return None if value is None else abs(value)


def evaluate(suite: Mapping[str, Any],
             criteria: Sequence[Criterion] = CRITERIA) -> AcceptanceReport:
    """Judge a suite record against the frozen criteria."""
    measurements: dict[str, float | None] = {}
    for name, extractor in _MEASUREMENTS:
        try:
            measurements[name] = extractor(suite)
        except (KeyError, TypeError):
            measurements[name] = None
    outcomes: dict[str, CriterionOutcome] = {}
    for crit in criteria:
        value = measurements.get(crit.id)
        outcomes[crit.id] = CriterionOutcome(
            id=crit.id, value=value, threshold=crit.threshold,
            comparison=crit.comparison, gate_class=crit.gate_class,
            outcome=crit.judged(value), requirement=crit.requirement,
        )
    return AcceptanceReport(measurements=measurements, outcomes=outcomes,
                            criteria_digest=criteria_digest(criteria))


def format_table(report: AcceptanceReport) -> str:
    lines = [
        f"{'criterion':32s} {'value':>14s} {'threshold':>12s} {'cmp':>6s}  verdict",
        "-" * 80,
    ]
    for key, out in report.outcomes.items():
        value = "n/a" if out.value is None else f"{out.value:.6e}"
        lines.append(
            f"{key:32s} {value:>14s} {out.threshold:>12.3e} {out.comparison:>6s}  "
            f"{out.outcome.value}"
        )
    counts = report.counts()
    lines.append("-" * 80)
    lines.append(
        f"verdict={report.verdict}  PASS={counts['PASS']} FAIL={counts['FAIL']} "
        f"WITHHELD={counts['WITHHELD']}  digest={report.criteria_digest[:16]}"
    )
    return "\n".join(lines)


def audit_record(record: Mapping[str, Any]) -> AcceptanceReport:
    """Re-gate a stored ``verification.json`` (does not trust stored booleans)."""
    if "suite" not in record:
        raise KeyError("record has no 'suite' section to re-gate")
    report = evaluate(record["suite"])
    stored = record.get("criteria_digest")
    if stored is not None and stored != report.criteria_digest:
        raise ValueError(
            "criteria digest mismatch: the record was judged against a different "
            f"criterion set ({stored[:16]} != {report.criteria_digest[:16]})"
        )
    return report