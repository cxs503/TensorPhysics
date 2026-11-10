"""Machine-checkable acceptance gates for the LBM/DEM momentum audit.

The module holds **no physics**: it decides whether the numbers produced by
:mod:`tensorphysics.lbm_dem_momentum.coupling` certify the contract.  Two design
choices are deliberate:

**Frozen, hashed criteria.**  Every gate is a :class:`Criterion` with an id, a
threshold and a rationale; the set is serialised canonically and hashed into
:data:`CRITERIA_DIGEST`, so a stored record can be re-gated against the exact
criterion set it claims to satisfy and a later threshold edit shows up as a
digest change instead of a silent re-score.

**Three-valued outcomes.**  ``PASS`` / ``FAIL`` / ``WITHHELD``.  A gate whose
input is missing returns ``WITHHELD`` -- never ``PASS``.  "We did not look" and
"the contract holds" must not look the same.

The two residuals are deliberately *not* interchangeable statements:

* ``momentum_residual`` is **structural** (the exchange is adjoint; round-off),
* ``energy_residual`` is a **discretisation** statement (percent level).

The criterion ``residual-separation`` makes that hierarchy explicit and
machine-checkable: the energy residual must dominate the momentum residual by at
least six decades, otherwise the momentum bookkeeping is not structural.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

__all__ = [
    "CRITERIA",
    "CRITERIA_DIGEST",
    "COUPLED_CASES",
    "AcceptanceReport",
    "Criterion",
    "Outcome",
    "criteria_digest",
    "evaluate",
    "verdict_of",
    "PASS",
    "FAIL",
    "WITHHELD",
]

COUPLED_CASES = ("floe_drift_decay", "ice_field_shear", "ship_ice_push")

PASS = "PASS"
FAIL = "FAIL"
WITHHELD = "WITHHELD"


@dataclass(frozen=True)
class Criterion:
    id: str
    requirement: str
    quantity: str
    comparison: str          # "<=" or ">="
    threshold: float
    role: str                # "mandatory" | "evidence"
    rationale: str

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "requirement": self.requirement,
            "quantity": self.quantity,
            "comparison": self.comparison,
            "threshold": self.threshold,
            "role": self.role,
            "rationale": self.rationale,
        }


CRITERIA: tuple = (
    Criterion(
        "mom-conservation",
        "system momentum change equals the accumulated external impulse in every coupled case",
        "momentum_residual_max", "<=", 1e-10, "mandatory",
        "the momentum-exchange boundary is adjoint by construction, so the only "
        "error left is round-off; a value above 1e-10 means the bookkeeping leaks "
        "(node conversion, wall tally or forced-collision term)",
    ),
    Criterion(
        "energy-conservation",
        "system energy change equals external work minus viscous dissipation",
        "energy_residual_max", "<=", 5e-2, "mandatory",
        "2 mu S:S is an independent continuum estimate of the sink, calibrated on "
        "a smooth reference flow at the same grid and relaxation time",
    ),
    Criterion(
        "residual-separation",
        "energy residual dominates the momentum residual by >= 6 decades",
        "residual_separation", ">=", 1e6, "mandatory",
        "the momentum contract is structural (round-off) while the energy contract "
        "is a discretisation statement (percent); if the two residuals were of the "
        "same magnitude the momentum bookkeeping would not be adjoint, it would "
        "merely be as accurate as the energy one",
    ),
    Criterion(
        "momentum-order",
        "grid refinement of the integrated interface impulse shows order p >= 1",
        "momentum_order_p", ">=", 1.0, "mandatory",
        "the impulse a user reads off the simulation is not the adjoint tally; its "
        "discretisation error must vanish at least linearly with dx for the error "
        "bound to be usable",
    ),
    Criterion(
        "negative-control",
        "DEM off (pure LBM) residual is below 1e-12",
        "control_residual", "<=", 1e-12, "mandatory",
        "a control arm with no interface at all proves the coupled residual is not "
        "an artefact of the instrument",
    ),
    Criterion(
        "interface-power",
        "interface energy flux (lattice) matches interface mechanical work F.u",
        "consistency_residual_max", "<=", 5e-2, "evidence",
        "momentum and energy bookkeeping of the same interface should agree at the "
        "discretisation level; reported as evidence because a staircase boundary "
        "makes the lattice energy flux noisier than the momentum tally",
    ),
    Criterion(
        "mass-drift",
        "lattice mass drift stays below 1% (moving bounce back is not mass exact)",
        "mass_drift_max", "<=", 1e-2, "evidence",
        "moving-wall bounce back exchanges mass at O(Ma^2); reported so the momentum "
        "claim is not silently tied to a mass-conserving scheme",
    ),
    Criterion(
        "mach-envelope",
        "lattice Mach number stays below 0.1",
        "max_mach", "<=", 0.1, "evidence",
        "beyond Ma = 0.1 the O(Ma^2) compressibility error contaminates both "
        "budgets and the 5% energy tolerance would be meaningless",
    ),
)


def criteria_digest(criteria=CRITERIA) -> str:
    payload = json.dumps([c.as_dict() for c in criteria], sort_keys=True,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


CRITERIA_DIGEST = criteria_digest()


@dataclass
class Outcome:
    criterion: Criterion
    value: float | None
    outcome: str
    detail: str = ""

    def as_dict(self) -> dict:
        return {
            "id": self.criterion.id,
            "value": None if self.value is None else float(self.value),
            "outcome": self.outcome,
            "threshold": self.criterion.threshold,
            "comparison": self.criterion.comparison,
            "role": self.criterion.role,
            "detail": self.detail,
        }


@dataclass
class AcceptanceReport:
    verdict: str
    digest: str
    outcomes: list = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "criteria_digest": self.digest,
            "outcomes": [o.as_dict() for o in self.outcomes],
            "detail": self.detail,
        }

    def failed(self) -> list:
        return [o for o in self.outcomes if o.outcome == FAIL]

    def withheld(self) -> list:
        return [o for o in self.outcomes if o.outcome == WITHHELD]


def verdict_of(outcomes) -> str:
    outcomes = list(outcomes)
    mandatory = [o for o in outcomes if o.criterion.role == "mandatory"]
    if any(o.outcome == FAIL for o in mandatory):
        return FAIL
    if any(o.outcome == WITHHELD for o in mandatory):
        return WITHHELD
    return PASS


def _cases(record: dict) -> dict:
    return record.get("cases", {}) or {}


def _max_over_cases(record: dict, key: str):
    cases = _cases(record)
    try:
        return max(abs(cases[n][key]) for n in COUPLED_CASES)
    except (KeyError, TypeError):
        return None


def _measure(record: dict, quantity: str):
    if quantity == "momentum_residual_max":
        return _max_over_cases(record, "momentum_residual")
    if quantity == "energy_residual_max":
        return _max_over_cases(record, "energy_residual")
    if quantity == "consistency_residual_max":
        return _max_over_cases(record, "consistency_residual")
    if quantity == "max_mach":
        return _max_over_cases(record, "max_mach")
    if quantity == "mass_drift_max":
        return _max_over_cases(record, "mass_drift")
    try:
        if quantity == "momentum_order_p":
            return float(record["grid"]["order_p"])
        if quantity == "control_residual":
            return float(_cases(record)["negative_control_lbm_only"]["momentum_residual"])
        if quantity == "residual_separation":
            mom = _max_over_cases(record, "momentum_residual")
            en = _max_over_cases(record, "energy_residual")
            if mom is None or en is None:
                return None
            return float(en / max(mom, 1e-300))
    except (KeyError, TypeError, ValueError):
        return None
    return None


def evaluate(record: dict) -> AcceptanceReport:
    """Gate a record of measurements against the frozen criterion set."""
    outcomes = []
    for crit in CRITERIA:
        value = _measure(record, crit.quantity)
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            outcomes.append(Outcome(crit, None, WITHHELD, "measurement absent"))
            continue
        ok = value <= crit.threshold if crit.comparison == "<=" else value >= crit.threshold
        detail = ""
        if crit.id == "residual-separation":
            mom = _max_over_cases(record, "momentum_residual")
            en = _max_over_cases(record, "energy_residual")
            detail = (f"max energy residual {en:.3e} vs max momentum residual {mom:.3e}"
                      if mom is not None and en is not None else "")
        if crit.id == "momentum-order":
            detail = (f"Richardson order p = {value:.3f} on the integrated interface "
                      f"impulse over dx = {record.get('grid', {}).get('dx', [])}")
        if crit.id == "energy-conservation":
            detail = "; ".join(
                f"{n}: {_cases(record)[n]['energy_residual']:.3f}"
                for n in COUPLED_CASES if n in _cases(record)
            )
        if crit.id == "mass-drift":
            detail = "; ".join(
                f"{n}: {_cases(record)[n]['mass_drift']:+.3f}"
                for n in COUPLED_CASES if n in _cases(record)
            )
        outcomes.append(Outcome(crit, value, PASS if ok else FAIL, detail))
    return AcceptanceReport(verdict=verdict_of(outcomes), digest=CRITERIA_DIGEST,
                            outcomes=outcomes,
                            detail={"case_count": len(_cases(record))})