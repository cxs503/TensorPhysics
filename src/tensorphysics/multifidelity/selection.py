"""Budget-constrained multi-fidelity model selection with explicit uncertainty.

Given a query point (operating/geometry parameters), an accuracy budget
``epsilon`` and a compute budget ``C``, the selector solves one of two problems
over the *admissible* models of a :class:`~tensorphysics.multifidelity.model
.ModelRegistry`:

``min_cost``    ``argmin E[cost]  s.t.  P(error <= epsilon) >= alpha  and  P(cost <= C) >= beta``
``min_error``   ``argmin median(error)  s.t.  P(cost <= C) >= beta  and  P(error <= epsilon) >= alpha``

Both constraints are *probabilistic*: the answer to a query is a distribution,
not a number, and a selector that compared medians would silently violate the
budget half of the time.  ``alpha``/``beta`` are therefore part of the contract.

Refusal is a first-class outcome.  ``Selection.status`` is ``PASS`` or ``FAIL``
with one of the reason codes in :data:`REASON_CODES`; in particular an
``epsilon`` below the accuracy floor of the admissible set (the smallest
achievable median error) is refused up front -- that is the negative control
that keeps the engine from recommending a model that cannot possibly meet the
budget ("1 % resistance on a Holtrop regression" is not a recommendation, it is
wishful thinking).

The uncertainty block returned with every recommendation carries

* ``error_upper`` -- one-sided bound at ``interval_level`` (prior quantile, or
  conformal order statistic when the registry is calibrated);
* ``prob_error_le_epsilon``, ``prob_cost_le_cost`` -- the two margins the
  decision was made on;
* ``cost_upper`` at ``interval_level`` and ``expected_cost``;
* ``slack`` -- how much accuracy budget is left over, i.e. a cheap measure of how
  risky the choice is at the operating point.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .model import ModelRegistry
from .stats import _check_level

PASS = "PASS"
FAIL = "FAIL"

MIN_COST = "min_cost"
MIN_ERROR = "min_error"
OBJECTIVES = (MIN_COST, MIN_ERROR)

REASON_CODES = {
    "OK": "a model satisfies every constraint",
    "NO_ADMISSIBLE_MODEL": "every registered model is outside its validity domain at this query",
    "EPSILON_BELOW_MODEL_FLOOR": "accuracy budget tighter than the best admissible model's median error",
    "COST_BELOW_MODEL_FLOOR": "compute budget below the cheapest admissible model's median cost",
    "NO_MODEL_MEETS_ERROR_BUDGET": "no admissible model reaches the required confidence on epsilon",
    "NO_MODEL_MEETS_COST_BUDGET": "no admissible model reaches the required confidence on the cost budget",
    "NO_MODEL_MEETS_BOTH_BUDGETS": "each admissible model violates one of the two budgets",
    "UNKNOWN_PROBES": "internal: an ErrorView could not be probed",
}


@dataclass(frozen=True)
class Budget:
    """User-side requirement.  Either budget may be ``None`` (unconstrained)."""

    epsilon: Optional[float] = None
    cost: Optional[float] = None
    error_confidence: float = 0.95
    cost_confidence: float = 0.95
    interval_level: float = 0.90

    def __post_init__(self) -> None:
        _check_level(self.error_confidence, "error_confidence")
        _check_level(self.cost_confidence, "cost_confidence")
        _check_level(self.interval_level, "interval_level")
        if self.epsilon is not None and (not math.isfinite(self.epsilon) or self.epsilon <= 0.0):
            raise ValueError(f"epsilon must be finite and > 0, got {self.epsilon!r}")
        if self.cost is not None and (not math.isfinite(self.cost) or self.cost <= 0.0):
            raise ValueError(f"cost must be finite and > 0, got {self.cost!r}")
        if self.epsilon is None and self.cost is None:
            raise ValueError("at least one of epsilon / cost must be given")

    def to_dict(self) -> Dict[str, Any]:
        return {"epsilon": self.epsilon, "cost": self.cost,
                "error_confidence": self.error_confidence,
                "cost_confidence": self.cost_confidence,
                "interval_level": self.interval_level}


@dataclass
class Candidate:
    """Per-model evaluation of one query (returned even when it is not chosen)."""

    model_id: str
    label: str
    fidelity_level: int
    admissible: bool
    domain_violation: Optional[Dict[str, Any]]
    prob_error_le_epsilon: Optional[float]
    prob_cost_le_cost: Optional[float]
    median_error: float
    expected_cost: float
    cost_median: float
    error_upper: float
    cost_upper: float
    meets_error: bool
    meets_cost: bool
    objective_value: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "label": self.label,
            "fidelity_level": self.fidelity_level,
            "admissible": self.admissible,
            "domain_violation": self.domain_violation,
            "prob_error_le_epsilon": self.prob_error_le_epsilon,
            "prob_cost_le_cost": self.prob_cost_le_cost,
            "median_error": self.median_error,
            "expected_cost": self.expected_cost,
            "cost_median": self.cost_median,
            "error_upper": self.error_upper,
            "cost_upper": self.cost_upper,
            "meets_error": self.meets_error,
            "meets_cost": self.meets_cost,
            "objective_value": self.objective_value,
        }


@dataclass
class Selection:
    status: str
    objective: str
    budget: Budget
    params: Dict[str, float]
    model_id: Optional[str]
    reason_code: str
    reason: str
    uncertainty: Dict[str, Any] = field(default_factory=dict)
    candidates: List[Candidate] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == PASS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "objective": self.objective,
            "budget": self.budget.to_dict(),
            "params": dict(self.params),
            "model_id": self.model_id,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "uncertainty": dict(self.uncertainty),
            "candidates": [c.to_dict() for c in self.candidates],
        }


def _prob(view, x: float, params: Mapping[str, float]) -> float:
    try:
        return float(view.prob_le(x, params))
    except TypeError:  # an ErrorView that ignores params
        return float(view.prob_le(x))


def _upper(view, level: float, params: Mapping[str, float]) -> float:
    try:
        return float(view.upper_bound(level, params))
    except TypeError:
        return float(view.upper_bound(level))


def evaluate_candidates(registry: ModelRegistry, params: Mapping[str, float],
                        budget: Budget) -> List[Candidate]:
    """Score every registered model at the query (admissible or not)."""
    candidates: List[Candidate] = []
    for model in registry.models:
        violation = model.domain.violations(params)
        admissible = violation is None
        if budget.epsilon is None:
            p_err = None
            meets_err = True
        else:
            p_err = _prob(model.error, budget.epsilon, params)
            meets_err = p_err >= budget.error_confidence
        if budget.cost is None:
            p_cost = None
            meets_cost = True
        else:
            p_cost = model.cost.prob_le(budget.cost)
            meets_cost = p_cost >= budget.cost_confidence
        median_err = _upper(model.error, 0.5, params)
        expected_cost = model.cost.mean_s
        cost_median = model.cost.median_s
        err_upper = _upper(model.error, budget.interval_level, params)
        cost_upper = model.cost.quantile(budget.interval_level)
        candidates.append(
            Candidate(
                model_id=model.model_id,
                label=model.label or model.model_id,
                fidelity_level=model.fidelity_level,
                admissible=admissible,
                domain_violation=violation,
                prob_error_le_epsilon=p_err,
                prob_cost_le_cost=p_cost,
                median_error=median_err,
                expected_cost=expected_cost,
                cost_median=cost_median,
                error_upper=err_upper,
                cost_upper=cost_upper,
                meets_error=bool(meets_err),
                meets_cost=bool(meets_cost),
                objective_value=expected_cost,  # refined by the caller
            )
        )
    return candidates


def select_model(registry: ModelRegistry,
                 params: Mapping[str, float],
                 budget: Budget,
                 objective: str = MIN_COST) -> Selection:
    """Solve the budget-constrained selection problem for one query."""
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
    params = {str(k): float(v) for k, v in params.items()}
    candidates = evaluate_candidates(registry, params, budget)
    admissible = [c for c in candidates if c.admissible]

    def _fail(code: str, reason: str) -> Selection:
        # A refusal is still actionable: name the best-effort candidate, but label
        # it advisory so that no downstream consumer mistakes it for a guarantee.
        advisory = None
        if admissible:
            advisory = min(admissible, key=lambda c: c.median_error).model_id
        return Selection(status=FAIL, objective=objective, budget=budget, params=dict(params),
                         model_id=None, reason_code=code, reason=reason,
                         uncertainty={"reason_code": code,
                                      "advisory_fallback_model_id": advisory,
                                      "advisory_only": True,
                                      "n_admissible": len(admissible)},
                         candidates=candidates)

    if not admissible:
        violated = {c.model_id: (c.domain_violation or {}).get("axis") for c in candidates}
        return _fail("NO_ADMISSIBLE_MODEL",
                     f"no registered model admits this query; violated axes: {violated}")

    if budget.epsilon is not None:
        floor = min(_upper(m.error, 0.5, params) for m in registry.admissible(params))
        if budget.epsilon < floor:
            return _fail(
                "EPSILON_BELOW_MODEL_FLOOR",
                f"epsilon={budget.epsilon:.4g} is below the accuracy floor {floor:.4g} "
                f"(median error of the best admissible model); no model can meet it",
            )
    if budget.cost is not None:
        cost_floor = min(m.cost.median_s for m in registry.admissible(params))
        if budget.cost < cost_floor:
            return _fail(
                "COST_BELOW_MODEL_FLOOR",
                f"cost budget {budget.cost:.4g} s is below the cheapest admissible model "
                f"median cost {cost_floor:.4g} s",
            )

    feasible = [c for c in admissible if c.meets_error and c.meets_cost]
    if not feasible:
        err_fail = [c for c in admissible if not c.meets_error]
        cost_fail = [c for c in admissible if not c.meets_cost]
        if len(err_fail) == len(admissible) and budget.epsilon is not None:
            code = "NO_MODEL_MEETS_ERROR_BUDGET"
            reason = (f"no admissible model reaches P(error <= {budget.epsilon:g}) >= "
                      f"{budget.error_confidence:g}")
        elif len(cost_fail) == len(admissible) and budget.cost is not None:
            code = "NO_MODEL_MEETS_COST_BUDGET"
            reason = (f"no admissible model reaches P(cost <= {budget.cost:g}) >= "
                      f"{budget.cost_confidence:g}")
        else:
            code = "NO_MODEL_MEETS_BOTH_BUDGETS"
            reason = ("each admissible model fails at least one budget: "
                      + ", ".join(f"{c.model_id}(err={c.prob_error_le_epsilon}, "
                                  f"cost={c.prob_cost_le_cost})" for c in admissible))
        return _fail(code, reason)

    if objective == MIN_COST:
        key = lambda c: (round(c.expected_cost, 12), c.median_error)  # noqa: E731
    else:
        key = lambda c: (round(c.median_error, 15), c.expected_cost)  # noqa: E731
    chosen = min(feasible, key=key)
    model = registry.get(chosen.model_id)

    err_upper = _upper(model.error, budget.interval_level, params)
    cost_upper = model.cost.quantile(budget.interval_level)
    slack = None if budget.epsilon is None else budget.epsilon - chosen.median_error
    uncertainty = {
        "reason_code": "OK",
        "interval_level": budget.interval_level,
        "error_upper": err_upper,
        "error_median": chosen.median_error,
        "prob_error_le_epsilon": chosen.prob_error_le_epsilon,
        "prob_cost_le_cost": chosen.prob_cost_le_cost,
        "expected_cost": chosen.expected_cost,
        "cost_median": chosen.cost_median,
        "cost_upper": cost_upper,
        "risk_error_above_epsilon": (None if budget.epsilon is None
                                     else 1.0 - (chosen.prob_error_le_epsilon or 0.0)),
        "slack_error": slack,
        "n_admissible": len(admissible),
        "n_feasible": len(feasible),
        "error_view": getattr(model.error, "kind", "unknown"),
    }
    return Selection(status=PASS, objective=objective, budget=budget, params=dict(params),
                     model_id=chosen.model_id, reason_code="OK",
                     reason=f"{chosen.model_id} selected by {objective} with "
                            f"P(error<={budget.epsilon})={chosen.prob_error_le_epsilon}",
                     uncertainty=uncertainty, candidates=candidates)


def pareto_front(registry: ModelRegistry, params: Mapping[str, float],
                 use_upper: bool = False) -> List[str]:
    """Non-dominated model ids on ``(expected cost, error)`` at the query.

    With ``use_upper=True`` the comparison uses the ``interval_level`` upper
    bounds instead of the medians, which is the honest front when the decision
    has to survive the uncertainty rather than the central value.
    """
    pts: List[Tuple[str, float, float]] = []
    for model in registry.admissible(params):
        level = 0.90
        err = _upper(model.error, level, params) if use_upper else _upper(model.error, 0.5, params)
        cost = model.cost.quantile(level) if use_upper else model.cost.mean_s
        pts.append((model.model_id, cost, err))
    front: List[str] = []
    for mid, cost, err in pts:
        dominated = any(
            (c2 <= cost and e2 <= err) and (c2 < cost or e2 < err)
            for mid2, c2, e2 in pts if mid2 != mid
        )
        if not dominated:
            front.append(mid)
    order = {m.model_id: i for i, m in enumerate(registry.models)}
    return sorted(front, key=lambda mid: order[mid])


def frontier_curve(registry: ModelRegistry, params: Mapping[str, float]) -> List[Dict[str, Any]]:
    """Sort the Pareto set by cost so it can be plotted as a staircase."""
    front = set(pareto_front(registry, params))
    rows = []
    for model in registry.models:
        if model.model_id not in front:
            continue
        rows.append({
            "model_id": model.model_id,
            "fidelity_level": model.fidelity_level,
            "cost_median_s": model.cost.median_s,
            "cost_upper_s": model.cost.quantile(0.90),
            "error_median": _upper(model.error, 0.5, params),
            "error_upper": _upper(model.error, 0.90, params),
        })
    return sorted(rows, key=lambda r: r["cost_median_s"])