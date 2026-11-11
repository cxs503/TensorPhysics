"""Backtest harness: does the selector keep its promises?

The selector promises three things per query: ``P(error <= epsilon) >= alpha``,
``P(cost <= C) >= beta`` and an interval with coverage ``>= interval_level``.
Those are claims about a *distribution*, so they can only be falsified against
realisations that the selector never saw.  This module supplies that missing
half: a **truth generator** whose parameters are declared independently of the
registry, plus the metric bookkeeping that turns realisations into rates.

Truth generator (``TruthSpec``) adds three failure mechanisms that a two-parameter
log-normal prior cannot represent:

``sigma_inflation``
    the registered shape parameter is optimistic by this factor (validation set
    smaller/narrower than the deployment population);
``jump_prob`` / ``jump_median``
    a fraction of hulls lands on a flow regime the fit never saw, multiplying the
    error by a factor (heavy tail: propeller--hull interaction, separation);
``regime_rules``
    an axis box in which the error is further scaled (heteroscedasticity: error
    growth outside the calibration envelope).

Latent draws are pre-generated per ``(query, model)`` pair so that all registry
variants (prior-only, pooled conformal, stratified conformal) see the *same*
realisations -- the comparison between configurations is paired, not merely
same-seed.

The truth generator is a verification instrument, not a physics model: it says
nothing about resistance, only about how badly a registered prior can be wrong.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .calibration import StratumSpec
from .model import ModelRegistry, ModelSpec
from .selection import Budget, MIN_COST, PASS, select_model
from .stats import wilson_interval

LATENT_DIM = 4  # z_error, u_jump, z_jump, z_cost


# ---------------------------------------------------------------------------
# query sampling
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AxisSpec:
    """Sampling span of one query axis (``log`` scale for Reynolds-like axes)."""

    lo: float
    hi: float
    scale: str = "linear"

    def __post_init__(self) -> None:
        if self.scale not in {"linear", "log"}:
            raise ValueError(f"unknown scale {self.scale!r}")
        if self.scale == "log" and self.lo <= 0.0:
            raise ValueError("log-scaled axis needs lo > 0")
        if self.lo >= self.hi:
            raise ValueError(f"axis span must satisfy lo < hi, got {self.lo} >= {self.hi}")

    def sample(self, rng, n: int) -> np.ndarray:
        if self.scale == "log":
            return np.exp(rng.uniform(math.log(self.lo), math.log(self.hi), n))
        return rng.uniform(self.lo, self.hi, n)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AxisSpec":
        if isinstance(data, Mapping):
            return cls(lo=float(data["lo"]), hi=float(data["hi"]), scale=str(data.get("scale", "linear")))
        lo, hi = data
        return cls(lo=float(lo), hi=float(hi))

    def to_dict(self) -> Dict[str, Any]:
        return {"lo": self.lo, "hi": self.hi, "scale": self.scale}


def sample_queries(axes: Mapping[str, AxisSpec],
                   n: int,
                   seed: int,
                   epsilon_range: Tuple[float, float] = (0.015, 0.20),
                   cost_range: Tuple[float, float] = (1.0, 1.0e6),
                   error_confidence: float = 0.95,
                   cost_confidence: float = 0.95,
                   interval_level: float = 0.90) -> List[Dict[str, Any]]:
    """Draw ``n`` independent queries: parameter point + accuracy/compute budget."""
    if n <= 0:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    names = list(axes)
    columns = {name: axes[name].sample(rng, n) for name in names}
    # log-uniform budgets: one decade of epsilon is as likely as the next
    eps = np.exp(rng.uniform(math.log(epsilon_range[0]), math.log(epsilon_range[1]), n))
    cost = np.exp(rng.uniform(math.log(cost_range[0]), math.log(cost_range[1]), n))
    queries = []
    for i in range(n):
        params = {name: float(columns[name][i]) for name in names}
        queries.append({
            "index": i,
            "params": params,
            "budget": Budget(epsilon=float(eps[i]), cost=float(cost[i]),
                             error_confidence=error_confidence,
                             cost_confidence=cost_confidence,
                             interval_level=interval_level),
        })
    return queries


# ---------------------------------------------------------------------------
# truth generator
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RegimeRule:
    """Multiplicative error/cost inflation inside an axis box."""

    name: str
    bounds: Mapping[str, Tuple[Optional[float], Optional[float]]]
    error_scale: float = 1.0
    cost_scale: float = 1.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RegimeRule":
        bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        for axis, span in dict(data.get("bounds", {})).items():
            if isinstance(span, Mapping):
                bounds[axis] = (span.get("lo"), span.get("hi"))
            else:
                lo, hi = span
                bounds[axis] = (lo, hi)
        return cls(name=str(data.get("name", "")), bounds=bounds,
                   error_scale=float(data.get("error_scale", 1.0)),
                   cost_scale=float(data.get("cost_scale", 1.0)))

    def matches(self, params: Mapping[str, float]) -> bool:
        for axis, (lo, hi) in self.bounds.items():
            if axis not in params:
                return False
            value = float(params[axis])
            if lo is not None and value < float(lo):
                return False
            if hi is not None and value > float(hi):
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "bounds": {k: list(v) for k, v in self.bounds.items()},
                "error_scale": self.error_scale, "cost_scale": self.cost_scale}


@dataclass(frozen=True)
class TruthSpec:
    """Declared truth for a backtest (see module docstring)."""

    label: str = "truth"
    sigma_inflation: float = 1.0
    jump_prob: float = 0.0
    jump_median: float = 2.2
    jump_sigma: float = 0.4
    cost_sigma_inflation: float = 1.0
    cost_median_inflation: float = 1.0
    regime_rules: Tuple[RegimeRule, ...] = ()

    def __post_init__(self) -> None:
        for value, name in ((self.sigma_inflation, "sigma_inflation"),
                            (self.cost_sigma_inflation, "cost_sigma_inflation"),
                            (self.cost_median_inflation, "cost_median_inflation")):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be > 0, got {value!r}")
        if not 0.0 <= self.jump_prob <= 1.0:
            raise ValueError("jump_prob must lie in [0, 1]")
        if self.jump_median <= 0.0 or self.jump_sigma < 0.0:
            raise ValueError("jump parameters must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "sigma_inflation": self.sigma_inflation,
                "jump_prob": self.jump_prob, "jump_median": self.jump_median,
                "jump_sigma": self.jump_sigma,
                "cost_sigma_inflation": self.cost_sigma_inflation,
                "cost_median_inflation": self.cost_median_inflation,
                "regime_rules": [r.to_dict() for r in self.regime_rules]}


@dataclass(frozen=True)
class TruthGenerator:
    """Turns (model, query, latent) into a realised error and a realised cost."""

    spec: TruthSpec

    def latents(self, n_queries: int, n_models: int, seed: int) -> np.ndarray:
        """Latent draws per ``(query, model)`` pair.

        Columns are ``(z_error, u_jump, z_jump, z_cost)``; ``u_jump`` is
        **uniform on [0, 1]** because it is compared against a probability.
        Drawing it from a normal instead silently fabricates a 50 % heavy-tail
        rate -- a defect caught by the prior-consistent control truth, which
        fell to 0.84 accuracy / 0.55 coverage where theory demands
        alpha = 0.95 / level = 0.90.
        """
        rng = np.random.default_rng(seed)
        out = rng.standard_normal((n_queries, n_models, LATENT_DIM))
        out[..., 1] = rng.random((n_queries, n_models))
        return out

    # -- helpers ---------------------------------------------------------
    def _regimes(self, params: Mapping[str, float]) -> Tuple[float, float]:
        err_scale, cost_scale = 1.0, 1.0
        for rule in self.spec.regime_rules:
            if rule.matches(params):
                err_scale *= rule.error_scale
                cost_scale *= rule.cost_scale
        return err_scale, cost_scale

    # -- realisation -----------------------------------------------------
    def error(self, model: ModelSpec, params: Mapping[str, float], latent: np.ndarray) -> float:
        z_err, u_jump, z_jump, _ = (float(v) for v in latent)
        prior = model.prior
        if prior is None:  # pragma: no cover - every registry model carries a prior
            raise ValueError(f"model {model.model_id} has no registered prior to distort")
        sigma = prior.sigma * self.spec.sigma_inflation
        value = prior.median * math.exp(sigma * z_err)
        if u_jump < self.spec.jump_prob:
            value *= math.exp(math.log(self.spec.jump_median) + self.spec.jump_sigma * z_jump)
        err_scale, _ = self._regimes(params)
        return value * err_scale

    def cost(self, model: ModelSpec, params: Mapping[str, float], latent: np.ndarray) -> float:
        _, _, _, z_cost = (float(v) for v in latent)
        sigma = model.cost.sigma * self.spec.cost_sigma_inflation
        median = model.cost.median_s * self.spec.cost_median_inflation
        value = median * math.exp(sigma * z_cost)
        _, cost_scale = self._regimes(params)
        return value * cost_scale


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------
@dataclass
class QueryRecord:
    index: int
    params: Dict[str, float]
    epsilon: float
    cost_budget: float
    status: str
    reason_code: str
    model_id: Optional[str]
    realised_error: Optional[float]
    realised_cost: Optional[float]
    error_upper: Optional[float]
    prob_error_le_epsilon: Optional[float]
    prob_cost_le_cost: Optional[float]
    stratum: str
    n_feasible: int
    advisory_fallback_model_id: Optional[str] = None
    advisory_realised_error: Optional[float] = None
    advisory_realised_cost: Optional[float] = None
    oracle_model_id: Optional[str] = None
    oracle_cost: Optional[float] = None
    cost_ratio_vs_oracle: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class BacktestResult:
    label: str
    objective: str
    truth: TruthSpec
    nominal_level: float
    error_confidence: float
    cost_confidence: float
    records: List[QueryRecord]

    # -- derived metrics -------------------------------------------------
    @property
    def recommended(self) -> List[QueryRecord]:
        return [r for r in self.records if r.status == PASS]

    def recommendation_histogram(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in self.recommended:
            counts[r.model_id] = counts.get(r.model_id, 0) + 1
        return dict(sorted(counts.items()))

    def fail_reasons(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in self.records:
            if r.status != PASS:
                counts[r.reason_code] = counts.get(r.reason_code, 0) + 1
        return dict(sorted(counts.items()))

    def metrics(self) -> Dict[str, Any]:
        rec = self.recommended
        n = len(self.records)
        if not rec:
            raise ValueError("backtest produced no recommendation; metrics undefined")
        realised_err = [r.realised_error for r in rec]
        realised_cost = [r.realised_cost for r in rec]
        eps = [r.epsilon for r in rec]
        cost_budget = [r.cost_budget for r in rec]
        bounds = [r.error_upper for r in rec]
        hits_err = sum(1 for e, b in zip(realised_err, eps) if e <= b)
        hits_cost = sum(1 for c, b in zip(realised_cost, cost_budget) if c <= b)
        hits_cov = sum(1 for e, b in zip(realised_err, bounds) if e <= b)
        by_stratum: Dict[str, Dict[str, Any]] = {}
        for name in sorted({r.stratum for r in rec}):
            sub = [r for r in rec if r.stratum == name]
            if not sub:
                continue
            by_stratum[name] = {
                "n": len(sub),
                "accuracy": sum(1 for r in sub if r.realised_error <= r.epsilon) / len(sub),
                "cost_efficiency": sum(1 for r in sub if r.realised_cost <= r.cost_budget) / len(sub),
                "coverage": sum(1 for r in sub if r.realised_error <= r.error_upper) / len(sub),
                "mean_bound": float(np.mean([r.error_upper for r in sub])),
                "mean_realised_error": float(np.mean([r.realised_error for r in sub])),
            }
        acc_lo, acc_hi = wilson_interval(hits_err, len(rec))
        cov_lo, cov_hi = wilson_interval(hits_cov, len(rec))
        cost_lo, cost_hi = wilson_interval(hits_cost, len(rec))
        worst_cov = min((v["coverage"] for v in by_stratum.values()), default=hits_cov / len(rec))
        ratios = [r.cost_ratio_vs_oracle for r in rec if r.cost_ratio_vs_oracle is not None]
        n_oracle = sum(1 for r in rec if r.oracle_model_id is not None)
        advisory = [r for r in self.records
                    if r.status != PASS and r.advisory_realised_error is not None]
        n_advisory = len(advisory)
        advisory_hits = sum(1 for r in advisory if r.advisory_realised_error <= r.epsilon)
        oracle_hits = sum(1 for r in rec if r.oracle_model_id is not None
                          and abs((r.cost_ratio_vs_oracle or 0.0) - 1.0) < 1e-12)
        return {
            "label": self.label,
            "objective": self.objective,
            "truth": self.truth.label,
            "n_queries": n,
            "n_recommended": len(rec),
            "recommendation_rate": len(rec) / n,
            "accuracy": hits_err / len(rec),
            "accuracy_wilson95": [acc_lo, acc_hi],
            "cost_efficiency": hits_cost / len(rec),
            "cost_efficiency_wilson95": [cost_lo, cost_hi],
            "coverage": hits_cov / len(rec),
            "coverage_wilson95": [cov_lo, cov_hi],
            "coverage_worst_stratum": worst_cov,
            "nominal_interval_level": self.nominal_level,
            "error_confidence": self.error_confidence,
            "cost_confidence": self.cost_confidence,
            "recommendation_histogram": self.recommendation_histogram(),
            "fail_reasons": self.fail_reasons(),
            "by_stratum": by_stratum,
            "mean_error_ratio_realised_over_predicted": (
                float(np.mean([r.realised_error / max(r.error_upper, 1e-15) for r in rec]))),
            "n_oracle_available": n_oracle,
            "oracle_optimal_rate": (oracle_hits / n_oracle) if n_oracle else None,
            "median_cost_ratio_vs_oracle": float(np.median(ratios)) if ratios else None,
            "p90_cost_ratio_vs_oracle": float(np.quantile(ratios, 0.9)) if ratios else None,
            "n_advisory_fallbacks": n_advisory,
            "advisory_accuracy": (advisory_hits / n_advisory) if n_advisory else None,
            "mean_bound": float(np.mean(bounds)),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"metrics": self.metrics(),
                "records": [r.to_dict() for r in self.records]}


def run_backtest(registry: ModelRegistry,
                 truth: TruthGenerator,
                 queries: Sequence[Mapping[str, Any]],
                 objective: str = MIN_COST,
                 label: str = "config",
                 latent_seed: int = 7,
                 strata: Sequence[StratumSpec] = ()) -> BacktestResult:
    """Run the selector over ``queries`` against ``truth`` and collect realisations."""
    if not queries:
        raise ValueError("run_backtest needs at least one query")
    model_ids = registry.ids()
    latents = truth.latents(len(queries), len(model_ids), latent_seed)
    index_of = {mid: i for i, mid in enumerate(model_ids)}
    records: List[QueryRecord] = []
    first = queries[0]["budget"]
    for qi, query in enumerate(queries):
        params = dict(query["params"])
        budget: Budget = query["budget"]
        selection = select_model(registry, params, budget, objective=objective)
        stratum = "__unstratified__"
        for spec in strata:
            if spec.matches(params):
                stratum = spec.name
                break
        if selection.status != PASS:
            advisory_id = selection.uncertainty.get("advisory_fallback_model_id")
            advisory_error = advisory_cost = None
            if advisory_id is not None:
                adv_model = registry.get(advisory_id)
                adv_latent = latents[qi, index_of[advisory_id]]
                advisory_error = truth.error(adv_model, params, adv_latent)
                advisory_cost = truth.cost(adv_model, params, adv_latent)
            records.append(QueryRecord(
                index=qi, params=params, epsilon=budget.epsilon, cost_budget=budget.cost,
                status=selection.status, reason_code=selection.reason_code, model_id=None,
                realised_error=None, realised_cost=None, error_upper=None,
                prob_error_le_epsilon=None, prob_cost_le_cost=None,
                stratum=stratum, n_feasible=int(selection.uncertainty.get("n_feasible", 0)),
                advisory_fallback_model_id=advisory_id,
                advisory_realised_error=advisory_error,
                advisory_realised_cost=advisory_cost))
            continue
        # realisations of *every* admissible model (latents are pre-drawn per pair,
        # so this costs nothing and gives the ex-post oracle for the same query)
        realised = {}
        for cand in selection.candidates:
            if not cand.admissible:
                continue
            cand_model = registry.get(cand.model_id)
            cand_latent = latents[qi, index_of[cand.model_id]]
            realised[cand.model_id] = (truth.error(cand_model, params, cand_latent),
                                       truth.cost(cand_model, params, cand_latent))
        mid = selection.model_id
        realised_error, realised_cost = realised[mid]
        oracle_pool = [(c, e, k) for c, (e, k) in realised.items()
                       if e <= budget.epsilon and k <= budget.cost]
        oracle_model_id = None
        oracle_cost = None
        ratio = None
        if oracle_pool:
            oracle_model_id, oracle_error, oracle_cost = min(oracle_pool, key=lambda row: row[2])
            ratio = realised_cost / oracle_cost if oracle_cost > 0 else None
        records.append(QueryRecord(
            index=qi, params=params, epsilon=budget.epsilon, cost_budget=budget.cost,
            status=selection.status, reason_code=selection.reason_code, model_id=mid,
            realised_error=realised_error, realised_cost=realised_cost,
            error_upper=float(selection.uncertainty["error_upper"]),
            prob_error_le_epsilon=selection.uncertainty["prob_error_le_epsilon"],
            prob_cost_le_cost=selection.uncertainty["prob_cost_le_cost"],
            stratum=stratum, n_feasible=int(selection.uncertainty.get("n_feasible", 0)),
            oracle_model_id=oracle_model_id, oracle_cost=oracle_cost,
            cost_ratio_vs_oracle=ratio))
    return BacktestResult(label=label, objective=objective, truth=truth.spec,
                          nominal_level=first.interval_level,
                          error_confidence=first.error_confidence,
                          cost_confidence=first.cost_confidence,
                          records=records)


def generate_validation_residuals(registry: ModelRegistry,
                                  truth: TruthGenerator,
                                  axes: Mapping[str, AxisSpec],
                                  strata: Sequence[StratumSpec] = (),
                                  n_per_stratum: int = 250,
                                  n_pooled_min: int = 400,
                                  seed: int = 11) -> Dict[str, Dict[str, List[float]]]:
    """Sample validation residuals per model (and per stratum) from the truth.

    This is the data a user would obtain from a validation campaign: error
    magnitudes of each model on hulls that are *not* the deployment queries
    (different seed), including the heavy tail and the regime inflation.  It is
    the only input of the recalibration -- the recalibration never sees
    ``TruthSpec``.
    """
    rng = np.random.default_rng(seed)
    names = list(axes)
    out: Dict[str, Dict[str, List[float]]] = {}
    for model in registry.models:
        groups: Dict[str, List[float]] = {s.name: [] for s in strata}
        pooled: List[float] = []
        attempts = 0
        max_attempts = 200 * (n_per_stratum + 1) * (len(strata) + 1)
        while attempts < max_attempts:
            attempts += 1
            if (len(pooled) >= n_pooled_min
                    and all(len(values) >= n_per_stratum for values in groups.values())):
                break
            params = {name: float(axes[name].sample(rng, 1)[0]) for name in names}
            if not model.domain.admits(params):
                continue
            latent = rng.standard_normal(LATENT_DIM)
            latent[1] = rng.random()  # jump indicator must be uniform (see TruthGenerator.latents)
            value = truth.error(model, params, latent)
            # the pooled sample is *every* accepted sample; the stratum samples are
            # the subsets.  Pooling only the untagged draws would silently starve
            # the pooled law when the strata cover the whole axis (fixed 2026-11-11).
            pooled.append(value)
            for spec in strata:
                if spec.matches(params):
                    groups[spec.name].append(value)
                    break
        all_values = list(pooled)
        for values in groups.values():
            all_values.extend(values)
        if len(all_values) < 30:
            raise RuntimeError(f"could not collect validation residuals for {model.model_id}")
        payload = {"__pooled__": all_values}
        payload.update({k: v for k, v in groups.items() if len(v) >= 2})
        out[model.model_id] = payload
    return out