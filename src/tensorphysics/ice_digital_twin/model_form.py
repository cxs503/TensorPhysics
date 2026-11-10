"""Model-form uncertainty by Bayesian model averaging over the structural ensemble.

Weights
-------
Each ensemble member (:data:`tensorphysics.ice_digital_twin.twin.FORMS`) is
calibrated on the same campaign.  Members are compared by their **leave-one-out
predictive score**

.. math::

    \\mathrm{lpd}_k = \\sum_{i=1}^{N} \\log p(y_i \\mid D_{-i}, M_k)

and the weights are the pseudo-BMA (a.k.a. stabilized softmax of the LOO scores)
weights of Yao et al. (2018, eq. 12)::

    w_k = \\exp(\\mathrm{lpd}_k) / \\sum_j \\exp(\\mathrm{lpd}_j)

which keeps :math:`w_k \\in [0,1]` and :math:`\\sum_k w_k = 1` by construction and
penalises members whose posterior is dominated by a few high-leverage
observations (a plain in-sample likelihood would always prefer the most
flexible member).  Log-sum-exp stabilisation makes the computation overflow-free.

Predictive distribution
-----------------------
The BMA posterior predictive is the mixture

.. math::

    p(y^* \\mid D) = \\sum_k w_k\\, p(y^* \\mid D, M_k),

sampled here by stratified allocation of the posterior draws (each member
contributes ``round(w_k N)`` draws), so the mixture keeps both the
*within-model* (parameter + discrepancy) and the *between-model* (structural)
spread.  Because the members share the same physics decomposition, the
between-model term is the honest measure of how much the *structural*
assumption moves the answer; a member that omits a mechanism the data see has to
absorb the residual in its discrepancy, which is exactly what makes its LOO score
(and thus its weight) collapse.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .bayes import CalibrationData, ChainSet, coverage_rate, predictive_draws, predictive_interval
from .twin import FORMS, ModelForm, ShipGeometry, get_form

__all__ = [
    "pseudo_bma_weights",
    "uniform_weights",
    "bma_mixture",
    "single_form_interval",
    "bma_report",
]


def _logsumexp(values: Sequence[float]) -> float:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        raise ValueError("empty score vector")
    m = float(np.max(v))
    if not np.isfinite(m):
        return -math.inf
    return m + float(np.log(np.sum(np.exp(v - m))))


def pseudo_bma_weights(lpd: Mapping[str, float], *, temperature: float = 1.0) -> Dict[str, float]:
    """Stabilized pseudo-BMA weights from LOO log predictive scores."""
    if not lpd:
        raise ValueError("need at least one model's score")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    keys = sorted(lpd)
    scores = np.array([float(lpd[k]) for k in keys]) / float(temperature)
    if not np.all(np.isfinite(scores)):
        raise ValueError("non-finite LOO score; weights would be undefined")
    z = scores - float(np.max(scores))
    w = np.exp(z)
    w /= w.sum()
    return {k: float(w[i]) for i, k in enumerate(keys)}


def uniform_weights(keys: Iterable[str]) -> Dict[str, float]:
    ks = sorted(keys)
    if not ks:
        raise ValueError("no members")
    return {k: 1.0 / len(ks) for k in ks}


def bma_mixture(
    draws_by_form: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
    *,
    seed: int = 20236,
) -> np.ndarray:
    """Stratified mixture of per-member predictive draws ``(n_draws, n_records)``."""
    keys = sorted(draws_by_form)
    if set(keys) != set(weights):
        raise ValueError("weights must cover exactly the supplied members")
    wsum = float(sum(weights.values()))
    if abs(wsum - 1.0) > 1e-6:
        raise ValueError(f"weights must sum to 1, got {wsum:.6f}")
    sizes = [draws_by_form[k].shape[0] for k in keys]
    n_target = int(max(sizes))
    alloc = {k: max(1, int(round(weights[k] * n_target))) for k in keys}
    rng = np.random.default_rng(seed)
    parts: List[np.ndarray] = []
    for k in keys:
        a = np.asarray(draws_by_form[k], dtype=float)
        take = min(alloc[k], a.shape[0])
        idx = rng.choice(a.shape[0], size=take, replace=take > a.shape[0])
        parts.append(a[idx])
    return np.concatenate(parts, axis=0)


def single_form_interval(
    draws: np.ndarray, level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Central interval of one member's predictive draws and its mean width."""
    lo, hi = predictive_interval(draws, level)
    return lo, hi, float(np.mean(hi - lo))


def bma_report(
    draws_by_form: Mapping[str, np.ndarray],
    lpd: Mapping[str, float],
    observed: np.ndarray,
    *,
    level: float = 0.95,
    seed: int = 20236,
) -> Dict[str, Any]:
    """Weights, per-member intervals and the BMA predictive width comparison."""
    weights = pseudo_bma_weights(lpd)
    y = np.asarray(observed, dtype=float).reshape(-1)
    members: Dict[str, Any] = {}
    widths: List[float] = []
    for k in sorted(draws_by_form):
        draws = np.asarray(draws_by_form[k], dtype=float)
        lo, hi, width = single_form_interval(draws, level)
        inside, n, rate = coverage_rate(lo, hi, y)
        members[k] = {
            "weight": float(weights[k]),
            "loo_log_predictive": float(lpd[k]),
            "pi_mean_width_kN": width,
            "pi_coverage": rate,
            "pred_mean_kN": float(draws.mean()),
            "pred_sd_kN": float(np.mean(draws.std(axis=0, ddof=1))),
            "form": (
                FORMS[k].as_dict() if k in FORMS else {"key": k, "label": k, "external": True}
            ),
        }
        widths.append(width)
    mix = bma_mixture(draws_by_form, weights, seed=seed)
    lo_b, hi_b = predictive_interval(mix, level)
    inside_b, n_b, rate_b = coverage_rate(lo_b, hi_b, y)
    width_b = float(np.mean(hi_b - lo_b))
    uniform = uniform_weights(draws_by_form)
    mix_u = bma_mixture(draws_by_form, uniform, seed=seed + 1)
    lo_u, hi_u = predictive_interval(mix_u, level)
    width_u = float(np.mean(hi_u - lo_u))

    # between/within decomposition of the BMA predictive variance
    w = np.array([weights[k] for k in sorted(draws_by_form)])
    means = np.array([float(np.mean(draws_by_form[k])) for k in sorted(draws_by_form)])
    sds = np.array([float(np.mean(np.var(draws_by_form[k], axis=0))) for k in sorted(draws_by_form)])
    grand = float(np.sum(w * means))
    within = float(np.sum(w * sds))
    between = float(np.sum(w * (means - grand) ** 2))

    mean_single = float(np.mean(widths))
    narrower_mean = 1.0 - width_b / mean_single if mean_single > 0 else float("nan")
    narrowest = float(np.min(widths))
    return {
        "weights": weights,
        "weights_uniform": uniform,
        "members": members,
        "bma": {
            "pi_mean_width_kN": width_b,
            "pi_coverage": rate_b,
            "n_inside": int(inside_b),
            "n_records": int(n_b),
            "pred_mean_kN": float(np.mean(mix)),
            "pred_sd_kN": float(np.mean(mix.std(axis=0, ddof=1))),
            "level": float(level),
            "between_model_var_kN2": between,
            "within_model_var_kN2": within,
            "between_share": float(between / (between + within)) if (between + within) > 0 else 0.0,
        },
        "uniform_mixture": {"pi_mean_width_kN": width_u},
        "comparison": {
            "mean_single_pi_width_kN": mean_single,
            "narrowest_single_pi_width_kN": narrowest,
            "narrower_fraction_vs_mean_single": float(narrower_mean),
            "narrower_fraction_vs_uniform_mixture": float(1.0 - width_b / width_u)
            if width_u > 0
            else float("nan"),
            "narrower_vs_mean_single_over_20pct": bool(narrower_mean > 0.20),
            "narrower_than_narrowest_single": bool(width_b < narrowest),
        },
        "criteria": {
            "weights_in_unit_interval": bool(
                all(0.0 <= v <= 1.0 for v in weights.values())
            ),
            "weights_sum_to_one": bool(abs(sum(weights.values()) - 1.0) < 1e-9),
            "pi_narrower_than_single_over_20pct": bool(narrower_mean > 0.20),
        },
    }