"""Variance-based global sensitivity of the ice-resistance twin (Saltelli/Jansen).

What is computed
----------------
Variance decomposition of the twin response at a reference ice condition,

.. math::

    Y = R_\\text{twin}(\\theta; x_\\text{ref}),\\qquad \\theta \\sim U(\\text{box}),

into first-order (:math:`S_i`, share of the output variance explained by
parameter *i* alone) and total effects (:math:`S_{T,i}`, share that would be
removed if parameter *i* were frozen).  The estimators are the pick-freeze /
Saltelli-Jansen ones already implemented and independently benchmarked in
:func:`tensorphysics.uq.sobol_mc.saltelli_indices` -- this module **reuses** them
and only adds the digital-twin-specific wiring:

* the black box is the twin's resistance at a declared reference condition,
* the closure identities that make the numbers auditable are reported
  (:math:`\\sum_i S_i + \\sum_{i<j} S_{ij} \\to 1` when interactions are at most
  pairwise; :math:`\\sum_i S_{T,i} - 1` = interaction share),
* an **additive submodel** of the same physics (friction fixed at its literature
  value, only the three strength/submergence multipliers free) gives a case where
  :math:`\\sum_i S_{T,i} = 1` analytically, so the sampler/estimator closure can
  be checked against a closed-form answer rather than against itself.

Interpretation guard
--------------------
:math:`\\sum_i S_{T,i} = 1` is *not* a universal identity: it holds exactly when
the model is additive in the inputs and exceeds one by exactly the share of the
variance carried by interactions.  Reporting a twin model's
:math:`\\sum S_{T,i} > 1` as "wrong" would be a category error, so both the
additive closure (an estimator check) and the interaction share (a physics
statement about coupling between friction and the strength terms) are reported.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..ice.resistance import G, MU_HULL_ICE, RHO_ICE, SIGMA_C_REF, SIGMA_F_REF, ShipGeometry
from ..uq.sobol_mc import saltelli_indices
from .twin import (
    PARAM_BOUNDS,
    PARAM_NAMES,
    PARAM_REFERENCE,
    get_form,
    resistance_kN_batch,
    unit_to_physical,
)

__all__ = [
    "reference_condition",
    "twin_sobol",
    "additive_submodel_exact",
    "additive_submodel_sobol",
    "closure_metrics",
    "sensitivity_report",
]


@dataclass(frozen=True)
class ReferenceCondition:
    """Reference ice condition at which the sensitivity is evaluated."""

    h_m: float
    V_ms: float
    label: str = "h=1.0 m, V=4 kn"

    def as_dict(self) -> Dict[str, Any]:
        return {"h_m": self.h_m, "V_ms": self.V_ms, "label": self.label}


def reference_condition(ship: ShipGeometry, h_m: float = 1.0, V_kn: float = 4.0) -> ReferenceCondition:
    """Default reference condition, clipped into the law's validity domain."""
    from ..ice.resistance import DomainLimits

    limits = DomainLimits()
    h = min(h_m, 0.9 * limits.h_limit(ship))
    v = min(V_kn * 0.514444, 0.9 * limits.v_limit(ship))
    return ReferenceCondition(h_m=float(h), V_ms=float(v), label=f"h={h:.2f} m, V={V_kn:.1f} kn")


def twin_sobol(
    ship: ShipGeometry,
    ref: Optional[ReferenceCondition] = None,
    *,
    n_base: int = 1 << 13,
    seed: int = 20234,
    second_order: bool = True,
    bootstrap: int = 200,
    form: Any = "full",
) -> Dict[str, Any]:
    """Sobol' indices of the twin resistance w.r.t. the four calibration parameters."""
    ref = ref or reference_condition(ship)
    f = get_form(form)
    if form != "full":
        raise ValueError("global sensitivity is defined for the reference form 'full'")
    d = len(PARAM_NAMES)

    def evaluate(unit: np.ndarray) -> np.ndarray:
        phys = unit_to_physical(unit, PARAM_BOUNDS)  # (n, d)
        return resistance_kN_batch(f, phys, ref.h_m, ref.V_ms, ship)

    res = saltelli_indices(
        evaluate,
        d,
        n_base=n_base,
        seed=seed,
        second_order=second_order,
        bootstrap=bootstrap,
    )
    second = {
        f"{PARAM_NAMES[i]}~{PARAM_NAMES[j]}": float(v)
        for (i, j), v in res.second.items()
    }
    total_sum = float(np.sum(res.total))
    first_sum = float(np.sum(res.first))
    second_sum = float(sum(res.second.values()))
    out: Dict[str, Any] = {
        "reference": ref.as_dict(),
        "n_base": int(res.n_base),
        "n_model_evaluations": int(res.n_model_evaluations),
        "seed": int(seed),
        "bootstrap": int(bootstrap),
        "output_mean_kN": float(res.mean),
        "output_sd_kN": float(math.sqrt(res.variance)),
        "first": {n: float(res.first[i]) for i, n in enumerate(PARAM_NAMES)},
        "first_saltelli": {
            n: float(res.first_saltelli[i]) for i, n in enumerate(PARAM_NAMES)
        },
        "total": {n: float(res.total[i]) for i, n in enumerate(PARAM_NAMES)},
        "second": second,
        "sum_first": first_sum,
        "sum_total": total_sum,
        "sum_second": second_sum,
        "interaction_share": total_sum - 1.0,
        "all_order_sum": first_sum + second_sum,
        "all_order_closure_error": abs(first_sum + second_sum - 1.0),
        "ranking_by_total": [
            n
            for n, _ in sorted(
                ((n, float(res.total[i])) for i, n in enumerate(PARAM_NAMES)),
                key=lambda kv: kv[1],
                reverse=True,
            )
        ],
    }
    if res.first_ci is not None and res.total_ci is not None:
        out["first_ci"] = {
            n: [float(res.first_ci[0][i]), float(res.first_ci[1][i])]
            for i, n in enumerate(PARAM_NAMES)
        }
        out["total_ci"] = {
            n: [float(res.total_ci[0][i]), float(res.total_ci[1][i])]
            for i, n in enumerate(PARAM_NAMES)
        }
    return out


def additive_submodel_exact(
    ship: ShipGeometry, ref: ReferenceCondition
) -> Dict[str, Any]:
    """Closed-form Sobol' indices of the additive (fixed-friction) submodel.

    With ``mu`` frozen at its literature value the twin response is

    .. math::  R = A\\,\\theta_c + B\\,\\theta_f + C\\,\\theta_s

    with ``A = 0.5 (tan a + mu/cos a) sigma_c^ref h^2``,
    ``B = 0.5 (sin(phi/2)+mu cos(phi/2)) sigma_f^ref h B_ship / k_bend`` and
    ``C = (0.15+0.3 Fr_h) rho_i V^2 h B_ship``.  For independent uniform
    parameters on ``[lo_i, hi_i]`` the Sobol' indices are exact::

        S_i = S_T,i = (A_i w_i)^2 / sum_j (A_j w_j)^2,   w_i = hi_i - lo_i
    """
    from .twin import k_bend_reference

    alpha, phi = ship.flare_rad, ship.bow_angle_rad
    h, v = ref.h_m, ref.V_ms
    A = 0.5 * (math.tan(alpha) + MU_HULL_ICE / math.cos(alpha)) * SIGMA_C_REF * h**2
    B = (
        0.5
        * (math.sin(0.5 * phi) + MU_HULL_ICE * math.cos(0.5 * phi))
        / k_bend_reference()
        * SIGMA_F_REF
        * h
        * ship.B
    )
    C = (0.15 + 0.3 * v / math.sqrt(G * h)) * RHO_ICE * v**2 * h * ship.B
    names = ("sigma_c_scale", "sigma_f_scale", "C_sub_scale")
    coeffs = {"sigma_c_scale": A, "sigma_f_scale": B, "C_sub_scale": C}
    widths = {}
    for n in names:
        i = PARAM_NAMES.index(n)
        widths[n] = float(PARAM_BOUNDS[i, 1] - PARAM_BOUNDS[i, 0])
    contrib = {n: (coeffs[n] * widths[n]) ** 2 for n in names}
    total = sum(contrib.values())
    idx = {n: contrib[n] / total for n in names}
    return {
        "coefficients_kN": {n: coeffs[n] / 1e3 for n in names},
        "indices": idx,
        "sum_indices": float(sum(idx.values())),
        "mean_kN": float(
            sum(coeffs[n] * (PARAM_REFERENCE[n] + 0.5 * widths[n]) for n in names) / 1e3
        ),
        "reference": ref.as_dict(),
    }


def additive_submodel_sobol(
    ship: ShipGeometry,
    ref: ReferenceCondition,
    *,
    n_base: int = 1 << 13,
    seed: int = 20235,
    bootstrap: int = 200,
) -> Dict[str, Any]:
    """Pick-freeze estimate of the additive submodel indices (3 inputs)."""
    names = ("sigma_c_scale", "sigma_f_scale", "C_sub_scale")
    idxs = [PARAM_NAMES.index(n) for n in names]
    bounds = PARAM_BOUNDS[idxs]

    def evaluate(unit: np.ndarray) -> np.ndarray:
        phys = unit_to_physical(unit, bounds)
        full = np.tile(
            np.array([PARAM_REFERENCE[n] for n in PARAM_NAMES], dtype=float),
            (unit.shape[0], 1),
        )
        full[:, idxs] = phys
        return resistance_kN_batch("full", full, ref.h_m, ref.V_ms, ship)

    res = saltelli_indices(
        evaluate, len(names), n_base=n_base, seed=seed, second_order=False,
        bootstrap=bootstrap,
    )
    est = {n: float(res.first[i]) for i, n in enumerate(names)}
    est_tot = {n: float(res.total[i]) for i, n in enumerate(names)}
    return {
        "names": list(names),
        "n_base": int(res.n_base),
        "n_model_evaluations": int(res.n_model_evaluations),
        "seed": int(seed),
        "first": est,
        "total": est_tot,
        "sum_first": float(sum(est.values())),
        "sum_total": float(sum(est_tot.values())),
    }


def closure_metrics(exact: Dict[str, Any], estimate: Dict[str, Any]) -> Dict[str, Any]:
    """Compare estimated vs analytic indices for the additive submodel."""
    names = exact.get("indices", {})
    first = estimate.get("first", {})
    total = estimate.get("total", {})
    per_param = {
        n: {
            "exact": float(names[n]),
            "first": float(first[n]),
            "total": float(total[n]),
            "first_error": float(abs(first[n] - names[n])),
            "total_error": float(abs(total[n] - names[n])),
        }
        for n in names
    }
    max_total_err = max(v["total_error"] for v in per_param.values())
    sum_total_err = abs(estimate["sum_total"] - 1.0)
    return {
        "per_parameter": per_param,
        "max_total_index_error": float(max_total_err),
        "sum_total_estimate": float(estimate["sum_total"]),
        "sum_total_error_vs_one": float(sum_total_err),
        "closure_ok_5pct": bool(max(max_total_err, sum_total_err) < 0.05),
    }


def sensitivity_report(
    ship: ShipGeometry,
    *,
    n_base: int = 1 << 13,
    seed: int = 20234,
    bootstrap: int = 200,
    ref: Optional[ReferenceCondition] = None,
) -> Dict[str, Any]:
    """Full sensitivity section of the UQ report (twin + additive closure)."""
    ref = ref or reference_condition(ship)
    twin = twin_sobol(ship, ref, n_base=n_base, seed=seed, bootstrap=bootstrap)
    exact = additive_submodel_exact(ship, ref)
    est = additive_submodel_sobol(
        ship, ref, n_base=n_base, seed=seed + 1, bootstrap=bootstrap
    )
    closure = closure_metrics(exact, est)
    return {
        "twin": twin,
        "additive_exact": exact,
        "additive_estimate": est,
        "closure": closure,
        "criteria": {
            "sum_total_closure_under_5pct": bool(closure["closure_ok_5pct"]),
            "interaction_share_reported": float(twin["interaction_share"]),
            "all_order_closure_error": float(twin["all_order_closure_error"]),
            "all_order_closure_under_5pct": bool(
                twin["all_order_closure_error"] < 0.05
            ),
        },
    }