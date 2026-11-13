"""Calibration of the pressure-area exponent ``ex`` from the source library.

Pipeline
========
1. every library record enters as an estimate ``ex_i +- sigma_i`` (``sigma_i``
   from the declared 90 % band);
2. ``ex`` is mapped to the reference area convention with law (T)
   (:func:`tensorphysics.ice_pressure_area.conventions.to_reference`);
3. records are pooled **per stratum** (global / measured-local / design-rule) by
   DerSimonian-Laird random effects, because the strata answer different
   physical questions and must not be averaged;
4. leave-one-out and the process-zone sweep ``delta/sqrt(A)`` give the
   sensitivity of the pooled value;
5. covariates that do not vary across the stratum (e.g. ice temperature) are
   reported as **not identifiable** instead of being fitted -- a regression on a
   rank-deficient design matrix would fabricate a slope.

Nothing here hides a judgement: pooled band, heterogeneity ``tau``, the
narrowing of the demand bandwidth and the residual scatter are reported side by
side so a reader can see *what* the calibration did and did not buy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .conventions import (
    required_area_ratio,
    required_area_ratio_solutions,
)
from .library import PressureAreaEntry, PressureAreaLibrary

Z90 = 1.6448536269514722
"""Two-sided 90 % normal quantile (band half-width -> sigma)."""


def band_to_sigma(lo: float, hi: float, confidence: float = 0.90) -> float:
    """Turn a ``confidence``-level band into a one-sigma standard deviation."""
    if hi <= lo:
        raise ValueError(f"band must be increasing, got ({lo}, {hi})")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0,1), got {confidence}")
    z = Z90 if abs(confidence - 0.90) < 1e-12 else _norm_ppf(0.5 + 0.5 * confidence)
    return (hi - lo) / (2.0 * z)


def _norm_ppf(p: float) -> float:
    """Rational approximation of the standard normal quantile (Acklam)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0,1), got {p}")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
    )


# ---------------------------------------------------------------------------
# power-law fit (used on tabulated p(A) data, e.g. the sha256-anchored assets)
# ---------------------------------------------------------------------------


def fit_power_law(
    areas_m2: Sequence[float],
    pressures_MPa: Sequence[float],
    *,
    weights: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Log-space least squares fit of ``p = p0 * A**-ex``.

    Returns ``ex``, ``p0`` (at A = 1 m^2), their 1-sigma standard errors and the
    log-space coefficient of determination.  Raises on non-positive input
    instead of silently dropping points.
    """
    a = np.asarray(areas_m2, dtype=float)
    p = np.asarray(pressures_MPa, dtype=float)
    if a.shape != p.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {p.shape}")
    if a.size < 2:
        raise ValueError("at least two (A, p) points are required")
    if np.any(a <= 0) or np.any(p <= 0):
        raise ValueError("areas and pressures must be strictly positive")
    w = np.ones_like(a) if weights is None else np.asarray(weights, dtype=float)
    if np.any(w <= 0):
        raise ValueError("weights must be strictly positive")

    x = np.log(a)
    y = np.log(p)
    sw = float(w.sum())
    xbar = float((w * x).sum() / sw)
    ybar = float((w * y).sum() / sw)
    sxx = float((w * (x - xbar) ** 2).sum())
    sxy = float((w * (x - xbar) * (y - ybar)).sum())
    slope = sxy / sxx
    intercept = ybar - slope * xbar
    resid = y - (intercept + slope * x)
    dof = max(int(a.size) - 2, 1)
    sigma2 = float((w * resid ** 2).sum() / max(sw / float(a.size), 1e-12) / dof)
    se_slope = math.sqrt(sigma2 / sxx) if sxx > 0 else float("inf")
    ss_tot = float((w * (y - ybar) ** 2).sum())
    r2 = 1.0 - float((w * resid ** 2).sum()) / ss_tot if ss_tot > 0 else 1.0
    return {
        "ex": -float(slope),
        "ex_se": float(se_slope),
        "p0_MPa": float(math.exp(intercept)),
        "p0_se_log": float(math.sqrt(sigma2 * (1.0 / sw + xbar ** 2 / sxx))),
        "r2_logspace": float(r2),
        "n_points": int(a.size),
        "area_range_m2": [float(a.min()), float(a.max())],
    }


# ---------------------------------------------------------------------------
# DerSimonian-Laird random effects pooling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Estimate:
    label: str
    ex: float
    sigma: float
    band: Tuple[float, float]
    meta: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "ex": self.ex,
            "sigma": self.sigma,
            "band90": list(self.band),
            "weight": 1.0 / self.sigma ** 2,
            **dict(self.meta),
        }


def dl_pool(estimates: Sequence[Estimate]) -> Dict[str, Any]:
    """DerSimonian-Laird random-effects pooled mean with a 90 % interval."""
    if not estimates:
        raise ValueError("need at least one estimate")
    y = np.array([e.ex for e in estimates], dtype=float)
    v = np.array([e.sigma ** 2 for e in estimates], dtype=float)
    if np.any(v <= 0):
        raise ValueError("every estimate needs a positive variance")
    w_fe = 1.0 / v
    mu_fe = float((w_fe * y).sum() / w_fe.sum())
    q = float((w_fe * (y - mu_fe) ** 2).sum())
    k = len(estimates)
    df = k - 1
    if k >= 2:
        c = float(w_fe.sum() - (w_fe ** 2).sum() / w_fe.sum())
        tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
        i2 = max(0.0, (q - df) / q) * 100.0 if q > 0 else 0.0
    else:
        tau2, i2 = 0.0, 0.0
    w_re = 1.0 / (v + tau2)
    mu_re = float((w_re * y).sum() / w_re.sum())
    se_re = math.sqrt(1.0 / float(w_re.sum()))
    return {
        "k": k,
        "ex_pooled": mu_re,
        "se": se_re,
        "ci90": [mu_re - Z90 * se_re, mu_re + Z90 * se_re],
        "ex_fixed": mu_fe,
        "Q": q,
        "df": df,
        "tau2": tau2,
        "tau": math.sqrt(tau2),
        "I2_pct": i2,
        "heterogeneity": ("homogeneous" if q <= df else ("moderate" if i2 < 50 else "substantial")),
        "weights_re": {e.label: float(w) for e, w in zip(estimates, w_re)},
        "members": [e.to_dict() for e in estimates],
    }


def leave_one_out(estimates: Sequence[Estimate]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if len(estimates) < 3:
        return out
    for i in range(len(estimates)):
        subset = [e for j, e in enumerate(estimates) if j != i]
        pooled = dl_pool(subset)
        out.append(
            {
                "omitted": estimates[i].label,
                "ex_pooled": pooled["ex_pooled"],
                "ci90": pooled["ci90"],
                "tau": pooled["tau"],
            }
        )
    return out


# ---------------------------------------------------------------------------
# meta-regression with identification check
# ---------------------------------------------------------------------------


def meta_regression(
    estimates: Sequence[Estimate], covariates: Mapping[str, Sequence[Optional[float]]]
) -> Dict[str, Any]:
    """Weighted least squares of ``ex`` on covariates, with an identification audit.

    A covariate is *identifiable* only if it takes at least two distinct values
    across the estimates **and** its column adds rank to the weighted design
    matrix.  Unidentifiable covariates are not silently dropped: they are listed
    with the reason, because "not measurable in the assembled data" is a result,
    not a gap to be papered over.
    """
    if not estimates:
        raise ValueError("need at least one estimate")
    n = len(estimates)
    names = ["intercept"] + list(covariates)
    columns: List[np.ndarray] = [np.ones(n)]
    for name in covariates:
        vals = list(covariates[name])
        if len(vals) != n:
            raise ValueError(f"covariate {name!r} has {len(vals)} values for {n} estimates")
        columns.append(np.array([np.nan if v is None else float(v) for v in vals], dtype=float))

    y = np.array([e.ex for e in estimates], dtype=float)
    w = np.array([1.0 / e.sigma ** 2 for e in estimates], dtype=float)

    identification: Dict[str, Any] = {}
    for name, col in zip(names, columns):
        if name == "intercept":
            identification[name] = {
                "n_finite": n,
                "distinct_values": 1,
                "identifiable": True,
                "reason": "intercept is always retained",
            }
            continue
        finite = np.isfinite(col)
        uniq = np.unique(col[finite])
        identification[name] = {
            "n_finite": int(finite.sum()),
            "distinct_values": int(uniq.size),
            "identifiable": bool(finite.all() and uniq.size >= 2),
            "reason": (
                "ok"
                if finite.all() and uniq.size >= 2
                else (
                    f"only {int(finite.sum())}/{n} sources report this covariate"
                    if not finite.all()
                    else "constant across sources (no contrast to regress on)"
                )
            ),
        }
    keep_idx = [0]
    for idx, name in enumerate(names[1:], start=1):
        if identification[name]["identifiable"]:
            keep_idx.append(idx)
    X = np.column_stack([columns[i] for i in keep_idx])
    Xw = X * np.sqrt(w)[:, None]
    yw = y * np.sqrt(w)
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = yw - Xw @ beta
    dof = max(n - X.shape[1], 1)
    sigma2 = float((resid ** 2).sum() / dof)
    cov = sigma2 * np.linalg.pinv(Xw.T @ Xw)
    se = np.sqrt(np.diag(cov))
    used = [names[i] for i in keep_idx]
    return {
        "coefficients": {name: float(b) for name, b in zip(used, beta)},
        "stderr": {name: float(s) for name, s in zip(used, se)},
        "dof": dof,
        "rank": int(np.linalg.matrix_rank(Xw)),
        "n_coefficients": len(used),
        "identification": identification,
        "dropped": [n for n in names[1:] if not identification[n]["identifiable"]],
        "withheld_note": (
            "a covariate with a single distinct value across sources cannot be "
            "calibrated; its effect stays WITHHELD rather than invented"
        ),
    }


# ---------------------------------------------------------------------------
# estimates from the library
# ---------------------------------------------------------------------------


def _amplification(entry: PressureAreaEntry, delta_over_sqrtA: float) -> float:
    from .conventions import ratio_factor

    return ratio_factor(
        entry.area_definition,
        entry.A_ref_m2 if entry.A_ref_m2 else 1.0,
        lambda_perimeter=entry.geometry.get("perimeter_factor_lambda") or 4.0,
        delta_over_sqrtA=delta_over_sqrtA,
    )


def estimates_from_library(
    library: PressureAreaLibrary,
    stratum: str,
    *,
    delta_over_sqrtA: float = 0.0,
    normalize: bool = True,
) -> List[Estimate]:
    """Build convention-normalized estimates for one stratum."""
    members = [e for e in library if e.stratum == stratum]
    if not members:
        raise ValueError(f"empty stratum {stratum!r}")
    out: List[Estimate] = []
    for e in members:
        amp = _amplification(e, delta_over_sqrtA) if normalize else 1.0
        centre = e.normalized_ex(delta_over_sqrtA) if normalize else e.ex
        lo = centre - (e.ex - e.ex_band[0]) * amp
        hi = centre + (e.ex_band[1] - e.ex) * amp
        out.append(
            Estimate(
                label=e.asset_id,
                ex=centre,
                sigma=(hi - lo) / (2.0 * Z90),
                band=(lo, hi),
                meta={
                    "family": e.family,
                    "scale": e.scale,
                    "role": e.role,
                    "area_definition": e.area_definition,
                    "primary_source_verified": e.primary_source_verified,
                    "ex_reported": e.ex,
                    "ex_band_reported": list(e.ex_band),
                    "convention_shift": centre - e.ex,
                    "amplification_r": amp,
                },
            )
        )
    return out


def covariate_matrix(library: PressureAreaLibrary, stratum: str) -> Dict[str, List[Optional[float]]]:
    """Covariates taken from each record's ``conditions`` (mid-band where a range)."""
    members = [e for e in library if e.stratum == stratum]

    def mid(rng: Optional[Sequence[float]]) -> Optional[float]:
        if not rng:
            return None
        return 0.5 * (float(rng[0]) + float(rng[1]))

    return {
        "ice_thickness_m": [mid(e.conditions.get("ice_thickness_m")) for e in members],
        "ice_temperature_C": [mid(e.conditions.get("ice_temperature_C")) for e in members],
    }


# ---------------------------------------------------------------------------
# demand bandwidth (the quantity the gap is actually about)
# ---------------------------------------------------------------------------


def demand_bandwidth_study(
    declared_band: Tuple[float, float],
    measured_bandwidth: Tuple[float, float],
    calibrated_bands: Mapping[str, Tuple[float, float]],
    *,
    area_ratio: Optional[float] = None,
) -> Dict[str, Any]:
    """Compare the declared exponent band with the calibrated strata.

    ``measured_bandwidth`` is the ship-structure demand band measured by the
    platform for ``ex in declared_band`` (``1.32x-1.74x`` in the gap).  The area
    ratio consistent with that pair is derived first, then each calibrated band
    is mapped through ``r**ex`` so the bands are directly comparable.
    """
    r_star = (
        area_ratio
        if area_ratio is not None
        else required_area_ratio(measured_bandwidth, declared_band)
    )
    sols = required_area_ratio_solutions(measured_bandwidth, declared_band)
    lo_d = r_star ** declared_band[0]
    hi_d = r_star ** declared_band[1]
    declared_width = hi_d - lo_d
    rows = []
    for name, band in calibrated_bands.items():
        lo, hi = r_star ** band[0], r_star ** band[1]
        rows.append(
            {
                "stratum": name,
                "ex_band": list(band),
                "bandwidth_band": [lo, hi],
                "width": hi - lo,
                "narrowing_factor_vs_declared": (declared_width / (hi - lo)) if hi > lo else None,
                "inside_declared": bool(lo >= lo_d - 1e-12 and hi <= hi_d + 1e-12),
            }
        )
    return {
        "declared_ex_band": list(declared_band),
        "declared_bandwidth_band": list(measured_bandwidth),
        "declared_width": declared_width,
        "reproduced_declared_bandwidth": [lo_d, hi_d],
        "area_ratio_used": r_star,
        "area_ratio_solutions": list(sols),
        "area_ratio_inconsistency_pct": 100.0 * (max(sols) - min(sols)) / r_star,
        "rows": rows,
    }