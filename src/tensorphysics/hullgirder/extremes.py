"""Extreme value statistics of the ice-induced hull girder bending moment.

Workflow
--------
1. draw ``n_real`` realisations of the correlated ice pressure field on the
   beam grid (one field per realisation, shared by every patch position - the
   field is stationary, so moving the contact patch is equivalent to sampling a
   different window of the same realisation);
2. for each realisation take the worst patch position along the ship
   (load envelope) and keep the signed extremes ``max M`` and ``min M``;
3. fit Gumbel (method of moments) and 2-parameter Weibull (method of moments)
   distributions to ``max |M|``;
4. read the design moment at the per-event probability level implied by the
   100-year exposure.

All operations are plain NumPy; the batch beam solver below reproduces
:func:`tensorphysics.hullgirder.girder.shear_and_moment` exactly (verified by
unit test) but is vectorised over realisations so that 1e4-5e4 realisations
stay cheap.
"""

from __future__ import annotations

import math

import numpy as np

from .ice_load import (
    IceCondition,
    contact_patch_length,
    global_ice_pressure,
    ice_load_scenarios,
    pressure_coefficient_of_variation,
)
from .model import ShipGirder
from .random_field import exponential_covariance, sample_correlated_field

__all__ = [
    "moment_batch",
    "ice_bending_extremes",
    "gumbel_fit",
    "gumbel_return_level",
    "weibull_fit",
    "weibull_return_level",
    "spearman_rank_correlation",
    "combine_design_moments",
]


def moment_batch(
    x: np.ndarray, q: np.ndarray, support: str = "free"
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised shear/moment for a batch of load distributions.

    ``q`` has shape ``(n_real, n)``; returns ``(V, M)`` of the same shape.
    """
    x = np.asarray(x, dtype=float)
    q = np.atleast_2d(np.asarray(q, dtype=float))
    if q.shape[1] != x.size:
        raise ValueError("load array must have shape (n_real, x.size)")
    dx = np.diff(x)
    length = float(x[-1] - x[0])

    if support == "free":
        net = np.trapezoid(q, x, axis=1)
        mom = np.trapezoid(q * x[None, :], x, axis=1)
        xx = float(np.trapezoid(x * x, x))
        a = net / length
        b = mom / xx
        q_used = q - a[:, None] - b[:, None] * x[None, :]
        v = np.concatenate(
            [np.zeros((q.shape[0], 1)), np.cumsum(0.5 * (q_used[:, 1:] + q_used[:, :-1]) * dx, axis=1)],
            axis=1,
        )
    elif support == "simple":
        r0 = np.trapezoid(q * (x[-1] - x)[None, :], x, axis=1) / length
        v = r0[:, None] - np.concatenate(
            [np.zeros((q.shape[0], 1)), np.cumsum(0.5 * (q[:, 1:] + q[:, :-1]) * dx, axis=1)],
            axis=1,
        )
    else:
        raise ValueError("support must be 'free' or 'simple'")
    m = np.concatenate(
        [np.zeros((q.shape[0], 1)), np.cumsum(0.5 * (v[:, 1:] + v[:, :-1]) * dx, axis=1)], axis=1
    )
    return v, m


def ice_bending_extremes(
    ship: ShipGirder,
    condition: IceCondition,
    lam: float,
    n_real: int = 10_000,
    seed: int = 20251031,
    n_positions: int = 9,
    model: str = "exponential",
) -> dict:
    """Monte-Carlo bending moment extremes for the correlated ice pressure field."""
    x = ship.grid()
    mu = global_ice_pressure(condition)
    cov = pressure_coefficient_of_variation()
    sigma = cov * mu
    z = sample_correlated_field(x, lam, sigma=1.0, n_real=n_real, seed=seed, model=model)
    pressure = np.clip(mu + sigma * z, 0.0, None)

    centres = ice_load_scenarios(x, condition, lam, n_positions=n_positions)
    patch = contact_patch_length(condition, lam)
    # line load [N/m] = pressure [N/m^2] * contact height [-] (ice belt height = ice thickness)
    height = condition.thickness
    hogging = np.full(n_real, -np.inf)
    sagging = np.full(n_real, np.inf)
    m_abs = np.zeros(n_real)
    for centre in centres:
        mask = (np.abs(x - centre) <= 0.5 * patch).astype(float)
        q = pressure * (height * mask[None, :])
        _, m = moment_batch(x, q, support="free")
        hogging = np.maximum(hogging, m.max(axis=1))
        sagging = np.minimum(sagging, m.min(axis=1))
        m_abs = np.maximum(m_abs, np.max(np.abs(m), axis=1))

    # analytic standard deviation of the fluctuation part (no MC noise)
    covariance = exponential_covariance(x, lam, sigma=sigma * height) if model == "exponential" else None
    analytic_std = None
    if covariance is not None:
        from .girder import moment_statistics

        centre = centres[len(centres) // 2]
        mask = (np.abs(x - centre) <= 0.5 * patch).astype(float)
        cov_masked = covariance * np.outer(mask, mask)
        analytic_std, _ = moment_statistics(x, cov_masked, support="free")

    return {
        "M_max_abs_Nm": m_abs,
        "M_hogging_Nm": hogging,
        "M_sagging_Nm": sagging,
        "mean_pressure_Pa": mu,
        "sigma_pressure_Pa": sigma,
        "correlation_length_m": float(lam),
        "contact_patch_length_m": float(patch),
        "n_realisations": int(n_real),
        "n_patch_positions": int(n_positions),
        "patch_centres_m": [float(c) for c in centres],
        "analytic_std_profile_Nm": None if analytic_std is None else analytic_std.tolist(),
        "mc_std_M_max_Nm": float(np.std(m_abs, ddof=1)),
        "sample": None,
    }


# ------------------------------------------------------------------- fitting
_EULER_GAMMA = 0.5772156649015329


def gumbel_fit(samples: np.ndarray) -> dict:
    """Gumbel distribution by the method of moments."""
    s = np.asarray(samples, dtype=float).ravel()
    if s.size < 10:
        raise ValueError("at least 10 samples are required")
    mean = float(s.mean())
    std = float(s.std(ddof=1))
    if std <= 0.0:
        raise ValueError("zero variance sample")
    beta = std * math.sqrt(6.0) / math.pi
    mu = mean - _EULER_GAMMA * beta
    return {"loc": mu, "scale": beta, "mean": mean, "std": std, "distribution": "gumbel"}


def gumbel_return_level(fit: dict, exceedance: float) -> float:
    """Quantile of the Gumbel fit at a per-event exceedance probability."""
    if not 0.0 < exceedance < 1.0:
        raise ValueError("exceedance probability must be in (0, 1)")
    return float(fit["loc"] - fit["scale"] * math.log(-math.log(1.0 - exceedance)))


def weibull_fit(samples: np.ndarray) -> dict:
    """Two-parameter Weibull by the method of moments (shape from the CoV)."""
    s = np.asarray(samples, dtype=float).ravel()
    if s.size < 10:
        raise ValueError("at least 10 samples are required")
    if np.any(s <= 0.0):
        raise ValueError("Weibull fit requires strictly positive samples")
    mean = float(s.mean())
    std = float(s.std(ddof=1))
    cv = std / mean
    if cv <= 0.0:
        raise ValueError("zero variance sample")
    k = cv ** (-1.086)
    scale = mean / math.gamma(1.0 + 1.0 / k)
    return {"shape": k, "scale": scale, "mean": mean, "std": std, "distribution": "weibull"}


def weibull_return_level(fit: dict, exceedance: float) -> float:
    """Quantile of the Weibull fit at a per-event exceedance probability."""
    if not 0.0 < exceedance < 1.0:
        raise ValueError("exceedance probability must be in (0, 1)")
    return float(fit["scale"] * (-math.log(exceedance)) ** (1.0 / fit["shape"]))


def spearman_rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation (Pearson correlation of ranks, tie-free)."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size != b.size or a.size < 3:
        raise ValueError("need two equal-length arrays with at least 3 points")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = math.sqrt(float(np.sum(ra ** 2) * np.sum(rb ** 2)))
    if denom == 0.0:
        raise ValueError("degenerate ranks")
    return float(np.sum(ra * rb) / denom)


def combine_design_moments(
    still_water: float,
    wave: float,
    ice: float,
    psi_ice: float = 0.7,
    psi_wave: float = 0.7,
) -> dict:
    """Combination of still-water, wave and ice bending moments.

    Two random processes with different exposure are combined with the
    Turkstra / Ferry Borges-Castanheta envelope rule: one action takes its
    design (extreme) value while the companion action takes a reduced value
    (``psi``).  The governing value is the larger of::

        M1 = M_sw + M_wave + psi_ice * M_ice     (wave leading)
        M2 = M_sw + M_ice  + psi_wave * M_wave   (ice leading)

    The rule replaces a naive linear sum of two independent extremes and is the
    standard engineering treatment of the non-simultaneity of wave and ice
    loading; no physical wave-ice interaction (e.g. ice damping of the wave
    response) is modelled.
    """
    m1 = still_water + wave + psi_ice * ice
    m2 = still_water + ice + psi_wave * wave
    governing = max(abs(m1), abs(m2))
    return {
        "wave_leading_Nm": float(m1),
        "ice_leading_Nm": float(m2),
        "governing_Nm": float(governing),
        "governing_case": "wave_leading" if abs(m1) >= abs(m2) else "ice_leading",
        "psi_ice": float(psi_ice),
        "psi_wave": float(psi_wave),
        "still_water_Nm": float(still_water),
        "wave_Nm": float(wave),
        "ice_Nm": float(ice),
        "rule": "Turkstra / Ferry Borges-Castanheta envelope combination",
    }