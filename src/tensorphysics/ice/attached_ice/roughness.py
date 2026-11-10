"""Equivalent sand-grain roughness and ITTC-1978 resistance penalty.

Physical model
==============
The ice layer on a submerged surface increases the hydrodynamic roughness,
which in turn increases the frictional resistance and reduces propulsive
efficiency.

Equivalent sand-grain roughness k_s
-----------------------------------
The equivalent sand-grain roughness is modelled as a combination of:

1. **Underlying surface roughness** (hull paint, corrosion, etc.)
2. **Ice roughness** which scales with ice thickness

The mapping is::

    k_s = k_s0 + alpha_ice * h_ice^beta_ice

where ``k_s0`` is the clean-hull roughness, and ``alpha_ice``, ``beta_ice``
are empirical coefficients. For thin ice (h < 1 mm), the roughness is
dominated by the underlying surface. For thicker ice, the roughness scales
approximately linearly with thickness (beta ~ 1).

ITTC-1978 roughness allowance
-----------------------------
The ITTC-1978 Performance Prediction Method defines the roughness allowance
of a hull with equivalent sand-grain roughness ``k_s`` as::

    Delta C_f(k_s) = (105 * (k_s / L)^(1/3) - 0.64) * 1e-3

(valid for ``1e-6 <= k_s/L <= 1e-4``).  The roughness *penalty* relative to the
clean hull is the increment of that allowance::

    Delta C_f = alpha * [(k_s / L)^(1/3) - (k_s0 / L)^(1/3)]

with ``alpha = 105e-3 = 0.105``, ``L`` the ship length [m], ``k_s`` the
roughened equivalent sand-grain roughness [m] and ``k_s0`` the baseline
(clean-hull) roughness [m].  Because the constant term cancels, the increment
form is exact with respect to :func:`ittc_1978_delta_cf`.

.. note::
   Until 2026-11-10 this module used ``alpha = 0.004``, which under-predicted the
   penalty by a factor ~15 (96% error against the ITTC-1978 expression for
   k_s = 0.5 mm -> 5 mm at L = 150 m).  See the maturity audit
   ``artifacts/attached_ice_maturity/*``.

Thrust loss
-----------
The thrust loss due to increased resistance is estimated as::

    Delta T / T = Delta C_f / C_f0

where ``C_f0`` is the clean-hull friction coefficient (from ITTC-1957
or similar). This is a first-order estimate; detailed propeller-hull
interaction requires a full propulsion analysis.

P50/P90 envelope
----------------
For probabilistic assessment, the P50 (median) and P90 (90th percentile)
envelopes of k_s and Delta C_f can be computed by propagating uncertainty
in the input parameters (ice thickness, roughness coefficients) through
the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import (
    ITTC_DELTA_CF_ALPHA, DEFAULT_HULL_ROUGHNESS_M, DEFAULT_SHIP_LENGTH_M,
)

# Ice roughness scaling coefficients
ALPHA_ICE = 0.5   # dimensionless, fraction of ice thickness contributing to roughness
BETA_ICE = 1.0    # exponent (linear scaling)

# ITTC-1957 friction coefficient (for thrust loss estimate)
# C_f0 = 0.075 / (log10(Re) - 2)^2, typical value ~0.0015 for ships
DEFAULT_CF0 = 0.0015


@dataclass(frozen=True)
class RoughnessState:
    """State of the roughness and resistance penalty analysis."""

    #: Ice thickness [m]
    h_ice: float
    #: Clean-hull roughness [m]
    k_s0: float
    #: Equivalent sand-grain roughness [m]
    k_s: float
    #: ITTC-1978 roughness penalty Delta C_f (dimensionless)
    delta_cf: float
    #: Thrust loss fraction Delta T / T (dimensionless)
    delta_t_over_t: float
    #: Ship length [m]
    ship_length: float

    def as_dict(self) -> dict:
        return {
            "h_ice_m": self.h_ice,
            "k_s0_m": self.k_s0,
            "k_s_m": self.k_s,
            "delta_cf": self.delta_cf,
            "delta_t_over_t": self.delta_t_over_t,
            "ship_length_m": self.ship_length,
        }


def equivalent_sand_roughness(
    h_ice: float,
    k_s0: float = DEFAULT_HULL_ROUGHNESS_M,
    alpha_ice: float = ALPHA_ICE,
    beta_ice: float = BETA_ICE,
) -> float:
    """Compute equivalent sand-grain roughness [m].

    Parameters
    ----------
    h_ice : float or array_like
        Ice thickness [m] (vectorised).
    k_s0 : float
        Clean-hull roughness [m].
    alpha_ice : float
        Ice roughness coefficient.
    beta_ice : float
        Ice roughness exponent.

    Returns
    -------
    float or numpy.ndarray
        Equivalent sand-grain roughness [m].
    """
    h = np.asarray(h_ice, dtype=float)
    if np.any(h < 0.0):
        raise ValueError("ice thickness must be non-negative")
    if k_s0 < 0.0:
        raise ValueError("baseline roughness must be non-negative")

    out = k_s0 + alpha_ice * h**beta_ice
    # scalar in -> float out; array in -> array out (vectorised for envelopes)
    return float(out) if out.ndim == 0 else out


def ittc_1978_roughness_penalty(
    k_s: float,
    k_s0: float = DEFAULT_HULL_ROUGHNESS_M,
    ship_length: float = DEFAULT_SHIP_LENGTH_M,
    alpha: float = ITTC_DELTA_CF_ALPHA,
) -> float:
    """ITTC-1978 roughness allowance Delta C_f.

    Parameters
    ----------
    k_s : float
        Roughened equivalent sand-grain roughness [m].
    k_s0 : float
        Baseline (clean hull) roughness [m].
    ship_length : float
        Ship length [m].
    alpha : float
        ITTC-1978 coefficient (default 0.004).

    Returns
    -------
    float
        Delta C_f (dimensionless).
    """
    k_s = float(k_s)
    k_s0 = float(k_s0)
    L = float(ship_length)

    if L <= 0.0:
        raise ValueError("ship length must be positive")
    if k_s < 0.0 or k_s0 < 0.0:
        raise ValueError("roughness values must be non-negative")

    # ITTC-1978 formula: Delta C_f = alpha * [(k_s/L)^(1/3) - (k_s0/L)^(1/3)]
    term_rough = (k_s / L) ** (1.0 / 3.0)
    term_baseline = (k_s0 / L) ** (1.0 / 3.0)

    return float(alpha * (term_rough - term_baseline))


def ittc_1978_thrust_loss(
    k_s: float,
    k_s0: float = DEFAULT_HULL_ROUGHNESS_M,
    ship_length: float = DEFAULT_SHIP_LENGTH_M,
    alpha: float = ITTC_DELTA_CF_ALPHA,
    cf0: float = DEFAULT_CF0,
    c_total: float | None = None,
) -> float:
    """Estimate thrust loss fraction Delta T / T.

    First-order ITTC-1978 propagation: with ``(1-t)`` unchanged, the required
    thrust scales with the resistance, hence

        Delta T / T = Delta C_f / C_denominator

    where ``C_denominator`` is the resistance coefficient the penalty is added
    to.  Passing the *total* resistance coefficient (``c_total``, supplied by the
    consuming resistance/propulsion chain) is the correct usage; the fallback
    ``cf0`` (friction only) is a conservative upper bound and over-predicts
    strongly once ``Delta C_f`` reaches the same order as ``C_f0``.

    Parameters
    ----------
    k_s : float
        Roughened equivalent sand-grain roughness [m].
    k_s0 : float
        Baseline (clean hull) roughness [m].
    ship_length : float
        Ship length [m].
    alpha : float
        ITTC-1978 coefficient.
    cf0 : float
        Friction coefficient fallback denominator (dimensionless).
    c_total : float, optional
        Total resistance coefficient supplied by the consumer; used as the
        denominator when given.

    Returns
    -------
    float
        Thrust loss fraction (dimensionless).
    """
    delta_cf = ittc_1978_roughness_penalty(k_s, k_s0, ship_length, alpha)
    denom = float(c_total) if c_total is not None else float(cf0)
    if denom <= 0.0:
        raise ValueError("denominator resistance coefficient must be positive")

    return float(delta_cf / denom)


def roughness_state(
    h_ice: float,
    k_s0: float = DEFAULT_HULL_ROUGHNESS_M,
    ship_length: float = DEFAULT_SHIP_LENGTH_M,
    alpha_ice: float = ALPHA_ICE,
    beta_ice: float = BETA_ICE,
    alpha_ittc: float = ITTC_DELTA_CF_ALPHA,
    cf0: float = DEFAULT_CF0,
) -> RoughnessState:
    """Compute the full roughness and penalty state.

    Parameters
    ----------
    h_ice : float
        Ice thickness [m].
    k_s0 : float
        Clean-hull roughness [m].
    ship_length : float
        Ship length [m].
    alpha_ice, beta_ice : float
        Ice roughness scaling coefficients.
    alpha_ittc : float
        ITTC-1978 coefficient.
    cf0 : float
        Clean-hull friction coefficient.

    Returns
    -------
    RoughnessState
    """
    k_s = equivalent_sand_roughness(h_ice, k_s0, alpha_ice, beta_ice)
    delta_cf = ittc_1978_roughness_penalty(k_s, k_s0, ship_length, alpha_ittc)
    delta_t = ittc_1978_thrust_loss(k_s, k_s0, ship_length, alpha_ittc, cf0)

    return RoughnessState(
        h_ice=float(h_ice),
        k_s0=float(k_s0),
        k_s=k_s,
        delta_cf=delta_cf,
        delta_t_over_t=delta_t,
        ship_length=float(ship_length),
    )


def p50_p90_envelope(
    h_ice_values: np.ndarray,
    k_s0: float = DEFAULT_HULL_ROUGHNESS_M,
    ship_length: float = DEFAULT_SHIP_LENGTH_M,
    alpha_ice: float = ALPHA_ICE,
    beta_ice: float = BETA_ICE,
    alpha_ittc: float = ITTC_DELTA_CF_ALPHA,
    cf0: float = DEFAULT_CF0,
) -> dict:
    """Compute P50/P90 envelope for k_s and Delta C_f.

    Parameters
    ----------
    h_ice_values : np.ndarray
        Array of ice thickness samples [m].
    k_s0, ship_length, alpha_ice, beta_ice, alpha_ittc, cf0 : float
        Model parameters.

    Returns
    -------
    dict
        Dictionary with keys: k_s_p50, k_s_p90, delta_cf_p50, delta_cf_p90.
    """
    h_arr = np.asarray(h_ice_values, dtype=float)
    if h_arr.size == 0:
        raise ValueError("h_ice_values must not be empty")

    k_s_arr = np.array([
        equivalent_sand_roughness(h, k_s0, alpha_ice, beta_ice) for h in h_arr
    ])
    delta_cf_arr = np.array([
        ittc_1978_roughness_penalty(k, k_s0, ship_length, alpha_ittc)
        for k in k_s_arr
    ])

    return {
        "k_s_p50": float(np.percentile(k_s_arr, 50)),
        "k_s_p90": float(np.percentile(k_s_arr, 90)),
        "delta_cf_p50": float(np.percentile(delta_cf_arr, 50)),
        "delta_cf_p90": float(np.percentile(delta_cf_arr, 90)),
    }


# ---------------------------------------------------------------------------
# ITTC-1978 reference formulation (canonical), used as the verification truth
# ---------------------------------------------------------------------------
#: Validity window of the ITTC-1978 roughness allowance in k_s / L
ITTC_K_OVER_L_MIN = 1.0e-6
ITTC_K_OVER_L_MAX = 1.0e-4


def ittc_1978_delta_cf(
    k_s: float,
    ship_length: float,
    strict: bool = False,
) -> float:
    """Canonical ITTC-1978 roughness allowance of one surface [dimensionless].

    ``Delta C_f(k_s) = (105 * (k_s/L)**(1/3) - 0.64) * 1e-3``

    Parameters
    ----------
    k_s : float
        Equivalent sand-grain roughness [m].
    ship_length : float
        Ship length [m].
    strict : bool
        If True, raise when ``k_s / L`` falls outside the ITTC-1978 validity
        window ``[1e-6, 1e-4]``.

    Returns
    -------
    float
        Roughness allowance (dimensionless).
    """
    k_s = float(k_s)
    L = float(ship_length)
    if k_s < 0.0:
        raise ValueError("roughness must be non-negative")
    if L <= 0.0:
        raise ValueError("ship length must be positive")
    ratio = k_s / L
    if strict and not (ITTC_K_OVER_L_MIN <= ratio <= ITTC_K_OVER_L_MAX):
        raise ValueError(
            f"k_s/L={ratio:.3e} outside ITTC-1978 validity window "
            f"[{ITTC_K_OVER_L_MIN:g}, {ITTC_K_OVER_L_MAX:g}]"
        )
    return float((105.0 * ratio ** (1.0 / 3.0) - 0.64) * 1e-3)


def ittc_1957_cf(reynolds: float) -> float:
    """ITTC-1957 friction coefficient ``C_F = 0.075 / (log10(Re) - 2)^2``."""
    re = float(reynolds)
    if re <= 1.0e4:
        raise ValueError("Reynolds number must exceed 1e4 for ITTC-1957")
    return float(0.075 / (np.log10(re) - 2.0) ** 2)


def monte_carlo_envelope(
    ship_length: float = DEFAULT_SHIP_LENGTH_M,
    duration_s: float = 6 * 3600.0,
    n_steps: int = 400,
    n_samples: int = 200,
    seed: int = 20261110,
    undercooling_k_range=(0.5, 3.0),
    u_star_range=(0.01, 0.06),
    c_f_range=(1e-6, 1e-4),
    eta_stick_range=(0.3, 1.0),
    t_water_k: float = 273.15 + 0.2,
    k_s0: float = DEFAULT_HULL_ROUGHNESS_M,
    alpha_ice: float = ALPHA_ICE,
    beta_ice: float = BETA_ICE,
    alpha_ittc: float = ITTC_DELTA_CF_ALPHA,
    cf0: float = DEFAULT_CF0,
    method: str = "stefan_split",
) -> dict:
    """Time-resolved P50/P90 envelope of ``h_ice``, ``k_s`` and ``Delta C_f``.

    Uncertainty of the driving state (undercooling, near-wall shear, frazil
    concentration, sticking efficiency) is propagated through the growth ODE by
    Monte Carlo.  Unlike :func:`p50_p90_envelope` — which only percentiles an
    already-sampled ``h_ice`` array — this returns an *envelope in time*, which is
    what the resistance/propulsion chain needs to bound an ice accretion event.

    All parameters are drawn independently and uniformly from the given ranges
    (``c_f`` log-uniform, because frazil concentrations span decades).

    Returns
    -------
    dict
        ``t_s`` (list), ``h_ice_p50/p90``, ``k_s_p50/p90``, ``delta_cf_p50/p90``
        (lists, same length as ``t_s``), plus scalar ``h_ice_final_p50/p90`` etc.
    """
    from .growth import solve_growth  # local import: avoids import cycle

    rng = np.random.default_rng(int(seed))
    n_samples = int(n_samples)
    if n_samples < 10:
        raise ValueError("n_samples must be >= 10 for a meaningful envelope")

    dt = np.log(c_f_range[1] / c_f_range[0])
    h_curves = np.zeros((n_samples, int(n_steps) + 1))
    for i in range(n_samples):
        dT = rng.uniform(*undercooling_k_range)
        u_s = rng.uniform(*u_star_range)
        c_f = c_f_range[0] * np.exp(rng.uniform(0.0, dt))
        eta = rng.uniform(*eta_stick_range)
        res = solve_growth(
            undercooling_k=float(dT),
            u_star=float(u_s),
            c_f=float(c_f),
            t_max=float(duration_s),
            eta_stick=float(eta),
            n_steps=int(n_steps),
            t_water_k=float(t_water_k),
            method=method,
        )
        h_curves[i] = res.h_ice

    t = np.linspace(0.0, float(duration_s), int(n_steps) + 1)
    k_curves = k_s0 + alpha_ice * h_curves**beta_ice
    dcf_curves = alpha_ittc * (
        (k_curves / ship_length) ** (1.0 / 3.0) - (k_s0 / ship_length) ** (1.0 / 3.0)
    )

    def _b(prefix, arr):
        return {
            f"{prefix}_p50": np.percentile(arr, 50, axis=0).tolist(),
            f"{prefix}_p90": np.percentile(arr, 90, axis=0).tolist(),
        }

    out = {"t_s": t.tolist(), "n_samples": n_samples, "seed": int(seed),
           "method": method, "ship_length_m": float(ship_length)}
    out.update(_b("h_ice", h_curves))
    out.update(_b("k_s", k_curves))
    out.update(_b("delta_cf", dcf_curves))
    out["h_ice_final_p50"] = float(np.percentile(h_curves[:, -1], 50))
    out["h_ice_final_p90"] = float(np.percentile(h_curves[:, -1], 90))
    out["delta_t_over_t_final_p50"] = float(np.percentile(dcf_curves[:, -1], 50) / cf0)
    out["delta_t_over_t_final_p90"] = float(np.percentile(dcf_curves[:, -1], 90) / cf0)
    return out
