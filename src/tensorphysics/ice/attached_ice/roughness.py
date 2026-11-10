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
The ITTC-1978 procedure recommends a roughness allowance for the friction
resistance coefficient::

    Delta C_f = alpha * [(k_s / L)^(1/3) - (k_s0 / L)^(1/3)]

where:
* ``alpha = 0.004`` (ITTC-1978 recommended value)
* ``L`` is the ship length [m]
* ``k_s`` is the roughened equivalent sand-grain roughness [m]
* ``k_s0`` is the baseline (clean hull) roughness [m]

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
    h_ice : float
        Ice thickness [m].
    k_s0 : float
        Clean-hull roughness [m].
    alpha_ice : float
        Ice roughness coefficient.
    beta_ice : float
        Ice roughness exponent.

    Returns
    -------
    float
        Equivalent sand-grain roughness [m].
    """
    h = float(h_ice)
    if h < 0.0:
        raise ValueError("ice thickness must be non-negative")
    if k_s0 < 0.0:
        raise ValueError("baseline roughness must be non-negative")

    return float(k_s0 + alpha_ice * h**beta_ice)


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
) -> float:
    """Estimate thrust loss fraction Delta T / T.

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
        Clean-hull friction coefficient.

    Returns
    -------
    float
        Thrust loss fraction (dimensionless).
    """
    delta_cf = ittc_1978_roughness_penalty(k_s, k_s0, ship_length, alpha)
    cf0 = float(cf0)
    if cf0 <= 0.0:
        raise ValueError("clean-hull friction coefficient must be positive")

    return float(delta_cf / cf0)


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
