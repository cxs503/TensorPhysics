"""Thin hand-off interface: attached-ice state -> resistance / propulsion chain.

The resistance and propulsion models (ITTC-1978 added-resistance chains,
propeller open-water corrections, ship-performance-platform) live *outside*
TensorPhysics.  This module is deliberately the only bridge: it exposes the
attached-ice state as a versioned, JSON-serialisable payload and writes no
domain resistance formula of its own.

Payload contract
----------------
``schema``  ``"tensorphysics.attached_ice/1"`` — bump on any field change.
``inputs``  echo of the physical inputs, each as ``{value, unit}``.
``time_series`` decimated trajectory: ``t_s``, ``h_ice_m``, ``k_s_m``,
    ``delta_cf`` (ITTC-1978 roughness penalty increment, dimensionless).
``scalars``  end-of-window values plus the closed-form equilibrium state and
    the detachment regime diagnostic.
``envelope`` P50/P90 of the end-of-window values from a Monte Carlo over the
    uncertain forcing (see :func:`tensorphysics.ice.attached_ice.roughness.monte_carlo_envelope`).
``limitations`` explicit list, so a downstream consumer cannot silently treat
    a model idealisation as validated physics.

All numbers are plain Python floats/ints (no NumPy scalars) so that
``json.dumps(payload)`` round-trips without a custom encoder.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np

from . import (
    C_P_WATER, DEFAULT_BOND_STRENGTH_PA, DEFAULT_HULL_ROUGHNESS_M,
    DEFAULT_SHIP_LENGTH_M, K_ICE, L_FUSION, RHO_ICE, RHO_WATER, T_FREEZE_FW,
)
from .detach import (
    bond_strength_from_adfreeze_pa, detachment_regime, friction_velocity_from_speed,
)
from .growth import solve_growth, equilibrium_thickness
from .roughness import (
    ALPHA_ICE, BETA_ICE, DEFAULT_CF0, ITTC_DELTA_CF_ALPHA,
    ittc_1978_roughness_penalty, ittc_1978_thrust_loss, monte_carlo_envelope,
)

SCHEMA = "tensorphysics.attached_ice/1"

LIMITATIONS = (
    "Frazil flux is a deposition-velocity closure (turbulent + Stokes), not a "
    "solved dispersed two-phase field.",
    "The thermal model is one lumped interface energy balance; no in-depth "
    "temperature profile inside the ice layer or the hull plating.",
    "Shear detachment uses a Frederking-type bond strength; at ship scale the "
    "required friction velocity is O(20 m/s), so the active removal mechanism "
    "is melting, not stripping (see scalars.detachment_regime).",
    "Roughness mapping k_s = k_s0 + alpha*h_ice is a linear surrogate; no "
    "calibrated attached-ice roughness database exists in this module.",
    "Thrust loss is the first-order Delta C_f / C_denominator propagation, valid "
    "only for delta_t_over_t_linearization_ok; pass the consumer's total "
    "resistance coefficient as cf_total instead of the friction-only fallback.",
)


def _v(x, unit):
    return {"value": float(x), "unit": unit}


def _plain(d):
    """numpy-free copy of a dict of scalars, preserving ``bool``."""
    out = {}
    for k, v in d.items():
        if isinstance(v, bool):
            out[k] = bool(v)
        elif isinstance(v, (int, float, np.integer, np.floating)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def build_handoff(
    undercooling_k: float,
    duration_s: float,
    water_temperature_k: float = T_FREEZE_FW + 0.2,
    u_star: Optional[float] = None,
    speed_m_s: Optional[float] = None,
    c_f: float = 1.0e-4,
    eta_stick: float = 1.0,
    ship_length_m: float = DEFAULT_SHIP_LENGTH_M,
    k_s0: float = DEFAULT_HULL_ROUGHNESS_M,
    ice_temperature_c: Optional[float] = None,
    bond_strength_pa: Optional[float] = DEFAULT_BOND_STRENGTH_PA,
    contact_time_s: float = 1.0e4,
    cf_total: Optional[float] = None,
    n_steps: int = 400,
    method: str = "stefan_split",
    n_mc: int = 0,
    seed: int = 20261110,
) -> dict:
    """Build the JSON payload consumed by the resistance / propulsion chain.

    Parameters
    ----------
    undercooling_k : float
        T_freeze - T_surface [K]; the driving temperature difference.
    duration_s : float
        Accretion window [s].
    water_temperature_k : float
        Bulk water temperature [K] (drives the melting flux).
    u_star : float, optional
        Friction velocity [m/s].  If omitted, it is derived from ``speed_m_s``.
    speed_m_s : float, optional
        Ship speed [m/s] (alternative to ``u_star``, cf = 0.0015).
    c_f : float
        Frazil volume concentration [-].
    eta_stick : float
        Sticking efficiency [-].
    ship_length_m : float
        Reference length for the ITTC-1978 allowance [m].
    k_s0 : float
        Clean-hull equivalent sand-grain roughness [m].
    ice_temperature_c : float, optional
        If given, the bond strength is taken from the adfreeze (Frederking 1984)
        model at this joint temperature instead of ``bond_strength_pa``.
    bond_strength_pa : float, optional
        Explicit bond strength [Pa].
    contact_time_s : float
        Contact time for the adfreeze bond [s].
    cf_total : float, optional
        Total resistance coefficient supplied by the consumer; used as the
        denominator of Delta T / T (recommended).  Defaults to a friction-only
        estimate (conservative upper bound).
    n_steps, method : int, str
        Growth integrator controls.
    n_mc : int
        If > 0, additionally attach a Monte Carlo P50/P90 envelope with this
        many samples.
    seed : int
        Monte Carlo seed.

    Returns
    -------
    dict
        Versioned payload, JSON-serialisable.
    """
    if u_star is None and speed_m_s is None:
        raise ValueError("provide either u_star or speed_m_s")
    if u_star is None:
        u_star = friction_velocity_from_speed(speed_m_s)

    if ice_temperature_c is not None:
        bond_pa = bond_strength_from_adfreeze_pa(
            ice_temperature_c=ice_temperature_c, contact_time_s=contact_time_s
        )
    else:
        bond_pa = float(bond_strength_pa if bond_strength_pa is not None
                        else DEFAULT_BOND_STRENGTH_PA)

    res = solve_growth(
        undercooling_k=undercooling_k,
        u_star=u_star,
        c_f=c_f,
        t_max=duration_s,
        eta_stick=eta_stick,
        n_steps=n_steps,
        t_water_k=water_temperature_k,
        tau_bond=bond_pa,
        method=method,
    )

    # identical mapping to roughness.equivalent_sand_roughness / roughness penalty
    k_s = k_s0 + ALPHA_ICE * np.asarray(res.h_ice) ** BETA_ICE
    delta_cf = ITTC_DELTA_CF_ALPHA * (
        (np.asarray(k_s) / ship_length_m) ** (1.0 / 3.0)
        - (k_s0 / ship_length_m) ** (1.0 / 3.0)
    )
    # cross-check against the module function (guards against drift)
    chk = ittc_1978_roughness_penalty(float(k_s[-1]), k_s0, ship_length_m)
    if abs(chk - float(delta_cf[-1])) > 1e-12 * max(abs(chk), 1e-30):
        raise AssertionError("roughness mapping drift between handoff and roughness.py")

    eq = equilibrium_thickness(
        undercooling_k=undercooling_k,
        u_star=u_star,
        c_f=c_f,
        eta_stick=eta_stick,
        t_water_k=water_temperature_k,
        tau_bond=bond_pa,
    )
    regime = detachment_regime(u_star=u_star, tau_bond=bond_pa, h_ice=res.h_final)

    stride = max(1, int(np.ceil(len(res.t) / 200.0)))
    payload = {
        "schema": SCHEMA,
        "inputs": {
            "undercooling_K": _v(undercooling_k, "K"),
            "duration_s": _v(duration_s, "s"),
            "water_temperature_K": _v(water_temperature_k, "K"),
            "u_star_m_s": _v(u_star, "m/s"),
            "speed_m_s": _v(float(speed_m_s) if speed_m_s is not None else np.nan, "m/s"),
            "frazil_concentration": _v(c_f, "-"),
            "sticking_efficiency": _v(eta_stick, "-"),
            "ship_length_m": _v(ship_length_m, "m"),
            "k_s0_m": _v(k_s0, "m"),
            "bond_strength_Pa": _v(bond_pa, "Pa"),
            "cf_total": _v(float(cf_total) if cf_total is not None else DEFAULT_CF0, "-"),
        },
        "time_series": {
            "t_s": [float(x) for x in res.t[::stride]],
            "h_ice_m": [float(x) for x in res.h_ice[::stride]],
            "k_s_m": [float(x) for x in k_s[::stride]],
            "delta_cf": [float(x) for x in delta_cf[::stride]],
        },
        "scalars": {
            "h_ice_end_m": float(res.h_final),
            "h_ice_max_m": float(np.max(res.h_ice)),
            "k_s_end_m": float(k_s[-1]),
            "delta_cf_end": float(delta_cf[-1]),
            "delta_t_over_t_end": float(
                ittc_1978_thrust_loss(
                    float(k_s[-1]), k_s0, ship_length_m,
                    cf0=DEFAULT_CF0, c_total=cf_total,
                )
            ),
            # first-order propagation is only meaningful while the penalty is a
            # small perturbation of the denominator resistance coefficient
            "delta_t_over_t_linearization_ok": bool(
                float(ittc_1978_thrust_loss(
                    float(k_s[-1]), k_s0, ship_length_m,
                    cf0=DEFAULT_CF0, c_total=cf_total)) < 0.2
            ),
            "detached_during_window": bool(res.detached),
            "equilibrium": _plain(eq),
            "detachment_regime": _plain(regime),
        },
        "envelope": None,
        "limitations": list(LIMITATIONS),
    }

    if n_mc and int(n_mc) > 0:
        env = monte_carlo_envelope(
            ship_length=ship_length_m,
            duration_s=duration_s,
            n_steps=n_steps,
            n_samples=int(n_mc),
            seed=seed,
            t_water_k=water_temperature_k,
            k_s0=k_s0,
            method=method,
        )
        payload["envelope"] = {
            "n_samples": int(n_mc),
            "seed": int(seed),
            "h_ice_final_p50_m": float(env["h_ice_final_p50"]),
            "h_ice_final_p90_m": float(env["h_ice_final_p90"]),
            "delta_t_over_t_final_p50": float(env["delta_t_over_t_final_p50"]),
            "delta_t_over_t_final_p90": float(env["delta_t_over_t_final_p90"]),
            "t_s": [float(x) for x in env["t_s"]],
            "h_ice_p50_m": [float(x) for x in env["h_ice_p50"]],
            "h_ice_p90_m": [float(x) for x in env["h_ice_p90"]],
            "k_s_p50_m": [float(x) for x in env["k_s_p50"]],
            "k_s_p90_m": [float(x) for x in env["k_s_p90"]],
        }
    return payload


def to_json(payload: dict, indent: int = 2) -> str:
    """Serialise a hand-off payload, asserting JSON round-trip fidelity."""
    text = json.dumps(payload, indent=indent, sort_keys=False, allow_nan=True)
    # round-trip guard: the consumer parses this string, so it must survive
    if json.loads(text)["schema"] != payload["schema"]:
        raise AssertionError("payload lost its schema tag during serialisation")
    return text


__all__ = ["SCHEMA", "LIMITATIONS", "build_handoff", "to_json"]