"""Stage 3 -- multirate field-exchange error audit (FEM 100 Hz, DEM 1 kHz, LBM 10 kHz).

When three solvers run at different rates, the exchanged field is only known at
the coarse instants and must be reconstructed on the fine rate.  This module
quantifies what that reconstruction costs in the quantities the conservation
audit cares about: the exchanged **impulse** ``int v dt`` (momentum) and the
exchanged **work** ``int tau v dt`` (energy), not just the pointwise L2 error.

Two reconstruction policies are compared:

``linear``
    piecewise linear between coarse samples (the usual default of coupling
    frameworks); second-order accurate.
``cubic``
    natural cubic spline through the coarse samples; fourth-order accurate for
    smooth data.

The audit is run twice: on an analytic broadband signal with a known spectrum
(so the order of accuracy can be measured), and on the *measured* coupled
history produced by the FEM->DEM->LBM chain, so the error band quoted in the
report is the one that actually applies to this case.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

__all__ = [
    "RATES_HZ",
    "reconstruct",
    "signal_errors",
    "multirate_sweep",
    "analytic_signal",
    "RATE_AUDIT",
]

RATES_HZ = {"fem": 100.0, "dem": 1000.0, "lbm": 10000.0}


def reconstruct(t_src: np.ndarray, v_src: np.ndarray, t_tgt: np.ndarray, scheme: str):
    t_src = np.asarray(t_src, dtype=float)
    v_src = np.asarray(v_src, dtype=float)
    t_tgt = np.asarray(t_tgt, dtype=float)
    if scheme == "linear":
        return np.interp(t_tgt, t_src, v_src)
    if scheme == "cubic":
        return CubicSpline(t_src, v_src, bc_type="natural")(t_tgt)
    raise ValueError(f"unknown scheme {scheme!r}")


def signal_errors(
    t_ref: np.ndarray,
    v_ref: np.ndarray,
    tau_ref: np.ndarray,
    t_src: np.ndarray,
    v_src: np.ndarray,
    tau_src: np.ndarray,
    scheme: str,
    t_tgt: np.ndarray | None = None,
) -> dict:
    """Reconstruction error in value, impulse and work."""
    t_tgt = t_ref if t_tgt is None else t_tgt
    v_rec = reconstruct(t_src, v_src, t_tgt, scheme)
    tau_rec = reconstruct(t_src, tau_src, t_tgt, scheme)
    v_true = np.interp(t_tgt, t_ref, v_ref)
    tau_true = np.interp(t_tgt, t_ref, tau_ref)

    def _rel(a, b):
        den = np.trapezoid(np.abs(b), t_tgt)
        return float(abs(np.trapezoid(a, t_tgt) - np.trapezoid(b, t_tgt)) / max(den, 1e-300))

    l2 = float(np.linalg.norm(v_rec - v_true) / max(np.linalg.norm(v_true), 1e-300))
    return {
        "scheme": scheme,
        "n_src": int(len(t_src)),
        "n_tgt": int(len(t_tgt)),
        "l2_rel": l2,
        "impulse_rel": _rel(v_rec, v_true),
        "work_rel": _rel(tau_rec * v_rec, tau_true * v_true),
        "momentum_ledger_rel": _rel(v_rec, v_true),  # same functional as the impulse
    }


def analytic_signal(t_end: float = 0.02):
    """Broadband but resolvable test signal (frequencies 400 / 1200 / 2600 Hz).

    The highest component is below the 5 kHz Nyquist of the 10 kHz LBM rate but
    above the 50 Hz Nyquist of the 100 Hz FEM rate, which is the point: the FEM
    rate alone cannot carry it, so any interpolation from FEM data is lossy.
    """
    def v(t):
        t = np.asarray(t, dtype=float)
        return (
            1.0 * np.sin(2 * np.pi * 400 * t)
            + 0.4 * np.sin(2 * np.pi * 1200 * t + 0.7)
            + 0.2 * np.sin(2 * np.pi * 2600 * t + 1.9)
        ) * np.exp(-t / 0.02)

    def tau(t):
        t = np.asarray(t, dtype=float)
        return 1.0 + 0.3 * np.exp(-t / 0.01) * np.cos(2 * np.pi * 800 * t)

    return v, tau


def multirate_sweep(
    f_true: float = 20000.0,
    rates_hz: dict | None = None,
    t_end: float = 0.02,
    schemes=("linear", "cubic"),
    v_fn=None,
    tau_fn=None,
):
    """Measure reconstruction error and observed order per scheme."""
    rates_hz = dict(RATES_HZ if rates_hz is None else rates_hz)
    if v_fn is None or tau_fn is None:
        av, atau = analytic_signal(t_end)
        v_fn = av if v_fn is None else v_fn
        tau_fn = atau if tau_fn is None else tau_fn

    n_true = int(round(f_true * t_end)) + 1
    t_true = np.linspace(0.0, t_end, n_true)
    v_true = v_fn(t_true)
    tau_true = tau_fn(t_true)

    results = []
    for name, f in rates_hz.items():
        n = max(int(round(f * t_end)) + 1, 3)
        t_src = np.linspace(0.0, t_end, n)
        v_src = v_fn(t_src)
        tau_src = tau_fn(t_src)
        for scheme in schemes:
            err = signal_errors(t_true, v_true, tau_true, t_src, v_src, tau_src, scheme)
            err.update({"rate_name": name, "rate_hz": f, "dt": t_end / (n - 1)})
            results.append(err)

    # observed order for each scheme from a rate sweep (both directions)
    orders = {}
    for scheme in schemes:
        rates = sorted({r["rate_hz"] for r in results})
        errs = []
        for f in rates:
            r = next(
                x for x in results if x["rate_hz"] == f and x["scheme"] == scheme
            )
            errs.append(r["l2_rel"])
        orders[scheme] = rates, errs
    return {
        "rates_hz": rates_hz,
        "t_end": t_end,
        "f_true": f_true,
        "results": results,
        "scheme_curves": orders,
    }


def observed_order(rates, errs) -> float:
    """Least-squares slope of log(err) vs log(dt) (dt = 1/rate)."""
    x = np.log(1.0 / np.asarray(rates, dtype=float))
    y = np.log(np.maximum(np.asarray(errs, dtype=float), 1e-300))
    return float(np.polyfit(x, y, 1)[0])


# Audit table used by the report: which exchange in this case chain has which rate.
RATE_AUDIT = [
    {
        "exchange": "FEM -> DEM",
        "producer_hz": RATES_HZ["fem"],
        "consumer_hz": RATES_HZ["dem"],
        "ratio": RATES_HZ["dem"] / RATES_HZ["fem"],
        "field": "failure-face traction / curvature",
    },
    {
        "exchange": "DEM -> LBM",
        "producer_hz": RATES_HZ["dem"],
        "consumer_hz": RATES_HZ["lbm"],
        "ratio": RATES_HZ["lbm"] / RATES_HZ["dem"],
        "field": "wall velocity v_ice(t)",
    },
]