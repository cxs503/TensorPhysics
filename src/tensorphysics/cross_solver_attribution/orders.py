"""Observed-order estimation for the arbitration protocol.

The order is measured in the L2 norm of each solver's **own reconstruction** on a
fine sampling grid, not in the cell averages.  Reason: for smooth periodic
advection the DG cell-average error is superconvergent (order ``2p+1`` for the
mean mode), so a cell-average norm would report ``p = 3`` for a DG-P1 scheme and
make the DG/FVM comparison look like an apples-to-oranges contest.  The
cross-solver *disagreement* ``D`` is still measured on cell averages, the common
space where no reconstruction choice enters (see ``attribution``); the two norms
answer different questions and both are reported.
"""
from __future__ import annotations

import math

import numpy as np

from .cases import ScalarCase, SWECase


def reconstruction_error(case, result, samples_per_cell: int = 8) -> float:
    """Relative L2 error of the solver's reconstructed field vs the exact one."""
    n = result.n
    n_s = n * samples_per_cell
    width = case.x1 - case.x0
    xs = np.linspace(case.x0, case.x1, n_s, endpoint=False) + width / (2.0 * n_s)
    recon = result.extra["reconstruct"](xs)
    if isinstance(case, ScalarCase):
        ref = np.asarray(case.exact(xs, case.t_end), dtype=float)
        return _rel(recon, ref)
    h_ref, u_ref = case.exact(xs, case.t_end)
    ref = np.stack([h_ref, h_ref * u_ref])
    return _rel(recon, ref)


def _rel(num, ref) -> float:
    num = np.asarray(num, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if not np.all(np.isfinite(num)):
        return float("inf")
    den = float(np.sqrt(np.mean(ref ** 2)))
    diff = float(np.sqrt(np.mean((num - ref) ** 2)))
    return diff / den if den > 0 else diff


def observed_order(errors, resolutions) -> float:
    """Least-squares slope of ``log e`` against ``log h``."""
    errors = np.asarray(errors, dtype=float)
    ns = np.asarray(resolutions, dtype=float)
    if not np.all(np.isfinite(errors)) or np.any(errors <= 0.0):
        return float("nan")
    slope = np.polyfit(np.log(1.0 / ns), np.log(errors), 1)[0]
    return float(slope)


def order_study(case, solve_fn, resolutions=(40, 80, 160), **cfg_kwargs) -> dict:
    """Run one solver over a grid sequence and return the observed order."""
    errors = []
    for n in resolutions:
        res = solve_fn(case, n, **cfg_kwargs)
        if not res.finite:
            errors.append(float("inf"))
        else:
            errors.append(reconstruction_error(case, res))
    return {
        "resolutions": list(resolutions),
        "errors": [float(e) for e in errors],
        "observed_order": observed_order(errors, resolutions),
    }


def expected_band(expected: float, tolerance: float = 0.15) -> tuple[float, float]:
    return expected * (1.0 - tolerance), expected * (1.0 + tolerance)


def in_band(observed: float, expected: float, tolerance: float = 0.15) -> bool:
    lo, hi = expected_band(expected, tolerance)
    return bool(math.isfinite(observed) and lo <= observed <= hi)