"""D2Q9 lattice constants and kinetic-theory helpers.

The whole LBM/DEM momentum audit is written in lattice units (``dx = dt = 1``,
``rho0 = 1``) and converted to physical units by the refinement map of
:mod:`tensorphysics.lbm_dem_momentum.gridscan`.  Keeping the solver in lattice
units is what makes the momentum ledger *exact*: every population that leaves a
fluid node is booked with the very same floating point number that the bounce
back wrote, so the residual is round-off and not a modelling choice.

Velocity set (standard D2Q9 ordering)::

    6   2   5
      \\ | /
    3 - 0 - 1
      / | \\
    7   4   8

so ``C[1] = (1, 0)``, ``C[2] = (0, 1)`` and ``OPP[i]`` is the reversed
direction (``OPP[1] = 3``, ``OPP[5] = 7`` ...).
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "C",
    "W",
    "Q",
    "OPP",
    "CS2",
    "E",
    "equilibrium",
    "shift",
    "viscosity",
    "strain_rate_from_noneq",
]

C = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
    dtype=np.int64,
)
W = np.array(
    [4.0 / 9.0]
    + [1.0 / 9.0] * 4
    + [1.0 / 36.0] * 4,
    dtype=np.float64,
)
Q = C.shape[0]
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)
CS2 = 1.0 / 3.0
#: Lattice energy weight of one population, ``c_i . c_i / 2``.
E = 0.5 * np.sum(C * C, axis=1).astype(np.float64)


def _pad(a):
    return np.asarray(a, dtype=np.float64).reshape(-1)


def equilibrium(rho, ux, uy):
    """Second-order Maxwellian at density ``rho`` and velocity ``(ux, uy)``.

    ``rho``, ``ux`` and ``uy`` may be scalars or arrays of the same shape; the
    returned array has shape ``(9,) + shape``.
    """
    rho = np.asarray(rho, dtype=np.float64)
    ux = np.asarray(ux, dtype=np.float64)
    uy = np.asarray(uy, dtype=np.float64)
    shape = np.broadcast(rho, ux, uy).shape
    ux = np.broadcast_to(ux, shape)
    uy = np.broadcast_to(uy, shape)
    usq = ux * ux + uy * uy
    cu = np.empty((Q,) + shape)
    for i in range(Q):
        cu[i] = C[i, 0] * ux + C[i, 1] * uy
    feq = np.empty((Q,) + shape)
    for i in range(Q):
        feq[i] = W[i] * rho * (
            1.0 + cu[i] / CS2 + 0.5 * cu[i] * cu[i] / (CS2 * CS2) - 0.5 * usq / CS2
        )
    return feq


def shift(arr, c, fill):
    """Return ``out[x] = arr[x + c]`` with ``fill`` outside the domain.

    ``arr`` is a 2-D array indexed ``[x, y]``; ``c = (cx, cy)``.
    """
    out = np.full(arr.shape, fill, dtype=arr.dtype)
    cx, cy = int(c[0]), int(c[1])
    nx, ny = arr.shape
    tx0, tx1 = max(0, -cx), min(nx, nx - cx)
    ty0, ty1 = max(0, -cy), min(ny, ny - cy)
    if tx1 <= tx0 or ty1 <= ty0:
        return out
    out[tx0:tx1, ty0:ty1] = arr[tx0 + cx: tx1 + cx, ty0 + cy: ty1 + cy]
    return out


def viscosity(tau: float) -> float:
    """Lattice kinematic viscosity ``nu = cs^2 (tau - 1/2)``."""
    return CS2 * (tau - 0.5)


def strain_rate_from_noneq(f, feq, rho, tau):
    """Rate-of-strain tensor from the non-equilibrium stress (Chapman-Enskog).

    ``Pi^neq_ab = sum_i (f_i - f_i^eq) c_ia c_ib = -2 rho cs^2 tau S_ab`` to
    first order in the Knudsen number, hence
    ``S_ab = -Pi^neq_ab / (2 rho cs^2 tau)``.
    This is used for the *independent* viscous-dissipation estimate
    ``Phi = sum 2 mu S:S`` that closes the energy budget.
    """
    d = f - feq
    sxx = np.zeros(rho.shape)
    syy = np.zeros(rho.shape)
    sxy = np.zeros(rho.shape)
    for i in range(Q):
        sxx += d[i] * C[i, 0] * C[i, 0]
        syy += d[i] * C[i, 1] * C[i, 1]
        sxy += d[i] * C[i, 0] * C[i, 1]
    norm = 2.0 * rho * CS2 * tau
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(norm > 0.0, 1.0 / norm, 0.0)
    Sxx = -sxx * inv
    Syy = -syy * inv
    Sxy = -sxy * inv
    return Sxx, Syy, Sxy