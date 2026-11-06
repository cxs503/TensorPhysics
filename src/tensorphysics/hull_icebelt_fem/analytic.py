"""Closed-form Navier solution of the simply supported plate under a patch load.

The double sine series is the *independent* reference for the finite element
results: it solves the Kirchhoff plate equation analytically for any piecewise
constant lateral load on a rectangular ``SSSS`` panel, so the same formula
covers both verification cases of the ice-belt cross-check (uniform pressure
and the IACS ice patch).

For a deflection expansion ``w = sum_mn W_mn sin(m pi x/a) sin(n pi y/b)`` the
Kirchhoff equation ``D laplacian^2 w = q`` gives

.. math::

    W_{mn} = \\frac{q_{mn}}{D \\pi^4 (m^2/a^2 + n^2/b^2)^2},
    \\qquad
    q_{mn} = \\frac{4}{ab}\\int_0^a\\!\\!\\int_0^b q \\sin\\frac{m\\pi x}{a}
             \\sin\\frac{n\\pi y}{b}\\,dy\\,dx

and for a load that is constant ``q`` on a rectangle ``[x0,x1] x [y0,y1]``

.. math::

    q_{mn} = \\frac{4 q}{ab}\\,\\frac{a}{m\\pi}\\frac{b}{n\\pi}
             [\\cos\\tfrac{m\\pi x_0}{a} - \\cos\\tfrac{m\\pi x_1}{a}]
             [\\cos\\tfrac{n\\pi y_0}{b} - \\cos\\tfrac{n\\pi y_1}{b}].

Moments follow from ``M_x = -D(w,xx + nu w,yy)``, ``M_y = -D(w,yy + nu w,xx)``
and the surface bending stress is ``sigma = 6 M / t^2``.  Signs are consistent
with the FE convention of :mod:`tensorphysics.plate_buckling.core`
(``kappa = [-w,xx, -w,yy, -2w,xy]``), so a sagging panel gives positive
moments in both chains.

The series converges slowly for a *local* patch (Gibbs-type overshoot next to
the load discontinuity), therefore the number of terms is an explicit argument
and the consumers report the term count they used.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np

from ..plate_buckling import PlateGeometry

__all__ = ["navier_field", "navier_grid", "patch_amplitudes"]

Box = Tuple[float, float, float, float]


def patch_amplitudes(geom: PlateGeometry, boxes: Sequence[Box], q: float,
                     m: int, n: int) -> float:
    """Sine-series amplitude ``q_mn`` of a piecewise constant load."""
    a, b = geom.a, geom.b
    total = 0.0
    for (x0, x1, y0, y1) in boxes:
        ix = (a / (m * math.pi)) * (math.cos(m * math.pi * x0 / a)
                                    - math.cos(m * math.pi * x1 / a))
        iy = (b / (n * math.pi)) * (math.cos(n * math.pi * y0 / b)
                                    - math.cos(n * math.pi * y1 / b))
        total += ix * iy
    return 4.0 * q / (a * b) * total


def navier_field(geom: PlateGeometry, boxes: Sequence[Box], q: float,
                 x: float, y: float, n_terms: int = 121) -> Dict[str, float]:
    """Single-point closed-form solution (deflection and bending moments).

    Parameters
    ----------
    geom:
        Simply supported rectangular plate (``SSSS``: ``w = 0`` and rotations
        free on all four edges, ``M_n = 0``).
    boxes:
        Load rectangles ``(x0, x1, y0, y1)`` carrying the uniform pressure
        ``q`` [Pa].  ``[]`` gives the trivial solution.
    x, y:
        Evaluation point [m].
    n_terms:
        Number of half-waves retained in each direction.

    Returns
    -------
    dict with ``w`` [m], ``Mx``, ``My`` [N] (moment per unit width) and the
    surface stress magnitudes ``sigma_x``, ``sigma_y`` [Pa].
    """
    if q < 0.0:
        raise ValueError(f"pressure must be non-negative, got {q!r}")
    D = geom.flexural_rigidity
    d = geom.t
    a, b = geom.a, geom.b
    w = 0.0
    Mx = 0.0
    My = 0.0
    for m in range(1, n_terms + 1):
        am = m * math.pi / a
        for n in range(1, n_terms + 1):
            qmn = patch_amplitudes(geom, boxes, q, m, n)
            if qmn == 0.0:
                continue
            bn = n * math.pi / b
            W = qmn / (D * (am ** 2 + bn ** 2) ** 2)
            sxy = math.sin(am * x) * math.sin(bn * y)
            w += W * sxy
            Mx += D * W * (am ** 2 + geom.nu * bn ** 2) * sxy
            My += D * W * (bn ** 2 + geom.nu * am ** 2) * sxy
    return {
        "w": float(w),
        "Mx": float(Mx),
        "My": float(My),
        "sigma_x": float(6.0 * abs(Mx) / d ** 2),
        "sigma_y": float(6.0 * abs(My) / d ** 2),
    }


def navier_grid(geom: PlateGeometry, boxes: Sequence[Box], q: float,
                xs: Iterable[float], ys: Iterable[float],
                n_terms: int = 61) -> Dict[str, np.ndarray]:
    """Closed-form solution on a grid (for field comparison and maxima).

    Returns arrays of shape ``(len(xs), len(ys))``: ``w``, ``Mx``, ``My``.
    """
    if q < 0.0:
        raise ValueError(f"pressure must be non-negative, got {q!r}")
    D = geom.flexural_rigidity
    a, b = geom.a, geom.b
    X = np.asarray(list(xs), dtype=float)
    Y = np.asarray(list(ys), dtype=float)
    SX = np.sin(np.pi * np.outer(X, np.arange(1, n_terms + 1)) / a)   # (nx, n)
    SY = np.sin(np.pi * np.outer(Y, np.arange(1, n_terms + 1)) / b)   # (ny, n)
    w = np.zeros((X.size, Y.size))
    Mx = np.zeros_like(w)
    My = np.zeros_like(w)
    for m in range(1, n_terms + 1):
        am = m * math.pi / a
        rows = []
        for n in range(1, n_terms + 1):
            qmn = patch_amplitudes(geom, boxes, q, m, n)
            if qmn == 0.0:
                rows.append(0.0)
                continue
            bn = n * math.pi / b
            W = qmn / (D * (am ** 2 + bn ** 2) ** 2)
            rows.append(W)
        Wn = np.asarray(rows)                                    # (n,)
        sx = SX[:, m - 1][:, None]                               # (nx, 1)
        bn = np.pi * np.arange(1, n_terms + 1) / b               # (n,)
        w += (sx * Wn[None, :]) @ SY.T
        cx = D * Wn * (am ** 2 + geom.nu * bn ** 2)
        cy = D * Wn * (bn ** 2 + geom.nu * am ** 2)
        Mx += (sx * cx[None, :]) @ SY.T
        My += (sx * cy[None, :]) @ SY.T
    return {"w": w, "Mx": Mx, "My": My}