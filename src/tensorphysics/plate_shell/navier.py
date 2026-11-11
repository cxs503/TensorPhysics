"""Navier double-series exact solution for a simply-supported plate (SSSS).

Kirchhoff thin-plate theory.  For a rectangular plate ``a x b`` (thickness
``t``, ``D = E t^3 / (12 (1 - nu^2))``) with all four edges simply supported and
a uniform pressure ``q`` the Levy/Navier solution is (Timoshenko &
Woinowsky-Krieger, *Theory of Plates and Shells*, 2nd ed., §30 for the
deflection, §44 for the uniform-load case):

.. math::

    w(x,y) = \\sum_{m,n\\ \\text{odd}} A_{mn}\\sin\\frac{m\\pi x}{a}
             \\sin\\frac{n\\pi y}{b}, \\qquad
    A_{mn} = \\frac{16 q}{\\pi^6 D\\, m n\\left(\\frac{m^2}{a^2}
             + \\frac{n^2}{b^2}\\right)^2}

because the uniform load itself expands as
``q = sum_{m,n odd} 16 q / (pi^2 m n) sin sin``.

Derived fields (all from the same series, so they are mutually consistent):

* bending moments ``Mx = -D (w,xx + nu w,yy)``, ``My = -D (w,yy + nu w,xx)``
* twisting moment ``Mxy = -D (1 - nu) w,xy``
* shear forces ``Qx = -D d/dx(nabla^2 w)``, ``Qy = -D d/dy(nabla^2 w)``

The moment series converge like ``m^-4`` but the shear/twist series only like
``m^-3``; the default ``n_terms`` is therefore generous and
:func:`series_convergence` is provided so callers can *prove* the truncation
error is below their gate instead of assuming it.

Sign conventions (identical to the finite element module): ``w`` positive
along ``+z`` (pressure acts in ``+z``), sagging moments positive, ``Qx``
positive on a face with outward normal ``+x`` in the ``+z`` direction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np

__all__ = [
    "navier_plate_fields",
    "navier_plate_center",
    "navier_alpha_coefficient",
    "series_convergence",
    "NAVIER_ALPHA_SQUARE_PUBLISHED",
]

#: Published centre-deflection coefficient for a uniformly loaded simply
#: supported square plate, ``w_max = alpha q a^4 / D`` (Timoshenko &
#: Woinowsky-Krieger, Theory of Plates and Shells, 2nd ed., Table 8 / §44;
#: also quoted as 0.00406 in most handbooks).
NAVIER_ALPHA_SQUARE_PUBLISHED = 0.00406


def _odd_terms(n_terms: int) -> np.ndarray:
    n_terms = int(n_terms)
    if n_terms < 1:
        raise ValueError("n_terms must be >= 1")
    return np.arange(1, 2 * n_terms, 2, dtype=float)


def _coefficients(a: float, b: float, t: float, E: float, nu: float, q: float,
                  n_terms: int) -> Dict[str, np.ndarray]:
    """Series coefficients ``A_mn`` and the modal wavenumbers."""
    if a <= 0 or b <= 0 or t <= 0:
        raise ValueError("a, b, t must be positive")
    D = E * t ** 3 / (12.0 * (1.0 - nu ** 2))
    odd = _odd_terms(n_terms)
    m, n = np.meshgrid(odd, odd, indexing="ij")
    alpha = m * math.pi / a
    beta = n * math.pi / b
    denom = m * n * (alpha ** 2 + beta ** 2) ** 2
    A = 16.0 * q / (math.pi ** 2 * D) / denom
    return {"A": A, "alpha": alpha, "beta": beta, "m": m, "n": n, "D": np.array(D)}


def navier_plate_fields(a: float, b: float, t: float, E: float, nu: float,
                        q: float, x, y, n_terms: int = 61,
                        ) -> Dict[str, np.ndarray]:
    """Evaluate the Navier series fields at points ``(x, y)``.

    Parameters
    ----------
    a, b, t, E, nu : float
        Plate side lengths [m], thickness [m], Young's modulus [Pa],
        Poisson's ratio.
    q : float
        Uniform pressure [Pa] acting in ``+z``.
    x, y : array_like
        Evaluation points [m]; any broadcastable shape.
    n_terms : int
        Number of odd half-wave numbers kept **per direction** (so
        ``n_terms**2`` terms).  The default 61 (3721 terms) keeps the shear
        truncation error below ~1e-4 relative away from the edges.

    Returns
    -------
    dict of ndarray
        ``w`` [m], ``Mx``, ``My``, ``Mxy`` [N m/m], ``Qx``, ``Qy`` [N/m],
        each broadcast to the shape of ``x``/``y``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x, y = np.broadcast_arrays(x, y)
    c = _coefficients(a, b, t, E, nu, q, n_terms)
    A, alpha, beta, D = c["A"], c["alpha"], c["beta"], float(c["D"])

    # trigonometric tables: (n_terms, *shape)
    sx = np.sin(np.multiply.outer(alpha[:, 0], x.ravel())).reshape(
        alpha.shape[0], *x.shape)
    cx = np.cos(np.multiply.outer(alpha[:, 0], x.ravel())).reshape(
        alpha.shape[0], *x.shape)
    sy = np.sin(np.multiply.outer(beta[0, :], y.ravel())).reshape(
        beta.shape[1], *y.shape)
    cy = np.cos(np.multiply.outer(beta[0, :], y.ravel())).reshape(
        beta.shape[1], *y.shape)

    am, bn = np.meshgrid(alpha[:, 0], beta[0, :], indexing="ij")
    k2 = am ** 2 + bn ** 2

    w = np.einsum("mn,m...,n...->...", A, sx, sy)
    Mx = D * np.einsum("mn,m...,n...->...", A * (am ** 2 + nu * bn ** 2), sx, sy)
    My = D * np.einsum("mn,m...,n...->...", A * (bn ** 2 + nu * am ** 2), sx, sy)
    Mxy = -D * (1.0 - nu) * np.einsum("mn,m...,n...->...",
                                      A * am * bn, cx, cy)
    Qx = D * np.einsum("mn,m...,n...->...", A * am * k2, cx, sy)
    Qy = D * np.einsum("mn,m...,n...->...", A * bn * k2, sx, cy)
    return {"w": w, "Mx": Mx, "My": My, "Mxy": Mxy, "Qx": Qx, "Qy": Qy}


def navier_plate_center(a: float, b: float, t: float, E: float, nu: float,
                        q: float, n_terms: int = 61) -> Dict[str, float]:
    """Closed-form centre values ``(a/2, b/2)`` of the Navier fields."""
    f = navier_plate_fields(a, b, t, E, nu, q, 0.5 * a, 0.5 * b,
                            n_terms=n_terms)
    D = E * t ** 3 / (12.0 * (1.0 - nu ** 2))
    return {
        "w": float(f["w"]),
        "Mx": float(f["Mx"]),
        "My": float(f["My"]),
        "Qx": float(f["Qx"]),
        "Qy": float(f["Qy"]),
        "D": float(D),
        "alpha_deflection": float(f["w"] * D / (q * a ** 4)),
    }


def navier_alpha_coefficient(n_terms: int = 61) -> float:
    """Dimensionless centre deflection ``alpha = w_max D / (q a^4)`` (square)."""
    return navier_plate_center(1.0, 1.0, 0.01, 1.0, 0.3, 1.0,
                               n_terms=n_terms)["alpha_deflection"]


def series_convergence(a: float, b: float, t: float, E: float, nu: float,
                       q: float, x, y, n_terms_list=(11, 21, 41, 61, 101),
                       ) -> Dict[str, list]:
    """Relative change of each field between successive truncation orders.

    Returns ``{"n_terms": [...], "w": [...], ...}`` where each value is the
    relative difference to the previous (coarser) truncation.  Used as an
    auditable proof that the analytic reference is converged to well below
    the acceptance gates.
    """
    out: Dict[str, list] = {"n_terms": list(n_terms_list)}
    prev = None
    keys = ("w", "Mx", "My", "Qx", "Qy")
    for k in keys:
        out[k] = []
    for nt in n_terms_list:
        cur = navier_plate_fields(a, b, t, E, nu, q, x, y, n_terms=nt)
        for k in keys:
            if prev is None:
                out[k].append(float("nan"))
            else:
                ref = np.max(np.abs(cur[k]))
                ref = ref if ref > 0 else 1.0
                out[k].append(float(np.max(np.abs(cur[k] - prev[k])) / ref))
        prev = cur
    return out