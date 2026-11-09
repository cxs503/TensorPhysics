"""Isoparametric shape functions for 3-D solid elements (Hex8 / Hex20).

Both element families use the same natural coordinate system
``xi, eta, zeta in [-1, 1]^3``.

Node ordering (right-handed, zeta = +-1 are the "top" / "bottom" faces):

Hex8  (trilinear):
    0 (-1,-1,-1)  1 (+1,-1,-1)  2 (+1,+1,-1)  3 (-1,+1,-1)
    4 (-1,-1,+1)  5 (+1,-1,+1)  6 (+1,+1,+1)  7 (-1,+1,+1)

Hex20 (quadratic serendipity): corners 0..7 as above, then edge mid-side nodes
     8 (0,-1,-1)  9 (+1,0,-1) 10 (0,+1,-1) 11 (-1,0,-1)     bottom face
    12 (0,-1,+1) 13 (+1,0,+1) 14 (0,+1,+1) 15 (-1,0,+1)     top face
    16 (-1,-1,0) 17 (+1,-1,0) 18 (+1,+1,0) 19 (-1,+1,0)     vertical edges

Everything below is analytic (no symbolic differentiation) so the derivative
tables can be checked to machine precision against central differences.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "HEX8_NATURAL",
    "HEX20_NATURAL",
    "hex8_shape",
    "hex20_shape",
    "shape",
    "n_nodes",
    "gauss_rule",
    "GAUSS_RULES",
]

#: Natural coordinates of the eight Hex8 nodes.
HEX8_NATURAL = np.array(
    [
        [-1.0, -1.0, -1.0],
        [+1.0, -1.0, -1.0],
        [+1.0, +1.0, -1.0],
        [-1.0, +1.0, -1.0],
        [-1.0, -1.0, +1.0],
        [+1.0, -1.0, +1.0],
        [+1.0, +1.0, +1.0],
        [-1.0, +1.0, +1.0],
    ],
    dtype=float,
)

#: Natural coordinates of the twenty Hex20 nodes.
HEX20_NATURAL = np.vstack(
    [
        HEX8_NATURAL,
        np.array(
            [
                [0.0, -1.0, -1.0],
                [+1.0, 0.0, -1.0],
                [0.0, +1.0, -1.0],
                [-1.0, 0.0, -1.0],
                [0.0, -1.0, +1.0],
                [+1.0, 0.0, +1.0],
                [0.0, +1.0, +1.0],
                [-1.0, 0.0, +1.0],
                [-1.0, -1.0, 0.0],
                [+1.0, -1.0, 0.0],
                [+1.0, +1.0, 0.0],
                [-1.0, +1.0, 0.0],
            ],
            dtype=float,
        ),
    ]
)


def n_nodes(element: str) -> int:
    """Number of nodes of ``element`` (``"hex8"`` or ``"hex20"``)."""
    key = element.lower()
    if key == "hex8":
        return 8
    if key == "hex20":
        return 20
    raise ValueError(f"unsupported solid element {element!r} (expected 'hex8' or 'hex20')")


def hex8_shape(xi: float, eta: float, zeta: float) -> tuple[np.ndarray, np.ndarray]:
    """Trilinear Hex8 shape functions and their natural derivatives.

    Returns
    -------
    N : (8,) array
    dN : (3, 8) array, rows are d/dxi, d/deta, d/dzeta.
    """
    s = np.array([xi, eta, zeta], dtype=float)
    N = np.empty(8)
    dN = np.empty((3, 8))
    for a in range(8):
        sa = HEX8_NATURAL[a]
        fx, fy, fz = 1.0 + sa[0] * xi, 1.0 + sa[1] * eta, 1.0 + sa[2] * zeta
        N[a] = 0.125 * fx * fy * fz
        dN[0, a] = 0.125 * sa[0] * fy * fz
        dN[1, a] = 0.125 * sa[1] * fx * fz
        dN[2, a] = 0.125 * sa[2] * fx * fy
    _ = s
    return N, dN


def hex20_shape(xi: float, eta: float, zeta: float) -> tuple[np.ndarray, np.ndarray]:
    """Quadratic serendipity Hex20 shape functions and natural derivatives.

    Corner nodes use the standard ``(1 + s*s_a)(s_a + s_b + s_c - 2) / 8`` form;
    mid-side nodes use ``(1 - s^2)`` in the parametric direction of the edge.
    """
    N = np.zeros(20)
    dN = np.zeros((3, 20))

    # --- corners -------------------------------------------------------
    for a in range(8):
        sa, sb, sc = HEX8_NATURAL[a]
        x, y, z = 1.0 + sa * xi, 1.0 + sb * eta, 1.0 + sc * zeta
        combo = sa * xi + sb * eta + sc * zeta - 2.0
        N[a] = 0.125 * x * y * z * combo
        dN[0, a] = 0.125 * sa * y * z * (2.0 * sa * xi + sb * eta + sc * zeta - 1.0)
        dN[1, a] = 0.125 * sb * x * z * (sa * xi + 2.0 * sb * eta + sc * zeta - 1.0)
        dN[2, a] = 0.125 * sc * x * y * (sa * xi + sb * eta + 2.0 * sc * zeta - 1.0)

    # --- bottom / top face mid-side nodes (zeta_a = +-1) ---------------
    for a, (sa, sb, sc) in zip(range(8, 16), HEX20_NATURAL[8:16]):
        x, y, z = 1.0 + sa * xi, 1.0 + sb * eta, 1.0 + sc * zeta
        two_dir = 0 if sa == 0.0 else (1 if sb == 0.0 else 2)
        # parametric direction with zero natural coordinate carries the (1 - s^2)
        if two_dir == 0:
            N[a] = 0.25 * (1.0 - xi * xi) * y * z
            dN[0, a] = -0.5 * xi * y * z
            dN[1, a] = 0.25 * (1.0 - xi * xi) * sb * z
            dN[2, a] = 0.25 * (1.0 - xi * xi) * y * sc
        elif two_dir == 1:
            N[a] = 0.25 * x * (1.0 - eta * eta) * z
            dN[0, a] = 0.25 * sa * (1.0 - eta * eta) * z
            dN[1, a] = -0.5 * eta * x * z
            dN[2, a] = 0.25 * x * (1.0 - eta * eta) * sc
        else:
            N[a] = 0.25 * x * y * (1.0 - zeta * zeta)
            dN[0, a] = 0.25 * sa * y * (1.0 - zeta * zeta)
            dN[1, a] = 0.25 * x * sb * (1.0 - zeta * zeta)
            dN[2, a] = -0.5 * zeta * x * y

    # --- vertical edge mid-side nodes (zeta_a = 0) ---------------------
    for a, (sa, sb, _sc) in zip(range(16, 20), HEX20_NATURAL[16:20]):
        x, y = 1.0 + sa * xi, 1.0 + sb * eta
        N[a] = 0.25 * x * y * (1.0 - zeta * zeta)
        dN[0, a] = 0.25 * sa * y * (1.0 - zeta * zeta)
        dN[1, a] = 0.25 * x * sb * (1.0 - zeta * zeta)
        dN[2, a] = -0.5 * zeta * x * y

    return N, dN


_SHAPE_TABLE = {"hex8": hex8_shape, "hex20": hex20_shape}


def shape(element: str, xi: float, eta: float, zeta: float) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to the shape-function routine of ``element``."""
    try:
        fn = _SHAPE_TABLE[element.lower()]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"unsupported solid element {element!r}") from exc
    return fn(xi, eta, zeta)


def _gauss_1d(n: int) -> tuple[np.ndarray, np.ndarray]:
    if n == 1:
        return np.array([0.0]), np.array([2.0])
    if n == 2:
        g = 1.0 / np.sqrt(3.0)
        return np.array([-g, g]), np.array([1.0, 1.0])
    if n == 3:
        g = np.sqrt(0.6)
        return np.array([-g, 0.0, g]), np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    # generic Gauss-Legendre fallback (numpy/scipy free for n > 3)
    from numpy.polynomial.legendre import leggauss

    pts, wts = leggauss(n)
    return pts, wts


def gauss_rule(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Tensor-product Gauss rule of order ``n`` on the parent cube.

    Returns
    -------
    points : (n**3, 3) array
    weights : (n**3,) array   (weights sum to 8, the parent volume)
    """
    p, w = _gauss_1d(n)
    pts = []
    wts = []
    for i in range(len(p)):
        for j in range(len(p)):
            for k in range(len(p)):
                pts.append([p[i], p[j], p[k]])
                wts.append(w[i] * w[j] * w[k])
    return np.array(pts), np.array(wts)


#: Pre-built rules: 1 point (reduced), 2x2x2 (Hex8 full), 3x3x3 (Hex20 full).
GAUSS_RULES = {
    "1pt": gauss_rule(1),
    "2x2x2": gauss_rule(2),
    "3x3x3": gauss_rule(3),
}


def default_rule(element: str) -> str:
    """Fully integrated default rule: 2x2x2 for Hex8, 3x3x3 for Hex20."""
    key = element.lower()
    if key == "hex8":
        return "2x2x2"
    if key == "hex20":
        return "3x3x3"
    raise ValueError(f"unsupported solid element {element!r}")