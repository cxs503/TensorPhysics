"""Kirchhoff thin-plate finite elements on a Winkler elastic foundation (ice sheet).

Purpose
-------
Level-ice bending failure of a floating ice sheet under a ship-bow indentation
load.  The sheet is idealised as a Kirchhoff (thin, small-deflection) plate of
thickness ``h`` resting on a buoyant Winkler foundation,

    D lap^4 w + k_w w = q(x, y),     D = E h^3 / [12 (1 - nu^2)],  k_w = rho_w g

with ``w`` positive downwards.  The transverse bending moment field follows from
the curvatures,

    M_x  = -D (w,xx + nu w,yy)
    M_y  = -D (w,yy + nu w,xx)
    M_xy = -D (1 - nu) w,xy

and the flexural (first-crack) limit of the ice is the moment capacity

    M_f = sigma_f h^2 / 6        (<=>  surface bending stress sigma = 6 M / h^2).

Sign convention: ``M > 0`` is sagging (bottom in tension), ``M < 0`` is hogging
(top in tension).  A downward bow load therefore drives a *ring crack* on the
top surface where the hogging principal moment first reaches ``M_f``
(:meth:`PlateField.extreme_hogging`); a bottom-surface radial crack under the
load footprint requires the sagging principal moment to reach ``M_f``.  The first
crack is the earlier (smaller load) of the two.  Inside a concentrated contact
zone the ice crushes before it bends, so only the hogging (ring) criterion is
used for the flake size; see the campaign report for the discussion.

Discretisation
--------------
Conforming ``C^1`` Bogner-Fox-Schmit (BFS) rectangular element: a bicubic
Hermite tensor-product patch with the 16 degrees of freedom
``(w, w,x, w,y, w,xy)`` at each of the four corners.  The element is
``C^1``-conforming on rectangular meshes and reproduces polynomials up to cubic
in each direction, so the discretisation converges at the optimal rate for a
degree-3 plate element.

Stiffness and Winkler terms are integrated with 3x3 Gauss quadrature (exact for
the integrands involved) and every element kernel is evaluated in vectorised
form (``numpy.einsum`` over all elements at once), which keeps meshes of 10^4
graded elements cheap.  Distributed loads enter through their consistent nodal
vector; because a bow contact patch can be much smaller than the characteristic
length, the loaded elements may be integrated with a refined Gauss rule
(``gauss_n``) and :meth:`BFSPlateSolver.add_fan_load` rescales the assembled
vector so that its force resultant equals the requested force exactly (factor
reported as ``force_correction``).  Meshes may be graded towards the load
(:func:`graded_axis`, :func:`graded_axis_centered`), so a ~1 m contact patch in
a ~15 m characteristic length is resolvable with ~10^4 elements.

Boundary-condition vocabulary (per edge; keys ``left``/``right``/``bottom``/``top``)
    * ``free``     - no constraint (natural Kirchhoff free edge)
    * ``simply``   - ``w = 0`` only (natural ``M_n = 0``)
    * ``sym``      - normal slope and twist zero (symmetry plane)
    * ``clamped``  - ``w``, both slopes and the twist zero

The Winkler term makes the assembled operator positive definite, hence even an
all-``free`` mesh (the model of an unbounded ice sheet on buoyancy) is solvable
without any artificial constraint.

Independent reference implemented here
    * :func:`kelvin_point_load_moments` - Hetenyi's closed-form Kelvin-function
      solution for a *point* load on an infinite plate on a Winkler foundation
      (moment field from the Kelvin ODE, not from finite differences).

References
----------
Bogner, F.K., Fox, R.L. & Schmit, L.A. (1967). The generation of interelement
    compatible stiffness and mass matrices by the use of interpolation formulas.
    Proc. Conf. Matrix Methods Struct. Anal., AFFDL-TR-66-80, 397-443.
Hetenyi, M. (1946). Beams on Elastic Foundation. Univ. of Michigan Press, Ch. 5.
Kerr, A.D. (1975). The bearing capacity of floating ice plates subjected to
    static or quasi-static loads. CRREL Report 75-1.
Kerr, A.D. (1996). Bearing capacity of floating ice covers subjected to static,
    moving and fluctuating loads. J. Glaciol. 42(142), 474-486.
Timoshenko, S. & Woinowsky-Krieger, S. (1959). Theory of Plates and Shells,
    2nd ed., McGraw-Hill, Ch. 8.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import kei, kelvin, ker

__all__ = [
    "KirchhoffWinklerPlate",
    "BFSPlateSolver",
    "PlateField",
    "kelvin_point_load",
    "kelvin_point_load_moments",
    "critical_point_load",
    "fan_sector_mask",
    "fan_sector_area",
    "graded_axis",
    "graded_axis_centered",
    "limiting_moment",
    "bending_stress",
]

# 3-point Gauss rule mapped to [0, 1]
_GP, _GW = np.polynomial.legendre.leggauss(3)
_GP = 0.5 * (_GP + 1.0)
_GW = 0.5 * _GW

# element corner order and the (x-corner, y-corner) indices used by the Hermite basis
_CORNERS = ((0, 0), (1, 0), (1, 1), (0, 1))
# dof kind k -> (x-derivative?, y-derivative?)
_KDERIV = {0: (False, False), 1: (True, False), 2: (False, True), 3: (True, True)}

# index of the 1D Hermite basis function attached to each of the 16 dofs
_IDX_I = np.array([2 * c[0] + (1 if _KDERIV[k][0] else 0) for c in _CORNERS for k in range(4)])
_IDX_J = np.array([2 * c[1] + (1 if _KDERIV[k][1] else 0) for c in _CORNERS for k in range(4)])


# ---------------------------------------------------------------------------
# material / failure scalars
# ---------------------------------------------------------------------------
def limiting_moment(sigma_f: float, h: float) -> float:
    """Flexural moment capacity ``M_f = sigma_f h^2 / 6`` [N m / m].

    ``|M| >= M_f`` means a surface bending stress ``6|M|/h^2 >= sigma_f``, i.e.
    first cracking of the ice sheet.
    """
    if not np.isfinite(sigma_f) or sigma_f <= 0.0:
        raise ValueError(f"sigma_f must be positive and finite, got {sigma_f!r}")
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError(f"h must be positive and finite, got {h!r}")
    return sigma_f * h * h / 6.0


def bending_stress(M, h: float):
    """Surface bending stress ``sigma = 6 |M| / h^2`` [Pa]."""
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError(f"h must be positive and finite, got {h!r}")
    return 6.0 * np.abs(np.asarray(M, dtype=float)) / (h * h)


def fan_sector_area(radius: float, half_angle_deg: float) -> float:
    """Area of a fan sector [m^2]."""
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius!r}")
    if not (0.0 < half_angle_deg <= 180.0):
        raise ValueError(f"half_angle_deg must be in (0, 180], got {half_angle_deg!r}")
    return 0.5 * radius * radius * math.radians(2.0 * half_angle_deg)


def fan_sector_mask(
    x0: float,
    y0: float,
    radius: float,
    half_angle_deg: float,
    direction_deg: float = 0.0,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Mask of a fan-shaped (circular sector) indentation footprint.

    Apex at ``(x0, y0)``, radius ``radius``, total opening angle
    ``2 * half_angle_deg``, bisector along ``direction_deg`` (degrees,
    counter-clockwise from +x).  Returns ``mask(x, y) -> bool array`` for
    :meth:`BFSPlateSolver.add_pressure`.
    """
    fan_sector_area(radius, half_angle_deg)  # validation

    def mask(x, y):
        dx = np.asarray(x, dtype=float) - x0
        dy = np.asarray(y, dtype=float) - y0
        inside = np.hypot(dx, dy) <= radius
        if half_angle_deg >= 180.0:
            return inside
        ang = np.degrees(np.arctan2(dy, dx)) - direction_deg
        ang = (ang + 180.0) % 360.0 - 180.0
        return inside & (np.abs(ang) <= half_angle_deg)

    return mask


def graded_axis(L: float, n: int, h0: float) -> np.ndarray:
    """Node coordinates on ``[0, L]`` with ``n`` elements, first spacing ``h0``.

    Element sizes grow geometrically towards ``L`` (mesh grading for a
    concentrated load).  If ``h0`` already fits the uniform spacing ``L / n`` the
    uniform mesh is returned unchanged.
    """
    if not np.isfinite(L) or L <= 0.0 or n < 1:
        raise ValueError("L must be positive and n >= 1")
    if not np.isfinite(h0) or h0 <= 0.0:
        raise ValueError(f"h0 must be positive and finite, got {h0!r}")
    if h0 * n >= L:
        return np.linspace(0.0, L, n + 1)

    def total(r: float) -> float:
        return h0 * (r**n - 1.0) / (r - 1.0)

    lo, hi = 1.0 + 1e-12, 2.0
    while total(hi) < L:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if total(mid) < L:
            lo = mid
        else:
            hi = mid
    r = 0.5 * (lo + hi)
    out = np.concatenate([[0.0], np.cumsum(h0 * r ** np.arange(n))])
    out[-1] = L  # close the round-off gap
    return out


def graded_axis_centered(L: float, n: int, h0: float) -> np.ndarray:
    """Node coordinates on ``[-L/2, L/2]`` (``n`` even elements) graded towards 0."""
    if n < 2 or n % 2:
        raise ValueError(f"n must be even and >= 2, got {n!r}")
    half = graded_axis(0.5 * L, n // 2, h0)
    return np.concatenate([-half[::-1], half[1:]])


# ---------------------------------------------------------------------------
# problem definition
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KirchhoffWinklerPlate:
    """Kirchhoff plate on a Winkler foundation (floating ice sheet), SI units.

    Construction validates physical admissibility: a non-positive thickness,
    modulus or foundation modulus yields a meaningless (negative or singular)
    discrete operator and is rejected with ``ValueError``.
    """

    h: float
    E: float = 5.0e9
    nu: float = 0.3
    k_w: float = 1025.0 * 9.81
    lx: float = 1.0
    ly: float = 1.0

    def __post_init__(self) -> None:
        for name, val in (("h", self.h), ("E", self.E), ("k_w", self.k_w)):
            if not np.isfinite(val) or val <= 0.0:
                raise ValueError(f"{name} must be positive and finite, got {val!r}")
        if not np.isfinite(self.nu) or not (-1.0 < self.nu < 0.5):
            raise ValueError(f"nu must lie in (-1, 0.5), got {self.nu!r}")
        for name, val in (("lx", self.lx), ("ly", self.ly)):
            if not np.isfinite(val) or val <= 0.0:
                raise ValueError(f"{name} must be positive and finite, got {val!r}")

    @property
    def D(self) -> float:
        """Flexural rigidity ``D = E h^3 / [12 (1 - nu^2)]`` [N m]."""
        return self.E * self.h**3 / (12.0 * (1.0 - self.nu**2))

    @property
    def char_length(self) -> float:
        """Hetenyi characteristic length ``l = (D / k_w)^(1/4)`` [m]."""
        return (self.D / self.k_w) ** 0.25

    def moment_capacity(self, sigma_f: float) -> float:
        """First-crack moment capacity ``M_f = sigma_f h^2 / 6`` [N m/m]."""
        return limiting_moment(sigma_f, self.h)

    def constitutive(self) -> np.ndarray:
        """``C`` such that the curvature energy density is ``kappa^T C kappa / 2``."""
        D, nu = self.D, self.nu
        return D * np.array(
            [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - nu)]], dtype=float
        )


# ---------------------------------------------------------------------------
# BFS element kernels (scalar and vectorised)
# ---------------------------------------------------------------------------
def _hermite_1d_arrays(xi, L):
    """1D cubic Hermite basis on an element of length ``L``.

    ``xi`` and ``L`` broadcast; returns ``(N, dN/dx, d2N/dx2)``, each with a
    trailing axis of length 4 ordered ``[w(0), w,x(0), w(1), w,x(1)]``.
    """
    xi, L = np.broadcast_arrays(np.asarray(xi, dtype=float), np.asarray(L, dtype=float))
    xi2, xi3 = xi * xi, xi * xi * xi
    N = np.stack(
        [1.0 - 3.0 * xi2 + 2.0 * xi3, L * (xi - 2.0 * xi2 + xi3), 3.0 * xi2 - 2.0 * xi3, L * (-xi2 + xi3)],
        axis=-1,
    )
    dN = np.stack(
        [
            (-6.0 * xi + 6.0 * xi2) / L,
            1.0 - 4.0 * xi + 3.0 * xi2,
            (6.0 * xi - 6.0 * xi2) / L,
            -2.0 * xi + 3.0 * xi2,
        ],
        axis=-1,
    )
    d2N = np.stack(
        [
            (-6.0 + 12.0 * xi) / (L * L),
            (-4.0 + 6.0 * xi) / L,
            (6.0 - 12.0 * xi) / (L * L),
            (-2.0 + 6.0 * xi) / L,
        ],
        axis=-1,
    )
    return N, dN, d2N


def bfs_shape(xi, eta, lx, ly):
    """BFS shape functions and derivatives for (arrays of) local coordinates.

    ``xi``, ``eta``, ``lx``, ``ly`` broadcast to a common shape ``S``; the result
    arrays have shape ``S + (16,)`` ordered as ``dof = 4 * corner + k`` with
    ``k = 0:w, 1:w_x, 2:w_y, 3:w_xy`` and corners ``(0,0), (1,0), (1,1), (0,1)``.

    Returns ``(N, Nx, Ny, Nxx, Nyy, Nxy)``.
    """
    Nx1, dNx1, d2Nx1 = _hermite_1d_arrays(xi, lx)
    Ny1, dNy1, d2Ny1 = _hermite_1d_arrays(eta, ly)
    N = Nx1[..., _IDX_I] * Ny1[..., _IDX_J]
    Nx = dNx1[..., _IDX_I] * Ny1[..., _IDX_J]
    Ny = Nx1[..., _IDX_I] * dNy1[..., _IDX_J]
    Nxx = d2Nx1[..., _IDX_I] * Ny1[..., _IDX_J]
    Nyy = Nx1[..., _IDX_I] * d2Ny1[..., _IDX_J]
    Nxy = dNx1[..., _IDX_I] * dNy1[..., _IDX_J]
    return N, Nx, Ny, Nxx, Nyy, Nxy


def _gauss_rule(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """``n``-point Gauss-Legendre rule mapped to ``[0, 1]``."""
    if n < 2:
        raise ValueError("need at least a 2-point rule")
    gp, gw = np.polynomial.legendre.leggauss(int(n))
    return 0.5 * (gp + 1.0), 0.5 * gw


# ---------------------------------------------------------------------------
# field container + post-processing
# ---------------------------------------------------------------------------
@dataclass
class PlateField:
    """Nodal solution container (arrays are ``(ny+1, nx+1)``, index ``[iy, ix]``)."""

    plate: KirchhoffWinklerPlate
    x: np.ndarray
    y: np.ndarray
    w: np.ndarray
    Mx: np.ndarray
    My: np.ndarray
    Mxy: np.ndarray
    foundation_model: str = "winkler"
    g_p: float = 0.0

    # -- sampling -----------------------------------------------------------
    def _bilinear(self, F: np.ndarray, xq, yq) -> np.ndarray:
        x, y = self.x, self.y
        xq = np.atleast_1d(np.asarray(xq, dtype=float))
        yq = np.atleast_1d(np.asarray(yq, dtype=float))
        tol = 1e-9 * max(x[-1] - x[0], y[-1] - y[0])
        if (
            np.any(xq < x[0] - tol)
            or np.any(xq > x[-1] + tol)
            or np.any(yq < y[0] - tol)
            or np.any(yq > y[-1] + tol)
        ):
            raise ValueError("sample point outside the mesh")
        i0 = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, len(x) - 2)
        j0 = np.clip(np.searchsorted(y, yq, side="right") - 1, 0, len(y) - 2)
        tx = np.clip((xq - x[i0]) / (x[i0 + 1] - x[i0]), 0.0, 1.0)
        ty = np.clip((yq - y[j0]) / (y[j0 + 1] - y[j0]), 0.0, 1.0)
        return (
            (1 - tx) * (1 - ty) * F[j0, i0]
            + tx * (1 - ty) * F[j0, i0 + 1]
            + (1 - tx) * ty * F[j0 + 1, i0]
            + tx * ty * F[j0 + 1, i0 + 1]
        )

    def sample_ray(
        self,
        origin: Tuple[float, float] = (0.0, 0.0),
        angle_deg: float = 0.0,
        r: Optional[np.ndarray] = None,
        n: int = 401,
    ) -> Dict[str, np.ndarray]:
        """Sample ``w`` and the radial/circumferential moments along a ray.

        ``M_r = Mx cos^2 + My sin^2 + 2 Mxy sin cos`` and ``M_t`` its conjugate.
        """
        x0, y0 = origin
        th = math.radians(angle_deg)
        c, s = math.cos(th), math.sin(th)
        if r is None:
            # longest ray that stays inside the mesh in this direction
            cands = []
            if c > 1e-12:
                cands.append((self.x[-1] - x0) / c)
            elif c < -1e-12:
                cands.append((self.x[0] - x0) / c)
            if s > 1e-12:
                cands.append((self.y[-1] - y0) / s)
            elif s < -1e-12:
                cands.append((self.y[0] - y0) / s)
            rmax = min(cands) if cands else 0.0
            r = np.linspace(0.0, 0.995 * rmax, n)
        r = np.atleast_1d(np.asarray(r, dtype=float))
        xq, yq = x0 + r * c, y0 + r * s
        Mx = self._bilinear(self.Mx, xq, yq)
        My = self._bilinear(self.My, xq, yq)
        Mxy = self._bilinear(self.Mxy, xq, yq)
        return {
            "r": r,
            "w": self._bilinear(self.w, xq, yq),
            "M_r": Mx * c * c + My * s * s + 2.0 * Mxy * s * c,
            "M_t": Mx * s * s + My * c * c - 2.0 * Mxy * s * c,
        }

    # -- failure analysis ---------------------------------------------------
    def principal_moments(self) -> Tuple[np.ndarray, np.ndarray]:
        """Nodal principal moments ``(M_1 >= M_2)``.

        ``M_2 < 0`` is the hogging principal moment (top-surface crack driver),
        ``M_1 > 0`` drives bottom-surface tension.
        """
        avg = 0.5 * (self.Mx + self.My)
        rad = np.sqrt((0.5 * (self.Mx - self.My)) ** 2 + self.Mxy**2)
        return avg + rad, avg - rad

    def _grid(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.x, self.y, indexing="xy")

    def extreme_hogging(self) -> Dict[str, float]:
        """Most negative principal moment and its location (first-crack site)."""
        _m1, m2 = self.principal_moments()
        k = np.unravel_index(int(np.argmin(m2)), m2.shape)
        X, Y = self._grid()
        return {
            "M_2": float(m2[k]),
            "x": float(X[k]),
            "y": float(Y[k]),
            "r": float(np.hypot(X[k], Y[k])),
        }

    def extreme_sagging(self) -> Dict[str, float]:
        """Largest positive principal moment and its location (bottom tension)."""
        m1, _m2 = self.principal_moments()
        k = np.unravel_index(int(np.argmax(m1)), m1.shape)
        X, Y = self._grid()
        return {
            "M_1": float(m1[k]),
            "x": float(X[k]),
            "y": float(Y[k]),
            "r": float(np.hypot(X[k], Y[k])),
        }

    def peak_hogging_moment(self, origin=(0.0, 0.0), angle_deg: float = 0.0):
        """``(M_r_min, r_at_min)`` of the radial moment along a ray."""
        s = self.sample_ray(origin, angle_deg)
        i = int(np.argmin(s["M_r"]))
        return float(s["M_r"][i]), float(s["r"][i])

    def peak_sagging_moment(self, origin=(0.0, 0.0), angle_deg: float = 0.0):
        """``(M_r_max, r_at_max)`` of the radial moment along a ray."""
        s = self.sample_ray(origin, angle_deg)
        i = int(np.argmax(s["M_r"]))
        return float(s["M_r"][i]), float(s["r"][i])

    def crack_radius(
        self, M_f: float, origin=(0.0, 0.0), angle_deg: float = 0.0, outermost: bool = True
    ) -> float:
        """Radius on the ray where ``M_r = -M_f`` (top-surface ring crack); ``nan`` if none."""
        s = self.sample_ray(origin, angle_deg)
        f = s["M_r"] + M_f
        r = s["r"]
        idx = np.where(np.sign(f[:-1]) * np.sign(f[1:]) < 0.0)[0]
        if idx.size == 0:
            return float("nan")
        i = int(idx[-1] if outermost else idx[0])
        f0, f1 = f[i], f[i + 1]
        return float(r[i] + (r[i + 1] - r[i]) * (-f0) / (f1 - f0))

    def cracked_extent(
        self, M_f: float, origin: Tuple[float, float] = (0.0, 0.0)
    ) -> Dict[str, float]:
        """Extent of the region where the hogging principal moment reaches ``-M_f``.

        Returns the maximum distance of a cracked node from ``origin`` (the broken
        flake length), the cracked node count and the extent along x.  ``r_max`` is
        ``nan`` when the load level does not crack the sheet.
        """
        if M_f <= 0.0:
            raise ValueError("M_f must be positive")
        _m1, m2 = self.principal_moments()
        mask = m2 <= -M_f
        if not mask.any():
            return {"r_max": float("nan"), "n_nodes": 0, "x_min": float("nan"), "x_max": float("nan")}
        X, Y = self._grid()
        dx = X[mask] - origin[0]
        r = np.hypot(dx, Y[mask] - origin[1])
        return {
            "r_max": float(r.max()),
            "n_nodes": int(mask.sum()),
            "x_min": float(dx.min()),
            "x_max": float(dx.max()),
        }

# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------
_EDGE_KINDS = ("free", "simply", "sym", "clamped")


class BFSPlateSolver:
    """Assembled BFS Kirchhoff plate on a Winkler foundation.

    Parameters
    ----------
    plate : KirchhoffWinklerPlate
    nx, ny : int
        Number of elements in x and y (``>= 1``).
    xs, ys : array_like, optional
        Node coordinates (lengths ``nx + 1`` / ``ny + 1``); defaults to a uniform
        mesh.  Pass :func:`graded_axis` / :func:`graded_axis_centered` output to
        refine the mesh locally.
    foundation_model : {"winkler", "pasternak", "none"}
        ``"winkler"`` (default) keeps the classical one-parameter subgrade
        ``p = k_w w``; ``"pasternak"`` adds the shear layer
        ``p = k_w w - g_p lap^2 w`` through the element term
        ``int g_p (N_,x N_,x^T + N_,y N_,y^T) dA``; ``"none"`` drops the
        foundation entirely (free-plate negative control, ``k_w`` still sets
        ``D`` and the post-processing).  ``"pasternak"`` with ``g_p = 0`` is
        bit-identical to ``"winkler"``.
    g_p : float
        Pasternak shear-layer modulus [N/m]; ignored unless
        ``foundation_model="pasternak"``.
    """

    def __init__(
        self,
        plate: KirchhoffWinklerPlate,
        nx: int,
        ny: int,
        xs: Optional[Iterable[float]] = None,
        ys: Optional[Iterable[float]] = None,
        foundation_model: str = "winkler",
        g_p: float = 0.0,
    ):
        if not isinstance(nx, int) or not isinstance(ny, int) or nx < 1 or ny < 1:
            raise ValueError(f"nx and ny must be positive integers, got nx={nx!r}, ny={ny!r}")
        if foundation_model not in ("winkler", "pasternak", "none"):
            raise ValueError(
                "foundation_model must be 'winkler', 'pasternak' or 'none', "
                f"got {foundation_model!r}"
            )
        if not np.isfinite(g_p) or g_p < 0.0:
            raise ValueError(f"g_p must be non-negative and finite, got {g_p!r}")
        if foundation_model == "winkler" and g_p != 0.0:
            raise ValueError(
                "g_p > 0 requires foundation_model='pasternak' "
                f"(got foundation_model={foundation_model!r}, g_p={g_p!r})"
            )
        self.plate = plate
        self.foundation_model = foundation_model
        self.g_p = float(g_p)
        # effective foundation coefficients used by the assembly
        self._k_eff = 0.0 if foundation_model == "none" else float(plate.k_w)
        self._g_eff = float(g_p) if foundation_model == "pasternak" else 0.0
        self.nx = nx
        self.ny = ny
        self.x = np.linspace(0.0, plate.lx, nx + 1) if xs is None else np.asarray(xs, dtype=float)
        self.y = np.linspace(0.0, plate.ly, ny + 1) if ys is None else np.asarray(ys, dtype=float)
        if self.x.size != nx + 1 or self.y.size != ny + 1:
            raise ValueError("xs/ys must contain nx+1 / ny+1 coordinates")
        if np.any(np.diff(self.x) <= 0.0) or np.any(np.diff(self.y) <= 0.0):
            raise ValueError("xs/ys must be strictly increasing")
        self.ndof = 4 * (nx + 1) * (ny + 1)
        self._f = np.zeros(self.ndof)
        self._load_total = 0.0
        self.force_correction = 1.0
        self._K: Optional[sp.csr_matrix] = None
        self._field: Optional[PlateField] = None
        self._u: Optional[np.ndarray] = None
        self._idx = self._element_dofs()

    # -- geometry helpers ---------------------------------------------------
    @property
    def hx(self) -> np.ndarray:
        """Element sizes in x, shape ``(nx,)``."""
        return np.diff(self.x)

    @property
    def hy(self) -> np.ndarray:
        """Element sizes in y, shape ``(ny,)``."""
        return np.diff(self.y)

    def node(self, ix: int, iy: int) -> int:
        return iy * (self.nx + 1) + ix

    def dof(self, ix: int, iy: int, k: int) -> int:
        return 4 * self.node(ix, iy) + k

    def _element_dofs(self) -> np.ndarray:
        """(nx*ny, 16) global DOF indices per element (element ``e = ey*nx + ex``)."""
        ex, ey = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing="xy")
        ex, ey = ex.ravel(), ey.ravel()
        idx = np.zeros((ex.size, 16), dtype=int)
        for corner, (cx, cy) in enumerate(_CORNERS):
            base = 4 * ((ey + cy) * (self.nx + 1) + (ex + cx))
            for k in range(4):
                idx[:, 4 * corner + k] = base + k
        return idx

    def _elem_sizes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Per-element ``(hx_e, hy_e)`` arrays of length ``nx * ny``."""
        e = np.arange(self._idx.shape[0])
        return self.hx[e % self.nx], self.hy[e // self.nx]

    def _elem_centers(self) -> Tuple[np.ndarray, np.ndarray]:
        e = np.arange(self._idx.shape[0])
        ex, ey = e % self.nx, e // self.nx
        return (
            0.5 * (self.x[ex] + self.x[ex + 1]),
            0.5 * (self.y[ey] + self.y[ey + 1]),
        )

    # -- assembly -----------------------------------------------------------
    def element_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """All element stiffness/foundation matrices, shapes ``(ne, 16, 16)``.

        The second array is the *total* foundation matrix: the Winkler subgrade
        ``k N N^T`` plus, for ``foundation_model="pasternak"``, the shear-layer
        term ``g_p (N_,x N_,x^T + N_,y N_,y^T)`` (the ``-g_p lap^2 w`` of the
        Pasternak pressure is the Euler-Lagrange form of
        ``+ g_p/2 int |grad w|^2 dA``).  For ``g_p = 0`` the shear term is never
        touched, so the Winkler assembly is reproduced bit for bit.
        """
        hx_e, hy_e = self._elem_sizes()
        C = self.plate.constitutive()
        jac = hx_e * hy_e
        Ke = np.zeros((hx_e.size, 16, 16))
        Kf = np.zeros((hx_e.size, 16, 16))
        for xi, wi in zip(_GP, _GW):
            for eta, wj in zip(_GP, _GW):
                N, Nx, Ny, Nxx, Nyy, Nxy = bfs_shape(xi, eta, hx_e, hy_e)
                B = np.stack([Nxx, Nyy, 2.0 * Nxy], axis=1)  # (ne, 3, 16)
                Ke += (wi * wj) * np.einsum("egi,gh,ehj->eij", B, C, B) * jac[:, None, None]
                Kf += (wi * wj) * np.einsum("ei,ej->eij", N, N) * self._k_eff * jac[:, None, None]
                if self._g_eff > 0.0:
                    Kf += (
                        (wi * wj)
                        * (
                            np.einsum("ei,ej->eij", Nx, Nx)
                            + np.einsum("ei,ej->eij", Ny, Ny)
                        )
                        * self._g_eff
                        * jac[:, None, None]
                    )
        return Ke, Kf

    def element_shear_matrix(self) -> np.ndarray:
        """Pasternak shear-layer element matrices ``int g_p grad N . grad N dA``.

        Shape ``(ne, 16, 16)``; all zeros for a Winkler foundation.  Exposed so
        tests can check the shear contribution independently of the bending and
        Winkler terms (integrated with the same 3x3 Gauss rule).
        """
        hx_e, hy_e = self._elem_sizes()
        jac = hx_e * hy_e
        Kg = np.zeros((hx_e.size, 16, 16))
        if self._g_eff <= 0.0:
            return Kg
        for xi, wi in zip(_GP, _GW):
            for eta, wj in zip(_GP, _GW):
                _, Nx, Ny, _, _, _ = bfs_shape(xi, eta, hx_e, hy_e)
                Kg += (
                    (wi * wj)
                    * (np.einsum("ei,ej->eij", Nx, Nx) + np.einsum("ei,ej->eij", Ny, Ny))
                    * self._g_eff
                    * jac[:, None, None]
                )
        return Kg

    def _stiffness(self) -> sp.csr_matrix:
        if self._K is not None:
            return self._K
        Ke, Kf = self.element_matrices()
        Kel = (Ke + Kf).reshape(Kel_shape := (self._idx.shape[0], 256))
        idx = self._idx
        rows = np.repeat(idx, 16, axis=1).ravel()
        cols = np.tile(idx, (1, 16)).ravel()
        K = sp.coo_matrix((Kel.ravel(), (rows, cols)), shape=(self.ndof, self.ndof)).tocsr()
        self._K = K
        return K

    def _scatter_load(self, fe: np.ndarray, elems: Optional[np.ndarray] = None) -> None:
        """Scatter per-element consistent load vectors ``fe`` (``fe.shape[0] == len(elems)``)."""
        if elems is None:
            np.add.at(self._f, self._idx.ravel(), np.ascontiguousarray(fe).ravel())
        else:
            np.add.at(self._f, self._idx[elems].ravel(), np.ascontiguousarray(fe).ravel())

    # -- loads --------------------------------------------------------------
    def _locate(self, x: float, y: float) -> Tuple[int, int, float, float]:
        ix = int(np.clip(np.searchsorted(self.x, x, side="right") - 1, 0, self.nx - 1))
        iy = int(np.clip(np.searchsorted(self.y, y, side="right") - 1, 0, self.ny - 1))
        xi = (x - self.x[ix]) / (self.x[ix + 1] - self.x[ix])
        eta = (y - self.y[iy]) / (self.y[iy + 1] - self.y[iy])
        if not (-1e-9 <= xi <= 1.0 + 1e-9 and -1e-9 <= eta <= 1.0 + 1e-9):
            raise ValueError(f"point ({x}, {y}) outside the plate")
        return ix, iy, min(max(xi, 0.0), 1.0), min(max(eta, 0.0), 1.0)

    def add_point_load(self, P: float, x: float, y: float) -> "BFSPlateSolver":
        """Consistent nodal load of a transverse point load ``P`` [N] at ``(x, y)``."""
        if not np.isfinite(P):
            raise ValueError("P must be finite")
        ix, iy, xi, eta = self._locate(x, y)
        hx, hy = self.x[ix + 1] - self.x[ix], self.y[iy + 1] - self.y[iy]
        N = bfs_shape(xi, eta, hx, hy)[0]
        dofs = np.array([self.dof(ix + cx, iy + cy, k) for (cx, cy) in _CORNERS for k in range(4)])
        np.add.at(self._f, dofs, P * N)
        self._load_total += P
        return self

    def add_pressure(
        self,
        q: Callable[[np.ndarray, np.ndarray], np.ndarray],
        inside: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        area: Optional[float] = None,
        gauss_n: int = 3,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> "BFSPlateSolver":
        """Consistent nodal load of a transverse pressure ``q(x, y)`` [Pa].

        Parameters
        ----------
        q : callable
            Pressure in Pa, evaluated on arrays of Gauss abscissae.
        inside : callable, optional
            Region test; Gauss points where it is false are skipped.
        area : float, optional
            Exact loaded area, booked for reporting only.
        gauss_n : int
            Gauss order per element (raise it when the patch is smaller than one
            element).
        bbox : (xmin, xmax, ymin, ymax), optional
            Only elements overlapping this box are integrated (speed-up; the
            caller is responsible for the box covering the loaded region).
        """
        gp, gw = _gauss_rule(gauss_n)
        wgt = np.outer(gw, gw).ravel()
        hx_e, hy_e = self._elem_sizes()
        cx_e, cy_e = self._elem_centers()
        if bbox is not None:
            x0, x1, y0, y1 = bbox
            sel = (
                (cx_e >= x0 - hx_e)
                & (cx_e <= x1 + hx_e)
                & (cy_e >= y0 - hy_e)
                & (cy_e <= y1 + hy_e)
            )
        else:
            sel = np.ones(hx_e.size, dtype=bool)
        elems = np.where(sel)[0]
        if elems.size == 0:
            return self
        e = np.repeat(elems, wgt.size)
        g = np.tile(np.arange(wgt.size), elems.size)
        gi, gj = np.divmod(g, gauss_n)
        xi = gp[gi]
        eta = gp[gj]
        hx_rep, hy_rep = hx_e[e], hy_e[e]
        ex, ey = e % self.nx, e // self.nx
        px = self.x[ex] + xi * hx_rep
        py = self.y[ey] + eta * hy_rep
        keep = np.ones(e.size, dtype=bool)
        if inside is not None:
            keep = np.asarray(inside(px, py), dtype=bool)
        if not keep.any():
            return self
        qv = np.broadcast_to(np.asarray(q(px, py), dtype=float), px.shape).ravel()
        keep &= qv != 0.0
        if not keep.any():
            return self
        N = bfs_shape(xi[keep], eta[keep], hx_rep[keep], hy_rep[keep])[0]
        contrib = (wgt[g[keep]] * qv[keep] * hx_rep[keep] * hy_rep[keep])[:, None] * N
        fe = np.zeros((hx_e.size, 16))
        np.add.at(fe, e[keep], contrib)
        self._scatter_load(fe[elems], elems)
        if area is not None:
            self._load_total += float(area)
        return self

    def add_fan_load(
        self,
        P_total: float,
        x0: float,
        y0: float,
        radius: float,
        half_angle_deg: float,
        direction_deg: float = 0.0,
        gauss_n: int = 6,
        normalize: bool = True,
        symmetry_factor: float = 1.0,
    ) -> "BFSPlateSolver":
        """Fan-shaped (sector) indentation load of total physical force ``P_total`` [N].

        The force is spread uniformly over the sector area
        ``A = radius^2 * radians(2 * half_angle_deg) / 2`` - the idealisation of a
        bow's plan-curvature contact patch on the ice.  With ``normalize`` the
        assembled vector is rescaled so that the *physical* force resultant equals
        ``P_total`` exactly (factor stored in ``force_correction``), removing both
        the resolution bias of a patch that is small compared with the
        characteristic length and the truncation of the patch by the domain.

        ``symmetry_factor`` is the ratio between the physical (full patch) force
        and the force actually present inside the computational domain: use 2.0
        when a symmetry plane cuts the patch in half (a patch centred on the
        symmetry edge of a half model).  The assembled nodal vector then carries
        ``P_total / symmetry_factor`` and ``applied_load`` reports ``P_total``.
        """
        if not np.isfinite(P_total):
            raise ValueError("P_total must be finite")
        A = fan_sector_area(radius, half_angle_deg)
        q = P_total / A
        mask = fan_sector_mask(x0, y0, radius, half_angle_deg, direction_deg)
        bbox = (x0 - radius, x0 + radius, y0 - radius, y0 + radius)
        f0 = self._f.copy()
        self.add_pressure(lambda xx, yy: np.full(np.shape(xx), q, dtype=float), inside=mask, gauss_n=gauss_n, bbox=bbox)
        added = self._f - f0
        raw = float(added[0::4].sum())
        if symmetry_factor <= 0.0:
            raise ValueError("symmetry_factor must be positive")
        if normalize and raw != 0.0:
            scale = P_total / (raw * symmetry_factor)
            self._f = f0 + added * scale
            self.force_correction = scale
        self._load_total += P_total
        return self

    @property
    def applied_load(self) -> float:
        """Total transverse force booked through the load API [N]."""
        return self._load_total

    def nodal_load_sum(self) -> float:
        """Sum of the ``w`` components of the assembled consistent load vector [N]."""
        return float(self._f[0::4].sum())

    def strain_energy(self, u: Optional[np.ndarray] = None) -> float:
        """Strain energy of a solution (bending + foundation) [J]."""
        u = self._u if u is None else u
        if u is None:
            raise ValueError("solve() first or pass a solution vector")
        return 0.5 * float(u @ (self._stiffness() @ u))

    # -- boundary conditions ------------------------------------------------
    def _constrained_dofs(self, bc: Dict[str, str]) -> np.ndarray:
        bad = {k: v for k, v in bc.items() if v not in _EDGE_KINDS}
        if bad:
            raise ValueError(f"unknown boundary condition(s) {bad}; allowed {_EDGE_KINDS}")
        unknown = set(bc) - {"left", "right", "bottom", "top"}
        if unknown:
            raise ValueError(f"unknown edge name(s) {sorted(unknown)}")
        dofs: List[int] = []
        for edge, kind in bc.items():
            if kind == "free":
                continue
            if edge in ("left", "right"):
                ix = 0 if edge == "left" else self.nx
                nodes = [(ix, iy) for iy in range(self.ny + 1)]
                normal_k = 1
            else:
                iy = 0 if edge == "bottom" else self.ny
                nodes = [(ix, iy) for ix in range(self.nx + 1)]
                normal_k = 2
            ks = [0] if kind == "simply" else ([0, 1, 2, 3] if kind == "clamped" else [normal_k, 3])
            for (ix, iy) in nodes:
                for k in ks:
                    dofs.append(self.dof(ix, iy, k))
        return np.unique(np.asarray(dofs, dtype=int))

    # -- solve --------------------------------------------------------------
    def solve(self, bc: Optional[Dict[str, str]] = None) -> PlateField:
        """Solve the static problem and recover the nodal moment field."""
        bc = dict(bc or {})
        K = self._stiffness()
        fixed = self._constrained_dofs(bc)
        free = np.setdiff1d(np.arange(self.ndof), fixed, assume_unique=True)
        u = np.zeros(self.ndof)
        u[free] = spla.spsolve(K[free][:, free].tocsc(), self._f[free])
        self._u = u
        self._field = self._recover(u)
        return self._field

    def _recover(self, u: np.ndarray) -> PlateField:
        plate = self.plate
        D, nu = plate.D, plate.nu
        nx1, ny1 = self.nx + 1, self.ny + 1
        Mx = np.zeros((ny1, nx1))
        My = np.zeros((ny1, nx1))
        Mxy = np.zeros((ny1, nx1))
        cnt = np.zeros((ny1, nx1))
        hx_e, hy_e = self._elem_sizes()
        ue = u[self._idx]
        ex = np.arange(hx_e.size) % self.nx
        ey = np.arange(hx_e.size) // self.nx
        for cx, cy in _CORNERS:
            _, _, _, Nxx, Nyy, Nxy = bfs_shape(float(cx), float(cy), hx_e, hy_e)
            wxx = np.einsum("ep,ep->e", ue, Nxx)
            wyy = np.einsum("ep,ep->e", ue, Nyy)
            wxy = np.einsum("ep,ep->e", ue, Nxy)
            np.add.at(Mx, (ey + cy, ex + cx), -D * (wxx + nu * wyy))
            np.add.at(My, (ey + cy, ex + cx), -D * (wyy + nu * wxx))
            np.add.at(Mxy, (ey + cy, ex + cx), -D * (1.0 - nu) * wxy)
            np.add.at(cnt, (ey + cy, ex + cx), 1.0)
        Mx /= np.maximum(cnt, 1.0)
        My /= np.maximum(cnt, 1.0)
        Mxy /= np.maximum(cnt, 1.0)
        w = u[0::4].reshape(ny1, nx1)
        return PlateField(
            plate=plate,
            x=self.x,
            y=self.y,
            w=w,
            Mx=Mx,
            My=My,
            Mxy=Mxy,
            foundation_model=self.foundation_model,
            g_p=self.g_p,
        )


# ---------------------------------------------------------------------------
# closed-form reference: point load on an infinite plate on a Winkler foundation
# ---------------------------------------------------------------------------
def kelvin_point_load(r, P: float, D: float, k_w: float) -> np.ndarray:
    """Hetenyi closed form ``w(r) = -P l^2 / (2 pi D) kei(r / l)`` [m], positive down.

    ``l = (D / k_w)^(1/4)``.  As ``kei(0) = -pi/4`` this reproduces the classical
    centre deflection ``w(0) = P l^2 / (8 D)``.
    """
    if D <= 0.0 or k_w <= 0.0:
        raise ValueError("D and k_w must be positive")
    l = (D / k_w) ** 0.25
    rr = np.atleast_1d(np.asarray(r, dtype=float))
    out = -(P * l * l / (2.0 * math.pi * D)) * kei(rr / l)
    return out if out.size > 1 else float(out[0])


def _kelvin_derivatives(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``kei(x)``, ``kei'(x)``, ``kei''(x)`` from the Kelvin ODE.

    With ``W = ker + i kei`` satisfying the order-0 Kelvin equation
    ``x^2 W'' + x W' - i x^2 W = 0`` (``z = x exp(i pi/4)``) the imaginary part is

        kei'' = ker - kei' / x.

    The first derivative is the imaginary part of :func:`scipy.special.kelvin`'s
    fourth return value (``Kep = ker' + i kei'``).
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    x = np.where(x > 0.0, x, np.finfo(float).tiny)
    _be, _ke, _bep, kep = kelvin(x)
    kep_im = np.asarray(kep, dtype=complex).imag
    return kei(x), kep_im, ker(x) - kep_im / x


def kelvin_point_load_moments(
    r, P: float, D: float, k_w: float, nu: float = 0.3
) -> Dict[str, np.ndarray]:
    """Radial/circumferential moment field of a point load on an infinite plate.

    With ``w(r) = -A kei(r/l)`` and ``A = P l^2 / (2 pi D)``:

        w,r  = -(A/l) kei'(r/l),   w,rr = -(A/l^2) kei''(r/l)
        M_r  = -D (w,rr + nu w,r / r),   M_t = -D (w,r / r + nu w,rr)

    ``M > 0`` sagging, ``M < 0`` hogging (top tension - ring-crack driver).
    """
    if D <= 0.0 or k_w <= 0.0:
        raise ValueError("D and k_w must be positive")
    l = (D / k_w) ** 0.25
    rr = np.atleast_1d(np.asarray(r, dtype=float))
    A = P * l * l / (2.0 * math.pi * D)
    kei_v, kei_p, kei_pp = _kelvin_derivatives(rr / l)
    w = -A * kei_v
    w_r = -(A / l) * kei_p
    w_rr = -(A / (l * l)) * kei_pp
    safe_r = np.where(rr > 0.0, rr, np.inf)
    return {
        "w": w,
        "w_r": w_r,
        "w_rr": w_rr,
        "M_r": -D * (w_rr + nu * w_r / safe_r),
        "M_t": -D * (w_r / safe_r + nu * w_rr),
    }


def critical_point_load(
    M_f: float, D: float, k_w: float, nu: float = 0.3, r_lo: float = 0.02, r_hi: float = 8.0, n: int = 8001
) -> Dict[str, float]:
    """Point load at which the first (ring) crack appears, and its radius.

    Returns ``P_crit_N`` (load for which ``min M_r = -M_f``), the radius of that
    minimum ``r_peak_m`` and ``r_peak_over_l``.  The response is linear in the
    load, so this is exact for the Kelvin solution.
    """
    if M_f <= 0.0:
        raise ValueError("M_f must be positive")
    l = (D / k_w) ** 0.25
    r = np.linspace(r_lo * l, r_hi * l, n)
    mom = kelvin_point_load_moments(r, 1.0, D, k_w, nu)
    i = int(np.argmin(mom["M_r"]))
    m_peak = float(mom["M_r"][i])
    return {
        "P_crit_N": float(M_f / abs(m_peak)),
        "r_peak_m": float(r[i]),
        "r_peak_over_l": float(r[i] / l),
        "M_peak_per_unit_load": m_peak,
    }
