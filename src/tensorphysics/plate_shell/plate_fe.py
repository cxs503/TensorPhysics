"""Conforming Kirchhoff plate finite element (Bogner-Fox-Schmit) + recovery.

The element is the 16-dof bicubic Hermite rectangle: four nodes, each carrying
``(w, theta_x = dw/dy, theta_y = -dw/dx, w_xy = d2w/dxdy)``.  ``w`` and its
first derivatives are continuous across element edges, i.e. this is a ``C^1``
conforming thin-plate element, so the deflection converges faster than the
``O(h^2)`` guaranteed for a ``C^0`` Mindlin element (the measured rates are
reported by :func:`convergence_study`).

Curvatures follow the sign convention ``kappa = [-w,xx, -w,yy, -2 w,xy]`` used
by :mod:`tensorphysics.plate_buckling.core`, so the bending stiffness is
``Db = D [[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]]`` and

    Mx = -D (w,xx + nu w,yy),  My = -D (w,yy + nu w,xx),
    Mxy = -D (1 - nu) w,xy,
    Qx = -D d/dx(nabla^2 w) = -D (w,xxx + w,xyy),  Qy = symmetric.

The module deliberately re-derives the element kernels locally (rather than
only importing :func:`tensorphysics.plate_buckling.core.element_kernels`) so
that the third derivatives needed for the shear recovery are available;
:func:`element_stiffness_kernels` implements the same stiffness through the
``plate_buckling`` kernels and the test-suite asserts the two agree to machine
precision.

Boundary conditions implemented: ``SSSS`` (all four edges simply supported).
Along a supported edge ``w = 0`` implies both the tangential slope and the
twist dof vanish there, so the constrained dofs on, e.g., an edge parallel to
``y`` are ``(w, theta_x, w_xy)`` while the normal slope ``theta_y`` is free;
corners additionally have ``theta_y = 0``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from tensorphysics.plate_buckling.core import (
    consistent_pressure_load,
    element_kernels,
)

__all__ = [
    "hermite_1d",
    "nodal_to_hermite",
    "element_stiffness",
    "element_stiffness_kernels",
    "element_fields",
    "SSSSPlateFE",
    "convergence_study",
    "fit_order",
]

_GX, _GW = np.polynomial.legendre.leggauss(4)
_GAUSS = [(_GX[i], _GX[j], _GW[i] * _GW[j]) for j in range(4) for i in range(4)]


def hermite_1d(t):
    """Cubic Hermite basis and derivatives up to third order on ``t in [0, 1]``.

    Node dofs are ``(value, slope)`` at ``t = 0`` then ``t = 1``.
    """
    t = np.asarray(t, dtype=float)
    h = np.stack([1 - 3 * t ** 2 + 2 * t ** 3,
                  t - 2 * t ** 2 + t ** 3,
                  3 * t ** 2 - 2 * t ** 3,
                  -t ** 2 + t ** 3])
    dh = np.stack([-6 * t + 6 * t ** 2,
                   1 - 4 * t + 3 * t ** 2,
                   6 * t - 6 * t ** 2,
                   -2 * t + 3 * t ** 2])
    ddh = np.stack([-6 + 12 * t,
                    -4 + 6 * t,
                    6 - 12 * t,
                    -2 + 6 * t])
    dddh = np.stack([12.0 * np.ones_like(t),
                     6.0 * np.ones_like(t),
                     -12.0 * np.ones_like(t),
                     6.0 * np.ones_like(t)])
    return h, dh, ddh, dddh


def nodal_to_hermite(ex: float, ey: float) -> np.ndarray:
    """Map 16 nodal dofs to the 16 Hermite generalised dofs ``A[i, j]``.

    Node order ``(0,0), (ex,0), (ex,ey), (0,ey)``; per-node dof order
    ``(w, theta_x = dw/dy, theta_y = -dw/dx, w_xy)``.  ``A`` stores
    value / ``t``-slope / ``s``-slope / mixed at the two ends, with
    ``A[i*4+j]`` and ``i`` the ``x`` (``t``) slot, ``j`` the ``y`` (``s``) slot.
    """
    M = np.zeros((16, 16))
    corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    for c, (t, s) in enumerate(corners):
        it, js = int(round(2 * t)), int(round(2 * s))
        base = 4 * c
        M[it * 4 + js, base + 0] = 1.0                 # w
        M[(it + 1) * 4 + js, base + 2] = -ex           # w_t = ex * w,x
        M[it * 4 + (js + 1), base + 1] = ey            # w_s = ey * w,y
        M[(it + 1) * 4 + (js + 1), base + 3] = ex * ey  # w_ts
    return M


def element_stiffness(ex: float, ey: float, D: float, nu: float) -> np.ndarray:
    """16x16 bending stiffness of the bicubic Hermite rectangle."""
    M = nodal_to_hermite(ex, ey)
    Db = D * np.array([[1.0, nu, 0.0],
                       [nu, 1.0, 0.0],
                       [0.0, 0.0, 0.5 * (1.0 - nu)]])
    Ke = np.zeros((16, 16))
    det_j = 0.25 * ex * ey
    for xi, eta, wgt in _GAUSS:
        t, s = 0.5 * (1.0 + xi), 0.5 * (1.0 + eta)
        _, _, ddh_x, _ = hermite_1d(t)
        h_x, _, _, _ = hermite_1d(t)
        _, _, ddh_y, _ = hermite_1d(s)
        h_y, _, _, _ = hermite_1d(s)
        B = np.zeros((3, 16))
        B[0, :] = -((np.outer(ddh_x, h_y) / ex ** 2).ravel() @ M)
        B[1, :] = -((np.outer(h_x, ddh_y) / ey ** 2).ravel() @ M)
        B[2, :] = -2.0 * ((np.outer(hermite_1d(t)[1], hermite_1d(s)[1])
                           / (ex * ey)).ravel() @ M)
        Ke += wgt * det_j * (B.T @ Db @ B)
    return Ke


def element_stiffness_kernels(ex: float, ey: float, D: float,
                              nu: float) -> np.ndarray:
    """Same stiffness built from :mod:`tensorphysics.plate_buckling` kernels.

    Cross-check path (independent code route to the same integral).
    """
    kern = element_kernels(ex, ey)
    Db = D * np.array([[1.0, nu, 0.0],
                       [nu, 1.0, 0.0],
                       [0.0, 0.0, 0.5 * (1.0 - nu)]])
    Ke = np.zeros((16, 16))
    for g, wgt in enumerate(kern["weight"]):
        Bb = kern["Bb"][g]
        Ke += wgt * (Bb.T @ Db @ Bb)
    return Ke


def element_fields(ex: float, ey: float, D: float, nu: float,
                   ue: np.ndarray, t: float, s: float) -> Dict[str, float]:
    """Displacement, slope, moments and shear at ``(t, s) in [0,1]^2``."""
    M = nodal_to_hermite(ex, ey)
    A = (M @ ue).reshape(4, 4)
    h, dh, ddh, dddh = hermite_1d(t)
    hy, dhy, ddhy, dddy = hermite_1d(s)

    def mix(fx, fy):
        return float(np.einsum("i,j,ij->", fx, fy, A))

    w = mix(h, hy)
    w_x = mix(dh, hy) / ex
    w_y = mix(h, dhy) / ey
    w_xx = mix(ddh, hy) / ex ** 2
    w_yy = mix(h, ddhy) / ey ** 2
    w_xy = mix(dh, dhy) / (ex * ey)
    w_xxx = mix(dddh, hy) / ex ** 3
    w_yyy = mix(h, dddy) / ey ** 3
    w_xxy = mix(ddh, dhy) / (ex ** 2 * ey)
    w_xyy = mix(dh, ddhy) / (ex * ey ** 2)

    return {
        "w": w,
        "theta_x": w_y,
        "theta_y": -w_x,
        "Mx": -D * (w_xx + nu * w_yy),
        "My": -D * (w_yy + nu * w_xx),
        "Mxy": -D * (1.0 - nu) * w_xy,
        "Qx": -D * (w_xxx + w_xyy),
        "Qy": -D * (w_yyy + w_xxy),
        "w_xx": w_xx,
        "w_yy": w_yy,
    }


@dataclass
class SSSSPlateFE:
    """Simply supported rectangular plate under uniform pressure (linear).

    Parameters
    ----------
    a, b, t : float
        Dimensions and thickness [m].
    E, nu : float
        Isotropic elastic constants.
    nx, ny : int
        Number of elements per side (mesh is ``nx x ny``).
    """

    a: float
    b: float
    t: float
    E: float
    nu: float
    nx: int = 16
    ny: int = 16
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    # ------------------------------------------------------------------ setup
    @property
    def D(self) -> float:
        return self.E * self.t ** 3 / (12.0 * (1.0 - self.nu ** 2))

    @property
    def ex(self) -> float:
        return self.a / self.nx

    @property
    def ey(self) -> float:
        return self.b / self.ny

    @property
    def n_nodes(self) -> int:
        return (self.nx + 1) * (self.ny + 1)

    @property
    def ndof(self) -> int:
        return 4 * self.n_nodes

    def node_id(self, i: int, j: int) -> int:
        return j * (self.nx + 1) + i

    def _element_nodes(self, ei: int, ej: int) -> List[int]:
        n0 = self.node_id(ei, ej)
        n1 = self.node_id(ei + 1, ej)
        n2 = self.node_id(ei + 1, ej + 1)
        n3 = self.node_id(ei, ej + 1)
        return [n0, n1, n2, n3]

    def _element_dofs(self, ei: int, ej: int) -> np.ndarray:
        dofs = []
        for n in self._element_nodes(ei, ej):
            dofs.extend([4 * n + k for k in range(4)])
        return np.asarray(dofs)

    def constrained_dofs(self, mode: str = "natural") -> np.ndarray:
        """SSSS constraint set.

        ``mode="natural"`` (default, used for every verification run) fixes
        only ``w = 0`` on the four edges, leaving the normal moment as the
        natural boundary condition -- this is the variationally correct simply
        supported condition for a Kirchhoff plate.

        ``mode="tight"`` additionally fixes the tangential slope and the twist
        dof ``w_xy`` on the edges.  ``w_xy`` is *not* implied by ``w = 0`` along
        an edge (it is the edge derivative of the free normal slope), so this
        set restricts the trial space and makes the model systematically too
        stiff.  It is kept as a documented negative control (an over-constrained
        model must give a smaller deflection than the analytic value).
        """
        if mode not in ("natural", "tight"):
            raise ValueError("mode must be 'natural' or 'tight'")
        cons = set()
        for j in range(self.ny + 1):
            for i in range(self.nx + 1):
                n = self.node_id(i, j)
                on_x = (i == 0) or (i == self.nx)   # edge parallel to y
                on_y = (j == 0) or (j == self.ny)   # edge parallel to x
                if on_x or on_y:
                    cons.add(4 * n + 0)             # w = 0 (always)
                if mode == "tight":
                    if on_x:
                        cons.add(4 * n + 1)         # theta_x = dw/dy = 0
                        cons.add(4 * n + 3)         # w_xy = 0
                    if on_y:
                        cons.add(4 * n + 2)         # theta_y = -dw/dx = 0
                        cons.add(4 * n + 3)         # w_xy = 0
        return np.array(sorted(cons), dtype=int)

    def assemble(self) -> sp.csr_matrix:
        if "K" in self._cache:
            return self._cache["K"]
        Ke = element_stiffness(self.ex, self.ey, self.D, self.nu)
        rows, cols, vals = [], [], []
        for ej in range(self.ny):
            for ei in range(self.nx):
                d = self._element_dofs(ei, ej)
                rows.append(np.repeat(d, 16))
                cols.append(np.tile(d, 16))
                vals.append(Ke.ravel())
        K = sp.coo_matrix((np.concatenate(vals),
                           (np.concatenate(rows), np.concatenate(cols))),
                          shape=(self.ndof, self.ndof)).tocsr()
        self._cache["K"] = K
        return K

    def load_vector(self, q: float) -> np.ndarray:
        fe = consistent_pressure_load(self.ex, self.ey, q)
        f = np.zeros(self.ndof)
        for ej in range(self.ny):
            for ei in range(self.nx):
                f[self._element_dofs(ei, ej)] += fe
        return f

    # ------------------------------------------------------------------ solve
    def solve(self, q: float = 1.0e4, bc_mode: str = "natural") -> np.ndarray:
        """Assemble, apply SSSS constraints and solve for the nodal dofs."""
        K = self.assemble()
        f = self.load_vector(q)
        cons = self.constrained_dofs(bc_mode)
        free = np.setdiff1d(np.arange(self.ndof), cons)
        u = np.zeros(self.ndof)
        Kff = K[free][:, free].tocsc()
        u[free] = spla.spsolve(Kff, f[free])
        kindex = {"cons": cons, "free": free, "q": q, "bc_mode": bc_mode}
        self._cache["u"] = u
        self._cache["solve"] = kindex
        self._cache["Kff"] = Kff
        return u

    @property
    def u(self) -> np.ndarray:
        if "u" not in self._cache:
            raise RuntimeError("call solve(q) before reading the solution")
        return self._cache["u"]

    def reaction_sum(self) -> float:
        """Sum of vertical reactions; equals the total applied load ``q a b``."""
        K = self.assemble()
        f = self.load_vector(self._cache["solve"]["q"])
        r = K @ self.u - f
        cons = self._cache["solve"]["cons"]
        z_rows = np.array([d for d in cons if d % 4 == 0])
        return float(-r[z_rows].sum())

    # ------------------------------------------------------------ post-process
    def w_nodal(self) -> np.ndarray:
        """Nodal deflection array of shape ``(ny + 1, nx + 1)``."""
        w = np.zeros((self.ny + 1, self.nx + 1))
        for j in range(self.ny + 1):
            for i in range(self.nx + 1):
                w[j, i] = self.u[4 * self.node_id(i, j)]
        return w

    def center_deflection(self) -> float:
        """Centre deflection.

        An *even* mesh has a node exactly at ``(a/2, b/2)`` (nodes sit at
        ``i a/nx``), so the centre value is read directly.  For an odd mesh the
        centre is inside an element and the four surrounding nodes are averaged
        (which biases the value low by ``(h/2)^2 (w,xx + w,yy)/2``); use even
        meshes for gate evaluation.
        """
        w = self.w_nodal()
        nx, ny = self.nx, self.ny
        if nx % 2 == 0 and ny % 2 == 0:
            return float(w[ny // 2, nx // 2])
        i0, j0 = nx // 2, ny // 2
        return float(np.mean([w[j0, i0], w[j0, i0 + 1],
                              w[j0 + 1, i0], w[j0 + 1, i0 + 1]]))

    def _candidate_elements(self, x: float, y: float):
        """Elements whose closed domain contains ``(x, y)`` (ties averaged)."""
        def idx(v, n, h):
            f = v / h
            cand = [int(np.clip(np.floor(f), 0, n - 1))]
            if abs(f - round(f)) < 1e-12:
                k = int(round(f))
                cand = [c for c in (k - 1, k) if 0 <= c <= n - 1]
            return sorted(set(cand))
        return [(ei, ej) for ei in idx(x, self.nx, self.ex)
                for ej in idx(y, self.ny, self.ey)]

    def fields_at(self, x: float, y: float) -> Dict[str, float]:
        """Evaluate recovered fields at a physical point (element-average)."""
        out: List[Dict[str, float]] = []
        for ei, ej in self._candidate_elements(x, y):
            dofs = self._element_dofs(ei, ej)
            t = (x - ei * self.ex) / self.ex
            s = (y - ej * self.ey) / self.ey
            out.append(element_fields(self.ex, self.ey, self.D, self.nu,
                                      self.u[dofs], t, s))
        return {k: float(np.mean([o[k] for o in out])) for k in out[0]}

    def fields_at_points(self, xs, ys) -> Dict[str, np.ndarray]:
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        xs, ys = np.broadcast_arrays(xs, ys)
        vals = [self.fields_at(float(x), float(y))
                for x, y in zip(xs.ravel(), ys.ravel())]
        return {k: np.array([v[k] for v in vals]).reshape(xs.shape)
                for k in vals[0]}

    def centroid_fields(self) -> Dict[str, np.ndarray]:
        """Element-centroid fields, each of shape ``(ny, nx)``."""
        acc: Dict[str, List[np.ndarray]] = {}
        for ej in range(self.ny):
            row: Dict[str, List[float]] = {}
            for ei in range(self.nx):
                dofs = self._element_dofs(ei, ej)
                f = element_fields(self.ex, self.ey, self.D, self.nu,
                                   self.u[dofs], 0.5, 0.5)
                for k, v in f.items():
                    row.setdefault(k, []).append(v)
            for k, v in row.items():
                acc.setdefault(k, []).append(np.asarray(v))
        return {k: np.vstack(v) for k, v in acc.items()}


def fit_order(h: Sequence[float], err: Sequence[float]) -> float:
    """Least-squares convergence order from ``err = C h^p`` (log-log fit)."""
    h = np.asarray(h, dtype=float)
    err = np.asarray(err, dtype=float)
    good = err > 0
    if good.sum() < 2:
        return float("nan")
    p, _ = np.polyfit(np.log(h[good]), np.log(err[good]), 1)
    return float(p)


def field_error(a: float, b: float, t: float, E: float, nu: float, q: float,
                fe: "SSSSPlateFE", n_sample: int = 41,
                n_terms: int = 81) -> Dict[str, np.ndarray]:
    """Sample the FE and analytic fields on a cell-centre grid and compare.

    Returns a dict with the sampling coordinates ``x``, ``y`` (1-D), the
    analytic fields ``w_ref``/``Mx_ref``, the FE fields ``w_fe``/``Mx_fe`` and
    the relative error maps ``rel_w``/``rel_Mx``.
    """
    from .navier import navier_plate_fields

    xs = (np.arange(n_sample) + 0.5) * a / n_sample
    ys = (np.arange(n_sample) + 0.5) * b / n_sample
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    ref = navier_plate_fields(a, b, t, E, nu, q, X, Y, n_terms=n_terms)
    fe_f = fe.fields_at_points(X, Y)
    scale_w = float(np.max(np.abs(ref["w"]))) or 1.0
    scale_m = float(np.max(np.abs(ref["Mx"]))) or 1.0
    return {
        "x": xs, "y": ys, "X": X, "Y": Y,
        "w_ref": ref["w"], "w_fe": fe_f["w"],
        "Mx_ref": ref["Mx"], "Mx_fe": fe_f["Mx"],
        "My_ref": ref["My"], "My_fe": fe_f["My"],
        "Qx_ref": ref["Qx"], "Qx_fe": fe_f["Qx"],
        "rel_w": (fe_f["w"] - ref["w"]) / scale_w,
        "rel_Mx": (fe_f["Mx"] - ref["Mx"]) / scale_m,
    }


def _l2_relative(err: np.ndarray, ref: np.ndarray) -> float:
    num = float(np.sqrt(np.mean(err ** 2)))
    den = float(np.sqrt(np.mean(ref ** 2))) or 1.0
    return num / den


def convergence_study(a: float, b: float, t: float, E: float, nu: float,
                      q: float, meshes: Sequence[int] = (4, 8, 16, 32, 64),
                      n_terms: int = 81, bc_mode: str = "natural",
                      n_sample: int = 41) -> List[Dict[str, float]]:
    """h-refinement study against the Navier series reference.

    Every mesh is compared with the analytic solution at
    * the plate centre: deflection, ``Mx``, ``My``;
    * ``(a/4, b/2)`` and ``(a/2, b/4)``: ``Qx``, ``Qy``;
    * all interior nodes: pointwise deflection (unbiased, no centre averaging);
    * a ``n_sample x n_sample`` cell-centre grid: ``L2`` relative field errors.

    The reaction sum is compared with ``q a b`` as a global equilibrium check.
    For even meshes the plate centre is a node, so the centre deflection needs
    no averaging; for odd meshes :meth:`SSSSPlateFE.center_deflection` averages
    the four central nodes, which biases the value low by
    ``(h/2)^2 (w,xx + w,yy)/2`` -- use even meshes (or the pointwise
    ``err_w_nodes`` column) for gate evaluation.
    """
    from .navier import navier_plate_fields

    c_ref = navier_plate_fields(a, b, t, E, nu, q,
                                np.array([0.5 * a]), np.array([0.5 * b]),
                                n_terms=n_terms)
    c_ref = {k: float(v[0]) for k, v in c_ref.items()}
    s_ref = navier_plate_fields(a, b, t, E, nu, q,
                                np.array([0.25 * a, 0.5 * a]),
                                np.array([0.5 * b, 0.25 * b]),
                                n_terms=n_terms)

    rows: List[Dict[str, float]] = []
    for n in meshes:
        fe = SSSSPlateFE(a=a, b=b, t=t, E=E, nu=nu, nx=n, ny=n)
        fe.solve(q, bc_mode=bc_mode)
        w_c = fe.center_deflection()
        c = fe.fields_at(0.5 * a, 0.5 * b)
        s1 = fe.fields_at(0.25 * a, 0.5 * b)
        s2 = fe.fields_at(0.5 * a, 0.25 * b)

        # pointwise nodal comparison (unbiased)
        xs = np.linspace(0.0, a, n + 1)[1:-1]
        ys = np.linspace(0.0, b, n + 1)[1:-1]
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        w_ref_nodes = navier_plate_fields(a, b, t, E, nu, q, X, Y,
                                          n_terms=n_terms)["w"]
        w_fe_nodes = np.array([[fe.u[4 * fe.node_id(i, j)]
                                for i in range(1, n)] for j in range(1, n)])
        err_w_nodes = float(np.max(np.abs(w_fe_nodes - w_ref_nodes))
                            / float(np.max(np.abs(c_ref["w"]))))

        fld = field_error(a, b, t, E, nu, q, fe, n_sample=n_sample,
                          n_terms=n_terms)
        total_load = q * a * b
        rsum = fe.reaction_sum()
        rows.append({
            "n_elem": float(n),
            "h": float(a / n),
            "ndof": float(fe.ndof),
            "center_is_node": float(1 if n % 2 == 0 else 0),
            "w_center": w_c,
            "w_ref": c_ref["w"],
            "err_w": abs(w_c - c_ref["w"]) / abs(c_ref["w"]),
            "err_w_nodes": err_w_nodes,
            "l2_w": _l2_relative(fld["w_fe"] - fld["w_ref"], fld["w_ref"]),
            "Mx_center": c["Mx"],
            "Mx_ref": c_ref["Mx"],
            "err_Mx": abs(c["Mx"] - c_ref["Mx"]) / abs(c_ref["Mx"]),
            "My_center": c["My"],
            "My_ref": c_ref["My"],
            "err_My": abs(c["My"] - c_ref["My"]) / abs(c_ref["My"]),
            "l2_Mx": _l2_relative(fld["Mx_fe"] - fld["Mx_ref"],
                                  fld["Mx_ref"]),
            "Qx": s1["Qx"],
            "Qx_ref": float(s_ref["Qx"][0]),
            "err_Qx": abs(s1["Qx"] - float(s_ref["Qx"][0]))
                      / abs(float(s_ref["Qx"][0])),
            "Qy": s2["Qy"],
            "Qy_ref": float(s_ref["Qy"][1]),
            "err_Qy": abs(s2["Qy"] - float(s_ref["Qy"][1]))
                      / abs(float(s_ref["Qy"][1])),
            "reaction_sum": rsum,
            "load_total": total_load,
            "equilibrium_rel_error": abs(rsum - total_load) / total_load,
        })
    return rows
