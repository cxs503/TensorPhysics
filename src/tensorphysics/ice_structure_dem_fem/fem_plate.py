"""Elasto-plastic Mindlin plate face of the ice-structure coupling node.

Element
-------
4-node Mindlin (Reissner) plate, 3 DOF per node ``(w, theta_x, theta_y)``,
bending integrated with 2x2 Gauss and transverse shear with **selective
reduced integration** (1 point) to remove shear locking in the thin-plate
limit (``a/t`` up to ~200 here).

Constitutive model
------------------
Through-thickness **layered J2 flow theory with linear isotropic hardening**.
At every bending Gauss point the strain at layer height ``z`` is
``eps = z kappa`` (small deflection, no membrane coupling -- declared
limitation), and each layer runs a genuine 3-D radial return map followed by a
**plane-stress condensation**: the thickness strain ``eps_z`` is solved so that
``sigma_z = 0`` (Newton on the scalar ``eps_z``), which is the exact plane
stress condition for a shell layer.  Stresses are hence *not* obtained from a
1-D reduction: the full 6-component return map is exercised.

The layer tangent ``d sigma / d eps_inplane`` is obtained by central
differences of that return map (3 extra evaluations per layer per Newton
iteration) -- no hand-derived algorithmic tangent is assumed.

Loading
-------
The panel is driven by a patch pressure time history (the DEM ice-load
spectrum, see :mod:`.coupling`).  The response is integrated **quasi
statically** (incremental Newton on the load path).  The quasi-static
assumption is quantified in :mod:`.verification` against the panel's
fundamental period.

Energy ledger
-------------
``external_work``, ``elastic_energy``, ``plastic_dissipation`` are tracked
every step so that the coupling node can report an energy-balance residual.

Units: SI (m, Pa, N, J).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

#: positions of the in-plane components (xx, yy, xy) inside the 6-vector Voigt
INPLANE = np.array([0, 1, 3])

__all__ = [
    "PlateMaterial",
    "PlateConfig",
    "PlateResponse",
    "solve_plate_history",
    "layer_return_map",
    "navier_reference_max_stress",
]


# ---------------------------------------------------------------------------
# constitutive model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PlateMaterial:
    E: float = 206.0e9        # steel
    nu: float = 0.3
    sigma_y: float = 355.0e6  # AH36 nominal yield
    H: float = 2.0e9          # linear isotropic hardening modulus

    @property
    def lam(self) -> float:
        return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

    @property
    def G(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))

    def D3(self) -> np.ndarray:
        lam, G = self.lam, self.G
        D = np.zeros((6, 6))
        D[:3, :3] = lam
        D[0, 0] = D[1, 1] = D[2, 2] = lam + 2 * G
        D[3, 3] = D[4, 4] = D[5, 5] = G
        return D


def _vmises(sig: np.ndarray) -> np.ndarray:
    """von Mises equivalent stress for 6-vectors (engineering shear)."""
    sxx, syy, szz, txy, tyz, tzx = (sig[..., i] for i in range(6))
    p = (sxx + syy + szz) / 3.0
    dxx, dyy, dzz = sxx - p, syy - p, szz - p
    return np.sqrt(1.5 * (dxx ** 2 + dyy ** 2 + dzz ** 2 + 2.0 * (txy ** 2 + tyz ** 2 + tzx ** 2)))


def layer_return_map(
    eps: np.ndarray,
    eps_p: np.ndarray,
    alpha: np.ndarray,
    mat: PlateMaterial,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised 3-D radial return map with linear isotropic hardening.

    ``eps`` has shape ``(n, 6)`` (Voigt, engineering shear); ``eps_p`` and
    ``alpha`` carry the previous plastic state (shape ``(n, 6)`` and ``(n,)``).
    """
    D = mat.D3()
    sig_tr = np.einsum("ij,nj->ni", D, eps - eps_p)
    q_tr = _vmises(sig_tr)
    sig_y = mat.sigma_y + mat.H * alpha
    f_tr = q_tr - sig_y
    # deviatoric direction of the trial stress
    p = (sig_tr[:, 0] + sig_tr[:, 1] + sig_tr[:, 2]) / 3.0
    s = np.empty_like(sig_tr)
    s[:, 0] = sig_tr[:, 0] - p
    s[:, 1] = sig_tr[:, 1] - p
    s[:, 2] = sig_tr[:, 2] - p
    s[:, 3:] = sig_tr[:, 3:]
    n = np.zeros_like(sig_tr)
    safe = np.where(q_tr > 1e-30, q_tr, 1.0)
    n[:, 0] = 1.5 * s[:, 0] / safe
    n[:, 1] = 1.5 * s[:, 1] / safe
    n[:, 2] = 1.5 * s[:, 2] / safe
    n[:, 3] = 3.0 * s[:, 3] / safe
    n[:, 4] = 3.0 * s[:, 4] / safe
    n[:, 5] = 3.0 * s[:, 5] / safe
    dgam = np.where(f_tr > 0.0, f_tr / (3.0 * mat.G + mat.H), 0.0)
    sig = sig_tr - 2.0 * mat.G * dgam[:, None] * n
    eps_p_new = eps_p + dgam[:, None] * n
    alpha_new = alpha + dgam
    return sig, eps_p_new, alpha_new


def _layer_state(
    eps_in: np.ndarray,
    eps_p: np.ndarray,
    alpha: np.ndarray,
    mat: PlateMaterial,
    eps_z_guess: np.ndarray,
    iters: int = 12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Plane-stress condensation: solve ``eps_z`` such that ``sigma_z = 0``."""
    eps_z = eps_z_guess.copy()
    n = eps_in.shape[0]
    eps = np.zeros((n, 6))
    sig = np.zeros((n, 6))
    eps_p_new, alpha_new = eps_p, alpha
    for _ in range(iters):
        eps[:, 0] = eps_in[:, 0]
        eps[:, 1] = eps_in[:, 1]
        eps[:, 2] = eps_z
        eps[:, 3] = eps_in[:, 2]
        sig, eps_p_new, alpha_new = layer_return_map(eps, eps_p, alpha, mat)
        eps_z = eps_z - sig[:, 2] / (mat.lam + 2.0 * mat.G)
    eps[:, 2] = eps_z
    sig, eps_p_new, alpha_new = layer_return_map(eps, eps_p, alpha, mat)
    return sig, eps_p_new, alpha_new, eps_z


# ---------------------------------------------------------------------------
# element / mesh
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PlateConfig:
    a: float = 2.4              # panel length in x [m]
    b: float = 3.6              # panel length in y [m]
    t: float = 0.020            # plate thickness [m]
    nx: int = 12                # elements in x
    ny: int = 18                # elements in y
    n_layers: int = 5           # Simpson layers through the thickness
    shear_correction: float = 5.0 / 6.0
    material: PlateMaterial = field(default_factory=PlateMaterial)

    def __post_init__(self) -> None:
        if self.a <= 0 or self.b <= 0 or self.t <= 0:
            raise ValueError("panel dimensions must be positive")
        if self.n_layers < 3 or self.n_layers % 2 == 0:
            raise ValueError("n_layers must be odd (Simpson integration)")
        if self.nx < 2 or self.ny < 2:
            raise ValueError("mesh too coarse")


def _shape(xi: float, eta: float) -> np.ndarray:
    return 0.25 * np.array([
        (1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
        (1 + xi) * (1 + eta), (1 - xi) * (1 + eta),
    ])


def _dshape(xi: float, eta: float, hx: float, hy: float) -> Tuple[np.ndarray, np.ndarray]:
    dN_dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
    dN_deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
    return dN_dxi / (0.5 * hx), dN_deta / (0.5 * hy)


def _B_curvature(xi, eta, hx, hy) -> np.ndarray:
    dNdx, dNdy = _dshape(xi, eta, hx, hy)
    B = np.zeros((3, 12))
    B[0, 1::3] = dNdx          # kappa_xx = d theta_x / dx
    B[1, 2::3] = dNdy          # kappa_yy = d theta_y / dy
    B[2, 1::3] = dNdy          # kappa_xy = d theta_x/dy + d theta_y/dx
    B[2, 2::3] = dNdx
    return B


def _B_shear(xi, eta, hx, hy) -> np.ndarray:
    N = _shape(xi, eta)
    dNdx, dNdy = _dshape(xi, eta, hx, hy)
    B = np.zeros((2, 12))
    B[0, 0::3] = dNdx
    B[0, 1::3] = N             # gamma_xz = dw/dx + theta_x
    B[1, 0::3] = dNdy
    B[1, 2::3] = N             # gamma_yz = dw/dy + theta_y
    return B


def _simpson(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Layer coordinates in [-0.5, 0.5] (fraction of t) and weights."""
    if n % 2 == 0:
        raise ValueError("need odd n_layers")
    zeta = np.linspace(-0.5, 0.5, n)
    h = zeta[1] - zeta[0]
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return zeta, w * h / 3.0


@dataclass
class PlateResponse:
    load_history: np.ndarray
    max_vm_stress: np.ndarray
    max_abs_sigma_x: np.ndarray
    centre_deflection: np.ndarray
    max_alpha_p: np.ndarray
    energy: Dict[str, float]
    config: PlateConfig
    converged: bool
    newton_iterations: List[int] = field(default_factory=list)

    @property
    def peak_stress(self) -> float:
        return float(np.max(self.max_vm_stress))

    def as_dict(self) -> Dict[str, object]:
        return {
            "panel_m": [self.config.a, self.config.b],
            "thickness_m": self.config.t,
            "mesh": [self.config.nx, self.config.ny],
            "n_layers": self.config.n_layers,
            "peak_von_mises_stress_Pa": self.peak_stress,
            "peak_abs_sigma_x_Pa": float(np.max(self.max_abs_sigma_x)),
            "peak_pressure_Pa": float(np.max(self.load_history)),
            "max_centre_deflection_m": float(np.max(np.abs(self.centre_deflection))),
            "max_equivalent_plastic_strain": float(np.max(self.max_alpha_p)),
            "yield_exceeded": bool(np.max(self.max_alpha_p) > 0.0),
            "energy": self.energy,
            "converged": self.converged,
            "newton_iterations": self.newton_iterations,
        }


def solve_plate_history(
    config: PlateConfig,
    load_history: Sequence[float],
    patch: Optional[Tuple[float, float, float, float]] = None,
    newton_tol: float = 1e-8,
    max_newton: int = 14,
    _retries: int = 1,
) -> PlateResponse:
    """Integrate the panel response along a uniform patch-pressure history.

    ``patch = (x0, x1, y0, y1)`` in metres; ``None`` means the whole panel.
    ``load_history`` is the pressure [Pa] on the patch at every step, applied
    incrementally (quasi-static).  Assembly is fully vectorised over elements,
    bending Gauss points and layers; the layer tangent is a central difference
    of the plane-stress return map.
    """
    cfg = config
    mat = cfg.material
    nx, ny = cfg.nx, cfg.ny
    hx, hy = cfg.a / nx, cfg.b / ny
    nnode_x, nnode_y = nx + 1, ny + 1
    ndof = 3 * nnode_x * nnode_y
    area = hx * hy

    node_id = np.arange(nnode_x * nnode_y).reshape(nnode_x, nnode_y)
    xs = np.linspace(0.0, cfg.a, nnode_x)
    ys = np.linspace(0.0, cfg.b, nnode_y)

    gauss = (-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0))
    zeta, w_zeta = _simpson(cfg.n_layers)
    z_layer = zeta * cfg.t
    Ds = mat.G * cfg.shear_correction * cfg.t * np.eye(2)

    dofs = np.zeros((nx * ny, 12), dtype=int)
    B = np.zeros((nx * ny, 4, 3, 12))
    Bsh = np.zeros((nx * ny, 2, 12))
    Bsh_edges = np.zeros((nx * ny, 4, 2, 12))
    e = 0
    for i in range(nx):
        for j in range(ny):
            nn = [node_id[i, j], node_id[i + 1, j], node_id[i + 1, j + 1], node_id[i, j + 1]]
            dofs[e] = np.array([[3 * n, 3 * n + 1, 3 * n + 2] for n in nn]).ravel()
            for g, (xi, eta) in enumerate([(a, b) for a in gauss for b in gauss]):
                B[e, g] = _B_curvature(xi, eta, hx, hy)
            Bsh[e] = _B_shear(0.0, 0.0, hx, hy)
            Bsh_edges[e, 0] = _B_shear(0.0, -1.0, hx, hy)   # eta = -1 midpoint
            Bsh_edges[e, 1] = _B_shear(1.0, 0.0, hx, hy)    # xi = +1 midpoint
            Bsh_edges[e, 2] = _B_shear(0.0, 1.0, hx, hy)    # eta = +1 midpoint
            Bsh_edges[e, 3] = _B_shear(-1.0, 0.0, hx, hy)   # xi = -1 midpoint
            e += 1
    nelem = e

    # consistent nodal loads for a uniform pressure on the patch
    if patch is None:
        patch = (0.0, cfg.a, 0.0, cfg.b)
    px0, px1, py0, py1 = patch
    f_ref = np.zeros(ndof)
    for i in range(nx):
        cx = 0.5 * (xs[i] + xs[i + 1])
        if not (px0 - 1e-12 <= cx <= px1 + 1e-12):
            continue
        for j in range(ny):
            cy = 0.5 * (ys[j] + ys[j + 1])
            if not (py0 - 1e-12 <= cy <= py1 + 1e-12):
                continue
            for n in (node_id[i, j], node_id[i + 1, j],
                      node_id[i + 1, j + 1], node_id[i, j + 1]):
                f_ref[3 * n] += area / 4.0

    # soft simple support: w = 0 on the edge, tangential rotation restrained
    fixed = []
    for i in range(nnode_x):
        fixed += [3 * node_id[i, 0], 3 * node_id[i, nnode_y - 1],
                  3 * node_id[i, 0] + 1, 3 * node_id[i, nnode_y - 1] + 1]
    for j in range(nnode_y):
        fixed += [3 * node_id[0, j], 3 * node_id[nnode_x - 1, j],
                  3 * node_id[0, j] + 2, 3 * node_id[nnode_x - 1, j] + 2]
    fixed = np.unique(np.array(fixed))
    free = np.setdiff1d(np.arange(ndof), fixed)

    n_lay = cfg.n_layers
    N = nelem * 4 * n_lay
    eps_p = np.zeros((N, 6))
    alpha_p = np.zeros(N)
    eps_z = np.zeros(N)
    w_layer = (z_layer ** 2 * w_zeta * cfg.t * area / 4.0)     # (n_lay,)
    w_layer_f = (z_layer * w_zeta * cfg.t * area / 4.0)

    def layer_fields(u_vec):
        kap = np.einsum("ngki,ni->ngk", B, u_vec[dofs])       # (nelem,4,3)
        eps_in = (z_layer[None, None, :, None] * kap[:, :, None, :]).reshape(N, 3)
        sig, ep_new, al_new, ez = _layer_state(eps_in, eps_p, alpha_p, mat, eps_z, iters=8)
        return kap, eps_in, sig, ep_new, al_new, ez

    def layer_tangent(eps_in):
        D = np.zeros((N, 3, 3))
        h = 1e-8
        for k in range(3):
            pert = np.zeros((N, 3))
            pert[:, k] = h
            sp_, _, _, _ = _layer_state(eps_in + pert, eps_p, alpha_p, mat, eps_z, iters=6)
            sm_, _, _, _ = _layer_state(eps_in - pert, eps_p, alpha_p, mat, eps_z, iters=6)
            D[:, :, k] = (sp_[:, INPLANE] - sm_[:, INPLANE]) / (2.0 * h)
        return D

    def assemble(u_vec, tangent: bool = True):
        kap, eps_in, sig, ep_new, al_new, ez = layer_fields(u_vec)
        f_int = np.zeros(ndof)
        sig4 = sig.reshape(nelem, 4, n_lay, 6)
        fb = np.einsum("ngci,nglc,ngl->ni", B, sig4[..., INPLANE],
                       np.broadcast_to(w_layer_f[None, None, :], (nelem, 4, n_lay)))
        np.add.at(f_int, dofs.ravel(), fb.ravel())
        shear = np.einsum("nki,nkl,nlj->nij", Bsh, np.broadcast_to(Ds, (nelem, 2, 2)), Bsh)
        fsh = np.einsum("nij,nj->ni", shear, u_vec[dofs])
        np.add.at(f_int, dofs.ravel(), fsh.ravel())
        if not tangent:
            return f_int, None, sig, ep_new, al_new, ez
        D = layer_tangent(eps_in).reshape(nelem, 4, n_lay, 3, 3)
        ke = np.einsum("ngci,nglcp,ngpj,ngl->nij", B, D, B,
                       np.broadcast_to(w_layer[None, None, :], (nelem, 4, n_lay)))
        ke += shear
        rows = np.repeat(dofs, 12, axis=1)
        cols = np.tile(dofs, (1, 12))
        K = sp.coo_matrix((ke.ravel(), (rows.ravel(), cols.ravel())),
                          shape=(ndof, ndof)).tocsr()
        return f_int, K, sig, ep_new, al_new, ez

    centre_node = node_id[nnode_x // 2, nnode_y // 2]

    u = np.zeros(ndof)
    u_prev = u.copy()
    f_prev = np.zeros(ndof)
    w_ext = 0.0
    nsteps = len(load_history)
    max_vm = np.zeros(nsteps)
    max_sx = np.zeros(nsteps)
    w_centre = np.zeros(nsteps)
    max_alpha = np.zeros(nsteps)
    newton_its: List[int] = []
    converged = True
    pd_total = 0.0
    w_int = 0.0
    f_int_prev = np.zeros(ndof)

    for step, p_now in enumerate(load_history):
        f_ext = f_ref * float(p_now)
        u_prev = u.copy()

        ok = False
        for its in range(max_newton):
            f_int, K, sig, ep_new, al_new, ez = assemble(u)
            res = f_ext - f_int
            rnorm = float(np.linalg.norm(res[free]))
            scale = max(float(np.linalg.norm(f_ext[free])), 1e-30)
            if rnorm <= newton_tol * scale:
                ok = True
                break
            Kff = K[free][:, free].tocsc()
            du = spla.spsolve(Kff, res[free])
            if not np.all(np.isfinite(du)):
                break
            # backtracking line search: the FD layer tangent can overshoot in the
            # plastic range, so the residual must not grow
            alp = 1.0
            accepted = False
            for _ in range(5):
                u_try = u.copy()
                u_try[free] += alp * du
                f_try, _, _, _, _, _ = assemble(u_try, tangent=False)
                r_try = float(np.linalg.norm((f_ext - f_try)[free]))
                if np.isfinite(r_try) and r_try < rnorm:
                    u = u_try
                    accepted = True
                    break
                alp *= 0.5
            if not accepted:
                break
        newton_its.append(its + 1)
        w_ext += 0.5 * float(np.dot(f_ext + f_prev, u - u_prev))
        w_int += 0.5 * float(np.dot(f_int + f_int_prev, u - u_prev))
        f_int_prev = f_int.copy()
        f_prev = f_ext
        if not ok:
            converged = False
        # commit state
        f_int, _, sig, ep_new, al_new, ez = assemble(u, tangent=False)
        d_alpha = al_new - alpha_p
        sy = mat.sigma_y + mat.H * (alpha_p + 0.5 * d_alpha)
        pd_total += float(np.sum(sy * d_alpha * np.repeat(w_zeta * cfg.t * area,
                                                          nelem * 4)))
        eps_p = ep_new
        alpha_p = al_new
        eps_z = ez
        vm = _vmises(sig)
        max_vm[step] = float(np.max(vm))
        max_sx[step] = float(np.max(np.abs(sig[:, 0])))
        w_centre[step] = u[3 * centre_node]
        max_alpha[step] = float(np.max(alpha_p))
        if not np.all(np.isfinite(u)):
            converged = False
            break

    # final energy ledger
    kap = np.einsum("ngki,ni->ngk", B, u[dofs])
    eps_in = (z_layer[None, None, :, None] * kap[:, :, None, :]).reshape(N, 3)
    eps_tot = np.zeros((N, 6))
    eps_tot[:, INPLANE] = eps_in
    eps_tot[:, 2] = eps_z
    eps_e = eps_tot - eps_p
    sig_ps = sig.copy()
    sig_ps[:, 2] = 0.0            # plane stress: sigma_z is identically zero
    wz_full = np.repeat(w_zeta * cfg.t * area / 4.0, nelem * 4)
    elastic = float(np.sum(0.5 * np.sum(sig_ps * eps_e, axis=1) * wz_full))
    gam = np.einsum("nki,ni->nk", Bsh, u[dofs])            # (nelem, 2)
    shear_e = 0.5 * float(np.einsum("ni,ij,nj->", gam, Ds, gam)) * area
    elastic += shear_e

    if not converged and _retries > 0:
        # refine the load path and retry (plastic collapse / snap-through would
        # otherwise be indistinguishable from a step-size problem)
        hist = np.asarray(load_history, dtype=float)
        refined = np.interp(
            np.linspace(0.0, 1.0, 2 * (hist.size - 1) + 1),
            np.linspace(0.0, 1.0, hist.size), hist)
        return solve_plate_history(config, refined, patch=patch,
                                   newton_tol=newton_tol, max_newton=max_newton,
                                   _retries=_retries - 1)

    return PlateResponse(
        load_history=np.asarray(load_history, dtype=float),
        max_vm_stress=max_vm,
        max_abs_sigma_x=max_sx,
        centre_deflection=w_centre,
        max_alpha_p=max_alpha,
        energy={
            "external_work_J": w_ext,
            "elastic_energy_J": elastic,
            "plastic_dissipation_J": pd_total,
            "internal_work_J": w_int,
            "work_energy_closure": abs(w_ext - w_int) / max(abs(w_ext), 1e-30),
            "layer_quadrature_residual": (w_ext - elastic - pd_total) / max(abs(w_ext), 1e-30),
        },
        config=cfg,
        converged=converged,
        newton_iterations=newton_its,
    )


def navier_reference_max_stress(
    config: PlateConfig,
    pressure: float,
    patch: Optional[Tuple[float, float, float, float]] = None,
    n_terms: int = 79,
) -> Dict[str, float]:
    """Kirchhoff Navier reference (independent check of the elastic limit).

    Simply supported panel, uniform pressure on ``patch``; returns the maximum
    surface bending stress ``6 M / t^2`` and the centre deflection.
    """
    cfg = config
    a, b, t = cfg.a, cfg.b, cfg.t
    mat = cfg.material
    D = mat.E * t ** 3 / (12.0 * (1 - mat.nu ** 2))
    if patch is None:
        patch = (0.0, a, 0.0, b)
    x0, x1, y0, y1 = patch
    W = np.zeros((n_terms, n_terms))
    for m in range(1, n_terms + 1, 2):
        for n in range(1, n_terms + 1, 2):
            qmn = (4.0 * pressure / (m * n * math.pi ** 2)
                   * (math.cos(m * math.pi * x0 / a) - math.cos(m * math.pi * x1 / a))
                   * (math.cos(n * math.pi * y0 / b) - math.cos(n * math.pi * y1 / b)))
            W[m - 1, n - 1] = qmn / (D * math.pi ** 4 * (m ** 2 / a ** 2 + n ** 2 / b ** 2) ** 2)
    xs = np.linspace(0, a, 81)
    ys = np.linspace(0, b, 81)
    w_max = 0.0
    m_max = 0.0
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            mm = np.arange(1, n_terms + 1)
            nn = np.arange(1, n_terms + 1)
            sx = np.sin(np.outer(mm * math.pi * x / a, nn * math.pi * y / b))
            w = float(np.sum(W * sx))
            kx = -float(np.sum(W * (-((mm * math.pi / a) ** 2)[:, None]) * sx))
            ky = -float(np.sum(W * (-((nn * math.pi / b) ** 2)[None, :]) * sx))
            w_max = max(w_max, abs(w))
            Mx = -D * (kx + mat.nu * ky)
            My = -D * (ky + mat.nu * kx)
            m_max = max(m_max, abs(Mx), abs(My))
    return {
        "max_deflection_m": w_max,
        "max_bending_moment_Nm_per_m": m_max,
        "max_surface_stress_Pa": 6.0 * m_max / t ** 2,
        "n_terms": n_terms,
    }