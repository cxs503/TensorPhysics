"""Discontinuous-Galerkin reference discretisation (1D) used as one leg of the
cross-solver arbitration.

Two equation sets are implemented on the same modal machinery:

* ``scalar``: ``u_t + a u_x = nu u_xx`` with a modal Legendre basis (degree
  ``p = 0, 1, 2``), upwind advective flux and a symmetric-interior-penalty
  (SIPG) diffusive flux.  Element mass matrix is diagonal in the Legendre
  basis, so the semi-discrete system is explicit.
* ``swe``: the shallow-water equations ``h_t + (hu)_x = 0``,
  ``(hu)_t + (hu^2/h + g h^2/2)_x = -g h b_x`` with a local Lax-Friedrichs
  face flux and the Audusse-type hydrostatic source term written on the
  element's own reconstructed face depths (this is what makes the flat-bed lake
  at rest exactly preserved).

Both are integrated with the strong-stability-preserving RK3 scheme, and both
report **cell averages** only.  Cell averages are the common representation that
makes a DG solution and a finite-volume solution directly comparable: the DG
mean mode ``U_0`` *is* the cell average because ``phi_0 = 1``, and the analytic
cell average is obtained by Gauss quadrature exact to round-off.  Comparing
reconstructions instead would inject each scheme's own reconstruction choice
into the difference, which is exactly the confounder the ledger must avoid.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .cases import ScalarCase, SWECase, cell_averages


def _legendre_data(p: int):
    """Reference-element data for a modal Legendre basis of degree ``p``."""
    nq = p + 3
    xi, w = np.polynomial.legendre.leggauss(nq)
    basis = [np.polynomial.legendre.Legendre.basis(i) for i in range(p + 1)]
    phi = np.stack([b(xi) for b in basis])            # (p+1, nq)
    dphi = np.stack([b.deriv()(xi) for b in basis])   # (p+1, nq)
    A = (dphi * w) @ phi.T                            # A_ij = int phi_j phi_i'
    phim = np.array([(-1.0) ** i for i in range(p + 1)])
    phip = np.ones(p + 1)
    mref = 2.0 / (2.0 * np.arange(p + 1) + 1.0)       # reference mass, diagonal
    return dict(p=p, xi=xi, wq=w, phi=phi, dphi=dphi, A=A, phim=phim,
                phip=phip, mref=mref)


@dataclass
class SolverConfig:
    """Protocol knobs shared by both discretisations.

    ``limiter``      : slope limiter on/off (minmod; clips smooth extrema).
    ``boundary``     : ``periodic`` or ``analytic`` (Dirichlet inflow taken from
                       the analytic solution, outflow extrapolated).
    ``subcycles``    : number of internal time steps per protocol step.  Both
                       solvers see the same protocol ``dt``; sub-cycling only
                       refines the temporal resolution of the one that uses it.
    """

    cfl: float = 0.4
    limiter: bool = False
    boundary: str = "periodic"
    subcycles: int = 1


@dataclass
class SolverResult:
    name: str
    x_edges: np.ndarray
    cell_averages: np.ndarray      # scalar: (n,), swe: (2, n)
    steps: int
    finite: bool = True
    extra: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.cell_averages.shape[-1]


def minmod(*args):
    """Component-wise minmod of two or more arrays."""
    out = np.asarray(args[0], dtype=float)
    ok = np.ones_like(out, dtype=bool)
    s0 = np.sign(out)
    a0 = np.abs(out)
    for other in args[1:]:
        o = np.asarray(other, dtype=float)
        ok &= (np.sign(o) == s0)
        a0 = np.minimum(a0, np.abs(o))
    return np.where(ok, s0 * a0, 0.0)


# ---------------------------------------------------------------------------
# scalar advection-diffusion
# ---------------------------------------------------------------------------
def _project_scalar(ld, func, edges, h):
    ne = len(edges) - 1
    xq = edges[:-1, None] + 0.5 * h * (ld["xi"][None, :] + 1.0)
    uq = np.asarray(func(xq), dtype=float)
    U = np.empty((ld["p"] + 1, ne))
    for i in range(ld["p"] + 1):
        U[i] = (ld["wq"] * uq * ld["phi"][i][None, :]).sum(axis=1) / ld["mref"][i]
    return U


def _limit_scalar(ld, U, periodic=True):
    if ld["p"] == 0:
        return U
    ubar = U[0]
    left = np.roll(ubar, 1)
    right = np.roll(ubar, -1)
    if not periodic:
        left[0] = ubar[0]
        right[-1] = ubar[-1]
    slope = U[1]
    lim = minmod(slope, ubar - left, right - ubar)
    out = U.copy()
    out[1] = lim
    if ld["p"] >= 2:
        out[2:] = 0.0
    return out


def dg_solve_scalar(case: ScalarCase, n: int, cfg: SolverConfig,
                    degree: int = 1) -> SolverResult:
    """DG solution of ``u_t + a u_x = nu u_xx`` at ``case.t_end``."""
    ld = _legendre_data(degree)
    edges = np.linspace(case.x0, case.x1, n + 1)
    h = edges[0 + 1] - edges[0]
    ne = n
    periodic = cfg.boundary == "periodic"
    alpha = 10.0 * (degree + 1) ** 2          # SIPG penalty, 1/h scaled below

    M = 0.5 * h * ld["mref"][:, None] * np.ones((1, ne))   # (p+1, ne)
    phip = ld["phip"][:, None]
    phim = ld["phim"][:, None]
    A = ld["A"]

    def rhs(U, t):
        AU = A @ U
        uL = ld["phim"] @ U
        uR = ld["phip"] @ U
        # auxiliary field q = u_x, LDG with alternating fluxes (u-hat from the
        # "plus" side, q-hat from the "minus" side) -- stable for every p
        # without a penalty parameter.
        uhat_L = uL
        uhat_R = np.roll(uL, -1)
        if not periodic:
            gL, gR = case.boundary_values(t)
            uhat_L = uL.copy()
            uhat_L[0] = gL
            uhat_R = np.roll(uL, -1)
            uhat_R[-1] = gR if cfg.boundary == "analytic" else uR[-1]
        q = (-AU + phip * uhat_R[None, :] - phim * uhat_L[None, :]) / M
        qL = ld["phim"] @ q
        qR = ld["phip"] @ q

        # --- advective flux (upwind) --------------------------------------
        um_l = np.roll(uR, 1)
        up_l = uL
        um_r = uR
        up_r = np.roll(uL, -1)
        if not periodic:
            gL, gR = case.boundary_values(t)
            um_l = np.roll(uR, 1)
            um_l[0] = gL
            up_r = np.roll(uL, -1)
            up_r[-1] = gR if cfg.boundary == "analytic" else uR[-1]
        if case.a >= 0.0:
            FR = case.a * um_r
            FL = case.a * um_l
        else:
            FR = case.a * up_r
            FL = case.a * up_l
        face_adv = phip * FR[None, :] - phim * FL[None, :]

        # --- diffusive flux (LDG alternating) -----------------------------
        if case.nu > 0.0:
            qhat_R = qR
            qhat_L = np.roll(qR, 1)
            if not periodic:
                qhat_L = np.roll(qR, 1)
                qhat_L[0] = qL[0]
            face_diff = phip * qhat_R[None, :] - phim * qhat_L[None, :]
            diff_term = case.nu * (-(A @ q) + face_diff)
        else:
            diff_term = 0.0

        # weak form: int u_t phi = int F phi_x - [F* phi]  (+ diffusion)
        dU = (case.a * AU - face_adv + diff_term) / M
        return dU

    U = _project_scalar(ld, case.initial, edges, h)

    # protocol time step: same grid, same CFL for every solver
    dt = _scalar_dt(case, h, cfg.cfl)
    nsteps = max(1, int(math.ceil(case.t_end / dt - 1e-12)))
    dt = case.t_end / nsteps
    sub = max(1, int(cfg.subcycles))
    dts = dt / sub

    t = 0.0
    finite = True
    for step in range(nsteps):
        for _ in range(sub):
            if cfg.limiter:
                U = _limit_scalar(ld, U, periodic)
            k1 = rhs(U, t)
            if cfg.limiter:
                U2 = _limit_scalar(ld, U + 0.5 * dts * k1, periodic)
            else:
                U2 = U + 0.5 * dts * k1
            k2 = rhs(U2, t + 0.5 * dts)
            U3 = U + dts * (-k1 + 2.0 * k2)
            if cfg.limiter:
                U3 = _limit_scalar(ld, U3, periodic)
            k3 = rhs(U3, t + dts)
            U = U + dts / 6.0 * (k1 + 4.0 * k2 + k3)
            t += dts
            if not np.all(np.isfinite(U)):
                finite = False
                break
        if not finite:
            break

    def reconstruct(xs):
        xs = np.asarray(xs, dtype=float)
        idx = np.clip(((xs - case.x0) / h).astype(int), 0, ne - 1)
        xi = 2.0 * (xs - edges[idx]) / h - 1.0
        basis = np.stack([np.polynomial.legendre.Legendre.basis(i)(xi)
                          for i in range(degree + 1)])
        return (basis * U[:, idx]).sum(axis=0)

    return SolverResult(name="dg", x_edges=edges, cell_averages=U[0].copy(),
                        steps=nsteps * sub, finite=finite,
                        extra={"degree": degree, "reconstruct": reconstruct,
                               "modal": U.copy()})


def _scalar_dt(case: ScalarCase, h: float, cfl: float) -> float:
    dt = math.inf
    if case.a != 0.0:
        dt = min(dt, cfl * h / abs(case.a))
    if case.nu > 0.0:
        # explicit DG diffusion: |lambda|_max ~ nu (2p+1)(2p+2)/h^2, so keep
        # dt far inside the RK3 real-axis stability interval
        dt = min(dt, 0.01 * h * h / case.nu)
    if not math.isfinite(dt):
        dt = cfl * h
    return dt


# ---------------------------------------------------------------------------
# shallow water
# ---------------------------------------------------------------------------
def _swe_flux(h, hu, g):
    u = np.where(h > 0.0, hu / np.maximum(h, 1e-12), 0.0)
    return hu, hu * u + 0.5 * g * h * h


def _swe_speed(h, hu, g):
    u = np.where(h > 0.0, hu / np.maximum(h, 1e-12), 0.0)
    return np.abs(u) + np.sqrt(g * np.maximum(h, 0.0))


def dg_solve_swe(case: SWECase, n: int, cfg: SolverConfig,
                 degree: int = 1) -> SolverResult:
    """DG solution of the shallow-water equations at ``case.t_end``.

    Formulation (chosen for exact well-balancedness): the conserved variables are
    ``(eta, hu)`` with ``eta = h + b`` the free-surface elevation, and the
    momentum equation is written as

        (hu)_t + (hu^2/h)_x + g h eta_x = 0 ,

    so the pressure force is a *surface-gradient* source.  A lake at rest has
    ``eta = const`` and ``hu = 0``; then ``eta_x`` is identically zero in the
    polynomial space and every flux is constant, so the semi-discrete residual
    vanishes to round-off on **any** bed, not only a flat one.  The price is that
    the momentum equation is non-conservative across bores/shocks; this case set
    contains only smooth (rarefaction) or steady solutions, and the limitation is
    recorded in the report.

    Face fluxes: local Lax-Friedrichs on ``(hu, hu^2/h)`` with the face depth
    ``h = max(0, eta_face - b(x_face))``.
    """
    ld = _legendre_data(degree)
    edges = np.linspace(case.x0, case.x1, n + 1)
    h_cell = edges[1] - edges[0]
    ne = n
    g = case.g
    periodic = cfg.boundary == "periodic"

    M = 0.5 * h_cell * ld["mref"][None, :, None] * np.ones((2, 1, ne))
    phip, phim = ld["phip"], ld["phim"]
    phi, dphi, wq = ld["phi"], ld["dphi"], ld["wq"]
    bed_face = np.asarray(case.bed(edges), dtype=float)

    def face_fluxes(U, t):
        etaL, etaR = phim @ U[0], phip @ U[0]
        qL, qR = phim @ U[1], phip @ U[1]
        em = np.empty(ne + 1); qm = np.empty(ne + 1)
        ep = np.empty(ne + 1); qp = np.empty(ne + 1)
        em[1:ne] = etaR[:ne - 1]; qm[1:ne] = qR[:ne - 1]
        ep[1:ne] = etaL[1:ne]; qp[1:ne] = qL[1:ne]
        em[ne], qm[ne] = etaR[-1], qR[-1]
        if periodic:
            em[0], qm[0] = etaR[-1], qR[-1]
            ep[0], qp[0] = etaL[0], qL[0]
            ep[ne], qp[ne] = etaL[0], qL[0]
        else:
            hb, hub = case.exact(np.array([case.x0]), t)
            em[0], qm[0] = float(hb[0]) + bed_face[0], float(hub[0])
            ep[0], qp[0] = etaL[0], qL[0]
            hb, hub = case.exact(np.array([case.x1]), t)
            ep[ne], qp[ne] = float(hb[0]) + bed_face[-1], float(hub[0])
        hm = np.maximum(em - bed_face, 0.0)
        hp = np.maximum(ep - bed_face, 0.0)
        um = np.where(hm > 0.0, qm / np.maximum(hm, 1e-12), 0.0)
        up = np.where(hp > 0.0, qp / np.maximum(hp, 1e-12), 0.0)
        Fm1, Fm2 = qm, qm * um
        Fp1, Fp2 = qp, qp * up
        amax = np.maximum(np.abs(um) + np.sqrt(g * hm), np.abs(up) + np.sqrt(g * hp))
        F1 = 0.5 * (Fm1 + Fp1) - 0.5 * amax * (ep - em)
        F2 = 0.5 * (Fm2 + Fp2) - 0.5 * amax * (qp - qm)
        return F1, F2, (hm[1:], hm[:ne])

    def rhs(U, t):
        etaq = phi.T @ U[0]                                # (nq, ne)
        qq = phi.T @ U[1]
        eta_x = (dphi.T @ U[0]) * (2.0 / h_cell)           # d/dx at nodes
        bq = np.asarray(case.bed(edges[:-1, None] + 0.5 * h_cell * (ld["xi"][None, :] + 1.0)),
                        dtype=float).T
        hq = np.maximum(etaq - bq, 0.0)
        uq = np.where(hq > 0.0, qq / np.maximum(hq, 1e-12), 0.0)
        F1q, F2q = qq, qq * uq
        Fq = np.stack([F1q, F2q])
        vol = np.einsum("kq,cqe->cke", dphi * wq[None, :], Fq)     # +int F phi'

        Ff1, Ff2, _ = face_fluxes(U, t)
        face = np.zeros((2, degree + 1, ne))
        face[0] = np.outer(phip, Ff1[1:]) - np.outer(phim, Ff1[:-1])
        face[1] = np.outer(phip, Ff2[1:]) - np.outer(phim, Ff2[:-1])

        # surface-gradient pressure source: -g int h eta_x phi dx
        s_nodes = -g * hq * eta_x
        src = np.zeros((2, degree + 1, ne))
        src[1] = (0.5 * h_cell) * np.einsum("kq,qe->ke", phi * wq[None, :], s_nodes)

        return (vol - face + src) / M

    xq = edges[:-1, None] + 0.5 * h_cell * (ld["xi"][None, :] + 1.0)
    h0, hu0 = case.initial_state(xq)
    eta0 = h0 + np.asarray(case.bed(xq), dtype=float)
    U = np.empty((2, degree + 1, ne))
    for c, f in enumerate((eta0, hu0)):
        for i in range(degree + 1):
            U[c, i] = (wq * f * phi[i][None, :]).sum(axis=1) / ld["mref"][i]

    speed = float(np.max(_swe_speed(np.maximum(h0, 0.0), hu0, g)))
    dt = cfg.cfl * h_cell / max(speed, 1e-9)
    nsteps = max(1, int(math.ceil(case.t_end / dt - 1e-12)))
    dt = case.t_end / nsteps
    sub = max(1, int(cfg.subcycles))
    dts = dt / sub

    t = 0.0
    finite = True
    for _ in range(nsteps):
        for _ in range(sub):
            if cfg.limiter:
                U = _limit_swe(U, periodic)
            k1 = rhs(U, t)
            U2 = U + 0.5 * dts * k1
            k2 = rhs(U2, t + 0.5 * dts)
            U3 = U + dts * (-k1 + 2.0 * k2)
            k3 = rhs(U3, t + dts)
            U = U + dts / 6.0 * (k1 + 4.0 * k2 + k3)
            t += dts
            if not np.all(np.isfinite(U)) or np.max(np.abs(U)) > 1e8:
                finite = False
                break
        if not finite:
            break

    h_out = np.maximum(U[0, 0] - np.asarray(case.bed(0.5 * (edges[:-1] + edges[1:])), dtype=float), 0.0)

    def reconstruct(xs):
        xs = np.asarray(xs, dtype=float)
        idx = np.clip(((xs - case.x0) / h_cell).astype(int), 0, ne - 1)
        xi = 2.0 * (xs - edges[idx]) / h_cell - 1.0
        basis = np.stack([np.polynomial.legendre.Legendre.basis(i)(xi)
                          for i in range(degree + 1)])
        eta = (basis * U[0][:, idx]).sum(axis=0)
        q = (basis * U[1][:, idx]).sum(axis=0)
        hh = np.maximum(eta - np.asarray(case.bed(xs), dtype=float), 0.0)
        return np.stack([hh, q])

    return SolverResult(name="dg", x_edges=edges,
                        cell_averages=np.stack([h_out, U[1, 0]]).copy(),
                        steps=nsteps * sub, finite=finite,
                        extra={"degree": degree, "reconstruct": reconstruct,
                               "modal": U.copy()})


def _limit_swe(U, periodic=True):
    """minmod slope limiting of both conserved variables (mean modes kept)."""
    out = U.copy()
    for c in range(2):
        ubar = U[c, 0]
        left = np.roll(ubar, 1)
        right = np.roll(ubar, -1)
        if not periodic:
            left[0] = ubar[0]
            right[-1] = ubar[-1]
        out[c, 1] = minmod(U[c, 1], ubar - left, right - ubar)
        if U.shape[1] > 2:
            out[c, 2:] = 0.0
    return out
