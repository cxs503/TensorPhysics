"""Finite-volume reference discretisation (1D) used as the other leg of the
cross-solver arbitration.

Design choices, stated because they are the axes of the attribution study:

* ``order = 1``      : first-order upwind advection / central diffusion.  This
  is the classic production default of an unstructured FVM code and it is the
  "truncation mismatch" that the ledger has to price.
* ``order = 2``      : MUSCL reconstruction of face values from cell averages
  (central slope when the limiter is off, minmod/van Leer when it is on) with
  the same RK3 integrator as the DG leg, so that a difference in the answers
  cannot be blamed on the time integrator.
* ``subcycles``      : number of internal substeps per protocol step, i.e. the
  sub-cycling knob of a coupled DG-FVM run.
* shallow water      : local Lax-Friedrichs flux with the Audusse hydrostatic
  reconstruction (``h* = max(0, h + b - b_face)``) and the momentum source
  written on the element's own reconstructed face depths, which makes the
  flat-bed lake at rest exact and the bumpy-bed lake at rest second-order
  accurate in the well-balancedness sense.

The output is cell averages, exactly as for the DG leg, so the two solutions
live in the same space and their difference is a meaningful "disagreement".
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .cases import ScalarCase, SWECase
from .dg import SolverConfig, SolverResult, _scalar_dt, minmod, _swe_flux, _swe_speed


# ---------------------------------------------------------------------------
# scalar advection-diffusion
# ---------------------------------------------------------------------------
def _face_values(ubar: np.ndarray, limiter: bool, order: int,
                 periodic: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return the left/right face values of every cell (reconstruction)."""
    if order <= 1:
        return ubar.copy(), ubar.copy()
    left = np.roll(ubar, 1)
    right = np.roll(ubar, -1)
    if not periodic:
        left[0] = ubar[0]
        right[-1] = ubar[-1]
    if limiter:
        slope = minmod(ubar - left, right - ubar)
    else:
        slope = 0.5 * (right - left)
    # value at the two faces of the cell
    face_lo = ubar - 0.5 * slope
    face_hi = ubar + 0.5 * slope
    return face_lo, face_hi


def fvm_solve_scalar(case: ScalarCase, n: int, cfg: SolverConfig,
                     order: int = 1) -> SolverResult:
    edges = np.linspace(case.x0, case.x1, n + 1)
    h = edges[1] - edges[0]
    periodic = cfg.boundary == "periodic"

    def rhs(u, t):
        lo, hi = _face_values(u, cfg.limiter, order, periodic)
        # face states
        if periodic:
            minus = np.roll(hi, 1)
            plus = lo
        else:
            gL, gR = case.boundary_values(t)
            minus = np.roll(hi, 1)
            plus = lo.copy()
            minus[0] = gL
            plus[-1] = gR if cfg.boundary == "analytic" else hi[-1]
        # advective upwind flux at each cell's two faces
        def flux(um, up):
            return case.a * (um if case.a >= 0.0 else up)

        F_left = flux(minus, plus)
        # right face of cell e is the left face of cell e+1
        if periodic:
            minus_r = hi
            plus_r = np.roll(lo, -1)
        else:
            gL, gR = case.boundary_values(t)
            minus_r = hi
            plus_r = np.roll(lo, -1)
            plus_r[-1] = gR if cfg.boundary == "analytic" else hi[-1]
        F_right = flux(minus_r, plus_r)

        dudt = -(F_right - F_left) / h

        if case.nu > 0.0:
            # central face gradient (2nd order, independent of the slope limiter)
            dudt += case.nu * (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / (h * h)
            if not periodic:
                gL, gR = case.boundary_values(t)
                gR_eff = gR if cfg.boundary == "analytic" else u[-1]
                lap = np.empty_like(u)
                lap[0] = (u[1] - 2.0 * u[0] + gL) / (h * h)
                lap[-1] = (gR_eff - 2.0 * u[-1] + u[-2]) / (h * h)
                lap[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (h * h)
                dudt = -(F_right - F_left) / h + case.nu * lap
        return dudt

    u = case.initial_averages(n)
    dt = _scalar_dt(case, h, cfg.cfl)
    nsteps = max(1, int(math.ceil(case.t_end / dt - 1e-12)))
    dt = case.t_end / nsteps
    sub = max(1, int(cfg.subcycles))
    dts = dt / sub

    t = 0.0
    finite = True
    for _ in range(nsteps):
        for _ in range(sub):
            k1 = rhs(u, t)
            k2 = rhs(u + 0.5 * dts * k1, t + 0.5 * dts)
            u = u + dts * k2                      # midpoint RK2
            t += dts
            if not np.all(np.isfinite(u)):
                finite = False
                break
        if not finite:
            break

    lo_f, hi_f = _face_values(u, cfg.limiter, order, periodic)
    slope = hi_f - lo_f

    def reconstruct(xs):
        xs = np.asarray(xs, dtype=float)
        idx = np.clip(((xs - case.x0) / h).astype(int), 0, n - 1)
        if order <= 1:
            return u[idx]
        xi = 2.0 * (xs - edges[idx]) / h - 1.0
        return u[idx] + 0.5 * slope[idx] * xi

    return SolverResult(name="fvm", x_edges=edges, cell_averages=u.copy(),
                        steps=nsteps * sub, finite=finite,
                        extra={"order": order, "reconstruct": reconstruct})


# ---------------------------------------------------------------------------
# shallow water
# ---------------------------------------------------------------------------
def fvm_solve_swe(case: SWECase, n: int, cfg: SolverConfig,
                  order: int = 1) -> SolverResult:
    """Finite-volume solution of the shallow-water equations.

    Same formulation as the DG leg (``eta = h + b`` conserved, surface-gradient
    pressure source ``-g h eta_x``) so that the two schemes are compared on the
    same equation set and the *only* differences are the reconstruction, the
    limiter, the flux dissipation and the boundary closure.  A lake at rest is
    preserved exactly for any bed because ``eta`` is constant and every flux
    reduces to its steady value.
    """
    edges = np.linspace(case.x0, case.x1, n + 1)
    h_cell = edges[1] - edges[0]
    g = case.g
    periodic = cfg.boundary == "periodic"
    bed_face = np.asarray(case.bed(edges), dtype=float)
    bed_cell = np.asarray(case.bed(0.5 * (edges[:-1] + edges[1:])), dtype=float)

    def slopes(vals):
        if order <= 1:
            return np.zeros_like(vals)
        left = np.roll(vals, 1)
        right = np.roll(vals, -1)
        if not periodic:
            left[0] = vals[0]
            right[-1] = vals[-1]
        return minmod(vals - left, right - vals) if cfg.limiter else 0.5 * (right - left)

    def reconstruct(vals, sl):
        return vals - 0.5 * sl, vals + 0.5 * sl

    def rhs(q, t):
        eta, hu = q[0], q[1]
        h_cellv = np.maximum(eta - bed_cell, 0.0)
        sl_e = slopes(eta)
        sl_q = slopes(hu)
        e_lo, e_hi = reconstruct(eta, sl_e)
        q_lo, q_hi = reconstruct(hu, sl_q)
        h_lo = np.maximum(e_lo - bed_face[:-1], 0.0)
        h_hi = np.maximum(e_hi - bed_face[1:], 0.0)
        u_lo = np.where(h_lo > 0.0, q_lo / np.maximum(h_lo, 1e-12), 0.0)
        u_hi = np.where(h_hi > 0.0, q_hi / np.maximum(h_hi, 1e-12), 0.0)
        # dry-state regularisation: a cell that is drying out must not generate an
        # unbounded velocity from a vanishing depth
        u_floor = np.sqrt(2.0 * g * np.maximum(h_lo + h_hi, 1e-12))
        u_lo = np.clip(u_lo, -10.0 * u_floor, 10.0 * u_floor)
        u_hi = np.clip(u_hi, -10.0 * u_floor, 10.0 * u_floor)
        q_lo, q_hi = h_lo * u_lo, h_hi * u_hi

        em = np.empty(n + 1); qm = np.empty(n + 1)
        ep = np.empty(n + 1); qp = np.empty(n + 1)
        em[1:n] = e_hi[:n - 1]; qm[1:n] = q_hi[:n - 1]
        ep[1:n] = e_lo[1:n]; qp[1:n] = q_lo[1:n]
        em[n], qm[n] = e_hi[-1], q_hi[-1]
        if periodic:
            em[0], qm[0] = e_hi[-1], q_hi[-1]
            ep[0], qp[0] = e_lo[0], q_lo[0]
            ep[n], qp[n] = e_lo[0], q_lo[0]
        else:
            hb, hub = case.exact(np.array([case.x0]), t)
            em[0], qm[0] = float(hb[0]) + bed_face[0], float(hub[0])
            ep[0], qp[0] = e_lo[0], q_lo[0]
            hb, hub = case.exact(np.array([case.x1]), t)
            ep[n], qp[n] = float(hb[0]) + bed_face[-1], float(hub[0])
        hm = np.maximum(em - bed_face, 0.0)
        hp = np.maximum(ep - bed_face, 0.0)
        um = np.where(hm > 0.0, qm / np.maximum(hm, 1e-12), 0.0)
        up = np.where(hp > 0.0, qp / np.maximum(hp, 1e-12), 0.0)
        amax = np.maximum(np.abs(um) + np.sqrt(g * hm), np.abs(up) + np.sqrt(g * hp))
        F1 = 0.5 * (qm + qp) - 0.5 * amax * (ep - em)
        F2 = 0.5 * (qm * um + qp * up) - 0.5 * amax * (qp - qm)

        dq = np.empty_like(q)
        dq[0] = -(F1[1:] - F1[:-1]) / h_cell
        h_face_avg = 0.5 * (h_lo + h_hi)
        dq[1] = -(F2[1:] - F2[:-1]) / h_cell - g * h_face_avg * (e_hi - e_lo) / h_cell
        return dq

    xc = 0.5 * (edges[:-1] + edges[1:])
    h0, hu0 = case.initial_state(xc)
    eta0 = np.asarray(h0, dtype=float) + bed_cell
    q = np.stack([eta0, np.asarray(hu0, dtype=float)])

    speed = float(np.max(_swe_speed(np.maximum(np.asarray(h0, dtype=float), 0.0),
                                    np.asarray(hu0, dtype=float), g)))
    dt = cfg.cfl * h_cell / max(speed, 1e-9)
    nsteps = max(1, int(math.ceil(case.t_end / dt - 1e-12)))
    dt = case.t_end / nsteps
    sub = max(1, int(cfg.subcycles))
    dts = dt / sub

    t = 0.0
    finite = True
    for _ in range(nsteps):
        for _ in range(sub):
            k1 = rhs(q, t)
            q2 = q + 0.5 * dts * k1
            k2 = rhs(q2, t + 0.5 * dts)
            q = q + dts * k2
            t += dts
            if not np.all(np.isfinite(q)) or np.max(np.abs(q)) > 1e8:
                finite = False
                break
        if not finite:
            break

    h_out = np.maximum(q[0] - bed_cell, 0.0)
    sl_e = slopes(q[0]); sl_q = slopes(q[1])

    def reconstruct_out(xs):
        xs = np.asarray(xs, dtype=float)
        idx = np.clip(((xs - case.x0) / h_cell).astype(int), 0, n - 1)
        if order <= 1:
            eta = q[0][idx]; qq = q[1][idx]
        else:
            xi = 2.0 * (xs - edges[idx]) / h_cell - 1.0
            eta = q[0][idx] + 0.5 * sl_e[idx] * xi
            qq = q[1][idx] + 0.5 * sl_q[idx] * xi
        hh = np.maximum(eta - np.asarray(case.bed(xs), dtype=float), 0.0)
        return np.stack([hh, qq])

    return SolverResult(name="fvm", x_edges=edges,
                        cell_averages=np.stack([h_out, q[1]]).copy(),
                        steps=nsteps * sub, finite=finite,
                        extra={"order": order, "reconstruct": reconstruct_out})
