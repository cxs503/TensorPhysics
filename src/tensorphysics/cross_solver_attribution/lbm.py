"""Lattice-Boltzmann legs of the arbitration: channel (Poiseuille) and the
linear shallow-water wave.

Two independent lattice Boltzmann models are implemented here because the two
anchors need different physics:

* ``lbm_poiseuille``  -- D2Q9 BGK with Guo forcing, half-way bounce-back walls and
  periodicity in the streamwise direction.  The steady solution is the exact
  parabolic Poiseuille profile ``u(y) = F y (H - y) / (2 nu)`` with peak
  ``F H^2 / (8 nu)``; half-way bounce-back places the wall exactly half a
  lattice node outside the last fluid node, so the effective channel height is
  ``ny`` lattice units (this is what makes the peak comparison meaningful).
* ``lbm_linear_wave`` -- D1Q3 isothermal lattice Boltzmann for the *linearised*
  shallow-water system.  The linear system

      eta_t + h0 u_x = 0 ,      u_t + g eta_x = 0

  is put in the symmetric acoustic form ``p_t + c q_x = 0``, ``q_t + c p_x = 0``
  with ``p = eta``, ``q = (h0 / c) u`` and ``c = sqrt(g h0)``; the lattice sound
  speed ``1/sqrt(3)`` is then matched to ``c`` by choosing ``dt``.  The scheme
  therefore returns the shallow-water wave celerity, but it is a *linearised*
  model: the nonlinear shallow-water LBM (Zhou-type forcing) is **not**
  implemented here and no claim about it is made.

Both return plain numpy arrays; nothing is sampled from a running state.
"""
from __future__ import annotations

import math

import numpy as np

# D2Q9 lattice
C9 = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
               [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=float)
W9 = np.array([4.0 / 9.0] + [1.0 / 9.0] * 4 + [1.0 / 36.0] * 4)
OPP9 = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
# D1Q3 lattice for the acoustic / linear-shallow-water leg
C3 = np.array([-1.0, 0.0, 1.0])
W3 = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])


def _feq9(rho, ux, uy):
    """Equilibrium populations, shape (ny, nx, 9)."""
    cu = 3.0 * (C9[None, None, :, 0] * ux[..., None]
                + C9[None, None, :, 1] * uy[..., None])
    usq = 1.5 * (ux ** 2 + uy ** 2)[..., None]
    return W9[None, None, :] * rho[..., None] * (1.0 + cu + 0.5 * cu * cu - usq)


def lbm_poiseuille(nx: int = 4, ny: int = 41, tau: float = 0.8,
                   force: float = 2.0e-5, max_steps: int = 200000,
                   tol: float = 1e-12) -> dict:
    """Steady D2Q9 Poiseuille channel with a body force in ``+x``."""
    rho = np.ones((ny, nx))
    ux = np.zeros((ny, nx))
    uy = np.zeros((ny, nx))
    f = _feq9(rho, ux, uy).copy()
    nu = (tau - 0.5) / 3.0
    # Guo forcing: F = (force, 0)
    gpre = (1.0 - 0.5 / tau)
    mom_err = np.inf
    step = 0
    for step in range(1, max_steps + 1):
        rho = f.sum(axis=2)
        ux = (f * C9[:, 0]).sum(axis=2) / rho + 0.5 * force
        uy = (f * C9[:, 1]).sum(axis=2) / rho
        feq = _feq9(rho, ux, uy)
        cu = 3.0 * (C9[None, None, :, 0] * ux[..., None]
                    + C9[None, None, :, 1] * uy[..., None])
        src = gpre * W9[None, None, :] * 3.0 * (C9[None, None, :, 0] * force
                                                + cu[:, :, :] * 0.0
                                                - 0.0)
        # full Guo term for a steady body force
        src = gpre * W9[None, None, :] * (
            3.0 * (C9[None, None, :, 0] - ux[..., None] + ux[..., None]) * force)
        f_post = f - (f - feq) / tau + src
        # streaming
        f_new = np.empty_like(f_post)
        for i in range(9):
            f_new[:, :, i] = np.roll(np.roll(f_post[:, :, i], int(C9[i, 0]), axis=1),
                                     int(C9[i, 1]), axis=0)
        # half-way bounce-back at y = 0.5 and y = ny + 0.5
        for i in range(9):
            if C9[i, 1] > 0:            # moving towards +y: reflected at top wall
                f_new[-1, :, OPP9[i]] = f_post[-1, :, i]
            elif C9[i, 1] < 0:          # moving towards -y: reflected at bottom wall
                f_new[0, :, OPP9[i]] = f_post[0, :, i]
        f = f_new
        ux_new = (f * C9[:, 0]).sum(axis=2) / f.sum(axis=2) + 0.5 * force
        err = float(np.max(np.abs(ux_new - ux)))
        ux, uy = ux_new, uy
        if err < tol and step > 100:
            break
        mom_err = err

    rho = f.sum(axis=2)
    ux = (f * C9[:, 0]).sum(axis=2) / rho + 0.5 * force
    y = np.arange(ny) + 0.5
    height = float(ny)
    analytic = force * y * (height - y) / (2.0 * nu)
    prof = ux.mean(axis=1)
    peak_num = float(prof.max())
    peak_ref = force * height ** 2 / (8.0 * nu)
    return {
        "kind": "lbm_d2q9_poiseuille",
        "ny": ny, "nx": nx, "tau": tau, "nu": float(nu), "force": force,
        "steps": int(step), "residual": float(mom_err),
        "y": y.tolist(), "profile": prof.tolist(), "analytic": analytic.tolist(),
        "peak_numeric": peak_num, "peak_analytic": float(peak_ref),
        "peak_rel_error": float(abs(peak_num - peak_ref) / peak_ref),
    }


def lbm_linear_wave(h0: float = 1.0, length: float = 20.0, cells: int = 400,
                    amplitude: float = 0.01, g: float = 9.81,
                    n_periods: float = 2.0, cfl: float = 0.3) -> dict:
    """Celerity of the linear shallow-water wave with a D1Q3 acoustic LBM."""
    c_phys = math.sqrt(g * h0)
    dx = length / cells
    # lattice: c_lattice = (dx/dt)/sqrt(3) must equal c_phys
    dt = dx / (math.sqrt(3.0) * c_phys)
    k = 2.0 * math.pi / length
    x = (np.arange(cells) + 0.5) * dx
    eta = amplitude * np.cos(k * x)
    beta = c_phys / h0                  # q = beta * u
    u0 = c_phys * eta / h0              # right-going linear wave
    p0 = 1.0 + eta                      # acoustic "density" with unit mean
    q0 = beta * u0
    tau = 0.8
    f = np.empty((cells, 3))
    for i in range(3):
        f[:, i] = W3[i] * p0 * (1.0 + 3.0 * C3[i] * q0 / p0)

    t_end = n_periods * 2.0 * math.pi / (k * c_phys)
    nsteps = int(math.ceil(t_end / dt))
    dt = t_end / nsteps
    probe = 0
    series = np.empty(nsteps)
    for n in range(nsteps):
        # macroscopic
        p = f.sum(axis=1)
        q = (f * C3).sum(axis=1)
        q_eq = q
        feq = np.stack([W3[i] * p * (1.0 + 3.0 * C3[i] * q_eq / p)
                        for i in range(3)], axis=1)
        f_post = f - (f - feq) / tau
        for i in range(3):
            f[:, i] = np.roll(f_post[:, i], int(C3[i]))
        series[n] = f[probe].sum() - 1.0
    # least-squares frequency estimate on a scan around the exact value
    t = (np.arange(nsteps) + 1) * dt
    omega_exact = k * c_phys

    def resid(omega):
        a = np.cos(omega * t)
        b = np.sin(omega * t)
        A = np.stack([a, b], axis=1)
        coef, *_ = np.linalg.lstsq(A, series, rcond=None)
        r = series - A @ coef
        return float(np.dot(r, r))

    grid = np.linspace(0.7 * omega_exact, 1.3 * omega_exact, 601)
    vals = np.array([resid(w) for w in grid])
    w0 = grid[int(np.argmin(vals))]
    fine = np.linspace(w0 - 0.002 * omega_exact, w0 + 0.002 * omega_exact, 401)
    vals2 = np.array([resid(w) for w in fine])
    omega_num = float(fine[int(np.argmin(vals2))])
    c_num = omega_num / k
    return {
        "kind": "lbm_d1q3_linear_shallow_wave",
        "h0": h0, "length": length, "cells": cells, "dx": dx, "dt": dt,
        "steps": nsteps, "c_exact": c_phys, "c_numeric": c_num,
        "celerity_rel_error": float(abs(c_num - c_phys) / c_phys),
        "omega_exact": omega_exact, "omega_numeric": omega_num,
        "points_per_wavelength": float(2.0 * math.pi / (k * dx)),
    }