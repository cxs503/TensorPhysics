"""SPH legs of the arbitration: viscous channel and shallow-water wave celerity.

Both use the cubic-spline kernel of the TensorSPH family (the same kernel shape is
re-used here in 1D so that the SPH discretisation under test is the standard one,
not an ad-hoc stencil):

    W(q) = (1/h) [2/3 - q^2 + q^3/2]        q = r/h < 1
    W(q) = (1/h) (2-q)^3 / 6                1 <= q < 2

``sph_poiseuille``
    The steady Stokes balance ``0 = F + nu lap(u)`` on a 1D particle lattice with
    two layers of fixed wall particles (the wall sits at the midpoint between the
    last wall particle and the first fluid particle, exactly as in a half-way
    bounce-back lattice).  The viscous operator is Monaghan's SPH Laplacian

        lap(u)_i = sum_j V_j (u_j - u_i) * ( -2 dW/dr / r )

    This is *not* a Navier-Stokes SPH run with pressure: the flow is unidirectional
    and the pressure force is exactly balanced by the body force, so solving the
    steady viscous balance is the honest statement of what SPH contributes to this
    benchmark.  The reported quantity is the peak velocity against the analytic
    parabola -- and the ratio peak/mean, which is where the classical SPH
    "plug-flow" tendency shows up.

``sph_shallow_wave``
    1D SPH discretisation of the linearised shallow-water system on a periodic
    lattice, with SPH continuity ``dh_i/dt = h_i sum_j V (u_i - u_j) dW/dx`` and
    the hydrostatic pressure force ``du_i/dt = -g sum_j V (h_j - h_i) dW/dx``.
    The celerity is measured by fitting the free-surface signal at one particle.
"""
from __future__ import annotations

import math

import numpy as np


def kernel_1d(r, h):
    """Cubic-spline kernel and its radial derivative (1D)."""
    q = np.abs(r) / h
    W = np.zeros_like(q)
    dW = np.zeros_like(q)
    m1 = q < 1.0
    m2 = (q >= 1.0) & (q < 2.0)
    W[m1] = (2.0 / 3.0 - q[m1] ** 2 + 0.5 * q[m1] ** 3) / h
    W[m2] = (2.0 - q[m2]) ** 3 / (6.0 * h)
    dW[m1] = (-2.0 * q[m1] + 1.5 * q[m1] ** 2) / (h * h)
    dW[m2] = -0.5 * (2.0 - q[m2]) ** 2 / (h * h)
    return W, dW


def sph_poiseuille(n_fluid: int = 80, height: float = 1.0, h_factor: float = 2.4,
                   nu: float = 0.01, force: float = 1.0, wall_layers: int = 2,
                   max_steps: int = 40000, tol: float = 1e-10) -> dict:
    """Steady SPH Poiseuille profile with fixed wall particles."""
    dy = height / n_fluid
    h = h_factor * dy
    y_fluid = (np.arange(n_fluid) + 0.5) * dy
    wall_lo = -0.5 * dy - dy * np.arange(wall_layers)
    wall_hi = height + 0.5 * dy + dy * np.arange(wall_layers)
    y_all = np.concatenate([wall_lo, y_fluid, wall_hi])
    n = y_all.size
    n_wall = wall_lo.size + wall_hi.size

    r = y_all[:, None] - y_all[None, :]
    rabs = np.abs(r)
    _, dWabs = kernel_1d(rabs, h)
    with np.errstate(divide="ignore", invalid="ignore"):
        K = np.where(rabs > 1e-14, -2.0 * dWabs / np.where(rabs > 1e-14, rabs, 1.0), 0.0)
    K = np.nan_to_num(K)
    np.fill_diagonal(K, 0.0)
    V = dy
    M = V * K
    # Monaghan SPH Laplacian: sum_j V K_ij (u_j - u_i) -> M u - diag(rowsum M) u
    lap_dense = M - np.diag(M.sum(axis=1))
    # sparse operator: dense BLAS mat-vec calls are thread-bound on many-core
    # nodes for these tiny matrices, the sparse product is single-threaded
    from scipy.sparse import csr_matrix
    lap = csr_matrix(lap_dense)

    wall_mask = np.zeros(n, dtype=bool)
    wall_mask[:wall_lo.size] = True
    wall_mask[n - wall_hi.size:] = True
    fluid = ~wall_mask

    # The steady balance 0 = force + nu * lap(u) with u = 0 on the wall particles
    # is a linear system.  Solving it directly (sparse LU) is algebraically the
    # same answer as pseudo-time relaxation, but without the O(N^2) relaxation
    # cost, so the resolution study can actually reach the asymptotic range.
    from scipy.sparse.linalg import spsolve
    L = lap.tocsr()
    Lff = L[fluid][:, fluid]
    rhs = -force * np.ones(int(fluid.sum()))
    u_fluid = spsolve(nu * Lff, rhs)
    u = np.zeros(n)
    u[fluid] = u_fluid
    step = 0
    prof = u[fluid]
    yc = y_fluid
    analytic = force * yc * (height - yc) / (2.0 * nu)
    peak_num = float(prof.max())
    peak_ref = float(analytic.max())
    mean_num = float(prof.mean())
    mean_ref = float(analytic.mean())
    # residual of the discrete steady balance, a direct measure of the SPH
    # operator's inconsistency at this resolution
    res = np.max(np.abs(force + nu * (lap @ u))[fluid])
    return {
        "kind": "sph_wcsp_like_poiseuille",
        "n_fluid": n_fluid, "height": height, "dy": dy, "h_factor": h_factor,
        "nu": nu, "force": force, "steps": int(step),
        "y": yc.tolist(), "profile": prof.tolist(), "analytic": analytic.tolist(),
        "peak_numeric": peak_num, "peak_analytic": peak_ref,
        "peak_rel_error": float(abs(peak_num - peak_ref) / peak_ref),
        "peak_over_mean_numeric": float(peak_num / mean_num),
        "waveform_momentum_residual": float(res),
    }


def sph_shallow_wave(h0: float = 1.0, length: float = 20.0,
                     n_particles: int = 200, h_factor: float = 1.3,
                     amplitude: float = 0.01, g: float = 9.81,
                     n_periods: float = 2.0) -> dict:
    """Celerity of the linear shallow-water wave with the SPH discretisation."""
    c_exact = math.sqrt(g * h0)
    dx = length / n_particles
    h_kernel = h_factor * dx
    x = np.arange(n_particles) * dx
    k = 2.0 * math.pi / length
    # periodic distance matrix and SPH kernel gradient
    r = x[:, None] - x[None, :]
    r = r - length * np.round(r / length)
    _, dW = kernel_1d(r, h_kernel)
    np.fill_diagonal(dW, 0.0)
    grad = dW * V_SIGN * (1.0)     # dW is already dW/dr; multiply by (dr/dx) = sign(r)
    # dW/dr * (x_i - x_j)/r  == dW/dr * sign(r)
    sign = np.sign(r)
    grad = dW * sign
    V = dx

    hs = h0 + amplitude * np.cos(k * x)
    us = c_exact * amplitude / h0 * np.cos(k * x)

    def rhs(state):
        hh, uu = state
        # dh_i/dt = h_i sum_j V (u_i - u_j) dW/dx
        dh = hh * (V * ((uu[:, None] - uu[None, :]) * grad).sum(axis=1))
        # du_i/dt = -g sum_j V (h_j - h_i) dW/dx
        du = -g * (V * ((hh[None, :] - hh[:, None]) * grad).sum(axis=1))
        return dh, du

    state = (hs.copy(), us.copy())
    dt = 0.2 * dx / c_exact
    t_end = n_periods * 2.0 * math.pi / (k * c_exact)
    nsteps = int(math.ceil(t_end / dt))
    dt = t_end / nsteps
    probe = 0
    series = np.empty(nsteps)
    for n in range(nsteps):
        k1 = rhs(state)
        s2 = (state[0] + 0.5 * dt * k1[0], state[1] + 0.5 * dt * k1[1])
        k2 = rhs(s2)
        s3 = (state[0] + 0.5 * dt * k2[0], state[1] + 0.5 * dt * k2[1])
        k3 = rhs(s3)
        s4 = (state[0] + dt * k3[0], state[1] + dt * k3[1])
        k4 = rhs(s4)
        state = (state[0] + dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
                 state[1] + dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]))
        series[n] = state[0][probe] - h0

    t = (np.arange(nsteps) + 1) * dt
    omega_exact = k * c_exact

    def resid(omega):
        A = np.stack([np.cos(omega * t), np.sin(omega * t)], axis=1)
        coef, *_ = np.linalg.lstsq(A, series, rcond=None)
        residv = series - A @ coef
        return float(np.dot(residv, residv))

    grid = np.linspace(0.7 * omega_exact, 1.3 * omega_exact, 601)
    vals = np.array([resid(w) for w in grid])
    w0 = grid[int(np.argmin(vals))]
    fine = np.linspace(w0 - 0.002 * omega_exact, w0 + 0.002 * omega_exact, 401)
    vals2 = np.array([resid(w) for w in fine])
    omega_num = float(fine[int(np.argmin(vals2))])
    c_num = omega_num / k
    return {
        "kind": "sph_linear_shallow_wave",
        "h0": h0, "length": length, "n_particles": n_particles,
        "dx": dx, "h_kernel": h_kernel, "steps": nsteps,
        "amplitude": amplitude, "c_exact": c_exact, "c_numeric": c_num,
        "celerity_rel_error": float(abs(c_num - c_exact) / c_exact),
        "omega_exact": omega_exact, "omega_numeric": omega_num,
        "points_per_wavelength": float(2.0 * math.pi / (k * dx)),
        "n_neighbours_per_side": int(math.floor(h_kernel * 2.0 / dx)),
    }


V_SIGN = 1.0