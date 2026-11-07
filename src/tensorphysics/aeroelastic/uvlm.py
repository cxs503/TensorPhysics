"""Linearised unsteady thin-aerofoil solver: Glauert-spectral vortex lattice.

Model
-----
Symmetric (zero-camber) aerofoil, e.g. NACA 0012 at small incidence, on a
straight chord line in incompressible inviscid flow of speed ``U`` along +x.
Small disturbances, hence the chord line stays on ``z = 0`` and the free wake
stays on ``z = 0`` convected at ``U`` (standard *linearised* fixed-wake UVLM).

The bound vorticity uses the Glauert basis

    x(theta) = -b cos(theta)                    (x measured from mid-chord)
    gamma(theta) = 2U [ a0 (1+cos)/sin + sum_{n>=1} a_n sin(n theta) ]

which inverts the Soehngen integral equation exactly: the velocity induced on
the chord line is

    w_ind(theta) = -U [ a0 + sum_{n>=1} a_n cos(n theta) ] ,

the Kutta condition ``gamma(trailing edge) = 0`` holds analytically, and the
leading-edge 1/sqrt singularity is exact rather than panel-smeared.

The wake is carried as a **vortex sheet** (piecewise constant strength on the
segments shed at each time step) rather than as point vortices: the induced
velocity of a constant-strength segment uses the bounded kernel

    w = (gamma / 2 pi) ln |(x - x_left) / (x - x_right)| ,

which removes the 1/r stiffness of a vortex point sitting just downstream of
the trailing edge (a vortex-point wake makes the discrete Kutta system
ill-conditioned and the time march diverges, see the campaign report).  As
``dt -> 0`` the segment contribution converges to the point-vortex value.

Kinematics (h up, alpha nose-up about an elastic axis at ``a*b`` from mid-chord):
wall normal velocity ``h_dot - x alpha_dot``, tangency on the chord line

    w_req(x) = h_dot - x alpha_dot - U alpha .

Loading from the linearised unsteady Bernoulli equation integrated along the
chord (s from the leading edge, ``Gamma_tail(x) = int_x^c gamma ds``)::

    L = rho U Gamma_b - rho d/dt ( int Gamma_tail dx )
    M = rho [ d/dt int Gamma_tail (x-x_ea) dx - U int gamma (x-x_ea) dx ]

Both d/dt terms reproduce the flat-plate apparent mass ``pi rho b^2`` and
apparent inertia ``pi rho b^4 (1/8 + a^2)`` (tested).

Shedding follows Kelvin's theorem ``Delta Gamma = Gamma_b^n - Gamma_b^{n+1}``;
the freshly shed sheet segment is an extra unknown of the same linear system,
so the bound-circulation / wake coupling carries no one-step lag.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TWO_PI = 2.0 * np.pi


@dataclass
class AeroState:
    """Instantaneous aerodynamic state of the aerofoil section."""

    lift: float          # N/m, positive up
    moment: float        # N, positive nose-up about the elastic axis
    gamma_b: float       # bound circulation, m^2/s
    gamma_new: float     # sheet strength shed during this step, m/s
    mom_l: float         # int Gamma_tail dx      (d/dt drives the lift)
    mom_m: float         # int Gamma_tail (x-x_ea) dx  (d/dt drives the moment)
    mom_x: float         # int gamma (x-x_ea) dx
    coeffs: np.ndarray   # Glauert coefficients a0..aN


def _sheet_kernel(x, x1, x2, core=0.0):
    """Induced vertical velocity of constant-strength vortex sheet segments.

    ``core`` is the Lamb/Oseen-type blob radius: the singular logarithmic kernel
    ``ln|r1/r2|`` is replaced by ``0.5 ln((r1^2+core^2)/(r2^2+core^2))`` which
    is bounded and identical to the singular kernel for ``core -> 0``.
    """
    r1 = x[:, None] - np.asarray(x1)[None, :]
    r2 = x[:, None] - np.asarray(x2)[None, :]
    # sign follows Biot-Savart for vorticity along +y in the (x, z) plane:
    # a point vortex induces w = -Gamma / (2 pi (x - x0))
    if core:
        return -0.25 * np.log((r1 * r1 + core * core) / (r2 * r2 + core * core)) / \
            np.pi
    return -np.log(np.abs(r1) / np.abs(r2)) / TWO_PI


class GlauertVortexLattice:
    """Time-marching linearised unsteady thin-aerofoil solver (2-D, unit span)."""

    def __init__(self, semi_chord=0.5, n_terms=20, rho=1.225,
                 elastic_axis=-0.5, wake_length_chords=20.0,
                 max_wake_segments=320, wake_coupling="implicit", core=0.0):
        self.b = float(semi_chord)
        self.n_terms = int(n_terms)
        self.rho = float(rho)
        self.a = float(elastic_axis)
        self.wake_length = float(wake_length_chords) * 2.0 * self.b
        self.max_wake_segments = int(max_wake_segments)
        self.wake_coupling = str(wake_coupling)
        self.core = float(core)
        self.x_ea = self.a * self.b

        n = self.n_terms
        self.theta_c = np.arange(1, n + 2) * np.pi / (n + 2)   # uniform in theta
        self.x_c = -self.b * np.cos(self.theta_c)
        nk = np.arange(0, n + 1)
        self.cos_mat = np.cos(np.outer(self.theta_c, nk))      # (n+1, n+1)
        # Quadrature grid in theta.  All integrands are trigonometric
        # polynomials of degree <= 2N+1, so a uniform grid with nq > 4N makes
        # trapezoidal quadrature EXACT; nq = 161 is far beyond that while still
        # cheap enough for the wake projection.
        # endpoints INCLUDED: gamma*sin(theta) is regular there, and the uniform
        # trapezoidal rule is then exact for the trigonometric polynomials.
        self.theta_q = np.linspace(0.0, np.pi, 161)
        self.sin_q = np.sin(self.theta_q)
        self.one_plus_cos_q = 1.0 + np.cos(self.theta_q)
        self.x_q = -self.b * np.cos(self.theta_q)
        self.s_q = self.b * (1.0 + np.cos(self.theta_q))
        self.basis_q = np.cos(np.outer(self.theta_q, nk))      # (nq, n+1)
        nk1 = np.arange(1, n + 1)
        self.sin_nq = np.sin(np.outer(self.theta_q, nk1)) * self.sin_q[:, None]
        # Galerkin (weighted-residual) mass matrix of the Glauert basis with the
        # natural weight sin(theta); well conditioned, unlike the interpolation
        # matrix which degenerates towards the trailing edge (cos(n theta) -> +-1).
        a_mat = (self.basis_q * self.sin_q[:, None])[:, :, None] * \
            self.basis_q[:, None, :]
        self.G = np.trapezoid(a_mat, x=self.theta_q, axis=0)
        self.P1 = self.G[0, :].copy()                          # proj of a constant
        self.P2 = np.trapezoid((self.basis_q * self.sin_q[:, None])
                               * self.x_q[:, None], x=self.theta_q, axis=0)
        self.U = 0.0
        self.reset(0.0)

    # -- bookkeeping ----------------------------------------------------
    def reset(self, U):
        self.U = float(U)
        self.wseg_x1 = np.zeros(0)
        self.wseg_x2 = np.zeros(0)
        self.wseg_g = np.zeros(0)
        self.gamma_b_prev = 0.0
        self.mom_l_prev = 0.0
        self.mom_m_prev = 0.0
        self._primed = False

    # -- wake ------------------------------------------------------------
    def _wake_downwash(self, x):
        if self.wseg_g.size == 0:
            return np.zeros_like(x)
        return _sheet_kernel(x, self.wseg_x1, self.wseg_x2,
                             self.core) @ self.wseg_g

    def _new_segment_kernel(self, x, dt):
        """Kernel of the segment shed during this step: [b, b + U dt]."""
        x1 = self.b
        x2 = self.b + self.U * dt
        return _sheet_kernel(x, [x1], [x2], self.core)[:, 0]

    def wake_circulation(self):
        return float(np.sum(self.wseg_g * (self.wseg_x2 - self.wseg_x1)))

    # -- circulation solve ---------------------------------------------
    def _project(self, field):
        """Galerkin projection of a chordwise field onto the Glauert basis."""
        return np.trapezoid((self.basis_q * self.sin_q[:, None])
                            * field[:, None], x=self.theta_q, axis=0)

    def solve(self, h_dot, alpha, alpha_dot, dt, steady=False):
        """Solve bound circulation + freshly shed sheet strength for this instant.

        A pure function of the *frozen* wake history, so the tight-coupling
        sub-iteration can call it repeatedly without corrupting the wake.
        ``steady=True`` returns the quasi-steady solution without shedding.
        """
        U = self.U
        b = self.b
        n = self.n_terms
        # wall normal velocity for a rigid section pitching about the ELASTIC
        # AXIS: h_dot - (x - x_ea) alpha_dot
        x_rel = self.x_c - self.x_ea
        w_req = h_dot - x_rel * alpha_dot - U * alpha
        w_old = self._wake_downwash(self.x_c)
        # projections of the requested downwash (analytic in the coefficients)
        b_req = (h_dot * self.P1 - alpha_dot * (self.P2 - self.x_ea * self.P1)
                 - U * alpha * self.P1)
        b_wake = self._project(self._wake_downwash(self.theta_q))
        rhs_top = (b_wake - b_req) / U

        if steady:
            a = np.linalg.solve(self.G, rhs_top)
            g_new = 0.0
        elif self.wake_coupling == "explicit":
            a = np.linalg.solve(self.G, rhs_top)
            gamma_b = self._bound_circulation(a, U, b)
            g_new = (self.gamma_b_prev - gamma_b) / (U * dt)
        else:
            x_new2 = b + U * dt
            knew_q = _sheet_kernel(self.theta_q, [b], [x_new2],
                                   self.core)[:, 0]
            mat = np.zeros((n + 2, n + 2))
            mat[: n + 1, : n + 1] = self.G
            mat[: n + 1, n + 1] = self._project(knew_q) / U
            rhs = np.zeros(n + 2)
            rhs[: n + 1] = rhs_top
            mat[n + 1, 0] = TWO_PI * U * b
            if n >= 1:
                mat[n + 1, 1] = np.pi * U * b
            mat[n + 1, n + 1] = U * dt
            rhs[n + 1] = self.gamma_b_prev
            sol = np.linalg.solve(mat, rhs)
            a = sol[: n + 1]
            g_new = sol[n + 1]

        gamma_b = self._bound_circulation(a, U, b)
        # gamma sin(theta) is smooth even though gamma itself has a 1/sqrt
        # leading-edge singularity: integrate everything through it.
        g_q = 2.0 * U * (a[0] * self.one_plus_cos_q
                         + (self.sin_nq * a[1:]).sum(axis=1))
        dth = self.theta_q[1] - self.theta_q[0]
        # Gamma_tail(x) = int_x^c gamma ds, integrated from the trailing edge
        tail = np.cumsum(g_q[::-1])[::-1] * dth * b
        tail = tail - 0.5 * g_q * dth * b          # half-cell end correction
        mom_x = b * np.trapezoid(g_q * (self.x_q - self.x_ea), x=self.theta_q)
        mom_l = b * np.trapezoid(tail * self.sin_q, x=self.theta_q)
        mom_m = b * np.trapezoid(tail * (self.x_q - self.x_ea) * self.sin_q,
                                 x=self.theta_q)
        if steady or not self._primed:
            d_l = 0.0
            d_m = 0.0
        else:
            d_l = (mom_l - self.mom_l_prev) / dt
            d_m = (mom_m - self.mom_m_prev) / dt
        rho = self.rho
        lift = rho * U * gamma_b - rho * d_l
        moment = rho * (d_m - U * mom_x)
        return AeroState(lift=lift, moment=moment, gamma_b=gamma_b,
                         gamma_new=g_new, mom_l=mom_l, mom_m=mom_m,
                         mom_x=mom_x, coeffs=a)

    @staticmethod
    def _bound_circulation(a, U, b):
        val = TWO_PI * U * b * a[0]
        if a.size >= 2:
            val += np.pi * U * b * a[1]
        return float(val)

    def commit(self, state, dt):
        """Accept ``state`` as the converged time level and shed its vorticity."""
        if self.wseg_g.size:
            self.wseg_x1 = self.wseg_x1 + self.U * dt
            self.wseg_x2 = self.wseg_x2 + self.U * dt
        if state.gamma_new != 0.0:
            self.wseg_x1 = np.append(self.wseg_x1, self.b)
            self.wseg_x2 = np.append(self.wseg_x2, self.b + self.U * dt)
            self.wseg_g = np.append(self.wseg_g, state.gamma_new)
        self._trim_wake()
        self.gamma_b_prev = state.gamma_b
        self.mom_l_prev = state.mom_l
        self.mom_m_prev = state.mom_m
        self._primed = True
        return state

    def _trim_wake(self):
        if self.wseg_g.size > self.max_wake_segments:
            # merge the outermost (farthest) pairs: conserves circulation and
            # leaves the near-wake interaction grid unchanged
            n_merge = self.wseg_g.size - int(0.75 * self.max_wake_segments)
            i0 = self.wseg_g.size - n_merge
            if i0 > 0:
                circ = np.sum(self.wseg_g[i0:] *
                              (self.wseg_x2[i0:] - self.wseg_x1[i0:]))
                x1 = self.wseg_x1[i0]
                x2 = self.wseg_x2[-1]
                self.wseg_x1 = np.append(self.wseg_x1[:i0], x1)
                self.wseg_x2 = np.append(self.wseg_x2[:i0], x2)
                self.wseg_g = np.append(self.wseg_g[:i0], circ / (x2 - x1))
        keep = self.wseg_x1 <= self.b + self.wake_length
        if not np.all(keep):
            self.wseg_x1 = self.wseg_x1[keep]
            self.wseg_x2 = self.wseg_x2[keep]
            self.wseg_g = self.wseg_g[keep]

    # -- diagnostics ----------------------------------------------------
    def gamma_at(self, x, coeffs):
        """Vortex-sheet strength gamma(x) for the given Glauert coefficients."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        theta = np.arccos(np.clip(-x / self.b, -1.0, 1.0))
        n = self.n_terms
        val = coeffs[0] * (1.0 + np.cos(theta)) / np.sin(theta)
        for k in range(1, n + 1):
            val = val + coeffs[k] * np.sin(k * theta)
        return 2.0 * self.U * val

    def pressure_jump(self, x, coeffs, d_dt_gamma_tail=None):
        """Linearised pressure jump ``p_upper - p_lower`` (Pa) on the chord line.

        ``d_dt_gamma_tail`` is d/dt of ``Gamma_b - int_LE^x gamma ds`` supplied by
        the caller (zero gives the quasi-steady field).
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        gamma_b = self._bound_circulation(coeffs, self.U, self.b)
        xs = np.linspace(-self.b, self.b, 4001)
        gs = self.gamma_at(np.clip(xs, -0.999999 * self.b, 0.999999 * self.b),
                           coeffs)
        cum = np.concatenate(([0.0], np.cumsum(0.5 * (gs[1:] + gs[:-1])
                                               * np.diff(xs))))
        tail = gamma_b - np.interp(x, xs, cum)
        dp = -self.rho * self.U * self.gamma_at(
            np.clip(x, -0.999999 * self.b, 0.999999 * self.b), coeffs)
        if d_dt_gamma_tail is not None:
            dp = dp + self.rho * d_dt_gamma_tail
        return dp, tail