"""Time-domain unsteady aerodynamics: Wagner indicial model in rational form.

This is the *fluid side* used by the coupled flutter solver.  It is the
classical reduced-order unsteady thin-aerofoil model (Wagner's indicial
function / Jones' rational approximation), written as a small ODE system so the
partitioned coupling can be exercised exactly like a full CFD solver would be.

Derivation
----------
Theodorsen's frequency-domain lift is

    L = 2 pi rho U b C(k) D ,     D = -h_dot + U alpha + b(1/2 - a) alpha_dot

with ``C(k)`` the lift-deficiency function.  Introducing the non-dimensional
Laplace variable ``s = i k = p b / U``, ``C`` is approximated by a 4-pole
rational function

    C(s) = 1 - sum_i a_i s / (s + beta_i) ,      sum_i a_i = 1/2

(so that ``C(0) = 1`` and ``C(inf) = 1/2``, the two exact limits).  Each pole
adds a lag state ``x_i`` with

    x_i(s) = s D(s) / (s + beta_i)   <=>   dx_i/dt = -(beta_i U/b) x_i + dD/dt ,

and the circulatory generalised force becomes the algebraic form

    L_c = 2 pi rho U b (D - sum_i a_i x_i) ,   M_c = 2 pi rho U b^2 (a + 1/2)(...) .

The non-circulatory (added-mass / apparent-inertia) part is exact:

    F_nc = - M_a q_ddot + [ pi rho b^2 U alpha_dot , -pi rho b^2 U b(1/2-a)
                            alpha_dot ] ,

with ``M_a = pi rho b^2 [[1, -ba], [-ba, b^2(1/8+a^2)]]``.  The second
bracket is the classical convective non-circulatory pair; its sign is fixed by
the requirement that a pure pitching motion about the elastic axis be damped
(positive torsional aerodynamic damping), which is a physical, not a
conventional, statement.

The four pole pairs are fitted to the *exact* Theodorsen function (see
``FIT_ERROR``); the remaining mismatch is the only approximation of the fluid
side and is reported with the acceptance numbers.  Integration of the lag
states uses the trapezoidal rule with ``dD/dt`` from the backward difference
``D(t+dt) - D(t)``, which is exact for that rule and keeps the force a pure
function of the structural state at the new time level (needed by the implicit
coupling).
"""

from __future__ import annotations

import numpy as np

from .theodorsen import apparent_mass, theodorsen_c
from .uvlm import AeroState

#: 4-pole fit of ``C(k)`` on ``k in [0.005, 20]`` (max relative error 0.185 %)
A_POLES = (0.02296842, 0.11671554, 0.26108063, 0.09876728)
BETA_POLES = (0.0081984, 0.05531121, 0.19734797, 0.64162866)
FIT_K_RANGE = (0.005, 20.0)
FIT_ERROR = 0.00185


def c_rational(k, a=A_POLES, beta=BETA_POLES):
    """Rational approximation of Theodorsen's ``C(k)`` (real reduced frequency)."""
    s = 1j * np.asarray(k, dtype=float)
    out = np.ones_like(s)
    for ai, bi in zip(a, beta):
        out = out - ai * s / (s + bi)
    return out


class RationalAero:
    """Wagner/Jones rational unsteady aerodynamics as a state-space fluid model."""

    def __init__(self, semi_chord=0.5, rho=1.225, elastic_axis=-0.5,
                 a_poles=A_POLES, beta_poles=BETA_POLES,
                 u_alpha_sign=1.0):
        self.b = float(semi_chord)
        self.rho = float(rho)
        self.a = float(elastic_axis)
        self.x_ea = self.a * self.b
        self.Ma = apparent_mass(self.b, self.a, self.rho)
        self.a_poles = np.asarray(a_poles, dtype=float)
        self.beta_poles = np.asarray(beta_poles, dtype=float)
        self.u_alpha_sign = float(u_alpha_sign)
        self.reset(0.0)

    def reset(self, U):
        self.U = float(U)
        self.x_lag = np.zeros(self.a_poles.size)
        self.D_prev = 0.0
        self._primed = False

    # -- kinematics -----------------------------------------------------
    def downwash(self, h_dot, alpha, alpha_dot):
        """Effective downwash ``D = -h_dot + U alpha + b(1/2-a) alpha_dot``."""
        return -h_dot + self.U * alpha + self.b * (0.5 - self.a) * alpha_dot

    # -- solve / commit --------------------------------------------------
    def solve(self, h_dot, alpha, alpha_dot, dt, steady=False):
        """Aerodynamic generalised force for the structural state at this level."""
        b, U, rho = self.b, self.U, self.rho
        D = self.downwash(h_dot, alpha, alpha_dot)
        if steady or not self._primed:
            x_new = np.zeros_like(self.x_lag)
        else:
            # exact integration of dx/dt = -gamma x + dD/dt over the step,
            # assuming dD/dt linear in t (piecewise-linear D).  The trapezoidal
            # rule was tried first and its O((omega dt)^2) numerical damping
            # (5 s^-1 at 240 steps/cycle) is enough to mask flutter, see report.
            gain = self.beta_poles * U / b
            decay = np.exp(-gain * dt)
            slope = (D - self.D_prev) / dt
            x_new = self.x_lag * decay + slope * (1.0 - decay) / gain
        circ = D - float(self.a_poles @ x_new)
        lift_c = 2.0 * np.pi * rho * U * b * circ
        moment_c = 2.0 * np.pi * rho * U * b * b * (self.a + 0.5) * circ
        # non-circulatory pair (convective part): depends on alpha_dot only
        lift_nc = self.u_alpha_sign * np.pi * rho * b * b * U * alpha_dot
        moment_nc = (-self.u_alpha_sign * np.pi * rho * b * b * U * b
                     * (0.5 - self.a) * alpha_dot)
        state = AeroState(
            lift=float(lift_c + lift_nc), moment=float(moment_c + moment_nc),
            gamma_b=float(2.0 * np.pi * U * b * circ / U),
            gamma_new=0.0, mom_l=float(D), mom_m=float(circ),
            mom_x=0.0, coeffs=np.array([circ]))
        state.x_lag = x_new
        state.D = D
        return state

    def commit(self, state, dt):
        self.x_lag = np.asarray(state.x_lag, dtype=float)
        self.D_prev = float(state.D)
        self._primed = True
        return state

    # -- apparent mass (used by the coupled solver) ----------------------
    def added_mass_force(self, h_ddot, alpha_ddot):
        """Non-circulatory inertia part ``-M_a q_ddot`` (N/m, N)."""
        return -(self.Ma @ np.array([h_ddot, alpha_ddot]))

    def fit_error(self, k_lo=None, k_hi=None, n=200):
        """Max relative error of the rational fit against exact ``C(k)``."""
        k_lo, k_hi = k_lo or FIT_K_RANGE[0], k_hi or FIT_K_RANGE[1]
        ks = np.geomspace(k_lo, k_hi, n)
        exact = theodorsen_c(ks)
        return float(np.max(np.abs(c_rational(ks) - exact) / np.abs(exact)))