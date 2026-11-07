"""Two-degree-of-freedom (plunge / pitch) aeroelastic section.

The section is the *modal reduction* of a wing-box FEM to its two lowest
bending/torsion modes, which is the standard typical-section model of classical
aeroelasticity: generalised coordinates ``q = (h, alpha)`` with

    M q_ddot + C q_dot + K q = F_aero

    M = [[m,            m b x_alpha],
         [m b x_alpha,  m b^2 r_alpha^2]]
    K = diag(m omega_h^2, m b^2 omega_alpha^2)

``h`` is plunge (up positive), ``alpha`` pitch nose-up about the elastic axis
located at ``a*b`` from mid-chord, ``x_alpha`` = static unbalance (CG offset aft
of the EA, divided by b) and ``r_alpha^2 = I_alpha / (m b^2)``.

Time integration is the unconditionally stable Newmark average-acceleration
scheme (gamma = 1/2, beta = 1/4), second order in the time step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SectionState:
    q: np.ndarray = field(default_factory=lambda: np.zeros(2))
    v: np.ndarray = field(default_factory=lambda: np.zeros(2))
    a: np.ndarray = field(default_factory=lambda: np.zeros(2))
    t: float = 0.0


class TypicalSection:
    """Plunge/pitch typical section with Newmark time integration."""

    def __init__(self, mass, semi_chord, omega_h, omega_alpha,
                 x_alpha=0.0, r_alpha2=0.25, elastic_axis=-0.5,
                 zeta_h=0.0, zeta_alpha=0.0):
        self.mass = float(mass)                 # kg/m
        self.b = float(semi_chord)              # m
        self.omega_h = float(omega_h)
        self.omega_alpha = float(omega_alpha)
        self.x_alpha = float(x_alpha)
        self.r_alpha2 = float(r_alpha2)
        self.a = float(elastic_axis)
        m, b = self.mass, self.b
        self.M = np.array([[m, m * b * self.x_alpha],
                           [m * b * self.x_alpha, m * b * b * self.r_alpha2]])
        self.K = np.array([[m * self.omega_h ** 2, 0.0],
                           [0.0, m * b * b * self.omega_alpha ** 2]])
        self.C = np.array([
            [2.0 * zeta_h * m * self.omega_h, 0.0],
            [0.0, 2.0 * zeta_alpha * m * b * b * self.omega_alpha],
        ])

    # -- mass ratio -----------------------------------------------------
    def mass_ratio(self, rho):
        """``mu = m / (pi rho b^2)`` (structure mass over displaced-fluid mass)."""
        return self.mass / (np.pi * rho * self.b ** 2)

    def added_mass_matrix(self, rho):
        """Flat-plate apparent mass / inertia matrix (per unit span)."""
        b, a = self.b, self.a
        return np.pi * rho * b * b * np.array([[1.0, -b * a],
                                               [-b * a, b * b * (0.125 + a * a)]])

    def added_mass_ratio(self, rho):
        """lambda_max(M_s^-1 M_added): > 1 means the fluid apparent mass exceeds
        the structural mass at the interface (the added-mass-instability regime)."""
        prod = np.linalg.solve(self.M, self.added_mass_matrix(rho))
        return float(np.max(np.abs(np.linalg.eigvals(prod))))

    # -- energies -------------------------------------------------------
    def energy(self, state):
        return float(0.5 * state.v @ self.M @ state.v
                     + 0.5 * state.q @ self.K @ state.q)

    def damping_loss(self, state):
        return float(state.v @ self.C @ state.v)

    def natural_frequencies(self):
        lam = np.linalg.eigvals(np.linalg.solve(self.M, self.K))
        return np.sort(np.sqrt(np.abs(lam)))

    # -- integration ----------------------------------------------------
    def newmark_matrices(self, dt, gamma=0.5, beta=0.25):
        M, C, K = self.M, self.C, self.K
        keff = M / (beta * dt * dt) + C * (gamma / (beta * dt)) + K
        return keff

    def newmark_predict(self, state, force, dt, gamma=0.5, beta=0.25,
                        M_eff=None):
        """Advance one step with a *given* force at t+dt (implicit in the force).

        ``force`` is the generalised aerodynamic force (L, M) at the new time
        level; ``M_eff`` optionally replaces the structural mass matrix (used to
        fold the fluid apparent mass in implicitly).  Returns the new
        :class:`SectionState`.
        """
        M, C = (self.M if M_eff is None else M_eff), self.C
        q, v, a = state.q, state.v, state.a
        keff = (M / (beta * dt * dt) + C * (gamma / (beta * dt))
                + self.K)
        rhs = (np.asarray(force, dtype=float)
               + M @ (q / (beta * dt * dt) + v / (beta * dt)
                      + (1.0 / (2.0 * beta) - 1.0) * a)
               + C @ (gamma / (beta * dt) * q + (gamma / beta - 1.0) * v
                      + dt * (gamma / (2.0 * beta) - 1.0) * a))
        q_new = np.linalg.solve(keff, rhs)
        a_new = ((q_new - q) / (beta * dt * dt) - v / (beta * dt)
                 - (1.0 / (2.0 * beta) - 1.0) * a)
        v_new = v + dt * ((1.0 - gamma) * a + gamma * a_new)
        return SectionState(q=q_new, v=v_new, a=a_new, t=state.t + dt)

    def initial_state(self, h=0.0, h_dot=0.0, alpha=0.0, alpha_dot=0.0):
        q = np.array([h, alpha], dtype=float)
        v = np.array([h_dot, alpha_dot], dtype=float)
        a = np.linalg.solve(self.M, -self.K @ q + np.zeros(2))
        return SectionState(q=q, v=v, a=a, t=0.0)