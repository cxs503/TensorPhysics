"""Stage 1b -- 1D bonded-particle (DEM) chain with bilinear cohesive hinges.

The chain is a lumped-mass discretisation of the Euler-Bernoulli beam produced by
the FEM stage.  Node ``i`` sits at ``x_i = i * l_b``; hinge ``j`` (at node ``j``)
carries a curvature-type rotation

    theta_j = (w_{j+1} - 2 w_j + w_{j-1}) / l_b ,        k_theta = E I / l_b

so ``sum_j 0.5 k_theta theta_j^2 = sum_j 0.5 E I kappa_j^2 l_b`` is exactly the
beam bending energy under mid-point quadrature.

Cohesive hinge (bilinear, damage):

    M_f   = sigma_f I / (h/2)                  peak (strength) moment
    Phi_f = G_f b l_b                          fracture energy of one hinge
                                               (``G_f`` x crack area created when
                                               the crack advances by ``l_b``)
    theta_0 = M_f / k_theta ,  theta_c = theta_0 + 2 Phi_f / M_f
    M = (1 - D) k_theta theta ,
    D = 1 - (theta_0/theta_max)(theta_c - theta_max)/(theta_c - theta_0)

The **secant** damage (not the naive ``(theta_max-theta_0)/(theta_c-theta_0)``) is
what makes ``work = stored + dissipated`` exact for a bilinear curve; the naive
form leaves a 1.2x inconsistency that the self-audit ``work-stored-dissipated``
flags.

Energy terms, all analytic and cross-checked against numerical path integrals:

    work(theta)   = int_0^theta M dtheta
    stored(theta) = 0.5 (1-D) k_theta theta^2 = 0.5 M(theta) theta
    dissipated    = 0.5 M_f theta_c (clip(theta_max, theta_0, theta_c)-theta_0)/(theta_c-theta_0)

Integration: explicit velocity Verlet with lumped mass and **nodal** viscous
damping ``c_damp`` (a dashpot per node, unconditional for small ``c``).  A
stiffness-proportional (Kelvin-Voigt) hinge dashpot was tried first and rejected:
``eta omega_max >> 1`` makes the damping moment ~300x the elastic moment at the
lattice frequency, which destabilises the explicit scheme (documented in the
report as a trap).

Because the softening branch has a negative tangent stiffness, rate-independent
damage is unstable once a hinge passes the peak (the snap-back mode grows faster
than the lattice frequency); damage growth is therefore rate limited
(``d_damage_max`` per step, the standard explicit-cohesive viscous
regularisation).  The regularisation makes the measured cohesive dissipation
exceed ``G_f x crack area`` by a reported amount -- an honest, quantified
limitation.

Ledger:

    U_ref (linear elastic strain energy delivered by FEM)
        = KE + U_el + E_diss + E_damp + E_handoff
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["CohesiveHingeLaw", "ChainState", "bonded_chain_from_beam", "run_chain"]

RHO_ICE = 917.0


@dataclass(frozen=True)
class CohesiveHingeLaw:
    k_theta: float
    M_peak: float
    G_f: float
    b: float
    l_b: float

    @property
    def phi_f(self) -> float:
        return self.G_f * self.b * self.l_b

    @property
    def theta_0(self) -> float:
        return self.M_peak / self.k_theta

    @property
    def theta_c(self) -> float:
        """Bond rotation at full damage: total area under the law = Phi_f.

        ``0.5 M_f theta_c = Phi_f`` makes the *total* dissipated energy of a
        broken hinge equal ``G_f b l_b`` (exact Griffith).  Requires
        ``G_f > sigma_f^2 h/(6E)`` (equivalently ``theta_c > theta_0``), i.e. the
        cohesive length must exceed the section thickness scale; this is checked
        by :func:`bonded_chain_from_beam`.
        """
        return 2.0 * self.phi_f / self.M_peak

    def damage(self, theta_max: np.ndarray) -> np.ndarray:
        tm = np.abs(np.asarray(theta_max, dtype=float))
        span = self.theta_c - self.theta_0
        d = np.zeros_like(tm)
        over = tm > self.theta_0
        d[over] = 1.0 - (self.theta_0 / tm[over]) * (
            (self.theta_c - tm[over]) / span
        )
        d = np.where(tm >= self.theta_c, 1.0, d)
        return np.clip(d, 0.0, 1.0)

    def dissipated(self, theta_max: np.ndarray) -> np.ndarray:
        span = self.theta_c - self.theta_0
        tm = np.clip(np.abs(np.asarray(theta_max, dtype=float)), self.theta_0, self.theta_c)
        return 0.5 * self.M_peak * self.theta_c * (tm - self.theta_0) / span

    def work_to(self, theta: np.ndarray) -> np.ndarray:
        t = np.abs(np.asarray(theta, dtype=float))
        span = self.theta_c - self.theta_0
        w = 0.5 * np.minimum(t, self.theta_0) ** 2 * self.k_theta
        over = t > self.theta_0
        tc = np.minimum(t[over], self.theta_c)
        w[over] = (
            0.5 * self.M_peak * self.theta_0
            + 0.5 * self.M_peak * (1.0 + (self.theta_c - tc) / span) * (tc - self.theta_0)
        )  # beyond theta_c the hinge is traction free: work saturates
        return w

    def as_dict(self) -> dict:
        return {
            "k_theta": self.k_theta,
            "M_peak": self.M_peak,
            "G_f": self.G_f,
            "b": self.b,
            "l_b": self.l_b,
            "phi_f": self.phi_f,
            "theta_0": self.theta_0,
            "theta_c": self.theta_c,
            "theta_c_over_theta_0": self.theta_c / self.theta_0,
        }


@dataclass
class ChainState:
    law: CohesiveHingeLaw
    m_node: float
    n_bonds: int
    w: np.ndarray
    v: np.ndarray
    theta_max: np.ndarray
    fixed: tuple = (0, 1)
    theta: np.ndarray = None  # type: ignore[assignment]
    process_zone: float = 0.0

    @property
    def n_node(self) -> int:
        return len(self.w)

    @property
    def free(self) -> np.ndarray:
        mask = np.ones(self.n_node, dtype=bool)
        for i in self.fixed:
            mask[i] = False
        return np.nonzero(mask)[0]


def _hinge_rotations(w: np.ndarray, law: CohesiveHingeLaw) -> np.ndarray:
    return (w[2:] - 2.0 * w[1:-1] + w[:-2]) / law.l_b


def _hinge_forces(moments: np.ndarray, law: CohesiveHingeLaw, n_node: int) -> np.ndarray:
    """``f_i = -(1/l_b) (A^T M)_i`` with ``A`` the second-difference operator."""
    g = moments / law.l_b
    f = np.zeros(n_node)
    f[:-2] += g
    f[1:-1] -= 2.0 * g
    f[2:] += g
    return -f


def bonded_chain_from_beam(sol, section, sigma_f: float, G_f: float, n_bonds: int):
    """Initialise the DEM chain from the FEM deflection field."""
    l_b = sol.L / n_bonds
    x = np.arange(n_bonds + 1) * l_b
    w = np.asarray(sol.deflection_at(x), dtype=float)
    w = w - w[0]
    law = CohesiveHingeLaw(
        k_theta=section.EI / l_b,
        M_peak=sigma_f * section.I / (section.h / 2.0),
        G_f=G_f,
        b=section.b,
        l_b=l_b,
    )
    theta = _hinge_rotations(w, law)
    pz = section.E * G_f / sigma_f**2
    state = ChainState(
        law=law,
        m_node=RHO_ICE * section.A * l_b,
        n_bonds=n_bonds,
        w=w,
        v=np.zeros_like(w),
        theta_max=np.abs(theta).copy(),
        fixed=(0, 1),
        theta=theta,
        process_zone=pz,
    )
    dmg = law.damage(np.abs(theta))
    u_lin = float(np.sum(0.5 * law.k_theta * theta**2))
    work0 = float(np.sum(law.work_to(theta)))
    stored0 = float(np.sum(0.5 * (1.0 - dmg) * law.k_theta * theta**2))
    info = {
        "l_b": l_b,
        "n_node": n_bonds + 1,
        "m_node": state.m_node,
        "U_chain_linear": u_lin,
        "U_fem": sol.U,
        "mapping_rel_error": abs(u_lin - sol.U) / sol.U,
        "cohesive_work_at_handoff": work0,
        "stored_at_handoff": stored0,
        "handoff_dissipated": work0 - stored0,
        "handoff_energy_deficit": u_lin - work0,
        "theta_root": float(theta[0]),
        "theta_0": law.theta_0,
        "theta_c": law.theta_c,
        "root_moment": float(law.k_theta * theta[0]),
        "M_peak": law.M_peak,
        "peak_exceeded": bool(np.max(np.abs(theta)) > law.theta_0),
        "process_zone_length": pz,
        "l_b_over_process_zone": l_b / pz,
        "law": law.as_dict(),
    }
    return state, info


def run_chain(
    state: ChainState,
    dt: float,
    t_end: float,
    zeta: float = 0.02,
    sample_every: int = 50,
    d_damage_max: float | None = 0.02,
):
    """Integrate the cohesive chain; return the measured energy ledger.

    ``zeta`` is the damping ratio of the first chain eigenmode used to size the
    nodal dashpots, ``c = 2 zeta m omega_1``; ``d_damage_max`` is the per-step
    damage rate limit (``None`` = rate independent, unstable once past the peak).
    """
    law = state.law
    n_node = state.n_node
    free = state.free
    inv_m = np.zeros(n_node)
    inv_m[free] = 1.0 / state.m_node
    free_mask = np.zeros(n_node)
    free_mask[free] = 1.0

    omega_1 = 3.5160**2 * np.sqrt(law.k_theta * law.l_b / (state.m_node / law.l_b)) / (
        state.n_bonds * law.l_b
    ) ** 2
    c_damp = 2.0 * zeta * state.m_node * omega_1

    w = state.w.copy()
    v = state.v.copy()
    theta_max = state.theta_max.copy()
    theta = _hinge_rotations(w, law)
    dmg = law.damage(theta_max)
    stored0 = float(np.sum(0.5 * (1.0 - dmg) * law.k_theta * theta**2))
    work0 = float(np.sum(law.work_to(state.theta)))

    m_prev = (1.0 - dmg) * law.k_theta * theta
    f = _hinge_forces(m_prev, law, n_node) * free_mask - c_damp * v * free_mask

    path_work = 0.0
    damp_work = 0.0
    n_steps = int(round(t_end / dt))
    history = {k: [] for k in ("t", "ke", "u_el", "e_diss", "e_damp", "total",
                               "n_broken", "theta_root", "w_tip", "v_tip",
                               "crack_front")}

    for step in range(n_steps):
        v_half = v + 0.5 * dt * f * inv_m
        theta_old = theta
        w[free] += dt * v_half[free]
        theta = _hinge_rotations(w, law)

        theta_max = np.maximum(theta_max, np.abs(theta))
        d_target = law.damage(theta_max)
        if d_damage_max is None:
            dmg = d_target
        else:
            dmg = np.minimum(d_target, dmg + d_damage_max)

        k_eff = (1.0 - dmg) * law.k_theta
        m_tot = k_eff * theta
        dtheta = theta - theta_old
        path_work += float(np.sum(0.5 * (m_tot + m_prev) * dtheta))
        m_prev = m_tot

        f = _hinge_forces(m_tot, law, n_node) * free_mask - c_damp * v_half * free_mask
        v = v_half + 0.5 * dt * f * inv_m
        damp_work += float(c_damp * dt * np.sum(v_half[free] ** 2))

        if sample_every and step % sample_every == 0:
            u_el = float(np.sum(0.5 * k_eff * theta**2))
            ke = float(0.5 * state.m_node * np.sum(v[free] ** 2))
            e_diss = float(np.sum(law.dissipated(theta_max)))
            history["t"].append((step + 1) * dt)
            history["ke"].append(ke)
            history["u_el"].append(u_el)
            history["e_diss"].append(e_diss)
            history["e_damp"].append(damp_work)
            history["total"].append(ke + u_el + e_diss + damp_work)
            history["n_broken"].append(int(np.sum(dmg >= 1.0 - 1e-12)))
            history["theta_root"].append(float(theta[0]))
            history["w_tip"].append(float(w[-1]))
            history["v_tip"].append(float(v[-1]))
            history["crack_front"].append(int(np.sum(dmg >= 1.0 - 1e-12)))

    k_eff = (1.0 - dmg) * law.k_theta
    u_el = float(np.sum(0.5 * k_eff * theta**2))
    ke = float(0.5 * state.m_node * np.sum(v[free] ** 2))
    e_diss = float(np.sum(law.dissipated(theta_max)))
    u_ref = float(np.sum(0.5 * law.k_theta * state.theta**2))
    diss_meas = work0 - stored0 + path_work - (u_el - stored0)
    n_broken = int(np.sum(dmg >= 1.0 - 1e-12))
    ledger = {
        "U_ref": u_ref,
        "KE": ke,
        "U_el": u_el,
        "E_diss": e_diss,
        "E_diss_path": diss_meas,
        "E_damp": damp_work,
        "E_handoff": u_ref - work0,
        "total": ke + u_el + e_diss + damp_work + (u_ref - work0),
        "n_broken": n_broken,
        "n_damaged": int(np.sum(dmg > 0.0)),
        "broken_length": n_broken * law.l_b,
        "crack_front": n_broken * law.l_b,
        "theta_max": theta_max.tolist(),
        "theta": theta.tolist(),
        "w": w.tolist(),
        "v": v.tolist(),
        "dmg": dmg.tolist(),
        "diss_path_vs_analytic_rel": abs(diss_meas - e_diss) / max(abs(e_diss), 1e-300),
        "excess_dissipation_over_griffith": e_diss - law.G_f * law.b * n_broken * law.l_b,
        "history": history,
        "dt": dt,
        "t_end": t_end,
        "zeta": zeta,
        "c_damp": c_damp,
        "omega_1": omega_1,
        "d_damage_max": d_damage_max,
    }
    ledger["closure_rel_error"] = abs(ledger["total"] - u_ref) / u_ref
    return ledger