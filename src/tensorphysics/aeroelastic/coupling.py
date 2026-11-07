"""Partitioned fluid-structure coupling: explicit staggered vs implicit sub-iteration.

Two schemes are provided so that the difference can be *measured*:

``loose``  explicit staggered (CSS): the aerodynamic load used to advance the
           structure is evaluated at the previous time level, and the fluid
           apparent mass enters the structural inertia *explicitly*.  One fluid
           solve and one structural solve per step -- cheap, but the added mass
           is then subtracted from the structural inertia with the wrong sign,
           the classical added-mass instability.

``tight``  implicit coupling: the structural apparent mass is folded into the
           effective mass matrix (exact treatment of the non-circulatory part)
           and the remaining (circulatory / lag-state) coupling is resolved by
           sub-iteration on ``q(t+dt) = G(q(t+dt))``, either by Aitken-relaxed
           fixed point or by a finite-difference Newton iteration on the
           2-dimensional residual.  Non-convergence is reported, never hidden.

Energy bookkeeping is recorded every step so the work done by the fluid can be
closed against the change of structural energy (gate ``C2``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CoupledRun:
    t: np.ndarray
    q: np.ndarray
    v: np.ndarray
    force: np.ndarray
    energy: np.ndarray
    work: np.ndarray
    iterations: np.ndarray
    converged: bool = True
    diverged: bool = False
    scheme: str = "tight"
    U: float = 0.0

    def growth_rate(self, t_start=None, quantity="alpha", n_windows=8):
        from .flutter import envelope_growth
        idx = 0 if quantity == "h" else 1
        return envelope_growth(self.t, self.q[:, idx], t_start=t_start,
                               n_windows=n_windows)

    def amplitude(self, quantity="alpha"):
        idx = 0 if quantity == "h" else 1
        return float(np.max(np.abs(self.q[:, idx])))

    def energy_residual(self, t_window=None):
        """Relative closure residual between fluid work and structural energy."""
        e = self.energy
        w = self.work
        if t_window is not None:
            m = self.t >= t_window
            if m.sum() < 3:
                return float("nan")
            e = e[m] - e[m][0]
            w = w[m] - w[m][0]
        den = max(np.max(np.abs(w)), np.max(np.abs(e)), 1e-300)
        return float(np.max(np.abs(e - w)) / den)


def _force(aero_state, aero, accel=None, with_added_mass=True):
    f = np.array([aero_state.lift, aero_state.moment])
    if with_added_mass and accel is not None and hasattr(aero, "Ma"):
        f = f - aero.Ma @ np.asarray(accel, dtype=float)
    return f


def _velocity_from_q(section, state, q_new, dt, gamma=0.5, beta=0.25):
    """Newmark velocity consistent with a trial displacement ``q_new``."""
    return (gamma / (beta * dt) * (q_new - state.q)
            + (1.0 - gamma / beta) * state.v
            + dt * (1.0 - gamma / (2.0 * beta)) * state.a)


def run_coupled(aero, section, U, dt, t_end, scheme="tight", tol=1e-10,
                max_iter=25, q0=(0.0, 1e-4), v0=(0.0, 0.0),
                diverge_factor=1e3, max_record=4000, relax="newton",
                with_added_mass=True):
    """March the coupled system and return a :class:`CoupledRun`."""
    aero.reset(U)
    state = section.initial_state(h=q0[0], h_dot=v0[0],
                                  alpha=q0[1], alpha_dot=v0[1])
    primed = aero.solve(state.v[0], state.q[1], state.v[1], dt, steady=True)
    aero.commit(primed, dt)

    n_steps = int(round(t_end / dt))
    stride = max(1, n_steps // max(1, max_record - 1))
    implicit_mass = scheme == "tight" and with_added_mass and hasattr(aero, "Ma")
    M_eff = section.M + aero.Ma if implicit_mass else section.M

    ts, qs, vs, fs, es, ws, its = [0.0], [state.q.copy()], [state.v.copy()], \
        [np.zeros(2)], [0.0], [0.0], [0]
    force_old = _force(primed, aero, state.a, with_added_mass=with_added_mass)
    work = 0.0
    e0 = section.energy(state)
    scale0 = max(abs(state.q[1]), 1e-30)
    converged = True
    diverged = False

    for step in range(1, n_steps + 1):
        if scheme == "loose":
            aero_state = aero.solve(state.v[0], state.q[1], state.v[1], dt)
            force = _force(aero_state, aero, state.a,
                           with_added_mass=with_added_mass)
            new_state = section.newmark_predict(state, force, dt, M_eff=M_eff)
            aero.commit(aero_state, dt)
            n_it = 1
        else:
            q_trial = state.q + dt * state.v
            v_trial = _velocity_from_q(section, state, q_trial, dt)
            f_trial = _force(aero.solve(v_trial[0], q_trial[1], v_trial[1], dt),
                             aero, None, with_added_mass=False)
            n_it = 0
            ok = False
            w_ait = 1.0
            res_hist = []
            jac_fd = None
            for it in range(max_iter):
                n_it = it + 1
                new_state = section.newmark_predict(state, f_trial, dt,
                                                    M_eff=M_eff)
                resid = new_state.q - q_trial
                rnorm = float(np.linalg.norm(resid))
                res_hist.append(rnorm)
                if rnorm <= tol * max(np.linalg.norm(new_state.q), 1e-14):
                    ok = True
                    break
                if relax == "newton":
                    if jac_fd is None or it % 3 == 2:
                        hstep = 1e-7 * max(abs(q_trial[0]), abs(q_trial[1]),
                                           1e-9)
                        jac_fd = np.zeros((2, 2))
                        for j in range(2):
                            dq = np.zeros(2)
                            dq[j] = hstep
                            qt = q_trial + dq
                            vt = _velocity_from_q(section, state, qt, dt)
                            ft = _force(aero.solve(vt[0], qt[1], vt[1], dt),
                                        aero, None, with_added_mass=False)
                            nt = section.newmark_predict(state, ft, dt,
                                                         M_eff=M_eff)
                            jac_fd[:, j] = (nt.q - new_state.q) / hstep
                    try:
                        # Newton on R(q) = G(q) - q : dR/dq = dG/dq - I
                        step_dir = np.linalg.solve(jac_fd - np.eye(2), -resid)
                    except np.linalg.LinAlgError:
                        step_dir = -0.5 * resid
                    q_trial = q_trial + step_dir
                else:
                    if len(res_hist) >= 2 and res_hist[-2] > 0.0:
                        w_ait = -w_ait * res_hist[-2] / (res_hist[-1]
                                                         - res_hist[-2])
                        w_ait = float(np.clip(w_ait, 0.05, 1.0))
                    q_trial = q_trial + w_ait * resid
                v_trial = _velocity_from_q(section, state, q_trial, dt)
                f_trial = _force(aero.solve(v_trial[0], q_trial[1], v_trial[1],
                                            dt), aero, None,
                                 with_added_mass=False)
            if not ok:
                converged = False
            final = aero.solve(new_state.v[0], new_state.q[1], new_state.v[1],
                               dt)
            aero.commit(final, dt)
            # book-keeping uses the *physical* total load (apparent mass
            # included) even when it was folded into the effective mass matrix
            force = _force(final, aero, new_state.a,
                           with_added_mass=with_added_mass)

        dq = new_state.q - state.q
        work += float(0.5 * (force_old + force) @ dq)
        state = new_state
        force_old = force
        energy = section.energy(state) - e0

        if step % stride == 0 or step == n_steps:
            ts.append(step * dt)
            qs.append(state.q.copy())
            vs.append(state.v.copy())
            fs.append(force.copy())
            es.append(energy)
            ws.append(work)
            its.append(n_it)

        if not np.isfinite(state.q).all() or abs(state.q[1]) > \
                diverge_factor * scale0:
            diverged = True
            converged = False
            break

    return CoupledRun(t=np.array(ts), q=np.array(qs), v=np.array(vs),
                      force=np.array(fs), energy=np.array(es),
                      work=np.array(ws), iterations=np.array(its),
                      converged=converged, diverged=diverged, scheme=scheme,
                      U=float(U))