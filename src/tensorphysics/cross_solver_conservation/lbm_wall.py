"""Stage 2 -- D2Q9 LBM channel with a moving wall and exact momentum bookkeeping.

The DEM stage hands over a boundary velocity history ``v_ice(t)`` (the tangential
velocity of the ice fragments sliding on the wall).  It is imposed on the LBM
channel as a **moving-wall bounce-back** condition (half-way wall between fluid
nodes and the wall plane), with Guo forcing for the streamwise pressure gradient.

Two independent measurements of the wall momentum flux are produced, plus a
third from the ice side:

``ladd``
    discrete momentum exchange of the bounce-back links,
    ``dp_wall = (2 f_i - 2 w_i rho (e_i . u_w)) e_i`` (Ladd 1994), accumulated
    over the averaging window.  The global discrete momentum balance
    ``P(t) - P(0) = sum force - sum dp_wall`` is checked every step and reported;
    it holds to round-off and therefore validates the boundary implementation.
``shear``
    wall shear stress from a second-order one-sided difference of the measured
    velocity profile, ``tau = rho nu du/dy``, averaged over the window.
``ice``
    the momentum lost by the ice fragment, ``Delta P_ice``, whose equation of
    motion is driven by ``tau`` and whose wall velocity is what LBM imposes.

The cross-solver closure is ``|Delta P_ice + J_ladd| / |Delta P_ice|`` and
``|J_shear - J_ladd| / |J_ladd|`` -- different discretisations (ice ODE, discrete
exchange, macroscopic gradient), so the comparison is not a tautology.

Dimensional note: the specification writes ``int rho v_ice dA = int tau_wall dt``.
The two sides differ by a length (``kg/s`` vs ``kg/(m s)``); the dimensionally
consistent pairing used here is the **momentum flux per unit wall area**
``int rho v_ice u_t dt / A`` against the impulse per unit area ``int tau dt``.
This is flagged in the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["D2Q9Channel", "run_channel", "run_ice_wall_coupling", "wall_shear"]

E = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
    dtype=int,
)
W = np.array(
    [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36], dtype=float
)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=int)
EX = E[:, 0]
EY = E[:, 1]
CS2 = 1.0 / 3.0


@dataclass
class D2Q9Channel:
    """D2Q9 BGK channel, periodic in x, bounce-back walls at ``y = -0.5, ny-0.5``."""

    nx: int = 48
    ny: int = 24
    tau: float = 0.8
    force: float = 1e-5
    f: np.ndarray = field(default_factory=lambda: np.zeros((9, 0, 0)))
    u_wall: float = 0.0
    impulse_top: float = 0.0
    impulse_bottom: float = 0.0
    balance_error: float = 0.0
    n_steps: int = 0

    def __post_init__(self):
        if self.f.size == 0:
            rho = np.ones((self.nx, self.ny))
            ux = np.zeros((self.nx, self.ny))
            uy = np.zeros((self.nx, self.ny))
            self.f = self._equilibrium(rho, ux, uy)

    @property
    def nu(self) -> float:
        return (self.tau - 0.5) / 3.0

    @staticmethod
    def _equilibrium(rho, ux, uy):
        feq = np.empty((9,) + rho.shape)
        for i in range(9):
            cu = EX[i] * ux + EY[i] * uy
            feq[i] = W[i] * rho * (1.0 + cu / CS2 + 0.5 * (cu / CS2) ** 2 - 0.5 * (ux**2 + uy**2) / CS2)
        return feq

    def macros(self):
        rho = self.f.sum(axis=0)
        ux = (self.f * EX[:, None, None]).sum(axis=0) / rho + 0.5 * self.force
        uy = (self.f * EY[:, None, None]).sum(axis=0) / rho
        return rho, ux, uy

    @property
    def momentum(self) -> float:
        return float((self.f * EX[:, None, None]).sum())

    def step(self, n: int = 1):
        for _ in range(n):
            self._step_once()
            self.n_steps += 1

    def _step_once(self):
        rho, ux, uy = self.macros()
        feq = self._equilibrium(rho, ux, uy)
        fpost = self.f - (self.f - feq) / self.tau
        # Guo forcing: F_i = (1 - 1/(2 tau)) w_i [3 (e_i - u).g + 9 (e_i.u)(e_i.g)]
        g = self.force
        for i in range(9):
            eu = EX[i] * ux + EY[i] * uy
            eg = EX[i] * g
            fpost[i] += (1.0 - 0.5 / self.tau) * W[i] * (
                3.0 * (eg - ux * g) + 9.0 * eu * eg
            )

        p_before = self.momentum
        fnew = np.zeros_like(self.f)
        fnew[0] = fpost[0]
        dtop = 0.0
        dbot = 0.0
        for i in range(1, 9):
            arr = np.roll(fpost[i], EX[i], axis=0)
            if EY[i] == 1:
                fnew[i, :, 1:] = arr[:, :-1]
                # top wall, moving with u_wall
                rho_w = rho[:, -1]
                corr = 6.0 * W[i] * rho_w * (EX[i] * self.u_wall)
                fnew[OPP[i], :, -1] += arr[:, -1] - corr
                # momentum handed to the wall (x component)
                dtop += float(np.sum((2.0 * arr[:, -1] - corr) * EX[i]))
            elif EY[i] == -1:
                fnew[i, :, :-1] = arr[:, 1:]
                fnew[OPP[i], :, 0] += arr[:, 0]
                dbot += float(np.sum(2.0 * arr[:, 0] * EX[i]))
            else:
                fnew[i] = arr
        self.f = fnew
        self.impulse_top += dtop
        self.impulse_bottom += dbot

        # discrete momentum balance: dP = force - wall impulse
        p_after = self.momentum
        rhs = self.nx * self.ny * self.force - dtop - dbot
        self.balance_error += abs((p_after - p_before) - rhs)

    def wall_shear(self, n_avg: int = 1):
        """Wall shear stress from one-sided velocity gradients (top and bottom)."""
        _, ux, _ = self.macros()
        prof = ux.mean(axis=0)
        # the half-way wall sits 0.5 below the first node centre and 0.5 above the
        # last one, so the wall gradient uses the exact 3-point parabola through
        # (wall, 0), (first node), (second node):  du/dy|wall = 3 u_1 - u_2/3
        dudy_bot = 3.0 * prof[0] - prof[1] / 3.0
        dudy_top = -(3.0 * (prof[-1] - self.u_wall) - (prof[-2] - self.u_wall) / 3.0)
        rho0 = float(self.f.sum(axis=0).mean())
        # stress the *fluid exerts on the wall*, +x positive (matches the sign of
        # the discrete momentum exchange): top = -mu du/dy, bottom = +mu du/dy
        return {
            "tau_top": -rho0 * self.nu * dudy_top,
            "tau_bottom": rho0 * self.nu * dudy_bot,
            "profile": prof,
            "rho": rho0,
        }


def wall_shear(profile: np.ndarray, u_wall_top: float, nu: float, rho: float = 1.0):
    """Wall stress from the exact 3-point parabola including the half-way wall."""
    dudy_bot = 3.0 * profile[0] - profile[1] / 3.0
    dudy_top = -(3.0 * (profile[-1] - u_wall_top) - (profile[-2] - u_wall_top) / 3.0)
    # stress on the wall (+x positive): top = -mu du/dy, bottom = +mu du/dy
    return -rho * nu * dudy_top, rho * nu * dudy_bot


def run_channel(nx=48, ny=24, tau=0.8, force=1e-5, n_warm=8000, n_avg=2000,
                u_wall=0.0, sample_every=100):
    """Spin up a channel to steady state and average the wall fluxes."""
    ch = D2Q9Channel(nx=nx, ny=ny, tau=tau, force=force, u_wall=u_wall)
    ch.step(n_warm)
    p0 = ch.momentum
    j0_top, j0_bot = ch.impulse_top, ch.impulse_bottom
    shear_top, shear_bot = [], []
    for _ in range(n_avg):
        ch.step(1)
        if ch.n_steps % sample_every == 0:
            s = ch.wall_shear()
            shear_top.append(s["tau_top"])
            shear_bot.append(s["tau_bottom"])
    dp = ch.momentum - p0
    jtop = ch.impulse_top - j0_top
    jbot = ch.impulse_bottom - j0_bot
    prof = ch.wall_shear()["profile"]
    y = (np.arange(ny) + 0.5) / ny
    u_analytic = force * (y * ny) * (ny - y * ny) / (2.0 * ch.nu)
    # steady Poiseuille with zero wall velocity: u(y)=g y(H-y)/(2 nu), H=ny
    yy = np.arange(ny) + 0.5
    u_analytic = force * yy * (ny - yy) / (2.0 * ch.nu)
    u_exact_max = force * ny**2 / (8.0 * ch.nu)
    prof_err = float(
        np.linalg.norm(prof - u_analytic) / np.linalg.norm(u_analytic)
    )
    return {
        "nx": nx,
        "ny": ny,
        "tau": tau,
        "nu": ch.nu,
        "force": force,
        "n_warm": n_warm,
        "n_avg": n_avg,
        "u_wall": u_wall,
        "profile": prof.tolist(),
        "u_analytic": u_analytic.tolist(),
        "u_max_measured": float(prof.max()),
        "u_max_analytic": u_exact_max,
        "profile_rel_error": prof_err,
        "momentum_change": dp,
        "impulse_top": jtop,
        "impulse_bottom": jbot,
        "tau_top_ladd": jtop / (nx * n_avg),
        "tau_bottom_ladd": jbot / (nx * n_avg),
        "tau_top_shear": float(np.mean(shear_top)),
        "tau_bottom_shear": float(np.mean(shear_bot)),
        "tau_analytic": force * ny / 2.0,
        "balance_error": ch.balance_error,
        "balance_error_rel": ch.balance_error / max(abs(jtop) + abs(jbot), 1e-300),
    }


def run_ice_wall_coupling(
    nx=48,
    ny=24,
    tau=0.8,
    force=1e-5,
    n_warm=8000,
    n_couple=6000,
    m_ice=500.0,
    v_ice0=0.02,
    contact_width=None,
):
    """Two-way weak coupling: LBM wall shear drives the ice fragment, which sets
    the LBM moving-wall velocity."""
    contact_width = nx if contact_width is None else contact_width
    v_ice = float(v_ice0)
    # pre-solve the flow with the wall already at the initial ice velocity, so the
    # impulsive start (u_wall: 0 -> v_ice0) is not charged to the audit window
    ch = D2Q9Channel(nx=nx, ny=ny, tau=tau, force=force, u_wall=v_ice)
    ch.step(n_warm)
    p_ice0 = m_ice * v_ice
    j_top0 = ch.impulse_top
    j_bot0 = ch.impulse_bottom
    hist = {k: [] for k in ("t", "v_ice", "tau_shear", "p_ice", "j_wall", "tau_ladd")}
    per_step = {"v_ice": [], "tau_shear": [], "impulse": []}
    tau_acc = 0.0
    for step in range(n_couple):
        s = ch.wall_shear()
        tau_w = s["tau_top"]
        # ice slab: m dv/dt = -tau * A (tangential drag on the sliding fragment)
        v_ice += tau_w * contact_width / m_ice
        ch.u_wall = v_ice
        ch.step(1)
        tau_acc += tau_w
        per_step["v_ice"].append(v_ice)
        per_step["tau_shear"].append(tau_w)
        per_step["impulse"].append(ch.impulse_top - j_top0)
        if step % 100 == 0:
            hist["t"].append(step + 1)
            hist["v_ice"].append(v_ice)
            hist["tau_shear"].append(tau_w)
            hist["p_ice"].append(m_ice * v_ice)
            hist["j_wall"].append(ch.impulse_top - j_top0)
            hist["tau_ladd"].append((ch.impulse_top - j_top0) / ((step + 1) * contact_width))

    j_top = ch.impulse_top - j_top0
    j_bot = ch.impulse_bottom - j_bot0
    dp_ice = m_ice * v_ice - p_ice0
    return {
        "nx": nx,
        "ny": ny,
        "tau": tau,
        "nu": ch.nu,
        "force": force,
        "m_ice": m_ice,
        "v_ice0": v_ice0,
        "v_ice_final": v_ice,
        "n_couple": n_couple,
        "dp_ice": dp_ice,
        "impulse_top_ladd": j_top,
        "impulse_bottom_ladd": j_bot,
        "impulse_shear": tau_acc * contact_width,
        "closure_ice_vs_ladd": abs(dp_ice - j_top) / abs(dp_ice),
        "closure_shear_vs_ladd": abs(tau_acc * contact_width - j_top) / abs(j_top),
        "balance_error_rel": ch.balance_error / max(abs(j_top) + abs(j_bot), 1e-300),
        "history": hist,
        "per_step": per_step,
        "contact_width": contact_width,
    }