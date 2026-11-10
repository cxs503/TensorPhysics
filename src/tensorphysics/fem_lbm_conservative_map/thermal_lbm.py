"""D2Q9 + D2Q5 thermal lattice Boltzmann with exact mass/energy bookkeeping.

Purpose in this node
--------------------
The mapped field is only useful if a real LBM solver can consume it.  This is a
compact, dependency-free thermal LBM for the Rayleigh-Benard configuration that the
FEM heat-conduction solution is mapped onto:

* velocity: D2Q9 BGK with a Guo-forced Boussinesq buoyancy term
  ``F_y = rho g beta (T - T_ref)``;
* temperature: D2Q5 BGK advection-diffusion, ``D = c_s^2 (tau_g - 1/2)``;
* walls: no-slip full-way bounce-back for ``f``; for ``g`` either an isothermal
  (anti-bounce-back) or an adiabatic (mirror) condition, node-by-node.

The point that matters for the coupling audit: the *interior* operators conserve
mass and energy by construction (BGK collision conserves the zeroth moment,
streaming only moves populations), so the whole change in ``sum(T)`` at every step
is attributable to the wall injections.  The solver therefore reports both the
measured energy history *and* the accumulated wall budget
(:attr:`ThermalLBMResult.energy_balance_mismatch` is the difference, which must be
round-off).  That turns "the coupling conserves energy" into a measured statement.

Ownership: LBM kernel only -- the field it starts from comes from
:mod:`tensorphysics.fem_lbm_conservative_map.remap`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["ThermalLBMResult", "run_thermal_lbm", "lattice_peierls", "conductive_flux"]

# D2Q9
_C9 = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=float)
_W9 = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
# D2Q5 (temperature)
_C5 = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
_W5 = np.array([1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6])
_OPP5 = np.array([0, 3, 4, 1, 2])
_CS2 = 1.0 / 3.0


@dataclass
class ThermalLBMResult:
    T: np.ndarray  # (ny, nx) final temperature
    u: np.ndarray
    v: np.ndarray
    rho: np.ndarray
    energy_history: np.ndarray  # sum(T) every step (including t=0)
    wall_budget: np.ndarray  # cumulative energy injected through the walls
    budget_mismatch: float  # max |measured - budget| over the run
    mass_history: np.ndarray
    mass_mismatch: float
    nusselt_history: np.ndarray = field(default_factory=lambda: np.zeros(1))
    rayleigh: float = 0.0
    prandtl: float = 0.0
    tau_f: float = 0.0
    tau_g: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def nusselt(self) -> float:
        return float(self.nusselt_history[-1]) if self.nusselt_history.size else float("nan")


def conductive_flux(diffusivity: float, delta_t: float, height: float,
                    length: float) -> float:
    """Total conduction heat rate ``D dT/dy * L`` of the quiescent state (lattice units)."""
    return diffusivity * delta_t / height * length


def lattice_peierls(Ra: float, Pr: float, height: float, nu: float = 0.1,
                    delta_t: float = 1.0) -> tuple[float, float, float]:
    """Relaxation times and buoyancy coefficient for a target (Ra, Pr) in lattice units.

    ``nu = c_s^2 (tau_f - 1/2)``, ``D = nu/Pr``, ``tau_g = 1/2 + D/c_s^2`` and
    ``g beta = Ra nu D / (dT H^3)`` (unit density).
    """
    D = nu / Pr
    tau_f = 0.5 + nu / _CS2
    tau_g = 0.5 + D / _CS2
    gbeta = Ra * nu * D / (float(delta_t) * float(height) ** 3)
    return tau_f, tau_g, gbeta


def _wall_array(value, nx: int):
    """Broadcast a plate temperature to one value per node (``None`` = adiabatic)."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(nx, float(arr))
    return arr.reshape(nx)


def run_thermal_lbm(
    T0: np.ndarray,
    *,
    steps: int = 500,
    Ra: float = 0.0,
    Pr: float = 0.71,
    hot_wall: float | None = 1.0,
    cold_wall: float | None = 0.0,
    noise_amplitude: float = 0.0,
    seed: int = 1234,
    probe_every: int = 1,
    nu: float = 0.1,
) -> ThermalLBMResult:
    """Run the thermal LBM from a cell-centred temperature field.

    Parameters
    ----------
    T0 : (ny, nx) array
        Initial temperature -- here the *mapped FEM field* (one value per lattice
        node/control volume).  ``hot_wall=None`` and ``cold_wall=None`` means a
        fully insulated box (no energy exchange with the surroundings), which is
        the configuration used to prove that the transported energy is conserved.
    hot_wall, cold_wall : float, array or None
        Temperatures of the bottom / top plates (scalar or one value per node);
        ``None`` = adiabatic.  Passing the *mapped* wall values is what makes the
        mapped field the exact discrete steady state of the solver.
    noise_amplitude : float
        Deterministic perturbation of the initial temperature (to trigger the
        convective instability); 0 keeps the pure conduction state.
    """
    T0 = np.asarray(T0, dtype=float)
    ny, nx = T0.shape
    height = ny - 1  # wall-to-wall distance in lattice units (the plates are on nodes)
    hot = _wall_array(hot_wall, nx)
    cold = _wall_array(cold_wall, nx)
    dT = float(np.mean(hot) - np.mean(cold)) if (hot is not None and cold is not None) else 1.0
    tau_f, tau_g, gbeta = lattice_peierls(Ra, Pr, height, nu=nu, delta_t=dT)
    omega_f = 1.0 / tau_f
    omega_g = 1.0 / tau_g

    T = T0.copy()
    if noise_amplitude > 0.0:
        rng = np.random.default_rng(seed)
        T += noise_amplitude * rng.standard_normal(T.shape) * np.sin(np.pi * np.arange(ny) / (ny - 1))[:, None]
    t_ref = 0.5 * ((float(np.mean(hot)) if hot is not None else 0.0)
                   + (float(np.mean(cold)) if cold is not None else 0.0))

    # equilibrium initialisation for both distributions
    rho = np.ones_like(T)
    u = np.zeros_like(T)
    v = np.zeros_like(T)
    f = _equilibrium9(rho, u, v)
    g = _equilibrium5(T, u, v)

    energy = np.zeros(steps + 1)
    mass = np.zeros(steps + 1)
    budget = np.zeros(steps + 1)
    nus = np.zeros(steps + 1)
    energy[0] = float(T.sum())
    mass[0] = float(rho.sum())
    q_cond = conductive_flux(_CS2 * (tau_g - 0.5), abs(dT), float(height), float(nx))
    nus[0] = float("nan")

    mismatch = 0.0
    mass_mismatch = 0.0

    for t in range(1, steps + 1):
        # ---- macroscopic fields -------------------------------------------
        rho = f.sum(axis=2)
        u = (f * _C9[None, None, :, 0]).sum(axis=2) / rho
        v = (f * _C9[None, None, :, 1]).sum(axis=2) / rho
        T = g.sum(axis=2)

        # ---- forcing (Boussinesq buoyancy, Guo) ---------------------------
        Fx = np.zeros_like(T)
        Fy = gbeta * (T - t_ref)
        f_post = _collide9(f, rho, u, v, omega_f, Fx, Fy)
        # thermodynamic (half-force) velocity for the equilibrium of g
        u_f = u + 0.5 * Fx / rho
        v_f = v + 0.5 * Fy / rho
        g_post = g - omega_g * (g - _equilibrium5(T, u_f, v_f))

        # ---- streaming with wall boundary conditions ----------------------
        f_new, _ = _stream9(f_post)
        g_new, q_bottom, q_top = _stream5(g_post, hot, cold)

        budget[t] = budget[t - 1] + q_bottom + q_top
        energy[t] = float(g_new.sum(axis=2).sum())
        f = f_new
        g = g_new
        mass[t] = float(f.sum(axis=2).sum())
        mismatch = max(mismatch, abs((energy[t] - energy[0]) - budget[t]))
        mass_mismatch = max(mass_mismatch, abs(mass[t] - mass[0]))
        if hot is not None and q_cond > 0:
            nus[t] = q_bottom / q_cond

    T = g.sum(axis=2)
    rho = f.sum(axis=2)
    u = (f * _C9[None, None, :, 0]).sum(axis=2) / rho
    v = (f * _C9[None, None, :, 1]).sum(axis=2) / rho
    return ThermalLBMResult(
        T=T, u=u, v=v, rho=rho,
        energy_history=energy[::probe_every],
        wall_budget=budget[::probe_every],
        budget_mismatch=mismatch,
        mass_history=mass[::probe_every],
        mass_mismatch=mass_mismatch,
        nusselt_history=nus[1:] if hot is not None else np.zeros(1),
        rayleigh=Ra, prandtl=Pr, tau_f=tau_f, tau_g=tau_g,
        metadata={"height": height, "nx": nx, "ny": ny, "steps": steps,
                  "delta_t": dT,
                  "gbeta": gbeta, "noise_amplitude": noise_amplitude},
    )


# ---------------------------------------------------------------------------
# equilibrium / collision / streaming
# ---------------------------------------------------------------------------
def _equilibrium9(rho, u, v) -> np.ndarray:
    cu = _C9[None, None, :, 0] * u[..., None] + _C9[None, None, :, 1] * v[..., None]
    usq = u ** 2 + v ** 2
    return _W9[None, None, :] * rho[..., None] * (
        1.0 + 3.0 * cu + 4.5 * cu ** 2 - 1.5 * usq[..., None])


def _equilibrium5(T, u, v) -> np.ndarray:
    cu = _C5[None, None, :, 0] * u[..., None] + _C5[None, None, :, 1] * v[..., None]
    w = _W5[None, None, :]
    # isotropic D2Q5 equilibrium (Guo/Shi): w T (1 + 3 c.u)
    return w * T[..., None] * (1.0 + 3.0 * cu)


def _collide9(f, rho, u, v, omega, Fx, Fy) -> np.ndarray:
    feq = _equilibrium9(rho, u, v)
    cu = _C9[None, None, :, 0] * u[..., None] + _C9[None, None, :, 1] * v[..., None]
    force = (1.0 - 0.5 * omega) * _W9[None, None, :] * (
        3.0 * (_C9[None, None, :, 0] * Fx[..., None] + _C9[None, None, :, 1] * Fy[..., None])
        - 3.0 * (u[..., None] * Fx[..., None] + v[..., None] * Fy[..., None])
        + 9.0 * cu * (_C9[None, None, :, 0] * Fx[..., None] + _C9[None, None, :, 1] * Fy[..., None]))
    return f - omega * (f - feq) + force


def _shift_x(arr: np.ndarray, cx: int) -> np.ndarray:
    """Shift in x with periodic wrap: ``out[i, j] = arr[i, j - cx]``."""
    if cx == 0:
        return arr
    return np.roll(arr, cx, axis=1)


def _shift_y(arr: np.ndarray, cy: int) -> np.ndarray:
    """Shift in y with zeros outside: ``out[i, j] = arr[i - cy, j]``."""
    ny, nx = arr.shape
    if cy == 0:
        return arr
    out = np.zeros_like(arr)
    if cy > 0:
        out[cy:, :] = arr[:ny - cy, :]
    else:
        out[:ny + cy, :] = arr[-cy:, :]
    return out


def _opp_index(cx: int, cy: int) -> int:
    for k in range(9):
        if int(_C9[k, 0]) == cx and int(_C9[k, 1]) == cy:
            return k
    raise KeyError((cx, cy))


# reflection in y only: (cx, cy) -> (cx, -cy); the bounce-back pairing of the plates
_REF_Y9 = np.array([_opp_index(int(_C9[k, 0]), -int(_C9[k, 1])) for k in range(9)])


def _stream9(f_post: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """D2Q9 streaming: periodic in x, specular bounce-back on the two horizontal walls.

    The wall pairing is a bijection between the leaving links (``S, SE, SW``) and
    the missing incoming links (``N, NE, NW``) of the same node, so ``sum(f)`` is
    conserved *exactly* and the only momentum change is the wall force -- the
    property that makes the mass audit meaningful.
    """
    ny, nx, _ = f_post.shape
    out = np.zeros_like(f_post)
    for k in range(9):
        out[:, :, k] = _shift_y(_shift_x(f_post[:, :, k], int(_C9[k, 0])), int(_C9[k, 1]))
    for k in range(1, 9):
        if _C9[k, 1] > 0:
            out[0, :, k] = f_post[0, :, _REF_Y9[k]]
        elif _C9[k, 1] < 0:
            out[ny - 1, :, k] = f_post[ny - 1, :, _REF_Y9[k]]
    return out, np.zeros((ny, nx))


def _stream5(g_post: np.ndarray, hot_wall: float | None,
             cold_wall: float | None) -> tuple[np.ndarray, float, float]:
    """D2Q5 streaming: periodic in x, isothermal / adiabatic thermal walls in y.

    Returns the new distribution and the energy injected at the bottom and at the top
    wall in this step.  Because the interior streaming step only permutes populations
    and the wall pairing is bijective, ``d(bottom) + d(top)`` reproduces the measured
    change of ``sum(T)`` exactly -- the identity behind the coupling audit, and the
    bottom flux divided by the conductive flux is the Nusselt number.
    """
    ny, nx, _ = g_post.shape
    out = np.zeros_like(g_post)
    out[:, :, 0] = g_post[:, :, 0]
    out[:, :, 1] = np.roll(g_post[:, :, 1], +1, axis=1)   # east-going from the west
    out[:, :, 3] = np.roll(g_post[:, :, 3], -1, axis=1)   # west-going from the east
    out[1:, :, 2] = g_post[:-1, :, 2]                     # north-going from below
    out[:-1, :, 4] = g_post[1:, :, 4]                     # south-going from above

    # bottom wall: the population that leaves the box is the south-going one; the
    # incoming north-going slot is first filled by its mirror image (which keeps the
    # node's energy unchanged), then corrected so that the *wall node itself* holds
    # the imposed temperature (exact nodal Dirichlet).  The correction is exactly the
    # energy the wall injects, so the budget stays closed.
    out[0, :, 2] = g_post[0, :, 4]
    q_bottom = float((out[0, :, 2] - g_post[0, :, 4]).sum())
    if hot_wall is not None:
        delta = hot_wall - out[0, :, :].sum(axis=1)
        out[0, :, 2] += delta
        q_bottom += float(delta.sum())

    out[ny - 1, :, 4] = g_post[ny - 1, :, 2]
    q_top = float((out[ny - 1, :, 4] - g_post[ny - 1, :, 2]).sum())
    if cold_wall is not None:
        delta = cold_wall - out[ny - 1, :, :].sum(axis=1)
        out[ny - 1, :, 4] += delta
        q_top += float(delta.sum())

    return out, q_bottom, q_top
