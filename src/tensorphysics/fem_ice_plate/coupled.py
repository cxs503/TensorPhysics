"""1-D bar with a cohesive interface and a J2-plastic bulk (energy audit model).

This is the smallest model in which *all three* dissipation mechanisms of the
task can be bookkept exactly:

* elastic strain energy in the bar,
* plastic dissipation in the bar (1-D reduction of the 3-D J2 return map with
  isotropic hardening),
* fracture energy released in the cohesive interface.

Model
-----
A bar of length ``L`` and unit cross-section contains a cohesive interface at
mid-span carrying the same force as the bulk::

    equilibrium:    sigma_bulk(eps) = t(delta)
    compatibility:  Delta = L * eps + delta

The interface opening ``delta`` is the *control parameter* (crack-mouth
opening control), which makes the softening branch -- including snap-back --
traceable: for every ``delta`` the bulk strain follows from the monotone
elastoplastic relation ``sigma_bulk(eps) = t(delta)`` (bisection), and the end
displacement follows from compatibility.

The external work is integrated along the equilibrium path,
``W_ext = int F dDelta``, and must equal
``U_elastic + W_plastic + W_fracture`` to round-off -- the energy-conservation
check of the task.  The model also provides the three negative controls of the
cohesive law (``G_f -> 0``, ``sigma_y -> inf``, interface stiffness ``-> inf``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .j2_plasticity import J2Material

__all__ = ["BarResult", "solve_bar_with_interface", "bulk_1d_return_map"]


def bulk_1d_return_map(
    mat: J2Material, eps: float, ep: float
) -> Tuple[float, float, float]:
    """1-D reduction of the J2 radial return map.

    Returns ``(sigma, ep_new, d_work)`` with ``d_work = int sigma_y dep`` the
    plastic dissipation density increment [J/m^3].
    """
    sigma_tr = mat.E * (eps - ep)
    sy = mat.yield_stress(ep)
    f = abs(sigma_tr) - sy
    if f <= 1e-12 * max(sy, 1.0):
        return sigma_tr, ep, 0.0
    Hp = mat.hardening_modulus(ep)
    dep = f / (mat.E + Hp)
    sigma = sigma_tr - np.sign(sigma_tr) * mat.E * dep
    ep_new = ep + dep
    d_work = 0.5 * (sy + mat.yield_stress(ep_new)) * dep
    return sigma, ep_new, d_work


@dataclass
class BarResult:
    """Result of :func:`solve_bar_with_interface`."""

    delta: np.ndarray  # end displacement history (not necessarily monotone)
    delta_interface: np.ndarray  # cohesive opening history (control)
    force: np.ndarray
    eps_bulk: np.ndarray
    ep_bulk: np.ndarray
    traction: np.ndarray
    W_ext: float
    U_elastic: float
    W_plastic: float
    W_fracture: float
    converged: bool

    @property
    def peak_force(self) -> float:
        return float(np.max(self.force))

    @property
    def peak_delta_interface(self) -> float:
        return float(np.max(self.delta_interface))

    def energy_balance(self) -> Dict[str, float]:
        total = self.U_elastic + self.W_plastic + self.W_fracture
        return {
            "W_ext": self.W_ext,
            "U_elastic": self.U_elastic,
            "W_plastic": self.W_plastic,
            "W_fracture": self.W_fracture,
            "sum": total,
            "residual_rel": abs(self.W_ext - total) / max(abs(self.W_ext), 1e-30),
        }


def solve_bar_with_interface(
    mat: J2Material,
    tsl,
    L: float = 0.10,
    delta_f_max: Optional[float] = None,
    n_steps: int = 200,
    tol: float = 1e-14,
    interface_area: float = 1.0,
    eps_cap: float = 0.2,
) -> BarResult:
    """Cohesive-interface bar under crack-mouth-opening control.

    Parameters
    ----------
    mat : J2Material
        Bulk material (1-D reduction).
    tsl : BilinearTSL or ExponentialTSL
        Cohesive traction-separation law.
    L : float
        Bar length [m].
    delta_f_max : float, optional
        Final interface opening [m]; defaults to ``1.2 * tsl.delta_f``.
    n_steps : int
        Number of control steps.
    """
    if delta_f_max is None:
        delta_f_max = 1.2 * float(tsl.delta_f)
    # quadratically clustered control grid: resolves the stiff elastic branch
    # (openings much smaller than delta_f) as well as the softening tail
    deltas = delta_f_max * np.linspace(0.0, 1.0, n_steps + 1) ** 2
    n = len(deltas)
    delta_end = np.zeros(n)
    force = np.zeros(n)
    eps_b = np.zeros(n)
    ep_b = np.zeros(n)
    traction = np.zeros(n)
    ep_state = 0.0
    delta_hist = 0.0
    converged = True

    for k in range(1, n):
        d_if = deltas[k]
        t = float(tsl.traction(d_if, delta_hist))
        # monotone solve sigma_bulk(eps) = t
        lo, hi = 0.0, eps_cap
        if bulk_1d_return_map(mat, hi, ep_state)[0] < t:
            sig_hi = bulk_1d_return_map(mat, hi, ep_state)[0]
            if sig_hi < t - 1e-9 * max(t, 1.0):
                # the bulk cannot equilibrate the interface traction: the bar
                # flows plastically (limit state) -> opening grows freely
                converged = False
                eps_sol = hi
            else:
                eps_sol = hi
        else:
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if bulk_1d_return_map(mat, mid, ep_state)[0] < t:
                    lo = mid
                else:
                    hi = mid
                if hi - lo < 1e-16:
                    break
            eps_sol = 0.5 * (lo + hi)

        sigma, ep_new, dw = bulk_1d_return_map(mat, eps_sol, ep_state)
        ep_state = ep_new
        delta_hist = max(delta_hist, d_if)
        delta_end[k] = L * eps_sol + d_if
        force[k] = sigma
        eps_b[k] = eps_sol
        ep_b[k] = ep_state
        traction[k] = t

    # external work along the equilibrium path
    W_ext = float(np.sum(0.5 * (force[1:] + force[:-1]) * np.diff(delta_end)))
    # plastic dissipation: integrate density along the path (same eps at every
    # section, so multiply by the bar length)
    W_plastic = 0.0
    ep_prev = 0.0
    sig_prev = 0.0
    for k in range(1, n):
        _, ep_new, _ = bulk_1d_return_map(mat, eps_b[k], ep_prev)
        # exact dissipation increment along the monotone loading path
        W_plastic += _plastic_work_increment(mat, ep_prev, ep_new) * L
        ep_prev = ep_new
        sig_prev = force[k]
    U_elastic = 0.5 * float(force[-1] * (eps_b[-1] - ep_b[-1])) * L
    W_frac = float(tsl.dissipate(delta_hist)) * interface_area
    return BarResult(
        delta=delta_end,
        delta_interface=deltas,
        force=force,
        eps_bulk=eps_b,
        ep_bulk=ep_b,
        traction=traction,
        W_ext=W_ext,
        U_elastic=U_elastic,
        W_plastic=W_plastic,
        W_fracture=W_frac,
        converged=converged,
    )


def _plastic_work_increment(mat: J2Material, ep_a: float, ep_b: float) -> float:
    """``int_{ep_a}^{ep_b} sigma_y(ep) dep`` (trapezoid on a fine sub-grid)."""
    if ep_b <= ep_a:
        return 0.0
    grid = np.linspace(ep_a, ep_b, 9)
    vals = np.array([mat.yield_stress(x) for x in grid])
    return float(np.trapezoid(vals, grid))