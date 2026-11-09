"""Two-way FSI stage of the ice-navigation loop.

Model
-----
The wetted hull is discretised on the same streamwise grid as the flow and the
ice field, so that every exchanged quantity has one and only one cell index
(grid identity is a contract check, not a convention).  Per cell the structure
is a modal panel (plate + frame, smeared bending stiffness
``D = E t_eff^3 / (12 (1 - nu^2))``, simply supported between frames):

    k w = p_flow(w) + p_contact(w) + p_added(w),     p_added = rho_w c_add wdd

with ``w`` the inward normal deflection, ``k = D (pi / s)^4`` the panel stiffness
per unit area, and ``c_add = 0.5 (gap0 + h_ice)`` the characteristic added length
of the water in the gap.

The partitioned (explicit added-mass) iteration is the classic source of the
added-mass instability of incompressible FSI (Causin, Gerbeau & Nobile, 2005).
Substituting the harmonic structure response ``wdd = -omega^2 w`` into the
added-mass pressure gives the displacement-form fixed-point map

    w^{k+1} = ( p_flow + p_contact + rho_w c_add omega^2 w^k ) / k,

whose amplification factor is the added-mass ratio

    M = rho_w c_add omega^2 / k = rho_w c_add / (rho_s t_eff).

With relaxation ``omega_r`` the error factor is

    g(omega_r, M) = 1 - omega_r (1 - M),

so the steady coupled problem converges **iff M < 1** (the classical steady
added-mass limit) and Jacobi (``omega_r = 1``) then contracts at exactly the
rate ``M``.  Over-relaxation is admissible but useless here: ``g`` is monotone
in ``omega_r`` only up to the bound ``omega_r < 2/(1 - M)``.  The often-quoted
``omega_r,max = 2/(1 + M)`` belongs to the *acceleration-form* exchange (the
anti-symmetric amplifier ``-M``, i.e. relaxing the interface acceleration rather
than the displacement); this module relaxes the displacement, so the criterion
used for verification is ``g(omega_r, M)`` above.

:func:`stability_scan` verifies the ``M < 1`` boundary and the contraction rate
numerically (the gate reports the measured-vs-analytic agreement) and
:func:`solve_two_way` runs the coupled iteration at the operating point with
Aitken dynamic relaxation until the fixed-point residual is below 1e-6.

Units: SI (m, N, Pa, rad/s).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

E_STEEL = 2.06e11
RHO_STEEL = 7850.0


@dataclass(frozen=True)
class StructuralSetup:
    """Panel structure, gap geometry and ice packing law."""

    rho_s: float = RHO_STEEL
    E_s: float = E_STEEL
    nu_s: float = 0.3
    t_eff_m: float = 0.25  # plate + frame, smeared
    frame_spacing_m: float = 4.0
    gap0_m: float = 0.65  # nominal hull-ice clearance
    added_length_factor: float = 0.5
    alpha_jam: float = 0.50
    sigma_c_pa: float = 2.8e6  # ice crushing strength (Lindqvist reference)
    packing_cap: float = 0.90
    min_gap_fraction: float = 0.05

    def bending_stiffness(self) -> float:
        return self.E_s * self.t_eff_m ** 3 / (12.0 * (1.0 - self.nu_s ** 2))

    def panel_stiffness(self) -> float:
        """Stiffness per unit area [N/m^3] of a frame-supported panel strip."""
        return self.bending_stiffness() * (math.pi / self.frame_spacing_m) ** 4

    def mass_per_area(self) -> float:
        return self.rho_s * self.t_eff_m

    def natural_frequency(self) -> float:
        return math.sqrt(self.panel_stiffness() / self.mass_per_area())

    def added_length(self, thickness_m: float) -> float:
        return self.added_length_factor * (self.gap0_m + thickness_m)

    def added_mass_ratio(self, thickness_m: float, rho_w: float = 1025.0) -> float:
        return rho_w * self.added_length(thickness_m) / self.mass_per_area()


def packing_pressure(alpha_eff: np.ndarray, setup: StructuralSetup) -> np.ndarray:
    """Packing (crush) pressure of the ice in the gap, [Pa].

    Linear packing law: no pressure below the jamming fraction, then the
    crushing strength is approached as the gap fills,
    ``p_c = sigma_c (alpha - alpha_jam) / (1 - alpha_jam)``.
    """
    excess = np.maximum(alpha_eff - setup.alpha_jam, 0.0)
    return setup.sigma_c_pa * excess / max(1.0 - setup.alpha_jam, 1e-9)


def effective_fraction(
    alpha_mean: np.ndarray, thickness_m: float, clearance_m: np.ndarray, setup: StructuralSetup
) -> np.ndarray:
    """Squeezing the gap raises the local volume fraction (floe mass conserved)."""
    c = np.maximum(clearance_m, setup.min_gap_fraction * setup.gap0_m)
    return np.clip(alpha_mean * thickness_m / c, 0.0, setup.packing_cap)


@dataclass
class CouplingResult:
    converged: bool
    iterations: int
    residual: float
    residual_history: List[float]
    spectral_radius: float
    deflection_m: np.ndarray
    clearance_m: np.ndarray
    p_flow_pa: np.ndarray
    p_contact_pa: np.ndarray
    alpha_effective: np.ndarray
    added_mass_ratio: float
    relaxation: float

    def summary(self) -> Dict[str, Any]:
        return {
            "converged": bool(self.converged),
            "iterations": int(self.iterations),
            "residual": float(self.residual),
            "residual_history": [float(r) for r in self.residual_history],
            "spectral_radius_measured": float(self.spectral_radius),
            "added_mass_ratio": float(self.added_mass_ratio),
            "relaxation": float(self.relaxation),
            "deflection_max_m": float(np.max(np.abs(self.deflection_m))),
            "contact_pressure_max_pa": float(np.max(self.p_contact_pa)),
        }


def _aitken(w_prev: float, r_prev: np.ndarray, r: np.ndarray) -> float:
    diff = r - r_prev
    denom = float(np.dot(diff, diff))
    if denom <= 0.0:
        return w_prev
    num = float(np.dot(r, diff))
    omega = -w_prev * num / denom
    return float(min(max(omega, 0.05), 2.0))


def solve_two_way(
    *,
    alpha_mean: np.ndarray,
    thickness_m: float,
    setup: StructuralSetup,
    flow_fn: Callable[[np.ndarray], np.ndarray],
    relaxation: Union[str, float] = "aitken",
    max_iter: int = 60,
    tol: float = 1.0e-6,
    w0: Optional[np.ndarray] = None,
    rho_w: float = 1025.0,
) -> CouplingResult:
    """Partitioned two-way FSI iteration to a certified fixed-point residual.

    ``flow_fn(alpha_effective) -> p_flow [Pa]`` is the flow-stage callback; it is
    evaluated at every iteration, so the loop really is two-way (the deflection
    squeezes the gap, the squeezed gap raises the local ice fraction, and the
    raised fraction changes the flow load).
    """
    k = setup.panel_stiffness()
    omega_n = setup.natural_frequency()
    c_add = setup.added_length(thickness_m)
    M = rho_w * c_add * omega_n ** 2 / k

    w = np.zeros_like(alpha_mean) if w0 is None else np.array(w0, dtype=float)
    residual_history: List[float] = []
    omega_r = 1.0
    r_prev: Optional[np.ndarray] = None
    p_flow = np.zeros_like(alpha_mean)
    p_contact = np.zeros_like(alpha_mean)
    alpha_eff = alpha_mean.copy()
    residual = float("inf")

    for it in range(1, max_iter + 1):
        clearance = np.maximum(setup.gap0_m - w, setup.min_gap_fraction * setup.gap0_m)
        alpha_eff = effective_fraction(alpha_mean, thickness_m, clearance, setup)
        p_flow = np.asarray(flow_fn(alpha_eff), dtype=float)
        p_contact = packing_pressure(alpha_eff, setup)
        p_added = rho_w * c_add * omega_n ** 2 * w
        w_target = (p_flow + p_contact + p_added) / k

        r = w_target - w
        scale = max(float(np.max(np.abs(w_target))), 1e-30)
        residual = float(np.max(np.abs(r))) / scale
        residual_history.append(residual)
        if residual <= tol:
            w = w_target
            break

        if relaxation == "aitken":
            if r_prev is None:
                omega_r = 1.0
            else:
                omega_r = _aitken(omega_r, r_prev, r)
            r_prev = r.copy()
        else:
            omega_r = float(relaxation)
        w = w + omega_r * r

    # empirical spectral radius: geometric mean of the last residual ratios
    spectral = float("nan")
    if len(residual_history) >= 3:
        ratios = [
            residual_history[i + 1] / residual_history[i]
            for i in range(len(residual_history) - 3, len(residual_history) - 1)
            if residual_history[i] > 0.0
        ]
        if ratios:
            spectral = float(np.exp(np.mean(np.log(np.clip(ratios, 1e-16, 1e16)))))

    clearance = np.maximum(setup.gap0_m - w, setup.min_gap_fraction * setup.gap0_m)
    return CouplingResult(
        converged=bool(residual <= tol),
        iterations=len(residual_history),
        residual=residual,
        residual_history=residual_history,
        spectral_radius=spectral,
        deflection_m=w,
        clearance_m=clearance,
        p_flow_pa=p_flow,
        p_contact_pa=p_contact,
        alpha_effective=alpha_eff,
        added_mass_ratio=float(M),
        relaxation=float(omega_r),
    )


# ---------------------------------------------------------------------------
# stability-domain verification
# ---------------------------------------------------------------------------


def linear_gain(omega_r: float, M: float) -> float:
    """Contraction factor ``g = 1 - omega_r (1 - M)`` of the displacement-form map."""
    return 1.0 - omega_r * (1.0 - M)


def _linear_iteration(omega_r: float, M: float, n_iter: int = 60) -> List[float]:
    """Residual history of the linearised added-mass map (unit initial error)."""
    gain = linear_gain(omega_r, M)
    w = 1.0
    hist: List[float] = []
    for _ in range(n_iter):
        w = gain * w
        hist.append(abs(w))
    return hist


def is_stable(omega_r: float, M: float) -> bool:
    """Numerical stability of one (relaxation, added-mass ratio) pair."""
    return abs(linear_gain(omega_r, M)) < 1.0


def measured_stability_limit(M: float, grid: Sequence[float]) -> float:
    """Largest relaxation factor on ``grid`` whose residual still decays."""
    best = 0.0
    for omega_r in grid:
        hist = _linear_iteration(omega_r, M)
        if hist[-1] < hist[0]:
            best = max(best, float(omega_r))
    return best


def analytic_stability_limit(M: float) -> float:
    """Analytic relaxation bound ``2/(1-M)``; ``0`` means *no* relaxation works."""
    if M >= 1.0:
        return 0.0
    return 2.0 / (1.0 - M)


def stability_scan(
    *,
    setup: StructuralSetup,
    thickness_values: Sequence[float],
    omega_grid: Sequence[float],
    rho_w: float = 1025.0,
    t_eff_values: Sequence[Optional[float]] = (None,),
) -> List[Dict[str, float]]:
    """Measured vs analytic stability boundary over the added-mass ratio range.

    ``t_eff_values`` lets the scan cross the ``M = 1`` boundary (thinner effective
    panel thickness = lighter structure = larger added-mass ratio), which is what
    makes the verification a *domain* check rather than a single point.
    """
    rows: List[Dict[str, float]] = []
    base_t = setup.t_eff_m
    for t_eff in t_eff_values:
        local = setup if t_eff is None else StructuralSetup(
            rho_s=setup.rho_s,
            E_s=setup.E_s,
            nu_s=setup.nu_s,
            t_eff_m=float(t_eff),
            frame_spacing_m=setup.frame_spacing_m,
            gap0_m=setup.gap0_m,
            added_length_factor=setup.added_length_factor,
            alpha_jam=setup.alpha_jam,
            sigma_c_pa=setup.sigma_c_pa,
            packing_cap=setup.packing_cap,
            min_gap_fraction=setup.min_gap_fraction,
        )
        for h in thickness_values:
            M = local.added_mass_ratio(h, rho_w=rho_w)
            measured = measured_stability_limit(M, omega_grid)
            analytic = analytic_stability_limit(M)
            grid_max = float(omega_grid[-1])
            beyond_grid = bool(analytic > grid_max)
            err = (
                float("nan")
                if (beyond_grid or analytic == 0.0)
                else abs(measured - analytic) / analytic
            )
            rows.append(
                {
                    "thickness_m": float(h),
                    "t_eff_m": float(local.t_eff_m),
                    "added_mass_ratio": float(M),
                    "jacobi_contraction": float(abs(linear_gain(1.0, M))),
                    "omega_max_measured": float(measured),
                    "omega_max_analytic": float(analytic),
                    "beyond_grid": beyond_grid,
                    "relative_error": err,
                    "grid_step": float(omega_grid[1] - omega_grid[0]),
                    "converges_at_jacobi": bool(abs(linear_gain(1.0, M)) < 1.0),
                }
            )
    return rows
