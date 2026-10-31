"""Quasi-1-D unsteady homogeneous-mixture solver with cavitation and ice coupling.

Numerics
--------
Finite-volume, staggered (MAC) arrangement on a quasi-1-D duct of given area
distribution ``A(x)``: velocity lives on faces, pressure and vapour volume
fraction on cell centres.  A fractional-step / pressure-correction scheme is
used:

1. explicit momentum predictor on faces (first-order upwind convection,
   face-centred pressure gradient, ice drag momentum source),
2. variable-density pressure Poisson equation with the volume source produced
   by the liquid/vapour mass transfer,
3. velocity correction and conservative transport of the vapour volume
   fraction with the Schnerr-Sauer / Zwart-Gerber-Belamri source, sub-cycled so
   that the stiff mass-transfer term stays explicit-stable.

The scheme is deliberately operator-split and first order in space, which keeps
it verifiable against the analytic Bernoulli inception threshold of a
converging-diverging duct (see ``inception_sigma``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_banded

from .cavitation import (  # noqa: F401  (P_V, RHO_* re-exported for callers)
    P_V,
    RHO_L,
    RHO_V,
    CavitationParams,
    cavitation_source,
    equilibrium_alpha,
)
from .particles import IcePhaseParams, drag_force_per_volume, pressure_fluctuation_rms


@dataclass(frozen=True)
class Quasi1DNozzle:
    """Smooth converging-diverging duct used as a propeller blade-passage analogue."""

    length: float = 0.30
    n_cells: int = 120
    area_in: float = 1.0e-2
    area_throat: float = 5.0e-3
    area_out: float | None = None
    inlet_fraction: float = 0.30
    throat_fraction: float = 0.50

    @property
    def area_ratio(self) -> float:
        return self.area_in / self.area_throat

    def area(self, x: np.ndarray) -> np.ndarray:
        """Area distribution ``A(x)`` (smooth cosine contractions/expansions)."""
        x = np.asarray(x, dtype=float)
        a_in = self.area_in
        a_t = self.area_throat
        a_out = self.area_out if self.area_out is not None else self.area_in
        x_c1 = self.inlet_fraction * self.length
        x_t = self.throat_fraction * self.length

        a = np.full_like(x, a_in)
        # inlet -> throat (contraction)
        mask = (x > x_c1) & (x <= x_t)
        s = (x[mask] - x_c1) / (x_t - x_c1)
        a[mask] = a_t + (a_in - a_t) * 0.5 * (1.0 + np.cos(np.pi * s))
        # throat -> outlet (diffuser)
        mask = x > x_t
        s = (x[mask] - x_t) / (self.length - x_t)
        a[mask] = a_t + (a_out - a_t) * 0.5 * (1.0 - np.cos(np.pi * s))
        return a

    def face_coordinates(self) -> np.ndarray:
        dx = self.length / self.n_cells
        return np.linspace(0.0, self.length, self.n_cells + 1)

    def face_areas(self) -> np.ndarray:
        return self.area(self.face_coordinates())

    def cell_areas(self) -> np.ndarray:
        af = self.face_areas()
        return 0.5 * (af[:-1] + af[1:])

    def cell_centres(self) -> np.ndarray:
        dx = self.length / self.n_cells
        return (np.arange(self.n_cells) + 0.5) * dx


@dataclass
class FlowState:
    """Snapshot of a converged or time-resolved flow state."""

    x: np.ndarray
    area: np.ndarray
    u: np.ndarray            # face velocity  (n+1)
    p: np.ndarray            # cell pressure  (n)
    alpha: np.ndarray        # cell vapour volume fraction (n)
    u_particle: np.ndarray   # cell ice-particle velocity (n)
    ice_concentration: np.ndarray
    time: float = 0.0

    @property
    def cavitation_volume(self) -> float:
        dx = float(self.x[0]) * 0.0 + (self.x[-1] - self.x[0])  # cell thickness
        return float(np.sum(self.alpha * self.area * dx))

    def vapour_volume_fraction_max(self) -> float:
        return float(np.max(self.alpha))

    def pressure_min(self) -> float:
        return float(np.min(self.p))


class MixtureFlow1D:
    """Unsteady quasi-1-D cavitating mixture flow with dilute ice particles."""

    def __init__(
        self,
        nozzle: Quasi1DNozzle | None = None,
        u_in: float = 2.0,
        p_out: float = 1.0e5,
        alpha0: float | None = None,
        cavitation: CavitationParams | None = None,
        ice: IcePhaseParams | None = None,
        ice_concentration: float = 0.0,
        nu: float = 2.0e-6,
        alpha_threshold: float | None = None,
    ) -> None:
        self.nozzle = nozzle or Quasi1DNozzle()
        self.cav = cavitation or CavitationParams()
        self.ice = ice or IcePhaseParams()
        self.n = self.nozzle.n_cells
        self.dx = self.nozzle.length / self.n
        self.x = self.nozzle.cell_centres()
        self.x_faces = self.nozzle.face_coordinates()
        self.af = self.nozzle.face_areas()
        self.ac = self.nozzle.cell_areas()

        self.alpha0 = float(self.cav.alpha_nuc if alpha0 is None else alpha0)
        self.alpha_threshold = (
            4.0 * self.alpha0 if alpha_threshold is None else alpha_threshold
        )

        self.u_in = float(u_in)
        self.p_out = float(p_out)
        self.nu = float(nu)
        self.ice_concentration = float(ice_concentration)

        self.u = np.full(self.n + 1, self.u_in)
        self.p = np.full(self.n, self.p_out)
        self.alpha = np.full(self.n, self.alpha0)
        self.u_particle = np.full(self.n, self.u_in)
        self.time = 0.0
        self.steps = 0
        self.initialise_inviscid_state()

        # c(x) from particle-flux conservation, refreshed every advance()
        self.concentration = np.zeros(self.n)

    # -- initialisation ---------------------------------------------------
    def initialise_inviscid_state(self) -> None:
        """Start from the divergence-free, Bernoulli-consistent 1-D solution.

        A uniform-velocity start violates continuity in a duct with varying
        area and forces an enormous first pressure correction, so the state is
        seeded with ``u A = const`` and the corresponding inviscid pressure
        profile.  This is only an initial guess: the transient solver still
        decides the final state.
        """
        flux = self.nozzle.area_in * self.u_in
        self.u = flux / self.af
        self.u[0] = self.u_in
        self.u_particle = 0.5 * (self.u[:-1] + self.u[1:])
        u_out = self.u[-1]
        p_face = self.p_out + 0.5 * self.cav.rho_l * (u_out ** 2 - self.u ** 2)
        self.p = 0.5 * (p_face[:-1] + p_face[1:])
        self.alpha = np.full(self.n, self.alpha0)
        self.time = 0.0
        self.steps = 0
        self.update_concentration()

    # -- derived fields ---------------------------------------------------
    def mixture_density(self, alpha: np.ndarray | None = None) -> np.ndarray:
        a = self.alpha if alpha is None else alpha
        return (1.0 - a) * self.cav.rho_l + a * self.cav.rho_v

    def sigma(self) -> float:
        """Cavitation number built on the inlet static pressure."""
        return float(
            (self.p[0] - self.cav.p_v) / (0.5 * self.cav.rho_l * self.u_in ** 2)
        )

    def cell_velocity(self) -> np.ndarray:
        return 0.5 * (self.u[:-1] + self.u[1:])

    def ice_slip(self) -> np.ndarray:
        return self.cell_velocity() - self.u_particle

    def update_concentration(self) -> None:
        """Conserve particle flux: ``c A u_p = c_in A_in u_in``."""
        if self.ice_concentration <= 0.0:
            self.concentration = np.zeros(self.n)
            return
        flux_in = self.ice_concentration * self.nozzle.area_in * self.u_in
        u_p = np.maximum(self.u_particle, 1.0e-6)
        c = flux_in / (self.ac * u_p)
        np.clip(c, 0.0, 0.60, out=c)
        self.concentration = c

    def drag_source_cells(self) -> np.ndarray:
        """Liquid->ice momentum source per cell [N/m^3] (positive decelerates liquid)."""
        if self.ice_concentration <= 0.0:
            return np.zeros(self.n)
        slip = self.ice_slip()
        return np.array(
            [
                drag_force_per_volume(s, c, self.ice)
                for s, c in zip(slip, self.concentration)
            ]
        )

    def pressure_fluctuation_rms_profile(self) -> np.ndarray:
        if self.ice_concentration <= 0.0:
            return np.zeros(self.n)
        slip = self.ice_slip()
        return np.array(
            [
                pressure_fluctuation_rms(s, c, self.ice)
                for s, c in zip(slip, self.concentration)
            ]
        )

    # -- one time step ----------------------------------------------------
    def advance(
        self,
        dt: float,
        *,
        u_in: float | None = None,
        p_out: float | None = None,
        pressure_fluctuation: np.ndarray | None = None,
        particle_relaxation: bool = True,
        n_piso: int = 3,
    ) -> None:
        """Advance the solution by ``dt`` seconds.

        The liquid/vapour mass transfer is linearised in the pressure equation
        (``dS/dp`` term below).  Without it the explicit volume source makes the
        pressure-correction problem violently stiff as soon as a cavity forms
        (``|dS/dp| ~ 0.1 1/(Pa s)`` against a Poisson diagonal of
        ``~0.03 1/(Pa s)``); with it the mixture behaves as a slightly
        compressible medium, which is the standard remedy in cavitating-flow
        solvers.
        """
        if u_in is not None:
            self.u_in = float(u_in)
        if p_out is not None:
            self.p_out = float(p_out)
        n = self.n
        m = n - 1

        # --- particle sub-step (relaxation towards the liquid velocity) ---
        if particle_relaxation and self.ice_concentration > 0.0:
            tau_p = self.ice.d_p ** 2 * (self.ice.rho_ice + 0.5 * self.cav.rho_l) / (
                18.0 * self.ice.mu_l
            )
            tau_p = max(tau_p, 1.0e-9)
            u_cell = self.cell_velocity()
            relax = 1.0 - np.exp(-dt / tau_p)
            self.u_particle = self.u_particle + relax * (u_cell - self.u_particle)
        else:
            self.u_particle = self.cell_velocity()
        self.update_concentration()
        drag = self.drag_source_cells()

        p_eff = self.p if pressure_fluctuation is None else self.p + pressure_fluctuation

        for _ in range(max(n_piso, 1)):
            rho = self.mixture_density()
            rho_f = np.empty(n + 1)
            rho_f[1:-1] = 0.5 * (rho[:-1] + rho[1:])
            rho_f[0] = rho[0]
            rho_f[-1] = rho[-1]

            mdot = np.asarray(cavitation_source(p_eff, self.alpha, self.cav), dtype=float)
            volume_source = mdot * (1.0 / self.cav.rho_v - 1.0 / self.cav.rho_l)

            # momentum predictor on interior faces
            uf = self.u
            upwind = np.where(uf[1:-1] > 0.0, uf[:-2], uf[2:])
            conv = uf[1:-1] * (uf[1:-1] - upwind) / self.dx
            visc = self.nu * (uf[2:] - 2.0 * uf[1:-1] + uf[:-2]) / self.dx ** 2
            grad_p = (self.p[1:] - self.p[:-1]) / self.dx
            drag_face = 0.5 * (drag[:-1] + drag[1:])
            accel = -conv + visc - grad_p / rho_f[1:-1] - drag_face / rho_f[1:-1]
            u_star = uf.copy()
            u_star[1:-1] = uf[1:-1] + dt * accel
            u_star[0] = self.u_in

            div_star = (self.af[1:] * u_star[1:] - self.af[:-1] * u_star[:-1]) / (
                self.ac * self.dx
            )
            rhs = (div_star - volume_source)[:m]

            # linearised mass-transfer derivative dS_vol/dp  [1/(Pa s)]
            probe = 0.01 * np.maximum(np.abs(self.p), 1.0e3)
            s_plus = np.asarray(
                cavitation_source(p_eff + probe, self.alpha, self.cav), dtype=float
            ) * (1.0 / self.cav.rho_v - 1.0 / self.cav.rho_l)
            s_minus = np.asarray(
                cavitation_source(p_eff - probe, self.alpha, self.cav), dtype=float
            ) * (1.0 / self.cav.rho_v - 1.0 / self.cav.rho_l)
            dsdp = (s_plus - s_minus) / (2.0 * probe)

            a_r = self.af[1:m + 1] * dt / (rho_f[1:m + 1] * self.dx)
            a_l = self.af[0:m] * dt / (rho_f[0:m] * self.dx)
            # The linearised damping is capped at a few times the Poisson
            # diagonal: dS/dp diverges as p -> p_v (sqrt singularity) while the
            # source itself vanishes there, so the unconstrained derivative
            # would freeze the pressure equation exactly where it must correct.
            poisson_scale = (a_r + a_l) / (self.ac[:m] * self.dx)
            # Equilibrium mixture compressibility of a cavity at p ~ p_v
            # (Blake relation): the pressure cannot dive through p_v without
            # creating volume, so this term makes the pressure equation bounded.
            vol_gain = 1.0 - self.cav.rho_v / self.cav.rho_l
            a_eq_p = equilibrium_alpha(p_eff + probe, self.cav)
            a_eq_m = equilibrium_alpha(p_eff - probe, self.cav)
            daeq_dp = (a_eq_p - a_eq_m) / (2.0 * probe)
            damp_eq = np.maximum(-daeq_dp[:m], 0.0) * vol_gain / dt
            damping = np.maximum(-dsdp[:m], 0.0) + damp_eq
            damping = np.minimum(damping, 300.0 * poisson_scale)
            diag = -(a_r + a_l) / (self.ac[:m] * self.dx) - damping
            diag[0] += a_l[0] / (self.ac[0] * self.dx)     # inlet: dp'/dx = 0
            upper = a_r[:m - 1] / (self.ac[:m - 1] * self.dx)
            lower = a_l[1:m] / (self.ac[1:m] * self.dx)

            banded = np.zeros((3, m))
            banded[0, 1:] = upper
            banded[1, :] = diag
            banded[2, :-1] = lower
            # Strong Dirichlet row for the pressure level (outlet cell p' = 0).
            # Merely dropping that cell's equation would leave the constant mode
            # anchored only through a ~1e-6 off-diagonal, i.e. a numerically
            # singular system whose constant mode explodes.
            banded[0, -1] = 0.0
            banded[1, -1] = 1.0
            rhs[-1] = 0.0
            p_corr = solve_banded((1, 1), banded, rhs)
            p_corr = np.concatenate([p_corr, [0.0]])

            self.p = self.p + p_corr
            self.u[1:-1] = u_star[1:-1] - dt / (rho_f[1:-1] * self.dx) * (
                p_corr[1:] - p_corr[:-1]
            )
            self.u[-1] = (
                self.ac[-1] * self.dx * volume_source[-1] + self.af[-2] * self.u[-2]
            ) / self.af[-1]
            self.u[0] = self.u_in
            p_eff = self.p if pressure_fluctuation is None else self.p + pressure_fluctuation

        # --- vapour volume fraction transport (conservative upwind) ------
        mdot = np.asarray(cavitation_source(p_eff, self.alpha, self.cav), dtype=float)
        rate = mdot / self.cav.rho_v
        n_sub = int(np.clip(np.ceil(np.max(np.abs(rate)) * dt / 0.05), 1, 400))
        alpha = self.alpha.copy()
        if n_sub > 1:
            dts = dt / n_sub
            for _ in range(n_sub):
                alpha = alpha + dts * np.asarray(
                    cavitation_source(p_eff, alpha, self.cav), dtype=float
                ) / self.cav.rho_v
                np.clip(alpha, self.cav.alpha_floor, 1.0 - 1.0e-6, out=alpha)
        else:
            alpha = alpha + dt * rate
            np.clip(alpha, self.cav.alpha_floor, 1.0 - 1.0e-6, out=alpha)

        up_a = np.where(
            self.u > 0.0,
            np.concatenate([[self.alpha0], alpha]),
            np.concatenate([alpha, [alpha[-1]]]),
        )
        flux = self.af * self.u * up_a
        self.alpha = alpha - dt * (flux[1:] - flux[:-1]) / (self.ac * self.dx)
        np.clip(self.alpha, self.cav.alpha_floor, 1.0 - 1.0e-6, out=self.alpha)

        self.time += dt
        self.steps += 1

    # -- driver -----------------------------------------------------------
    def state(self) -> FlowState:
        return FlowState(
            x=self.x.copy(),
            area=self.ac.copy(),
            u=self.u.copy(),
            p=self.p.copy(),
            alpha=self.alpha.copy(),
            u_particle=self.u_particle.copy(),
            ice_concentration=self.concentration.copy(),
            time=self.time,
        )

    def divergence(self, u: np.ndarray | None = None) -> np.ndarray:
        """Volume divergence ``(1/A) d(Au)/dx`` of a face-velocity field [1/s]."""
        uu = self.u if u is None else u
        return (self.af[1:] * uu[1:] - self.af[:-1] * uu[:-1]) / (self.ac * self.dx)

    def stable_dt(self, cfl: float = 0.4, source_limit: float = 0.02) -> float:
        """Explicit-stability limit: convection CFL and mass-transfer stiffness.

        The barotropic source reaches ``|mdot/rho_v| ~ 1e3 s^-1`` once a cavity
        forms, so the CFL step alone (``~4e-4 s``) is far too large for the
        explicit coupling; the step is therefore also bounded by
        ``source_limit / max|S|``.
        """
        u_max = max(float(np.max(np.abs(self.u))), 1.0e-3)
        dt = cfl * self.dx / u_max
        rate = np.abs(
            np.asarray(cavitation_source(self.p, self.alpha, self.cav), dtype=float)
        ) / self.cav.rho_v
        s_max = float(np.max(rate)) if rate.size else 0.0
        if s_max > 0.0:
            dt = min(dt, source_limit / s_max)
        return float(dt)

    def run_to_steady(
        self,
        max_steps: int = 4000,
        tol: float = 1.0e-6,
        cfl: float = 0.4,
        check_every: int = 50,
    ) -> dict:
        """March until the velocity field stops changing (or ``max_steps``)."""
        residual = np.inf
        history = []
        for k in range(max_steps):
            dt = self.stable_dt(cfl)
            u_old = self.u.copy()
            self.advance(dt)
            if (k + 1) % check_every == 0:
                residual = float(np.max(np.abs(self.u - u_old)))
                history.append(residual)
                if residual < tol * max(self.u_in, 1.0e-6):
                    break
        return {
            "steps": self.steps,
            "residual": residual,
            "dv_per_step": history[-1] if history else np.nan,
            "time": self.time,
        }

    def cavitation_volume(self) -> float:
        return float(np.sum(self.alpha * self.ac * self.dx))

    def cavitates(self) -> bool:
        return bool(np.max(self.alpha) > self.alpha_threshold)


# ---------------------------------------------------------------------------
# Cavitation inception
# ---------------------------------------------------------------------------
def analytic_inception_sigma(area_ratio: float) -> float:
    """Bernoulli inception number of an inviscid 1-D duct: ``(A_in/A_t)^2 - 1``."""
    return float(area_ratio ** 2 - 1.0)


def _run_at_p_out(
    nozzle: Quasi1DNozzle,
    u_in: float,
    p_out: float,
    ice_concentration: float,
    cav: CavitationParams,
    ice: IcePhaseParams,
    max_steps: int,
    pressure_fluctuation_factor: float = 0.0,
    seed: int = 0,
) -> tuple[MixtureFlow1D, dict]:
    solver = MixtureFlow1D(
        nozzle=nozzle,
        u_in=u_in,
        p_out=p_out,
        cavitation=cav,
        ice=ice,
        ice_concentration=ice_concentration,
    )
    # mean-field pressure fluctuation amplitude for the ice phase
    p_rms = np.zeros(solver.n)
    if ice_concentration > 0.0 and pressure_fluctuation_factor > 0.0:
        p_rms = pressure_fluctuation_factor * solver.pressure_fluctuation_rms_profile()
    info = solver.run_to_steady(max_steps=max_steps)
    return solver, info


def inception_sigma(
    nozzle: Quasi1DNozzle | None = None,
    u_in: float = 2.0,
    ice_concentration: float = 0.0,
    cav: CavitationParams | None = None,
    ice: IcePhaseParams | None = None,
    max_steps: int = 3000,
    n_bisect: int = 18,
    p_bounds: tuple[float, float] | None = None,
    mean_pressure_shift: float = 0.0,
) -> dict:
    """Bisect the outlet pressure to find the cavitation inception number.

    Inception is declared when the peak vapour fraction exceeds
    ``alpha_threshold`` (4 x the nuclei background), i.e. when a numerically
    resolvable cavity has formed.

    ``mean_pressure_shift`` (Pa) lowers the *time-mean* pressure that the
    inception test is referred to without touching the fluctuation model; it is
    used to test the sensitivity of the criterion.
    """
    nozzle = nozzle or Quasi1DNozzle()
    cav = cav or CavitationParams()
    ice = ice or IcePhaseParams()

    def onset(p_out: float) -> bool:
        solver, _ = _run_at_p_out(
            nozzle, u_in, p_out, ice_concentration, cav, ice, max_steps
        )
        return solver.cavitates() and (
            float(np.min(solver.p)) - mean_pressure_shift <= cav.p_v
        )

    if p_bounds is None:
        # upper bound: no cavitation, lower bound: deep cavitation
        p_hi = cav.p_v + 12.0 * 0.5 * cav.rho_l * u_in ** 2
        p_lo = cav.p_v - 2.0 * 0.5 * cav.rho_l * u_in ** 2
    else:
        p_lo, p_hi = p_bounds

    for _ in range(n_bisect):
        p_mid = 0.5 * (p_lo + p_hi)
        if onset(p_mid):
            p_lo = p_mid      # cavitating: need higher pressure to suppress
        else:
            p_hi = p_mid
    p_crit = 0.5 * (p_lo + p_hi)

    solver, info = _run_at_p_out(nozzle, u_in, p_crit, ice_concentration, cav, ice, max_steps)
    # inlet pressure at the critical point
    p_in = float(solver.p[0])
    sigma = (p_in - cav.p_v) / (0.5 * cav.rho_l * u_in ** 2)
    return {
        "sigma": sigma,
        "p_out_crit": p_crit,
        "p_in": p_in,
        "p_min": float(np.min(solver.p)),
        "u_throat": float(np.max(np.abs(solver.u))),
        "alpha_max": float(np.max(solver.alpha)),
        "ice_concentration": ice_concentration,
        "steps": info["steps"],
        "p_out_bracket": [p_lo, p_hi],
    }


def sweep_sigma(
    nozzle: Quasi1DNozzle | None = None,
    u_in: float = 2.0,
    sigma_values: list[float] | None = None,
    ice_concentration: float = 0.0,
    cav: CavitationParams | None = None,
    ice: IcePhaseParams | None = None,
    max_steps: int = 3000,
) -> list[dict]:
    """Steady-state cavitation volume over a range of imposed cavitation numbers.

    For each prescribed ``sigma`` the outlet pressure is chosen so that the
    inlet static pressure satisfies ``sigma = (p_in - p_v)/(0.5 rho u_in^2)``.
    Because the inlet pressure depends weakly on the solution, the outlet
    pressure is fixed by prior iteration (2 correction sweeps).
    """
    nozzle = nozzle or Quasi1DNozzle()
    cav = cav or CavitationParams()
    ice = ice or IcePhaseParams()
    sigmas = sigma_values if sigma_values is not None else [2.0, 2.5, 3.0, 4.0]
    out = []
    for sigma in sigmas:
        p_out = cav.p_v + 0.5 * cav.rho_l * u_in ** 2 * sigma
        for _ in range(3):
            solver, info = _run_at_p_out(
                nozzle, u_in, p_out, ice_concentration, cav, ice, max_steps
            )
            sigma_now = (solver.p[0] - cav.p_v) / (0.5 * cav.rho_l * u_in ** 2)
            p_out += 0.5 * cav.rho_l * u_in ** 2 * (sigma - sigma_now)
        out.append(
            {
                "sigma_target": sigma,
                "sigma_achieved": float(sigma_now),
                "p_out": float(p_out),
                "cavitation_volume": solver.cavitation_volume(),
                "alpha_max": float(np.max(solver.alpha)),
                "p_min": float(np.min(solver.p)),
                "ice_concentration": ice_concentration,
                "steps": info["steps"],
            }
        )
    return out