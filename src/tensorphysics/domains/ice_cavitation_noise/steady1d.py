"""Steady quasi-1-D cavitating mixture flow: ODE marching with ice coupling.

Why a second solver?
--------------------
The unsteady projection solver in :mod:`flow1d` is the natural tool for
time-resolved pressure histories, but it is stiff once a real cavity forms: the
vapour volume source (``|mdot/rho_v| ~ 1e3 1/s``) must be balanced by the
liquid, and an incompressible projection scheme then reacts on the acoustic
time scale of the duct, which is shorter than the affordable explicit time
step.  For the *steady* operating points that the acceptance criteria refer to
(the inception number and the cavity volume), the steady equations are marched
directly as an ODE system with an implicit integrator.  This is the classical
quasi-1-D treatment of a cavitating duct and it is both faster and far more
robust.

Steady equations (one-way coupled two-fluid, isothermal mixture)
----------------------------------------------------------------
With the total mass flux ``M = rho u A = const`` and the mixture density
``rho(alpha) = (1-alpha) rho_l + alpha rho_v``:

* vapour fraction transport::

      d(alpha u A)/dx = A mdot / rho_v

* mixture momentum (no wall friction, inviscid core)::

      rho u du/dx = -dp/dx - S_drag      (S_drag = ice drag per unit volume)

* ice particle momentum (steady, nonlinear drag)::

      u_p du_p/dx = (u - u_p) / tau_p(w)

which is integrated in ``x`` together with ``alpha`` and ``p``.  The inlet
state is ``(p_in, alpha_nuc, u_in)``; the exit pressure, the cavitation volume
``V_cav = int alpha A dx`` and the cavitation number
``sigma = (p_in - p_v) / (0.5 rho_l u_in^2)`` follow from the march.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .cavitation import CavitationParams, cavitation_source
from .flow1d import Quasi1DNozzle
from .particles import IcePhaseParams, drag_force_per_particle, pressure_fluctuation_rms


@dataclass
class SteadyResult:
    x: np.ndarray
    area: np.ndarray
    p: np.ndarray
    alpha: np.ndarray
    u: np.ndarray
    u_particle: np.ndarray
    concentration: np.ndarray
    sigma: float
    p_in: float
    p_out: float
    cavitation_volume: float
    alpha_max: float
    p_min: float
    x_throat: float
    p_throat: float
    pressure_fluctuation_rms_max: float
    converged: bool = True
    message: str = ""

    def at_throat(self) -> dict:
        return {
            "x": float(self.x_throat),
            "p": float(self.p_throat),
            "alpha": float(np.interp(self.x_throat, self.x, self.alpha)),
            "u": float(np.interp(self.x_throat, self.x, self.u)),
        }


class SteadyNozzleFlow:
    """Steady cavitating flow through a quasi-1-D duct with a dilute ice phase."""

    def __init__(
        self,
        nozzle: Quasi1DNozzle | None = None,
        u_in: float = 2.0,
        ice_concentration: float = 0.0,
        cav: CavitationParams | None = None,
        ice: IcePhaseParams | None = None,
        alpha_in: float | None = None,
    ) -> None:
        self.nozzle = nozzle or Quasi1DNozzle()
        self.u_in = float(u_in)
        self.ice_concentration = float(ice_concentration)
        self.cav = cav or CavitationParams()
        self.ice = ice or IcePhaseParams()
        self.alpha_in = float(self.cav.alpha_nuc if alpha_in is None else alpha_in)
        self.rho_in = (1.0 - self.alpha_in) * self.cav.rho_l + self.alpha_in * self.cav.rho_v
        self.mass_flux = self.rho_in * self.u_in * self.nozzle.area_in
        self.particle_flux = (
            self.ice_concentration * self.nozzle.area_in * self.u_in
        )
        x_faces = self.nozzle.face_coordinates()
        self.x_throat = 0.5 * (
            self.nozzle.throat_fraction * self.nozzle.length
            + self.nozzle.throat_fraction * self.nozzle.length
        )
        self._af = self.nozzle.area(x_faces)

    # -- kinematics -------------------------------------------------------
    def area(self, x: float) -> float:
        return float(self.nozzle.area(np.asarray(x)))

    def mixture_density(self, alpha: float) -> float:
        return (1.0 - alpha) * self.cav.rho_l + alpha * self.cav.rho_v

    def sigma_of(self, p_in: float) -> float:
        return (p_in - self.cav.p_v) / (0.5 * self.cav.rho_l * self.u_in ** 2)

    def p_in_of(self, sigma: float) -> float:
        return self.cav.p_v + sigma * 0.5 * self.cav.rho_l * self.u_in ** 2

    # -- ODE right-hand side ---------------------------------------------
    def _rhs(self, x: float, y: np.ndarray, freeze_vapour: bool = False) -> list[float]:
        p, alpha, u_p = y
        area = self.area(x)
        darea_dx = (
            self.area(x + 1.0e-7) - self.area(x - 1.0e-7)
        ) / 2.0e-7
        rho = self.mixture_density(alpha)
        drho_dalpha = self.cav.rho_v - self.cav.rho_l
        u = self.mass_flux / (rho * area)
        mdot = float(np.asarray(cavitation_source(p, alpha, self.cav)).reshape(()))

        # d(alpha u A)/dx = A mdot/rho_v  with  u A = M/rho:
        #   dalpha/dx = (A mdot/rho_v) / (u A - alpha M drho/dalpha / rho^2)
        u_a = u * area
        denominator = u_a - alpha * self.mass_flux * drho_dalpha / rho ** 2
        dalpha_dx = (area * mdot / self.cav.rho_v) / denominator if denominator != 0.0 else 0.0
        if freeze_vapour:
            # pure single-phase flow: the vapour fraction is held at the nuclei
            # background, so the pressure field has no cavity feedback
            dalpha_dx = 0.0

        # ice phase: concentration from particle-flux conservation
        slip = u - u_p
        tau = self._particle_tau(slip)
        du_p_dx = (u - u_p) / (max(abs(u_p), 1.0e-6) * tau)
        concentration = (
            self.particle_flux / (area * max(abs(u_p), 1.0e-6))
            if self.ice_concentration > 0.0
            else 0.0
        )
        concentration = min(concentration, 0.6)
        drag = 0.0
        if self.ice_concentration > 0.0 and concentration > 0.0:
            force = drag_force_per_particle(
                slip, self.ice.d_p, self.ice.rho_l, self.ice.mu_l
            )
            n_p = concentration / (np.pi * self.ice.d_p ** 3 / 6.0)
            drag = n_p * force

        du_dx = -u * ((drho_dalpha / rho) * dalpha_dx + darea_dx / area)
        dp_dx = -rho * u * du_dx - drag
        return [dp_dx, dalpha_dx, du_p_dx]

    def _particle_tau(self, slip: float) -> float:
        """Effective particle relaxation time for the local slip."""
        d = self.ice.d_p
        stokes = d ** 2 * (self.ice.rho_ice + 0.5 * self.ice.rho_l) / (18.0 * self.ice.mu_l)
        if self.ice_concentration <= 0.0 or abs(slip) < 1.0e-9:
            return stokes
        force = abs(
            drag_force_per_particle(slip, d, self.ice.rho_l, self.ice.mu_l)
        )
        mass = self.ice.rho_ice * np.pi * d ** 3 / 6.0
        return max(mass * abs(slip) / force, 1.0e-9) if force > 0.0 else stokes

    # -- march ------------------------------------------------------------
    def _flow_state(self, p, alpha, u_p, x):
        """Algebraic quantities at a station (velocity, density, sources)."""
        area = self.area(x)
        h = 1.0e-7
        darea_dx = (self.area(x + h) - self.area(x - h)) / (2.0 * h)
        rho = self.mixture_density(alpha)
        drho_dalpha = self.cav.rho_v - self.cav.rho_l
        u = self.mass_flux / (rho * area)
        mdot = float(np.asarray(cavitation_source(p, alpha, self.cav)).reshape(()))
        concentration = 0.0
        drag = 0.0
        slip = u - u_p
        tau = self._particle_tau(slip)
        if self.ice_concentration > 0.0:
            concentration = min(
                self.particle_flux / (area * max(abs(u_p), 1.0e-6)), 0.6
            )
            if concentration > 0.0:
                force = drag_force_per_particle(
                    slip, self.ice.d_p, self.ice.rho_l, self.ice.mu_l
                )
                n_p = concentration / (np.pi * self.ice.d_p ** 3 / 6.0)
                drag = n_p * force
        u_a = u * area
        denominator = u_a - alpha * self.mass_flux * drho_dalpha / rho ** 2
        dalpha_dx = (area * mdot / self.cav.rho_v) / denominator if denominator != 0.0 else 0.0
        if freeze_vapour:
            # pure single-phase flow: the vapour fraction is held at the nuclei
            # background, so the pressure field has no cavity feedback
            dalpha_dx = 0.0
        du_dx = -u * ((drho_dalpha / rho) * dalpha_dx + darea_dx / area)
        dp_dx = -rho * u * du_dx - drag
        du_p_dx = (u - u_p) / (max(abs(u_p), 1.0e-6) * tau)
        return {
            "area": area,
            "darea_dx": darea_dx,
            "rho": rho,
            "u": u,
            "mdot": mdot,
            "dalpha_dx": dalpha_dx,
            "dp_dx": dp_dx,
            "du_p_dx": du_p_dx,
            "concentration": concentration,
            "drag": drag,
        }

    def march(
        self,
        sigma_of_inlet: float | None = None,
        p_in: float | None = None,
        n_samples: int = 120,
        rtol: float = 1.0e-7,
        atol: float = 1.0e-10,
        alpha_cap: float = 0.15,
        freeze_vapour: bool = False,
    ) -> SteadyResult:
        """Integrate the steady equations from the inlet to the exit.

        ``freeze_vapour=True`` keeps ``alpha = alpha_nuc`` (no cavity feedback)
        and returns the *single-phase* pressure field.  That is the field used
        for the cavitation-susceptible volume metric: it is well posed for any
        pressure level, whereas the barotropic cavity feedback has a sharp
        bifurcation at inception (see the report's limitations section).
        """
        if p_in is None:
            if sigma_of_inlet is None:
                raise ValueError("provide p_in or sigma_of_inlet")
            p_in = self.p_in_of(sigma_of_inlet)
        p_in = float(p_in)
        y0 = [p_in, self.cav.alpha_nuc if freeze_vapour else self.alpha_in, self.u_in]

        def rhs(x, y):
            return self._rhs(x, y, freeze_vapour=freeze_vapour)

        def cap_event(x, y):
            return y[1] - alpha_cap

        cap_event.terminal = True
        cap_event.direction = 1.0

        x_end = self.nozzle.length
        x_eval = np.linspace(0.0, x_end, n_samples)
        sol = solve_ivp(
            rhs,
            (0.0, x_end),
            y0,
            method="Radau",
            t_eval=x_eval,
            rtol=rtol,
            atol=atol,
            events=[cap_event],
        )
        converged = bool(sol.success)
        message = str(sol.message)
        x = sol.t
        if x.size < 2:
            x = np.array([0.0, x_end])
            y = np.column_stack([y0, y0])
        else:
            y = sol.y
        p = y[0]
        alpha = np.clip(y[1], self.cav.alpha_floor, 1.0 - 1.0e-9)
        u_p = y[2]
        area = self.area_array(x)
        rho = np.array([self.mixture_density(a) for a in alpha])
        u = self.mass_flux / (rho * area)
        concentration = np.zeros_like(x)
        if self.ice_concentration > 0.0:
            concentration = np.clip(
                self.particle_flux / (area * np.maximum(np.abs(u_p), 1.0e-6)), 0.0, 0.6
            )
        slip = u - u_p
        p_rms = np.array(
            [
                pressure_fluctuation_rms(s, c, self.ice)
                for s, c in zip(slip, concentration)
            ]
        )
        cav_volume = float(np.trapezoid(alpha * area, x)) if x.size > 1 else 0.0
        idx_throat = int(np.argmin(area))
        return SteadyResult(
            x=x,
            area=area,
            p=p,
            alpha=alpha,
            u=u,
            u_particle=u_p,
            concentration=concentration,
            sigma=self.sigma_of(p_in),
            p_in=p_in,
            p_out=float(p[-1]),
            cavitation_volume=cav_volume,
            alpha_max=float(np.max(alpha)),
            p_min=float(np.min(p)),
            x_throat=float(x[idx_throat]),
            p_throat=float(p[idx_throat]),
            pressure_fluctuation_rms_max=float(np.max(p_rms)) if p_rms.size else 0.0,
            converged=converged,
            message=message,
        )

    # -- cavitation-susceptible volume ------------------------------------
    def suscept_volume(self, sigma_of_inlet: float, p_shift: float = 0.0) -> dict:
        """Duct volume where the *single-phase* pressure is below ``p_v``.

        ``p_shift`` lowers the mean pressure field by an externally supplied
        fluctuation level (used for the ice-induced pressure-fluctuation
        sensitivity), see the report's model description.
        """
        result = self.march(
            sigma_of_inlet=sigma_of_inlet, freeze_vapour=True, n_samples=400
        )
        below = (result.p - p_shift) <= self.cav.p_v
        area = result.area
        volume = float(np.trapezoid(np.where(below, area, 0.0), result.x))
        return {
            "sigma": float(result.sigma),
            "p_in": float(result.p_in),
            "p_min": float(result.p_min),
            "volume": volume,
            "length": float(np.trapezoid(below.astype(float), result.x)),
            "result": result,
        }

    def area_array(self, x: np.ndarray) -> np.ndarray:
        return self.nozzle.area(np.asarray(x, dtype=float))

    # -- inception --------------------------------------------------------
    def inception_sigma(
        self,
        alpha_threshold_factor: float = 4.0,
        sigma_bracket: tuple[float, float] = (1.0, 40.0),
        n_bisect: int = 40,
        fluctuation_factor: float = 0.0,
        kappa: float | None = None,
    ) -> dict:
        """Cavitation inception number from the steady solution family.

        Inception is declared when the peak vapour fraction exceeds
        ``alpha_threshold_factor * alpha_nuc``.  With ``fluctuation_factor`` > 0
        the criterion becomes the *instantaneous* one: a cavity appears once the
        local mean pressure minus ``fluctuation_factor * p'_rms`` (the
        ice-induced pressure fluctuation) drops to the vapour pressure.  This is
        the mechanism by which a particle-laden inflow raises the apparent
        inception number.
        """
        cav = self.cav
        if kappa is not None:
            ice = IcePhaseParams(
                rho_ice=self.ice.rho_ice,
                d_p=self.ice.d_p,
                kappa=kappa,
                strouhal=self.ice.strouhal,
                rho_l=self.ice.rho_l,
                mu_l=self.ice.mu_l,
            )
        else:
            ice = self.ice

        def cavitates(sigma: float) -> tuple[bool, SteadyResult]:
            result = self.march(sigma_of_inlet=sigma)
            trigger = alpha_threshold_factor * cav.alpha_nuc
            ok = result.alpha_max > trigger
            if ok and fluctuation_factor > 0.0:
                ok = (
                    result.p_min - fluctuation_factor * result.pressure_fluctuation_rms_max
                    <= cav.p_v
                )
            elif ok:
                ok = result.p_min <= cav.p_v
            return ok, result

        s_lo, s_hi = sigma_bracket
        ok_lo, _ = cavitates(s_lo)
        ok_hi, res_hi = cavitates(s_hi)
        if not ok_lo or ok_hi:
            # bracket does not contain the onset
            return {
                "sigma": float("nan"),
                "bracket": [s_lo, s_hi],
                "alpha_max": res_hi.alpha_max,
                "note": "onset outside bracket",
                "ice_concentration": self.ice_concentration,
            }
        for _ in range(n_bisect):
            s_mid = 0.5 * (s_lo + s_hi)
            ok, _ = cavitates(s_mid)
            if ok:
                s_lo = s_mid
            else:
                s_hi = s_mid
        sigma_i = 0.5 * (s_lo + s_hi)
        _, result = cavitates(sigma_i)
        return {
            "sigma": float(sigma_i),
            "bracket": [float(s_lo), float(s_hi)],
            "p_in": float(result.p_in),
            "p_min": float(result.p_min),
            "p_throat": float(result.p_throat),
            "alpha_max": float(result.alpha_max),
            "cavitation_volume": float(result.cavitation_volume),
            "ice_concentration": self.ice_concentration,
            "p_fluctuation_rms_max": float(result.pressure_fluctuation_rms_max),
        }

    def inception_scan(
        self,
        sigma_bracket: tuple[float, float],
        n_scan: int = 9,
        threshold_factor: float = 2.0,
    ) -> dict:
        """Inception number by scanning the sigma axis (robust, cheap).

        The barotropic cavity has a sharp bifurcation at inception, so the
        crossing of ``alpha_max = threshold_factor * alpha_nuc`` is located by
        scanning from the cavitating side and interpolating ``log(alpha_max)``
        linearly in sigma.  A bisection is deliberately avoided: deep-cavity
        states are very stiff and would dominate the run time.
        """
        s_lo, s_hi = sigma_bracket
        sigmas = np.linspace(s_lo, s_hi, n_scan)
        alphas = []
        records = []
        for sigma in sigmas:
            result = self.march(sigma_of_inlet=float(sigma))
            alphas.append(result.alpha_max)
            records.append((float(sigma), float(result.alpha_max), float(result.p_min)))
        alphas = np.asarray(alphas)
        threshold = threshold_factor * self.cav.alpha_nuc
        above = np.flatnonzero(alphas > threshold)
        if above.size == 0:
            return {
                "sigma": float("nan"),
                "records": records,
                "note": "no cavitation within the scanned band",
                "threshold": float(threshold),
            }
        k = int(above.min())        # lowest sigma that cavitates
        if k == 0:
            sigma_i = float(sigmas[0])
        else:
            x0, x1 = sigmas[k - 1], sigmas[k]
            y0 = max(alphas[k - 1], 0.5 * self.cav.alpha_nuc)
            y1 = alphas[k]
            t = (np.log(threshold) - np.log(y0)) / (np.log(y1) - np.log(y0))
            sigma_i = float(x0 + np.clip(t, 0.0, 1.0) * (x1 - x0))
        return {
            "sigma": sigma_i,
            "records": records,
            "threshold": float(threshold),
            "alpha_at_crossing": float(np.interp(sigma_i, sigmas, alphas)),
            "p_min_at_crossing": float(np.interp(sigma_i, sigmas, [r[2] for r in records])),
        }

    def sigma_sweep(
        self, sigmas: list[float], **kwargs
    ) -> list[SteadyResult]:
        return [self.march(sigma_of_inlet=s, **kwargs) for s in sigmas]