"""Interface energy-balance ODE for underwater attached ice growth.

Physical model
==============
The ice layer grows by freezing of frazil particles that impact and stick
to the surface, and by conductive freezing of the water film at the ice-water
interface. The governing energy balance at the interface is::

    rho_ice * L_f * dh/dt = q_cond - q_conv + q_frazil

where:

* ``q_cond = k_ice * (T_f - T_surface) / h_ice`` is the conductive heat
  flux through the ice layer (positive when heat flows out of the interface).
* ``q_conv = h_c * (T_water - T_f)`` is the convective heat flux from the
  water to the interface (positive when water is warmer than freezing).
* ``q_frazil = eta_stick * J_mass * L_f`` is the latent heat release from
  frazil particles that stick to the surface.

The convective heat transfer coefficient is estimated from the friction
velocity using a Reynolds analogy::

    h_c = rho_water * C_p * u_star / Pr_t

where ``Pr_t`` is the turbulent Prandtl number (~1 for water).

Stefan limit
------------
When frazil flux is zero and convection is negligible (large undercooling),
the ODE reduces to the classical Stefan problem::

    h(t) = sqrt(2 * k_ice * dT * t / (rho_ice * L_f))

This analytic solution is used as a verification benchmark (C2: <3% error).

Negative control
----------------
When the sticking efficiency ``eta_stick = 0`` and there is no conductive
growth (``dT = 0``), the ice thickness must remain identically zero
(C3: <1e-15).

Detachment limiting
-------------------
During integration, if the wall shear stress exceeds the adhesive strength
of the ice-surface bond, the ice layer is stripped and growth resets to zero.
This ensures the monotonicity property (C4): higher velocity → lower steady
state thickness.

Numerical integration
---------------------
The ODE is integrated using forward Euler (numpy-only). The integration
stops or resets when detachment occurs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import (
    RHO_ICE, RHO_WATER, K_ICE, L_FUSION, C_P_WATER, T_FREEZE_FW,
    DEFAULT_STICKING_EFFICIENCY, DEFAULT_BOND_STRENGTH_PA,
)
from .flux import frazil_impact_flux, FrazilFluxState
from .detach import wall_shear_stress, adhesive_strength, H_REF_M

#: Reference skin-friction coefficient relating u_star to the mean flow speed
CF_REF = 1.5e-3
#: Prandtl number of the water film in the Colburn analogy (water at 0 degC)
PR_T = 13.0


@dataclass(frozen=True)
class GrowthResult:
    """Result of ice growth integration."""

    #: Time array [s]
    t: np.ndarray
    #: Ice thickness array [m]
    h_ice: np.ndarray
    #: Energy residual array [W/m^2] (should be ~0 at each step)
    energy_residual: np.ndarray
    #: Final thickness [m]
    h_final: float
    #: Time to steady state or detachment [s] (inf if not reached)
    t_steady: float
    #: Whether detachment occurred during integration
    detached: bool
    #: Time of first detachment [s] (inf if never)
    t_detach: float

    def as_dict(self) -> dict:
        return {
            "t_s": self.t.tolist(),
            "h_ice_m": self.h_ice.tolist(),
            "h_final_m": self.h_final,
            "t_steady_s": self.t_steady,
            "detached": self.detached,
            "t_detach_s": self.t_detach,
        }


def stefan_analytic(
    undercooling_k: float,
    time_s: float,
    k_ice: float = K_ICE,
    rho_ice: float = RHO_ICE,
    l_fusion: float = L_FUSION,
) -> float:
    """Analytic Stefan solution for ice thickness [m].

    ``h(t) = sqrt(2 * k * dT * t / (rho * L))``

    Parameters
    ----------
    undercooling_k : float
        Temperature difference T_freeze - T_surface [K].
    time_s : float
        Time [s].
    k_ice, rho_ice, l_fusion : float
        Physical constants.

    Returns
    -------
    float
        Ice thickness [m].
    """
    dT = float(undercooling_k)
    t = float(time_s)
    if dT < 0.0 or t < 0.0:
        raise ValueError("undercooling and time must be non-negative")
    if dT == 0.0 or t == 0.0:
        return 0.0
    return float(np.sqrt(2.0 * k_ice * dT * t / (rho_ice * l_fusion)))


def energy_residual(
    h_ice: float,
    dh_dt: float,
    undercooling_k: float,
    convective_flux: float,
    frazil_flux_kg_m2s: float,
    eta_stick: float,
    k_ice: float = K_ICE,
    rho_ice: float = RHO_ICE,
    l_fusion: float = L_FUSION,
) -> float:
    """Compute the energy balance residual [W/m^2].

    Residual = rho*L*dh/dt - (q_cond - q_conv + q_frazil)
    Should be ~0 when the ODE is satisfied.

    Parameters
    ----------
    h_ice : float
        Current ice thickness [m].
    dh_dt : float
        Growth rate [m/s].
    undercooling_k : float
        T_freeze - T_surface [K].
    convective_flux : float
        Convective heat flux from water [W/m^2].
    frazil_flux_kg_m2s : float
        Frazil mass flux [kg/(m^2*s)].
    eta_stick : float
        Sticking efficiency.
    k_ice, rho_ice, l_fusion : float
        Physical constants.

    Returns
    -------
    float
        Energy residual [W/m^2].
    """
    if h_ice <= 0.0:
        q_cond = 0.0
    else:
        q_cond = k_ice * undercooling_k / h_ice

    q_frazil = eta_stick * frazil_flux_kg_m2s * l_fusion

    lhs = rho_ice * l_fusion * dh_dt
    rhs = q_cond - convective_flux + q_frazil

    return float(lhs - rhs)


class IceGrowthODE:
    """Ice growth ODE: dh/dt = f(h, params).

    The ODE right-hand side is::

        dh/dt = (q_cond - q_conv + q_frazil) / (rho_ice * L_f)

    where q_cond depends on h (nonlinear), making this a nonlinear ODE.

    Detachment is checked at each step: if wall shear stress exceeds
    adhesive strength, the ice layer is stripped (h reset to 0).
    """

    def __init__(
        self,
        undercooling_k: float,
        u_star: float,
        c_f: float,
        eta_stick: float = DEFAULT_STICKING_EFFICIENCY,
        t_water_k: Optional[float] = None,
        particle_diameter_m: float = 1.0e-4,
        k_ice: float = K_ICE,
        rho_ice: float = RHO_ICE,
        l_fusion: float = L_FUSION,
        c_p_water: float = C_P_WATER,
        rho_water: float = RHO_WATER,
        tau_bond: float = DEFAULT_BOND_STRENGTH_PA,
        cf_ref: float = CF_REF,
        pr_t: float = PR_T,
    ):
        """Initialise the growth ODE.

        Parameters
        ----------
        undercooling_k : float
            T_freeze - T_surface [K].
        u_star : float
            Friction velocity [m/s].
        c_f : float
            Frazil volume concentration.
        eta_stick : float
            Sticking efficiency (0 to 1).
        t_water_k : float, optional
            Water temperature [K]. If None, assumed = T_freeze.
        particle_diameter_m : float
            Frazil particle diameter [m].
        k_ice, rho_ice, l_fusion, c_p_water, rho_water : float
            Physical constants.
        tau_bond : float
            Bond strength [Pa] for detachment check.
        cf_ref : float
            Reference skin-friction coefficient used to relate the friction
            velocity to the mean-flow speed, ``u_star = U sqrt(cf_ref/2)``.
        pr_t : float
            Molecular/turbulent Prandtl number of the water film (default 13,
            water at 0 degC) in the Colburn analogy ``St = (cf/2) Pr**(-2/3)``.
        """
        self.undercooling_k = float(undercooling_k)
        self.u_star = float(u_star)
        self.c_f = float(c_f)
        self.eta_stick = float(eta_stick)
        self.t_water_k = t_water_k if t_water_k is not None else T_FREEZE_FW
        self.particle_diameter_m = float(particle_diameter_m)
        self.k_ice = float(k_ice)
        self.rho_ice = float(rho_ice)
        self.l_fusion = float(l_fusion)
        self.c_p_water = float(c_p_water)
        self.rho_water = float(rho_water)
        self.tau_bond = float(tau_bond)

        # Pre-compute frazil flux
        self._flux_state = frazil_impact_flux(
            c_f=self.c_f,
            u_star=self.u_star,
            particle_diameter_m=self.particle_diameter_m,
        )
        self._frazil_mass_flux = self._flux_state.mass_flux

        # Convective heat transfer coefficient.
        #
        # Colburn/Reynolds analogy on the wall shear:  St = (cf/2) * Pr**(-2/3)
        # with  h_c = rho Cp U St  and  U = u_star / sqrt(cf/2), hence
        #     h_c = rho Cp u_star sqrt(cf/2) Pr**(-2/3).
        # Before 2026-11-10 the module used h_c = rho Cp u_star, i.e. it equated
        # the friction velocity with the mean-flow speed inside the Stanton
        # number (St = u_star/U = sqrt(cf/2) instead of cf/2).  That
        # over-predicted h_c — and therefore the melting flux — by
        # 1/[sqrt(cf/2) Pr**(-2/3)] ~ 2e2, which made any accretion at ship
        # speeds impossible.  See artifacts/attached_ice_maturity/.
        self.cf_ref = float(cf_ref)
        self.pr_t = float(pr_t)
        self._h_c = (
            self.rho_water * self.c_p_water * self.u_star
            * np.sqrt(self.cf_ref / 2.0) * self.pr_t ** (-2.0 / 3.0)
        )

        # Convective flux (positive when water is warmer than freezing)
        delta_t_conv = self.t_water_k - T_FREEZE_FW
        self._convective_flux = self._h_c * delta_t_conv

        # Wall shear stress (constant for given u_star)
        self._tau_wall = wall_shear_stress(self.u_star, self.rho_water)

    def _check_detachment(self, h_ice: float) -> bool:
        """Check if ice detaches at given thickness."""
        tau_adh = adhesive_strength(h_ice, self.tau_bond)
        return self._tau_wall > tau_adh

    def rhs(self, h_ice: float) -> float:
        """Right-hand side of the growth ODE: dh/dt [m/s].

        Parameters
        ----------
        h_ice : float
            Current ice thickness [m].

        Returns
        -------
        float
            Growth rate [m/s].
        """
        h = float(h_ice)
        if h < 0.0:
            raise ValueError("ice thickness must be non-negative")

        # Check detachment first
        if self._check_detachment(h):
            return 0.0  # no growth when detached

        # Conductive flux through ice
        if h <= 1e-15:
            q_cond = 0.0
        else:
            q_cond = self.k_ice * self.undercooling_k / h

        # Frazil contribution
        q_frazil = self.eta_stick * self._frazil_mass_flux * self.l_fusion

        # Net energy flux
        q_net = q_cond - self._convective_flux + q_frazil

        return float(q_net / (self.rho_ice * self.l_fusion))

    def solve_euler(
        self,
        t_max: float,
        n_steps: int = 1000,
        h_init: float = 0.0,
        method: str = "euler",
    ) -> GrowthResult:
        """Integrate the growth ODE (numpy-only, no scipy).

        Parameters
        ----------
        t_max : float
            Maximum integration time [s].
        n_steps : int
            Number of time steps.
        h_init : float
            Initial ice thickness [m].
        method : {"euler", "stefan_split"}
            ``euler`` — classic explicit forward Euler on the full right-hand
            side (first order; the ``1/h`` conduction term needs many steps).
            ``stefan_split`` — operator split: the singular conduction term is
            advanced by its *exact* quadrature ``h**2 -> h**2 + 2 k dT dt/(rho L)``
            while the (bounded) frazil and convective source terms are advanced
            explicitly.  Exact for the pure Stefan problem at any step size, so
            the C2 benchmark measures only the source-term splitting error.

        Returns
        -------
        GrowthResult
        """
        if method not in ("euler", "stefan_split"):
            raise ValueError(f"unknown integration method {method!r}")
        t = np.linspace(0.0, float(t_max), int(n_steps) + 1)
        dt = t[1] - t[0]
        h = np.zeros_like(t)
        h[0] = float(h_init)
        res = np.zeros_like(t)
        detached = False
        t_detach = float("inf")

        for i in range(n_steps):
            # Check detachment before growth
            if self._check_detachment(h[i]) and h[i] > 1e-15:
                h[i + 1] = 0.0  # strip ice
                if not detached:
                    detached = True
                    t_detach = float(t[i])
                res[i] = 0.0
                continue

            if method == "euler":
                dh = self.rhs(h[i]) * dt
                h_next = h[i] + dh
            else:
                # exact advance of the singular Stefan (conduction) term ...
                src = (self.eta_stick * self._frazil_mass_flux * self.l_fusion
                       - self._convective_flux) / (self.rho_ice * self.l_fusion)
                arg = h[i] ** 2 + 2.0 * self.k_ice * self.undercooling_k * dt / (
                    self.rho_ice * self.l_fusion)
                h_cond = float(np.sqrt(max(arg, 0.0)))
                h_next = h_cond + src * dt
            h[i + 1] = max(h_next, 0.0)  # enforce non-negativity

            # Compute energy residual at this step
            if h[i] > 1e-15:
                dh_dt = (h[i + 1] - h[i]) / dt if dt > 0 else 0.0
                res[i] = energy_residual(
                    h_ice=h[i],
                    dh_dt=dh_dt,
                    undercooling_k=self.undercooling_k,
                    convective_flux=self._convective_flux,
                    frazil_flux_kg_m2s=self._frazil_mass_flux,
                    eta_stick=self.eta_stick,
                    k_ice=self.k_ice,
                    rho_ice=self.rho_ice,
                    l_fusion=self.l_fusion,
                )
            else:
                res[i] = 0.0

        # Final residual
        if h[-1] > 1e-15:
            dh_dt = self.rhs(h[-1])
            res[-1] = energy_residual(
                h_ice=h[-1],
                dh_dt=dh_dt,
                undercooling_k=self.undercooling_k,
                convective_flux=self._convective_flux,
                frazil_flux_kg_m2s=self._frazil_mass_flux,
                eta_stick=self.eta_stick,
                k_ice=self.k_ice,
                rho_ice=self.rho_ice,
                l_fusion=self.l_fusion,
            )

        # Estimate steady-state time
        initial_rate = self.rhs(1e-10) if self.rhs(1e-10) > 0 else self.rhs(1e-6)
        if initial_rate > 0:
            threshold = 0.01 * initial_rate
            rates = np.array([self.rhs(hi) for hi in h])
            steady_idx = np.where(rates <= threshold)[0]
            t_steady = float(t[steady_idx[0]]) if len(steady_idx) > 0 else float(t_max)
        else:
            t_steady = 0.0

        return GrowthResult(
            t=t,
            h_ice=h,
            energy_residual=res,
            h_final=float(h[-1]),
            t_steady=t_steady,
            detached=detached,
            t_detach=t_detach,
        )


    def energy_closure(self, result: GrowthResult, skip_steps: int = 1) -> dict:
        """Integral energy-balance closure of an integration result.

        Criterion C1 of the maturity spec asks for the *interface energy
        residual* to stay below 2%.  A per-step residual recomputed with the very
        same flux expressions the right-hand side uses is identically zero and
        proves nothing, so the closure is measured on the integral form

            rho_ice * L_f * (h(T) - h(0)) = int_0^T (q_cond(t) - q_conv + q_frazil) dt

        with ``q_cond = k_ice * dT / h(t)`` taken from the returned trajectory and
        the integral evaluated by trapezoidal quadrature.  The residual is a
        genuine discretisation-error measure (it vanishes only if the scheme
        conserves latent heat).

        Parameters
        ----------
        result : GrowthResult
            Integration output of :meth:`solve_euler`.
        skip_steps : int
            Number of leading steps excluded from the integral.  The very first
            step carries the singular ``1/h`` nucleation jump from ``h_init``;
            trapezoidal quadrature of ``1/h(t)`` there is invalid (it reflects
            the kink of the initial condition, not the scheme), so the closure
            is measured on the regular part and the nucleation step is reported
            separately in ``relative_error_including_nucleation``.
            Starting from a finite seed ``h_init`` the skipped part accounts for
            exactly ``rho L (h[skip] - h[0])``, which is subtracted below.

        Returns
        -------
        dict
            ``latent_heat_J_m2``, ``flux_integral_J_m2``,
            ``relative_error`` (|difference| / max of the two),
            ``relative_error_including_nucleation`` and
            ``max_step_relative_residual``.
        """
        t = np.asarray(result.t, dtype=float)
        h = np.asarray(result.h_ice, dtype=float)
        h_safe = np.maximum(h, 1e-12)
        q_cond = np.where(h > 1e-12, self.k_ice * self.undercooling_k / h_safe, 0.0)
        q = (q_cond - self._convective_flux
             + self.eta_stick * self._frazil_mass_flux * self.l_fusion)
        k = int(max(0, min(skip_steps, len(t) - 1)))
        flux_integral = float(np.trapezoid(q[k:], t[k:]))
        latent = float(self.rho_ice * self.l_fusion * (h[-1] - h[k]))
        scale = max(abs(latent), abs(flux_integral), 1e-30)
        flux_integral_full = float(np.trapezoid(q, t))
        latent_full = float(self.rho_ice * self.l_fusion * (h[-1] - h[0]))
        scale_full = max(abs(latent_full), abs(flux_integral_full), 1e-30)
        # per-step trapezoidal residual, normalised by the local flux magnitude
        q_scale = max(float(np.max(np.abs(q))), 1e-30)
        step_res = np.abs(
            self.rho_ice * self.l_fusion * np.diff(h)
            - 0.5 * (q[1:] + q[:-1]) * np.diff(t)
        ) / q_scale
        return {
            "latent_heat_J_m2": latent,
            "flux_integral_J_m2": flux_integral,
            "relative_error": abs(latent - flux_integral) / scale,
            "relative_error_including_nucleation": abs(latent_full - flux_integral_full) / scale_full,
            "max_step_relative_residual": float(np.max(step_res)) if step_res.size else 0.0,
        }


def stefan_with_initial(
    initial_thickness_m: float,
    undercooling_k: float,
    time_s: float,
    k_ice: float = K_ICE,
    rho_ice: float = RHO_ICE,
    l_fusion: float = L_FUSION,
) -> float:
    """Exact solution of the pure-conduction Stefan ODE from a finite seed.

    ``dh/dt = k_ice * dT / (rho_ice * L_f * h)`` has the closed-form solution

        h(t) = sqrt( h0**2 + 2 * k_ice * dT * t / (rho_ice * L_f) )

    which reduces to :func:`stefan_analytic` for ``h0 = 0``.  This is the exact
    reference used for the C2 verification of the numerical integrators: the
    discretisation error can therefore be measured without any model
    approximation leaking into the benchmark.

    Parameters
    ----------
    initial_thickness_m : float
        Seed thickness ``h0`` [m] (>= 0).
    undercooling_k : float
        T_freeze - T_surface [K].
    time_s : float
        Elapsed time [s].

    Returns
    -------
    float
        Ice thickness [m].
    """
    h0 = float(initial_thickness_m)
    dT = float(undercooling_k)
    t = float(time_s)
    if h0 < 0.0:
        raise ValueError("initial thickness must be non-negative")
    if t < 0.0:
        raise ValueError("time must be non-negative")
    arg = h0 * h0 + 2.0 * float(k_ice) * dT * t / (float(rho_ice) * float(l_fusion))
    return float(np.sqrt(max(arg, 0.0)))


def solve_growth(
    undercooling_k: float,
    u_star: float,
    c_f: float,
    t_max: float,
    eta_stick: float = DEFAULT_STICKING_EFFICIENCY,
    n_steps: int = 1000,
    method: str = "euler",
    **kwargs,
) -> GrowthResult:
    """Convenience function to solve the ice growth ODE.

    Parameters
    ----------
    undercooling_k : float
        T_freeze - T_surface [K].
    u_star : float
        Friction velocity [m/s].
    c_f : float
        Frazil volume concentration.
    t_max : float
        Maximum integration time [s].
    eta_stick : float
        Sticking efficiency.
    n_steps : int
        Number of time steps.
    method : {"euler", "stefan_split"}
        Integrator, see :meth:`IceGrowthODE.solve_euler`.

    Returns
    -------
    GrowthResult
    """
    ode = IceGrowthODE(
        undercooling_k=undercooling_k,
        u_star=u_star,
        c_f=c_f,
        eta_stick=eta_stick,
        **kwargs,
    )
    return ode.solve_euler(t_max=t_max, n_steps=n_steps, method=method)


def equilibrium_thickness(
    undercooling_k: float,
    u_star: float,
    c_f: float = 0.0,
    eta_stick: float = DEFAULT_STICKING_EFFICIENCY,
    t_water_k: Optional[float] = None,
    tau_bond: float = DEFAULT_BOND_STRENGTH_PA,
    h_ref: float = H_REF_M,
    particle_diameter_m: float = 1.0e-4,
    k_ice: float = K_ICE,
    rho_ice: float = RHO_ICE,
    rho_water: float = RHO_WATER,
    c_p_water: float = C_P_WATER,
    l_fusion: float = L_FUSION,
) -> dict:
    """Closed-form steady-state thickness ``h_ss(u_star)`` of the attached layer.

    Two independent limits compete, and the steady state is the smaller of them:

    * **growth equilibrium** — the layer stops growing when the conductive heat
      flux ``k_ice dT / h`` balances the net sink ``q_conv - q_frazil``::

          h_growth = k_ice * dT / (q_conv - q_frazil),
          q_conv = rho_water c_p u_star (T_water - T_f),
          q_frazil = eta_stick * C_f rho_ice v_dep u_star-correlated * L_f

      i.e. the layer is in thermal equilibrium against melting.  If
      ``q_conv <= q_frazil`` no equilibrium exists (the layer grows without
      bound; the returned value is ``inf``) — a physical statement, not a
      numerical failure: with the water at the freezing point and a persistent
      frazil supply nothing melts the layer.

    * **detachment limit** — the largest thickness whose adhesive strength still
      carries the wall shear stress, from inversion of
      ``tau_bond / (1 + h/h_ref) = rho_water u_star**2``::

          h_detach = h_ref * (tau_bond / (rho_water u_star**2) - 1)

    Both limits are strictly decreasing functions of ``u_star`` (``q_conv`` and
    ``q_frazil`` grow linearly with ``u_star``; ``tau_wall`` grows
    quadratically), therefore ``h_ss = min(h_growth, h_detach)`` is monotone
    decreasing — the property asserted by criterion C4.  Unlike the sawtooth
    produced by stripping ice inside a finite integration window, ``h_ss`` is a
    well-defined single-valued function of the forcing.

    Parameters
    ----------
    undercooling_k : float
        T_freeze - T_surface [K].
    u_star : float
        Friction velocity [m/s].
    c_f : float
        Frazil volume concentration.
    eta_stick : float
        Sticking efficiency.
    t_water_k : float, optional
        Bulk water temperature [K]; default = freezing point (no melting).
    tau_bond : float
        Bond strength [Pa] (from the adfreeze/Frederking model).
    h_ref : float
        Adhesive-strength reference thickness [m].
    Others : physical constants.

    Returns
    -------
    dict
        ``h_growth_m``, ``h_detach_m``, ``h_eq_m``, ``equilibrium_exists`` and
        ``limited_by`` in {"growth", "detachment", "none"}.  When no melting sink
        removes the frazil input (``q_conv <= q_frazil``) the growth limit is
        infinite, ``equilibrium_exists`` is False, ``h_eq_m`` is ``inf`` and
        ``limited_by`` is ``"none"`` — the detachment cap is reported separately
        in ``h_detach_m`` but is not promoted to a steady state.
    """
    u_s = float(u_star)
    if u_s < 0.0:
        raise ValueError("friction velocity must be non-negative")
    ode = IceGrowthODE(
        undercooling_k=undercooling_k,
        u_star=u_s,
        c_f=c_f,
        eta_stick=eta_stick,
        t_water_k=t_water_k,
        particle_diameter_m=particle_diameter_m,
        k_ice=k_ice,
        rho_ice=rho_ice,
        l_fusion=l_fusion,
        c_p_water=c_p_water,
        rho_water=rho_water,
        tau_bond=tau_bond,
    )
    q_conv = ode._convective_flux
    q_frazil = ode.eta_stick * ode._frazil_mass_flux * l_fusion
    sink = q_conv - q_frazil
    if sink > 0.0:
        h_growth = float(k_ice * float(undercooling_k) / sink)
    else:
        h_growth = float("inf")

    tau_wall = wall_shear_stress(u_s, rho_water)
    if tau_wall <= 0.0:
        h_detach = float("inf")
    else:
        ratio = float(tau_bond) / tau_wall
        h_detach = float(h_ref * (ratio - 1.0)) if ratio > 1.0 else 0.0

    # An *equilibrium* only exists if a melting sink removes the frazil latent
    # input; otherwise the layer grows without bound and the detachment cap is a
    # degenerate (km-scale) artefact that must not be reported as a steady state.
    equilibrium_exists = bool(np.isfinite(h_growth))
    if equilibrium_exists:
        h_eq = min(h_growth, h_detach)
        limited_by = "detachment" if h_detach < h_growth else "growth"
    else:
        h_eq = float("inf")
        limited_by = "none"

    return {
        "u_star_m_s": u_s,
        "tau_wall_Pa": tau_wall,
        "q_conv_W_m2": q_conv,
        "q_frazil_W_m2": q_frazil,
        "h_growth_m": h_growth,
        "h_detach_m": h_detach,
        "h_eq_m": h_eq,
        "equilibrium_exists": equilibrium_exists,
        "limited_by": limited_by,
    }
