"""Melt-film thickness evolution: mass + energy balance with three mechanisms.

Governing ODE
-------------
The film is thin enough to be isothermal at the *local* melting temperature
``T_m(p)``, so the energy balance reduces to a latent-heat bookkeeping of the
ice/water interface plus a plug-flow drainage of meltwater out of the contact:

    dh/dt = [ q_fric + q_ice + q_struct ] / (rho_ice L_f)  -  c_d h V / L

    q_fric   = mu(h, p, V, T) p V                       friction dissipation
    q_ice    = k_ice (T_ice - T_m(p)) / sqrt(alpha_ice L / V)
    q_struct = (T_struct - T_m(p)) / (h / k_water + sqrt(alpha_s L / V) / k_s)

Signs are chosen so that a positive flux *melts* ice and feeds the film:

* ``q_fric`` is always positive -- frictional heating;
* ``q_ice`` is negative for ice colder than its melting point, i.e. heat leaks
  into the ice and the film *refreezes* (this is the Stefan condition, and the
  ``sqrt(alpha L / V)`` length is the thermal penetration depth during one
  contact residence);
* ``q_struct`` is the same Stefan condition on the structural side, written as
  two resistances in series: the film itself (``h/k_water``) and the
  semi-infinite structure (``delta_s / k_s``).  A cold structure therefore
  refreezes the film, and a thick film insulates itself from that sink.

Pressure melting enters only through ``T_m(p)`` (Clausius-Clapeyron): the melting
point of the interface is depressed, which both weakens the refreezing sinks and,
above a threshold pressure, makes otherwise cold ice superheated so that it
melts with no frictional heating at all.

Drainage ``c_d h V / L`` is the mass balance: meltwater produced inside a contact
of length ``L`` must be expelled at the trailing edge, where the ice surface
recedes past at speed ``V``.  ``c_d`` is an O(1) squeeze/drag coefficient.

Analytic steady state
---------------------
Setting ``dh/dt = 0`` with constant ``mu`` gives a quadratic in ``h``,

    (a/k_w) h^2 + (a R_s - Q/k_w) h - (Q R_s + D) = 0,
    a = c_d V rho_ice L_f / L,  Q = q_fric + q_ice,
    R_s = delta_s / k_s,        D = T_struct - T_m(p),

whose positive root :meth:`FilmModel.steady_state_analytic` evaluates in closed
form.  :meth:`FilmModel.integrate` and :meth:`FilmModel.steady_state` are
independent of that formula, so the comparison is a genuine verification
(criterion (b)).  When no positive root exists the energy balance cannot sustain
a film: the model then reports the dry/boundary branch (``h -> 0``).

Limitations
-----------
Single-node (0-D) film, no lateral flow or Reynolds equation, no ice
deformation, no asperity-scale pressure.  The structure-side sink uses a
semi-infinite penetration depth, so the result depends on the assumed structure
surface temperature and on the thermal history of the slider (a slider that has
already warmed to the melting point removes that sink entirely; both cases are
scanned in the verification harness).
"""

from dataclasses import dataclass, field
from math import sqrt

from .adhesion import IceAdhesionModel
from .pressure_melting import melt_temperature, melt_point_depression
from .properties import IceWaterProperties, StructureProperties
from .tribology import (
    StribeckParams,
    friction_coefficient,
    lubrication_regime,
)

__all__ = ["ContactState", "FilmModel"]


@dataclass(frozen=True)
class ContactState:
    """Kinematic and thermal state of the ice/structure contact."""

    pressure: float                 # Pa, nominal contact pressure
    velocity: float                 # m/s, sliding speed
    ice_temperature: float          # K
    structure_temperature: float    # K
    contact_length: float = 0.1     # m, length of the contact along sliding

    def __post_init__(self) -> None:
        if self.pressure <= 0.0:
            raise ValueError("pressure must be positive")
        if self.velocity < 0.0:
            raise ValueError("velocity must be non-negative")
        if self.ice_temperature <= 0.0 or self.structure_temperature <= 0.0:
            raise ValueError("temperatures must be positive absolute temperatures")
        if self.contact_length <= 0.0:
            raise ValueError("contact_length must be positive")


@dataclass(frozen=True)
class FilmModel:
    """Coupled thermodynamics/tribology model of the melt film."""

    props: IceWaterProperties = field(default_factory=IceWaterProperties)
    structure: StructureProperties = field(default_factory=StructureProperties)
    tribology: StribeckParams = field(default_factory=StribeckParams)
    adhesion: IceAdhesionModel | None = None
    drainage_coefficient: float = 1.0
    film_floor: float = 1.0e-9        # m, numerical floor / dry-contact scale
    minimum_velocity: float = 1.0e-6  # m/s, guards the penetration-depth scale

    def __post_init__(self) -> None:
        if self.drainage_coefficient <= 0.0:
            raise ValueError("drainage_coefficient must be positive")
        if self.film_floor <= 0.0:
            raise ValueError("film_floor must be positive")
        if self.minimum_velocity <= 0.0:
            raise ValueError("minimum_velocity must be positive")

    # -- kinematics ---------------------------------------------------------
    def penetration_depths(self, state: ContactState) -> tuple[float, float]:
        """Thermal penetration depths in ice and structure [m]."""
        velocity = max(state.velocity, self.minimum_velocity)
        residence = state.contact_length / velocity
        return sqrt(self.props.alpha_ice * residence), sqrt(self.structure.alpha * residence)

    def melting_temperature(self, state: ContactState) -> float:
        """Local melting temperature [K] at the contact pressure."""
        return melt_temperature(state.pressure, self.props)

    # -- tribology ----------------------------------------------------------
    def friction_coefficient(self, film_thickness: float, state: ContactState) -> float:
        """Extended-Stribeck friction coefficient at the current film thickness."""
        # adhesion strength is a property of the supercooling of the interface
        # material, so it is evaluated at the ice temperature, not at the
        # (pressure-depressed) melting point of the film
        return friction_coefficient(
            film_thickness,
            state.pressure,
            state.velocity,
            state.ice_temperature,
            params=self.tribology,
            props=self.props,
            adhesion_model=self.adhesion,
        )

    def regime(self, film_thickness: float) -> str:
        """Lubrication regime label for the current film thickness."""
        return lubrication_regime(film_thickness, self.tribology)

    # -- heat and mass fluxes ----------------------------------------------
    def fluxes(self, film_thickness: float, state: ContactState) -> dict:
        """Heat fluxes [W/m^2] and the mass-balance drain [m/s]."""
        h = max(float(film_thickness), self.film_floor)
        t_melt = self.melting_temperature(state)
        delta_ice, delta_structure = self.penetration_depths(state)
        mu = self.friction_coefficient(h, state)
        q_friction = mu * state.pressure * state.velocity
        q_ice = self.props.k_ice * (state.ice_temperature - t_melt) / delta_ice
        resistance = h / self.props.k_water + delta_structure / self.structure.k
        q_structure = (state.structure_temperature - t_melt) / resistance
        drain = self.drainage_coefficient * h * state.velocity / state.contact_length
        return {
            "film_thickness": h,
            "friction_coefficient": mu,
            "melting_temperature": t_melt,
            "delta_ice": delta_ice,
            "delta_structure": delta_structure,
            "q_friction": q_friction,
            "q_ice": q_ice,
            "q_structure": q_structure,
            "drain_rate": drain,
            "latent_scale": self.props.latent_heat_per_volume_ice,
        }

    def rhs(self, film_thickness: float, state: ContactState) -> float:
        """Right-hand side of the film-thickness ODE [m/s]."""
        f = self.fluxes(film_thickness, state)
        growth = (f["q_friction"] + f["q_ice"] + f["q_structure"]) / f["latent_scale"]
        return growth - f["drain_rate"]

    def decompose(self, film_thickness: float, state: ContactState) -> dict:
        """Rate decomposition [m/s] into the three competing mechanisms.

        ``drain_rate`` closes the mass balance, so
        ``dh/dt = friction_melting + pressure_melting + refreezing - drain_rate``.
        """
        f = self.fluxes(film_thickness, state)
        latent = f["latent_scale"]
        t_melt = f["melting_temperature"]
        t_melt_ref = self.props.melting_temperature
        reference_flux = self.props.k_ice * (state.ice_temperature - t_melt_ref) / f["delta_ice"]
        friction_melting = f["q_friction"] / latent
        pressure_melting = (f["q_ice"] - reference_flux) / latent
        passive = (reference_flux + f["q_structure"]) / latent
        refreezing = max(0.0, -passive)
        return {
            "friction_melting": friction_melting,
            "pressure_melting": pressure_melting,
            "refreezing": refreezing,
            "passive_melting": max(0.0, passive),
            "drain_rate": f["drain_rate"],
            "dh_dt": friction_melting + pressure_melting + passive - f["drain_rate"],
            "melting_temperature_depression": melt_point_depression(state.pressure, self.props),
        }

    # -- steady states ------------------------------------------------------
    def steady_state_analytic(
        self, state: ContactState, friction_coefficient_value: float | None = None
    ) -> float | None:
        """Closed-form steady film thickness [m], or ``None`` if no film exists.

        Positive root of ``(a/k_w) h^2 + (a R_s - Q/k_w) h - (Q R_s + D) = 0``.
        With ``friction_coefficient_value`` omitted the boundary value is used.
        """
        mu = (
            self.tribology.mu_boundary
            if friction_coefficient_value is None
            else float(friction_coefficient_value)
        )
        t_melt = self.melting_temperature(state)
        delta_ice, delta_structure = self.penetration_depths(state)
        q_friction = mu * state.pressure * state.velocity
        q_ice = self.props.k_ice * (state.ice_temperature - t_melt) / delta_ice
        q_total = q_friction + q_ice
        resistance_structure = delta_structure / self.structure.k
        driving = state.structure_temperature - t_melt
        a = (
            self.drainage_coefficient
            * state.velocity
            * self.props.latent_heat_per_volume_ice
            / state.contact_length
        )
        if a <= 0.0:
            return None
        quad_a = a / self.props.k_water
        quad_b = a * resistance_structure - q_total / self.props.k_water
        quad_c = -(q_total * resistance_structure + driving)
        discriminant = quad_b * quad_b - 4.0 * quad_a * quad_c
        if discriminant < 0.0:
            return None
        root = (-quad_b + sqrt(discriminant)) / (2.0 * quad_a)
        if root <= 0.0 or root != root:  # NaN guard
            return None
        return root

    def steady_state(
        self, state: ContactState, max_iterations: int = 80, tolerance: float = 1e-12
    ) -> dict:
        """Self-consistent steady state of the ODE, independent of the closed form.

        The friction coefficient depends on the film thickness, so the fixed point
        of ``rhs(h) = 0`` is found by a bracketing search on ``h`` (bisection),
        entirely without the analytic quadratic.  ``mu`` is evaluated at each trial
        thickness, so the result is the coupled tribological steady state.
        """
        def residual(h: float) -> float:
            return self.rhs(h, state)

        h_lo = self.film_floor
        if residual(h_lo) <= 0.0:
            return {
                "exists": False,
                "film_thickness": 0.0,
                "friction_coefficient": self.friction_coefficient(h_lo, state),
                "regime": "boundary",
                "dh_dt_at_floor": residual(h_lo),
            }
        h_hi = self.film_floor
        for _ in range(max_iterations):
            h_hi *= 2.0
            if residual(h_hi) <= 0.0:
                break
            if h_hi > 1.0:  # pragma: no cover - unphysical runaway
                return {
                    "exists": False,
                    "film_thickness": 0.0,
                    "friction_coefficient": self.friction_coefficient(h_hi, state),
                    "regime": self.regime(h_hi),
                    "dh_dt_at_floor": residual(h_lo),
                }
        else:
            return {
                "exists": False,
                "film_thickness": 0.0,
                "friction_coefficient": self.friction_coefficient(h_lo, state),
                "regime": "boundary",
                "dh_dt_at_floor": residual(h_lo),
            }
        for _ in range(200):
            h_mid = 0.5 * (h_lo + h_hi)
            if residual(h_mid) > 0.0:
                h_lo = h_mid
            else:
                h_hi = h_mid
            if h_hi - h_lo <= tolerance * max(1.0, h_hi):
                break
        h_star = 0.5 * (h_lo + h_hi)
        return {
            "exists": True,
            "film_thickness": h_star,
            "friction_coefficient": self.friction_coefficient(h_star, state),
            "regime": self.regime(h_star),
            "dh_dt_at_floor": residual(self.film_floor),
        }

    # -- time integration ---------------------------------------------------
    def integrate(
        self,
        state: ContactState,
        initial_thickness: float = 1.0e-6,
        t_end: float = 1.0,
        n_samples: int = 201,
        tolerance: float = 1.0e-6,
    ) -> dict:
        """Integrate the film ODE with adaptive RK4 (step doubling).

        Returns time histories of film thickness, friction coefficient, regime and
        the three mechanism rates.  Step doubling gives a local error estimate, so
        the stiff relaxation onto the steady state and the (much faster) refreezing
        collapse are both handled without a hand-tuned time step.  The dry branch is
        latched when the film reaches ``film_floor`` while still thinning, so a
        "no persistent film" outcome is reported as such.
        """
        if initial_thickness <= 0.0:
            raise ValueError("initial_thickness must be positive")
        if t_end <= 0.0:
            raise ValueError("t_end must be positive")
        if not isinstance(n_samples, int) or isinstance(n_samples, bool) or n_samples < 2:
            raise ValueError("n_samples must be an integer >= 2")
        if not 0.0 < tolerance < 1.0:
            raise ValueError("tolerance must lie in (0, 1)")

        def advance(thickness: float, step: float) -> float:
            k1 = self.rhs(thickness, state)
            k2 = self.rhs(thickness + 0.5 * step * k1, state)
            k3 = self.rhs(thickness + 0.5 * step * k2, state)
            k4 = self.rhs(thickness + step * k3, state)
            return thickness + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        sample_times = [t_end * i / (n_samples - 1) for i in range(n_samples)]
        h = max(initial_thickness, self.film_floor)
        time = 0.0
        dry = False
        step = t_end * 1.0e-4
        history = {
            "time": [],
            "film_thickness": [],
            "friction_coefficient": [],
            "regime": [],
            "friction_melting": [],
            "pressure_melting": [],
            "refreezing": [],
            "drain_rate": [],
            "melting_temperature": [],
            "dry_branch": [],
        }
        max_internal_steps = 4_000_000
        internal_steps = 0
        for target in sample_times:
            while time < target - 1e-18:
                if dry and self.rhs(h, state) <= 0.0:
                    time = target
                    break
                trial = min(step, target - time)
                full = advance(h, trial)
                half = advance(advance(h, 0.5 * trial), 0.5 * trial)
                error = abs(full - half) / max(0.5 * (abs(full) + abs(half)), self.film_floor)
                if error <= tolerance:
                    h = half
                    time += trial
                    step = min(trial * 1.6, target - time if target - time > 0 else trial * 1.6)
                    step = max(step, 1.0e-16)
                    internal_steps += 1
                    if h <= self.film_floor:
                        h = self.film_floor
                        dry = True
                    if internal_steps > max_internal_steps:  # pragma: no cover
                        raise RuntimeError("film ODE integration exceeded its step budget")
                else:
                    step = trial * 0.5
                    if step < 1.0e-18:  # pragma: no cover - numerical guard
                        raise RuntimeError("film ODE step underflow")
            f = self.fluxes(h, state)
            d = self.decompose(h, state)
            history["time"].append(time)
            history["film_thickness"].append(h)
            history["friction_coefficient"].append(f["friction_coefficient"])
            history["regime"].append(self.regime(h))
            history["friction_melting"].append(d["friction_melting"])
            history["pressure_melting"].append(d["pressure_melting"])
            history["refreezing"].append(d["refreezing"])
            history["drain_rate"].append(d["drain_rate"])
            history["melting_temperature"].append(f["melting_temperature"])
            history["dry_branch"].append(dry)
        return history
