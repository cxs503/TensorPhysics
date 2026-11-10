"""Coupled drift-roughness-drag closure for a free-drifting ice cover.

Physics
-------
A sea-ice sheet of thickness ``h`` and concentration ``C`` drifting with velocity
``U_i`` in a water current ``U_w`` under wind ``U_a`` obeys, per unit area, the
momentum balance

    rho_i h dU_i/dt = tau_a + tau_w - rho_i h f (k x U_i) + rho_i h g grad(eta)   (1)

(the Coriolis term turns the drift to the right of the wind in the northern
hemisphere, f > 0)

with the surface (wind) stress

    tau_a = rho_a C_a |U_a - U_i| (U_a - U_i)                                    (2)

and the ice-ocean stress

    tau_w = -rho_w C_d_ice |U_i - U_w| (U_i - U_w) .                             (3)

The drag coefficient in (3) is *not* a constant: it follows from the under-ice
roughness length, which in turn depends on the keel morphology and -- through the
viscous sublayer term ``0.11 nu/u_star`` -- on the drift velocity itself:

    C_d_ice = F( z0_geom(keel depth, keel spacing), |U_i - U_w| )                (4)

(see :mod:`tensorphysics.ice_drift_roughness.z0_model` and
:mod:`tensorphysics.ice_drift_roughness.drag_coeff`).

Equations (1)-(4) form the *drift-roughness-drag closure*: for steady drift the
momentum balance is solved for ``U_i`` while (4) is solved for ``C_d_ice`` inside
every residual evaluation.  The outer nonlinear system is solved with Newton's
method on the two velocity components (numerical Jacobian, 2x2 direct solve), so
the drift-velocity increment of successive iterates falls below the acceptance
tolerance in a handful of iterations (criterion C3).

Diagnostics returned per solution: drift speed, drift/wind ratio (Nansen's rule
gives ~0.02), the turn angle between wind and drift (~30 deg to the right in the
northern hemisphere), the friction velocity, the roughness length and the
Reynolds regime.

Interfacing with other ice-drift models
--------------------------------------
:func:`solve_free_drift` accepts ``drift_provider`` -- a callable
``(h_ice, wind, current) -> (u_x, u_y)`` -- whose output is used as the initial
guess (and, with ``freeze_drift=True``, as the fixed drift velocity).  This is the
hook for an external ice-drift model (e.g. an ice-ocean momentum model or a
measured drift record): the module then only performs the roughness / drag
diagnosis, e.g. :func:`drift_roughness_diagnostics`.

Units: SI (m, s, kg, Pa, N/m^2 per unit area).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from .drag_coeff import (
    FrictionState,
    solve_ice_area_friction,
)
from .z0_model import (
    KAPPA,
    NU_SEAWATER,
    PARAMETER_PROVENANCE as _Z0_PROVENANCE,
    Z_REF,
    IceDriftRoughnessError,
    IceUndersideRoughness,
    KeelMorphology,
    ice_underside_roughness,
    surface_interface_roughness,
)

PARAMETER_PROVENANCE: Dict[str, str] = {
    **_Z0_PROVENANCE,
    "RHO_ICE": "917 kg/m^3, sea ice density at -5 C",
    "RHO_WATER": "1025 kg/m^3, seawater density at 0 C / 34 psu",
    "RHO_AIR": "1.30 kg/m^3, air density at 0 C, 1 atm",
    "C_AIR_ICE": "1.5e-3, neutral 10 m drag coefficient of a snow/ice surface",
    "OMEGA_EARTH": "7.292e-5 rad/s, Earth rotation rate",
}

RHO_ICE = 917.0
RHO_WATER = 1025.0
RHO_AIR = 1.30
C_AIR_ICE = 1.5e-3
OMEGA_EARTH = 7.292e-5
GRAVITY = 9.81

Vec2 = Tuple[float, float]


class ClosureError(IceDriftRoughnessError):
    """Raised when the drift closure cannot be solved."""


# ---------------------------------------------------------------------------
# Vector helpers (kept dependency free)
# ---------------------------------------------------------------------------
def _norm(v: Vec2) -> float:
    return math.hypot(v[0], v[1])


def _sub(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] - b[0], a[1] - b[1])


def _scale(v: Vec2, s: float) -> Vec2:
    return (v[0] * s, v[1] * s)


def _rotate(v: Vec2, angle_rad: float) -> Vec2:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return (v[0] * c - v[1] * s, v[0] * s + v[1] * c)


def _angle_between_deg(a: Vec2, b: Vec2) -> float:
    na, nb = _norm(a), _norm(b)
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    cosang = max(-1.0, min(1.0, (a[0] * b[0] + a[1] * b[1]) / (na * nb)))
    return math.degrees(math.acos(cosang))


def _signed_turn_deg(wind: Vec2, drift: Vec2) -> float:
    """Signed angle from wind to drift (positive = counter-clockwise, NH = left)."""
    mag = _angle_between_deg(wind, drift)
    if mag == 0.0:
        return 0.0
    cross = wind[0] * drift[1] - wind[1] * drift[0]
    return mag if cross > 0.0 else -mag


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DriftForcing:
    """Environmental forcing for the free-drift closure."""

    wind: Vec2 = (10.0, 0.0)
    current: Vec2 = (0.0, 0.0)
    latitude_deg: float = 75.0
    air_density: float = RHO_AIR
    water_density: float = RHO_WATER
    ice_density: float = RHO_ICE
    air_drag_coefficient: float = C_AIR_ICE
    kinematic_viscosity: float = NU_SEAWATER

    def coriolis(self) -> float:
        return 2.0 * OMEGA_EARTH * math.sin(math.radians(self.latitude_deg))


@dataclass(frozen=True)
class ClosureSolution:
    """Output of the drift-roughness-drag closure."""

    ice_velocity: Vec2
    drift_speed: float
    relative_velocity: Vec2
    relative_speed: float
    drag_coefficient: float
    z0_ice_m: float
    u_star: float
    roughness_regime: str
    surface_drag_coefficient: float
    z0_surface_m: float
    wind_stress: Vec2
    water_stress: Vec2
    coriolis_stress: Vec2
    force_residual: float
    relative_force_residual: float
    iterations: int
    last_increment: float
    converged: bool
    drift_ratio: float
    turn_angle_deg: float
    concentration: float
    h_ice_m: float
    ice_roughness: IceUndersideRoughness = field(repr=False)
    history: Tuple[Tuple[int, float, float, float], ...] = field(default=(), repr=False)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ice_velocity_m_s": list(self.ice_velocity),
            "drift_speed_m_s": self.drift_speed,
            "relative_speed_m_s": self.relative_speed,
            "cd_ice": self.drag_coefficient,
            "z0_ice_m": self.z0_ice_m,
            "u_star_m_s": self.u_star,
            "roughness_regime": self.roughness_regime,
            "cd_surface": self.surface_drag_coefficient,
            "z0_surface_m": self.z0_surface_m,
            "wind_stress_pa": list(self.wind_stress),
            "water_stress_pa": list(self.water_stress),
            "coriolis_stress_pa": list(self.coriolis_stress),
            "force_residual_pa": self.force_residual,
            "relative_force_residual": self.relative_force_residual,
            "iterations": self.iterations,
            "last_increment_m_s": self.last_increment,
            "converged": self.converged,
            "drift_ratio": self.drift_ratio,
            "turn_angle_deg": self.turn_angle_deg,
            "concentration": self.concentration,
            "h_ice_m": self.h_ice_m,
            "history": [list(h) for h in self.history],
        }


# ---------------------------------------------------------------------------
# Stress terms
# ---------------------------------------------------------------------------
def wind_stress(wind: Vec2, ice_velocity: Vec2, forcing: DriftForcing) -> Vec2:
    """``tau_a = rho_a C_a |U_a - U_i| (U_a - U_i)``."""
    rel = _sub(wind, ice_velocity)
    speed = _norm(rel)
    return _scale(rel, forcing.air_density * forcing.air_drag_coefficient * speed)


def water_stress(
    relative_velocity: Vec2, cd: float, water_density: float = RHO_WATER
) -> Vec2:
    """``tau_w = -rho_w C_d |U_rel| U_rel``."""
    speed = _norm(relative_velocity)
    return _scale(relative_velocity, -water_density * cd * speed)


def coriolis_stress(
    ice_velocity: Vec2, h_ice_m: float, forcing: DriftForcing
) -> Vec2:
    """``-rho_i h f (k x U_i)`` per unit ice area (Coriolis, right of velocity in NH)."""
    f = forcing.coriolis()
    return (forcing.ice_density * h_ice_m * f * ice_velocity[1],
            -forcing.ice_density * h_ice_m * f * ice_velocity[0])


# ---------------------------------------------------------------------------
# Diagnostics for a prescribed drift velocity
# ---------------------------------------------------------------------------
def drift_roughness_diagnostics(
    ice_velocity: Vec2,
    current: Vec2,
    h_ice_m: float,
    concentration: float,
    morphology: KeelMorphology,
    forcing: Optional[DriftForcing] = None,
    z_ref_m: float = Z_REF,
    kappa: float = KAPPA,
) -> Dict[str, Any]:
    """Roughness / drag state for a *given* drift velocity (external model hook).

    Returns ``z0_ice``, ``C_d_ice`` (per unit ice area) plus the surface-mosaic
    values (``z0_surface``, ``C_d_surface``) that include leads.
    """
    forcing = forcing or DriftForcing()
    if concentration < 0.0 or concentration > 1.0:
        raise ClosureError(f"concentration must lie in [0, 1], got {concentration}")
    ice_roughness = ice_underside_roughness(morphology, h_ice_m, z_ref_m, kappa)
    rel = _sub(ice_velocity, current)
    rel_speed = max(_norm(rel), 1e-6)
    friction: FrictionState = solve_ice_area_friction(
        rel_speed, ice_roughness, forcing.kinematic_viscosity, z_ref_m, kappa
    )
    mosaic = surface_interface_roughness(ice_roughness, concentration, z_ref_m, kappa)
    out = {
        "ice_velocity_m_s": list(ice_velocity),
        "current_m_s": list(current),
        "relative_speed_m_s": rel_speed,
        "h_ice_m": h_ice_m,
        "concentration": concentration,
        "keel_morphology": morphology.as_dict(),
        "ice_underside": ice_roughness.as_dict(),
        "z0_ice_m": friction.z0_m,
        "cd_ice": friction.drag_coefficient,
        "cd_ice_geometric": ice_roughness.drag_coefficient,
        "z0_ice_geometric_m": ice_roughness.z0_m,
        "u_star_m_s": friction.u_star,
        "roughness_reynolds": friction.roughness_reynolds,
        "roughness_regime": friction.regime,
        "z0_surface_m": mosaic["z0_surface_m"],
        "cd_surface": mosaic["cd_surface"],
        "cd_water": mosaic["cd_water"],
    }
    return out


# ---------------------------------------------------------------------------
# Free-drift closure
# ---------------------------------------------------------------------------
DriftProvider = Callable[[float, Vec2, Vec2], Vec2]


def _residual(
    ice_velocity: Vec2,
    h_ice_m: float,
    morphology: KeelMorphology,
    forcing: DriftForcing,
    ice_roughness: IceUndersideRoughness,
    z_ref_m: float,
    kappa: float,
) -> Tuple[Vec2, FrictionState]:
    rel = _sub(ice_velocity, forcing.current)
    rel_speed = max(_norm(rel), 1e-6)
    friction = solve_ice_area_friction(
        rel_speed, ice_roughness, forcing.kinematic_viscosity, z_ref_m, kappa
    )
    tau_a = wind_stress(forcing.wind, ice_velocity, forcing)
    tau_w = water_stress(rel, friction.drag_coefficient, forcing.water_density)
    tau_c = coriolis_stress(ice_velocity, h_ice_m, forcing)
    total = (tau_a[0] + tau_w[0] + tau_c[0], tau_a[1] + tau_w[1] + tau_c[1])
    return total, friction


def solve_free_drift(
    h_ice_m: float,
    concentration: float,
    morphology: KeelMorphology,
    forcing: Optional[DriftForcing] = None,
    z_ref_m: float = Z_REF,
    kappa: float = KAPPA,
    tol: float = 1e-4,
    max_iter: int = 10,
    drift_provider: Optional[DriftProvider] = None,
    freeze_drift: bool = False,
) -> ClosureSolution:
    """Solve the steady drift-roughness-drag closure (Newton, 2 velocity comps).

    Parameters
    ----------
    h_ice_m, concentration, morphology : ice state and keel statistics.
    forcing : wind / current / Coriolis environment.
    tol : acceptance tolerance on the drift-velocity increment, m/s.
    max_iter : iteration budget for the outer Newton loop.
    drift_provider : optional external drift model ``(h, wind, current) -> U``.
    freeze_drift : if True, keep the provider's drift velocity fixed and only
        diagnose roughness and drag (pure coupling / interface mode).

    Returns
    -------
    :class:`ClosureSolution`
    """
    forcing = forcing or DriftForcing()
    if not (h_ice_m > 0.0):
        raise ClosureError(f"h_ice_m must be > 0, got {h_ice_m}")
    if not (0.0 <= concentration <= 1.0):
        raise ClosureError(f"concentration must lie in [0, 1], got {concentration}")

    ice_roughness = ice_underside_roughness(morphology, h_ice_m, z_ref_m, kappa)
    wind_speed = _norm(forcing.wind)

    if drift_provider is not None:
        guess = tuple(float(v) for v in drift_provider(h_ice_m, forcing.wind, forcing.current))
        if len(guess) != 2:
            raise ClosureError("drift_provider must return a 2-component velocity")
    else:
        # first guess: Nansen-like 2 % of the wind, turned 25 deg to the right (NH)
        guess = _rotate(_scale(forcing.wind, 0.02), math.radians(-25.0))

    if freeze_drift:
        ice_velocity = (guess[0], guess[1])
        total, friction = _residual(
            ice_velocity, h_ice_m, morphology, forcing, ice_roughness, z_ref_m, kappa
        )
        rel = _sub(ice_velocity, forcing.current)
        mosaic = surface_interface_roughness(ice_roughness, concentration, z_ref_m, kappa)
        magnitudes = max(_norm(water_stress(rel, friction.drag_coefficient,
                                            forcing.water_density)), 1e-30)
        return ClosureSolution(
            ice_velocity=ice_velocity,
            drift_speed=_norm(ice_velocity),
            relative_velocity=rel,
            relative_speed=_norm(rel),
            drag_coefficient=friction.drag_coefficient,
            z0_ice_m=friction.z0_m,
            u_star=friction.u_star,
            roughness_regime=friction.regime,
            surface_drag_coefficient=mosaic["cd_surface"],
            z0_surface_m=mosaic["z0_surface_m"],
            wind_stress=wind_stress(forcing.wind, ice_velocity, forcing),
            water_stress=water_stress(rel, friction.drag_coefficient, forcing.water_density),
            coriolis_stress=coriolis_stress(ice_velocity, h_ice_m, forcing),
            force_residual=_norm(total),
            relative_force_residual=_norm(total) / magnitudes,
            iterations=0,
            last_increment=float("nan"),
            converged=False,
            drift_ratio=(_norm(ice_velocity) / wind_speed) if wind_speed > 0 else float("nan"),
            turn_angle_deg=_signed_turn_deg(forcing.wind, ice_velocity) if wind_speed > 0 else 0.0,
            concentration=concentration,
            h_ice_m=h_ice_m,
            ice_roughness=ice_roughness,
        )

    ice_velocity = guess
    history: list = []
    last_increment = float("inf")
    converged = False
    iterations = 0
    friction: FrictionState = None  # type: ignore[assignment]
    total: Vec2 = (0.0, 0.0)

    for iterations in range(1, max_iter + 1):
        total, friction = _residual(
            ice_velocity, h_ice_m, morphology, forcing, ice_roughness, z_ref_m, kappa
        )
        # numerical Jacobian of the residual w.r.t. the drift velocity
        step = 1e-4 * (1.0 + _norm(ice_velocity))
        jac = []
        for comp in range(2):
            perturbed = list(ice_velocity)
            perturbed[comp] += step
            r_p, _ = _residual(
                (perturbed[0], perturbed[1]), h_ice_m, morphology, forcing,
                ice_roughness, z_ref_m, kappa,
            )
            jac.append(((r_p[0] - total[0]) / step, (r_p[1] - total[1]) / step))
        # jac[comp] = d R / d v_comp ; build J with columns for v0, v1
        j00, j10 = jac[0]
        j01, j11 = jac[1]
        det = j00 * j11 - j01 * j10
        if abs(det) < 1e-30:
            raise ClosureError("singular drift-closure Jacobian")
        dx = (-total[0] * j11 + total[1] * j01) / det
        dy = (-total[1] * j00 + total[0] * j10) / det
        new_velocity = (ice_velocity[0] + dx, ice_velocity[1] + dy)
        last_increment = math.hypot(new_velocity[0] - ice_velocity[0],
                                    new_velocity[1] - ice_velocity[1])
        ice_velocity = new_velocity
        history.append((iterations, ice_velocity[0], ice_velocity[1], last_increment))
        if last_increment <= tol:
            converged = True
            break

    total, friction = _residual(
        ice_velocity, h_ice_m, morphology, forcing, ice_roughness, z_ref_m, kappa
    )
    rel = _sub(ice_velocity, forcing.current)
    mosaic = surface_interface_roughness(ice_roughness, concentration, z_ref_m, kappa)
    w_stress = water_stress(rel, friction.drag_coefficient, forcing.water_density)
    magnitudes = max(_norm(w_stress), 1e-30)
    return ClosureSolution(
        ice_velocity=ice_velocity,
        drift_speed=_norm(ice_velocity),
        relative_velocity=rel,
        relative_speed=_norm(rel),
        drag_coefficient=friction.drag_coefficient,
        z0_ice_m=friction.z0_m,
        u_star=friction.u_star,
        roughness_regime=friction.regime,
        surface_drag_coefficient=mosaic["cd_surface"],
        z0_surface_m=mosaic["z0_surface_m"],
        wind_stress=wind_stress(forcing.wind, ice_velocity, forcing),
        water_stress=w_stress,
        coriolis_stress=coriolis_stress(ice_velocity, h_ice_m, forcing),
        force_residual=_norm(total),
        relative_force_residual=_norm(total) / magnitudes,
        iterations=iterations,
        last_increment=last_increment,
        converged=converged,
        drift_ratio=(_norm(ice_velocity) / wind_speed) if wind_speed > 0 else float("nan"),
        turn_angle_deg=_signed_turn_deg(forcing.wind, ice_velocity) if wind_speed > 0 else 0.0,
        concentration=concentration,
        h_ice_m=h_ice_m,
        ice_roughness=ice_roughness,
        history=tuple(history),
    )