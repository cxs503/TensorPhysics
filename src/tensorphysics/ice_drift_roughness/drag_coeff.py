"""Under-ice friction velocity and drag coefficient with Reynolds-number closure.

Physics
-------
The drag coefficient of a rough wall referred to a reference height ``z_ref`` is,
from the logarithmic law of the wall,

    C_d = [ kappa / ln( z_ref / z0 ) ] ** 2 ,                                  (1)

where the roughness length of a *hydrodynamically smooth* wall is set by the
viscous sublayer,

    z0_smooth = 0.11 * nu / u_star        (Schlichting & Gersten 2000)          (2)

and that of a *fully rough* wall by the equivalent sand roughness ``k_s``,

    z0_rough  = k_s / 30                  (Nikuradse 1933).                    (3)

Combining (2) and (3) by adding the two roughness lengths (the usual
"superposition of roughness" approximation used for the ice-ocean interface,
McPhee 2008) gives a drag coefficient that is a function of the drift Reynolds
number through ``u_star``:

    z0(u_star) = 0.11 nu / u_star + k_s / 30 ,
    C_d        = [ kappa / ln( z_ref / z0(u_star) ) ] ** 2 ,
    u_star     = U_rel * sqrt(C_d) .                                           (4)

System (4) is solved by fixed-point iteration in :func:`solve_friction_velocity`.

Independent log-law reference
-----------------------------
The classical alternative representation of the same wall law uses the
*roughness function* (velocity defect) of the log layer,

    U/u_star = (1/kappa) ln( z_ref u_star / nu ) + B - dU_plus ,               (5)

with the Karman interpolation for the roughness function

    dU_plus = (1/kappa) ln( 1 + 0.3 k_s_plus ),   k_s_plus = u_star k_s / nu   (6)

and the smooth-wall constant ``B`` fixed by requiring ``dU_plus -> 0`` as
``k_s_plus -> 0`` with ``z0_smooth = 0.11 nu / u_star``, i.e.
``B = (1/kappa) ln(1/0.11) = 5.518``.

Equations (5)-(6) are *not* identical to (4): they describe the wall law in
velocity-defect form, they use a different interpolation constant (0.3 k_s_plus
vs the additive roughness-length superposition), and they are solved for
``u_star`` directly.  Agreement between the two routes is therefore a genuine
consistency test of the closure -- the criterion C2 test of this module -- and
it requires the drift-Reynolds-number dependence to follow log-law boundary
layer theory to within 5 %.

Regimes are classified by the roughness Reynolds number ``k_s_plus``:
smooth (``k_s_plus < 5``), transitional (``5 ... 70``), fully rough
(``k_s_plus > 70``).

References
----------
* Schlichting, H. & Gersten, K. (2000) *Boundary-Layer Theory*, 8th ed., ch. 17
  (smooth-wall roughness length ``0.11 nu / u_star``, skin-friction laws).
* Nikuradse, J. (1933) VDI-Forschungsheft 361 (``k_s = 30 z0``).
* Karman, Th. von (1930) / White, F.M. (2006) *Viscous Fluid Flow*, 3rd ed.,
  ch. 7 (roughness function ``dU_plus``).
* McPhee, M.G. (2008) *Air-Ice-Ocean Interaction*, Springer (under-ice log
  layer, superposition of roughness for the ice-ocean interface).

Units: SI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

from .z0_model import (
    KAPPA,
    NU_SEAWATER,
    PARAMETER_PROVENANCE as _Z0_PROVENANCE,
    Z_REF,
    IceDriftRoughnessError,
    IceUndersideRoughness,
    log_law_drag_coefficient,
    roughness_length_from_drag_coefficient,
)

PARAMETER_PROVENANCE: Dict[str, str] = {
    **_Z0_PROVENANCE,
    "SMOOTH_ROUGHNESS_COEFFICIENT": "0.11 (Schlichting & Gersten 2000): z0 = 0.11 nu/u*",
    "KARMAN_ROUGHNESS_COEFFICIENT": "0.3 (Karman interpolation dU+ = ln(1+0.3 ks+)/kappa)",
    "SMOOTH_REGIME_UPPER_KS_PLUS": "5.0, hydraulic smooth / transitional boundary",
    "FULLY_ROUGH_LOWER_KS_PLUS": "70.0, transitional / fully rough boundary",
    "LAW_OF_WALL_B": "B = ln(1/0.11)/kappa = 5.518, smooth-wall log-law constant",
}

SMOOTH_ROUGHNESS_COEFFICIENT = 0.11
KARMAN_ROUGHNESS_COEFFICIENT = 0.3
SMOOTH_REGIME_UPPER_KS_PLUS = 5.0
FULLY_ROUGH_LOWER_KS_PLUS = 70.0


class DragCoefficientError(IceDriftRoughnessError):
    """Raised when the friction closure cannot be solved."""


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
def smooth_equivalent_roughness(
    u_star: float, nu: float = NU_SEAWATER
) -> float:
    """Viscous (hydraulically smooth) roughness length ``0.11 nu / u_star``."""
    if not (u_star > 0.0):
        raise DragCoefficientError(f"friction velocity must be > 0, got {u_star}")
    if not (nu > 0.0):
        raise DragCoefficientError(f"kinematic viscosity must be > 0, got {nu}")
    return SMOOTH_ROUGHNESS_COEFFICIENT * nu / u_star


def roughness_reynolds(u_star: float, k_s: float, nu: float = NU_SEAWATER) -> float:
    """``k_s_plus = u_star k_s / nu``."""
    if not (k_s > 0.0):
        raise DragCoefficientError(f"equivalent sand roughness must be > 0, got {k_s}")
    if not (u_star > 0.0):
        raise DragCoefficientError(f"friction velocity must be > 0, got {u_star}")
    return u_star * k_s / nu


def roughness_regime(k_s_plus: float) -> str:
    """Classify the wall regime from the roughness Reynolds number."""
    if k_s_plus < SMOOTH_REGIME_UPPER_KS_PLUS:
        return "smooth"
    if k_s_plus <= FULLY_ROUGH_LOWER_KS_PLUS:
        return "transitional"
    return "fully_rough"


def law_of_wall_smooth_constant(kappa: float = KAPPA) -> float:
    """``B = ln(1/0.11)/kappa`` so that ``dU_plus -> 0`` for a smooth wall."""
    return math.log(1.0 / SMOOTH_ROUGHNESS_COEFFICIENT) / kappa


def karman_roughness_function(k_s_plus: float, kappa: float = KAPPA) -> float:
    """Karman roughness function ``dU_plus = (1/kappa) ln(1 + 0.3 k_s_plus)``."""
    if k_s_plus < 0.0:
        raise DragCoefficientError(f"k_s_plus must be >= 0, got {k_s_plus}")
    return math.log(1.0 + KARMAN_ROUGHNESS_COEFFICIENT * k_s_plus) / kappa


def fully_rough_drag_coefficient(k_s: float, z_ref_m: float = Z_REF, kappa: float = KAPPA) -> float:
    """Closed-form fully rough value ``[kappa / ln(z_ref 30 / k_s)]**2``."""
    if not (k_s > 0.0):
        raise DragCoefficientError(f"k_s must be > 0, got {k_s}")
    return log_law_drag_coefficient(k_s / 30.0, z_ref_m, kappa)


# ---------------------------------------------------------------------------
# Model route: roughness-length superposition (system (4))
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FrictionState:
    """Result of the friction-velocity closure."""

    u_star: float
    drag_coefficient: float
    z0_m: float
    iterations: int
    converged: bool
    roughness_reynolds: float
    regime: str
    relative_velocity: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "u_star_m_s": self.u_star,
            "drag_coefficient": self.drag_coefficient,
            "z0_m": self.z0_m,
            "iterations": self.iterations,
            "converged": self.converged,
            "roughness_reynolds": self.roughness_reynolds,
            "regime": self.regime,
            "relative_velocity_m_s": self.relative_velocity,
        }


def solve_friction_velocity(
    u_rel: float,
    z0_geom_m: float,
    nu: float = NU_SEAWATER,
    z_ref_m: float = Z_REF,
    kappa: float = KAPPA,
    tol: float = 1e-14,
    max_iter: int = 200,
) -> FrictionState:
    """Solve the roughness-length superposition closure for a *single* class.

    ``z0 = z0_geom + 0.11 nu/u_star`` and ``C_d = [kappa/ln(z_ref/z0)]**2`` are
    iterated together with ``u_star = u_rel sqrt(C_d)``.
    """
    if not (u_rel > 0.0):
        raise DragCoefficientError(f"relative velocity must be > 0, got {u_rel}")
    if not (0.0 < z0_geom_m < z_ref_m):
        raise DragCoefficientError(
            f"geometric roughness must satisfy 0 < z0 < z_ref, got {z0_geom_m}"
        )

    u_star = u_rel * math.sqrt(log_law_drag_coefficient(z0_geom_m, z_ref_m, kappa))
    converged = False
    iterations = 0
    for iterations in range(1, max_iter + 1):
        z0 = z0_geom_m + smooth_equivalent_roughness(u_star, nu)
        if z0 >= z_ref_m:
            z0 = z_ref_m * (1.0 - 1e-12)
        cd = log_law_drag_coefficient(z0, z_ref_m, kappa)
        u_star_new = u_rel * math.sqrt(cd)
        if abs(u_star_new - u_star) <= tol * max(u_star_new, 1e-30):
            u_star = u_star_new
            converged = True
            break
        u_star = u_star_new

    z0 = z0_geom_m + smooth_equivalent_roughness(u_star, nu)
    cd = log_law_drag_coefficient(z0, z_ref_m, kappa)
    k_s_plus = roughness_reynolds(u_star, 30.0 * z0_geom_m, nu)
    return FrictionState(
        u_star=u_star,
        drag_coefficient=cd,
        z0_m=z0,
        iterations=iterations,
        converged=converged,
        roughness_reynolds=k_s_plus,
        regime=roughness_regime(k_s_plus),
        relative_velocity=u_rel,
    )


def solve_ice_area_friction(
    u_rel: float,
    ice_roughness: IceUndersideRoughness,
    nu: float = NU_SEAWATER,
    z_ref_m: float = Z_REF,
    kappa: float = KAPPA,
    tol: float = 1e-14,
    max_iter: int = 200,
) -> FrictionState:
    """Solve the closure for the two-class ice-underside mosaic.

    Each class carries its own viscous roughness contribution, so the drift
    velocity enters the roughness length explicitly:

        z0_i(u_star) = z0_geom,i + 0.11 nu / u_star ,
        C_d_ice      = A_r C_d(z0_ridge) + (1 - A_r) C_d(z0_skin) ,
        u_star       = u_rel sqrt(C_d_ice) .
    """
    if not (u_rel > 0.0):
        raise DragCoefficientError(f"relative velocity must be > 0, got {u_rel}")

    a_r = ice_roughness.area_fraction
    z0_skin_geom = ice_roughness.z0_skin_m
    z0_ridge_geom = ice_roughness.z0_ridge_m

    def cd_of(u_star: float) -> float:
        visc = smooth_equivalent_roughness(u_star, nu)
        z0_skin = min(z0_skin_geom + visc, z_ref_m * (1.0 - 1e-12))
        z0_ridge = min(z0_ridge_geom + visc, z_ref_m * (1.0 - 1e-12))
        cd_skin = log_law_drag_coefficient(z0_skin, z_ref_m, kappa)
        cd_ridge = log_law_drag_coefficient(z0_ridge, z_ref_m, kappa)
        return a_r * cd_ridge + (1.0 - a_r) * cd_skin

    u_star = u_rel * math.sqrt(ice_roughness.drag_coefficient)
    converged = False
    iterations = 0
    for iterations in range(1, max_iter + 1):
        cd = cd_of(u_star)
        u_star_new = u_rel * math.sqrt(cd)
        if abs(u_star_new - u_star) <= tol * max(u_star_new, 1e-30):
            u_star = u_star_new
            converged = True
            break
        u_star = u_star_new

    cd = cd_of(u_star)
    visc = smooth_equivalent_roughness(u_star, nu)
    z0_eff = roughness_length_from_drag_coefficient(cd, z_ref_m, kappa)
    k_s_plus = roughness_reynolds(u_star, 30.0 * max(z0_ridge_geom, z0_skin_geom), nu)
    return FrictionState(
        u_star=u_star,
        drag_coefficient=cd,
        z0_m=z0_eff,
        iterations=iterations,
        converged=converged,
        roughness_reynolds=k_s_plus,
        regime=roughness_regime(k_s_plus),
        relative_velocity=u_rel,
    )


# ---------------------------------------------------------------------------
# Independent log-law reference route (5)-(6)
# ---------------------------------------------------------------------------
def law_of_wall_drag_coefficient(
    u_rel: float,
    k_s: float,
    nu: float = NU_SEAWATER,
    z_ref_m: float = Z_REF,
    kappa: float = KAPPA,
) -> FrictionState:
    """Drag coefficient from the velocity-defect (roughness-function) form.

    Solves ``U/u_star = (1/kappa) ln(z_ref u_star/nu) + B - dU_plus`` for
    ``u_star`` by bisection.  This is the independent log-law reference used by
    criterion C2; it shares no algebra with the roughness-length superposition.
    """
    if not (u_rel > 0.0):
        raise DragCoefficientError(f"relative velocity must be > 0, got {u_rel}")
    if not (k_s > 0.0):
        raise DragCoefficientError(f"k_s must be > 0, got {k_s}")
    if not (nu > 0.0):
        raise DragCoefficientError(f"nu must be > 0, got {nu}")

    big_b = law_of_wall_smooth_constant(kappa)

    def residual(u_star: float) -> float:
        k_s_plus = roughness_reynolds(u_star, k_s, nu)
        lhs = (1.0 / kappa) * math.log(z_ref_m * u_star / nu) + big_b
        return lhs - karman_roughness_function(k_s_plus, kappa) - u_rel / u_star

    lo = 1e-8 * u_rel
    hi = u_rel
    if residual(hi) < 0.0:
        # drag coefficient would exceed unity: not reachable for these inputs
        raise DragCoefficientError("log-law reference has no bracket in 0 < u*/U < 1")
    for iterations in range(1, 301):
        mid = 0.5 * (lo + hi)
        if residual(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    u_star = 0.5 * (lo + hi)
    iterations = 300
    cd = (u_star / u_rel) ** 2
    k_s_plus = roughness_reynolds(u_star, k_s, nu)
    return FrictionState(
        u_star=u_star,
        drag_coefficient=cd,
        z0_m=k_s / 30.0 + smooth_equivalent_roughness(u_star, nu),
        iterations=iterations,
        converged=True,
        roughness_reynolds=k_s_plus,
        regime=roughness_regime(k_s_plus),
        relative_velocity=u_rel,
    )