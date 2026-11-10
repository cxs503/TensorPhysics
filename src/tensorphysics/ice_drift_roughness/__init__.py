"""Ice-drift roughness closure: under-ice z0 and the ice-ocean drag coefficient.

Module map
----------
``z0_model``
    Two-class (undeformed ice / ridge keel) mosaic roughness of the ice
    underside from keel depth and keel spacing, with the log-law-consistent
    effective roughness length and the ice+lead surface mixture.
``drag_coeff``
    Friction velocity and drag coefficient with the Reynolds-number closure
    (viscous-sublayer + fully-rough superposition) and the independent
    velocity-defect (Karman roughness function) log-law reference.
``closure``
    Steady momentum balance of a free-drifting ice cover solved together with the
    roughness/drag closure (Newton on the drift velocity, 2 components, Coriolis
    and ocean current included) -- plus the injection hook for external drift
    models.
``beaufort``
    Beaufort Sea under-ice drag-coefficient anchors, calibration of the single
    free parameter (wake/form-drag amplification) and the C4 cross-validation.

Top-level entry point
---------------------
:func:`ice_drift_roughness` implements the mission API: given the drift velocity
``U_ice`` (or, in closure mode, the wind ``U_wind``), the ice concentration ``C``,
the level-ice thickness ``h`` and the keel morphology, it returns ``z0_ice`` and
``C_d_ice`` together with the friction velocity, the Reynolds regime and the
surface (ice + lead) values.

Quick example
-------------
>>> out = ice_drift_roughness(
...     ice_velocity=(0.15, -0.05), concentration=0.9, h_ice_m=2.0,
...     wind=(10.0, 0.0), keel_depth_m=6.0, keel_spacing_m=70.0)
>>> round(out["z0_ice_m"], 5) > 0.0 and 2.0e-3 < out["cd_ice"] < 9.0e-3
True

References: see the module docstrings (McPhee 2008; Timco & Burden 1997;
Nikuradse 1933; Schlichting & Gersten 2000; Lu et al. 2011).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

from .beaufort import (
    ANCHORS,
    CD_TOLERANCE,
    DragAnchor,
    anchor_cd_model,
    anchor_roughness,
    calibrate_wake_amplification,
    cross_validate,
    validation_summary,
)
from .closure import (
    C_AIR_ICE,
    ClosureError,
    ClosureSolution,
    DriftForcing,
    DriftProvider,
    RHO_AIR,
    RHO_ICE,
    RHO_WATER,
    coriolis_stress,
    drift_roughness_diagnostics,
    solve_free_drift,
    water_stress,
    wind_stress,
)
from .drag_coeff import (
    DragCoefficientError,
    FrictionState,
    fully_rough_drag_coefficient,
    karman_roughness_function,
    law_of_wall_drag_coefficient,
    law_of_wall_smooth_constant,
    roughness_regime,
    roughness_reynolds,
    smooth_equivalent_roughness,
    solve_friction_velocity,
    solve_ice_area_friction,
)
from .z0_model import (
    KEEL_WIDTH_RATIO,
    KAPPA,
    NU_SEAWATER,
    PARAMETER_PROVENANCE,
    WAKE_AMPLIFICATION_DEFAULT,
    Z_REF,
    Z0_CEILING_DEFAULT,
    Z0_SKIN_DEFAULT,
    Z0_WATER_SURFACE,
    IceDriftRoughnessError,
    IceUndersideRoughness,
    KeelMorphology,
    ice_underside_roughness,
    log_law_drag_coefficient,
    open_water_drag_coefficient,
    roughness_length_from_drag_coefficient,
    surface_interface_roughness,
)

__all__ = [
    # top level
    "ice_drift_roughness",
    # z0 model
    "KeelMorphology",
    "IceUndersideRoughness",
    "IceDriftRoughnessError",
    "ice_underside_roughness",
    "surface_interface_roughness",
    "log_law_drag_coefficient",
    "roughness_length_from_drag_coefficient",
    "open_water_drag_coefficient",
    "PARAMETER_PROVENANCE",
    "KAPPA",
    "Z_REF",
    "NU_SEAWATER",
    "Z0_SKIN_DEFAULT",
    "Z0_CEILING_DEFAULT",
    "Z0_WATER_SURFACE",
    "KEEL_WIDTH_RATIO",
    "WAKE_AMPLIFICATION_DEFAULT",
    # drag coefficient
    "FrictionState",
    "DragCoefficientError",
    "solve_friction_velocity",
    "solve_ice_area_friction",
    "smooth_equivalent_roughness",
    "roughness_reynolds",
    "roughness_regime",
    "karman_roughness_function",
    "law_of_wall_smooth_constant",
    "law_of_wall_drag_coefficient",
    "fully_rough_drag_coefficient",
    # closure
    "DriftForcing",
    "ClosureSolution",
    "ClosureError",
    "DriftProvider",
    "solve_free_drift",
    "drift_roughness_diagnostics",
    "wind_stress",
    "water_stress",
    "coriolis_stress",
    "RHO_ICE",
    "RHO_WATER",
    "RHO_AIR",
    "C_AIR_ICE",
    # Beaufort anchors
    "ANCHORS",
    "DragAnchor",
    "CD_TOLERANCE",
    "anchor_roughness",
    "anchor_cd_model",
    "calibrate_wake_amplification",
    "cross_validate",
    "validation_summary",
]


def ice_drift_roughness(
    ice_velocity: Optional[Tuple[float, float]] = None,
    *,
    concentration: float,
    h_ice_m: float,
    wind: Tuple[float, float] = (10.0, 0.0),
    morphology: Optional[KeelMorphology] = None,
    keel_depth_m: Optional[float] = None,
    keel_spacing_m: Optional[float] = None,
    current: Tuple[float, float] = (0.0, 0.0),
    latitude_deg: float = 75.0,
    wake_amplification: float = WAKE_AMPLIFICATION_DEFAULT,
    z_ref_m: float = Z_REF,
    kappa: float = KAPPA,
    tol: float = 1e-4,
    max_iter: int = 10,
    drift_provider: Optional[DriftProvider] = None,
) -> Dict[str, Any]:
    """Ice-underside roughness ``z0_ice`` and drag coefficient ``C_d_ice``.

    Parameters
    ----------
    ice_velocity : (u_x, u_y) drift velocity in m/s.  If ``None`` the steady
        free-drift momentum balance is solved from ``wind``/``current`` and the
        roughness closure is evaluated at the resulting drift (closure mode).
    concentration : ice concentration ``C`` in [0, 1].
    h_ice_m : level-ice thickness in m.
    wind : 10 m wind vector in m/s (used by the closure and reported as forcing).
    morphology : :class:`KeelMorphology`; alternatively pass ``keel_depth_m`` and
        ``keel_spacing_m``.
    current : ocean surface current vector in m/s.
    latitude_deg : latitude for the Coriolis parameter (closure mode).
    wake_amplification : calibrated form-drag footprint amplification.
    drift_provider : optional external drift model hook (closure mode).

    Returns
    -------
    dict with ``z0_ice_m``, ``cd_ice`` (per unit ice area), the surface-mosaic
    ``z0_surface_m``/``cd_surface``, friction velocity, roughness regime and, in
    closure mode, the drift velocity, turn angle and iteration diagnostics.
    """
    if morphology is None:
        if keel_depth_m is None or keel_spacing_m is None:
            raise IceDriftRoughnessError(
                "provide either morphology=KeelMorphology(...) or both "
                "keel_depth_m and keel_spacing_m"
            )
        morphology = KeelMorphology(
            keel_depth_m=keel_depth_m,
            keel_spacing_m=keel_spacing_m,
            wake_amplification=wake_amplification,
        )

    forcing = DriftForcing(
        wind=tuple(float(v) for v in wind),  # type: ignore[arg-type]
        current=tuple(float(v) for v in current),  # type: ignore[arg-type]
        latitude_deg=latitude_deg,
    )

    if ice_velocity is not None:
        diag = drift_roughness_diagnostics(
            (float(ice_velocity[0]), float(ice_velocity[1])),
            forcing.current,
            h_ice_m,
            concentration,
            morphology,
            forcing,
            z_ref_m,
            kappa,
        )
        diag["mode"] = "prescribed_drift"
        diag["wind_m_s"] = list(forcing.wind)
        diag["wind_stress_pa"] = list(wind_stress(forcing.wind, ice_velocity, forcing))
        return diag

    solution: ClosureSolution = solve_free_drift(
        h_ice_m,
        concentration,
        morphology,
        forcing,
        z_ref_m=z_ref_m,
        kappa=kappa,
        tol=tol,
        max_iter=max_iter,
        drift_provider=drift_provider,
        freeze_drift=False,
    )
    out = solution.as_dict()
    out["mode"] = "free_drift_closure"
    out["wind_m_s"] = list(forcing.wind)
    out["current_m_s"] = list(forcing.current)
    out["keel_morphology"] = morphology.as_dict()
    out["ice_underside"] = solution.ice_roughness.as_dict()
    out["cd_ice_geometric"] = solution.ice_roughness.drag_coefficient
    out["z0_ice_geometric_m"] = solution.ice_roughness.z0_m
    return out