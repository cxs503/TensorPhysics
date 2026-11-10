"""The three audit cases of campaign node ``p367seed-n07`` plus the control arm.

All three cases are 2-D *horizontal-plane* problems: an ice floe floating in
water is in vertical equilibrium (buoyancy balances weight) and therefore has no
net external force in the plane, which is exactly the property that makes the
momentum contract testable -- any drift of the system momentum must be produced
by the solver, not by physics.

``floe_drift_decay``
    One circular floe released with a velocity in still water inside a closed
    box.  Drag removes momentum from the floe and gives it to the water; no
    external force at all, so ``P_fluid + P_floe`` is a constant of the motion.
``ice_field_shear``
    Eight floes in a Couette shear layer between a fixed south wall and a north
    wall moving at ``U``.  Several floe pairs are in contact, so the DEM
    internal impulses must cancel (Newton's third law) and the only momentum
    input is the tallied impulse of the moving wall.
``ship_ice_push``
    A prescribed circular "ship" body advances into a fragmented ice field.  The
    ship can supply momentum (it is prescribed, not solved), so its bounce-back
    and node-conversion impulses are booked as an external account, and its
    mechanical work ``sum I_ship . u_ship`` is the energy input of the case.
``negative_control_lbm_only``
    The same box, grid, time step and window with the DEM coupling switched off:
    a pure LBM run (Taylor-Green vortex).  Its momentum residual has no
    interface at all and must be *smaller* than the coupled residuals and below
    ``1e-12`` -- otherwise the instrument, not the coupling, is the problem.
"""
from __future__ import annotations

import math

import numpy as np

from .coupling import AuditCase
from .dem import DEMConfig, Floe
from .gridscan import RefinementLevel

__all__ = [
    "floe_drift_decay",
    "ice_field_shear",
    "ship_ice_push",
    "negative_control_lbm_only",
    "drag_convergence",
    "all_cases",
]

RHO_WATER = 1000.0
RHO_ICE = 920.0
#: Areal density ratio of the floes, i.e. the physical sea-ice value
#: ``rho_ice/rho_water = 0.92``.  In a horizontal-plane floe model this puts the
#: body *lighter* than the fluid it displaces, i.e. inside the added-mass window
#: of an explicitly staggered fluid/rigid coupling; the audit therefore runs a
#: density-ratio sweep (0.92 ... 10) as part of the evidence and reports what the
#: coupling does there (see docs/LBM_DEM_MOMENTUM_P367N07.md).
FLOE_DENSITY_RATIO = 0.92


def _floe_mass(radius_lat: float) -> float:
    """Areal mass of a floe in lattice units (``rho_water = 1``)."""
    return FLOE_DENSITY_RATIO * math.pi * radius_lat ** 2


def _dem_config(radius_lat: float) -> DEMConfig:
    """Contact law scaled with the floe mass.

    The dimensionless contact time ``sqrt(k/m) dt`` must be grid independent,
    otherwise the same physical configuration becomes mechanically stiffer as the
    grid is refined and the coarse levels of a refinement study are not solving
    the same problem.  ``k = 0.5 m`` gives ``sqrt(k/m) dt = 0.7`` per step, which
    is integrated with eight sub-steps.
    """
    return DEMConfig(stiffness=0.5 * _floe_mass(radius_lat), substeps=8)


def _physical(level: RefinementLevel) -> dict:
    return {"dx": level.dx, "dt": level.dt}


def _taylor_green(nx: int, ny: int, amp: float = 0.01):
    k = 2.0 * math.pi / nx
    gx = np.arange(nx)[:, None] + 0.5
    gy = np.arange(ny)[None, :] + 0.5
    ux = amp * np.sin(k * gx) * np.cos(k * gy)
    uy = -amp * np.cos(k * gx) * np.sin(k * gy)
    return ux, uy


def floe_drift_decay(level: RefinementLevel) -> AuditCase:
    """Single floe, free drift in still water, closed box, no contacts."""
    n = level.n
    radius = n / 8.0
    # the floe starts with twice the reference speed
    floe = Floe(
        x=np.array([0.42 * n, 0.5 * n]),
        v=np.array([2.0 * level.u_lat, 0.0]),
        radius=radius,
        mass=_floe_mass(radius),
    )
    return AuditCase(
        name="floe_drift_decay",
        title="Single buoyant floe: free-drift momentum decay in still water",
        nx=n, ny=n, tau=level.tau, steps=level.steps,
        floes=[floe],
        dem_config=_dem_config(radius),
        u_ref=level.u_lat,
        physical=_physical(level),
    )


def ice_field_shear(level: RefinementLevel) -> AuditCase:
    """Eight floes sheared between a fixed wall and a wall moving at ``U``."""
    n = level.n
    radius = n / 14.0
    pitch = 2.0 * radius * 0.99          # 2% nominal overlap -> contact chain
    floes = []
    for row, y in enumerate((0.38 * n, 0.62 * n)):
        for col in range(4):
            x = (0.22 + 0.18 * col) * n
            floes.append(
                Floe(x=np.array([x, y + 0.1 * radius * (-1) ** (row + col)]),
                     v=np.zeros(2), radius=radius, mass=_floe_mass(radius))
            )
    return AuditCase(
        name="ice_field_shear",
        title="Fragmented-ice field in a Couette shear layer (8 floes, contacts)",
        nx=n, ny=n, tau=level.tau, steps=level.steps,
        floes=floes,
        dem_config=_dem_config(radius),
        u_ref=level.u_lat,
        wall_velocity={"north": np.array([level.u_lat, 0.0])},
        physical=_physical(level),
    )


def ship_ice_push(level: RefinementLevel) -> AuditCase:
    """Prescribed ship body advancing into three floes."""
    n = level.n
    radius = n / 9.0                      # floe radius
    ship_radius = 0.30 * n
    ship_start = 1.16 * n                 # only a cap of the ship is inside
    ship_speed = 2.0 * level.u_lat
    floes = []
    for k, (frac, y) in enumerate(((0.62, 0.5), (0.40, 0.36), (0.40, 0.64))):
        floes.append(
            Floe(x=np.array([frac * n, y * n]), v=np.zeros(2),
                 radius=radius, mass=_floe_mass(radius))
        )
    ship = Floe(x=np.array([ship_start, 0.5 * n]), v=np.array([-ship_speed, 0.0]),
                radius=ship_radius, mass=float("inf"), fixed=True)
    floes.append(ship)

    def move(step: int, body: Floe) -> None:
        body.x = np.array([ship_start - ship_speed * (step + 1), 0.5 * n])

    return AuditCase(
        name="ship_ice_push",
        title="Prescribed ship body pushing three floating ice floes",
        nx=n, ny=n, tau=level.tau, steps=level.steps,
        floes=floes,
        dem_config=_dem_config(radius),
        u_ref=level.u_lat,
        ship_index=len(floes) - 1,
        ship_velocity=(-ship_speed, 0.0),
        ship_move=move,
        physical=_physical(level),
    )


def drag_convergence(level: RefinementLevel) -> AuditCase:
    """Prescribed-velocity floe (tow tank): the quantity that must converge.

    The floe is *prescribed* (``fixed``, infinite mass), so there is no
    added-mass feedback and no transient history: the window-averaged interface
    force is the drag of a fixed geometry, which is the quantity whose grid
    convergence bounds the momentum error of a non-adjoint measurement.
    """
    n = level.n
    radius = n / 8.0
    floe = Floe(x=np.array([0.5 * n, 0.5 * n]),
                v=np.array([2.0 * level.u_lat, 0.0]),
                radius=radius, mass=float("inf"), fixed=True)
    return AuditCase(
        name="drag_convergence",
        title="Prescribed-velocity floe: grid convergence of the interface force",
        nx=n, ny=n, tau=level.tau, steps=level.steps,
        floes=[floe],
        u_ref=level.u_lat,
        physical=_physical(level),
    )


def negative_control_lbm_only(level: RefinementLevel) -> AuditCase:
    """Pure LBM arm: no DEM, no interface, momentum must be conserved to 1e-12."""
    return AuditCase(
        name="negative_control_lbm_only",
        title="Negative control: DEM coupling switched off (pure LBM)",
        nx=level.n, ny=level.n, tau=level.tau, steps=level.steps,
        floes=[],
        u_ref=level.u_lat,
        initial_flow=_taylor_green,
        dem_coupling=False,
        physical=_physical(level),
    )


def all_cases(level: RefinementLevel) -> list[AuditCase]:
    return [
        floe_drift_decay(level),
        ice_field_shear(level),
        ship_ice_push(level),
        drag_convergence(level),
        negative_control_lbm_only(level),
    ]