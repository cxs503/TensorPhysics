"""Step loop, momentum/energy ledgers and the audit cases.

The audit is built on **two independent momentum books** that must agree:

1. the *snapshot* book -- system momentum measured from the fields
   ``P_sys(t) = sum_i f_i c_i (fluid) + sum_k m_k v_k (floes)``;
2. the *impulse* book -- everything the surroundings delivered, accumulated from
   the exact bounce-back tallies (``-wall impulse``, ``-ship impulse``) and the
   prescribed external forces.

``momentum_residual = |P_sys(T) - P_sys(0) - sum I_ext| / scale`` is therefore a
real test and not a restatement: the two books are computed from different
numbers (field moments vs per-link momentum exchange).

The energy book closes the same way with one physical sink:

``d(KE_fluid + KE_floes + PE_contact) = W_ship + W_walls + W_ext - c * Phi_visc``

with ``Phi_visc = sum 2 mu S:S`` from the non-equilibrium stress.  Using the
lattice collision loss instead would make the budget a tautology, but the raw
continuum estimate carries a systematic, Mach-independent bias (the
non-equilibrium stress truncated at second order); the bias is measured on a
smooth reference flow at the *same* grid and relaxation time and enters as the
calibration gain ``c`` (:func:`calibrate_dissipation`).  Both the raw and the
calibrated residuals are reported -- the calibration is an instrument property
and cannot hide a coupling error, because the reference run has no coupling.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .dem import DEMConfig, DEMField, Floe
from .fluid import LBMFluid2D

__all__ = [
    "AuditCase",
    "CaseResult",
    "run_case",
    "calibrate_dissipation",
    "momentum_scale_of",
]


def momentum_scale_of(case: "AuditCase", dem: DEMField, fl: LBMFluid2D) -> float:
    """Reference momentum for the relative residual of a case.

    Never let the scale collapse: the scale is the largest of the initial system
    momentum and the bulk momentum ``rho U L`` of the configuration.
    """
    p0 = np.linalg.norm(fl.momentum() + dem.momentum())
    bulk = fl.rho0 * case.u_ref * min(case.nx, case.ny)
    return float(max(p0, bulk, 1e-30))


@dataclass
class AuditCase:
    """A fully specified audit configuration (already in lattice units)."""

    name: str
    title: str
    nx: int
    ny: int
    tau: float
    steps: int
    floes: list = field(default_factory=list)
    dem_config: DEMConfig = field(default_factory=DEMConfig)
    u_ref: float = 0.02
    initial_flow: Callable[[int, int], tuple] | None = None
    external_force: Callable[[int, DEMField], np.ndarray] | None = None
    ship_index: int | None = None
    ship_velocity: tuple = (0.0, 0.0)
    ship_move: Callable[[int, Floe], None] | None = None
    wall_velocity: dict = field(default_factory=dict)
    dem_coupling: bool = True
    calibration_gain: float = 1.0
    window_start: int | None = None
    physical: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.window_start is None:
            # skip the impulsive-start transient: the reversible pressure work of
            # the acoustic pulse is O(Ma) and the linearised KE budget does not
            # hold during it
            self.window_start = self.steps // 4

    def build_dem(self) -> DEMField | None:
        if not self.floes or not self.dem_coupling:
            return None
        return DEMField([Floe(f.x, f.v, f.radius, f.mass, f.fixed) for f in self.floes],
                        self.dem_config)


@dataclass(frozen=True)
class CaseResult:
    """Measured audit numbers of one case (one grid level)."""

    name: str
    title: str
    nx: int
    ny: int
    tau: float
    steps: int
    window_start: int
    dt: float
    dx: float
    momentum_residual: float
    momentum_scale: float
    impulse_residual: float
    energy_residual: float
    energy_residual_raw: float
    calibration_gain: float
    consistency_residual: float
    interface_impulse_x: float
    window_impulse_x: float
    window_mean_force_x: float
    external_impulse: float
    wall_impulse: float
    ke_initial: float
    ke_final: float
    ke_window_initial: float
    ke_window_final: float
    conversion_energy: float
    mass_energy: float
    dissipation: float
    dissipation_raw_gain: float
    external_work: float
    max_mach: float
    max_velocity: float
    mass_drift: float
    series: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        out = {k: v for k, v in self.__dict__.items() if k != "series"}
        out["series"] = {k: [float(x) for x in v] for k, v in self.series.items()}
        return out


def _rasterize(dem: DEMField, nx: int, ny: int):
    solid = np.zeros((nx, ny), dtype=bool)
    node_id = np.full((nx, ny), -1, dtype=np.int64)
    gx = np.arange(nx, dtype=np.float64)[:, None] + 0.5
    gy = np.arange(ny, dtype=np.float64)[None, :] + 0.5
    for k, f in enumerate(dem.floes):
        # Particle mapping: the body is rasterised at the nearest lattice point
        # that keeps the staircase mask mirror symmetric in x and y.  A mask that
        # is not symmetric carries a spurious momentum-exchange force of order
        # rho w c per unbalanced link (~0.2 here, ten times the physical drag)
        # which pumps the fluid instead of damping the floe; snapping costs one
        # O(dx) geometry error (first order, standard for a staircase boundary).
        xc = round(f.x[0] - 0.5) + 0.5
        yc = round(f.x[1] - 0.5) + 0.5
        inside = (gx - xc) ** 2 + (gy - yc) ** 2 < f.radius * f.radius
        solid = solid | inside
        node_id = np.where(inside, k, node_id)
    return solid, node_id


def run_case(case: AuditCase, record: bool = True) -> CaseResult:
    """Run one audit configuration and return its measured numbers."""
    fl = LBMFluid2D(case.nx, case.ny, tau=case.tau)
    if case.initial_flow is not None:
        ux, uy = case.initial_flow(case.nx, case.ny)
        fl.set_state(np.ones((case.nx, case.ny)), ux, uy)
    for side, vel in case.wall_velocity.items():
        fl.wall_velocity[side] = np.asarray(vel, dtype=np.float64)

    dem = case.build_dem()
    ship = case.ship_index if dem is not None else None
    scale = momentum_scale_of(case, dem if dem is not None else DEMField([]), fl)

    p_sys0 = np.asarray(fl.momentum() + (dem.momentum() if dem is not None else 0.0),
                        dtype=np.float64)

    impulse_ext = np.zeros(2)     # delivered to the system by the prescription
    impulse_wall = np.zeros(2)    # delivered to the system by the outer walls
    impulse_ship = np.zeros(2)    # delivered to the system by the ship
    impulse_iface = np.zeros(2)   # exchanged between fluid and floes (internal)
    impulse_iface_win = np.zeros(2)
    iface_force_win = np.zeros(2)
    w0 = max(0, min(int(case.window_start), case.steps - 1))
    w_ext = 0.0
    w_wall = 0.0
    w_ship = 0.0
    w_iface_lattice = 0.0
    w_iface_mech = 0.0
    dissipation = 0.0
    e_conv = 0.0
    e_mass = 0.0
    mass0 = fl.mass()
    max_mach = 0.0
    max_velocity = 0.0
    ke_start = float(
        fl.kinetic_energy()
        + (dem.kinetic_energy() if dem is not None else 0.0)
        + (dem.contacts()[1] if dem is not None else 0.0)
    )

    keys = ("t", "momentum_x", "momentum_y", "impulse_x", "ke_fluid", "ke_floes",
            "ke_total", "dissipation_cum", "work_cum", "floe_ux", "floe_uy")
    series = {k: [] for k in keys}

    for step in range(case.steps):
        # ---------------------------------------------------------- geometry
        if dem is not None:
            if ship is not None and case.ship_move is not None:
                case.ship_move(step, dem.floes[ship])
                dem.floes[ship].v = np.asarray(case.ship_velocity, dtype=np.float64)
            solid, node_id = _rasterize(dem, case.nx, case.ny)
            vel = dem.velocities()
            _fluid_conv, per_body_conv = fl.apply_mask(solid, node_id, vel)
            e_conv += fl.conversion_energy
            for k, imp in per_body_conv.items():
                if k == ship or dem.floes[k].fixed or not np.isfinite(dem.floes[k].mass):
                    # a prescribed body can supply momentum and energy: both are
                    # booked with *its* velocity (not the ship's) so that any
                    # prescribed actuator closes the budget
                    impulse_ship -= imp
                    w_ship -= float(np.asarray(imp) @ np.asarray(dem.floes[k].v))
                else:
                    dem.floes[k].v = dem.floes[k].v + np.asarray(imp) / dem.floes[k].mass

        # ------------------------------------------------------------- fluid
        mass_before = fl.mass()
        ke_before = fl.kinetic_energy()
        diag = fl.step()
        impulse_wall -= np.asarray(diag.wall_impulse)
        # Mechanical work of the outer walls on the fluid (identically zero for a
        # stationary wall, so only the sheared case contributes).
        w_wall += float(diag.wall_work)
        if dem is not None:
            v = dem.velocities()
            body_force = np.zeros_like(v)
            if step >= (case.window_start or 0):
                for _k, _imp in diag.particle_impulse.items():
                    iface_force_win += np.asarray(_imp)
            for k, imp in diag.particle_impulse.items():
                if k == ship or dem.floes[k].fixed or not np.isfinite(dem.floes[k].mass):
                    impulse_ship -= np.asarray(imp)
                    w_ship -= float(np.asarray(imp) @ np.asarray(v[k]))
                else:
                    body_force[k] += np.asarray(imp)          # dt = 1 -> force
                    impulse_iface += np.asarray(imp)
                    if step >= (case.window_start or 0):
                        impulse_iface_win += np.asarray(imp)
                    w_iface_lattice += float(diag.particle_power.get(k, 0.0))
                    w_iface_mech += float(np.asarray(imp) @ v[k])
            if case.external_force is not None:
                fext = np.asarray(case.external_force(step, dem), dtype=np.float64)
                body_force += fext
                impulse_ext += fext.sum(axis=0)
                w_ext += float(np.sum(fext * v))
            fwall, _ = dem.wall_forces((case.nx, case.ny))
            body_force += fwall
            impulse_ext += fwall.sum(axis=0)
            w_ext += float(np.sum(fwall * v))
            dem.advance(1.0, body_force)
        # mass repair (momentum neutral): its kinetic-energy side effect is a
        # numerical line of the ledger, not physics
        _ke_pre = fl.kinetic_energy()
        if fl.restore_mass(mass0) != 0.0:
            e_mass += fl.kinetic_energy() - _ke_pre
        dissipation += diag.dissipation
        # Mass sink of the moving bounce back: a lost mass element carries its
        # specific kinetic energy out of the lattice.  Booked explicitly (it is
        # a property of the boundary scheme, not of the coupling).
        dm = diag.mass - mass_before
        if dm != 0.0:
            ke_after = diag.kinetic_energy
            spec = ke_before / max(mass_before, 1e-30)
            e_mass += dm * spec

        ke_fluid = diag.kinetic_energy
        ke_floes = dem.kinetic_energy() if dem is not None else 0.0
        pe = dem.contacts()[1] if dem is not None else 0.0
        speed = float(np.sqrt(np.sum(fl.ux[~fl.solid] ** 2 + fl.uy[~fl.solid] ** 2)
                              / max(int((~fl.solid).sum()), 1)))
        max_velocity = max(max_velocity, speed)
        max_mach = max(max_mach, speed / math.sqrt(1.0 / 3.0))

        if record:
            p_sys = fl.momentum() + (dem.momentum() if dem is not None else 0.0)
            series["t"].append(float(step + 1))
            series["momentum_x"].append(float(p_sys[0]))
            series["momentum_y"].append(float(p_sys[1]))
            series["impulse_x"].append(float((impulse_ext + impulse_wall + impulse_ship)[0]))
            series["ke_fluid"].append(float(ke_fluid))
            series["ke_floes"].append(float(ke_floes))
            series["ke_total"].append(float(ke_fluid + ke_floes + pe))
            series["dissipation_cum"].append(float(dissipation))
            series["work_cum"].append(float(w_ext + w_wall + w_ship))
            if dem is not None and dem.n_dynamic() > 0:
                series["floe_ux"].append(float(np.mean([f.v[0] for f in dem.floes
                                                        if not f.fixed])))
                series["floe_uy"].append(float(np.mean([f.v[1] for f in dem.floes
                                                        if not f.fixed])))

    p_sysT = np.asarray(fl.momentum() + (dem.momentum() if dem is not None else 0.0))
    total_impulse = impulse_ext + impulse_wall + impulse_ship
    momentum_residual = float(np.linalg.norm(p_sysT - p_sys0 - total_impulse) / scale)

    ke_end = float(fl.kinetic_energy()
                   + (dem.kinetic_energy() if dem is not None else 0.0)
                   + (dem.contacts()[1] if dem is not None else 0.0))
    work_in = w_ext + w_wall + w_ship

    if record and series["ke_total"]:
        e_w0 = series["ke_total"][w0]
        e_w1 = series["ke_total"][-1]
        phi_win = series["dissipation_cum"][-1] - series["dissipation_cum"][w0]
        work_win = series["work_cum"][-1] - series["work_cum"][w0]
    else:                                     # pragma: no cover - degenerate
        e_w0 = ke_start
        e_w1 = ke_end
        phi_win = dissipation
        work_win = work_in

    # node conversion is a numerical source/sink of kinetic energy (momentum is
    # exact, energy is not): it is carried as its own ledger line
    e_conv_win = e_conv
    delta_e = e_w1 - e_w0 - e_conv_win - e_mass
    escale = max(abs(phi_win), abs(work_win), abs(delta_e), 1e-30)
    raw_gain = abs(delta_e) / phi_win if phi_win > 0.0 else 1.0
    energy_raw = abs(delta_e - (work_win - phi_win)) / escale
    energy_cal = abs(delta_e - (work_win - case.calibration_gain * phi_win)) / escale

    if abs(w_iface_mech) > 0.0 or abs(w_iface_lattice) > 0.0:
        consistency = float(abs(w_iface_lattice - w_iface_mech)
                            / max(abs(w_iface_lattice), abs(w_iface_mech), 1e-30))
    else:
        consistency = 0.0

    return CaseResult(
        name=case.name,
        title=case.title,
        nx=case.nx,
        ny=case.ny,
        tau=case.tau,
        steps=case.steps,
        window_start=w0,
        dt=float(case.physical.get("dt", 1.0)),
        dx=float(case.physical.get("dx", 1.0)),
        momentum_residual=momentum_residual,
        momentum_scale=float(scale),
        impulse_residual=float(
            abs(total_impulse[0] - impulse_iface[0])
            + abs(total_impulse[1] - impulse_iface[1])
        ) / max(float(np.linalg.norm(total_impulse)), 1e-30),
        energy_residual=float(energy_cal),
        energy_residual_raw=float(energy_raw),
        calibration_gain=float(case.calibration_gain),
        consistency_residual=consistency,
        interface_impulse_x=float(impulse_iface[0]),
        window_impulse_x=float(impulse_iface_win[0]),
        window_mean_force_x=float(iface_force_win[0] / max(case.steps - w0, 1)),
        external_impulse=float(np.linalg.norm(total_impulse)),
        wall_impulse=float(np.linalg.norm(impulse_wall)),
        ke_initial=ke_start,
        ke_final=ke_end,
        ke_window_initial=float(e_w0),
        ke_window_final=float(e_w1),
        conversion_energy=float(e_conv_win),
        mass_energy=float(e_mass),
        dissipation=float(phi_win),
        dissipation_raw_gain=float(raw_gain),
        external_work=float(work_win),
        max_mach=float(max_mach),
        max_velocity=float(max_velocity),
        mass_drift=float((fl.mass() - mass0) / mass0),
        series=series if record else {},
    )


def calibrate_dissipation(level, amp: float = 0.01, steps: int | None = None) -> float:
    """Measure the systematic bias of ``sum 2 mu S:S`` on a smooth reference flow.

    A Taylor-Green vortex at the same grid, relaxation time, window and time step
    as the audit case: it has no coupling and no moving boundary, so
    ``c = |dKE| / Phi`` isolates the *estimator* (second-order truncation of the
    non-equilibrium stress).  Applying ``c`` to a coupled case cannot mask a
    coupling error -- it can only remove a property of the instrument.
    """
    from .cases import negative_control_lbm_only

    case = negative_control_lbm_only(level)
    case.steps = steps or level.steps
    res = run_case(case, record=True)
    phi = res.dissipation
    delta = res.ke_window_final - res.ke_window_initial
    return abs(delta) / phi if phi > 0.0 else 1.0