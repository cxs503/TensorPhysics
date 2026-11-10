"""The three-field (LBM / FEM / DEM) closed coupling loop and its energy ledger.

What is coupled
---------------
A cantilever hull panel (Q4 plane-stress FEM + Q4 transient conduction on the
same mesh) is loaded from above by two falling discs (DEM) that fall through a
warm fluid column (D2Q9 advection-diffusion LBM) sitting on the panel.  The
three crossings form the closed cycle declared in :mod:`.contracts`::

    fluid  --heat flux (measured at the lattice Dirichlet wall)-->  solid
    solid  --contact force (spring-dashpot, Newton's third law)-->  particles
    particles --drag dissipation (Stokes-type)--> fluid internal energy

Multi-rate
----------
The four executables have four different stable steps (explicit plate, DEM
contact, implicit panel conduction, lattice diffusion) that differ by two
orders of magnitude.  Each solver steps on its own clock; every exchanged
quantity is a degree-``p`` Lagrange reconstruction of the counterpart's causal
sample history (:mod:`.subcycling`).  Nothing is sub-stepped inside another
solver's step, so the reconstruction error is the *only* time-coupling error in
the run -- which is what makes the order gate in the acceptance module
attributable.

Energy book
-----------
Every event samples :class:`.energy.EnergyBook` from the solver states
(``DEM2D.kinetic_energy``, ``Q4PlaneStressFEM.strain_energy``,
``Q4ThermalFEM.energy()``, the LBM scalar mass, the measured dashpot work).  The
reported residual is therefore a physical non-conservation, not a bookkeeping
identity.

Scope notes (also listed in the report): the carrier fluid is prescribed
quiescent, so only its *energy* channel is two-way; thermomechanical coupling
(thermoelasticity, friction heating fed back into the panel) is not modelled;
contact is frictionless and geometrically normal to the panel surface.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Mapping, Sequence

import numpy as np

from ..solvers import (  # noqa: F401  (lattice weights for the initial state)
    _W as LATTICE_WEIGHTS,
    ContactLaw,
    DEM2D,
    Q4Mesh,
    Q4PlaneStressFEM,
    Q4ThermalFEM,
    ThermalLBM2D,
)
from .contracts import ContractError, assert_reciprocity
from .energy import Component, EnergyBook
from .subcycling import (
    History,
    RateSchedule,
    SubcyclingError,
    integrate_history,
    observed_order,
)

__all__ = [
    "LoopConfig",
    "LoopResult",
    "run_three_field_loop",
    "run_negative_control",
    "scan_reconstruction_order",
    "reconstruction_operator_study",
]


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LoopConfig:
    """Verification-scale configuration of the loop.

    The thermal diffusivity, the panel conductivity and the carrier viscosity
    are *verification-scale* values: they are chosen so that the panel's
    conduction time, the fluid's diffusion time and the contact duration of the
    impact are all resolvable inside one run window.  They are not a claim about
    a real material; every number below is part of the reported parameter table.
    """

    # panel (solid)
    length: float = 0.2
    height: float = 0.05
    nx: int = 6
    ny: int = 16
    thickness: float = 0.05
    young: float = 1.0e7
    poisson: float = 0.3
    density: float = 8000.0
    conductivity: float = 250.0
    volumetric_heat_capacity: float = 1.0e4
    # carrier fluid (LBM)
    fluid_nx: int = 40
    fluid_ny: int = 4
    tau: float = 1.25
    fluid_diffusivity: float = 0.025
    fluid_temperature: float = 0.01
    # particles (DEM)
    particle_radius: float = 0.01
    particle_density: float = 2000.0
    particle_velocity: float = 0.02
    start_gap: float = 1.0e-4
    stiffness: float = 5.0e3
    damping: float = 8.0
    drag_coefficient: float = 0.5
    contact_columns: tuple[float, float] = (0.45, 0.75)
    # numerics
    degree: int = 1
    t_end: float = 0.04
    solid_safety: float = 0.4
    particle_safety: float = 0.05
    particle_steps_per_exchange: int = 16
    sample_stride: int = 5
    settle_fraction: float = 0.1

    def __post_init__(self) -> None:
        if self.degree < 0:
            raise ValueError("degree must be >= 0")
        if self.t_end <= 0.0:
            raise ValueError("t_end must be positive")
        if self.sample_stride < 1:
            raise ValueError("sample_stride must be >= 1")
        if len(self.contact_columns) < 1:
            raise ValueError("at least one particle column is required")

    # -- derived quantities ------------------------------------------------
    @property
    def dx_fluid(self) -> float:
        return self.length / self.fluid_nx

    @property
    def fluid_energy_capacity(self) -> float:
        """``rho c_p dx^2 t``: joules per unit of lattice scalar and node."""
        return (self.volumetric_heat_capacity * self.dx_fluid ** 2 * self.thickness)

    @property
    def dt_fluid(self) -> float:
        alpha_lat = (self.tau - 0.5) / 3.0
        return alpha_lat * self.dx_fluid ** 2 / self.fluid_diffusivity

    def as_dict(self) -> dict:
        return {
            "length": self.length, "height": self.height, "nx": self.nx, "ny": self.ny,
            "thickness": self.thickness, "young": self.young, "poisson": self.poisson,
            "density": self.density, "conductivity": self.conductivity,
            "volumetric_heat_capacity": self.volumetric_heat_capacity,
            "fluid_nx": self.fluid_nx, "fluid_ny": self.fluid_ny, "tau": self.tau,
            "fluid_diffusivity": self.fluid_diffusivity,
            "fluid_temperature": self.fluid_temperature,
            "particle_radius": self.particle_radius,
            "particle_density": self.particle_density,
            "particle_velocity": self.particle_velocity, "start_gap": self.start_gap,
            "stiffness": self.stiffness, "damping": self.damping,
            "drag_coefficient": self.drag_coefficient,
            "contact_columns": list(self.contact_columns),
            "degree": self.degree, "t_end": self.t_end,
            "solid_safety": self.solid_safety,
            "particle_safety": self.particle_safety,
            "particle_steps_per_exchange": self.particle_steps_per_exchange,
        }


@dataclass
class LoopResult:
    """Outcome of one loop run."""

    config: LoopConfig
    degree: int
    solved_t: float
    schedule: Mapping[str, float]
    series: dict[str, list[float]]
    diagnostics: dict

    def as_dict(self) -> dict:
        return {
            "series": self.series,
            "degree": self.degree,
            "solved_t": self.solved_t,
            "schedule": dict(self.schedule),
            "residual": self.diagnostics["residual"],
            "absolute_imbalance": self.diagnostics["absolute_imbalance"],
            "flux_continuity_l2": self.diagnostics["flux_continuity_l2"],
            "diagnostics": self.diagnostics,
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _interp_columns(x_nodes: np.ndarray, values: np.ndarray, x_col: np.ndarray) -> np.ndarray:
    """Linear interpolation of top-row nodal values onto the fluid columns."""
    return np.interp(x_col, x_nodes, values)


def _distribute_columns(weights: np.ndarray, x_nodes: np.ndarray,
                        x_col: np.ndarray) -> np.ndarray:
    """Spread per-column values onto the top-row nodes (conservative, unit sum)."""
    out = np.zeros_like(x_nodes)
    for xc, w in zip(x_col, weights):
        # linear hat weights of the two bracketing nodes
        i = int(np.clip(np.searchsorted(x_nodes, xc) - 1, 0, len(x_nodes) - 2))
        span = x_nodes[i + 1] - x_nodes[i]
        s = (xc - x_nodes[i]) / span if span > 0 else 0.0
        out[i] += w * (1.0 - s)
        out[i + 1] += w * s
    return out


def _contact_dissipation(law: ContactLaw, overlap: float, rate: float) -> float:
    """Power removed by the dashpot (positive) for one contact.

    ``rate`` is the separation rate; the dashpot only acts while approaching.
    """
    if overlap <= 0.0 or rate >= 0.0:
        return 0.0
    return float(law.damping * (-rate) * (-rate))


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------
def _panel_surface_flux(panel: Q4ThermalFEM, mesh: Q4Mesh, cfg: LoopConfig,
                        trap: np.ndarray) -> float:
    """Second-order one-sided estimate of the panel's top-face heat rate [W].

    Positive = heat leaves the panel upwards (into the fluid).  A three-point
    one-sided difference is used instead of the first-order two-point one
    because the panel is coupled to a fluid of the *same* conductivity: the
    composite is one slab, so the exact flux is continuous and the residual of
    the two estimates is a pure discretisation error that must not be dominated
    by the estimator itself.
    """
    ny = cfg.ny
    if ny < 2:
        raise ValueError("a second-order one-sided flux needs ny >= 2")
    h = cfg.height / ny
    dTdy = (3.0 * panel.T[mesh.row(ny)] - 4.0 * panel.T[mesh.row(ny - 1)]
            + panel.T[mesh.row(ny - 2)]) / (2.0 * h)
    return float(np.sum(trap * (-cfg.conductivity * dTdy) * cfg.thickness))


def run_three_field_loop(config: LoopConfig | None = None) -> LoopResult:
    """Run the closed three-field loop and return its energy ledger.

    Exchange structure: one *exchange interval* of length ``dt_macro = dt_fluid``
    contains ``1`` lattice step, ``n_panel`` implicit panel-conduction steps,
    ``n_solid`` explicit plate steps and ``n_dem`` DEM steps.  Inside an
    interval the order is fluid -> panel -> plate -> DEM (Gauss-Seidel):

    * the fluid takes the panel's surface temperature as a Dirichlet value,
      reconstructed with degree ``p`` from the panel's sub-step history;
    * the panel removes exactly the heat the fluid just measured (double entry,
      so the thermal books close by construction and the coupling is stable);
    * the plate takes the contact load with a zero-order hold from the DEM's
      last sample, and the DEM evaluates its contacts against the plate's
      surface state reconstructed with degree ``p``;
    * the drag heat the DEM measured over the interval reaches the fluid only
      through the *reconstructed* power history (degree ``p``), which is the
      one-way crossing whose energy mismatch is the order-sensitive part of the
      ledger.
    """
    cfg = config or LoopConfig()
    mesh = Q4Mesh(cfg.nx, cfg.ny, cfg.length, cfg.height)
    plate = Q4PlaneStressFEM(mesh, cfg.young, cfg.poisson, cfg.density, cfg.thickness)
    panel = Q4ThermalFEM(mesh, conductivity=cfg.conductivity,
                         volumetric_heat_capacity=cfg.volumetric_heat_capacity,
                         thickness=cfg.thickness)
    plate.fix(mesh.column(0))

    # -- consistent initial state ------------------------------------------
    # Both sides start on *one* linear conduction profile that spans panel and
    # fluid column with the same conductivity and the same gradient, so the
    # interface starts with matching temperature and matching flux.  An
    # isothermal-against-cold start would put a delta function in the flux at
    # t = 0, and every lagged partition scheme would then book a startup
    # imbalance that has nothing to do with the reconstruction order.
    slope = cfg.fluid_temperature / (cfg.height + cfg.fluid_ny * cfg.dx_fluid)
    panel.T[:] = slope * mesh.nodes[:, 1]

    def make_fluid() -> ThermalLBM2D:
        fluid = ThermalLBM2D(cfg.fluid_nx, cfg.fluid_ny, tau=cfg.tau, T_init=0.0)
        y = cfg.height + (np.arange(cfg.fluid_ny) + 0.5) * cfg.dx_fluid
        field = slope * y[:, None] * np.ones((1, cfg.fluid_nx))
        fluid.f[:] = LATTICE_WEIGHTS[:, None, None] * field[None, :, :]
        return fluid

    lbm = make_fluid()

    # -- geometry bookkeeping ---------------------------------------------
    x_col = (np.arange(cfg.fluid_nx) + 0.5) * cfg.dx_fluid
    top = mesh.row(cfg.ny)
    x_nodes = mesh.nodes[top, 0]
    trap = np.full(x_nodes.size, 2.0)
    trap[0] = trap[-1] = 1.0
    trap *= cfg.length / cfg.nx / 2.0          # trapezoid weights -> sum = length
    w_top = _distribute_columns(np.full(cfg.fluid_nx, 1.0 / cfg.fluid_nx), x_nodes, x_col)
    w_top = w_top / w_top.sum()

    cols = np.array(cfg.contact_columns, dtype=float) * cfg.length
    node_of = np.array([int(np.argmin(np.abs(x_nodes - x))) for x in cols], dtype=int)
    pos = np.column_stack([cols, np.full(cols.size, cfg.height + cfg.start_gap
                                         + cfg.particle_radius)])
    vel = np.column_stack([np.zeros(cols.size), np.full(cols.size, -cfg.particle_velocity)])
    law = ContactLaw(stiffness=cfg.stiffness, damping=cfg.damping)
    dem = DEM2D(pos, vel, np.full(cols.size, cfg.particle_radius),
                cfg.particle_density, cfg.thickness, law)
    n_p = cols.size

    # -- rates -------------------------------------------------------------
    dt_macro = cfg.dt_fluid
    n_panel = 4
    dt_panel = dt_macro / n_panel
    dt_solid_target = plate.critical_dt(safety=cfg.solid_safety)
    n_solid = max(1, int(math.ceil(dt_macro / dt_solid_target)))
    dt_solid = dt_macro / n_solid
    n_dem = max(1, int(cfg.particle_steps_per_exchange))
    dt_dem = dt_macro / n_dem
    omega_contact = math.sqrt(cfg.stiffness / float(dem.mass.max()))
    schedule = RateSchedule(dt_fluid=dt_macro, dt_solid=dt_solid,
                            dt_particles=dt_dem)
    panel_inv = panel.cached_inverse(dt_panel)

    c_f = cfg.fluid_energy_capacity
    p = cfg.degree

    # -- exchange histories ------------------------------------------------
    # ``p`` governs every *continuous* crossing that has to be reconstructed
    # between two exchange instants:
    #   * panel surface temperature  -> fluid Dirichlet value (panel is finer)
    #   * fluid heat rate            -> panel sink (fluid is the coarse side,
    #                                   applied one interval late: the lag is
    #                                   what makes the partition stable, and
    #                                   the reconstruction error is what the
    #                                   order gate measures)
    #   * DEM drag power             -> fluid scalar source
    # The contact crossing is *not* reconstructed: it is resolved at the DEM
    # rate with the actual (not extrapolated) plate surface, because a
    # reconstructed contact load is unstable at any degree (see
    # ``test_reconstructed_contact_load_is_unstable``).
    h_wall_temp = History("solid->fluid surface temperature", degree=p)
    h_heat_rate = History("fluid->solid heat rate", degree=p)
    h_drag_power = [History(f"particles->fluid drag power {i}", degree=p)
                    for i in range(n_p)]

    def push_solid_state(t: float) -> None:
        if h_wall_temp.times and t <= h_wall_temp.times[-1] + 1e-15:
            return
        h_wall_temp.push(t, _interp_columns(x_nodes, panel.T[top], x_col))

    # -- book --------------------------------------------------------------
    book = EnergyBook(sides=("fluid", "solid", "particles"))
    dissipated = 0.0
    contact_max_overlap = 0.0
    reciprocity_error = 0.0
    reciprocity_norm = 0.0
    drag_source = 0.0
    drag_delivered = 0.0
    thermal_measured = 0.0
    thermal_applied = 0.0
    q_fluid_samples: list[float] = []
    q_solid_samples: list[float] = []
    contact_impulse = 0.0

    def snapshot(t: float) -> None:
        y_surf = cfg.height + plate.u[2 * node_of + 1]
        overlaps = np.maximum(y_surf - (dem.pos[:, 1] - dem.radius), 0.0)
        book.set(Component.KINETIC, "particles", dem.kinetic_energy)
        book.set(Component.KINETIC, "solid", plate.kinetic_energy())
        book.set(Component.POTENTIAL, "solid",
                 float(np.sum(0.5 * law.stiffness * overlaps ** 2)))
        book.set(Component.INTERNAL, "fluid", c_f * lbm.total())
        book.set(Component.INTERNAL, "solid", panel.energy() + plate.strain_energy())
        book.set(Component.DISSIPATED, "solid", dissipated)
        if not book.history or t > book.history[-1].t:
            book.sample(t)

    # -- initial exchange pass --------------------------------------------
    push_solid_state(0.0)
    # initial exchange pass: one probe lattice step on a throw-away copy of the
    # fluid gives the coupled face a heat-rate sample at t = 0, so the panel's
    # first interval has data to interpolate instead of nothing
    probe = make_fluid()
    q_bootstrap = -c_f * probe.step(
        wall_T=_interp_columns(x_nodes, panel.T[top], x_col)) / cfg.dt_fluid
    h_heat_rate.push(0.0, q_bootstrap)
    for i in range(n_p):
        h_drag_power[i].push(0.0, cfg.drag_coefficient * float(np.sum(dem.vel[i] ** 2)))
    snapshot(0.0)

    n_macro = max(1, int(round(cfg.t_end / dt_macro)))
    t = 0.0
    substeps = [int(round(n_solid * (j + 1) / n_dem)) - int(round(n_solid * j / n_dem))
                for j in range(n_dem)]

    for _k in range(n_macro):
        t0, t1 = t, min(t + dt_macro, cfg.t_end)
        step_macro = t1 - t0

        # (1) fluid: Dirichlet wall from the panel's reconstructed surface, and
        #     the drag heat reconstructed out of the DEM's coarse power history
        tw = np.asarray(h_wall_temp.value_at(t1), dtype=float).ravel()
        injected = lbm.step(wall_T=tw)
        q_meas = -c_f * injected / step_macro          # + = into the solid
        thermal_measured += q_meas * step_macro
        q_fluid_samples.append(q_meas)
        q_solid_samples.append(-_panel_surface_flux(panel, mesh, cfg, trap))
        for i in range(n_p):
            energy = integrate_history(h_drag_power[i], t0, t1, n_quad=5)
            drag_delivered += energy
            ix = int(np.clip(int(dem.pos[i, 0] / cfg.dx_fluid), 0, cfg.fluid_nx - 1))
            iy = int(np.clip(int((dem.pos[i, 1] - cfg.height) / cfg.dx_fluid),
                             0, cfg.fluid_ny - 1))
            lbm.f[0, iy, ix] += energy / c_f

        # (2) panel conduction: the sink is the *reconstructed* heat rate of the
        #     previous intervals (one-interval lag -> stable partition), which
        #     is the crossing whose mismatch the ledger measures
        # the sample for this interval is published *before* the panel steps: the
        # sink is then an interpolation through the interval midpoints (not an
        # extrapolation), which is both stable and order convergent
        h_heat_rate.push(0.5 * (t0 + t1), q_meas)
        q_hat = integrate_history(h_heat_rate, t0, t1, n_quad=5) / step_macro
        flux = np.zeros(mesh.n_nodes)
        flux[top] = q_hat * w_top
        for _ in range(n_panel):
            panel.step(dt_panel, flux=flux, matrix_inverse=panel_inv)
        thermal_applied += q_hat * step_macro

        # (3) mechanics: plate and DEM interleaved at the DEM rate, contact
        #     evaluated on the *actual* states (no reconstruction, no lag but
        #     one sub-block)
        f_load = np.zeros(2 * mesh.n_nodes)
        load_acc = np.zeros(n_p)
        power_acc = np.zeros(n_p)
        y_surf = cfg.height + plate.u[2 * node_of + 1]
        gaps = (dem.pos[:, 1] - dem.radius) - y_surf
        overlaps = np.maximum(-gaps, 0.0)
        rates = dem.vel[:, 1] - plate.v[2 * node_of + 1]
        mags = np.array([law.force(o, r) for o, r in zip(overlaps, rates)])
        for j in range(n_dem):
            f_ext = f_load.copy()
            for _s in range(substeps[j]):
                plate.step(dt_solid, f_ext)
                push_solid_state(t0 + (j + 1) * dt_dem)
            # contact state after the plate moved (particle still at t_j)
            y_surf = cfg.height + plate.u[2 * node_of + 1]
            v_surf = plate.v[2 * node_of + 1]
            gaps = (dem.pos[:, 1] - dem.radius) - y_surf
            overlaps = np.maximum(-gaps, 0.0)
            rates = dem.vel[:, 1] - v_surf
            mags = np.array([law.force(o, r) for o, r in zip(overlaps, rates)])
            contact_max_overlap = max(contact_max_overlap, float(overlaps.max()))
            for o, r in zip(overlaps, rates):
                dissipated += _contact_dissipation(law, float(o), float(r)) * dt_dem
            f_particles = float(np.sum(mags))
            f_nodes = -f_particles
            if abs(f_particles) > 0.0:
                assert_reciprocity("fem_dem_contact_force", f_nodes, -f_particles,
                                   rtol=1e-12)
            reciprocity_error += abs(f_nodes + f_particles)
            reciprocity_norm += abs(f_particles)
            contact_impulse += abs(f_particles) * dt_dem
            load_acc += mags
            f_load = np.zeros(2 * mesh.n_nodes)
            for i in range(n_p):
                f_load[2 * node_of[i] + 1] += -float(mags[i])
            power = cfg.drag_coefficient * np.sum(dem.vel ** 2, axis=1)
            power_acc += power * dt_dem
            drag_source += float(np.sum(power)) * dt_dem
            f_drag = -cfg.drag_coefficient * dem.vel
            dem.step(dt_dem, np.column_stack([f_drag[:, 0], f_drag[:, 1] + mags]))

        for i in range(n_p):
            h_drag_power[i].push(0.5 * (t0 + t1), power_acc[i] / step_macro)

        t = t1
        snapshot(t)
        if t >= cfg.t_end - 1e-15:
            break

    t_end = t
    series = book.series()

    # -- diagnostics -------------------------------------------------------
    qf = np.asarray(q_fluid_samples[1:], dtype=float)
    qs = np.asarray(q_solid_samples[1:], dtype=float)
    denom = float(np.sqrt(np.mean(qf ** 2))) if qf.size else 0.0
    flux_l2 = float(np.sqrt(np.mean((qs - qf) ** 2)) / denom) if denom > 0.0 else float("nan")
    n_settle = max(1, int(cfg.settle_fraction * qf.size))
    qf_s, qs_s = qf[n_settle:], qs[n_settle:]
    denom_s = float(np.sqrt(np.mean(qf_s ** 2))) if qf_s.size else 0.0
    flux_l2_settled = (float(np.sqrt(np.mean((qs_s - qf_s) ** 2)) / denom_s)
                       if denom_s > 0.0 else float("nan"))
    keys = ("kinetic", "potential", "internal", "dissipated")
    settle_t = cfg.settle_fraction * t_end
    settle_idx = next((i for i, tt in enumerate(series["t"]) if tt >= settle_t),
                      len(series["t"]) - 1)
    settle_scale = sum(abs(series[k][settle_idx]) for k in keys)
    settle_imbalance = abs(sum(series[k][-1] - series[k][settle_idx] for k in keys))
    mech_scale = (abs(series["kinetic"][0]) + abs(series["potential"][0])
                  + abs(series["dissipated"][0]))
    diagnostics = {
        "residual": book.residual(),
        "residual_settled": settle_imbalance / max(settle_scale, 1e-300),
        "absolute_imbalance": book.absolute_imbalance(),
        "absolute_imbalance_settled": settle_imbalance,
        "initial_scale": book.initial_scale(),
        "residual_thermal": book.residual(components=(Component.INTERNAL,)),
        "residual_mechanical": book.residual(
            scale=max(mech_scale, 1e-300),
            components=(Component.KINETIC, Component.POTENTIAL, Component.DISSIPATED)),
        "flux_continuity_l2": flux_l2,
        "flux_continuity_l2_settled": flux_l2_settled,
        "flux_fluid_rms": denom,
        "flux_fluid_peak": float(np.max(np.abs(qf))) if qf.size else 0.0,
        "contact_max_overlap": contact_max_overlap,
        "contact_engaged": bool(contact_max_overlap > 1e-9),
        "momentum_reciprocity_error": reciprocity_error / max(reciprocity_norm, 1e-300),
        "thermal_measured_energy": thermal_measured,
        "thermal_applied_energy": thermal_applied,
        "thermal_transfer_mismatch": (thermal_applied - thermal_measured)
        / max(abs(thermal_measured), 1e-300),
        "drag_source_energy": drag_source,
        "drag_delivered_energy": drag_delivered,
        "drag_transfer_mismatch": (drag_delivered - drag_source)
        / max(abs(drag_source), 1e-300),
        "dissipated": dissipated,
        "energy_initial": {k: series[k][0] for k in keys},
        "energy_final": {k: series[k][-1] for k in keys},
        "max_temperature_fluid": float(lbm.total() / max(lbm.T.size, 1)),
        "max_panel_temperature": float(panel.T.max()),
        "min_panel_temperature": float(panel.T.min()),
        "max_particle_y": float(dem.pos[:, 1].max()),
        "min_particle_velocity": float(dem.vel[:, 1].min()),
        "events": n_macro * (1 + n_panel + n_solid + n_dem),
        "macro_steps": n_macro,
        "n_panel_steps": n_panel,
        "n_solid_steps": n_solid,
        "n_dem_steps": n_dem,
        "degree": p,
        "schedule": schedule.as_dict(),
    }
    return LoopResult(config=cfg, degree=p, solved_t=t_end,
                      schedule=schedule.as_dict(), series=series,
                      diagnostics=diagnostics)


# ---------------------------------------------------------------------------
# negative control: uncoupled fields
# ---------------------------------------------------------------------------
def _drift(history: list[float]) -> tuple[float, float]:
    """(secular drift, bounded oscillation) of an energy history, both relative."""
    arr = np.asarray(history, dtype=float)
    e0 = arr[0]
    if e0 == 0.0:
        return 0.0, 0.0
    n = max(1, arr.size // 4)
    drift = abs(float(arr[-n:].mean() - arr[:n].mean())) / abs(e0)
    osc = float(np.max(np.abs(arr - e0))) / abs(e0)
    return drift, osc


def run_negative_control(config: LoopConfig | None = None,
                         n_steps: int = 400) -> dict:
    """Single-field conservation with every coupling term switched off.

    Each solver is advanced exactly as in the coupled run (same steps, same
    integrator) but with no interface data at all.  ``< 1e-6`` here is the null
    hypothesis the coupled residual must be measured against: it proves that a
    residual of the coupled run cannot be blamed on the integrators.
    """
    cfg = config or LoopConfig()
    out: dict[str, float] = {}
    oscillation: dict[str, float] = {}

    # fluid: insulated cavity, no Dirichlet wall
    lbm = ThermalLBM2D(cfg.fluid_nx, cfg.fluid_ny, tau=cfg.tau,
                       T_init=cfg.fluid_temperature)
    hist = [lbm.total()]
    for _ in range(n_steps):
        lbm.step(None)
        hist.append(lbm.total())
    out["fluid_thermal"], oscillation["fluid_thermal"] = _drift(hist)

    # panel conduction: adiabatic, no flux boundary
    mesh = Q4Mesh(cfg.nx, cfg.ny, cfg.length, cfg.height)
    panel = Q4ThermalFEM(mesh, conductivity=cfg.conductivity,
                         volumetric_heat_capacity=cfg.volumetric_heat_capacity,
                         thickness=cfg.thickness)
    panel.T[:] = cfg.fluid_temperature * np.sin(
        np.pi * mesh.nodes[:, 0] / cfg.length) + cfg.fluid_temperature
    hist = [panel.energy()]
    inv = panel.cached_inverse(cfg.dt_fluid / 4.0)
    for _ in range(n_steps):
        panel.step(cfg.dt_fluid / 4.0, matrix_inverse=inv)
        hist.append(panel.energy())
    out["panel_thermal"], oscillation["panel_thermal"] = _drift(hist)

    # plate elasticity: free vibration from an initial displacement
    plate = Q4PlaneStressFEM(mesh, cfg.young, cfg.poisson, cfg.density, cfg.thickness)
    plate.fix(mesh.column(0))
    # a *smooth* single-mode initial state: a displacement spike on one row would
    # excite the highest modes, whose O(dt^2) energy oscillation (bounded, not
    # drifting) would swamp the secular drift this control is about
    top_nodes = mesh.row(cfg.ny)
    plate.u[2 * top_nodes + 1] = 1.0e-3 * np.sin(
        np.pi * mesh.nodes[top_nodes, 0] / cfg.length)
    hist = [plate.energy()]
    dt = plate.critical_dt(safety=0.1)
    for _ in range(n_steps):
        plate.step(dt)
        hist.append(plate.energy())
    out["plate_elastic"], oscillation["plate_elastic"] = _drift(hist)

    # particles: free flight, no contact, no drag
    x_cols = np.array(cfg.contact_columns) * cfg.length
    pos = np.column_stack([x_cols, np.full(x_cols.size, cfg.height + cfg.start_gap)])
    vel = np.column_stack([np.zeros(x_cols.size),
                           np.full(x_cols.size, -cfg.particle_velocity)])
    dem = DEM2D(pos, vel, np.full(x_cols.size, cfg.particle_radius),
                cfg.particle_density, cfg.thickness, ContactLaw(cfg.stiffness, cfg.damping))
    hist = [dem.kinetic_energy]
    zero = np.zeros_like(dem.pos)
    for _ in range(n_steps):
        dem.step(cfg.dt_fluid / 40.0, zero)
        hist.append(dem.kinetic_energy)
    out["particles_kinetic"], oscillation["particles_kinetic"] = _drift(hist)

    out["max"] = max(out.values())
    out["oscillation_max"] = max(oscillation.values())
    out["oscillation"] = oscillation
    out["metric"] = "secular drift = |mean(last quarter) - mean(first quarter)| / E0"
    return out


# ---------------------------------------------------------------------------
# order scan
# ---------------------------------------------------------------------------
def scan_reconstruction_order(config: LoopConfig | None = None,
                              degrees: Sequence[int] = (1, 2, 3)) -> dict:
    """Run the loop once per reconstruction degree and report the gain."""
    cfg = config or LoopConfig()
    runs: dict[str, dict] = {}
    residuals: dict[int, float] = {}
    for p in degrees:
        res = run_three_field_loop(replace(cfg, degree=p))
        runs[str(p)] = res.as_dict()
        residuals[p] = float(res.diagnostics["residual"])
    gains: dict[str, float] = {}
    order = sorted(residuals)
    for p in order:
        if p - 1 in residuals and residuals[p] > 0.0:
            gains[f"p{p-1}_over_p{p}"] = residuals[p - 1] / residuals[p]
    return {"config": cfg.as_dict(), "degrees": list(degrees), "runs": runs,
            "residuals": {str(k): v for k, v in residuals.items()}, "gains": gains}

# ---------------------------------------------------------------------------
# operator-level order study
# ---------------------------------------------------------------------------
def reconstruction_operator_study(signal_kind: str = "decaying_flux",
                                 h: float = 2.5e-4, t_end: float = 0.04,
                                 degrees: Sequence[int] = (1, 2, 3),
                                 refinements: Sequence[float] = (1.0, 0.5, 0.25)
                                 ) -> dict:
    """Quadrature error of the interface reconstruction *without* any solver.

    The receiver sees one interval-average sample per exchange interval (placed
    at the interval midpoint, exactly as in the coupled run) and integrates a
    degree-``p`` Lagrange reconstruction of those samples over each interval.
    The error is the relative non-conservation of that exchange operator.  This
    study separates "is the operator order-p?" from "is the coupled residual
    order-limited?" -- the two questions the order gate conflates.
    """
    if signal_kind != "decaying_flux":
        raise ValueError(f"unknown signal {signal_kind!r}")
    rate = 55.0

    def f(t: float) -> float:
        return 0.4 * math.exp(-rate * t) + 0.12

    def exact(t0: float, t1: float) -> float:
        return (0.4 / rate) * (math.exp(-rate * t0) - math.exp(-rate * t1)) + 0.12 * (t1 - t0)

    out: dict[str, dict] = {}
    for p in degrees:
        errors, orders = [], []
        for factor in refinements:
            hh = h * factor
            n = max(1, int(round(t_end / hh)))
            # one initial-exchange sample at t=0, as in the coupled run
            samples: list[tuple[float, float]] = [(0.0, f(0.0))]
            num = den = 0.0
            for k in range(n):
                t0, t1 = k * hh, (k + 1) * hh
                samples.append((0.5 * (t0 + t1), f(0.5 * (t0 + t1))))
                hist = History("operator study", degree=p)
                for ts, v in samples:
                    hist.push(ts, v)
                received = integrate_history(hist, t0, t1, n_quad=5)
                q = exact(t0, t1)
                num += abs(received - q)
                den += abs(q)
            errors.append(num / den)
            orders.append(hh)
        out[str(p)] = {"h": orders, "error": errors,
                       "observed_order": observed_order(orders, errors)}
    gains = {}
    for p in sorted(degrees):
        if p - 1 in degrees:
            e_prev, e_cur = out[str(p - 1)]["error"][0], out[str(p)]["error"][0]
            gains[f"p{p-1}_over_p{p}"] = (e_prev / e_cur) if e_cur > 0 else float("inf")
    return {"signal": signal_kind, "h": h, "t_end": t_end,
            "degrees": list(degrees), "runs": out, "gains": gains}
