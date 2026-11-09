"""Executable three-field coupling scenarios and their conservation ledgers.

Campaign node ``P40W3C-N09``.  Three couplings are run with *real* discrete
solvers (no synthetic energy histories):

``run_thermal_coupling``
    LBM (D2Q9 passive scalar) fluid over a Q4 transient-conduction plate.
    ``L = 0.1 m`` square cavity, ``dT = 50 K``.  The plate interface temperature
    is the Dirichlet data of the lattice wall; the scalar that wall actually
    injects is *measured* step by step and is the interface heat.  The plate
    receives the exchange reconstructed by the temporal scheme.

``run_mechanical_coupling``
    Q4 plane-stress plate (explicit velocity-Verlet) hit by rigid DEM discs.
    One scalar per contact is booked into both sides, so Newton's third law and
    the momentum budget are structural; the ``independent`` scheme (each side
    evaluates its own contact force) is the negative control that must go red.

``run_three_field``
    Both channels on the same plate plus the full energy chain
    ``E_total = E_fluid + E_solid + E_particles + U_contact``.  This scenario is
    run in *dimensionless* units on purpose (see :class:`ThreeFieldConfig`): a
    diffusive thermal transient is O(s) while contact dynamics is O(us), so a
    dimensionally-honest joint run needs a time-scale study that is out of
    scope here; the conservation contract is scale free and is what is
    verified.  The SI parameters of the corresponding ship-ice-water problem
    are listed in the accompanying document.

Every number the acceptance harness and the tests quote is produced by these
functions, so report, gates and tests cannot drift apart.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .exchange import (SCHEME_ORDER, ExchangeSample, TemporalScheme,
                       reconstruct_interval)
from .ledger import EnergyLedger, MomentumLedger, observed_order
from .solvers import (DEM2D, ContactLaw, Q4Mesh, Q4PlaneStressFEM, Q4ThermalFEM,
                      ThermalLBM2D)

__all__ = [
    "scatter_columns_to_nodes",
    "gather_nodes_to_columns",
    "interval_exchange",
    "ThermalConfig",
    "MechanicalConfig",
    "ThreeFieldConfig",
    "run_thermal_coupling",
    "run_mechanical_coupling",
    "run_three_field",
    "subcycling_scan",
    "order_fit",
    "solver_verification",
]


# ---------------------------------------------------------------------------
# interface operators (1D, conservative scatter / adjoint gather)
# ---------------------------------------------------------------------------
def scatter_columns_to_nodes(col: np.ndarray) -> np.ndarray:
    """Scatter ``nx`` cell-centred values to ``nx+1`` nodes, exactly conserving.

    Each column gives half of its value to each bounding node, so the nodal sum
    equals the column sum to round-off.  This is the flux side of the interface
    operator pair.
    """
    col = np.asarray(col, dtype=float)
    nodes = np.zeros(col.size + 1)
    nodes[:-1] += 0.5 * col
    nodes[1:] += 0.5 * col
    return nodes


def gather_nodes_to_columns(nodes: np.ndarray) -> np.ndarray:
    """Adjoint of :func:`scatter_columns_to_nodes` (consistent interpolation).

    ``<scatter(q), T>_nodes == <q, gather(T)>_cols`` for all ``q, T``, so the
    flux and the Dirichlet data use one and the same duality pairing.
    """
    nodes = np.asarray(nodes, dtype=float)
    return 0.5 * (nodes[:-1] + nodes[1:])


def interval_exchange(scheme: TemporalScheme, previous: ExchangeSample | None,
                      current: ExchangeSample, h: float,
                      t_start: float | None = None) -> float:
    """Exchange handed to the receiver over the interval ``[t_start, t_start+h]``.

    Causality: only samples up to ``current`` are used, so ``zoh`` holds the
    last known rate and ``linear`` extrapolates with the last known slope.
    Evaluating the reconstruction at the *target* window (and not at
    ``[current.t, current.t+h]``) is what makes the extrapolation second order;
    sampling the window half an interval early leaves an O(h) lag.
    """
    scheme = TemporalScheme(scheme)
    if scheme is TemporalScheme.INDEPENDENT:
        raise ValueError("the independent scheme has no shared history")
    if previous is None:
        return current.value * h
    t0 = current.t if t_start is None else float(t_start)
    if t0 < current.t:
        raise ValueError("the exchange window must start at or after the last sample")
    q0 = reconstruct_interval(scheme, previous, current, t0)
    q1 = reconstruct_interval(scheme, previous, current, t0 + h)
    return 0.5 * (q0 + q1) * h


def _nodal_rate(energy: float, h: float, shape: np.ndarray) -> np.ndarray:
    """Split a total interface energy ``energy`` [J] over ``[0, h]`` to nodes.

    The per-column *shape* of the current measured exchange is kept and only
    the total is scaled, so the plate sees the same conservative scatter that
    the fluid's columns would give; the sum over nodes is ``energy / h``.
    """
    shape = np.asarray(shape, dtype=float)
    total = float(shape.sum())
    if abs(total) <= 1e-300:
        shape = np.ones(shape.size)
        total = float(shape.sum())
    return scatter_columns_to_nodes(energy / h * shape / total)


def order_fit(hs: Sequence[float], errors: Sequence[float]) -> dict:
    """Observed order of a refinement series (``order=None`` at round-off)."""
    est = observed_order(hs, errors)
    return est.as_dict()


# ---------------------------------------------------------------------------
# (a) LBM <-> FEM thermal coupling
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ThermalConfig:
    """SI parameters of the 2D cavity / plate thermal problem.

    The lattice diffusivity is ``(tau-1/2)/3``; ``dt`` follows from requiring
    the physical diffusivity ``alpha_fluid``, i.e. ``dt = alpha_lat dx^2/alpha``.
    The fluid's effective conductivity ``alpha*rho_c`` (0.588 W/m/K for the
    defaults, water-like) is the value the plate has to see at the interface.
    """

    grid: int = 20                 # lattice columns == plate elements in x
    solid_rows: int = 4            # plate elements in y
    length: float = 0.1            # L [m]
    delta_T: float = 50.0          # initial fluid-solid temperature gap [K]
    t_cold: float = 300.0          # plate initial temperature [K]
    tau: float = 1.0               # LBM relaxation time
    alpha_fluid: float = 1.4e-7    # m^2/s
    rho_c_fluid: float = 4.2e5     # J/m^3/K (water/10: keeps the plate conduction-limited)
    k_solid: float = 6.0           # W/m/K
    rho_c_solid: float = 4.0e6     # J/m^3/K
    solid_thickness: float = 0.05  # m  (massive plate: staggered gain < 1)
    heater_watts: float = 0.0      # optional periodic external load [W]
    heater_period_steps: float = 320.0
    n_fine_steps: int = 96

    @property
    def dx(self) -> float:
        return self.length / self.grid

    @property
    def alpha_lattice(self) -> float:
        return (self.tau - 0.5) / 3.0

    @property
    def dt(self) -> float:
        return self.alpha_lattice * self.dx ** 2 / self.alpha_fluid

    @property
    def t_hot(self) -> float:
        return self.t_cold + self.delta_T

    @property
    def heater_period(self) -> float:
        return self.heater_period_steps * self.dt

    def heater_rate(self, t: float) -> float:
        return self.heater_watts * math.sin(2.0 * math.pi * t / self.heater_period)


def run_thermal_coupling(ratio: int = 1,
                         scheme: TemporalScheme | str = TemporalScheme.ZOH,
                         config: ThermalConfig | None = None,
                         n_fine_steps: int | None = None,
                         keep_fields: bool = False) -> dict:
    """Run the LBM<->FEM thermal coupling and return its energy ledger.

    Headline numbers
    ----------------
    ``residual_total``
        ``|E_total(T) - E_total(0)| / |E_total(0)|`` of the coupled system,
        with the periodic external load booked as an external input.
    ``exchange_mismatch``
        ``sum_k |E_booked,k - E_actual,k| / sum_k |E_actual,k|``: the relative
        heat the coupling operator moved without the two sides agreeing.
    ``flux_consistency``
        ``|Q_fluid - Q_solid| / |Q_fluid|`` on the time-averaged interface heat
        rate, ``Q_fluid`` measured from the injected lattice scalar, ``Q_solid``
        from the plate's one-sided Fourier flux.
    """
    cfg = config or ThermalConfig()
    scheme = TemporalScheme(scheme)
    dx, dt = cfg.dx, cfg.dt
    n_fine = int(n_fine_steps if n_fine_steps is not None else cfg.n_fine_steps)
    h = ratio * dt
    n_coarse = max(4, int(round(n_fine / ratio)))

    lbm = ThermalLBM2D(cfg.grid, cfg.grid, tau=cfg.tau, T_init=cfg.t_hot)
    mesh = Q4Mesh(cfg.grid, cfg.solid_rows, cfg.length, cfg.solid_thickness)
    plate = Q4ThermalFEM(mesh, cfg.k_solid, cfg.rho_c_solid, thickness=1.0,
                         T_init=cfg.t_cold)
    inv = plate.cached_inverse(h)
    top = mesh.row(cfg.solid_rows)
    band = cfg.solid_thickness / cfg.solid_rows
    bottom = mesh.row(0)
    e_scale = cfg.rho_c_fluid * dx * dx

    ledger = EnergyLedger(("fluid", "solid"))
    e_ext = 0.0
    samples: list[ExchangeSample] = []
    mismatch = 0.0
    exchanged = 0.0
    mismatch_late = 0.0
    exchanged_late = 0.0
    late_from = n_coarse // 2
    rows: list[dict] = []

    for k in range(n_coarse):
        t0 = k * h
        # --- fluid: Dirichlet wall temperature from the plate interface ------
        tw_col = gather_nodes_to_columns(plate.T[top])
        inj = np.zeros(cfg.grid)
        for _ in range(ratio):
            inj += lbm.step(wall_T=tw_col)
        e_act = float(e_scale * inj.sum())
        q_fluid = e_act / h
        q_lbm_est = -float(e_scale * lbm.nonequilibrium_wall_flux().sum() / dt)
        # positive = heat leaving the plate into the fluid (same convention as
        # ``q_fluid``, which is the fluid's gain measured from the injection)
        q_solid_est = float(plate.interface_flux_estimate(
            top, dx, band, below_row=mesh.row(cfg.solid_rows - 1)).sum())

        # Causal reconstruction: at t0 the plate only knows the *completed*
        # intervals, so the last sample is the mean rate of interval k-1.  The
        # very first interval has no shared history and is booked from the
        # fluid's own measurement (documented start-up, one interval only).
        if scheme is TemporalScheme.INDEPENDENT:
            # classic Dirichlet-Dirichlet: the plate books its own flux
            e_book = q_solid_est * h
        elif not samples:
            e_book = e_act
        else:
            cur = samples[-1]
            prev = samples[-2] if len(samples) >= 2 else None
            e_book = interval_exchange(scheme, prev, cur, h, t_start=t0)

        # --- solid: Neumann rate on the interface + periodic external load ---
        heater_energy = 0.5 * (cfg.heater_rate(t0) + cfg.heater_rate(t0 + h)) * h
        e_ext += heater_energy
        b = np.zeros(mesh.n_nodes)
        b[top] += _nodal_rate(-e_book, h, inj)
        b[bottom] += heater_energy / h / bottom.size
        plate.step(h, flux=b, matrix_inverse=inv)

        mismatch += abs(e_book - e_act)
        exchanged += abs(e_act)
        if k >= late_from:
            mismatch_late += abs(e_book - e_act)
            exchanged_late += abs(e_act)
        # the interval mean is a midpoint sample: no built-in lag
        samples.append(ExchangeSample(t0 + 0.5 * h, q_fluid))

        ledger.set("fluid", e_scale * float(lbm.T.sum()))
        ledger.set("solid", plate.energy())
        ledger.book_external(heater_energy)   # book the increment, not the sum
        s = ledger.sample(t0 + h)
        rows.append({
            "t": t0 + h,
            "e_fluid": s.sides["fluid"],
            "e_solid": s.sides["solid"],
            "e_total": s.total,
            "q_fluid": q_fluid,
            "q_booked": e_book / h,
            "q_solid_est": q_solid_est,
            "q_lbm_est": q_lbm_est,
            "residual": abs(s.total - ledger.history[0].total)
            / abs(ledger.history[0].total),
        })

    half = rows[len(rows) // 2:]
    q_f = float(np.mean([r["q_fluid"] for r in half]))
    q_s = float(np.mean([r["q_solid_est"] for r in half]))
    q_l = float(np.mean([r["q_lbm_est"] for r in half]))
    denom = abs(q_f) if abs(q_f) > 0 else 1.0

    out = {
        "scenario": "lbm_fem_thermal",
        "ratio": int(ratio),
        "scheme": scheme.value,
        "scheme_order": SCHEME_ORDER[scheme],
        "grid": cfg.grid,
        "dt": dt,
        "h": h,
        "n_coarse": n_coarse,
        "n_fine_steps": n_coarse * ratio,
        "t_end": n_coarse * h,
        "delta_T": cfg.delta_T,
        "length": cfg.length,
        "e_total_0": ledger.history[0].total,
        "e_total_end": ledger.history[-1].total,
        "residual_total": ledger.residual(),
        "residual_max": ledger.max_residual(),
        "drift_slope": ledger.drift_slope(),
        "exchange_mismatch": mismatch / exchanged if exchanged else 0.0,
        "exchange_mismatch_late": (mismatch_late / exchanged_late
                                   if exchanged_late else 0.0),
        "q_fluid_mean": q_f,
        "q_solid_mean": q_s,
        "q_lbm_mean": q_l,
        "flux_consistency": abs(q_f - q_s) / denom,
        "lbm_flux_consistency": abs(q_f - q_l) / denom,
        "history": rows,
    }
    if keep_fields:
        out["field_fluid_T"] = lbm.T.tolist()
        out["field_solid_T"] = plate.T.reshape(
            cfg.grid + 1, cfg.solid_rows + 1).tolist()
    return out


# ---------------------------------------------------------------------------
# (b) FEM <-> DEM mechanical coupling
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MechanicalConfig:
    """Free-free elastic plate hit by rigid discs (SI)."""

    nx: int = 20
    ny: int = 2
    lx: float = 0.2
    ly: float = 0.02
    young: float = 2.0e9
    poisson: float = 0.3
    density: float = 900.0
    thickness: float = 1.0
    particle_radius: float = 1.0e-3
    particle_density: float = 900.0
    particle_speed: float = 2.0
    n_particles: int = 2
    stiffness: float = 2.0e6
    damping: float = 0.0
    n_fine_steps: int = 1200
    safety: float = 0.4
    impact_fraction: float = 0.35

    @property
    def particle_mass(self) -> float:
        return self.particle_density * self.thickness * math.pi * self.particle_radius ** 2

    @property
    def contact_omega(self) -> float:
        return math.sqrt(self.stiffness / self.particle_mass)


def run_mechanical_coupling(ratio: int = 1,
                            scheme: TemporalScheme | str = TemporalScheme.ZOH,
                            config: MechanicalConfig | None = None,
                            n_fine_steps: int | None = None) -> dict:
    """Run the FEM<->DEM impact and return its momentum/energy ledger."""
    cfg = config or MechanicalConfig()
    scheme = TemporalScheme(scheme)
    n_fine = int(n_fine_steps if n_fine_steps is not None else cfg.n_fine_steps)

    mesh = Q4Mesh(cfg.nx, cfg.ny, cfg.lx, cfg.ly)
    plate = Q4PlaneStressFEM(mesh, cfg.young, cfg.poisson, cfg.density,
                             thickness=cfg.thickness)
    dt = plate.critical_dt(safety=cfg.safety)
    h = ratio * dt
    n_coarse = max(4, int(round(n_fine / ratio)))
    edge = mesh.column(cfg.nx)

    law = ContactLaw(stiffness=cfg.stiffness, damping=cfg.damping)
    r = cfg.particle_radius
    t_run = n_coarse * h
    gap = cfg.particle_speed * cfg.impact_fraction * t_run
    pos = np.array([[cfg.lx + r + gap, cfg.ly * (0.35 + 0.3 * i)]
                    for i in range(cfg.n_particles)])
    vel = np.array([[-cfg.particle_speed, 0.0] for _ in range(cfg.n_particles)])
    dem = DEM2D(pos, vel, np.full(cfg.n_particles, r), cfg.particle_density,
                thickness=cfg.thickness, law=law)
    node_of = np.linspace(0, edge.size - 1, cfg.n_particles).round().astype(int)

    omega = cfg.contact_omega
    contact_dt = 0.2 / omega
    stable = dt <= contact_dt

    p_ledger = MomentumLedger(("solid", "particles"),
                              scale=cfg.particle_mass * cfg.particle_speed)
    # seed the accounts *before* the first record, otherwise the initial total
    # is taken as zero and the reported drift is the whole initial momentum
    p_ledger.set("solid", 0.0, 0.0)
    p_ledger.set("particles", float(dem.momentum()[0]), float(dem.momentum()[1]))
    p_ledger.record()
    e0 = plate.energy() + dem.kinetic_energy
    e_scale = max(abs(e0), 1e-30)
    e_hist: list[float] = []
    rows: list[dict] = []
    force_hist: list[float] = []
    forced_prev: np.ndarray | None = None
    mismatch = 0.0
    exchanged = 0.0
    antisym = 0.0
    antisym_scale = 0.0
    peak_force = 0.0

    for k in range(n_coarse):
        t0 = k * h
        wall_x = mesh.nodes[edge, 0] + plate.u[2 * edge]
        wall_vx = plate.v[2 * edge]
        wall_y = mesh.nodes[edge, 1] + plate.u[2 * edge + 1]
        f_p, f_nodes, overlaps, _ = dem.contact_forces(wall_x, wall_vx, wall_y, node_of)
        total_f = float(f_p[:, 0].sum())
        peak_force = max(peak_force, float(np.abs(f_p[:, 0]).max()))

        if scheme is TemporalScheme.INDEPENDENT:
            # loose (asynchronous) coupling: the discs get the current contact
            # force, the plate gets the force of the *previous* exchange.  The
            # two impulses therefore differ by O(h dF/dt) and Newton's third law
            # fails on the discrete level -- this is the negative control.
            f_on_particles = f_p
            f_on_plate = -(forced_prev if forced_prev is not None
                           else np.zeros((edge.size, 2)))
            _ = f_nodes
        else:
            if scheme is TemporalScheme.LINEAR and len(force_hist) >= 1:
                prev = ExchangeSample(t0 - h, force_hist[-1])
                cur = ExchangeSample(t0, total_f)
                q0 = reconstruct_interval(scheme, prev, cur, t0)
                q1 = reconstruct_interval(scheme, prev, cur, t0 + h)
                f_eq = 0.5 * (q0 + q1)
            else:
                f_eq = total_f
            # one and the same magnitude for both sides => Newton-consistent
            f_on_particles = np.column_stack([np.full(cfg.n_particles, f_eq / cfg.n_particles),
                                              np.zeros(cfg.n_particles)])
            f_on_plate = np.zeros((edge.size, 2))
            np.add.at(f_on_plate, node_of,
                      np.column_stack([np.full(cfg.n_particles, -f_eq / cfg.n_particles),
                                       np.zeros(cfg.n_particles)]))
        force_hist.append(total_f)
        forced_prev = np.zeros((edge.size, 2))
        np.add.at(forced_prev, node_of,
                  np.column_stack([np.full(cfg.n_particles,
                                           -total_f / cfg.n_particles),
                                   np.zeros(cfg.n_particles)]))

        f_ext = np.zeros(2 * mesh.n_nodes)
        for j, node in enumerate(edge):
            f_ext[2 * node] += f_on_plate[j, 0]
            f_ext[2 * node + 1] += f_on_plate[j, 1]
        impulse_p = float(np.sum(f_on_particles[:, 0])) * ratio
        impulse_s = float(np.sum(f_on_plate[:, 0])) * ratio
        mismatch += abs(impulse_p + impulse_s)
        exchanged += max(abs(impulse_p), 1e-30)
        antisym += abs(impulse_p + impulse_s)
        antisym_scale += max(abs(impulse_p), 1e-30)

        for _ in range(ratio):
            plate.step(dt, f_ext=f_ext)
            dem.step(dt, f_on_particles)

        p_ledger.set("solid", float(np.dot(plate.mass[0::2], plate.v[0::2])),
                     float(np.dot(plate.mass[1::2], plate.v[1::2])))
        p_ledger.set("particles", float(dem.momentum()[0]), float(dem.momentum()[1]))
        p_ledger.record()

        wall_x = mesh.nodes[edge, 0] + plate.u[2 * edge]
        _, _, ov2, _ = dem.contact_forces(wall_x, plate.v[2 * edge],
                                          mesh.nodes[edge, 1], node_of)
        u_contact = float(sum(law.potential(o) for o in ov2))
        e = plate.energy() + dem.kinetic_energy + u_contact
        e_hist.append(e)
        rows.append({
            "t": t0 + h,
            "p_residual": p_ledger.residual(),
            "e_total": e,
            "e_plate_strain": plate.strain_energy(),
            "e_plate_kinetic": plate.kinetic_energy(),
            "e_particles": dem.kinetic_energy,
            "u_contact": u_contact,
            "force": total_f,
            "overlap": float(np.max(ov2)),
        })

    return {
        "scenario": "fem_dem_mechanical",
        "ratio": int(ratio),
        "scheme": scheme.value,
        "scheme_order": SCHEME_ORDER[scheme],
        "dt": dt,
        "h": h,
        "n_coarse": n_coarse,
        "n_fine_steps": n_coarse * ratio,
        "t_end": n_coarse * h,
        "contact_period": 2.0 * math.pi / omega,
        "contact_dt_limit": contact_dt,
        "explicitly_stable": stable,
        "peak_contact_force": peak_force,
        "e_total_0": e0,
        "momentum_residual": p_ledger.residual(),
        "energy_residual": abs(e_hist[-1] - e0) / e_scale,
        "energy_residual_max": max(abs(x - e0) for x in e_hist) / e_scale,
        "force_antisymmetry": antisym / antisym_scale if antisym_scale else 0.0,
        "impulse_mismatch": mismatch / exchanged if exchanged else 0.0,
        "history": rows,
    }


# ---------------------------------------------------------------------------
# (c) three-field chain LBM <-> FEM <-> DEM
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ThreeFieldConfig:
    """Dimensionless three-field verification case (see the module docstring).

    ``dx = 1``, ``dt_lbm = 1``, ``alpha_lat = 1/6`` for both the lattice and the
    plate (``k_solid/rho_c_solid = 1/6``), plate wave speed ``sqrt(E/rho)=1.2``.
    A physical ship-ice-water instance keeps the same dimensionless groups; the
    SI values of the corresponding problem are given in the report.
    """

    grid: int = 16
    solid_rows: int = 2
    t_hot: float = 350.0
    t_cold: float = 300.0
    tau: float = 1.0
    rho_c_fluid: float = 1.0
    rho_c_solid: float = 1.0
    k_solid: float = 1.0 / 6.0
    density: float = 1.0
    young: float = 1.44
    poisson: float = 0.3
    thickness: float = 1.0
    particle_radius: float = 0.5
    particle_speed: float = 0.05
    stiffness: float = 10.0
    n_lbm_steps: int = 96
    impact_fraction: float = 0.45


def run_three_field(ratio: int = 1,
                    scheme: TemporalScheme | str = TemporalScheme.ZOH,
                    config: ThreeFieldConfig | None = None,
                    keep_fields: bool = False) -> dict:
    """Full energy chain ``E_fluid + E_solid + E_particles + U_contact``."""
    cfg = config or ThreeFieldConfig()
    scheme = TemporalScheme(scheme)
    ratios = int(ratio)
    n_exchange = max(4, int(round(cfg.n_lbm_steps / ratios)))
    h = float(ratios)                      # dt_lbm = 1, dx = 1
    dx = 1.0

    lbm = ThermalLBM2D(cfg.grid, cfg.grid, tau=cfg.tau, T_init=cfg.t_hot)
    mesh = Q4Mesh(cfg.grid, cfg.solid_rows, float(cfg.grid), float(cfg.solid_rows))
    thermal = Q4ThermalFEM(mesh, cfg.k_solid, cfg.rho_c_solid,
                           thickness=cfg.thickness, T_init=cfg.t_cold)
    mechanical = Q4PlaneStressFEM(mesh, cfg.young, cfg.poisson, cfg.density,
                                  thickness=cfg.thickness)
    inv = thermal.cached_inverse(h)
    top = mesh.row(cfg.solid_rows)
    edge = mesh.column(cfg.grid)
    band = 1.0

    r_p = cfg.particle_radius
    law = ContactLaw(stiffness=cfg.stiffness, damping=0.0)
    m_p = cfg.density * cfg.thickness * math.pi * r_p * r_p
    omega_c = math.sqrt(cfg.stiffness / m_p)
    dt_crit_plate = mechanical.critical_dt(safety=1.0)
    n_mech = max(1, int(math.ceil(1.0 / (0.5 * min(dt_crit_plate, 0.4 / omega_c)))))
    dt_mech = 1.0 / n_mech                 # integer substeps per lattice step

    t_run = n_exchange * h
    gap = cfg.particle_speed * cfg.impact_fraction * t_run
    pos = np.array([[cfg.grid + r_p + gap, 1.6 * r_p],
                    [cfg.grid + r_p + gap + 2.0 * r_p, 0.6 * r_p]])
    vel = np.array([[-cfg.particle_speed, 0.0], [-cfg.particle_speed, 0.0]])
    dem = DEM2D(pos, vel, np.full(2, r_p), cfg.density, thickness=cfg.thickness,
                law=law)
    node_of = np.array([edge.size // 2, edge.size // 2 + 1])

    ledger = EnergyLedger(("fluid", "solid", "particles", "contact"))
    samples: list[ExchangeSample] = []
    mismatch = 0.0
    exchanged = 0.0
    forced_prev: np.ndarray | None = None
    m_mismatch = 0.0
    m_exchanged = 0.0
    mismatch_late = 0.0
    exchanged_late = 0.0
    late_from = n_exchange // 2
    rows: list[dict] = []

    for k in range(n_exchange):
        t0 = k * h
        # --- thermal channel --------------------------------------------------
        tw_col = gather_nodes_to_columns(thermal.T[top])
        inj = np.zeros(cfg.grid)
        for _ in range(ratios):
            inj += lbm.step(wall_T=tw_col)
        e_act = float(cfg.rho_c_fluid * dx * dx * inj.sum())
        if scheme is TemporalScheme.INDEPENDENT:
            q_solid = float(thermal.interface_flux_estimate(
                top, dx, band, below_row=mesh.row(cfg.solid_rows - 1)).sum())
            e_book = q_solid * h
        elif not samples:
            e_book = e_act
        else:
            cur = samples[-1]
            prev = samples[-2] if len(samples) >= 2 else None
            e_book = interval_exchange(scheme, prev, cur, h, t_start=t0)
        samples.append(ExchangeSample(t0 + 0.5 * h, e_act / h))
        mismatch += abs(e_book - e_act)
        exchanged += abs(e_act)
        if k >= late_from:
            mismatch_late += abs(e_book - e_act)
            exchanged_late += abs(e_act)
        b = np.zeros(mesh.n_nodes)
        b[top] += _nodal_rate(-e_book, h, inj)
        thermal.step(h, flux=b, matrix_inverse=inv)

        # --- mechanical channel ----------------------------------------------
        wall_x = mesh.nodes[edge, 0] + mechanical.u[2 * edge]
        wall_vx = mechanical.v[2 * edge]
        wall_y = mesh.nodes[edge, 1] + mechanical.u[2 * edge + 1]
        f_p, f_nodes, ov, _ = dem.contact_forces(wall_x, wall_vx, wall_y, node_of)
        total_f = float(f_p[:, 0].sum())
        if scheme is TemporalScheme.INDEPENDENT:
            f_on_particles = f_p
            f_on_plate = (forced_prev if forced_prev is not None
                          else np.zeros((edge.size, 2)))
        else:
            f_on_particles = np.column_stack([np.full(2, 0.5 * total_f),
                                              np.zeros(2)])
            f_on_plate = np.zeros((edge.size, 2))
            np.add.at(f_on_plate, node_of,
                      np.column_stack([np.full(2, -0.5 * total_f), np.zeros(2)]))
        forced_prev = np.zeros((edge.size, 2))
        np.add.at(forced_prev, node_of,
                  np.column_stack([np.full(2, -0.5 * total_f), np.zeros(2)]))
        m_mismatch += abs(float(f_on_particles[:, 0].sum())
                          + float(f_on_plate[:, 0].sum()))
        m_exchanged += max(abs(total_f), 1e-30)

        f_ext = np.zeros(2 * mesh.n_nodes)
        for j, node in enumerate(edge):
            f_ext[2 * node] += f_on_plate[j, 0]
            f_ext[2 * node + 1] += f_on_plate[j, 1]
        for _ in range(ratios * n_mech):
            mechanical.step(dt_mech, f_ext=f_ext)
            dem.step(dt_mech, f_on_particles)

        wall_x = mesh.nodes[edge, 0] + mechanical.u[2 * edge]
        _, _, ov2, _ = dem.contact_forces(wall_x, mechanical.v[2 * edge],
                                          mesh.nodes[edge, 1], node_of)
        u_contact = float(sum(law.potential(o) for o in ov2))
        ledger.set("fluid", cfg.rho_c_fluid * dx * dx * float(lbm.T.sum()))
        ledger.set("solid", thermal.energy() + mechanical.energy())
        ledger.set("particles", dem.kinetic_energy)
        ledger.set("contact", u_contact)
        s = ledger.sample(t0 + h)
        rows.append({
            "t": t0 + h,
            "e_fluid": s.sides["fluid"],
            "e_solid": s.sides["solid"],
            "e_particles": s.sides["particles"],
            "u_contact": s.sides["contact"],
            "e_total": s.total,
            "residual": abs(s.total - ledger.history[0].total)
            / abs(ledger.history[0].total),
            "q_fluid": e_act / h,
            "q_booked": e_book / h,
        })

    out = {
        "scenario": "three_field_chain",
        "ratio": int(ratio),
        "scheme": scheme.value,
        "scheme_order": SCHEME_ORDER[scheme],
        "h": h,
        "dt_mech": dt_mech,
        "mech_substeps_per_lattice": n_mech,
        "n_exchange": n_exchange,
        "n_lbm_steps": n_exchange * ratios,
        "t_end": n_exchange * h,
        "e_total_0": ledger.history[0].total,
        "residual_total": ledger.residual(),
        "residual_max": ledger.max_residual(),
        "drift_slope": ledger.drift_slope(),
        "exchange_mismatch": mismatch / exchanged if exchanged else 0.0,
        "exchange_mismatch_late": (mismatch_late / exchanged_late
                                   if exchanged_late else 0.0),
        "momentum_mismatch": m_mismatch / m_exchanged if m_exchanged else 0.0,
        "history": rows,
    }
    if keep_fields:
        out["field_fluid_T"] = lbm.T.tolist()
        out["field_solid_T"] = thermal.T.reshape(
            cfg.grid + 1, cfg.solid_rows + 1).tolist()
    return out


# ---------------------------------------------------------------------------
# scans
# ---------------------------------------------------------------------------
def subcycling_scan(ratios: Sequence[int] = (1, 2, 4, 8),
                    schemes: Sequence[TemporalScheme | str] = (
                        TemporalScheme.INDEPENDENT, TemporalScheme.ZOH,
                        TemporalScheme.LINEAR),
                    thermal: ThermalConfig | None = None,
                    mechanical: MechanicalConfig | None = None,
                    three_field: ThreeFieldConfig | None = None,
                    n_fine_steps: int | None = None,
                    progress: bool = False) -> dict:
    """Full ratio x scheme matrix for all three scenarios (JSON-ready)."""
    rows = {"thermal": [], "mechanical": [], "three_field": []}
    for ratio in ratios:
        for scheme in schemes:
            name = TemporalScheme(scheme).value
            tr = run_thermal_coupling(ratio, scheme, config=thermal,
                                      n_fine_steps=n_fine_steps)
            mr = run_mechanical_coupling(ratio, scheme, config=mechanical,
                                         n_fine_steps=n_fine_steps)
            cr = run_three_field(ratio, scheme, config=three_field)
            for key, res in (("thermal", tr), ("mechanical", mr), ("three_field", cr)):
                rows[key].append({
                    "ratio": res["ratio"],
                    "scheme": name,
                    "scheme_order": res["scheme_order"],
                    "h": res["h"],
                    "residual_total": res.get("residual_total",
                                              res.get("momentum_residual")),
                    "exchange_mismatch": res.get("exchange_mismatch",
                                                 res.get("impulse_mismatch")),
                    "energy_residual": res.get("energy_residual"),
                    "momentum_residual": res.get("momentum_residual"),
                    "flux_consistency": res.get("flux_consistency"),
                })
            if progress:
                print(f"  ratio={ratio} scheme={name} done", flush=True)
    return rows


# ---------------------------------------------------------------------------
# solver-level verification (kernels, independent of the coupling)
# ---------------------------------------------------------------------------
def solver_verification(grids: Sequence[int] = (16, 32, 48),
                        steps: int = 200) -> dict:
    """Verify each kernel against an analytic reference.

    * lattice: initial ``cos(pi x / N)`` decays as ``exp(-alpha pi^2 t / N^2)``
      (adiabatic walls, exact eigenmode) -> L2 error vs the closed form;
    * plate: a flux dipole ``(+Q, -Q)`` on the top/bottom row relaxes to the
      exact linear profile ``dT = Q t_s/(k A)`` and the discrete energy identity
      ``dE = dt * sum(b)`` must hold to round-off;
    * DEM: single disc on a fixed wall, undamped -> contact time ``pi/omega``
      and unit restitution.
    """
    from .solvers import Q4ThermalFEM

    # -- lattice diffusion ---------------------------------------------------
    lbm_errs, lbm_steps = [], []
    for n in grids:
        sim = ThermalLBM2D(n, n, tau=1.0, T_init=1.0)
        x = np.arange(n) + 0.5
        amp = np.cos(math.pi * x / n)
        sim.f[:] = _equilibrium(1.0 + amp[None, :] * np.ones((n, 1)))
        alpha = sim.diffusivity
        exact_t = steps
        decay = math.exp(-alpha * math.pi ** 2 * exact_t / n ** 2)
        for _ in range(steps):
            sim.step(None)
        ref = 1.0 + amp[None, :] * np.ones((n, 1)) * decay
        num = sim.T
        lbm_errs.append(float(np.sqrt(np.mean((num - ref) ** 2))
                              / np.sqrt(np.mean((ref - 1.0) ** 2))))
        lbm_steps.append(1.0 / n)

    # -- plate conduction ----------------------------------------------------
    fem_rows = []
    for n in grids:
        mesh = Q4Mesh(n, 3, 1.0, 0.06)
        fem = Q4ThermalFEM(mesh, conductivity=0.6,
                           volumetric_heat_capacity=2.0e6,
                           thickness=1.0, T_init=300.0)
        dt = 1000.0                                # unconditionally stable
        inv = fem.cached_inverse(dt)
        q_watt = 25.0
        top, bottom = mesh.row(3), mesh.row(0)
        b = np.zeros(mesh.n_nodes)
        b[top] += q_watt / top.size
        b[bottom] -= q_watt / bottom.size
        e_before = fem.energy()
        fem.step(dt, flux=b, matrix_inverse=inv)
        dE_expected = dt * float(b.sum())          # exact for lumped capacity
        dE_actual = fem.energy() - e_before
        for _ in range(40):
            fem.step(dt, flux=b, matrix_inverse=inv)
        # exact steady state of the flux dipole: q = Q/(lx t), dT = q ly / k
        dT_ref = q_watt * mesh.ly / (0.6 * mesh.lx * 1.0)
        dT_num = float(np.mean(fem.T[top]) - np.mean(fem.T[bottom]))
        fem_rows.append({
            "grid": n,
            # normalise by the *absolute* flux budget: the dipole has zero net
            # input, so |dE_expected| alone is not a usable scale
            "energy_identity_error": abs(dE_actual - dE_expected)
            / max(dt * float(np.abs(b).sum()), 1e-300),
            "delta_T_numeric": dT_num,
            "delta_T_reference": dT_ref,
            "flux_error": abs(dT_num - dT_ref) / abs(dT_ref),
        })

    # -- DEM contact against a fixed wall ------------------------------------
    cfg = MechanicalConfig()
    mesh = Q4Mesh(4, 1, cfg.lx, cfg.ly)
    plate = Q4PlaneStressFEM(mesh, cfg.young, cfg.poisson, cfg.density,
                             thickness=cfg.thickness)
    plate.fix(range(mesh.n_nodes))
    r = cfg.particle_radius
    law = ContactLaw(stiffness=cfg.stiffness, damping=0.0)
    gap = 1e-5
    dem = DEM2D(np.array([[cfg.lx + r + gap, cfg.ly * 0.5]]),
                np.array([[-cfg.particle_speed, 0.0]]), np.array([r]),
                cfg.particle_density, thickness=cfg.thickness, law=law)
    dt = 0.2 / cfg.contact_omega
    edge = mesh.column(4)
    node_of = np.array([edge.size // 2])
    contact_steps = 0
    e_hist = [dem.kinetic_energy]
    wall_x = mesh.nodes[edge, 0]
    wall_vx = np.zeros(edge.size)
    wall_y = mesh.nodes[edge, 1]

    def contact_force(p, v):
        return dem.contact_forces(wall_x, wall_vx, wall_y, node_of)[0]

    for _ in range(int(6 * math.pi / cfg.contact_omega / dt)):
        f_p, _, ov, _ = dem.contact_forces(wall_x, wall_vx, wall_y, node_of)
        if float(ov.max()) > 0.0:
            contact_steps += 1
        dem.step(dt, f_p, force_fn=contact_force)
        e_hist.append(dem.kinetic_energy)
    exit_speed = -float(dem.vel[0, 0])

    e_scale = max(abs(e_hist[0]), 1e-300)
    return {
        "lattice_diffusion": {
            "grids": list(grids),
            "relative_l2": lbm_errs,
            "order": order_fit(lbm_steps, lbm_errs),
            "steps": steps,
        },
        "plate_conduction": fem_rows,
        "dem_contact": {
            "contact_time_numeric": contact_steps * dt,
            "contact_time_reference": math.pi / cfg.contact_omega,
            "restitution": exit_speed / cfg.particle_speed,
            "kinetic_energy_drift": abs(e_hist[-1] - e_hist[0]) / e_scale,
            "kinetic_energy_swing": max(abs(e - e_hist[0]) for e in e_hist) / e_scale,
        },
    }


def _equilibrium(T: np.ndarray) -> np.ndarray:
    """D2Q9 equilibrium populations for a scalar field ``T`` (zeros velocity)."""
    from .solvers import _W
    return _W[:, None, None] * T[None, :, :]