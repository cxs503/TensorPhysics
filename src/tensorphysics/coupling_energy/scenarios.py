"""Three cross-solver energy-conservation scenarios (campaign node P40W3B-N09).

The three experiments share one contract: at every exchange instant the
conserved quantity is measured **from the solvers themselves** and compared with
what the coupling booked as external input or as an interface transfer.  The
reported residual is therefore a physical non-conservation, not a bookkeeping
identity:

    eps(t) = |sum_sides E_side(t) - Q_external_in(t) - (same at t=0)| / scale

Scenarios
---------
``thermal``     LBM <-> FEM conjugate heat transfer, 2D square cavity L = 0.1 m,
                dT = 50 K, coupled through a 1D interface at y = L/2.
``mechanical``  FEM <-> DEM contact: a *free* plane-stress plate hit edge-on by
                two DEM discs, so the contact force must satisfy Newton's third
                law for the total momentum to stay constant.
``three_field`` LBM <-> FEM <-> DEM energy chain on a clamped hull panel with a
                hot lower face, a water column above and ice discs hitting the
                free edge.

Sub-cycling and interpolation
-----------------------------
Every scenario runs the ratio set ``r = 1, 2, 4, 8`` crossed with the interface
reconstruction orders zero / first / second (:class:`TemporalScheme`), and each
combination is repeated in ``exchange_mode="independent"`` where the two sides
evaluate their own interface quantity instead of sharing one number.  That
control is the falsification arm of the experiment: the ledger must stay at
round-off for the shared scheme and must *not* stay at round-off for the
independent one, otherwise the instrument would be blind.

Time-scale honesty
------------------
In ``three_field`` the thermal (seconds) and mechanical (microseconds) clocks
differ by ~1e5 and are **not** synchronised: each macro step advances both
physics by one of their own stable steps and the ledger is sampled at the common
macro index.  The energy ledger is a snapshot statement (sum of stored energies
minus input is constant), so it does not require a common physical time.  This
is stated in the report and is the reason the two physics are energetically
additive here: thermomechanical coupling (thermal stress / expansion) is *not*
modelled.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .exchange import (
    MultiRateSchedule, TemporalScheme, reconstruction_poly,
    midpoint_exchange_value, midpoint_interval_integral,
)
from .ledger import EnergyLedger, MomentumLedger
from .solvers import DEM2D, ContactLaw, Q4Mesh, Q4PlaneStressFEM, Q4ThermalFEM, ThermalLBM2D

__all__ = [
    "ScenarioResult",
    "slab_analytic_temperature",
    "run_thermal_lbm_fem",
    "plate_first_frequency",
    "run_mechanical_fem_dem",
    "run_three_field",
    "run_suite",
    "SCHEMES",
    "RATIOS",
]

#: sub-cycling ratios required by the campaign node.
RATIOS: tuple[int, ...] = (1, 2, 4, 8)
#: interface reconstruction orders (zero / first / second).
SCHEMES: tuple[TemporalScheme, ...] = (
    TemporalScheme.ZOH,
    TemporalScheme.LINEAR,
    TemporalScheme.QUADRATIC,
)
#: Midpoint scheme (Crank-Nicolson type) - used explicitly for AC1-AC4 tests.
MIDPOINT_SCHEME = TemporalScheme.MIDPOINT


class _Sample:
    """Minimal exchange sample (an interface row plus its instant)."""

    __slots__ = ("t", "value")

    def __init__(self, t: float, value) -> None:
        self.t = t
        self.value = value


@dataclass
class ScenarioResult:
    """Per-step ledger state plus the scalars used for the acceptance gates."""

    name: str
    ratio: int
    scheme: str
    exchange_mode: str
    n_steps: int
    n_substeps: int
    dt_coarse: float
    residual_max: float
    residual_final: float
    scale: float
    series: dict[str, list[float]]
    metrics: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "ratio": self.ratio,
            "scheme": self.scheme,
            "exchange_mode": self.exchange_mode,
            "n_steps": self.n_steps,
            "n_substeps": self.n_substeps,
            "dt_coarse": self.dt_coarse,
            "residual_max": self.residual_max,
            "residual_final": self.residual_final,
            "scale": self.scale,
            "metrics": dict(self.metrics),
            "series": {k: [float(x) for x in v] for k, v in self.series.items()},
        }


# ---------------------------------------------------------------------------
# analytic reference: one-dimensional slab, Dirichlet at y=0, insulated at y=L
# ---------------------------------------------------------------------------
def slab_analytic_temperature(y, t: float, length: float, dT: float, alpha: float,
                              n_terms: int = 400) -> np.ndarray:
    """Exact solution of ``T_t = alpha T_yy`` with ``T(0,t)=dT``, ``T_y(L,t)=0``.

    Initial condition ``T(y,0)=0``, series form

        T(y,t) = dT - (2 dT / L) sum_n sin(lam_n y)/lam_n exp(-alpha lam_n^2 t)

    with ``lam_n = (2n+1) pi / (2L)``.
    """
    y = np.asarray(y, dtype=float)
    T = np.full_like(y, float(dT))
    for n in range(n_terms):
        lam = (2 * n + 1) * math.pi / (2.0 * length)
        T = T - (2.0 * dT / length) * np.sin(lam * y) / lam * math.exp(-alpha * lam * lam * t)
    return T


# ---------------------------------------------------------------------------
# (a) LBM <-> FEM conjugate heat transfer
# ---------------------------------------------------------------------------
def run_thermal_lbm_fem(ratio: int = 1, scheme: TemporalScheme = TemporalScheme.LINEAR,
                        exchange_mode: str = "shared", t_end: float = 2000.0,
                        nx: int = 40, ny: int = 20, length: float = 0.1,
                        dT: float = 50.0, alpha: float = 1.0e-6,
                        rho_cp: float = 1.0e6, tau: float = 1.25,
                        thickness: float = 1.0, relax: float = 0.2,
                        sample_times: Sequence[float] | None = None,
                        ) -> ScenarioResult:
    """D2Q9 thermal LBM (upper half) coupled to a Q4 conduction FEM (lower half).

    * 2D square cavity ``L = 0.1 m``; solid ``0 <= y <= L/2``, fluid
      ``L/2 <= y <= L``; interface at ``y = L/2``.
    * solid lower face ``y = 0``: Dirichlet ``dT = 50 K`` (the only external
      energy input); fluid top face and all side faces adiabatic.
    * the solid hands its interface temperature to the fluid (LBM Dirichlet
      wall); the fluid hands back the *measured* interface heat rate
      ``Q = C_f * sum(injected populations)`` as a Neumann flux on the solid's
      upper row.  ``exchange_mode="shared"`` uses that single number on both
      sides; ``"independent"`` lets the solid use its own one-sided Fourier
      estimate instead.
    """
    dx = length / nx
    ly = length / 2.0
    dy = ly / ny
    alpha_lat = (tau - 0.5) / 3.0
    dt_fine = alpha_lat * dx * dx / alpha
    sched = MultiRateSchedule(dt_fine=dt_fine, ratio=ratio)
    h = sched.dt_coarse
    n_coarse = max(1, int(round(t_end / h)))

    mesh = Q4Mesh(nx, ny, length, ly)
    fem = Q4ThermalFEM(mesh, conductivity=alpha * rho_cp,
                       volumetric_heat_capacity=rho_cp, thickness=thickness)
    lbm = ThermalLBM2D(nx, ny, tau=tau)
    inv = fem.cached_inverse(h)

    C_f = rho_cp * dx * dy * thickness
    top = mesh.row(ny)
    bottom = mesh.row(0)
    below = mesh.row(ny - 1)
    w_top = fem.M[top] / fem.M[top].sum()
    flux_pattern = np.zeros(mesh.n_nodes)
    flux_pattern[top] = w_top
    x_node = mesh.nodes[top, 0]
    x_col = (np.arange(nx) + 0.5) * dx
    band = dy
    k_solid = alpha * rho_cp

    ledger = EnergyLedger(sides=("fluid", "solid"))
    keys = ("t", "e_fluid", "e_solid", "e_total", "q_iface", "q_solid_est",
            "q_external", "residual")
    series: dict[str, list[float]] = {k: [] for k in keys}
    profiles: list[dict] = []
    sample_times = list(sample_times) if sample_times is not None else []

    state = {"t": 0.0, "q_external": 0.0, "q_sum": 0.0, "q_est": 0.0,
             "abs_q": 0.0, "abs_est": 0.0, "ref": None}

    relaxed = {"value": None}

    def interface_temperature() -> np.ndarray:
        """Interface temperature handed to the fluid, with interface relaxation.

        The explicit staggered Dirichlet-Neumann exchange is *unstable* in this
        configuration (see the module report): the LBM wall conductance is large
        compared with the heat capacity of the solid's surface row, so a full
        update (``relax = 1``) amplifies the interface error every exchange.
        ``relax`` in (0, 1] is the standard partitioned-coupling under
        relaxation; it only affects the *boundary condition* of the fluid, the
        measured flux still crosses the ledger unchanged, so conservation is
        not affected.
        """
        target = np.interp(x_col, x_node, fem.T[top])
        if relaxed["value"] is None or relax >= 1.0:
            relaxed["value"] = target
        else:
            relaxed["value"] = (1.0 - relax) * relaxed["value"] + relax * target
        return relaxed["value"]

    def record(tt: float) -> None:
        e_f = C_f * lbm.total()
        e_s = fem.energy()
        ledger.set("fluid", e_f)
        ledger.set("solid", e_s)
        ledger.external_in = state["q_external"]
        total = ledger.total()
        scale = max(abs(state["q_external"]), abs(e_f) + abs(e_s), 1e-300)
        if state["ref"] is None:
            state["ref"] = total
        series["t"].append(tt)
        series["e_fluid"].append(e_f)
        series["e_solid"].append(e_s)
        series["e_total"].append(total)
        series["q_iface"].append(state["q_sum"])
        series["q_solid_est"].append(state["q_est"])
        series["q_external"].append(state["q_external"])
        series["residual"].append(abs(total - state["ref"]) / scale)
        for target in sample_times:
            if abs(tt - target) <= 0.5 * h + 1e-12:
                # x-averaged vertical profile (the exact solution is 1D)
                y_solid = np.array([mesh.nodes[mesh.row(j), 1][0] for j in range(ny + 1)])
                T_solid = np.array([float(fem.T[mesh.row(j)].mean()) for j in range(ny + 1)])
                y_fluid = ly + (np.arange(ny) + 0.5) * dy
                y = np.concatenate([y_solid, y_fluid])
                profiles.append({
                    "t": tt,
                    "y": y.tolist(),
                    "T": np.concatenate([T_solid, lbm.T.mean(axis=1)]).tolist(),
                    "T_analytic": slab_analytic_temperature(y, tt, length, dT, alpha).tolist(),
                })

    history: list[_Sample] = []
    record(0.0)
    history.append(_Sample(0.0, interface_temperature()))

    for _ in range(n_coarse):
        previous = history[-2] if len(history) >= 2 else None
        previous2 = history[-3] if len(history) >= 3 else None
        current = history[-1]
        if scheme is TemporalScheme.MIDPOINT:
            # Crank-Nicolson midpoint: evaluate at interval midpoint
            q_mid = midpoint_exchange_value(previous, current, h / 2.0)
            injected = 0.0
            for k in range(ratio):
                injected += lbm.step(q_mid)
        else:
            f0, a1, a2 = reconstruction_poly(scheme, previous, current, previous2)
            injected = 0.0
            for k in range(ratio):
                s = (k + 1) * dt_fine
                wall = f0 + a1 * s + a2 * s * s
                injected += lbm.step(wall)
        q_iface = injected * C_f

        q_est = float(np.sum(k_solid * (fem.T[top] - fem.T[below]) / band
                             * (length / nx) * thickness))
        state["q_sum"] += q_iface
        state["q_est"] += q_est
        state["abs_q"] += abs(q_iface)
        state["abs_est"] += abs(q_est)

        imposed = q_iface if exchange_mode == "shared" else q_est
        fem.step(h, flux=-imposed / h * flux_pattern, matrix_inverse=inv)
        t_before = fem.T[bottom].copy()
        fem.T[bottom] = dT
        step_q = float(np.sum(fem.M[bottom] * (dT - t_before)))
        state["q_external"] += step_q
        state["t"] += h
        history.append(_Sample(state["t"], interface_temperature()))
        record(state["t"])

    resid = np.asarray(series["residual"])
    denom = max(state["abs_q"], state["abs_est"], 1e-300)
    metrics = {
        "flux_consistency": abs(state["q_sum"] - state["q_est"]) / denom,
        "q_iface_total_J": state["q_sum"],
        "q_solid_estimate_total_J": state["q_est"],
        "q_external_total_J": state["q_external"],
        "dt_fine_s": dt_fine,
        "dt_coarse_s": h,
        "t_end_s": state["t"],
        "fourier_number_interface": alpha * state["t"] / (length ** 2),
        "alpha_m2_per_s": alpha,
        "interface_relaxation": relax,
    }
    if profiles:
        errs = [float(np.sqrt(np.mean((np.asarray(p["T"]) - np.asarray(p["T_analytic"])) ** 2)) / dT)
                for p in profiles]
        metrics["profile_rel_l2_error"] = max(errs)
        metrics["profile_errors"] = errs
        metrics["profiles"] = profiles
    return ScenarioResult(
        name="thermal", ratio=ratio, scheme=scheme.value, exchange_mode=exchange_mode,
        n_steps=n_coarse, n_substeps=n_coarse * ratio, dt_coarse=h,
        residual_max=float(resid.max()), residual_final=float(resid[-1]),
        scale=float(max(abs(state["q_external"]), 1e-300)), series=series, metrics=metrics,
    )


# ---------------------------------------------------------------------------
# support: first axial frequency of a clamped-free plate (FEM physics check)
# ---------------------------------------------------------------------------
def plate_first_frequency(nx: int = 20, ny: int = 2, length: float = 0.2,
                          height: float = 0.04, young: float = 2.0e9,
                          poisson: float = 0.3, density: float = 900.0,
                          thickness: float = 0.05, n_steps: int = 4000) -> dict:
    """Measured first axial clamped-free frequency of the Q4 plane-stress plate.

    Analytic reference for a bar clamped at ``x = 0`` and free at ``x = L``:
    ``f1 = c / (4L)``, ``c = sqrt(E/rho)``.
    """
    mesh = Q4Mesh(nx, ny, length, height)
    plate = Q4PlaneStressFEM(mesh, young, poisson, density, thickness)
    plate.fix(mesh.column(0))
    dt = plate.critical_dt(safety=0.3)
    free = mesh.column(nx)
    plate.v[0::2] = 1.0e-3 * mesh.nodes[:, 0] / length
    history = []
    for _ in range(n_steps):
        plate.step(dt)
        history.append(float(plate.u[2 * free[len(free) // 2]]))
    u = np.asarray(history)
    u = u - u.mean()
    crossings = np.where(np.diff(np.sign(u)) != 0)[0]
    if len(crossings) < 3:
        raise RuntimeError("not enough zero crossings to estimate the frequency")
    t_cross = crossings * dt
    period = 2.0 * float(np.mean(np.diff(t_cross)))
    f_meas = 1.0 / period
    f_analytic = math.sqrt(young / density) / (4.0 * length)
    return {
        "nx": nx, "ny": ny, "dt": dt, "n_steps": n_steps,
        "frequency_hz": f_meas, "analytic_hz": f_analytic,
        "rel_error": abs(f_meas - f_analytic) / f_analytic,
    }


# ---------------------------------------------------------------------------
# (b) FEM <-> DEM : contact force and Newton's third law
# ---------------------------------------------------------------------------
def _taylor(scheme: TemporalScheme, u0, v0, a0, s: float):
    if scheme is TemporalScheme.ZOH:
        return u0, 0.0 * u0
    if scheme is TemporalScheme.LINEAR:
        return u0 + v0 * s, v0
    return u0 + v0 * s + 0.5 * a0 * s * s, v0 + a0 * s


def run_mechanical_fem_dem(ratio: int = 1, scheme: TemporalScheme = TemporalScheme.LINEAR,
                           exchange_mode: str = "shared", n_macro: int = 500,
                           nx: int = 10, ny: int = 3, length: float = 0.2,
                           height: float = 0.04, young: float = 2.0e9,
                           poisson: float = 0.3, density: float = 900.0,
                           thickness: float = 0.05, radius: float = 0.008,
                           velocity: float = 3.0, gap: float = 1.0e-3,
                           stiffness: float = 5.0e6, damping: float = 0.0,
                           ) -> ScenarioResult:
    """Two DEM discs hitting the free edge of an unsupported plane-stress plate.

    With no supports the total momentum of plate + particles is conserved
    exactly iff the contact force is reciprocal.  ``exchange_mode="independent"``
    substitutes the plate's own reaction impulse, evaluated from a frozen
    (start-of-interval) particle state, for the impulse the DEM actually
    applied.
    """
    mesh = Q4Mesh(nx, ny, length, height)
    plate = Q4PlaneStressFEM(mesh, young, poisson, density, thickness)
    h = plate.critical_dt(safety=0.5)
    dt_fine = h / ratio
    MultiRateSchedule(dt_fine=dt_fine, ratio=ratio)   # validation of the ratio

    edge = mesh.column(nx)
    edge_y = mesh.nodes[edge, 1]
    particles_y = np.array([height * 0.25, height * 0.75])
    node_of = np.array([int(np.argmin(np.abs(edge_y - y))) for y in particles_y])
    pos = np.column_stack([np.full(len(particles_y), length + gap + radius), particles_y])
    vel = np.column_stack([np.full(len(particles_y), -velocity), np.zeros(len(particles_y))])
    law = ContactLaw(stiffness=stiffness, damping=damping)
    dem = DEM2D(pos, vel, np.full(len(particles_y), radius), density, thickness, law)

    ledger = EnergyLedger(sides=("solid", "particles", "contact"))
    mom = MomentumLedger(sides=("particles", "solid"))

    def plate_momentum() -> tuple[float, float]:
        return (float(np.sum(plate.mass[0::2] * plate.v[0::2])),
                float(np.sum(plate.mass[1::2] * plate.v[1::2])))

    p0 = dem.momentum() + np.array(plate_momentum())
    p_scale = float(max(float(np.hypot(*p0)), 1e-300))

    keys = ("t", "e_solid", "e_particles", "e_contact", "e_total", "momentum_x",
            "residual", "momentum_residual", "impulse", "overlap")
    series: dict[str, list[float]] = {k: [] for k in keys}
    state = {"t": 0.0, "j": 0.0, "overlap": 0.0, "potential": 0.0, "ref": None}
    last_impulse = np.zeros((edge.size, 2))

    def record(tt: float) -> None:
        e_s = plate.energy()
        e_p = dem.kinetic_energy
        e_c = state["potential"]
        ledger.set("solid", e_s)
        ledger.set("particles", e_p)
        ledger.set("contact", e_c)
        total = ledger.total()
        if state["ref"] is None:
            state["ref"] = total
        e_scale = max(abs(state["ref"]), 1e-300)
        px_p, py_p = dem.momentum()
        px_s, py_s = plate_momentum()
        mom.set("particles", px_p, py_p)
        mom.set("solid", px_s, py_s)
        mom.record()
        series["t"].append(tt)
        series["e_solid"].append(e_s)
        series["e_particles"].append(e_p)
        series["e_contact"].append(e_c)
        series["e_total"].append(total)
        series["momentum_x"].append(px_p + px_s)
        series["residual"].append(abs(total - state["ref"]) / e_scale)
        series["momentum_residual"].append(mom.residual())
        series["impulse"].append(state["j"])
        series["overlap"].append(state["overlap"])

    record(0.0)
    for _ in range(n_macro):
        ux0 = plate.u[2 * edge].copy()
        vy0 = plate.v[2 * edge].copy()
        a0 = plate.accelerations()
        ax0 = a0[2 * edge].copy()
        pos0 = dem.pos.copy()
        vel0 = dem.vel.copy()

        impulse_nodes = np.zeros((edge.size, 2))
        for k in range(ratio):
            s = (k + 0.5) * dt_fine
            wall_x, wall_v = _taylor(scheme, ux0, vy0, ax0, s)
            wall_x = length + wall_x
            f_p, f_n, overlaps, _ = dem.contact_forces(wall_x, wall_v, edge_y, node_of)
            impulse_nodes += f_n * dt_fine
            state["overlap"] = float(overlaps.max()) if overlaps.size else 0.0
            # total contact potential energy (sum over contacts, not the max)
            state["potential"] = float(0.5 * stiffness * np.sum(overlaps ** 2))
            dem.step(dt_fine, f_p)

        if exchange_mode == "shared":
            applied = impulse_nodes
        else:
            wall_x_end, wall_v_end = _taylor(scheme, ux0, vy0, ax0, h)
            wall_x_end = length + wall_x_end
            x_pred = pos0[:, 0] + vel0[:, 0] * h
            overlaps = np.maximum(-(x_pred - wall_x_end[node_of] - radius), 0.0)
            rates = -(vel0[:, 0] - wall_v_end[node_of])
            mags = np.array([law.force(o, r) for o, r in zip(overlaps, rates)])
            applied = np.zeros((edge.size, 2))
            np.add.at(applied, node_of, np.column_stack([mags, np.zeros_like(mags)]))
            applied = applied * h
        last_impulse = applied
        force = np.zeros(2 * mesh.n_nodes)
        np.add.at(force, 2 * edge, applied[:, 0] / h)
        np.add.at(force, 2 * edge + 1, applied[:, 1] / h)
        plate.step(h, force)
        state["j"] += float(np.sum(applied[:, 0]))
        state["t"] += h
        record(state["t"])

    resid = np.asarray(series["residual"])
    mresid = np.asarray(series["momentum_residual"])
    metrics = {
        "dt_coarse_s": h,
        "dt_fine_s": dt_fine,
        "t_end_s": state["t"],
        "impulse_total_Ns": state["j"],
        "momentum_initial_kgm_s": float(np.hypot(*p0)),
        "energy_initial_J": float(state["ref"]),
        "plate_first_frequency": None,
        "max_overlap_m": float(max(series["overlap"])),
        "contact_stiffness": stiffness,
        "damping": damping,
    }
    return ScenarioResult(
        name="mechanical", ratio=ratio, scheme=scheme.value, exchange_mode=exchange_mode,
        n_steps=n_macro, n_substeps=n_macro * ratio, dt_coarse=h,
        residual_max=float(resid.max()), residual_final=float(resid[-1]),
        scale=p_scale, series=series, metrics=metrics,
    )


# ---------------------------------------------------------------------------
# (c) LBM <-> FEM <-> DEM energy chain
# ---------------------------------------------------------------------------
def run_three_field(ratio: int = 1, scheme: TemporalScheme = TemporalScheme.LINEAR,
                    exchange_mode: str = "shared", n_macro: int = 200,
                    nx: int = 10, ny: int = 3, length: float = 0.2,
                    height: float = 0.04, young: float = 2.0e9,
                    poisson: float = 0.3, density: float = 900.0,
                    thickness: float = 0.05, radius: float = 0.008,
                    velocity: float = 6.0, gap: float = 1.0e-3,
                    stiffness: float = 5.0e6, dT: float = 20.0,
                    alpha: float = 1.0e-6, rho_cp: float = 1.0e6,
                    fluid_nx: int = 20, fluid_ny: int = 4, tau: float = 1.25,
                    ) -> ScenarioResult:
    """Hull panel (FEM) with a water column (thermal LBM) and ice discs (DEM).

    Thermal chain: plate lower face Dirichlet ``dT``, plate upper face coupled to
    the LBM south wall (water column above the plate, adiabatic free surface).
    Mechanical chain: the plate is clamped at ``x = 0`` and the DEM discs hit its
    free edge, so the fixture does no work and the mechanical energy stays
    closed.  The two chains are energetically additive; thermomechanical
    coupling is not modelled (see the module docstring).
    """
    mesh = Q4Mesh(nx, ny, length, height)
    plate = Q4PlaneStressFEM(mesh, young, poisson, density, thickness)
    fem = Q4ThermalFEM(mesh, conductivity=alpha * rho_cp,
                       volumetric_heat_capacity=rho_cp, thickness=thickness)

    dx = length / fluid_nx
    alpha_lat = (tau - 0.5) / 3.0
    dt_th_fine = alpha_lat * dx * dx / alpha
    h_th = dt_th_fine * ratio
    inv = fem.cached_inverse(h_th)
    lbm = ThermalLBM2D(fluid_nx, fluid_ny, tau=tau, T_init=0.0)
    C_f = rho_cp * dx * dx * thickness

    top = mesh.row(ny)
    bottom = mesh.row(0)
    below = mesh.row(ny - 1)
    w_top = fem.M[top] / fem.M[top].sum()
    flux_pattern = np.zeros(mesh.n_nodes)
    flux_pattern[top] = w_top
    x_node = mesh.nodes[top, 0]
    x_col = (np.arange(fluid_nx) + 0.5) * dx
    k_solid = alpha * rho_cp

    h_mech = plate.critical_dt(safety=0.4)
    edge = mesh.column(nx)
    edge_y = mesh.nodes[edge, 1]
    particles_y = np.array([height * 0.25, height * 0.75])
    node_of = np.array([int(np.argmin(np.abs(edge_y - y))) for y in particles_y])
    pos = np.column_stack([np.full(len(particles_y), length + gap + radius), particles_y])
    vel = np.column_stack([np.full(len(particles_y), -velocity), np.zeros(len(particles_y))])
    law = ContactLaw(stiffness=stiffness, damping=0.0)
    dem = DEM2D(pos, vel, np.full(len(particles_y), radius), density, thickness, law)
    plate.fix(mesh.column(0))

    ledger = EnergyLedger(sides=("fluid", "solid_thermal", "solid_mech",
                                 "particles", "contact"))
    keys = ("t_thermal", "t_mech", "e_fluid", "e_solid_thermal", "e_solid_mech",
            "e_particles", "e_contact", "e_total", "q_external", "residual",
            "mech_residual", "thermal_residual", "momentum_x")
    series: dict[str, list[float]] = {k: [] for k in keys}
    state = {"t_th": 0.0, "t_m": 0.0, "q_ext": 0.0, "ref": None,
             "mech_ref": None, "th_ref": None, "overlap": 0.0, "potential": 0.0}

    def record() -> None:
        e_f = C_f * lbm.total()
        e_st = fem.energy()
        e_sm = plate.energy()
        e_p = dem.kinetic_energy
        e_c = state["potential"]
        ledger.set("fluid", e_f)
        ledger.set("solid_thermal", e_st)
        ledger.set("solid_mech", e_sm)
        ledger.set("particles", e_p)
        ledger.set("contact", e_c)
        ledger.external_in = state["q_ext"]
        total = ledger.total()
        if state["ref"] is None:
            state["ref"] = total
            state["mech_ref"] = e_sm + e_p + e_c
            state["th_ref"] = e_f + e_st
        scale = max(abs(state["q_ext"]), abs(e_f) + abs(e_st) + abs(e_sm) + abs(e_p) + abs(e_c),
                    1e-300)
        mech = e_sm + e_p + e_c
        th = e_f + e_st - state["q_ext"]
        series["t_thermal"].append(state["t_th"])
        series["t_mech"].append(state["t_m"])
        series["e_fluid"].append(e_f)
        series["e_solid_thermal"].append(e_st)
        series["e_solid_mech"].append(e_sm)
        series["e_particles"].append(e_p)
        series["e_contact"].append(e_c)
        series["e_total"].append(total)
        series["q_external"].append(state["q_ext"])
        series["residual"].append(abs(total - state["ref"]) / scale)
        series["mech_residual"].append(
            abs(mech - state["mech_ref"]) / max(abs(state["mech_ref"]), 1e-300))
        series["thermal_residual"].append(
            abs(th - state["th_ref"]) / max(abs(state["q_ext"]), 1e-300))
        series["momentum_x"].append(float(dem.momentum()[0]
                                          + np.sum(plate.mass[0::2] * plate.v[0::2])))

    record()
    for _ in range(n_macro):
        # --- thermal exchange (LBM sub-cycled inside one FEM thermal step) ---
        history = [(state["t_th"] - h_th, np.interp(x_col, x_node, fem.T[top]))]
        wall_cur = _Sample(state["t_th"], np.interp(x_col, x_node, fem.T[top]))
        prev = _Sample(state["t_th"] - h_th, np.interp(x_col, x_node, fem.T[top]))
        f0, a1, a2 = reconstruction_poly(scheme, prev, wall_cur, None)
        injected = 0.0
        for k in range(ratio):
            s = (k + 1) * dt_th_fine
            injected += lbm.step(f0 + a1 * s + a2 * s * s)
        q_iface = injected * C_f
        q_est = float(np.sum(k_solid * (fem.T[top] - fem.T[below]) / (height / ny)
                             * (length / nx) * thickness))
        imposed = q_iface if exchange_mode == "shared" else q_est
        fem.step(h_th, flux=-imposed / h_th * flux_pattern, matrix_inverse=inv)
        t_before = fem.T[bottom].copy()
        fem.T[bottom] = dT
        state["q_ext"] += float(np.sum(fem.M[bottom] * (dT - t_before)))
        state["t_th"] += h_th

        # --- mechanical exchange (DEM sub-cycled inside one plate step) ---
        ux0 = plate.u[2 * edge].copy()
        vx0 = plate.v[2 * edge].copy()
        a0 = plate.accelerations()[2 * edge].copy()
        impulse_nodes = np.zeros((edge.size, 2))
        for k in range(ratio):
            s = (k + 0.5) * (h_mech / ratio)
            wall_x, wall_v = _taylor(scheme, ux0, vx0, a0, s)
            f_p, f_n, overlaps, _ = dem.contact_forces(length + wall_x, wall_v, edge_y, node_of)
            impulse_nodes += f_n * (h_mech / ratio)
            state["overlap"] = float(overlaps.max()) if overlaps.size else 0.0
            state["potential"] = float(0.5 * stiffness * np.sum(overlaps ** 2))
            dem.step(h_mech / ratio, f_p)
        force = np.zeros(2 * mesh.n_nodes)
        np.add.at(force, 2 * edge, impulse_nodes[:, 0] / h_mech)
        np.add.at(force, 2 * edge + 1, impulse_nodes[:, 1] / h_mech)
        plate.step(h_mech, force)
        state["t_m"] += h_mech
        record()

    resid = np.asarray(series["residual"])
    return ScenarioResult(
        name="three_field", ratio=ratio, scheme=scheme.value, exchange_mode=exchange_mode,
        n_steps=n_macro, n_substeps=n_macro * ratio, dt_coarse=h_mech,
        residual_max=float(resid.max()), residual_final=float(resid[-1]),
        scale=float(max(state["q_ext"], 1e-300)), series=series,
        metrics={
            "dt_thermal_s": h_th,
            "dt_mech_s": h_mech,
            "t_thermal_end_s": state["t_th"],
            "t_mech_end_s": state["t_m"],
            "time_scale_separation": h_th / h_mech,
            "mech_residual_max": float(np.max(series["mech_residual"])),
            "thermal_residual_max": float(np.max(series["thermal_residual"])),
            "max_contact_energy_J": float(max(series["e_contact"])) if series["e_contact"] else 0.0,
            "q_external_total_J": state["q_ext"],
        },
    )


# ---------------------------------------------------------------------------
# suite
# ---------------------------------------------------------------------------
def run_suite(ratios: Sequence[int] = RATIOS, schemes: Sequence[TemporalScheme] = SCHEMES,
              include_control: bool = True, thermal_kw: dict | None = None,
              mechanical_kw: dict | None = None, three_field_kw: dict | None = None,
              ) -> dict[str, list[dict]]:
    """Run the full ratio x scheme matrix for all three couplings."""
    thermal_kw = dict(thermal_kw or {})
    mechanical_kw = dict(mechanical_kw or {})
    three_field_kw = dict(three_field_kw or {})
    out: dict[str, list[dict]] = {"thermal": [], "mechanical": [], "three_field": []}
    for ratio in ratios:
        for scheme in schemes:
            out["thermal"].append(
                run_thermal_lbm_fem(ratio, scheme, "shared", **thermal_kw).as_dict())
            out["mechanical"].append(
                run_mechanical_fem_dem(ratio, scheme, "shared", **mechanical_kw).as_dict())
            out["three_field"].append(
                run_three_field(ratio, scheme, "shared", **three_field_kw).as_dict())
        if include_control:
            out["thermal"].append(
                run_thermal_lbm_fem(ratio, TemporalScheme.LINEAR, "independent",
                                    **thermal_kw).as_dict())
            out["mechanical"].append(
                run_mechanical_fem_dem(ratio, TemporalScheme.LINEAR, "independent",
                                       **mechanical_kw).as_dict())
            out["three_field"].append(
                run_three_field(ratio, TemporalScheme.LINEAR, "independent",
                                **three_field_kw).as_dict())
    return out