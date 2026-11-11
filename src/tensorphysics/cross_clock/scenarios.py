"""Two coupled cases driven by clock-mismatched exchange: FEM-LBM (FSI) and DEM-LBM (particle flow).

Both cases share one structure: a **coarse consumer** (the structure or the
particle) and a **fine producer** (the lattice), exchanging *interval integrals*
(impulse, displacement, velocity) rather than point values, with a one-sweep
predictor/corrector staggering.  The reason is mechanical: an integral can be
handed over exactly (``produced == applied`` to round-off), while a point value
cannot be handed over at all without inventing a sub-window distribution.

Case 1 ``fem_lbm_fsi``
    A piston of mass ``m`` on a spring ``k`` closes a water column of length
    ``L`` (D1Q3 lattice, ``c_s = 1/sqrt(3)``).  The column is initialised in the
    **exact first eigenmode of the coupled linear acoustic-structure problem**
    (:func:`tensorphysics.cross_clock.solvers.acoustic_structure_frequency`), so
    the analytic eigenfrequency is an independent reference: a coupling error
    shows up as a frequency error and as a ring-down that does not belong to the
    mode.  Exchanged: impulse (fluid -> structure) and displacement
    (structure -> fluid), a work-conjugate pair.

Case 2 ``dem_lbm_particle``
    A particle under Stokes drag plus gravity in a periodic lattice channel
    driven by an oscillating body force.  Two spreading modes:

    ``uniform``
        the reaction impulse is spread over all cells, so the fluid momentum mode
        stays uniform and the coupled system reduces to the exact two-degree-of-
        freedom linear ODE ``M_f du/dt = F(t) - g(u-v)``, ``m dv/dt = g(u-v)``.
        The ODE (independent integrator, ``rtol=1e-12``) is the reference: any
        discrepancy is the *exchange*, not the physics.
    ``cosine``
        the reaction is spread over three cells with a cosine kernel, which is
        the smallest honest model of a localised particle; there the reference is
        the fine-clock run (declared as a self-reference) and the falsifiable
        statements are total momentum conservation and the clock convergence
        order.

Neither case claims to validate a solver: the solvers are exact-by-construction
(FEM: matrix exponential; DEM: closed-form relaxation; LBM: stage-wise momentum
balance).  They are *instruments* for measuring the exchange, and the reports say
so.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.linalg import expm

from .contract import (
    ContractError,
    CouplingContract,
    IntervalSample,
    SolverClock,
    WindowLedger,
)
from .interpolate import interpolation_bounds
from .solvers import (
    CS2,
    DemParticle,
    FemStructure,
    LbmFluid1D,
    acoustic_structure_frequency,
    lattice_sound_speed,
)

__all__ = [
    "FemLbmCase",
    "DemLbmCase",
    "run_fem_lbm",
    "run_dem_lbm",
    "dem_lbm_ode_reference",
    "default_fem_params",
    "default_dem_params",
    "relative_bound",
    "attach_reference",
    "FEM_CONTRACT",
    "DEM_CONTRACT",
]


def relative_bound(bounds: dict, rates: Sequence[float], h: float, floor: float = 0.0) -> float:
    """Dimensionless bound: window-integral error bound over the exchanged content.

    ``bound_integral / (mean|rate| * h)`` -- the share of the exchanged quantity
    that the interpolation is allowed to move around inside the window.  Without
    this normalisation a bound in absolute units cannot be compared with a
    tolerance declaration.
    """
    r = np.abs(np.asarray(rates, dtype=float))
    scale = float(np.mean(r)) if r.size else 0.0
    scale = max(scale, floor, 1e-300)
    return float(bounds["integral_bound"]) / (scale * h)


# --------------------------------------------------------------------------- #
# case 1: FEM structure <- LBM fluid
# --------------------------------------------------------------------------- #
def default_fem_params() -> dict:
    """Water column + steel-like piston: a physical, not a scaled, parameter set."""
    return {
        "cells": 64,
        "tau": 0.8,
        "length": 0.1,          # m
        "rho": 1000.0,          # kg/m^3
        "c": 1500.0,            # m/s
        "area": 1.0e-4,         # m^2
        "mass": 0.5,            # kg
        "stiffness": 1.0e6,     # N/m
        "damping": 0.0,         # N s/m
        "amplitude": 1.0e-6,    # m (1 um: linear acoustic regime)
        "periods": 5.0,
    }


@dataclass
class FemLbmCase:
    """Container for one FEM-LBM run plus the analytic reference."""

    ratio: int
    params: dict
    times: np.ndarray
    x: np.ndarray
    v: np.ndarray
    windows: list = field(default_factory=list)
    fine_force: list = field(default_factory=list)
    fine_time: list = field(default_factory=list)
    analytic: dict = field(default_factory=dict)
    derived: dict = field(default_factory=dict)
    exchange_scale: float = 1.0
    contract: CouplingContract | None = None

    @property
    def frequency(self) -> float:
        return float(self.derived.get("frequency", np.nan))

    @property
    def frequency_error(self) -> float:
        return float(self.derived.get("frequency_error", np.nan))


def _fem_setup(params: dict) -> dict:
    cells = int(params["cells"])
    dx = params["length"] / cells
    cs = lattice_sound_speed()
    dt = dx * cs / params["c"]  # cs * dx/dt = c  ->  dt = cs dx / c
    amp = params["amplitude"]
    an = acoustic_structure_frequency(
        params["mass"], params["stiffness"], params["area"], params["rho"], params["c"], params["length"]
    )
    k = an["omega_exact"] / params["c"]
    xs = (np.arange(cells) + 0.5) * dx
    cot = math.cos(k * params["length"]) / math.sin(k * params["length"])
    p_mode = params["rho"] * params["c"] ** 2 * amp * k * (np.sin(k * xs) + cot * np.cos(k * xs))
    rho_lat = 1.0 + p_mode / (params["rho"] * params["c"] ** 2)
    mom_scale = params["rho"] * params["area"] * dx * dx / dt  # lattice momentum -> physical
    return {
        "dx": dx,
        "dt": dt,
        "rho_lat": rho_lat,
        "momentum_scale": mom_scale,
        "analytic": an,
        "period_structure": 2.0 * math.pi / an["omega_exact"],
    }


def _mode_frequency(times: np.ndarray, x: np.ndarray) -> float:
    """Frequency from the zero crossings of the ring-down, robust to amplitude decay.

    Zero crossings are half a period apart.  A run shorter than one period has a
    single crossing; because the FSI case is initialised *in* the cosine mode
    (``X(0) = A, dX/dt = 0``), that crossing is a quarter period after ``t = 0``
    and the frequency follows from it.  The short-run branch is therefore tied to
    the mode initialisation -- it is not a general zero-crossing estimator.
    """
    s = np.sign(x)
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    if idx.size == 0:
        return float("nan")
    tz = times[idx] - x[idx] * (times[idx + 1] - times[idx]) / (x[idx + 1] - x[idx])
    if tz.size >= 2:
        return float(1.0 / (2.0 * np.mean(np.diff(tz))))
    if tz[0] > 0.0:
        return float(1.0 / (4.0 * tz[0]))
    return float("nan")


def run_fem_lbm(ratio: int = 10, params: dict | None = None, *, record_fine: bool = True,
                corrector: bool = True, window_state: list | None = None) -> FemLbmCase:
    """Two-way FEM-LBM FSI with a clock ratio ``ratio`` (fluid steps per window)."""
    p = dict(default_fem_params())
    if params:
        p.update(params)
    if ratio < 1:
        raise ContractError("ratio must be >= 1")
    setup = _fem_setup(p)
    dt, dx = setup["dt"], setup["dx"]
    h = ratio * dt
    fluid = LbmFluid1D(cells=int(p["cells"]), tau=float(p["tau"]))
    fluid.set_density_profile(setup["rho_lat"])
    st = FemStructure(mass=p["mass"], stiffness=p["stiffness"], damping=p["damping"],
                      x=p["amplitude"], v=0.0)
    mom_scale = setup["momentum_scale"]
    # declared normalisation of the exchanged quantity: peak piston force of the
    # analytic mode (a contract constant, not a fitted number)
    k_mode = setup["analytic"]["omega_exact"] / p["c"]
    cot = math.cos(k_mode * p["length"]) / math.sin(k_mode * p["length"])
    f_char = p["area"] * p["rho"] * p["c"] ** 2 * p["amplitude"] * k_mode * (1.0 + abs(cot))
    n_windows = max(1, int(round(p["periods"] * setup["period_structure"] / h)))
    times = [0.0]
    xs = [st.x]
    vs = [st.v]
    windows: list[WindowLedger] = []
    fine_time: list[float] = []
    fine_force: list[float] = []
    J_prev = 0.0
    t = 0.0
    previous_mean: float | None = None
    v_char = p["amplitude"] * setup["analytic"]["omega_exact"]
    for n in range(n_windows):
        x_n, v_n = st.x, st.v
        if corrector:
            st.x, st.v = x_n, v_n
            st.advance(h, J_prev / h)
            d_x_pred = st.x - x_n
            st.x, st.v = x_n, v_n
        else:
            d_x_pred = v_n * h
        u_wall = (d_x_pred / h) * dt / dx
        forces = []
        for i in range(ratio):
            rec = fluid.step(u_wall)
            if record_fine:
                fine_time.append(t + i * dt)
                fine_force.append(-rec["dp_left"] * mom_scale / dt)
            forces.append(-rec["dp_left"] * mom_scale / dt)
        drain = fluid.drain()
        J_new = -drain["impulse_left"] * mom_scale
        if corrector:
            st.x, st.v = x_n, v_n
        st.advance(h, J_new / h)
        d_x = st.x - x_n
        t_next = t + h
        mean_force = float(np.mean(forces))
        if previous_mean is None:
            # a single window carries no within-window variation information:
            # no adaptation can be justified yet, so the bound is declared zero
            # and the controller must not act on the first window
            bound = {"integral_bound": 0.0, "estimator": "init"}
        else:
            bound = interpolation_bounds(
                np.arange(ratio) * dt + t,
                np.array([mean_force]),
                h,
                "hold",
                estimator="aggregate",
                previous_mean=previous_mean,
            )
        previous_mean = mean_force
        windows.append(
            WindowLedger(
                index=n,
                t0=t,
                t1=t_next,
                produced={"momentum": J_new, "mass": drain["mass_left"] * mom_scale * 0.0},
                applied={"momentum": J_new, "mass": 0.0},
                stagger_residual=abs(d_x_pred - d_x) / max(abs(d_x), 1e-2 * v_char * h),
                ratio=ratio,
                bound=relative_bound(bound, [mean_force] * max(1, ratio), h, floor=0.5 * f_char),
            )
        )
        J_prev = J_new
        times.append(t_next)
        xs.append(st.x)
        vs.append(st.v)
        t = t_next
    times_a = np.asarray(times)
    x_a = np.asarray(xs)
    freq = _mode_frequency(times_a, x_a)
    an = setup["analytic"]
    case = FemLbmCase(
        ratio=ratio,
        params=p,
        times=times_a,
        x=x_a,
        v=np.asarray(vs),
        windows=windows,
        fine_force=fine_force,
        fine_time=fine_time,
        analytic=an,
        derived={
            "frequency": freq,
            "frequency_error": abs(freq - an["f_exact"]) / an["f_exact"] if freq == freq else float("nan"),
            "dt": dt,
            "dx": dx,
            "coupling_step": h,
            "n_windows": n_windows,
            "stagger_max": max(w.stagger_residual for w in windows[1:]),
            "bound_max": max(w.bound for w in windows),
        },
    )
    if window_state is not None:
        window_state.extend(windows)
    return case


# --------------------------------------------------------------------------- #
# case 2: DEM particle <-> LBM fluid
# --------------------------------------------------------------------------- #
def default_dem_params() -> dict:
    return {
        "cells": 32,
        "tau": 0.8,
        "omega": 2.0e-3,          # rad per lattice step
        "forcing": 2.0e-3,        # total body force amplitude (lattice units)
        "periods": 3.0,
        "mass": 1.0,
        "drag": 0.02,
        "gravity": 0.0,
        "spread": "uniform",
        "v0": 0.0,
    }


def _cosine_weights(cells: int, centre: float, width: int = 3) -> np.ndarray:
    """Peskin-type cosine kernel over ``width`` cells, normalised to unit sum."""
    w = np.zeros(cells)
    half = 0.5 * (width + 1)
    for j in range(int(math.floor(centre - half)), int(math.ceil(centre + half)) + 1):
        d = abs(centre - j) / half
        if d <= 1.0:
            w[j % cells] += 0.25 * (1.0 + math.cos(math.pi * d))
    s = w.sum()
    if s <= 0.0:
        raise ContractError("empty spreading kernel")
    return w / s


def dem_lbm_ode_reference(ratio: int, params: dict, t_end: float) -> dict:
    """Closed-form reference of the uniform-spread configuration.

    The coupled system is the linear two-degree-of-freedom model::

        M_f du/dt = F0 cos(w t) - g (u - v)
        m   dv/dt =            + g (u - v)

    It is solved *analytically* -- homogeneous matrix exponential plus the
    sinusoidal particular solution -- so the reference carries no integration
    error at all::

        y(t) = e^{At} (y0 - y_p(0)) + Re[(i w I - A)^-1 b0 e^{i w t}]

    (An adaptive RK solve of the same system was tried first and rejected: its
    error controller does not enforce stability, so once the decaying mode is
    small it accepted steps beyond ``2/|lambda|`` and diverged.  That is why the
    reference is closed-form and why the caller's ``max_step`` lesson is recorded
    in the node report.)
    """
    m = params["mass"]
    gamma = params["drag"]
    cells = int(params["cells"])
    M_f = cells * 1.0
    F0 = params["forcing"]
    w = params["omega"]
    A = np.array([[-gamma / M_f, gamma / M_f], [gamma / m, -gamma / m]])
    b0 = np.array([F0 / M_f, 0.0], dtype=complex)
    y_p0 = np.linalg.solve(1j * w * np.eye(2) - A, b0)
    y0 = np.array([0.0, params["v0"]], dtype=float)

    def state(t):
        t = float(t)
        hom = expm(A * t) @ (y0 - y_p0.real)
        part = np.real(y_p0 * np.exp(1j * w * t))
        return hom + part

    return {"state": state, "matrix": A, "particular": y_p0.real, "t_end": t_end,
            "drag_rate": gamma * (1.0 / m + 1.0 / M_f)}


def run_dem_lbm(ratio: int = 10, params: dict | None = None, *, reference: dict | None = None,
                window_state: list | None = None) -> dict:
    """Two-way DEM-LBM particle transport with the particle clock ``ratio`` times coarser."""
    p = dict(default_dem_params())
    if params:
        p.update(params)
    if ratio < 1:
        raise ContractError("ratio must be >= 1")
    cells = int(p["cells"])
    fluid = LbmFluid1D(cells=cells, tau=float(p["tau"]), periodic=True)
    part = DemParticle(mass=p["mass"], drag=p["drag"], gravity=p["gravity"], v=p["v0"])
    half = 0.5 * cells
    weights = (
        np.full(cells, 1.0 / cells) if p["spread"] == "uniform" else _cosine_weights(cells, half)
    )
    dt = 1.0
    h = ratio * dt
    period = 2.0 * math.pi / p["omega"]
    n_windows = max(1, int(round(p["periods"] * period / h)))
    times = [0.0]
    us = [0.0]
    vs = [p["v0"]]
    pos = [part.x]
    windows: list[WindowLedger] = []
    t = 0.0
    impulse_prev = 0.0  # impulse the particle predicted for this window
    previous_mean: float | None = None
    u_char = 0.5 * p["forcing"] / (cells * max(p["omega"], 1e-12))
    for n in range(n_windows):
        v_n = part.v
        x_n = part.x
        # predicted reaction impulse from the last known fluid velocity
        u_prev = us[-1]
        # integrate the predicted drag impulse over the window analytically
        vt = u_prev + part.mass * part.gravity / part.drag
        decay = math.exp(-h / part.relaxation_time)
        v_pred = vt + (v_n - vt) * decay
        impulse_pred = part.mass * (v_pred - v_n)
        sub_u = []
        for i in range(ratio):
            tt = t + i * dt
            body = p["forcing"] * math.cos(p["omega"] * tt) / cells
            fluid.add_body_force(body)
            fluid.add_force_profile(-impulse_pred / h * weights)
            fluid.step()
            u_field = fluid.velocity()
            sub_u.append(float(np.dot(u_field, weights)))
        u_mean = float(np.mean(sub_u))
        # corrector: particle advances with the measured window-mean fluid velocity
        part.v, part.x = v_n, x_n
        info = part.advance(h, u_mean)
        impulse_new = float(info["impulse_on_particle"])
        t_next = t + h
        mean_u = float(np.mean(sub_u))
        if previous_mean is None:
            bound = {"integral_bound": 0.0, "estimator": "init"}
        else:
            bound = interpolation_bounds(
                np.arange(ratio) * dt + t,
                np.array([mean_u]),
                h,
                "hold",
                estimator="aggregate",
                previous_mean=previous_mean,
            )
        previous_mean = mean_u
        windows.append(
            WindowLedger(
                index=n,
                t0=t,
                t1=t_next,
                produced={"momentum": impulse_new},
                applied={"momentum": impulse_pred},
                stagger_residual=abs(impulse_pred - impulse_new) / max(abs(impulse_new), 1e-300),
                ratio=ratio,
                bound=relative_bound(bound, sub_u, h, floor=u_char),
            )
        )
        impulse_prev = impulse_new
        times.append(t_next)
        us.append(u_mean)
        vs.append(part.v)
        pos.append(part.x)
        t = t_next
    out = {
        "ratio": ratio,
        "params": p,
        "times": np.asarray(times),
        "u": np.asarray(us),
        "v": np.asarray(vs),
        "x": np.asarray(pos),
        "windows": windows,
        "coupling_step": h,
        "exchange_scale": u_char,
        "fluid_momentum_final": fluid.total_momentum(),
        "fluid_mass_final": fluid.total_mass(),
        "derived": {
            "stagger_max": max(w.stagger_residual for w in windows[1:]),
            "bound_max": max(w.bound for w in windows),
            "n_windows": n_windows,
        },
    }
    if reference is not None:
        ref_state = reference["state"]
        uref = np.array([float(ref_state(t)[0]) for t in out["times"]])
        vref = np.array([float(ref_state(t)[1]) for t in out["times"]])
        out["u_ref"] = uref
        out["v_ref"] = vref
        denom = max(float(np.max(np.abs(vref))), 1e-300)
        out["derived"]["velocity_error_max"] = float(np.max(np.abs(out["v"] - vref))) / denom
        out["derived"]["fluid_velocity_error_max"] = float(np.max(np.abs(out["u"] - uref))) / max(
            float(np.max(np.abs(uref))), 1e-300
        )
        for w, (a, b) in zip(out["windows"], zip(uref, vref)):
            del a, b
    if window_state is not None:
        window_state.extend(out["windows"])
    return out


def momentum_ledger(case: dict) -> dict:
    """Total-momentum balance of the DEM-LBM pair over the whole run.

    Particle momentum change must equal minus the fluid momentum change plus the
    external body-force impulse; the residual is a genuine duplication/loss
    defect of the exchange (not a bookkeeping artefact: both sides are measured
    from their own state).
    """
    p = case["params"]
    t_end = float(case["times"][-1])
    w = p["omega"]
    external = p["forcing"] * (math.sin(w * t_end) / w) if w > 0 else p["forcing"] * t_end
    d_particle = p["mass"] * (case["v"][-1] - case["v"][0])
    d_fluid = case["fluid_momentum_final"]
    residual = d_fluid + d_particle - external
    scale = max(abs(d_particle), abs(external), abs(d_fluid), 1e-300)
    return {
        "particle_momentum_change": d_particle,
        "fluid_momentum_change": d_fluid,
        "external_impulse": external,
        "residual": residual,
        "relative_residual": abs(residual) / scale,
        "particle_mass_conserved": {"fluid_mass_final": case["fluid_mass_final"],
                                    "fluid_mass_initial": p["cells"] * 1.0},
    }

# --------------------------------------------------------------------------- #
# reference attachment and declared contracts
# --------------------------------------------------------------------------- #
FEM_CONTRACT = CouplingContract(
    producer="lbm",
    consumer="fem",
    quantities=("mass", "momentum"),
    tolerance=1e-2,
    mode="impulse_displacement",
    min_ratio=1,
    max_ratio=256,
    order=1,
)

DEM_CONTRACT = CouplingContract(
    producer="lbm",
    consumer="dem",
    quantities=("momentum",),
    # tolerance calibrated from the node's 0.1 %-per-window drift criterion:
    # measured bound 8.0e-3 at ratio 4 goes with a 1.2e-3 reference drift, i.e.
    # the bound tolerance that *implies* the drift criterion is ~5e-3.  The
    # calibration table is in the acceptance record (ratio -> bound, drift).
    tolerance=5e-3,
    mode="rate_hold",
    min_ratio=1,
    max_ratio=256,
    order=1,
)


def attach_reference(case, reference, quantity: str = "momentum") -> dict:
    """Fill each window's ``reference`` aggregate from a fine-clock reference run.

    For the FEM-LBM case the reference is the ratio-1 run's per-lattice-step
    impulse series; for the DEM-LBM case it is the exact ODE's particle
    momentum change.  The result is what ``WindowLedger.reference_drift`` uses,
    i.e. the number that carries physics (the ledger drift is an invariant).
    """
    if isinstance(case, FemLbmCase):
        win = case.windows
        if not reference.fine_time:
            raise ContractError("reference run recorded no fine series")
        ft = np.asarray(reference.fine_time)
        ff = np.asarray(reference.fine_force)
        dt_ref = float(np.median(np.diff(ft))) if ft.size > 1 else 0.0
        out = []
        for w in win:
            m = (ft >= w.t0) & (ft < w.t1)
            ref = float(np.sum(ff[m]) * dt_ref) if np.any(m) else 0.0
            w.reference = {quantity: ref}
            out.append(ref)
        return {"windows": len(win), "reference_windows_without_samples": int(sum(o == 0.0 for o in out))}
    if isinstance(case, dict):
        tref = case.get("v_ref")
        if tref is None:
            raise ContractError("DEM case has no ODE reference attached")
        t = case["times"]
        for i, w in enumerate(case["windows"]):
            if i + 1 >= len(tref):
                break
            ref = case["params"]["mass"] * (tref[i + 1] - tref[i])
            w.reference = {quantity: float(ref)}
        return {"windows": len(case["windows"])}
    raise ContractError("unsupported case type for attach_reference")
