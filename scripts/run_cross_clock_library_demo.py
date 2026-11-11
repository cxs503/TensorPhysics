"""Cross-clock library binding demo: TensorLBM fluid + TensorDEM particles.

What this script establishes, in order:

1. **Library anchors** (each leg alone, against a third-party closed form):
   * ``tensorlbm`` D2Q9: Taylor-Green decay rate vs the analytic
     ``exp(-2 nu k^2 t)`` — fixes the ``tau -> nu`` and ``dx, dt`` semantics.
   * ``tensordem`` Stokes drag: velocity relaxation vs the analytic exponential
     ``u + (v0-u) exp(-t/tau_p)`` — fixes the drag law and the step semantics.
2. **Coupled run over three clock ratios** (window ``h`` = 1, 5, 10 ms with the
   particle leg always at ``dt = 50 us``): per-window ledger, system momentum
   bookkeeping, clock round-trip, sync coincidence, and the deviation from the
   exact piecewise-constant (ZOH) solution of the same coupling problem.
3. **Reference comparison**: the finest-window run is the reference; coarser
   windows report the reference drift they cost.
4. Writes ``runs/m14792_cross_clock_library/{results.json,figures/,SHA256SUMS}``.

Run: ``python scripts/run_cross_clock_library_demo.py`` (needs tensorlbm and
tensordem importable; ~1 min on CPU).
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import platform
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.cross_clock.contract import SolverClock  # noqa: E402
from tensorphysics.cross_clock.library_binding import (  # noqa: E402
    FROZEN_PINS,
    CoupledClockRun,
    SyncSchedule,
    TensorDEMLeg,
    TensorLBMLeg,
    UnitMap,
    pin_library,
    verify_pins,
)

OUT = ROOT / "runs" / "m14792_cross_clock_library"
FIG = OUT / "figures"

# ---------------------------------------------------------------- parameters
DX = 1.0e-4           # m, lattice spacing
TAU = 0.80            # BGK relaxation time (lattice)
NU_SI = 1.0e-6        # m^2/s, water at 20 C
RHO0 = 1000.0         # kg/m^3
DT_LBM = (TAU - 0.5) / 3.0 * DX**2 / NU_SI   # 1.0e-3 s (locked by nu)
NX, NY = 64, 8        # periodic D2Q9 domain, 512 cells
U_FLUID0 = 1.0e-3     # m/s, lattice Mach 0.01
RADIUS = 2.1e-3       # m
RHO_P = 1000.0        # kg/m^3
MU = 1.0e-3           # Pa s
DT_DEM = 5.0e-5       # s
T_FINAL = 0.2         # s
BOX = 0.05            # m half-width of the particle domain
CASES = [("fine", 1, 20), ("medium", 5, 100), ("coarse", 10, 200)]


def dem_config():
    from tensordem.coupling_kernel import DEMKernelConfig

    return DEMKernelConfig(
        radius=RADIUS,
        density=RHO_P,
        dt=DT_DEM,
        gravity=(0.0, 0.0, 0.0),
        fluid_density=RHO0,
        fluid_viscosity=MU,
        fluid_velocity=(U_FLUID0, 0.0, 0.0),
        drag_model="stokes",
        box_lo=(-BOX, -BOX, -BOX),
        box_hi=(BOX, BOX, BOX),
        wall_axes=(False, False, False),
    )


# ------------------------------------------------------------------ anchors
def taylor_green_anchor(n_steps: int = 200, n: int = 32) -> dict:
    """Decay of a 2-D Taylor-Green vortex, tensorlbm only, vs exp(-2 nu k^2 t)."""
    import torch

    import tensorlbm as tl

    units = UnitMap(dx=DX, dt=DT_LBM, rho0=RHO0)
    nu_lat = (TAU - 0.5) / 3.0
    k = 2.0 * math.pi / n
    rho = torch.ones((n, n), dtype=torch.float64)
    x = torch.arange(n, dtype=torch.float64)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    u0 = 1e-3
    f = tl.equilibrium(rho, u0 * torch.sin(k * X) * torch.cos(k * Y),
                       -u0 * torch.cos(k * X) * torch.sin(k * Y))

    def amp(field):
        _, ux, uy = tl.macroscopic(field)
        return float((ux**2 + uy**2).sqrt().max())

    a0 = amp(f)
    for _ in range(n_steps):
        f = tl.stream(tl.collide_bgk(f, TAU))
    a1 = amp(f)
    measured = math.log(a0 / a1) / (n_steps * nu_lat * 2.0 * k**2)  # 1.0 == analytic
    return {
        "cells": n * n,
        "steps": n_steps,
        "u0_lattice": u0,
        "nu_lattice": nu_lat,
        "nu_si_from_units_m2_s": units.viscosity_si(TAU),
        "decay_rate_ratio_measured_over_analytic": measured,
        "rel_error": abs(measured - 1.0),
        "amplitude_first": a0,
        "amplitude_last": a1,
    }


def dem_relaxation_anchor(n_steps: int = 4000) -> dict:
    """Single particle relaxing in a held Stokes flow, tensordem only."""
    cfg = dem_config()
    clock = SolverClock("dem", DT_DEM)
    leg = TensorDEMLeg("dem", clock, cfg, positions=[[0.0, 0.0, 0.0]])
    tau_p = leg.mass / cfg.stokes_coefficient
    leg.advance(n_steps)
    v = leg.velocities()[0][0]
    analytic = U_FLUID0 * (1.0 - math.exp(-leg.time / tau_p))
    return {
        "steps": n_steps,
        "t_s": leg.time,
        "particle_mass_kg": leg.mass,
        "stokes_coefficient_n_s_m": cfg.stokes_coefficient,
        "relaxation_time_s": tau_p,
        "dt_over_tau": DT_DEM / tau_p,
        "v_measured_m_s": v,
        "v_analytic_m_s": analytic,
        "rel_error": abs(v - analytic) / abs(analytic),
    }


def zoh_reference(u_history, v0: float, dt: float, tau_p: float) -> np.ndarray:
    """Exact solution of dv/dt = -(v-u_k)/tau_p with u held per window."""
    v = np.empty(len(u_history) + 1)
    v[0] = v0
    decay = math.exp(-dt / tau_p)
    for i, u in enumerate(u_history):
        v[i + 1] = u + (v[i] - u) * decay
    return v


# ------------------------------------------------------------- coupled runs
def coupled_case(label: str, r_lbm: int, r_dem: int, reference: dict | None = None) -> dict:
    units = UnitMap(dx=DX, dt=DT_LBM, rho0=RHO0)
    window = r_lbm * DT_LBM
    clocks = {
        "lbm": SolverClock("lbm", DT_LBM),
        "dem": SolverClock("dem", DT_DEM),
    }
    schedule = SyncSchedule(window, clocks)
    ratios = schedule.validate()
    if ratios["dem"] != r_dem:  # the case table must agree with the clocks
        raise SystemExit(f"{label}: r_dem {ratios['dem']} != declared {r_dem}")

    n_windows = int(round(T_FINAL / window))
    fluid = TensorLBMLeg(
        "lbm", clocks["lbm"], units, nx=NX, ny=NY, tau=TAU,
        dtype="float64", initial_velocity_si=(U_FLUID0, 0.0),
    )
    particles = TensorDEMLeg("dem", clocks["dem"], dem_config(), positions=[[0.0, 0.0, 0.0]])
    tau_p = particles.mass / particles.base_config.stokes_coefficient
    rebuild_err = particles.rebuild_roundtrip_error()

    t0 = time.perf_counter()
    result = CoupledClockRun(fluid, particles, schedule).run(n_windows)
    wall = time.perf_counter() - t0

    series = result["series"]
    u_hist = np.array([u[0] for u in series["u_fluid_si"]])
    v_hist = np.array([v[0] for v in series["v_particle_si"]])
    v_ref = zoh_reference(u_hist, 0.0, window, tau_p)

    out = {
        "label": label,
        "window_s": window,
        "r_lbm": ratios["lbm"],
        "r_dem": ratios["dem"],
        "n_windows": n_windows,
        "t_final_s": result["t_final_s"],
        "tau_particle_s": tau_p,
        "window_over_tau": window / tau_p,
        "rebuild_roundtrip_rel_error": rebuild_err,
        "ledger_drift_max": result["ledger"]["momentum"]["ledger_drift_max"],
        "book_vs_produced_rel": result["book_vs_produced_rel"],
        "stagger_residual_max": result["ledger"]["momentum"]["stagger_residual_max"],
        "measured_system_momentum_residual_rel": result["measured_system_momentum_residual_rel"],
        "produced_impulse_si": result["produced_impulse_si"][0],
        "measured_fluid_momentum_delta_si": result["measured_fluid_momentum_delta_si"][0],
        "v_final_m_s": float(v_hist[-1]),
        "v_zoh_analytic_final_m_s": float(v_ref[-1]),
        "v_vs_zoh_rel_error": float(abs(v_hist[-1] - v_ref[-1]) / abs(v_ref[-1])),
        "v_vs_zoh_max_rel_error": float(np.max(np.abs(v_hist - v_ref[1:]) / np.abs(v_ref[1:]))),
        "clock_roundtrip": result["clock_roundtrip"],
        "wall_seconds": wall,
        "series": {
            "t_s": [x for x in series["t_s"]],
            "u_fluid_si": [u[0] for u in series["u_fluid_si"]],
            "v_particle_si": v_hist.tolist(),
            "v_zoh_analytic_si": v_ref[1:].tolist(),
            "impulse_si": [j[0] for j in series["impulse_si"]],
        },
    }
    out["clock_roundtrip_ok"] = (
        out["clock_roundtrip"]["fluid_steps"] == out["clock_roundtrip"]["fluid_steps_expected"]
        and out["clock_roundtrip"]["particle_steps"] == out["clock_roundtrip"]["particle_steps_expected"]
        and abs(out["clock_roundtrip"]["fluid_time_s"] - out["clock_roundtrip"]["schedule_time_s"]) < 1e-15
        and abs(out["clock_roundtrip"]["particle_time_s"] - out["clock_roundtrip"]["schedule_time_s"]) < 1e-15
    )
    if reference is not None:
        ref_series = reference["series"]
        # compare at the times the coarse run actually has
        ref_t = np.array(ref_series["t_s"])
        ref_v = np.array(ref_series["v_particle_si"])
        ref_u = np.array(ref_series["u_fluid_si"])
        idx = [int(np.argmin(np.abs(ref_t - t))) for t in series["t_s"]]
        v_ref_at = ref_v[idx]
        u_ref_at = ref_u[idx]
        out["reference_drift"] = {
            "v_rel": float(np.max(np.abs(v_hist - v_ref_at) / np.abs(v_ref_at))),
            "u_fluid_rel": float(np.max(np.abs(u_hist - u_ref_at) / np.abs(u_ref_at))),
            "reference_label": reference["label"],
            "reference_window_s": reference["window_s"],
        }
    return out


# ------------------------------------------------------------------ figures
def make_figures(cases) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    figures = []

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    colors = {"fine": "#1f77b4", "medium": "#ff7f0e", "coarse": "#d62728"}
    for c in cases:
        s = c["series"]
        ax.plot(s["t_s"], np.array(s["v_particle_si"]) * 1e3, "-",
                lw=1.6, color=colors.get(c["label"], None),
                label=f"{c['label']}: h={c['window_s']*1e3:.0f} ms, r={c['r_lbm']}:{c['r_dem']}")
        ax.plot(s["t_s"], np.array(s["v_zoh_analytic_si"]) * 1e3, ":", lw=1.0,
                color=colors.get(c["label"], None), alpha=0.9)
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"particle velocity $v_x$ [mm/s]")
    ax.set_title("TensorDEM particle dragged by the TensorLBM fluid (dotted = exact ZOH)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    p = FIG / "velocity_vs_time.png"
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    figures.append({"path": p, "caption": "Particle velocity vs time for three coupling-window sizes; dotted lines are the exact solution of the same piecewise-constant (ZOH) coupling problem."})

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    h = np.array([c["window_s"] for c in cases if "reference_drift" in c])
    dv = np.array([c["reference_drift"]["v_rel"] for c in cases if "reference_drift" in c])
    stag = np.array([c["stagger_residual_max"] for c in cases if "reference_drift" in c])
    if h.size:
        ax.loglog(h * 1e3, dv, "o-", label="reference drift, $v_x$ (vs finest window)")
        ax.loglog(h * 1e3, stag, "s--", label="stagger residual (held vs realised $u_f$)")
        ax.set_xlabel("coupling window h [ms]")
        ax.set_ylabel("relative difference")
        ax.set_title("Cost of coarsening the coupling window")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    p = FIG / "reference_drift_vs_window.png"
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    figures.append({"path": p, "caption": "Reference drift and stagger residual versus coupling window; both grow with h, the ledger residual stays at round-off."})
    return figures


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pins = {name: pin_library(p.library, p.distribution, tuple(p.api)).as_dict()
            for name, p in FROZEN_PINS.items()}
    drift = verify_pins()

    anchors = {
        "tensorlbm_taylor_green": taylor_green_anchor(),
        "tensordem_stokes_relaxation": dem_relaxation_anchor(),
    }

    cases = []
    reference = None
    for label, r_lbm, r_dem in CASES:
        case = coupled_case(label, r_lbm, r_dem, reference=reference)
        if reference is None:
            reference = case
        cases.append(case)

    figures = make_figures(cases)
    results = {
        "mission": "m14792",
        "anchor": "GAP-CROSS-CLOCK",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "contract_pins": pins,
        "contract_pin_drift": drift,
        "parameters": {
            "dx_m": DX, "tau": TAU, "nu_si_m2_s": NU_SI, "rho0_kg_m3": RHO0,
            "dt_lbm_s": DT_LBM, "dt_dem_s": DT_DEM, "nx": NX, "ny": NY,
            "u_fluid0_m_s": U_FLUID0, "particle_radius_m": RADIUS,
            "particle_density_kg_m3": RHO_P, "mu_pa_s": MU, "t_final_s": T_FINAL,
        },
        "anchors": anchors,
        "cases": cases,
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2))

    lines = []
    for f in sorted(OUT.rglob("*")):
        if f.is_file() and f.name != "SHA256SUMS":
            lines.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(OUT)}")
    (OUT / "SHA256SUMS").write_text("\n".join(lines) + "\n")

    print(json.dumps({
        "pin_drift": drift,
        "anchors": {k: {kk: vv for kk, vv in v.items() if "error" in kk or "ratio" in kk}
                    for k, v in anchors.items()},
        "cases": [{k: c[k] for k in (
            "label", "window_s", "r_lbm", "r_dem", "ledger_drift_max", "book_vs_produced_rel",
            "stagger_residual_max", "measured_system_momentum_residual_rel",
            "v_vs_zoh_rel_error", "clock_roundtrip_ok")} | {"reference_drift": c.get("reference_drift")}
            for c in cases],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())