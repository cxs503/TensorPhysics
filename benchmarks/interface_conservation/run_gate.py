"""Driver for the P189-N01 interface-conservation gate.

Runs the three measurement arms of the node and writes a machine-readable record
plus the report figures:

1. **FVM arm** -- staggered finite-volume cavity at ``Re = 100``: Ghia et al.
   (1982) centreline comparison and the interface (``x = 0.5``) fluxes.
2. **Transfer arm** -- the conservative (flux-derived) versus state (velocity)
   mass-flux transfer on the synthetic channel ``channel_coupling_02``, at a
   single resolution and across a refinement sequence in two Mach scalings.
3. **Closure exactness arm** -- the unit gates that make the transfer arm
   trustworthy: uniform advection must be reproduced to round-off, a closed box
   must conserve mass to round-off, and the momentum-exchange accounting of an
   internal solid must balance.

Usage::

    python benchmarks/interface_conservation/run_gate.py --out artifacts/interface_conservation
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorphysics.interface_conservation.fvm import StaggeredFVM          # noqa: E402
from tensorphysics.interface_conservation.lbm import D2Q9                  # noqa: E402
from tensorphysics.interface_conservation.quantities import (              # noqa: E402
    InterfaceFlux, fit_order, flux_residual, local_flux_residual)
from tensorphysics.interface_conservation.transfer import (                # noqa: E402
    CHANNEL_DOMAIN, run_channel_transfer, transfer_refinement, _inlet_profile)

#: Ghia, Ghia & Shin (1982), J. Comput. Phys. 48, 387, Table I, Re = 100.
#: ``u`` along the vertical centreline ``x = 0.5``.
GHIA_RE100 = np.array([
    (0.0000, 0.00000), (0.0547, -0.04192), (0.0625, -0.04192),
    (0.0703, -0.04775), (0.1016, -0.06434), (0.1719, -0.10150),
    (0.2813, -0.15662), (0.4531, -0.21090), (0.5000, -0.20581),
    (0.6172, -0.13641), (0.7344, 0.00332), (0.8516, 0.23151),
    (0.9531, 0.68717), (0.9609, 0.73722), (0.9688, 0.78871),
    (0.9766, 0.84123), (1.0000, 1.00000),
])


def fvm_arm(n: int = 48, steps: int | None = None) -> dict:
    """Staggered FVM cavity: Ghia comparison + interface fluxes at x = 0.5."""
    f = StaggeredFVM(n, n, re=100.0, kappa=0.8)
    f.set_closed_cavity(1.0)
    dt = 0.2 / n
    steps = steps or int(round(20.0 / dt))
    for _ in range(steps):
        f.step(dt)
    # centreline u at x = 0.5
    ys = GHIA_RE100[:, 0]
    u_num = np.array([f.velocity_at(0.5, min(max(y, 1e-6), 1 - 1e-6))[0]
                      for y in ys])
    u_ref = GHIA_RE100[:, 1]
    mask = ys > 0.02
    err = np.abs(u_num[mask] - u_ref[mask])
    # interface (east *interior* section at x = 0.5 is an interior column here;
    # take the column face that sits exactly on x = 0.5)
    face = f.face_flux("east")
    w = f.h
    return {
        "n": n,
        "steps": steps,
        "dt": dt,
        "final_divergence_max": f.max_divergence(),
        "u_max": float(np.abs(f.u).max()),
        "ghia": {
            "y": ys.tolist(),
            "u_fvm": u_num.tolist(),
            "u_ghia": u_ref.tolist(),
            "max_abs_error": float(err.max()),
            "u_centre_error": float(abs(u_num[list(ys).index(0.5)] - u_ref[list(ys).index(0.5)])),
            "max_abs_error_frac_of_urange": float(err.max() / (u_ref.max() - u_ref.min())),
        },
        "interface": {
            "x": 0.5,
            "mass_integral": float(np.sum(face["mass"]) * w),
            "momentum_integral": float(np.sum(face["momentum"]) * w),
            "energy_integral": float(np.sum(face["energy"]) * w),
            "mass_profile": (face["mass"] * w).tolist(),
            "momentum_profile": (face["momentum"] * w).tolist(),
        },
    }


def closure_arm(n: int = 16) -> dict:
    """Unit gates: uniform advection, closed-box mass, momentum exchange."""
    u = 0.05
    lbm = D2Q9(n, n, tau=0.6)
    lbm.set_equilibrium((u, 0.0))
    bcs = {s: ("velocity", u, 0.0) for s in ("north", "south", "east", "west")}
    for _ in range(200):
        lbm.step(bcs)
    vel = lbm.velocity()
    uniform_err = float(max(np.abs(vel[0] - u).max(), np.abs(vel[1]).max()))
    rho_err = float(np.abs(lbm.rho - 1.0).max())

    box = D2Q9(n, n, tau=0.6)
    box.set_equilibrium((0.02, 0.0))
    m0 = float(box.f.sum())
    walls = {s: ("wall", 0.0, 0.0) for s in ("north", "south", "east", "west")}
    for _ in range(300):
        box.step(walls)
    mass_drift = abs(float(box.f.sum()) - m0) / m0

    # momentum exchange with an internal solid disc (Ladd bounce-back), measured
    # by the total (fluid + solid) momentum budget of the collision-stream step
    yy, xx = np.mgrid[0:n, 0:n]
    cx, cy, r = n / 2.0, n / 2.0, max(2.0, n / 8.0)
    solid = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    part = D2Q9(n, n, tau=0.6, solid=solid)
    part.set_equilibrium((0.02, 0.0))    # relative motion -> a real exchange
    part.wall_u[0][solid] = 0.0
    p_before = part.momentum()
    total_exchange = np.zeros(2)
    for _ in range(50):
        part.step(walls, track_exchange=True)
        total_exchange += part.last_exchange_momentum
    p_after = part.momentum()
    exchange = total_exchange / 50.0
    # the momentum handed to the solid must be exactly the momentum the fluid
    # lost (the fluid is otherwise closed: no open boundary, no body force)
    budget_residual = float(np.linalg.norm(p_after - p_before + total_exchange)
                            / max(np.linalg.norm(p_before), 1e-30))
    return {
        "uniform_flow_velocity_error": uniform_err,
        "uniform_flow_density_error": rho_err,
        "closed_box_mass_drift": mass_drift,
        "exchange_force_per_step": exchange.tolist(),
        "exchange_budget_residual": budget_residual,
    }


def transfer_arm(n: int = 32, steps: int = 1200) -> dict:
    out = {"domain": CHANNEL_DOMAIN, "n": n, "steps": steps}
    for mode in ("flux", "velocity"):
        s = run_channel_transfer(n=n, u_lat=0.05, tau=0.6, mode=mode, steps=steps)
        out[mode] = s.as_dict()
    return out


def order_arm(sizes=(16, 32), steps_scale=800) -> dict:
    res = {}
    for scaling in ("fixed", "diffusive"):
        res[scaling] = transfer_refinement(
            sizes=sizes, scaling=scaling,
            steps=int(steps_scale * (sizes[-1] / 16) ** 2))
    return res


def figures(rec: dict, outdir: Path) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = []
    g = rec["fvm"]["ghia"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    ax[0].plot(g["u_fvm"], g["y"], "o-", ms=3, label="FVM (this work)")
    ax[0].plot(g["u_ghia"], g["y"], "s--", ms=3, label="Ghia et al. (1982)")
    ax[0].set_xlabel("u/U  at x = 0.5")
    ax[0].set_ylabel("y/L")
    ax[0].set_title(f"Re = 100 lid-driven cavity, n = {rec['fvm']['n']}")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    prof = np.array(rec["fvm"]["interface"]["mass_profile"])
    yc = (np.arange(len(prof)) + 0.5) / len(prof)
    ax[1].plot(prof, yc, "-", label="mass flux  rho u_n")
    mom = np.array(rec["fvm"]["interface"]["momentum_profile"])
    ax[1].plot(mom, yc, "--", label="normal momentum flux")
    ax[1].set_xlabel("flux per unit length  (rho U^2 units)")
    ax[1].set_ylabel("y/L")
    ax[1].set_title("FVM interface fluxes at x = 0.5")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    p1 = outdir / "fig_fvm_cavity_interface.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=140)
    plt.close(fig)
    figs.append({"name": "fig_fvm_cavity_interface.png",
                 "caption": "Left: FVM centreline velocity against Ghia et al. (1982) "
                            "at Re = 100. Right: the conserved interface fluxes "
                            "(mass and normal momentum) on the x = 0.5 section.",
                 "path": str(p1)})

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for scaling, col in (("fixed", "C3"), ("diffusive", "C0")):
        arm = rec["order"][scaling]
        ax.loglog(arm["h"], arm["residual_state_exchange"], "o-", color=col,
                  label=f"state exchange, {scaling} (p = {arm['order_state_exchange']:.2f})")
        ax.loglog(arm["h"], arm["residual_conservative"], "s--", color=col,
                  label=f"conservative, {scaling}")
    ax.set_xlabel("h  (1/n)")
    ax.set_ylabel("mass-flux residual")
    ax.set_title("interface mass residual vs grid")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    p2 = outdir / "fig_transfer_order.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    figs.append({"name": "fig_transfer_order.png",
                 "caption": "Interface mass-flux residual against grid spacing for the "
                            "state-exchange and conservative transfer operators, in fixed "
                            "and diffusive Mach-number scaling.",
                 "path": str(p2)})
    return figs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/interface_conservation")
    ap.add_argument("--quick", action="store_true", help="small sizes for a smoke run")
    args = ap.parse_args(argv)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        rec = {
            "fvm": fvm_arm(n=24, steps=3000),
            "closure": closure_arm(n=16),
            "transfer": transfer_arm(n=16, steps=400),
            "order": order_arm(sizes=(16, 32), steps_scale=400),
        }
    else:
        rec = {
            "fvm": fvm_arm(n=32, steps=12000),
            "closure": closure_arm(n=16),
            "transfer": transfer_arm(n=32, steps=800),
            "order": order_arm(sizes=(16, 32), steps_scale=200),
        }
    rec["node"] = "P189-N01"
    rec["domain"] = {"cavity": "box_coupling_01", "channel": CHANNEL_DOMAIN}
    # criterion verdicts, computed from the measurements (no hand-typed numbers)
    t = rec["transfer"]
    rec["verdicts"] = {
        "mass_residual_conservative": t["flux"]["integral_residual"],
        "mass_residual_state_exchange": t["velocity"]["integral_residual"],
        "mass_conservative_below_1e6": bool(t["flux"]["integral_residual"] < 1e-6),
        "state_exchange_below_1e6": bool(t["velocity"]["integral_residual"] < 1e-6),
        "observed_order_fixed_mach": rec["order"]["fixed"]["order_state_exchange"],
        "observed_order_diffusive": rec["order"]["diffusive"]["order_state_exchange"],
        "order_at_least_1_diffusive": bool(
            rec["order"]["diffusive"]["order_state_exchange"] >= 1.0),
        "closure_uniform_exact": rec["closure"]["uniform_flow_velocity_error"],
        "closure_closed_box_mass_drift": rec["closure"]["closed_box_mass_drift"],
        "ghia_max_error": rec["fvm"]["ghia"]["max_abs_error"],
        "ghia_max_error_frac": rec["fvm"]["ghia"]["max_abs_error_frac_of_urange"],
    }
    figs = figures(rec, outdir)
    rec["figures"] = figs
    (outdir / "gate_report.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps({k: v for k, v in rec["verdicts"].items()}, indent=2))
    print(f"artifacts written to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())