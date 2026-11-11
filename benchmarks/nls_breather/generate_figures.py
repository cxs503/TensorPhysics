"""Generate figures for the NLS / Peregrine breather extreme-wave-focusing
report (p472w2-N08).

Figures:
1.  Peregrine breather amplitude |A(x,t)| -- the exact-solution anchor (C1).
2.  Benjamin-Feir dimensionless growth rate vs wavenumber -- analytic vs
    numerical (C2).
3.  Conservation invariant drift vs time (C3).
4.  Second-order crest vs breather crest comparison (C4).
"""

from __future__ import annotations

import base64
import io
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorphysics.nls_breather import (
    NLSSolver,
    benjamin_feir_dimensionless,
    benjamin_feir_scan_verification,
    breather_vs_second_order,
    conservation_drift_study,
    peregrine_envelope,
    peregrine_phase_aligned_l2_error,
)


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def fig1_peregrine_breather() -> dict:
    """Peregrine breather amplitude field and the numerical reproduction."""
    n_x = 1024
    L = 30.0
    dt = 1e-3
    t0 = -1.0
    solver = NLSSolver(n_x=n_x, L=L, dt=dt, n_t=2000)
    A0 = peregrine_envelope(solver.x, t0)
    n_steps = int(round((0.0 - t0) / dt))
    res = solver.solve(A0, n_steps=n_steps, record_every=1, t0=t0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    # left: analytic breather at t=0
    x = solver.x
    A_ana = peregrine_envelope(x, 0.0)
    axes[0].plot(x, np.abs(A_ana), "k-", lw=2, label="analytic (t=0)")
    axes[0].plot(x, np.abs(res.A[-1]), "r--", lw=1.5, label="numerical (t=0)")
    axes[0].axhline(1.0, color="gray", ls=":", lw=1)
    axes[0].set_xlim(-8, 8)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("|A|")
    axes[0].set_title("Peregrine breather: analytic vs numerical")
    axes[0].legend()
    # right: space-time amplitude
    t_sel = np.linspace(0, len(res.t) - 1, 20).astype(int)
    X, T = np.meshgrid(x, res.t[t_sel])
    axes[1].contourf(X, T, np.abs(res.A[t_sel]), levels=30, cmap="viridis")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("t")
    axes[1].set_title("Numerical |A(x,t)| (focusing)")
    fig.tight_layout()
    return {"name": "fig1_peregrine", "caption": (
        "Peregrine breather: analytic (black) vs numerical (red) amplitude at "
        "the peak time t=0 (left), and the numerical space-time focusing "
        "|A(x,t)| (right).  The numerical solution reproduces the analytic "
        "breather to a phase-aligned L2 error of "
        f"{peregrine_phase_aligned_l2_error(res.A[-1], x, res.t[-1]):.4f} "
        "(<= 2%, C1)."), "png_base64": _fig_to_b64(fig)}


def fig2_benjamin_feir() -> dict:
    """Benjamin-Feir dimensionless growth rate vs wavenumber."""
    hk = np.linspace(0.05, 1.8, 100)
    ana = np.array([benjamin_feir_dimensionless(k) for k in hk])
    scan = benjamin_feir_scan_verification(np.linspace(0.05, 1.8, 50), A0=1.0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(hk, ana, "k-", lw=2, label="analytic")
    ax.plot(scan["hat_kappa"], scan["hat_sigma"], "ro", ms=4,
            label="numerical (scan)")
    ax.axvline(np.sqrt(2.0), color="gray", ls=":", lw=1,
               label=r"$\kappa_{max}=\sqrt{2}$")
    ax.set_xlabel(r"$\hat{\kappa} = \kappa / A_0$")
    ax.set_ylabel(r"$\hat{\sigma} = \sigma / A_0^2$")
    ax.set_title("Benjamin-Feir dimensionless growth rate")
    ax.legend()
    fig.tight_layout()
    return {"name": "fig2_bf", "caption": (
        "Benjamin-Feir dimensionless growth rate vs wavenumber: analytic "
        "(black) and numerical scan (red).  The maximum relative error is "
        f"{scan['max_relative_error']:.4f} (<= 3%, C2)."), "png_base64": _fig_to_b64(fig)}


def fig3_conservation() -> dict:
    """Conservation invariant drift vs time."""
    c3_solver = NLSSolver(n_x=256, L=15.0, dt=1e-4, n_t=20000)
    c3_A0 = 1.0 + 0.1 * np.exp(-c3_solver.x ** 2 / 4.0) * np.exp(
        1j * 0.3 * c3_solver.x)
    c3_res = c3_solver.solve(c3_A0, n_steps=20000, record_every=2000)
    drift = conservation_drift_study(c3_res.A, c3_solver.dx, c3_res.t)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    # compute the invariant time series
    n_t = c3_res.A.shape[0]
    mass = np.array([drift["mass"]["initial"]] * n_t)
    # use the recorded values
    from tensorphysics.nls_breather import nls_mass, nls_momentum, nls_energy
    mass_ts = np.array([nls_mass(c3_res.A[i], c3_solver.dx) for i in range(n_t)])
    mom_ts = np.array([nls_momentum(c3_res.A[i], c3_solver.dx) for i in range(n_t)])
    ene_ts = np.array([nls_energy(c3_res.A[i], c3_solver.dx) for i in range(n_t)])
    # relative drift
    ax.plot(c3_res.t, np.abs(mass_ts - mass_ts[0]) / abs(mass_ts[0]),
            "b-", lw=1.5, label="mass")
    ax.plot(c3_res.t, np.abs(mom_ts - mom_ts[0]) / (abs(mom_ts[0]) + 1e-30),
            "r-", lw=1.5, label="momentum")
    ax.plot(c3_res.t, np.abs(ene_ts - ene_ts[0]) / abs(ene_ts[0]),
            "g-", lw=1.5, label="energy")
    ax.axhline(1e-6, color="k", ls="--", lw=1, label="1e-6 threshold")
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel("relative drift")
    ax.set_title("NLS conservation invariant drift")
    ax.legend()
    fig.tight_layout()
    return {"name": "fig3_conservation", "caption": (
        "NLS conservation invariant drift vs time: mass (blue), momentum "
        "(red), energy (green).  The maximum absolute drift is "
        f"{drift['max_abs_drift']:.2e} (<= 1e-6, C3)."), "png_base64": _fig_to_b64(fig)}


def fig4_second_order() -> dict:
    """Second-order crest vs breather crest comparison."""
    H_s = 4.0
    T_p = 10.0
    comp = breather_vs_second_order(H_s, T_p)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["1st-order\ncrest", "2nd-order\ncrest", "Peregrine\nbreather"]
    vals = [comp["eta_first_order"], comp["eta_second_order"],
            comp["eta_breather"]]
    colors = ["#4C72B0", "#DD8452", "#C44E52"]
    bars = ax.bar(labels, vals, color=colors, alpha=0.8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.3f}",
                ha="center", fontsize=10)
    ax.set_ylabel("crest (m)")
    ax.set_title(f"Second-order vs breather crest (H_s={H_s} m, T_p={T_p} s)")
    ax.axhline(0, color="k", lw=0.5)
    fig.tight_layout()
    return {"name": "fig4_second_order", "caption": (
        "Second-order spectral-synthesis crest vs the Peregrine breather peak "
        "crest for the same sea state (H_s=4 m, T_p=10 s).  The breather "
        f"crest ({comp['eta_breather']:.3f} m) exceeds the second-order crest "
        f"({comp['eta_second_order']:.3f} m) by a factor of "
        f"{comp['ratio_breather_over_second_order']:.3f} (C4)."), "png_base64": _fig_to_b64(fig)}


def main() -> None:
    figs = [fig1_peregrine_breather(), fig2_benjamin_feir(),
            fig3_conservation(), fig4_second_order()]
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "artifacts")
    os.makedirs(out_dir, exist_ok=True)
    for f in figs:
        path = os.path.join(out_dir, f["name"] + ".png")
        with open(path, "wb") as fh:
            fh.write(base64.b64decode(f["png_base64"]))
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
