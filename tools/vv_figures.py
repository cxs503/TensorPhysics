"""Figures for the V&V gate report.

Reads the artifacts produced by ``tools/vv_controls.py`` and
``tools/vv_crossrepo_tensorlbm.py`` (plus the TensorFEM evidence run when
present) and writes:

* ``fig_gci_vs_r.png``        GCI of the manufactured control versus refinement ratio;
* ``fig_order_comparison.png`` observed order of accuracy of the cross-repository
  runs against the theoretical order;
* ``fig_convergence.png``      log-log convergence of the real benchmark error
  norms with the fitted slopes;
* ``fig_vv20.png``             ASME V&V 20 |E|/U_val for the ten table cases;
* ``fig_lbm_error_field.png``  axial-velocity error profile of the TensorLBM
  channel case on three grids (the field being verified).

Usage::  python tools/vv_figures.py [--out artifacts/vv_gate/figures]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from tensorphysics.vv import GatePolicy, Observation, evaluate  # noqa: E402

LBM_RAW = ROOT / "artifacts" / "vv_gate" / "tensorlbm_poiseuille_raw.json"
CONTROLS = ROOT / "artifacts" / "vv_gate" / "controls"
FEM_EVIDENCE = ROOT / "artifacts" / "vv_gate" / "tensorfem_hconv" / "evidence.json"
MMS_RAW = ROOT / "artifacts" / "vv_gate" / "tensorfvm_mms_raw.json"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def fig_gci_vs_r(out: Path) -> Path | None:
    path = CONTROLS / "positive_control.json"
    if not path.exists():
        return None
    data = _load(path)
    rs = [row["r"] for row in data["gci_vs_r"]]
    gcis = [row["gci"] for row in data["gci_vs_r"]]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(rs, gcis, "o-", color="#1f4e79", lw=1.8, ms=6, label="GCI of the fine result")
    for r, g in zip(rs, gcis):
        ax.annotate(f"{g:.4f}", (r, g), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("refinement ratio  r = h_coarse / h_fine  (h_coarse fixed)")
    ax.set_ylabel("GCI$_{fine}$  (F$_s$ = 1.25)")
    ax.set_title("Manufactured control f(h)=3+0.5h$^2$: GCI decreases under refinement")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    target = out / "fig_gci_vs_r.png"
    fig.savefig(target, dpi=170)
    plt.close(fig)
    return target


def fig_order_comparison(out: Path) -> Path | None:
    labels, orders, colours = [], [], []
    if LBM_RAW.exists():
        raw = _load(LBM_RAW)
        for obs, tag in (("rel_l2_error", "LBM rel-L2"), ("flow_rate_error", "LBM flow rate"),
                         ("peak_velocity_error", "LBM peak u")):
            study = evaluate(
                [Observation(h=row["h"], f=row[obs]) for row in raw["levels"]],
                quantity=obs, policy=__import__("tensorphysics.vv", fromlist=["GatePolicy"]).GatePolicy(expected_order=2.0),
            ).study
            labels.append(tag)
            orders.append(study.order_p)
            colours.append("#c00000")
    if FEM_EVIDENCE.exists():
        ev = _load(FEM_EVIDENCE)
        for study_name, key in (("manufactured", "relative_l2_displacement_error"),
                                ("bending", "relative_strain_energy_error")):
            node = ev.get(study_name)
            if not node:
                continue
            levels = [Observation(h=row["mesh_size"], f=row[key]) for row in node["levels"]]
            study = evaluate(levels, quantity=study_name).study
            if study.order_p is None:
                continue
            labels.append(f"FEM {study_name}")
            orders.append(study.order_p)
            colours.append("#2e7d32")
    if not labels:
        return None
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    bars = ax.bar(range(len(labels)), orders, color=colours, width=0.55)
    ax.axhline(2.0, color="#1f4e79", ls="--", lw=1.4, label="theoretical order p = 2")
    ax.axhspan(1.8, 2.2, color="#1f4e79", alpha=0.10, label="±0.2 tolerance")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=12, fontsize=8)
    ax.set_ylabel("observed order of accuracy p")
    ax.set_title("Cross-repository runs through the same gate")
    for rect, value in zip(bars, orders):
        ax.annotate(f"{value:.3f}", (rect.get_x() + rect.get_width() / 2, value),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    target = out / "fig_order_comparison.png"
    fig.savefig(target, dpi=170)
    plt.close(fig)
    return target


def fig_convergence(out: Path) -> Path | None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    plotted = False
    if LBM_RAW.exists():
        raw = _load(LBM_RAW)
        hs = [row["h"] for row in raw["levels"]]
        for obs, tag, colour in (("rel_l2_error", "LBM rel-L2 error", "#c00000"),
                                 ("flow_rate_error", "LBM flow-rate error", "#e08a00"),
                                 ("peak_velocity_error", "LBM peak-velocity error", "#7b3f00")):
            fs = [row[obs] for row in raw["levels"]]
            p, _src, _res = __import__("tensorphysics.vv", fromlist=["observed_order"]).observed_order(
                [Observation(h=h, f=f) for h, f in zip(hs, fs)])
            ax.loglog(hs, fs, "o-", color=colour, lw=1.6, ms=5,
                      label=f"{tag}  (slope {p:.2f})")
            plotted = True
    if FEM_EVIDENCE.exists():
        ev = _load(FEM_EVIDENCE)
        node = ev.get("manufactured")
        if node:
            hs = [row["mesh_size"] for row in node["levels"]]
            fs = [row["relative_l2_displacement_error"] for row in node["levels"]]
            p, _src, _res = __import__("tensorphysics.vv", fromlist=["observed_order"]).observed_order(
                [Observation(h=h, f=f) for h, f in zip(hs, fs)])
            ax.loglog(hs, fs, "s-", color="#2e7d32", lw=1.6, ms=6,
                      label=f"FEM manufactured error  (slope {p:.2f})")
            plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel("grid length h")
    ax.set_ylabel("relative error norm")
    ax.set_title("Measured convergence of the cross-repository benchmark errors")
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    target = out / "fig_convergence.png"
    fig.savefig(target, dpi=170)
    plt.close(fig)
    return target


def fig_vv20(out: Path) -> Path | None:
    path = CONTROLS / "vv20_table.json"
    if not path.exists():
        return None
    data = _load(path)
    rows = data["rows"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ratios = [row["abs_error_over_u_val"] for row in rows]
    colours = ["#2e7d32" if row["status_gate"] == "PASS" else "#c00000" for row in rows]
    bars = ax.bar(range(len(rows)), ratios, color=colours, width=0.6)
    ax.axhline(1.0, color="k", ls="--", lw=1.2, label="acceptance limit |E| = U$_{val}$")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([row["label"] for row in rows], fontsize=8)
    ax.set_ylabel("|E| / U$_{val}$")
    ax.set_title("ASME V&V 20 acceptance: 10 table cases, gate vs hand computation")
    for rect, row in zip(bars, rows):
        ax.annotate(f"{row['abs_error_over_u_val']:.3f}",
                    (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    target = out / "fig_vv20.png"
    fig.savefig(target, dpi=170)
    plt.close(fig)
    return target


def fig_lbm_error_field(out: Path) -> Path | None:
    """Axial-velocity error profile of the TensorLBM channel case (the field verified)."""
    src = Path("/root/TensorLBM/src")
    if not src.exists():
        return None
    dt = Path(ROOT / "artifacts" / "vv_gate" / "lbm_profiles.json")
    if not dt.exists():
        return None
    data = _load(dt)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    ax = axes[0]
    for row in data["profiles"]:
        ys = [i / (len(row["ux"]) - 1) for i in range(len(row["ux"]))]  # normalised height
        ax.plot(row["ux"], ys, lw=1.6, label=f"ny={row['ny']}")
    ax.plot(data["analytic"], data["analytic_y"], "k--", lw=1.2, label="analytic")
    ax.set_xlabel("axial velocity $u_x$ (lattice units)")
    ax.set_ylabel("normalised channel height  y / (ny-1)")
    ax.set_title("TensorLBM Poiseuille: velocity profiles")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    for row in data["profiles"]:
        ys = [i / (len(row["error"]) - 1) for i in range(len(row["error"]))]
        ax.plot(row["error"], ys, lw=1.6, label=f"ny={row['ny']}")
    ax.axvline(0.0, color="k", lw=0.8)
    ax.set_xlabel("$u_x - u_x^{analytic}$ (lattice units)")
    ax.set_ylabel("normalised channel height  y / (ny-1)")
    ax.set_title("Error profile on three grids")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    target = out / "fig_lbm_error_field.png"
    fig.savefig(target, dpi=170)
    plt.close(fig)
    return target



def fig_fvm_mms(out: Path) -> Path | None:
    """TensorFVM manufactured-solution study: error norms, fitted orders, h^2 rail."""
    if not MMS_RAW.exists():
        return None
    raw = _load(MMS_RAW)
    hs = [row["h"] for row in raw["levels"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    styles = (("l2_interior", "interior L2 (wall layer excluded)", "#1a4f9c", "o"),
              ("l2_global", "global L2 (all cells)", "#c00000", "s"),
              ("linf_global", "global L-infinity", "#6b6b6b", "^"))
    h0 = min(hs)
    anchor = None
    for obs, label, colour, marker in styles:
        fs = [row[obs] for row in raw["levels"]]
        verdict = evaluate([Observation(h=h, f=f) for h, f in zip(hs, fs)],
                           quantity=obs, policy=GatePolicy(expected_order=2.0))
        p = verdict.study.order_p
        ax.loglog(hs, fs, marker + "-", color=colour, lw=1.6, ms=5,
                  label=f"{label}: p = {p:.3f}, gate {verdict.status}")
        if anchor is None:
            anchor = fs[-1] / h0 ** 2
    hh = [h0, max(hs)]
    ax.loglog(hh, [anchor * h ** 2 for h in hh], "k--", lw=1.0, label="slope h^2 (design order)")
    ax.set_xlabel("cell size h = 1/n (n^3 hexahedral box)")
    ax.set_ylabel("error norm (m/s)")
    ax.set_title("TensorFVM manufactured solution: gate-measured order vs design order 2")
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    target = out / "fig_fvm_mms.png"
    fig.savefig(target, dpi=170)
    plt.close(fig)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/vv_gate/figures")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    made = []
    for fn in (fig_gci_vs_r, fig_order_comparison, fig_convergence, fig_vv20, fig_lbm_error_field, fig_fvm_mms):
        try:
            target = fn(out)
        except Exception as exc:  # pragma: no cover - figure best effort
            print(f"{fn.__name__}: skipped ({exc})")
            continue
        if target:
            made.append(target)
            print(f"wrote {target}")
        else:
            print(f"{fn.__name__}: inputs missing")
    (out / "manifest.json").write_text(
        json.dumps({"figures": [p.name for p in made]}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())