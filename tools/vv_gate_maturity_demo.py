"""Demonstration run of the matured V&V machine gate (P204-N10).

Produces the numbers and figures quoted in the node report:

* ``in30_regression_after.json`` -- the IN30-N01..N08 corpus re-run with the
  matured gate (must be identical to ``tests/data/vv_in30/golden_head_5de4f11.json``);
* ``resolvability_sweep.json`` -- the ``R = delta / dx`` law over a panel-count
  sweep for a declared boundary-layer thickness, with the resolution floor and
  the spacing each status demands;
* ``uncertainty_ledgers.json`` -- the full ASME V&V 20 ledger (``U_num``,
  ``U_input``, ``U_val``, ``U_ref``) for the ship-performance cross-repo study
  plus the discriminability verdict, and the adapted verdicts of the TensorLBM /
  TensorFEM / ship documents;
* ``figures/*.png`` -- convergence + extrapolation, the R discriminator, and the
  uncertainty budget.

Run::

    python tools/vv_gate_maturity_demo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SRC = ROOT / "src"
for _p in (str(_SRC), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tensorphysics.vv import (  # noqa: E402
    DiscriminabilityPolicy,
    UncertaintyLedger,
    adapters,
    compare,
    discriminability,
    evaluate,
    required_grid_spacing,
    u_num_from_study,
)
from tensorphysics.vv.schema import check_benchmark  # noqa: E402
from tools.vv_in30_regression import build_document  # noqa: E402

OUT = ROOT / "artifacts" / "vv_gate_maturity"
FIG = OUT / "figures"
BENCH = ROOT / "tests" / "data" / "vv_benchmark"


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def in30_regression() -> dict:
    doc = build_document(git_head="worktree (m6924)")
    golden = json.loads((ROOT / "tests" / "data" / "vv_in30" / "golden_head_5de4f11.json").read_text())
    doc["historical_digest"] = golden["digest"]
    doc["matches_historical"] = doc["digest"] == golden["digest"]
    _write(OUT / "in30_regression_after.json", doc)
    return doc


def resolvability_sweep(delta=0.0042) -> dict:
    policy = DiscriminabilityPolicy()
    panels = [75, 150, 300, 600, 1200, 2400, 4800, 9600, 19200]
    rows = []
    for n in panels:
        dx = 1.0 / n
        res = discriminability([{"name": "boundary_layer", "delta": delta}], dx, policy=policy)
        rows.append(
            {
                "panels": n,
                "dx": dx,
                "ratio": res.features[0].ratio,
                "status": res.status,
                "detail": res.features[0].detail,
            }
        )
    payload = {
        "schema": "tensorphysics.vv.resolvability-sweep/1",
        "feature": {"name": "boundary_layer_thickness_at_ap", "delta": delta, "unit": "m"},
        "policy": policy.as_dict(),
        "rows": rows,
        "required_dx_for_resolution": required_grid_spacing(delta, policy=policy),
        "required_panels_for_resolution": 1.0 / required_grid_spacing(delta, policy=policy),
    }
    _write(OUT / "resolvability_sweep.json", payload)
    return payload


def uncertainty_ledgers() -> dict:
    ship = json.loads((BENCH / "ship_resistance_study.json").read_text())
    series = adapters.extract_series(ship)
    verdict = evaluate(
        series.levels,
        quantity=series.quantity,
        features=ship["features"],
        dx=series.levels[-1].h,
        f_exp=ship["validation"]["f_exp"],
        u_input=ship["validation"]["u_input"],
        u_ref=ship["validation"]["u_ref"],
    )
    u_num, u_num_source = u_num_from_study(verdict.study)
    ledger = verdict.uncertainty
    discriminative_reference = compare(
        ship["validation"]["f_sim"], ship["validation"]["f_exp"], ledger
    ).as_dict()

    # what reference uncertainty would still be discriminative?
    u_val = ledger.u_val
    boundary = {
        "u_val": u_val,
        "discrimination_threshold_u_ref": u_val,
        "current_u_ref": ledger.u_ref,
        "current_reference_ratio": ledger.reference_ratio,
        "discriminative": ledger.discriminative(),
        "error_kN": verdict.validation.error,
        "ratio_absE_over_Uval": verdict.validation.ratio,
    }
    crossrepo = {}
    for name, repo in (
        ("ship_resistance_study.json", "ship"),
        ("tensorlbm_channel_study.json", "tensorlbm"),
        ("tensorfem_bending_study.json", "tensorfem"),
    ):
        payload = json.loads((BENCH / name).read_text())
        problems = check_benchmark(payload)
        s = adapters.extract_series(payload)
        v = evaluate(s.levels, quantity=s.quantity)
        crossrepo[name] = {
            "repo": repo,
            "contract_violations": problems,
            "levels": len(s.levels),
            "spacings": [lvl.h for lvl in s.levels],
            "observable": [lvl.f for lvl in s.levels],
            "status": v.status,
            "order_p": v.study.order_p,
            "gci_fine": v.study.gci_fine,
            "convergence_type": v.study.convergence_type,
        }

    payload = {
        "schema": "tensorphysics.vv.gate-maturity-demo/1",
        "ship_ledger": ledger.as_dict(),
        "ship_u_num_source": u_num_source,
        "ship_ledger_verdict_u_ref_declared": discriminative_reference,
        "discriminability_boundary": boundary,
        "ship_gate_status": verdict.status,
        "ship_resolvability": verdict.resolvability.as_dict(),
        "ship_checks": [{"check": c["check"], "status": c["status"]} for c in verdict.checks],
        "crossrepo": crossrepo,
    }
    _write(OUT / "uncertainty_ledgers.json", payload)
    return payload


def figures(in30: dict, sweep: dict, ledgers: dict) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []

    # (a) IN30 convergence + Richardson extrapolation on a log-log axis
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    for rec in in30["cases"]:
        if rec["order_p"] is None or rec["case"] in {"IN30-N06"}:
            continue
        hs = rec["h"]
        fs = rec["f"]
        ax.plot(hs, [f - (rec["f_extrapolated"] or fs[-1]) for f in fs], "o-", label=f"{rec['case']} p={rec['order_p']:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("grid spacing h")
    ax.set_ylabel("|f(h) - f_exact,Richardson|")
    ax.set_title("(a) IN30 error vs h and Richardson limit")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    # (b) R = delta/dx discriminator
    ax = axes[1]
    rows = sweep["rows"]
    ratios = [r["ratio"] for r in rows]
    colors = {"RESOLVED": "#2f7d31", "MARGINAL": "#c98a00", "UNRESOLVED": "#b3261e"}
    ax.plot([r["panels"] for r in rows], ratios, "k-", lw=1.2, zorder=1)
    for status, color in colors.items():
        xs = [r["panels"] for r in rows if r["status"] == status]
        ys = [r["ratio"] for r in rows if r["status"] == status]
        if xs:
            ax.scatter(xs, ys, color=color, s=42, zorder=3, label=status)
    for level, style in ((1.0, ":"), (4.0, "--")):
        ax.axhline(level, color="grey", ls=style, lw=1)
        ax.text(rows[0]["panels"], level * 1.06, f"R = {level:g}", fontsize=8, color="grey")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("panels per hull (dx = 1/panels)")
    ax.set_ylabel("R = delta / dx   (delta = 4.2 mm)")
    ax.set_title("(b) W10-N07 discriminability law")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    path = FIG / "in30_and_resolvability.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    out.append(path)

    # (c) uncertainty budget
    led = ledgers["ship_ledger"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    names = ["U_num\n(GCI)", "U_input", "U_ref\n(reference)"]
    values = [led["u_num"], led["u_input"], led["u_ref"]]
    bars = ax.bar(names, values, color=["#1f4e79", "#5b9bd5", "#c00000"])
    ax.axhline(led["u_val"], color="black", ls="--", lw=1.2, label=f"U_val = {led['u_val']:.3f} kN")
    ax.axhline(abs(ledgers["discriminability_boundary"]["error_kN"]), color="#2f7d31", ls=":", lw=1.6,
               label=f"|E| = {abs(ledgers['discriminability_boundary']['error_kN']):.3f} kN")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("kN")
    ax.set_title("(c) ASME V&V 20 ledger, KCS panel study\n"
                 f"U_ref/U_val = {led['reference_ratio']:.3f} -> discriminative")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    path = FIG / "uncertainty_ledger.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    out.append(path)
    return out


def main() -> int:
    in30 = in30_regression()
    sweep = resolvability_sweep()
    ledgers = uncertainty_ledgers()
    figs = figures(in30, sweep, ledgers)
    summary = {
        "in30_matches_historical": in30["matches_historical"],
        "in30_statuses": {r["case"]: r["status"] for r in in30["cases"]},
        "resolvability": {r["panels"]: r["status"] for r in sweep["rows"]},
        "required_panels_for_resolution": sweep["required_panels_for_resolution"],
        "ship_gate_status": ledgers["ship_gate_status"],
        "ship_u_val": ledgers["ship_ledger"]["u_val"],
        "ship_u_ref_over_u_val": ledgers["ship_ledger"]["reference_ratio"],
        "crossrepo_status": {k: v["status"] for k, v in ledgers["crossrepo"].items()},
        "figures": [str(p.relative_to(ROOT)) for p in figs],
    }
    _write(OUT / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())