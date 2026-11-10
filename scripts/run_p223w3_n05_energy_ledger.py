#!/usr/bin/env python3
"""Run the ``p223w3-n05`` coupling energy-ledger study and emit artifacts.

The study is shardable because the full three-chain ladder is minutes of work on
a control-plane pod; the merge step recomputes the gate from the merged record
and never trusts a stored verdict.

    python scripts/run_p223w3_n05_energy_ledger.py --chains viv,weld --out shards/a.json
    python scripts/run_p223w3_n05_energy_ledger.py --chains settling --out shards/b.json
    python scripts/run_p223w3_n05_energy_ledger.py --merge

Outputs (all inside the worktree):

    artifacts/p223w3_n05/verification.json      merged study + gate verdicts + provenance
    artifacts/p223w3_n05/energy_ladders.json    compact per-arm ladder table
    artifacts/p223w3_n05/fig1_energy_order.png
    artifacts/p223w3_n05/fig2_attribution_loss.png
    artifacts/p223w3_n05/fig3_balance_quadrature.png
    artifacts/p223w3_n05/fig4_flux_vs_energy.png
    artifacts/p223w3_n05/SHA256SUMS
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.coupling_energy.exchange_energy import (  # noqa: E402
    ENERGY_ARMS,
    format_table,
    gate_energy,
    run_energy_study,
)

OUT = ROOT / "artifacts" / "p223w3_n05"
SHARDS = OUT / "shards"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # pragma: no cover - provenance only
        return "unavailable"


def _provenance() -> dict:
    import matplotlib as mpl
    import scipy

    return {
        "node": "p223w3-n05-physics-time-exchange",
        "mission": 10561,
        "repo": "TensorPhysics",
        "module": "tensorphysics.coupling_energy.exchange_energy",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": _git("rev-parse", "--short", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "platform": platform.machine(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": mpl.__version__,
        "arms": sorted(ENERGY_ARMS),
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _slope_label(xs, ys) -> str:
    if len(xs) < 2:
        return ""
    order = np.polyfit(np.log(xs), np.log(ys), 1)[0]
    return f"slope {order:.2f}"


def plot_energy_order(study: dict) -> Path:
    chains = sorted(study["chains"])
    arms = [("lag_endpoint", "lag (end-stamped hold)", "o", "-"),
            ("centered_endpoint", "centered / CN midpoint", "s", "-"),
            ("predictor_corrector_endpoint", "predictor-corrector", "^", "--"),
            ("centered_destamp", "centered + de-stamp (window stamp)", "d", ":")]
    fig, axes = plt.subplots(1, len(chains), figsize=(4.4 * len(chains), 3.9),
                             squeeze=False)
    for ax, name in zip(axes[0], chains):
        for arm, label, marker, style in arms:
            levels = study["chains"][name]["arms"][arm]["levels"]
            xs = [lv["dt"] for lv in levels]
            ys = [max(lv["energy_defect"], 1e-17) for lv in levels]
            ax.loglog(xs, ys, marker=marker, linestyle=style, ms=4.5, lw=1.4,
                      label=f"{label} ({_slope_label(xs, ys)})")
        ax.axhline(1e-5, color="k", lw=0.8, ls="-.", alpha=0.6)
        ax.text(ax.get_xlim()[0], 1.2e-5, "criterion (d) 1e-5", fontsize=7)
        ax.set_title(f"{name} chain", fontsize=9)
        ax.set_xlabel("exchange step dt [s]")
        ax.set_ylabel("interface energy defect (L1, -)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=6.5, loc="upper left")
    fig.suptitle("Interface-energy defect versus exchange step: the delay is "
                 "first order, the time-centred operators are second order", fontsize=9)
    fig.tight_layout()
    path = OUT / "fig1_energy_order.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_attribution_loss(study: dict) -> Path:
    tail = {row["ratio"]: row for row in study["booking"]["tail"]}
    comp = {row["ratio"]: row for row in study["booking"]["complete"]}
    ratios = sorted(tail)
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(ratios, [(r - 1.0) / r for r in ratios], "k--", lw=1.2,
            label="(r-1)/r analytic")
    ax.plot(ratios, [tail[r]["unattributed_share"] for r in ratios], "o-",
            color="crimson", label="tail-only booking (shipped practice)")
    ax.plot(ratios, [comp[r]["unattributed_share"] for r in ratios], "s-",
            color="seagreen", label="interval-complete booking")
    ax.axhspan(0.87, 0.94, color="crimson", alpha=0.10)
    ax.text(2.05, 0.905, "reported 87-94 % loss band", fontsize=7, color="crimson")
    for r in ratios:
        ax.annotate(f"{tail[r]['unattributed_share']:.3f}", (r, tail[r]["unattributed_share"]),
                    textcoords="offset points", xytext=(3, -11), fontsize=7,
                    color="crimson")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ratios)
    ax.set_xticklabels([str(r) for r in ratios])
    ax.set_xlabel("sub-cycling ratio r (fine steps per exchange interval)")
    ax.set_ylabel("unattributed share of the exchanged energy")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Interval completeness in the conserved quantity (weld chain)", fontsize=9)
    fig.tight_layout()
    path = OUT / "fig2_attribution_loss.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_balance_quadrature(study: dict) -> Path:
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for name, rows in sorted(study["quadrature"].items()):
        xs = [row["substeps"] for row in rows]
        ax.loglog(xs, [max(row["balance_residual"], 1e-16) for row in rows], "o-",
                  label=f"{name}: balance residual")
        ax.loglog(xs, [max(row["energy_defect"], 1e-16) for row in rows], "x--",
                  alpha=0.7, label=f"{name}: energy defect (operator)")
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16])
    ax.set_xticklabels(["4", "8", "16"])
    ax.set_xlabel("quadrature sub-steps per fine window")
    ax.set_ylabel("relative residual")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6.5)
    ax.set_title("Balance closure is grid-limited; the energy defect is not", fontsize=9)
    fig.tight_layout()
    path = OUT / "fig3_balance_quadrature.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_flux_vs_energy(study: dict) -> Path:
    chains = sorted(study["cross_check"])
    width = 0.2
    xs = np.arange(len(chains))
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for i, (scheme, colour) in enumerate((("lag", "crimson"), ("centered", "seagreen"))):
        flux = [max(study["cross_check"][c]["flux_defect"][scheme], 1e-18) for c in chains]
        energy = [max(study["cross_check"][c]["energy_defect"][scheme], 1e-18)
                  for c in chains]
        ax.bar(xs + (i * 2 - 1.5) * width, flux, width, color=colour, alpha=0.45,
               label=f"{scheme}: flux defect")
        ax.bar(xs + (i * 2 - 0.5) * width, energy, width, color=colour,
               label=f"{scheme}: energy defect")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(chains)
    ax.set_ylabel("relative defect (same configuration)")
    ax.grid(axis="y", which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Flux and energy are different functionals, same verdict", fontsize=9)
    fig.tight_layout()
    path = OUT / "fig4_flux_vs_energy.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------
def run_shard(chains: list[str], out: Path) -> dict:
    study = run_energy_study(chains=tuple(chains))
    record = {"provenance": _provenance(), "study": study,
              "shard": {"chains": chains,
                        "generated_at": datetime.now(timezone.utc).isoformat(
                            timespec="seconds")}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1, default=str))
    print(f"shard {chains} -> {out} sha256 {_sha256(out)}")
    return record


def merge(shard_paths: list[Path] | None = None) -> dict:
    shard_paths = shard_paths or sorted(SHARDS.glob("*.json"))
    if not shard_paths:
        raise SystemExit("no shards found; run with --chains first")
    merged: dict = {"engine": None, "dt_fractions": None, "arms": None,
                    "chains": {}, "booking": None, "quadrature": {},
                    "cross_check": {}}
    shard_records = []
    for path in shard_paths:
        record = json.loads(path.read_text())
        study = record["study"]
        shard_records.append({"path": str(path.relative_to(ROOT)),
                              "sha256": _sha256(path),
                              "chains": record["shard"]["chains"],
                              "generated_at": record["shard"]["generated_at"]})
        for key in ("node", "dt_fractions", "arms"):
            if key in study:
                merged[key if key != "node" else "engine"] = study[key]
        merged["chains"].update(study["chains"])
        merged["quadrature"].update(study["quadrature"])
        merged["cross_check"].update(study["cross_check"])
        if merged["booking"] is None:
            merged["booking"] = study["booking"]
        elif merged["booking"] != study["booking"]:
            # the booking sweep uses the weld chain only; every shard must agree
            raise SystemExit(f"booking sweep disagrees between shards ({path})")

    verdict = gate_energy(merged)
    OUT.mkdir(parents=True, exist_ok=True)
    figures = [plot_energy_order(merged), plot_attribution_loss(merged),
               plot_balance_quadrature(merged), plot_flux_vs_energy(merged)]

    compact = {
        "arms": merged["arms"],
        "chains": {
            name: {
                "meta": payload["meta"],
                "arms": {arm: {"energy_defect_order": ladder["energy_defect_order"],
                               "energy_defect_r2": ladder["energy_defect_r2"],
                               "reference_energy_order": ladder["reference_energy_order"],
                               "finest_energy_defect": ladder["finest_energy_defect"],
                               "finest_reference_energy_error":
                                   ladder["finest_reference_energy_error"],
                               "balance_residual": ladder["balance_residual"],
                               "levels": ladder["levels"]}
                          for arm, ladder in payload["arms"].items()},
            } for name, payload in merged["chains"].items()},
        "booking": merged["booking"],
        "quadrature": merged["quadrature"],
        "cross_check": merged["cross_check"],
    }
    ladders_path = OUT / "energy_ladders.json"
    ladders_path.write_text(json.dumps(compact, indent=1, default=str))

    record = {
        "provenance": _provenance(),
        "shards": shard_records,
        "study": merged,
        "gate": verdict,
        "persisted_baseline": {
            "flux_gate": "artifacts/p223w3_n06/verification.json",
            "note": "the flux/state-side gate of the same exchange algebra; "
                    "this record is the energy-space instrument on top of it",
        },
        "figures": [{"path": str(p.relative_to(ROOT)), "sha256": _sha256(p)}
                    for p in figures],
        "artifacts": {"energy_ladders.json": _sha256(ladders_path)},
    }
    out = OUT / "verification.json"
    out.write_text(json.dumps(record, indent=1, default=str))
    sums = [f"{_sha256(p)}  {p.relative_to(OUT)}"
            for p in sorted([out, ladders_path, *figures])]
    (OUT / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(format_table(verdict))
    print(f"verification -> {out} sha256 {_sha256(out)}")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chains", default="",
                        help="comma separated chain subset for a shard")
    parser.add_argument("--out", default="",
                        help="shard path (default artifacts/p223w3_n05/shards/<tag>.json)")
    parser.add_argument("--merge", action="store_true",
                        help="merge every shard into verification.json")
    args = parser.parse_args(argv)
    if args.merge:
        record = merge()
        return 0 if record["gate"]["verdict"] == "pass" else 1
    chains = [c for c in args.chains.split(",") if c] or ["viv", "weld", "settling"]
    tag = "_".join(chains)
    out = Path(args.out) if args.out else SHARDS / f"{tag}.json"
    run_shard(chains, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())