#!/usr/bin/env python3
"""Run the ``p223w3-n06`` coupling time-exchange study and emit artifacts.

Outputs (all inside the worktree):

    artifacts/p223w3_n06/verification.json     study + gate verdicts + provenance
    artifacts/p223w3_n06/cadence_ladders.json  compact per-arm ladder table
    artifacts/p223w3_n06/fig1_operator_order.png
    artifacts/p223w3_n06/fig2_staircase_repair.png
    artifacts/p223w3_n06/fig3_attribution_loss.png
    artifacts/p223w3_n06/fig4_chain_trajectories.png
"""
from __future__ import annotations

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

from tensorphysics.coupling_energy.exchange_gate import (  # noqa: E402
    ARMS,
    RATIOS,
    format_table,
    gate_study,
    run_exchange_study,
)
from tensorphysics.coupling_energy.time_exchange import (  # noqa: E402
    build_chain,
    run_ladder,
)

OUT = ROOT / "artifacts" / "p223w3_n06"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # pragma: no cover - provenance only
        return "unavailable"


def plot_operator_order(study: dict) -> Path:
    mms = study["mms"]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    styles = {
        "lag": ("lag (end stamp)", "o-", "C3"),
        "centered": ("centered (CN midpoint)", "s-", "C0"),
        "predictor_corrector_provisional": ("predictor-corrector, provisional", "^-", "C2"),
        "predictor_corrector": ("predictor-corrector, corrected", "d--", "C2"),
        "window_raw": ("window average, end-stamped (raw)", "v-", "C1"),
        "window_destamp": ("window average, de-stamped", "*--", "C1"),
    }
    for key, (label, marker, colour) in styles.items():
        levels = mms[key]["levels"]
        hs = np.array([lv["h"] for lv in levels])
        errs = np.array([lv["defect_l1"] for lv in levels])
        order, r2 = study["mms"][key]["observed_order"], study["mms"][key]["fit_r2"]
        ax.loglog(hs, errs, marker, color=colour, label=f"{label}: p={order:.2f} (R²={r2:.4f})")
    ax.set_xlabel("exchange interval h (period = 1)")
    ax.set_ylabel(r"$\sum_n|\int q\,\mathrm{d}t - Q_n| / \sum_n|\int q\,\mathrm{d}t|$")
    ax.set_title("Exchange operator order on the coupling MMS (harmonic flux)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    path = OUT / "fig1_operator_order.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_staircase_repair(study: dict) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.0))
    for ax, name in zip(axes, ("viv", "weld", "settling")):
        arms = study["chains"][name]["arms"]
        for arm, marker, colour, label in (
                ("lag_endstamp", "o-", "C3", "lag + end stamp"),
                ("centered_endstamp", "s-", "C1", "CN + end stamp (reported practice)"),
                ("centered_destamp", "^-", "C0", "CN + de-stamp (repair)"),
                ("subcycle_complete", "d--", "C2", "every sub-window (exact arm)")):
            levels = arms[arm]["levels"]
            hs = np.array([lv["h"] for lv in levels])
            errs = np.array([lv["defect_l1"] for lv in levels])
            ax.loglog(hs, errs, marker, color=colour, label=label)
        ax.set_title(f"{name} chain")
        ax.set_xlabel("coarse exchange step h [s]")
        ax.set_ylabel("handover defect (L1, relative)")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=6.5, loc="lower right")
    fig.suptitle("Staircase saturation of the end-stamped exchange and its repair", fontsize=11)
    fig.tight_layout()
    path = OUT / "fig2_staircase_repair.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_attribution(study: dict) -> Path:
    rows = study["attribution"]["tail_sweep"]
    ratios = np.array([r["ratio"] for r in rows])
    loss = np.array([r["unattributed_share"] for r in rows])
    predicted = np.array([r["predicted_loss"] for r in rows])
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.axhspan(0.87, 0.94, color="grey", alpha=0.18,
               label="reported loss band of the coupling contract (87–94 %)")
    ax.semilogx(ratios, 100 * loss, "o-", color="C3", label="tail-only booking (measured)")
    ax.semilogx(ratios, 100 * predicted, "k:", label="$1 - 1/r$ (analytic)")
    ax.scatter([8, 16], [100 * loss[np.where(ratios == 8)][0], 100 * loss[np.where(ratios == 16)][0]],
               s=90, facecolors="none", edgecolors="C3", zorder=5)
    ax.set_xlabel("sub-cycling ratio r (fine steps per exchange interval)")
    ax.set_ylabel("exchanged quantity never booked [%]")
    ax.set_title("Interval attribution: heat carried by r-1 sub-intervals is dropped")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = OUT / "fig3_attribution_loss.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_trajectories() -> Path:
    """Deviation of the receiver from the solve_ivp oracle: broken vs repaired."""
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0))
    for ax, name, ylabel in zip(axes, ("viv", "weld", "settling"),
                                ("$v_s$ [m/s]", "$T_s$ [K]", "$v_p$ [m/s]")):
        chain = build_chain(name)
        ratio = 8
        broken = run_ladder(chain, "centered", ratio, stamp_offset=0.5 * chain.dt_fine)
        repaired = run_ladder(chain, "centered", ratio, stamp_offset=0.5 * chain.dt_fine,
                              recenter=True)
        t = np.array(broken.time_trace)
        oracle = np.array(broken.oracle_trace)
        span = max(np.ptp(oracle), 1e-300)
        ax.plot(t, (np.array(broken.state_trace) - oracle) / span, "s--", ms=3,
                color="C1", label=f"CN + end stamp (final {broken.state_error:.1e})")
        ax.plot(t, (np.array(repaired.state_trace) - oracle) / span, "^:", ms=3,
                color="C0", label=f"CN + de-stamp (final {repaired.state_error:.1e})")
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_xlabel("t [s]")
        ax.set_ylabel(f"({ylabel.split('[')[0].strip()} - oracle) / span")
        ax.set_title(f"{name} chain, exchange every {ratio} fine steps")
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5, loc="best")
    fig.suptitle("Exchange-induced receiver deviation from the oracle "
                 "(span = ptp of the oracle trajectory)", fontsize=10)
    fig.tight_layout()
    path = OUT / "fig4_chain_trajectories.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def compact_ladders(study: dict) -> dict:
    table: dict[str, dict] = {}
    for name, payload in study["chains"].items():
        table[name] = {"meta": payload["meta"], "arms": {}}
        for arm, ladder in payload["arms"].items():
            table[name]["arms"][arm] = {
                "scheme": ladder["scheme"],
                "stamp_offset": ladder["stamp_offset"],
                "recenter": ladder["recenter"],
                "tail_booking": ladder["tail_booking"],
                "subcycle": ladder["subcycle"],
                "h": [lv["h"] for lv in ladder["levels"]],
                "defect_l1": [lv["defect_l1"] for lv in ladder["levels"]],
                "max_interval_defect": [lv["max_interval_defect"] for lv in ladder["levels"]],
                "state_error": [lv["state_error"] for lv in ladder["levels"]],
                "unattributed_share": [lv["unattributed_share"] for lv in ladder["levels"]],
                "deficit_share": [lv["deficit_share"] for lv in ladder["levels"]],
                "observed_order": ladder["observed_order"],
                "fit_r2": ladder["fit_r2"],
                "saturation": ladder["saturation"],
            }
    return table


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    study = run_exchange_study(ratios=RATIOS)
    verdict = gate_study(study)

    figs = [plot_operator_order(study), plot_staircase_repair(study),
            plot_attribution(study), plot_trajectories()]

    ladders_path = OUT / "cadence_ladders.json"
    ladders_path.write_text(json.dumps(compact_ladders(study), indent=1))

    payload = {
        "node": "p223w3-n06-physics-coupling-exchange",
        "title": "Coupling time-exchange operator algebra, second order, and the "
                 "interval-complete conservation ledger",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "machine": platform.machine(),
        },
        "provenance": {
            "repo": "TensorPhysics",
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "commit": _git("rev-parse", "HEAD"),
            "worktree": str(ROOT),
        },
        "artifacts": {
            "verification": "artifacts/p223w3_n06/verification.json",
            "ladders": "artifacts/p223w3_n06/cadence_ladders.json",
            "figures": [f"artifacts/p223w3_n06/{p.name}" for p in figs],
            "doc": "docs/P223W3_N06_COUPLING_TIME_EXCHANGE.md",
            "record": "artifacts/p223w3_n06/SHA256SUMS",
        },
        "study": study,
        "gate": verdict,
        "gate_table": format_table(verdict),
    }
    verification_path = OUT / "verification.json"
    verification_path.write_text(json.dumps(payload, indent=1))

    # A record cannot carry its own hash: writing the hash changes the file.  The
    # artifact hashes of everything else live in the record, and the record's own
    # hash lives in a sibling SHA256SUMS file.
    hashes = {p.name: _sha256(p) for p in [ladders_path, *figs]}
    payload["sha256"] = hashes
    verification_path.write_text(json.dumps(payload, indent=1))
    sums = {"verification.json": _sha256(verification_path), **hashes}
    sums_path = OUT / "SHA256SUMS"
    sums_path.write_text("".join(f"{digest}  {name}\n"
                                 for name, digest in sorted(sums.items())))

    print(payload["gate_table"])
    print(json.dumps(sums, indent=1))
    print(f"verification: {verification_path}")
    return 0 if verdict["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())