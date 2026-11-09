#!/usr/bin/env python3
"""m8695 — evidence + figures for the durable criterion freeze store.

Runs six synthetic freeze→audit scenarios through the real API (no mocks) and
writes:

* ``artifacts/m8695_freeze_store_evidence.json``  — measured outcomes
* ``artifacts/m8695_freeze_store_matrix.png``     — detection matrix figure
* ``artifacts/m8695_freeze_store_digests.png``    — digest-layer figure

Synthetic criteria only (zero domain semantics); every number in the JSON is
produced by programmatic execution of
``tensorphysics.campaign.criterion_registry``.

Usage:  PYTHONPATH=src python scripts/m8695_freeze_store_evidence.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensorphysics.campaign.criterion_registry import (  # noqa: E402
    CriterionLedger,
    PreregistrationConflict,
    StoreCorruptError,
    audit_from_store,
    bundle_path,
    freeze_to_store,
    load_bundle,
)
from tensorphysics.campaign.criterion_registry.store import FrozenBundle  # noqa: E402

ART = ROOT / "artifacts"
MISSION = "m8695-demo"
GOAL = "- recall >= 0.92 on the held-out split\n- latency <= 25 ms of the p95 baseline\n"
LOOSENED = "- recall >= 0.90 on the held-out split\n- latency <= 25 ms of the p95 baseline\n"


def _forge(path: Path, mutate, recompute_bundle: bool) -> None:
    raw = json.loads(path.read_text(encoding="utf-8"))
    mutate(raw)
    if recompute_bundle:
        forged = FrozenBundle(
            mission_id=raw["mission_id"],
            frozen_by=raw["frozen_by"],
            frozen_at=raw["frozen_at"],
            criteria=raw["criteria"],
        )
        raw["bundle_sha256"] = forged.compute_sha256()
    path.write_text(json.dumps(raw, sort_keys=True, separators=(",", ":")),
                    encoding="utf-8")


def run_scenarios() -> dict:
    scenarios = []
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)

        # S1 honest round trip
        store = base / "s1"
        freeze_to_store(store, MISSION, GOAL, frozen_by="agent/m8695")
        audit = audit_from_store(store, MISSION, GOAL, ledger=CriterionLedger())
        scenarios.append(_row("S1 honest re-freeze + unchanged report", audit,
                              len(audit.results)))

        # S2 post-hoc loosening
        store = base / "s2"
        freeze_to_store(store, MISSION, GOAL, frozen_by="agent/m8695")
        audit = audit_from_store(store, MISSION, LOOSENED, ledger=CriterionLedger())
        scenarios.append(_row("S2 post-hoc threshold loosening", audit, len(audit.results)))

        # S3 hand-edited threshold (bundle digest left stale)
        store = base / "s3"
        freeze_to_store(store, MISSION, GOAL, frozen_by="agent/m8695")
        _forge(bundle_path(store, MISSION),
               lambda raw: raw["criteria"][0]["fields"]["threshold"].__setitem__("value", 0.5),
               recompute_bundle=False)
        try:
            load_bundle(store, MISSION)
            outcome, classes, failed = "loaded (BUG)", [], None
        except StoreCorruptError:
            outcome, classes, failed = "StoreCorruptError", ["store_corrupt"], True
        scenarios.append({"scenario": "S3 hand-edited level, stale bundle digest",
                          "classifications": classes, "failed_closed": failed,
                          "kpi": None, "outcome": outcome})

        # S4 hand-edited threshold + recomputed bundle digest
        store = base / "s4"
        freeze_to_store(store, MISSION, GOAL, frozen_by="agent/m8695")
        _forge(bundle_path(store, MISSION),
               lambda raw: raw["criteria"][0]["fields"]["threshold"].__setitem__("value", 0.5),
               recompute_bundle=True)
        audit = audit_from_store(store, MISSION, GOAL, ledger=CriterionLedger())
        scenarios.append(_row("S4 hand-edited level, bundle digest recomputed",
                              audit, len(audit.results)))

        # S5 never frozen
        audit = audit_from_store(base / "s5", "never-frozen", GOAL,
                                 ledger=CriterionLedger())
        scenarios.append(_row("S5 mission never pre-registered", audit, len(audit.results)))

        # S6 conflicting re-freeze attempt
        store = base / "s6"
        freeze_to_store(store, MISSION, GOAL, frozen_by="agent/m8695")
        try:
            freeze_to_store(store, MISSION, LOOSENED, frozen_by="agent/m8695")
            outcome = "overwrote (BUG)"
        except PreregistrationConflict:
            outcome = "PreregistrationConflict"
        kept = load_bundle(store, MISSION)
        scenarios.append({
            "scenario": "S6 attempt to re-freeze with looser level",
            "classifications": [], "failed_closed": True, "kpi": None,
            "outcome": outcome,
            "frozen_threshold_preserved": kept.criteria[0]["fields"]["threshold"]["value"],
        })

        # digest sizes for the second figure
        store = base / "digests"
        bundle = freeze_to_store(store, MISSION, GOAL, frozen_by="agent/m8695")
        digests = {
            "bundle_file_bytes": bundle.path.stat().st_size,
            "bundle_sha256": bundle.bundle_sha256,
            "per_criterion_sha256": [c["sha256"] for c in bundle.criteria],
            "criteria_count": len(bundle.criteria),
        }

    return {"scenarios": scenarios, "digests": digests}


def _row(name: str, audit, n: int) -> dict:
    return {
        "scenario": name,
        "classifications": [r.classification for r in audit.results],
        "failed_closed": audit.failed_closed,
        "kpi": audit.kpi.to_dict() if audit.kpi else None,
        "outcome": "audited",
        "criteria": n,
    }


def plot_matrix(evidence: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["unchanged", "loosened", "tampered", "unregistered", "store_corrupt"]
    rows = evidence["scenarios"]
    grid = [[1 if cls in r["classifications"] else 0 for cls in labels] for r in rows]

    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    ax.imshow(grid, cmap="Greys", vmin=0, vmax=1.6, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['scenario']}" for r in rows], fontsize=7)
    for i, row in enumerate(grid):
        for j, val in enumerate(row):
            ax.text(j, i, "detected" if val else "·", ha="center", va="center",
                    fontsize=7, color="white" if val else "#666")
    ax.set_title("Freeze-store audit outcomes per scenario (black = classification emitted)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_digests(evidence: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = evidence["digests"]
    labels = ["bundle file\n(bytes)", "criteria\n(count)", "per-criterion\nhash len"]
    values = [d["bundle_file_bytes"], d["criteria_count"],
              len(d["per_criterion_sha256"][0]) if d["per_criterion_sha256"] else 0]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    bars = ax.bar(labels, values, color=["#31556b", "#7fa1b3", "#c08b4b"])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value}",
                ha="center", va="bottom", fontsize=8)
    ax.set_title("On-disk freeze bundle (mission m8695-demo)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    evidence = run_scenarios()
    (ART / "m8695_freeze_store_evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plot_matrix(evidence, ART / "m8695_freeze_store_matrix.png")
    plot_digests(evidence, ART / "m8695_freeze_store_digests.png")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())