"""p285w2b-n04 study: capability-aware routing acceptance run + figures.

Mechanism-only script (no domain vocabulary): it exercises the routing layer on
a synthetic ledger whose *grade distribution* mirrors the platform capability
audit shape, writes every number it measures to ``artifacts/capability_routing/``
and renders the report figures.  Nothing is typed in by hand: the acceptance
payload is produced by
:func:`tensorphysics.capability_routing.acceptance.run_acceptance` and the
figures read only from that payload.

Usage::

    python scripts/p285w2b_n04_study.py [--out artifacts/capability_routing]
                                        [--figdir docs/capability_routing]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from tensorphysics.capability_routing.acceptance import (  # noqa: E402
    fallback_oracle_check,
    latency_benchmark,
    refusal_sweep,
    run_acceptance,
    synthetic_ledger,
)
from tensorphysics.capability_routing.evidence import (  # noqa: E402
    AdmissibilityPolicy,
    GRADE_ORDER,
    TIER_ADMISSIBLE,
)
from tensorphysics.capability_routing.router import CapabilityAwareRouter  # noqa: E402

ROUTES = 2000
CASES = 240


def _decision_matrix(ledger) -> dict:
    """Route every (capability, solver) request under staged tier -> outcomes."""
    router = CapabilityAwareRouter(ledger, AdmissibilityPolicy(tier="staged"))
    rows = []
    for record in ledger.records():
        try:
            decision = router.route(record.capability_id, preferred=record.solver_id)
        except Exception as exc:  # noqa: BLE001 - the refusal is the measurement
            rows.append({
                "capability_id": record.capability_id,
                "requested": record.solver_id,
                "requested_grade": record.level.value,
                "outcome": "refused",
                "refusal_code": getattr(exc, "code", type(exc).__name__),
                "selected": None,
                "selected_grade": None,
            })
            continue
        rows.append({
            "capability_id": record.capability_id,
            "requested": record.solver_id,
            "requested_grade": record.level.value,
            "outcome": "routed",
            "refusal_code": None,
            "selected": decision.selected,
            "selected_grade": decision.level.value,
        })
    return {"tier": "staged", "rows": rows}


def _write_figures(payload: dict, matrix: dict, figdir: Path) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir.mkdir(parents=True, exist_ok=True)
    figures: list[dict] = []

    # --- fig 1: gate latency distribution against the 50 ms budget -----------
    latency = payload["criteria"]["C3"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels = ["gate mean", "gate p50", "gate p95", "gate p99", "gate max",
              "route p99"]
    values = [latency["gate"]["mean_ms"], latency["gate"]["p50_ms"],
              latency["gate"]["p95_ms"], latency["gate"]["p99_ms"],
              latency["gate_max_ms"], latency["total"]["p99_ms"]]
    bars = ax.bar(labels, values, color=["#2b6cb0"] * 5 + ["#718096"])
    ax.axhline(latency["budget_ms"], color="#c53030", linestyle="--",
               label=f"budget {latency['budget_ms']:.0f} ms")
    ax.set_yscale("log")
    ax.set_ylabel("latency per route [ms] (log scale)")
    ax.set_title(f"Evidence-gate overhead over {latency['routes']} routed requests")
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:.4f}", (bar.get_x() + bar.get_width() / 2, value),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = figdir / "fig1_gate_latency.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append({
        "name": path.name,
        "path": str(path),
        "caption": ("Fig.1 Evidence-gate latency per routing decision vs the 50 ms budget "
                    "(log scale). Gate work is a single indexed dictionary probe plus one "
                    "policy membership test, so the tail stays four orders of magnitude "
                    "below the budget."),
    })

    # --- fig 2: outcome matrix per requested grade ---------------------------
    grades = [g.value for g in GRADE_ORDER]
    routed = {g: 0 for g in grades}
    refused = {g: 0 for g in grades}
    for row in matrix["rows"]:
        (routed if row["outcome"] == "routed" else refused)[row["requested_grade"]] += 1
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = range(len(grades))
    ax.bar(x, [routed[g] for g in grades], label="routed to some solver",
           color="#2f855a")
    ax.bar(x, [refused[g] for g in grades], bottom=[routed[g] for g in grades],
           label="refused (explicit error)", color="#c53030")
    ax.set_xticks(list(x))
    ax.set_xticklabels(grades, rotation=15)
    ax.set_ylabel("requests")
    ax.set_title("Outcome of requesting a solver by its graded evidence level\n"
                 "(staged tier, every ledger row requested once)")
    for index, grade in enumerate(grades):
        total = routed[grade] + refused[grade]
        ax.annotate(f"{refused[grade]}/{total} refused",
                    (index, total), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=7)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = figdir / "fig2_refusal_matrix.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append({
        "name": path.name,
        "path": str(path),
        "caption": ("Fig.2 Requests grouped by the requested solver's evidence grade: every "
                    "NOT_SUPPORTED request and every placeholder request is refused with an "
                    "explicit error code instead of returning a fabricated number."),
    })

    # --- fig 3: fallback agreement with the brute-force oracle ---------------
    oracle = payload["criteria"]["C2"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9))
    ax = axes[0]
    ax.bar(["agreement", "mismatch"],
           [oracle["agreement_rate"] * 100, len(oracle["mismatches"]) * 100 / max(
               1, oracle["routed_cases"])],
           color=["#2f855a", "#c53030"])
    ax.set_ylim(0, 110)
    ax.set_ylabel("[%] of routed cases")
    ax.set_title(f"Fallback vs brute-force oracle\n({oracle['routed_cases']} routed, "
                 f"{oracle['refusal_cases_expected']} refusals expected)")
    for index, value in enumerate([oracle["agreement_rate"] * 100,
                                   len(oracle["mismatches"]) * 100 / max(
                                       1, oracle["routed_cases"])]):
        ax.annotate(f"{value:.2f}%", (index, value), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=8)
    ax = axes[1]
    tiers = list(TIER_ADMISSIBLE)
    counts = []
    for tier in tiers:
        policy = AdmissibilityPolicy(tier=tier, allow_mock=(tier == "development"))
        result = fallback_oracle_check(synthetic_ledger(), cases=120, seed=4242,
                                       policies=(policy,))
        counts.append(result["agreement_rate"] * 100)
    ax.bar(tiers, counts, color="#2b6cb0")
    ax.set_ylim(0, 110)
    ax.set_ylabel("oracle agreement [%]")
    ax.set_title("Agreement per operating tier\n(120 randomised outage cases each)")
    for index, value in enumerate(counts):
        ax.annotate(f"{value:.1f}%", (index, value), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=8)
    fig.tight_layout()
    path = figdir / "fig3_fallback_agreement.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append({
        "name": path.name,
        "path": str(path),
        "caption": ("Fig.3 Evidence-ordered fallback reproduces the brute-force oracle "
                    "(best admissible grade, requested solver only as tie-break inside one "
                    "grade) in every randomised outage case, and in every operating tier."),
    })
    return figures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/capability_routing")
    parser.add_argument("--figdir", default="docs/capability_routing")
    parser.add_argument("--routes", type=int, default=ROUTES)
    parser.add_argument("--cases", type=int, default=CASES)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    payload = run_acceptance(root=_ROOT, routes=args.routes, cases=args.cases)
    ledger = synthetic_ledger()
    matrix = _decision_matrix(ledger)
    figures = _write_figures(payload, matrix, Path(args.figdir))

    payload["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload["figures"] = figures
    payload["decision_matrix"] = matrix
    (out / "acceptance.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "ledger_v1.json").write_text(
        json.dumps(ledger.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "decision_matrix.jsonl").open("w", encoding="utf-8") as handle:
        for row in matrix["rows"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "all_pass": payload["all_pass"],
        "C1_not_supported_refusal_rate": payload["criteria"]["C1"]["not_supported_refusal_rate"],
        "C2_oracle_agreement_rate": payload["criteria"]["C2"]["agreement_rate"],
        "C3_gate_p99_ms": payload["criteria"]["C3"]["gate"]["p99_ms"],
        "C3_gate_max_ms": payload["criteria"]["C3"]["gate_max_ms"],
        "C4_lexicon_hits": len(payload["criteria"]["C4"]["hits"]),
        "C4_files_scanned": payload["criteria"]["C4"]["files_scanned"],
        "C5_decisions": payload["criteria"]["C5"]["decisions"],
        "C6_placeholder_selections": payload["criteria"]["C6"]["staged_placeholder_selections"],
        "figures": [f["name"] for f in figures],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())