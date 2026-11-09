"""Figure generation for the campaign gate (matplotlib, optional dependency).

One figure, two panels, entirely data-driven from the report dict:

* left: per-node stacked bars of criterion status (PASS / FAIL / WITHHELD);
* right: campaign-level evidence coverage (artifact families parsed, pytest
  suites executed green, contract checks passed, campaign criteria passed).

No hand-typed numbers: every bar is a count taken from the report.  When
matplotlib is unavailable the gate records ``figure_png=None`` and states the
reason in the markdown instead of silently shipping no figure.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

STATUS_COLORS = {"PASS": "#2e7d32", "FAIL": "#c62828", "WITHHELD": "#9e9e9e",
                 "NOT_SUPPORTED": "#616161"}


def write_gate_figure(report: Mapping[str, Any], path: Path) -> tuple[Path | None, str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment without matplotlib
        return None, f"matplotlib unavailable: {type(exc).__name__}: {exc}"

    nodes: Sequence[Mapping[str, Any]] = report.get("nodes", [])
    labels = [n["node_key"].split("-")[-1] for n in nodes]
    statuses = ("PASS", "FAIL", "WITHHELD")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.05, 1.0]})
    left = [0] * len(nodes)
    for status in statuses:
        counts = [sum(1 for c in n.get("criteria", []) if c["status"] == status) for n in nodes]
        ax0.barh(labels, counts, left=left, color=STATUS_COLORS[status], label=status,
                 edgecolor="white", linewidth=0.6)
        left = [a + b for a, b in zip(left, counts)]
    ax0.set_xlabel("declared node criteria (count)")
    ax0.set_title("P84-W2C per-node acceptance criteria\n(N01-N10, gate P84W2C-N11)")
    ax0.legend(loc="lower right", fontsize=8, frameon=False)
    ax0.invert_yaxis()
    ax0.grid(axis="x", alpha=0.25)

    contracts = report.get("contracts", [])
    criteria = report.get("criteria", [])
    def frac(items: Sequence[Mapping[str, Any]]) -> float:
        if not items:
            return 0.0
        return sum(1 for item in items if item.get("status") == "PASS") / len(items)

    families = []
    for node in nodes:
        jsons = [a for a in node.get("artifacts", []) if a.get("kind") == "json"]
        families.append(1.0 if jsons and all(a.get("parseable") for a in jsons) else 0.0)
    pytest_ok = [1.0 if n.get("pytest", {}).get("status") == "PASS" else 0.0 for n in nodes]
    metrics = {
        "artifact family\nparseable": (sum(families) / len(families)) if families else 0.0,
        "pytest suite\nexecuted green": (sum(pytest_ok) / len(pytest_ok)) if pytest_ok else 0.0,
        "cross-domain\ncontracts": frac(contracts),
        "campaign criteria\nC1-C4": frac(criteria),
    }
    names = list(metrics)
    values = [metrics[k] for k in names]
    bars = ax1.bar(names, [v * 100 for v in values], color="#1565c0", width=0.55)
    for bar, value in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f"{value * 100:.1f}%", ha="center", va="bottom", fontsize=9)
    ax1.set_ylim(0, 110)
    ax1.set_ylabel("coverage [%]")
    ax1.set_title(f"Evidence coverage — overall {report.get('overall_status', '?')}")
    ax1.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path, "ok"