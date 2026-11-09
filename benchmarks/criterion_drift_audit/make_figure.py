"""Figure for the drift-audit artifact report (generated, zero hand-copying).

Left panel: headline metrics for the frozen v1 extractor vs the hardened one, read
straight from the committed artifacts.  Right panel: per-field precision/recall
delta, i.e. *where* the hardening gained — the interesting part, because the
aggregate F1 alone hides which structural field was broken.

Usage::

    python benchmarks/criterion_drift_audit/make_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "criterion_drift_audit"
OUT = ROOT / "docs" / "figures" / "criterion_drift_audit_delta.png"

FIELDS = ("threshold", "unit", "comparator", "denominator", "exclusion", "reference_path")


def main() -> int:
    v2 = json.loads((ART / "bench_v2_hardened.json").read_text())
    v1 = json.loads((ART / "bench_v1_legacy.json").read_text())

    labels = ["extraction\nprecision", "extraction\nrecall", "extraction\nF1",
              "held-out\nF1", "drift\ndetection F1"]
    legacy = [v1["extraction"]["precision"], v1["extraction"]["recall"],
              v1["extraction"]["f1"], v1["held_out"]["f1"],
              v1["detection"]["f1"]]
    hardened = [v2["extraction"]["precision"], v2["extraction"]["recall"],
                v2["extraction"]["f1"], v2["held_out"]["f1"],
                v2["detection"]["f1"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    x = np.arange(len(labels))
    w = 0.38
    b1 = ax1.bar(x - w / 2, legacy, w, label="v1 legacy (pre-hardening)", color="#b2432f")
    b2 = ax1.bar(x + w / 2, hardened, w, label="v2 hardened (shipped)", color="#2f6fb2")
    for bars in (b1, b2):
        ax1.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax1.axhline(0.95, ls="--", lw=1, color="grey")
    ax1.text(len(labels) - 0.5, 0.957, "gate 0.95", ha="right", fontsize=8, color="grey")
    ax1.set_xticks(x, labels, fontsize=8.5)
    ax1.set_ylim(0, 1.12)
    ax1.set_ylabel("metric value")
    ax1.set_title("Criterion extraction/drift: legacy vs hardened\n"
                  f"same corpus sha256={v2['corpus_sha256'][:16]}…", fontsize=10)
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(axis="y", alpha=0.25)

    fx = np.arange(len(FIELDS))
    fw = 0.2
    for i, (metric, col) in enumerate((("precision", "#2f6fb2"), ("recall", "#7fb2e5"))):
        lv = [v1["extraction"]["per_field"][f][metric] for f in FIELDS]
        hv = [v2["extraction"]["per_field"][f][metric] for f in FIELDS]
        ax2.bar(fx + (i - 1.5) * fw, lv, fw, color=col, alpha=0.45,
                label=f"v1 {metric}", hatch="//")
        ax2.bar(fx + (i + 0.5) * fw, hv, fw, color=col, label=f"v2 {metric}")
    ax2.set_xticks(fx, [f.replace("_", "\n") for f in FIELDS], fontsize=8.5)
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("field-level score")
    ax2.set_title("Where the hardening gained: per-field P/R\n"
                  "(v1 lacks dedicated parsers for the last three fields)", fontsize=10)
    ax2.legend(fontsize=7.5, ncol=2, loc="lower left")
    ax2.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"aggregate F1 {v1['extraction']['f1']:.4f} -> {v2['extraction']['f1']:.4f}; "
          f"held-out {v1['held_out']['f1']:.4f} -> {v2['held_out']['f1']:.4f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())