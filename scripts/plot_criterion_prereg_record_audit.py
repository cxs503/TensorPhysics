#!/usr/bin/env python
"""Plot the m8598 criterion pre-registration record audit (report figure).

Left panel: outcome of ``tests/test_criterion_prereg_doc_report.py`` before and
after the sealed mission record was restored (7 failed / 21 passed -> 28 passed);
the red/green counts are the ones measured in ``docs/m8598/PROVENANCE.md`` §3.
Right panel: every criterion measurement carried by the sealed record against its
own threshold.

Numbers are read from ``docs/m8598/outputs/acceptance_criteria.json`` -- nothing
here re-types a threshold or a measured value, so the figure cannot drift from
the record it illustrates.

Usage::

    python scripts/plot_criterion_prereg_record_audit.py [--out PNG]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs" / "m8598" / "outputs" / "acceptance_criteria.json"
OUT = ROOT / "docs" / "m8598" / "figures" / "fig_record_gate_audit.png"

#: (passed, failed) counts of the gated test file, measured per PROVENANCE.md §3.
GATE_OUTCOMES = (("pre-fix\n(stale record on main)", 21, 7),
                 ("post-fix\n(sealed record)", 28, 0))


def build(record_path: Path = RECORD, out_path: Path = OUT) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    acc = json.loads(Path(record_path).read_text(encoding="utf-8"))["acceptance"]
    a, b = acc["a_detection_24cases"], acc["b_extractor_field_f1"]

    fig, (ax_gate, ax_metric) = plt.subplots(1, 2, figsize=(11, 4.2))

    xs = range(len(GATE_OUTCOMES))
    passed = [c[1] for c in GATE_OUTCOMES]
    failed = [c[2] for c in GATE_OUTCOMES]
    ax_gate.bar(xs, passed, color="#2e7d32", label="passed")
    ax_gate.bar(xs, failed, bottom=passed, color="#c62828", label="failed")
    for i, (_, p, f) in enumerate(GATE_OUTCOMES):
        ax_gate.text(i, p + f + 0.4, f"{p} pass / {f} fail", ha="center", fontsize=9)
    ax_gate.set_xticks(list(xs), [c[0] for c in GATE_OUTCOMES], fontsize=8.5)
    ax_gate.set_ylabel("tests in tests/test_criterion_prereg_doc_report.py")
    ax_gate.set_title("Gate outcome: stale vs sealed mission record", fontsize=10)
    ax_gate.set_ylim(0, 34)
    ax_gate.legend(fontsize=8, loc="lower right")

    names = ["recall(a)", "precision(a)", "F1(b)", "held-out F1(b)", "OOD F1(b)"]
    measured = [a["recall"], a["precision"], b["f1"], b["heldout_seed_f1"],
                b["ood_undelimited_cjk_f1"]]
    thresholds = [a["thresholds"]["recall"], a["thresholds"]["precision"],
                  b["thresholds"]["f1"], b["thresholds"]["f1"], b["thresholds"]["f1"]]
    ys = range(len(names))
    ax_metric.barh(list(ys), measured, color="#1565c0", height=0.5, label="measured")
    ax_metric.plot(thresholds, list(ys), "r|", markersize=16, label="threshold")
    for i, (m, t) in enumerate(zip(measured, thresholds)):
        ax_metric.text(min(m + 0.01, 1.06), i, f"{m:.3f} (>= {t:.2f})",
                       va="center", fontsize=8)
    ax_metric.set_yticks(list(ys), names, fontsize=8.5)
    ax_metric.invert_yaxis()
    ax_metric.set_xlim(0, 1.15)
    ax_metric.set_xlabel("metric value (1.0 = perfect)")
    ax_metric.set_title("Criterion measurements carried by the record", fontsize=10)
    ax_metric.legend(fontsize=8, loc="lower right")

    fig.suptitle("m8598 criterion pre-registration gate: record provenance audit",
                 fontsize=11)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--record", default=str(RECORD))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    path = build(Path(args.record), Path(args.out))
    label = path.relative_to(ROOT) if ROOT in path.parents else path
    print(f"wrote {label} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())