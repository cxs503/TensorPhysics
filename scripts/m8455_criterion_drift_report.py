#!/usr/bin/env python3
"""Render figures + machine-readable summary for the criterion drift-audit node.

Inputs (produced by ``python -m tensorphysics.campaign.criterion_registry.bench``):
  artifacts/criterion_drift_audit/bench_v1_legacy.json      (legacy extractor)
  artifacts/criterion_drift_audit/bench_v2_hardened.json    (hardened extractor)

Outputs:
  artifacts/criterion_drift_audit/figures/fig1_field_f1.png
  artifacts/criterion_drift_audit/figures/fig2_drift_confusion.png
  artifacts/criterion_drift_audit/report_metrics.json
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "criterion_drift_audit"
FIGS = OUT / "figures"
FIELDS = ("threshold", "unit", "denominator", "exclusion", "reference_path", "comparator")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def field_f1(record: dict) -> list[float]:
    out = []
    for name in FIELDS:
        m = record["extraction"]["per_field"][name]
        p, r = m["precision"], m["recall"]
        out.append(2 * p * r / (p + r) if (p + r) else 0.0)
    return out


def fig_field_f1(legacy: dict, hardened: dict) -> Path:
    labels = list(FIELDS) + ["micro-F1\n(48 cases)"]
    legacy_vals = field_f1(legacy) + [legacy["extraction"]["f1"]]
    hardened_vals = field_f1(hardened) + [hardened["extraction"]["f1"]]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.bar(x - w / 2, legacy_vals, w, label="legacy extractor", color="#b0413e")
    ax.bar(x + w / 2, hardened_vals, w, label="hardened extractor", color="#3b6ea5")
    ax.axhline(0.95, ls="--", lw=1.2, color="#444", label="gate 0.95")
    ax.set_xticks(x, labels, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("field-level F1")
    ax.set_title(
        "Extraction F1 per structural field (same 48-case synthetic corpus,\n"
        f"sha256 {hardened['corpus_sha256'][:12]}…): "
        f"micro-F1 {legacy['extraction']['f1']:.3f} → {hardened['extraction']['f1']:.3f};"
        " held-out 20-case micro-F1 "
        f"{hardened['held_out']['f1']:.3f} (legacy {legacy['held_out']['f1']:.3f})"
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = FIGS / "fig1_field_f1.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_confusion(hardened: dict) -> Path:
    conf = hardened["detection"]["confusion"]
    labels = [k for k in ("unchanged", "tightened", "loosened", "scope_changed",
                          "undefined", "unregistered") if k in conf]
    mat = np.array([[conf[e].get(g, 0) for g in labels] for e in labels], dtype=float)
    fig, ax = plt.subplots(figsize=(5.9, 5.1))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=max(3.0, mat.max()))
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("expected")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if mat[i, j]:
                ax.text(j, i, int(mat[i, j]), ha="center", va="center",
                        color="white" if mat[i, j] > mat.max() * 0.6 else "black")
    ax.set_title(
        "Drift classification, 24-case synthetic set\n"
        f"binary precision {hardened['detection']['precision']:.3f}, "
        f"recall {hardened['detection']['recall']:.3f}, "
        f"exact {hardened['detection']['exact_accuracy']:.3f}"
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cases")
    fig.tight_layout()
    path = FIGS / "fig2_drift_confusion.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    legacy_p = OUT / "bench_v1_legacy.json"
    hardened_p = OUT / "bench_v2_hardened.json"
    legacy = json.loads(legacy_p.read_text())
    hardened = json.loads(hardened_p.read_text())
    assert legacy["corpus_sha256"] == hardened["corpus_sha256"]

    fig1 = fig_field_f1(legacy, hardened)
    fig2 = fig_confusion(hardened)

    metrics = {
        "corpus_sha256": hardened["corpus_sha256"],
        "corpus": {
            "extraction_cases": hardened["extraction"]["cases"],
            "held_out_cases": hardened["held_out"]["cases"],
            "drift_cases": hardened["detection"]["cases"],
            "drift_class_counts": {
                k: sum(1 for c in __import__(
                    "tensorphysics.campaign.criterion_registry.bench",
                    fromlist=["DRIFT_CASES"]).DRIFT_CASES if c.expected == k)
                for k in ("unchanged", "tightened", "loosened", "scope_changed",
                          "undefined", "unregistered")
            },
        },
        "hardened": {
            "extraction_precision": hardened["extraction"]["precision"],
            "extraction_recall": hardened["extraction"]["recall"],
            "extraction_f1": hardened["extraction"]["f1"],
            "detection_precision": hardened["detection"]["precision"],
            "detection_recall": hardened["detection"]["recall"],
            "detection_exact_accuracy": hardened["detection"]["exact_accuracy"],
            "held_out_f1": hardened["held_out"]["f1"],
            "held_out_precision": hardened["held_out"]["precision"],
            "held_out_recall": hardened["held_out"]["recall"],
            "held_out_failures": hardened["held_out"]["failures"],
            "gates": hardened["gates"],
            "all_gates_pass": hardened["all_gates_pass"],
        },
        "legacy_baseline": {
            "extraction_f1": legacy["extraction"]["f1"],
            "extraction_precision": legacy["extraction"]["precision"],
            "extraction_recall": legacy["extraction"]["recall"],
            "detection_precision": legacy["detection"]["precision"],
            "detection_recall": legacy["detection"]["recall"],
            "detection_exact_accuracy": legacy["detection"]["exact_accuracy"],
            "held_out_f1": legacy["held_out"]["f1"],
            "gates": legacy["gates"],
            "all_gates_pass": legacy["all_gates_pass"],
        },
        "deltas": {
            "extraction_f1": round(hardened["extraction"]["f1"]
                                   - legacy["extraction"]["f1"], 4),
            "detection_recall": round(hardened["detection"]["recall"]
                                      - legacy["detection"]["recall"], 4),
            "held_out_f1": round(hardened["held_out"]["f1"]
                                 - legacy["held_out"]["f1"], 4),
        },
        "artifacts": {
            p.name: _sha256(p)
            for p in (legacy_p, hardened_p, fig1, fig2)
        },
        "fail_closed": hardened["fail_closed"],
    }
    out = OUT / "report_metrics.json"
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"\nfigures: {fig1.name}, {fig2.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())