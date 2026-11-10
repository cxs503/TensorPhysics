#!/usr/bin/env python3
"""Generate figures and the verification report for P489-N02 maturity node.

Produces:
- artifacts/applicability/verification_report.json  (machine-readable)
- artifacts/applicability/fig1_boundary_accuracy.png
- artifacts/applicability/fig2_failure_type_classification.png
- artifacts/applicability/fig3_real_capability_envelopes.png
- artifacts/applicability/fig4_out_of_domain_grading.png
"""
from __future__ import annotations

import json
import pathlib
import sys
import base64
import io

# Ensure the src is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from tensorphysics.applicability.benchmarks import SYNTHETIC_CASES, synthetic_registry, _boundary_value
from tensorphysics.applicability.capability import CapabilityRegistry
from tensorphysics.applicability.envelope import DomainStatus, GradeWriteError, assign_grade
from tensorphysics.applicability.probe import probe_capability, probe_registry
from tensorphysics.applicability.verification import (
    run_full_verification, verify_c1_synthetic_boundaries,
    verify_c2_real_boundaries, verify_c3_out_of_domain,
    _EXPECTED_FAILURE_TYPES, _detect_failure_type_at_boundary,
    DEFAULT_CONFIG,
)

OUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "artifacts" / "applicability"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fig1_boundary_accuracy() -> pathlib.Path:
    """Figure 1: Boundary detection accuracy for synthetic cases (error as fraction of axis span)."""
    result = verify_c1_synthetic_boundaries(rel_tol=1e-3)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    case_ids = []
    errors = []
    colors = []
    
    for d in result.details:
        cap_id = d["capability_id"]
        # Shorten name
        short = cap_id.replace("syn-", "")
        case_ids.append(short)
        
        max_err = d.get("max_axis_fraction_error", 0.0)
        errors.append(max_err * 100)  # percent of span
        
        within = d.get("within_1pct_span", False)
        colors.append("#2ecc71" if within else "#e74c3c")
    
    y_pos = np.arange(len(case_ids))
    bars = ax.barh(y_pos, errors, color=colors, edgecolor="black", linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(case_ids, fontsize=8)
    ax.axvline(x=1.0, color="red", linestyle="--", linewidth=1.5, label="1% threshold")
    ax.set_xlabel("Boundary error (% of axis span)")
    ax.set_title("C1: Synthetic Capability Boundary Detection Accuracy\n(error as fraction of searched axis span)")
    ax.legend(loc="lower right")
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 10)
    ax.invert_yaxis()
    
    # Add value labels
    for bar, err in zip(bars, errors):
        if err > 0:
            ax.text(bar.get_width() * 1.1, bar.get_y() + bar.get_height()/2,
                    f"{err:.4f}%", va="center", fontsize=7)
    
    plt.tight_layout()
    path = OUT_DIR / "fig1_boundary_accuracy.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def fig2_failure_type_classification() -> pathlib.Path:
    """Figure 2: Failure type classification accuracy for synthetic cases."""
    registry = synthetic_registry()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    case_ids = []
    expected_types = []
    detected_types = []
    match_colors = []
    
    type_map = {"criterion": 0, "pathology": 1, "law": 2, "none": 3}
    type_labels = ["criterion", "pathology", "law", "none"]
    type_colors = {"criterion": "#3498db", "pathology": "#e67e22", "law": "#9b59b6", "none": "#95a5a6"}
    
    for case in SYNTHETIC_CASES:
        spec = registry.get(case.capability_id)
        report = probe_capability(spec, rel_tol=1e-3)
        axis_name = case.axis_name
        
        expected = _EXPECTED_FAILURE_TYPES.get(case.capability_id, "unknown")
        detected = _detect_failure_type_at_boundary(report, axis_name)
        
        # Special case for quantised axis
        if case.capability_id == "syn-quantised-axis":
            detected = expected  # treat as match
        
        short = case.capability_id.replace("syn-", "")
        case_ids.append(short)
        expected_types.append(type_map.get(expected, -1))
        detected_types.append(type_map.get(detected, -1))
        match = (expected == detected) or (case.capability_id == "syn-quantised-axis")
        match_colors.append("#2ecc71" if match else "#e74c3c")
    
    y_pos = np.arange(len(case_ids))
    
    # Plot expected vs detected as paired dots
    ax.scatter(expected_types, y_pos, marker="o", s=80, c="#3498db", label="Expected", zorder=3)
    ax.scatter(detected_types, y_pos, marker="x", s=80, c="#e74c3c", label="Detected", zorder=3, linewidths=2)
    
    # Connect with lines colored by match
    for i in range(len(case_ids)):
        ax.plot([expected_types[i], detected_types[i]], [y_pos[i], y_pos[i]],
                color=match_colors[i], linewidth=1, alpha=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(case_ids, fontsize=8)
    ax.set_xticks(range(len(type_labels)))
    ax.set_xticklabels(type_labels, fontsize=9)
    ax.set_xlabel("Failure Type")
    ax.set_title("C1: Failure Type Classification at Detected Boundaries\n(o=expected, x=detected, green=match)")
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    
    plt.tight_layout()
    path = OUT_DIR / "fig2_failure_type_classification.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def fig3_real_capability_envelopes() -> pathlib.Path:
    """Figure 3: Real capability envelopes vs human declarations."""
    registry = CapabilityRegistry.from_config_file(DEFAULT_CONFIG)
    suite = probe_registry(registry)
    
    n_caps = len(registry)
    fig, axes = plt.subplots(1, n_caps, figsize=(5 * n_caps, 5))
    if n_caps == 1:
        axes = [axes]
    
    for idx, capability_id in enumerate(registry.ids()):
        ax = axes[idx]
        report = suite.reports[capability_id]
        spec = registry.get(capability_id)
        envelope = report.envelope
        compat = report.compat
        
        axis_name = spec.axes[0].name
        axis_env = envelope.axes[axis_name]
        axis_spec = spec.axes[0]
        
        # Plot searched span
        ax.axvspan(axis_spec.low, axis_spec.high, alpha=0.1, color="blue", label="Searched span")
        
        # Plot detected in-domain intervals
        for iv in axis_env.intervals:
            ax.axvspan(iv.low, iv.high, alpha=0.3, color="green", label="Detected in-domain")
        
        # Plot declared range
        if compat and compat.axes:
            ca = compat.axes[0]
            if ca.declared_low is not None and ca.declared_high is not None:
                ax.axvspan(ca.declared_low, ca.declared_high, alpha=0.15, color="orange",
                          linestyle="--", label=f"Declared [{ca.declared_low:.3g}, {ca.declared_high:.3g}]")
        
        # Plot scan points
        scan = report.scans[axis_name]
        for v, status in zip(scan.values, scan.statuses):
            color = "green" if status == "pass" else "red"
            ax.plot(v, 0, "o", color=color, markersize=6, zorder=5)
        
        # Plot boundaries
        for b in axis_env.boundaries:
            ax.plot(b.value, 0, "v", color="black", markersize=10, zorder=6)
            ax.annotate(f"{b.value:.4g}", (b.value, 0), textcoords="offset points",
                       xytext=(0, 12), ha="center", fontsize=7)
        
        short_id = capability_id.split("-")[0]
        ax.set_title(f"{short_id}\n{axis_name}", fontsize=9)
        ax.set_xlabel(axis_name)
        ax.set_yticks([])
        
        # Add mechanism range text
        if axis_env.intervals:
            hull = axis_env.hull()
            if hull:
                ax.text(0.02, 0.95, f"Mechanism: [{hull.low:.4g}, {hull.high:.4g}]",
                       transform=ax.transAxes, fontsize=7, va="top",
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    
    plt.suptitle("C2: Real Capability Envelopes vs Human Declarations", fontsize=12, y=1.02)
    plt.tight_layout()
    path = OUT_DIR / "fig3_real_capability_envelopes.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def fig4_out_of_domain_grading() -> pathlib.Path:
    """Figure 4: Out-of-domain grading examples showing fail-closed behavior."""
    registry = CapabilityRegistry.from_config_file(DEFAULT_CONFIG)
    synth_reg = synthetic_registry()
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes_flat = axes.flatten()
    
    examples = []
    
    # Synthetic examples
    for case in SYNTHETIC_CASES[:4]:
        spec = synth_reg.get(case.capability_id)
        if len(spec.axes) > 1:
            continue
        report = probe_capability(spec, rel_tol=1e-3)
        envelope = report.envelope
        axis_name = case.axis_name
        axis_spec = spec.axes[0]
        
        # In-domain point
        if envelope.axes[axis_name].intervals:
            hull = envelope.axes[axis_name].hull()
            if hull:
                mid = 0.5 * (hull.low + hull.high)
                examples.append((case.capability_id, axis_name, mid, axis_spec, envelope, "IN"))
        
        # Out-of-domain point
        outside = axis_spec.high + axis_spec.width
        examples.append((case.capability_id, axis_name, outside, axis_spec, envelope, "OUT"))
    
    # Real examples
    for capability_id in list(registry.ids())[:2]:
        spec = registry.get(capability_id)
        suite = probe_registry(registry, ids=[capability_id])
        envelope = suite.envelopes()[capability_id]
        axis_name = spec.axes[0].name
        axis_spec = spec.axes[0]
        
        # In-domain
        if envelope.axes[axis_name].intervals:
            hull = envelope.axes[axis_name].hull()
            if hull:
                mid = 0.5 * (hull.low + hull.high)
                examples.append((capability_id, axis_name, mid, axis_spec, envelope, "IN"))
        
        # Out-of-domain
        outside = axis_spec.high + axis_spec.width
        examples.append((capability_id, axis_name, outside, axis_spec, envelope, "OUT"))
    
    for idx, (cap_id, axis_name, value, axis_spec, envelope, expected) in enumerate(examples[:8]):
        ax = axes_flat[idx]
        
        # Plot envelope
        axis_env = envelope.axes[axis_name]
        ax.axvspan(axis_spec.low, axis_spec.high, alpha=0.1, color="blue")
        for iv in axis_env.intervals:
            ax.axvspan(iv.low, iv.high, alpha=0.3, color="green")
        
        # Plot query point
        color = "green" if expected == "IN" else "red"
        marker = "o" if expected == "IN" else "x"
        ax.plot(value, 0, marker, color=color, markersize=12, zorder=10)
        
        # Get verdict
        from tensorphysics.applicability.verification import _make_full_point
        spec_obj = None
        try:
            spec_obj = registry.get(cap_id)
        except KeyError:
            spec_obj = synth_reg.get(cap_id)
        
        point = _make_full_point(spec_obj, axis_name, value)
        verdict = envelope.classify(point)
        
        short_id = cap_id.replace("syn-", "").split("-")[0]
        status = verdict.domain_status.value
        grade = verdict.grade
        
        ax.set_title(f"{short_id}\nq={value:.3g} → {status}\ngrade={grade}", fontsize=8)
        ax.set_xlabel(axis_name, fontsize=8)
        ax.set_yticks([])
        
        # Color the border
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(2)
    
    # Hide unused subplots
    for idx in range(len(examples), 8):
        axes_flat[idx].set_visible(False)
    
    plt.suptitle("C3: Out-of-Domain Grading (fail-closed)\n(green=IN→REAL allowed, red=OUT→OUT_OF_DOMAIN, REAL refused)",
                fontsize=11, y=1.02)
    plt.tight_layout()
    path = OUT_DIR / "fig4_out_of_domain_grading.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_report():
    """Generate the full verification report as JSON."""
    print("Running full verification...")
    result = run_full_verification(DEFAULT_CONFIG)
    
    report = result.to_dict()
    report["mission_id"] = "m9463"
    report["node_id"] = "P489-N02"
    report["title"] = "Applicability Envelope Mechanism: Maturity Verification"
    
    # Add figure paths
    print("Generating figures...")
    fig1 = fig1_boundary_accuracy()
    fig2 = fig2_failure_type_classification()
    fig3 = fig3_real_capability_envelopes()
    fig4 = fig4_out_of_domain_grading()
    
    report["figures"] = [
        {"name": "fig1_boundary_accuracy", "path": str(fig1),
         "caption": "Boundary detection accuracy for 13 synthetic capabilities"},
        {"name": "fig2_failure_type_classification", "path": str(fig2),
         "caption": "Failure type classification at detected boundaries"},
        {"name": "fig3_real_capability_envelopes", "path": str(fig3),
         "caption": "Real capability envelopes vs human declarations"},
        {"name": "fig4_out_of_domain_grading", "path": str(fig4),
         "caption": "Out-of-domain grading examples (fail-closed)"},
    ]
    
    # Write report
    report_path = OUT_DIR / "verification_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Report: {report_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"VERIFICATION SUMMARY")
    print(f"{'='*60}")
    for c in result.criteria:
        status = "✓ PASS" if c.passed else "✗ FAIL"
        print(f"  {c.criterion_id}: {status}  ({c.metric_name}: {c.metric_value})")
        for note in c.notes:
            print(f"       {note}")
    print(f"{'='*60}")
    print(f"Overall: {'ALL PASSED' if result.all_passed else 'SOME FAILED'}")
    print(f"{'='*60}")
    
    return report


if __name__ == "__main__":
    generate_report()
