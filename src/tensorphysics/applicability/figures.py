"""Figures for the applicability-envelope report (matplotlib only, no hand data).

Two figures are produced:

``fig1_envelopes.png``
    one panel per capability: the probed scalar along its axis (log or linear in
    the axis scale), the in-domain samples filled, out-of-domain samples hollow,
    the refined boundary as a vertical line with its confidence, and the
    human-written declared interval as a shaded band.

``fig2_domain_handling.png``
    the fail-closed decision surface: for every capability, the probe samples in
    the (axis, measured value) plane with the acceptance window drawn, so the
    boundary is visually traceable to the criterion.

Every number comes from the probe artifact, never from a literal in this file.
"""
from __future__ import annotations

import pathlib

__all__ = ["make_figures"]


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def make_figures(suite, registry, outdir: pathlib.Path) -> list[pathlib.Path]:
    plt = _plt()
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths: list[pathlib.Path] = []

    capability_ids = list(registry.ids())
    columns = 3 if len(capability_ids) >= 3 else max(1, len(capability_ids))
    rows = (len(capability_ids) + columns - 1) // columns

    # ---- figure 1: envelopes -------------------------------------------
    fig, axes_grid = plt.subplots(rows, columns, figsize=(4.6 * columns, 3.4 * rows),
                                 squeeze=False)
    for index, capability_id in enumerate(capability_ids):
        spec = registry.get(capability_id)
        report = suite.reports[capability_id]
        ax = axes_grid[index // columns][index % columns]
        axis_name = spec.axes[0].name
        scan = report.scans[axis_name]
        axis_spec = spec.axes[0]
        measured = [v if v is not None else float("nan") for v in scan.measured()]
        statuses = scan.statuses
        inside_x = [x for x, s in zip(scan.values, statuses) if s == "pass"]
        inside_y = [y for y, s in zip(measured, statuses) if s == "pass"]
        outside_x = [x for x, s in zip(scan.values, statuses) if s != "pass"]
        outside_y = [y for y, s in zip(measured, statuses) if s != "pass"]
        if axis_spec.scale == "log":
            ax.set_xscale("log")
        ax.plot(scan.values, measured, "-", color="0.6", lw=0.9, zorder=1)
        ax.plot(inside_x, inside_y, "o", ms=5, color="#1f77b4", label="in domain", zorder=3)
        ax.plot(outside_x, outside_y, "x", ms=6, color="#d62728", label="out of domain", zorder=3)
        for boundary in report.envelope.axes[axis_name].boundaries:
            ax.axvline(boundary.value, color="#2ca02c", ls="--", lw=1.2, zorder=2)
            ax.annotate(f"{boundary.value:.4g}\n({boundary.confidence_label})",
                        xy=(boundary.value, ax.get_ylim()[1] if ax.get_ylim()[1] else 1.0),
                        xytext=(3, -12), textcoords="offset points", fontsize=7,
                        color="#2ca02c")
        declared = spec.declared.get(axis_name)
        if declared is not None and declared.low is not None and declared.high is not None:
            ax.axvspan(declared.low, declared.high, color="#ff7f0e", alpha=0.08, zorder=0,
                       label="declared")
        ax.set_title(capability_id, fontsize=8)
        ax.set_xlabel(f"{axis_name} ({axis_spec.scale})", fontsize=8)
        quantity = spec.labels.get("quantity", "measured scalar")
        ax.set_ylabel(quantity, fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, lw=0.5)
        if index == 0:
            ax.legend(fontsize=6, loc="best")
    for empty in range(len(capability_ids), rows * columns):
        axes_grid[empty // columns][empty % columns].axis("off")
    fig.suptitle("Applicability envelopes: detected boundaries vs. declared intervals",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = outdir / "fig1_envelopes.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    # ---- figure 2: samples vs the acceptance window ---------------------
    fig, axes_grid = plt.subplots(1, len(capability_ids),
                                  figsize=(4.4 * len(capability_ids), 3.2), squeeze=False)
    for index, capability_id in enumerate(capability_ids):
        spec = registry.get(capability_id)
        report = suite.reports[capability_id]
        axis_name = spec.axes[0].name
        scan = report.scans[axis_name]
        axis_spec = spec.axes[0]
        ax = axes_grid[0][index]
        measured = [v if v is not None else float("nan") for v in scan.measured()]
        if axis_spec.scale == "log":
            ax.set_xscale("log")
        ax.plot(scan.values, measured, "o-", ms=4, lw=0.9, color="#1f77b4", label="probe")
        criterion = spec.policy.criterion
        if criterion.kind == "range" and criterion.lower is not None:
            ax.axhline(criterion.lower, color="#d62728", lw=1.0, ls=":",
                       label=f"lower limit {criterion.lower:g}")
        if criterion.kind == "range" and criterion.upper is not None:
            ax.axhline(criterion.upper, color="#d62728", lw=1.2, ls="--",
                       label=f"upper limit {criterion.upper:g}")
        if criterion.kind in ("abs", "rel") and criterion.reference is not None \
                and criterion.tolerance is not None:
            scale = abs(criterion.reference) if criterion.kind == "rel" else 1.0
            ax.axhline(criterion.reference + criterion.tolerance * scale, color="#d62728",
                       lw=1.2, ls="--", label="acceptance limit")
        for boundary in report.envelope.axes[axis_name].boundaries:
            ax.axvline(boundary.value, color="#2ca02c", ls="--", lw=1.0)
        ax.set_title(capability_id, fontsize=8)
        ax.set_xlabel(f"{axis_name} ({axis_spec.scale})", fontsize=8)
        ax.set_ylabel("measured scalar", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=6, loc="best")
    fig.suptitle("Probe samples against the acceptance window", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = outdir / "fig2_acceptance_window.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths