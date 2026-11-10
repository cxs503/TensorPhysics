#!/usr/bin/env python
"""Render the numeric tables of the P40W4-N09 report from the record (no hand copying)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs" / "p40w4_n09" / "verification.json"
OUT = ROOT / "docs" / "p40w4_n09" / "RESULTS_TABLES.md"


def _fmt(x):
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, float):
        return f"{x:.6e}"
    return str(x)


def main() -> None:
    r = json.loads(RECORD.read_text())
    suite = r["suite"]
    cfg = suite["config"]
    lines = ["# P40W4-N09 numeric tables (generated from `verification.json`)", ""]

    lines += ["## 1. Configuration", "", "| key | value |", "| --- | --- |"]
    for k in sorted(cfg):
        lines.append(f"| `{k}` | {_fmt(cfg[k])} |")
    lines.append("")

    lines += ["## 2. Rates and step counts", "", "| quantity | value |", "| --- | --- |"]
    prod = suite["runs"][str(suite["production_degree"])]
    for k in sorted(prod["schedule"]):
        lines.append(f"| `{k}` | {_fmt(prod['schedule'][k])} |")
    diag = prod["diagnostics"]
    for k in ("macro_steps", "n_panel_steps", "n_solid_steps", "n_dem_steps", "events"):
        lines.append(f"| `{k}` | {_fmt(diag[k])} |")
    lines.append("")

    lines += ["## 3. Degree scan", "",
              "| p | ledger residual | abs. imbalance [J] | flux L2 (settled) | "
              "thermal mismatch | drag mismatch |",
              "| --- | --- | --- | --- | --- | --- |"]
    for p in sorted(suite["runs"], key=int):
        run = suite["runs"][p]
        d = run["diagnostics"]
        lines.append(
            f"| {p} | {_fmt(run['residual'])} | {_fmt(d['absolute_imbalance'])} | "
            f"{_fmt(d['flux_continuity_l2_settled'])} | "
            f"{_fmt(d['thermal_transfer_mismatch'])} | "
            f"{_fmt(d['drag_transfer_mismatch'])} |")
    lines.append("")

    lines += ["## 4. Energy book (production degree)", "",
              "| component | initial [J] | final [J] |", "| --- | --- | --- |"]
    for c in ("kinetic", "potential", "internal", "dissipated"):
        lines.append(f"| {c} | {_fmt(diag['energy_initial'][c])} | "
                     f"{_fmt(diag['energy_final'][c])} |")
    lines.append("")
    for k in ("initial_scale", "residual", "residual_settled", "residual_thermal",
              "residual_mechanical", "flux_fluid_peak", "contact_max_overlap",
              "contact_impulse", "dissipated", "drag_source_energy",
              "drag_delivered_energy", "max_panel_temperature",
              "max_temperature_fluid", "min_particle_velocity"):
        lines.append(f"* `{k}` = {_fmt(diag.get(k))}")
    lines.append("")

    lines += ["## 5. Negative control (coupling switched off)", "",
              "| field | secular drift | bounded oscillation |", "| --- | --- | --- |"]
    nc = suite["negative_control"]
    for f in ("fluid_thermal", "panel_thermal", "plate_elastic", "particles_kinetic"):
        lines.append(f"| {f} | {_fmt(nc[f])} | {_fmt(nc['oscillation'][f])} |")
    lines.append(f"| **max drift** | {_fmt(nc['max'])} | "
                 f"{_fmt(nc['oscillation_max'])} |")
    lines.append("")

    lines += ["## 6. Exchange-operator order study", "",
              "| p | h [s] | relative quadrature error | observed order |",
              "| --- | --- | --- | --- |"]
    for p, run in sorted(suite["operator_study"]["runs"].items()):
        for h, e in zip(run["h"], run["error"]):
            lines.append(f"| {p} | {_fmt(h)} | {_fmt(e)} | "
                         f"{run['observed_order']:.3f} |")
    lines.append("")
    lines += [f"gains: {json.dumps({k: round(v, 3) for k, v in suite['operator_order'].items()})}", ""]

    lines += ["## 7. Gate verdicts", "", "| criterion | value | threshold | cmp | verdict |",
              "| --- | --- | --- | --- | --- |"]
    for key, o in r["outcomes"].items():
        lines.append(f"| `{key}` | {_fmt(o['value'])} | {_fmt(o['threshold'])} | "
                     f"{o['comparison']} | **{o['outcome']}** |")
    lines.append("")
    lines += [f"verdict = **{r['verdict']}**, counts = {r['counts']}, "
              f"criteria digest = `{r['criteria_digest']}`, "
              f"git rev = `{r['environment']['git_rev']}`", ""]
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
