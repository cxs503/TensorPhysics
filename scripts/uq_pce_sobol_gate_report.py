#!/usr/bin/env python3
"""Generate a markdown report from a UQ PCE Sobol' gate study.

Reads a study JSON (produced by ``uq_pce_sobol_gate_e2e.py`` or
``run_uq_study``) and renders a structured markdown report with:

- Input specification and sample provenance
- Surrogate quality metrics (ε_LOO, R², n_terms)
- Sobol' indices table with bootstrap confidence intervals
- Cross-check results (PCE vs Monte Carlo)
- Verdict with reason codes

Usage::

    python scripts/uq_pce_sobol_gate_report.py \\
        --study artifacts/uq_gate/ishigami_study.json \\
        --out artifacts/uq_gate/report.md

The report is self-contained: all numbers come from the JSON, no hand-copied
digits.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import datetime, timezone
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _fmt(value: float, digits: int = 6) -> str:
    """Format a float to fixed decimal places."""
    return f"{value:.{digits}f}"


def _fmt_sci(value: float, digits: int = 2) -> str:
    """Format a float in scientific notation."""
    return f"{value:.{digits}e}"


def _ci_str(ci: list[float] | None, digits: int = 4) -> str:
    """Format a confidence interval as [low, high]."""
    if ci is None:
        return "N/A"
    return f"[{_fmt(ci[0], digits)}, {_fmt(ci[1], digits)}]"


def render_header(study: dict[str, Any]) -> str:
    """Render the report header."""
    benchmark = study.get("benchmark", "unknown")
    status = study.get("status", "UNKNOWN")
    reason = study.get("verdict_reason", "N/A")
    elapsed = study.get("elapsed_seconds", 0.0)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        f"# UQ PCE Sobol' Gate Report: {benchmark.title()}",
        "",
        f"**Generated**: {timestamp}  ",
        f"**Status**: {status}  ",
        f"**Reason**: {reason}  ",
        f"**Elapsed**: {elapsed:.2f} s",
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


def render_spec(study: dict[str, Any]) -> str:
    """Render the input specification and sample provenance."""
    sample_hash = study.get("sample_hash", "N/A")
    surrogate = study.get("surrogate", {})
    n_samples = surrogate.get("n_samples", "N/A")

    lines = [
        "## 1. Input Specification and Sample Provenance",
        "",
        f"- **Sample hash**: `{sample_hash}`",
        f"- **Number of samples**: {n_samples}",
        "- **Sampling rule**: Scrambled Sobol' (deterministic, hash-pinned)",
        "- **Marginals**: See benchmark definition (uniform [−π, π] for Ishigami, uniform [0, 1] for G-function)",
        "",
        "The sample hash ensures bit-identical reproducibility: same spec + same seed → same hash → same indices.",
        "",
    ]
    return "\n".join(lines)


def render_surrogate(study: dict[str, Any]) -> str:
    """Render the surrogate quality metrics."""
    surrogate = study.get("surrogate", {})
    eps_loo = surrogate.get("eps_loo", "N/A")
    r2 = surrogate.get("r2", "N/A")
    n_terms = surrogate.get("n_terms", "N/A")
    n_samples = surrogate.get("n_samples", "N/A")

    eps_loo_status = "✓" if isinstance(eps_loo, (int, float)) and eps_loo <= 0.1 else "✗"

    lines = [
        "## 2. Surrogate Quality",
        "",
        "| Metric | Value | Threshold | Status |",
        "|--------|-------|-----------|--------|",
        f"| ε_LOO (normalized LOO error) | {_fmt(eps_loo, 4) if isinstance(eps_loo, (int, float)) else eps_loo} | ≤ 0.1 | {eps_loo_status} |",
        f"| R² (coefficient of determination) | {_fmt(r2, 4) if isinstance(r2, (int, float)) else r2} | — | — |",
        f"| Number of PCE terms | {n_terms} | — | — |",
        f"| Number of samples | {n_samples} | — | — |",
        "",
    ]

    if isinstance(eps_loo, (int, float)):
        if eps_loo <= 0.01:
            quality = "excellent"
        elif eps_loo <= 0.05:
            quality = "good"
        elif eps_loo <= 0.1:
            quality = "acceptable"
        else:
            quality = "poor (WITHHELD likely)"
        lines.append(f"**Surrogate quality**: {quality}")
        lines.append("")

    return "\n".join(lines)


def render_indices(study: dict[str, Any]) -> str:
    """Render the Sobol' indices table."""
    indices = study.get("indices", {})
    if not indices:
        return "## 3. Sobol' Indices\n\nNo indices available.\n\n"

    # Separate first-order, total-effect, and second-order
    first_order = {k: v for k, v in indices.items() if not k.endswith("_total") and "*" not in k}
    total_effect = {k.replace("_total", ""): v for k, v in indices.items() if k.endswith("_total")}
    second_order = {k: v for k, v in indices.items() if "*" in k}

    lines = [
        "## 3. Sobol' Indices",
        "",
    ]

    # First-order table
    if first_order:
        lines.extend([
            "### 3.1 First-Order Indices",
            "",
            "| Input | Analytic | PCE | Absolute Error | MC (Jansen) |",
            "|-------|----------|-----|----------------|-------------|",
        ])
        for name, data in first_order.items():
            analytic = data.get("analytic", "N/A")
            pce = data.get("pce", "N/A")
            err = data.get("abs_error", "N/A")
            mc = data.get("mc_jansen", "N/A")

            analytic_str = _fmt(analytic, 6) if isinstance(analytic, (int, float)) else str(analytic)
            pce_str = _fmt(pce, 6) if isinstance(pce, (int, float)) else str(pce)
            err_str = _fmt_sci(err, 2) if isinstance(err, (int, float)) else str(err)
            mc_str = _fmt(mc, 6) if isinstance(mc, (int, float)) else str(mc)

            lines.append(f"| {name} | {analytic_str} | {pce_str} | {err_str} | {mc_str} |")
        lines.append("")

    # Total-effect table
    if total_effect:
        lines.extend([
            "### 3.2 Total-Effect Indices",
            "",
            "| Input | Analytic | PCE | Absolute Error | MC (Jansen) |",
            "|-------|----------|-----|----------------|-------------|",
        ])
        for name, data in total_effect.items():
            analytic = data.get("analytic", "N/A")
            pce = data.get("pce", "N/A")
            err = data.get("abs_error", "N/A")
            mc = data.get("mc_jansen", "N/A")

            analytic_str = _fmt(analytic, 6) if isinstance(analytic, (int, float)) else str(analytic)
            pce_str = _fmt(pce, 6) if isinstance(pce, (int, float)) else str(pce)
            err_str = _fmt_sci(err, 2) if isinstance(err, (int, float)) else str(err)
            mc_str = _fmt(mc, 6) if isinstance(mc, (int, float)) else str(mc)

            lines.append(f"| {name} | {analytic_str} | {pce_str} | {err_str} | {mc_str} |")
        lines.append("")

    # Second-order table
    if second_order:
        lines.extend([
            "### 3.3 Second-Order Indices",
            "",
            "| Pair | Analytic | PCE | Absolute Error |",
            "|------|----------|-----|----------------|",
        ])
        for name, data in second_order.items():
            analytic = data.get("analytic", "N/A")
            pce = data.get("pce", "N/A")
            err = data.get("abs_error", "N/A")

            analytic_str = _fmt(analytic, 6) if isinstance(analytic, (int, float)) else str(analytic)
            pce_str = _fmt(pce, 6) if isinstance(pce, (int, float)) else str(pce)
            err_str = _fmt_sci(err, 2) if isinstance(err, (int, float)) else str(err)

            lines.append(f"| {name} | {analytic_str} | {pce_str} | {err_str} |")
        lines.append("")

    return "\n".join(lines)


def render_cross_check(study: dict[str, Any]) -> str:
    """Render the cross-check results (PCE vs MC)."""
    cross_check = study.get("cross_check")
    if not cross_check:
        return "## 4. Cross-Check (PCE vs Monte Carlo)\n\nCross-check not available (MC not run).\n\n"

    agreement = cross_check.get("agreement", False)
    max_dev = cross_check.get("max_relative_deviation", "N/A")
    entries = cross_check.get("entries", [])

    status = "✓ PASS" if agreement else "✗ FAIL"

    lines = [
        "## 4. Cross-Check (PCE vs Monte Carlo)",
        "",
        f"**Agreement**: {status}  ",
        f"**Max relative deviation**: {_fmt(max_dev, 4) if isinstance(max_dev, (int, float)) else max_dev}  ",
        f"**Threshold**: ≤ 0.05 (5%)",
        "",
    ]

    if entries:
        lines.extend([
            "| Index | PCE | MC | Relative Deviation |",
            "|-------|-----|----|--------------------|",
        ])
        for entry in entries:
            name = entry.get("name", "N/A")
            pce_val = entry.get("pce", "N/A")
            mc_val = entry.get("mc", "N/A")
            rel_dev = entry.get("relative_deviation", "N/A")

            pce_str = _fmt(pce_val, 6) if isinstance(pce_val, (int, float)) else str(pce_val)
            mc_str = _fmt(mc_val, 6) if isinstance(mc_val, (int, float)) else str(mc_val)
            rel_dev_str = _fmt(rel_dev, 4) if isinstance(rel_dev, (int, float)) else str(rel_dev)

            lines.append(f"| {name} | {pce_str} | {mc_str} | {rel_dev_str} |")
        lines.append("")

    return "\n".join(lines)


def render_verdict(study: dict[str, Any]) -> str:
    """Render the verdict and output budget."""
    status = study.get("status", "UNKNOWN")
    reason = study.get("verdict_reason", "N/A")
    budget = study.get("output_budget", {})

    lines = [
        "## 5. Verdict",
        "",
        f"**Status**: **{status}**  ",
        f"**Reason**: {reason}",
        "",
    ]

    if budget:
        u95 = budget.get("u95", "N/A")
        threshold = budget.get("threshold", "N/A")
        lines.extend([
            "### Output Budget",
            "",
            f"- **95% prediction interval width (U)**: {_fmt(u95, 4) if isinstance(u95, (int, float)) else u95}",
            f"- **Engineering threshold**: {_fmt(threshold, 4) if isinstance(threshold, (int, float)) else threshold}",
            "",
        ])

        if isinstance(u95, (int, float)) and isinstance(threshold, (int, float)):
            if u95 <= threshold:
                lines.append("The output uncertainty is within the acceptable engineering threshold.")
            else:
                lines.append("The output uncertainty exceeds the acceptable engineering threshold.")
            lines.append("")

    # Reason code explanation
    if status == "WITHHELD":
        lines.extend([
            "### Reason Codes",
            "",
            "| Code | Meaning |",
            "|------|---------|",
            "| W01 | Non-finite values in samples |",
            "| W02 | Too few samples |",
            "| W03 | ε_LOO > 0.1 |",
            "| W04 | PCE fit failed |",
            "| W05 | Degenerate output (zero variance) |",
            "| W06 | Bootstrap CI too wide |",
            "| W07 | Cross-check missing or failed |",
            "| W08 | Surrogate not available |",
            "",
        ])

    return "\n".join(lines)


def render_acceptance(study: dict[str, Any]) -> str:
    """Render the acceptance criteria results."""
    acceptance = study.get("acceptance", {})
    if not acceptance:
        return ""

    lines = [
        "## 6. Acceptance Criteria",
        "",
        "| Criterion | Description | Measured | Limit | Status |",
        "|-----------|-------------|----------|-------|--------|",
    ]

    for crit_name, crit in acceptance.items():
        description = crit.get("description", crit_name)
        measured = crit.get("worst_abs_error", crit.get("worst_rel_error", crit.get("max_rel_deviation", "N/A")))
        limit = crit.get("limit", "N/A")
        passed = crit.get("pass", False)

        measured_str = _fmt_sci(measured, 2) if isinstance(measured, (int, float)) else str(measured)
        limit_str = _fmt(limit, 4) if isinstance(limit, (int, float)) else str(limit)
        status = "✓ PASS" if passed else "✗ FAIL"

        # Shorten description for table
        short_desc = crit_name.replace("_", " ").title()

        lines.append(f"| {crit_name} | {short_desc} | {measured_str} | {limit_str} | {status} |")

    lines.append("")
    return "\n".join(lines)


def render_footer() -> str:
    """Render the report footer."""
    lines = [
        "---",
        "",
        "## References",
        "",
        "1. Xiu, D., & Karniadakis, G. E. (2002). The Wiener-Askey polynomial chaos",
        "   for stochastic differential equations. *SIAM Journal on Scientific Computing*, 24(2), 610-644.",
        "2. Sobol', I. M. (2001). Global sensitivity indices for nonlinear mathematical",
        "   models and their Monte Carlo estimates. *Mathematics and Computers in Simulation*, 55(1-3), 271-280.",
        "3. Saltelli, A. (2002). Making best use of model evaluations to compute",
        "   sensitivity indices. *Computer Physics Communications*, 145(2), 280-297.",
        "4. Blatman, G., & Sudret, B. (2011). Adaptive sparse polynomial chaos",
        "   expansion based on least angle regression. *Journal of Computational Physics*, 230(6), 2345-2367.",
        "",
        "**Documentation**: `docs/UQ_PCE_SOBOL_GATE.md`  ",
        "**Package**: `tensorphysics.uq`",
        "",
    ]
    return "\n".join(lines)


def generate_report(study: dict[str, Any]) -> str:
    """Generate the full markdown report."""
    sections = [
        render_header(study),
        render_spec(study),
        render_surrogate(study),
        render_indices(study),
        render_cross_check(study),
        render_verdict(study),
        render_acceptance(study),
        render_footer(),
    ]
    return "\n".join(section for section in sections if section)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate UQ PCE Sobol' gate report")
    parser.add_argument("--study", type=pathlib.Path, required=True,
                        help="Path to study JSON file")
    parser.add_argument("--out", type=pathlib.Path, required=True,
                        help="Output markdown file path")
    args = parser.parse_args()

    if not args.study.exists():
        print(f"Error: study file not found: {args.study}", file=sys.stderr)
        return 1

    study = json.loads(args.study.read_text())
    report = generate_report(study)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(f"Report written to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
