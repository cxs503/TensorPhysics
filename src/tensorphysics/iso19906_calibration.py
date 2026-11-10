"""ISO 19906 ice-pressure design partial-factor calibration — top-level API.

This module provides a unified entry point for the ISO 19906 ice-pressure
partial-factor back-calibration workflow, integrating:

1. **p-A extreme value statistics** — ice pressure–area relationship with
   Gumbel/Weibull/GEV extreme value fitting (from field data or synthetic).
2. **Target reliability inverse** — finding ``(gamma_f, psi)`` such that a
   member designed by the ISO 19906 format attains ``beta_T = 4.0``.
3. **Monte Carlo verification** — FORM/IS/MC cross-checks at 10^6 samples.
4. **ISO 19906 table comparison** — row-by-row deviation map against the
   normative tables with non-conservatism quantification.

References
----------
- ISO 19906:2019 — Petroleum and natural gas industries — Fixed steel
  offshore structures in Arctic and ice-covered waters.
- ISO 2394:2015 — General principles on reliability of structures.
- ISO 19906 Annex A — Partial factor design format for ice actions.

Quick start
-----------
>>> from tensorphysics.iso19906_calibration import run_full_calibration
>>> results = run_full_calibration()
>>> results["criteria_summary"]["all_passed"]
True

The returned dictionary is JSON-serialisable and contains every number
quoted by the report (zero manual copying discipline).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from .ice_partial_factors import (
    PAreaForceModel,
    run_calibration,
    run_target_inverse_study,
    evaluate_criteria as evaluate_inverse_criteria,
)
from .ice_partial_factors.target_inverse_study import (
    DEFAULT_CONFIG,
    DEFAULT_TOLERANCES,
    beta_curve,
    run_study,
)
from .ice_partial_factors.target_reliability_inverse import (
    TARGET_BETA,
    TARGET_PF,
    calibrate_kappa,
    solve_factors,
    solve_factors_multistart,
)
from .ice_partial_factors.p_area_joint import (
    characteristic_force,
    force_moments,
    analytic_beta_ln,
)
from .ice_partial_factors.target_inverse_checks import (
    reliability_crosscheck,
    conditional_mc_pf,
    iso_deviation_map,
    negative_control,
    design_table_rows,
)

__all__ = [
    # Top-level orchestration
    "run_full_calibration",
    "run_iso19906_study",
    "export_results_json",
    # Re-exports for convenience
    "PAreaForceModel",
    "TARGET_BETA",
    "TARGET_PF",
    "DEFAULT_CONFIG",
    "DEFAULT_TOLERANCES",
    # Core functions
    "calibrate_kappa",
    "solve_factors",
    "solve_factors_multistart",
    "characteristic_force",
    "force_moments",
    "analytic_beta_ln",
    "reliability_crosscheck",
    "conditional_mc_pf",
    "iso_deviation_map",
    "negative_control",
    "design_table_rows",
    "beta_curve",
    "run_study",
]


def run_full_calibration(
    config: dict | None = None,
    tolerances: dict | None = None,
    with_provenance: bool = True,
) -> dict:
    """Run the complete ISO 19906 partial-factor calibration study.

    This is the main entry point for the maturity node P46W3-N09. It executes
    the full chain:

    1. Build the joint p-A ice-force model (lognormal marginals, ISO power-law
       correlation).
    2. Compute characteristic ice force (100-year return period).
    3. Solve the two-equation inverse for ``(gamma_f, psi)`` at ``beta_T = 4.0``.
    4. Cross-check with FORM/IS/MC/Rao-Blackwell.
    5. Compute adjoint sensitivities.
    6. Build the ISO 19906 table deviation map.
    7. Run the negative control (rho=0).
    8. Evaluate all machine criteria.

    Parameters
    ----------
    config : dict, optional
        Override default configuration (model parameters, sample sizes, etc.).
    tolerances : dict, optional
        Override default tolerance thresholds for machine criteria.
    with_provenance : bool
        If True, add provenance metadata (timestamp, runner info).

    Returns
    -------
    dict
        JSON-serialisable dictionary with all results. Keys include:
        - ``study``: metadata (node, title, repository, etc.)
        - ``model``: the PAreaForceModel parameters
        - ``force_statistics``: log-space moments of the ice force
        - ``inverse``: the calibrated ``(gamma_f, psi, kappa)``
        - ``crosscheck``: FORM/IS/MC comparison
        - ``sensitivity``: adjoint derivatives
        - ``iso_deviations``: table-by-table comparison
        - ``negative_control``: rho=0 control experiment
        - ``machine_criteria``: pass/fail rows
        - ``criteria_summary``: aggregate pass/fail count
    """
    results = run_study(config=config, tolerances=tolerances)

    if with_provenance:
        results["provenance"] = {
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "runner": "tensorphysics.iso19906_calibration.run_full_calibration",
            "node": "P46W3-N09",
            "mission": "ISO 19906 ice-pressure partial-factor back-calibration maturity",
            "repository": "TensorPhysics",
        }

    return results


def run_iso19906_study(
    model_params: dict | None = None,
    beta_target: float = 4.0,
    samples_mc: int = 4_000_000,
    samples_is: int = 200_000,
    seed: int = 20261109,
) -> dict:
    """Simplified entry point for the ISO 19906 study.

    Parameters
    ----------
    model_params : dict, optional
        Override PAreaForceModel parameters.
    beta_target : float
        Target reliability index (default 4.0 per ISO 2394).
    samples_mc : int
        Monte Carlo sample count for cross-check.
    samples_is : int
        Importance sampling count for cross-check.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Study results (same structure as run_full_calibration).
    """
    config = {
        "beta_target": beta_target,
        "samples_mc": samples_mc,
        "samples_is": samples_is,
        "seed": seed,
    }
    if model_params:
        config["model"] = model_params
    return run_full_calibration(config=config)


def export_results_json(
    results: dict,
    output_path: str,
    indent: int = 2,
) -> str:
    """Export study results to a JSON file.

    Parameters
    ----------
    results : dict
        Results from run_full_calibration or run_iso19906_study.
    output_path : str
        Path to write the JSON file.
    indent : int
        JSON indentation level.

    Returns
    -------
    str
        The absolute path of the written file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=indent, sort_keys=True)
    return os.path.abspath(output_path)


def summary_table(results: dict) -> str:
    """Generate a Markdown summary table of the machine criteria.

    Parameters
    ----------
    results : dict
        Results from run_full_calibration.

    Returns
    -------
    str
        Markdown-formatted table.
    """
    lines = [
        "| Criterion | Measured | Tolerance | Passed |",
        "|---|---|---|---|",
    ]
    for row in results.get("machine_criteria", []):
        if row.get("skipped"):
            continue
        name = row["criterion"]
        measured = row.get("measured", {})
        passed = "✅" if row["passed"] else "❌"
        # Extract key measured values
        if "beta_residual_abs" in measured:
            val = f"|β−β_T| = {measured['beta_residual_abs']:.2e}"
            tol = f"≤ {row.get('note', '')}"
        elif "form_vs_is_relative_error" in measured:
            val = f"FORM vs IS: {measured['form_vs_is_relative_error']:.2e}"
            tol = "≤ 3%"
        elif "delta_gamma_f_rel" in measured:
            val = f"Δγ_f = {100*measured['delta_gamma_f_rel']:+.1f}%"
            tol = "≥ 5%"
        else:
            val = str({k: v for k, v in measured.items() if isinstance(v, (int, float))})[:60]
            tol = "—"
        lines.append(f"| `{name}` | {val} | {tol} | {passed} |")
    return "\n".join(lines)
