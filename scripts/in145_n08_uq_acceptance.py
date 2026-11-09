#!/usr/bin/env python3
"""Acceptance run of mission **IN145-N08**: the UQ budget as a pre-evidence gate.

Every machine-checkable criterion of the node is executed here and dumped to
``artifacts/in145_n08/acceptance.json`` together with the figures of the report.
No number in the report is hand-typed: the report generator reads this file.

Criteria (node -> section)
--------------------------
C1  Ishigami first/second/total indices vs closed form
    -- PCE analytic: |dS| <= 1e-3; Saltelli/Jansen: relative error <= 5 %
C2  Refinement: eps_LOO must fall as the budget grows
    -- fixed-truncation axis and degree axis: Spearman <= -0.9 over >= 4 points;
       the adaptive-selection axis is reported as a *falsification* of the
       literal claim (eps_LOO non-monotone, holdout error monotone)
C3  Determinism: same spec + rule + seed -> bit-identical sample hash and
    Sobol' indices, in-process and across processes/hash seeds
C4  Injections: NaN samples / starved budget / eps_LOO > 0.1 / missing
    cross-check / tampered MC / hash mismatch / degenerate output / no surrogate
    / threshold knife-edge -> never PASS, always with the matching reason code
C5  PCE (analytic) vs Saltelli (sampled): relative deviation <= 5 % (1st order)
C6  End-to-end: one call of the unmodified Holtrop-1984 kernel -> UQ -> three
    state output -> chained ASME V&V 20 comparison (kernel file hash unchanged)
C7  Interval calibration: 95 % bootstrap CI coverage of the analytic indices
C8  Shift invariance of the sampled estimators (regression of the defect fixed
    in this node: a QoI with mean O(1e3) must not corrupt the indices)

Usage::

    python scripts/in145_n08_uq_acceptance.py            # full run (~3 min)
    python scripts/in145_n08_uq_acceptance.py --quick    # reduced coverage study

Exit code 0 iff every criterion passed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.uq import (  # noqa: E402
    FAIL,
    PASS,
    WITHHELD,
    chain_asme_validation,
    coverage_study,
    engineering_error_gate,
    holtrop_type_benchmark,
    injection_suite,
    ishigami,
    ishigami_analytic,
    ishigami_spec,
    pre_evidence,
    refinement_study,
    run_uq_study,
    saltelli_indices,
)
from tensorphysics.uq.refinement import adaptive_selection_diagnostic  # noqa: E402

KERNEL_FILES = (
    "src/tensorphysics/physics_kernels_holtrop1984.py",
    "src/tensorphysics/physics_kernels_resistance.py",
)
OUT_DIR = ROOT / "artifacts" / "in145_n08"

ISHIGAMI_TARGETS = (
    ("first", 0, 0.31390519),
    ("first", 1, 0.44241114),
    ("second", (0, 2), 0.24368366),
    ("total", 0, 0.55758886),
    ("total", 2, 0.24368366),
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _round(value: Any, digits: int = 9) -> Any:
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if not np.isfinite(number):
        return str(number)
    return float(np.round(number, digits))


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_round(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        return _round(obj)
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    return obj


def _sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_digest(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _criterion(cid: str, title: str, passed: bool, numbers: dict[str, Any],
               artifacts: Sequence[str] = (), notes: Sequence[str] = ()) -> dict[str, Any]:
    return {
        "id": cid,
        "title": title,
        "passed": bool(passed),
        "numbers": _jsonable(numbers),
        "artifacts": list(artifacts),
        "notes": list(notes),
    }


# ---------------------------------------------------------------------------
# C1 / C5: index accuracy and method cross-check
# ---------------------------------------------------------------------------
def criterion_c1_c5(evidence_dir: pathlib.Path) -> tuple[dict[str, Any], dict[str, Any], Any]:
    analytic = ishigami_analytic()
    study = run_uq_study(
        ishigami,
        ishigami_spec(),
        n_samples=1024,
        degree=8,
        n_boot=200,
        n_base_mc=1 << 15,
        bootstrap_mc=0,
        budget_n_mc=20000,
    )
    indices = study.pce_sobol
    targets: dict[str, dict[str, float]] = {}
    max_abs = 0.0
    for kind, key, truth in ISHIGAMI_TARGETS:
        if kind == "second":
            estimate = float(indices.second.get(key, float("nan")))
        elif kind == "first":
            estimate = float(indices.first[key])
        else:
            estimate = float(indices.total[key])
        err = abs(estimate - truth)
        max_abs = max(max_abs, err)
        targets[f"{kind}:{key}"] = {"estimate": estimate, "analytic": truth, "abs_error": err}

    ci_payload = {}
    if indices.first_ci is not None:
        lo, hi = indices.first_ci
        ci_payload["first"] = [[float(lo[i]), float(hi[i])] for i in range(3)]
        tlo, thi = indices.total_ci
        ci_payload["total"] = [[float(tlo[i]), float(thi[i])] for i in range(3)]

    mc_err: dict[str, float] = {}
    for i in range(3):
        for kind, truth in (("first", analytic["first"][i]), ("total", analytic["total"][i])):
            estimate = float(study.mc.first[i]) if kind == "first" else float(study.mc.total[i])
            if abs(truth) < 1e-12:
                mc_err[f"{kind}:x{i + 1}"] = abs(estimate - truth)
            else:
                mc_err[f"{kind}:x{i + 1}"] = abs(estimate - truth) / abs(truth)
    max_mc_rel = max(
        value for key, value in mc_err.items() if not key.startswith("first:x3")
    )

    cross = study.cross_check
    payload = {
        "study_status": study.status,
        "eps_loo": study.surrogate.eps_loo,
        "n_terms": study.surrogate.n_terms,
        "quoted": {"S1": 0.3139, "S2": 0.4424, "ST": 0.5574},
        "targets": targets,
        "max_abs_error": max_abs,
        "ci95": ci_payload,
        "mc_relative_error": mc_err,
        "mc_max_relative_error": max_mc_rel,
        "mc_n_base": study.mc.n_base,
        "mc_model_evaluations": study.mc.n_model_evaluations,
        "analytic_first": [float(v) for v in analytic["first"]],
        "analytic_total": [float(v) for v in analytic["total"]],
        "pce_first": [float(v) for v in indices.first],
        "pce_total": [float(v) for v in indices.total],
        "mc_first": [float(v) for v in study.mc.first],
        "mc_total": [float(v) for v in study.mc.total],
        "cross_check": None if cross is None else cross.as_dict(),
        "cross_check_max_relative_deviation": None if cross is None else cross.max_relative_deviation,
        "cross_check_max_second_relative_deviation": None if cross is None else cross.max_second_relative_deviation,
    }
    (evidence_dir / "c1_ishigami_indices.json").write_text(json.dumps(_jsonable(payload), indent=1))
    (evidence_dir / "c1_ishigami_study.json").write_text(json.dumps(_jsonable(study.as_dict()), indent=1))

    c1 = _criterion(
        "C1",
        "Ishigami indices: PCE |dS| <= 1e-3 and Saltelli |rel| <= 5 %",
        max_abs <= 1.0e-3 and max_mc_rel <= 0.05,
        {
            "pce_max_abs_error": max_abs,
            "mc_max_relative_error": max_mc_rel,
            "threshold_abs": 1.0e-3,
            "threshold_rel": 0.05,
            "targets": targets,
            "mc_relative_error": mc_err,
            "ci95": ci_payload,
            "eps_loo": study.surrogate.eps_loo,
            "analytic_first": payload["analytic_first"],
            "analytic_total": payload["analytic_total"],
            "pce_first": payload["pce_first"],
            "pce_total": payload["pce_total"],
            "mc_first": payload["mc_first"],
            "mc_total": payload["mc_total"],
        },
        artifacts=["artifacts/in145_n08/evidence/c1_ishigami_indices.json",
                   "artifacts/in145_n08/evidence/c1_ishigami_study.json"],
    )
    c5 = _criterion(
        "C5",
        "PCE (analytic) vs Saltelli (sampled): relative deviation <= 5 %",
        bool(cross is not None and cross.agreement and cross.max_relative_deviation <= 0.05),
        {
            "max_relative_deviation": None if cross is None else cross.max_relative_deviation,
            "max_second_relative_deviation": None if cross is None else cross.max_second_relative_deviation,
            "agreement": None if cross is None else cross.agreement,
            "n_compared": None if cross is None else cross.n_compared,
            "entries": None if cross is None else [e.as_dict() for e in cross.entries],
        },
        artifacts=["artifacts/in145_n08/evidence/c1_ishigami_indices.json"],
    )
    return c1, c5, study


# ---------------------------------------------------------------------------
# C2: refinement monotonicity
# ---------------------------------------------------------------------------
def criterion_c2(evidence_dir: pathlib.Path, quick: bool) -> dict[str, Any]:
    study = refinement_study(
        ishigami_spec(),
        ishigami,
        benchmark="ishigami",
        degrees=(2, 3, 4, 6, 8, 10, 12),
        degree_n_samples=1024,
        truncation_degree=4,
        adaptive_n_samples=(128, 256, 512, 1024, 2048),
        holdout_n=12000 if quick else 20000,
    )
    payload = study.as_dict()
    payload["adaptive_diagnostic"] = adaptive_selection_diagnostic(study)
    (evidence_dir / "c2_refinement.json").write_text(json.dumps(_jsonable(payload), indent=1))
    numbers = {
        "threshold_spearman": study.threshold,
        "estimation_axis": {
            "spearman_loo": study.estimation_axis.spearman_loo,
            "spearman_holdout": study.estimation_axis.spearman_holdout,
            "n_points": len(study.estimation_axis.points),
            "loo_monotone": study.estimation_axis.loo_monotone,
            "holdout_monotone": study.estimation_axis.holdout_monotone,
            "eps_loo": study.estimation_axis.eps_loo.tolist(),
            "holdout": study.estimation_axis.holdout.tolist(),
            "budgets": study.estimation_axis.budgets.tolist(),
            "loo_optimism": [p.loo_optimism for p in study.estimation_axis.points],
        },
        "truncation_axis": {
            "spearman_loo": study.truncation_axis.spearman_loo,
            "spearman_holdout": study.truncation_axis.spearman_holdout,
            "n_points": len(study.truncation_axis.points),
            "loo_monotone": study.truncation_axis.loo_monotone,
            "holdout_monotone": study.truncation_axis.holdout_monotone,
            "eps_loo": study.truncation_axis.eps_loo.tolist(),
            "holdout": study.truncation_axis.holdout.tolist(),
            "degrees": [p.degree for p in study.truncation_axis.points],
        },
        "adaptive_axis": {
            "spearman_loo": study.adaptive_axis.spearman_loo,
            "spearman_holdout": study.adaptive_axis.spearman_holdout,
            "loo_monotone": study.adaptive_axis.loo_monotone,
            "holdout_monotone": study.adaptive_axis.holdout_monotone,
            "eps_loo": study.adaptive_axis.eps_loo.tolist(),
            "holdout": study.adaptive_axis.holdout.tolist(),
            "budgets": study.adaptive_axis.budgets.tolist(),
            "loo_optimism": [p.loo_optimism for p in study.adaptive_axis.points],
            "n_terms": [p.n_terms for p in study.adaptive_axis.points],
        },
        "adaptive_diagnostic": payload["adaptive_diagnostic"],
        "all_met": study.all_met,
    }
    return _criterion(
        "C2",
        "Refinement: eps_LOO falls with budget (Spearman <= -0.9, >= 4 points)",
        study.all_met,
        numbers,
        artifacts=["artifacts/in145_n08/evidence/c2_refinement.json"],
        notes=[
            "the literal claim 'eps_LOO falls as the sample count grows' is FALSE "
            "under adaptive term selection; the axis that makes it true is the "
            "fixed-truncation one, and the holdout error proves the surrogate "
            "still improves -- see adaptive_diagnostic",
            "the adaptive axis is required to have a monotone *holdout* error and "
            "a non-monotone eps_LOO (the documented selection-optimism effect)",
        ],
    )


# ---------------------------------------------------------------------------
# C3: determinism
# ---------------------------------------------------------------------------
def criterion_c3(evidence_dir: pathlib.Path) -> dict[str, Any]:
    kwargs = dict(n_samples=512, degree=8, n_boot=0, n_base_mc=1 << 12,
                  bootstrap_mc=0, budget_n_mc=4000)
    first = run_uq_study(ishigami, ishigami_spec(), **kwargs)
    second = run_uq_study(ishigami, ishigami_spec(), **kwargs)
    in_process = {
        "sample_hash_a": first.samples.hash,
        "sample_hash_b": second.samples.hash,
        "indices_digest_a": _canonical_digest(first.pce_sobol.as_dict(ishigami_spec().names)),
        "indices_digest_b": _canonical_digest(second.pce_sobol.as_dict(ishigami_spec().names)),
    }
    in_process["identical"] = bool(
        in_process["sample_hash_a"] == in_process["sample_hash_b"]
        and in_process["indices_digest_a"] == in_process["indices_digest_b"]
    )

    across = []
    for hashseed in ("0", "1", "random"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = hashseed
        out = subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).resolve()), "--determinism-probe"],
            capture_output=True, text=True, env=env, check=True, cwd=str(ROOT),
        )
        across.append(json.loads(out.stdout.strip().splitlines()[-1]))
    hashes = {rec["sample_hash"] for rec in across}
    digests = {rec["indices_digest"] for rec in across}
    across_identical = len(hashes) == 1 and len(digests) == 1
    payload = {
        "study_kwargs": _jsonable(kwargs),
        "in_process": in_process,
        "cross_process": across,
        "cross_process_sample_hash": sorted(hashes),
        "cross_process_indices_digest": sorted(digests),
        "identical": bool(in_process["identical"] and across_identical),
    }
    (evidence_dir / "c3_determinism.json").write_text(json.dumps(payload, indent=1))
    return _criterion(
        "C3",
        "same spec + rule + seed -> bit-identical sample hash and Sobol' indices",
        payload["identical"],
        payload,
        artifacts=["artifacts/in145_n08/evidence/c3_determinism.json"],
        notes=[
            "the probe is run in three fresh interpreters with PYTHONHASHSEED in "
            "{0, 1, random}: a hash-order dependence of the sparse selector would "
            "show up as a differing index digest",
        ],
    )


# ---------------------------------------------------------------------------
# C4: injections
# ---------------------------------------------------------------------------
def criterion_c4(evidence_dir: pathlib.Path) -> dict[str, Any]:
    report = injection_suite()
    payload = report.as_dict()
    (evidence_dir / "c4_injections.json").write_text(json.dumps(_jsonable(payload), indent=1))
    return _criterion(
        "C4",
        "injections never yield PASS and always name their reason code",
        report.passed,
        payload,
        artifacts=["artifacts/in145_n08/evidence/c4_injections.json"],
        notes=[
            "the control (clean study) must PASS, otherwise the matrix would be "
            "trivially 'safe'",
            "the 15 % threshold knife-edge is pinned from both sides: exactly 15 % "
            "passes, 15 % + 1e-11 fails",
        ],
    )


# ---------------------------------------------------------------------------
# C6: end-to-end chain on the ship-resistance kernel
# ---------------------------------------------------------------------------
def criterion_c6(evidence_dir: pathlib.Path) -> dict[str, Any]:
    kernel_hashes_before = {rel: _sha256_file(ROOT / rel) for rel in KERNEL_FILES if (ROOT / rel).exists()}
    bench = holtrop_type_benchmark()
    narrow = run_uq_study(
        bench.resistance_kn, bench.spec,
        n_samples=256, degree=4, n_boot=0, n_base_mc=1 << 13,
        bootstrap_mc=100, budget_n_mc=8000,
    )
    wide_bench = holtrop_type_benchmark(rel_uncertainty=0.04)
    wide = run_uq_study(
        wide_bench.resistance_kn, wide_bench.spec,
        n_samples=256, degree=4, n_boot=0, run_mc=False,
        require_cross_check=False, budget_n_mc=8000,
    )
    kernel_hashes_after = {rel: _sha256_file(ROOT / rel) for rel in KERNEL_FILES if (ROOT / rel).exists()}

    # nominal case: one kernel call, then the gate chain
    anchor = float(bench.reference_case["RT_kN"])
    evidence = pre_evidence(narrow, label="holtrop-narrow")
    gate_ok = engineering_error_gate(study=narrow, relative_error=0.0, label="holtrop-narrow")
    gate_wide = engineering_error_gate(study=wide, label="holtrop-wide")
    gate_bare = engineering_error_gate(study=None, relative_error=0.02, label="no-evidence")
    chained = chain_asme_validation(
        study=narrow,
        f_sim=float(narrow.budget.mean),
        f_exp=anchor,
        u_num=float(narrow.surrogate.eps_loo_abs),
        label="holtrop-asme",
    )
    payload = {
        "kernel_hashes_before": kernel_hashes_before,
        "kernel_hashes_after": kernel_hashes_after,
        "kernel_unchanged": kernel_hashes_before == kernel_hashes_after,
        "anchor_RT_kN": anchor,
        "narrow": {
            "status": narrow.status,
            "band_relative_width": narrow.budget.relative_width,
            "band_mean_kN": narrow.budget.mean,
            "band_width_kN": narrow.budget.width,
            "eps_loo": narrow.surrogate.eps_loo,
            "dominant_input": bench.spec.names[int(np.argmax(narrow.pce_sobol.total))],
            "sample_hash": narrow.samples.hash,
            "solver_evaluations": int(narrow.results.n) + int(narrow.mc.n_model_evaluations),
            "indices_total": narrow.pce_sobol.as_dict(bench.spec.names)["total"],
        },
        "wide": {
            "status": wide.status,
            "band_relative_width": wide.budget.relative_width,
            "reason_codes": list(wide.verdict.reason_codes),
        },
        "gate_narrow": gate_ok.as_dict(),
        "gate_wide": gate_wide.as_dict(),
        "gate_no_evidence": gate_bare.as_dict(),
        "evidence_digest": evidence.digest,
        "chained_asme": chained,
    }
    (evidence_dir / "c6_end_to_end.json").write_text(json.dumps(_jsonable(payload), indent=1))
    passed = bool(
        narrow.status == PASS
        and wide.status == FAIL
        and gate_ok.status == PASS
        and gate_wide.status == FAIL
        and gate_bare.status == WITHHELD
        and chained["combined_status"] == PASS
        and payload["kernel_unchanged"]
    )
    return _criterion(
        "C6",
        "end-to-end: one Holtrop-1984 kernel call -> UQ -> PASS/FAIL/WITHHELD -> ASME chain",
        passed,
        payload,
        artifacts=["artifacts/in145_n08/evidence/c6_end_to_end.json"],
        notes=[
            "the kernel source files are hashed before and after the run: the "
            "chain must not modify the solver",
            "the WITHHELD branch is exercised by presenting a claim with no UQ "
            "record at all",
        ],
    )


# ---------------------------------------------------------------------------
# C7: interval calibration
# ---------------------------------------------------------------------------
def criterion_c7(evidence_dir: pathlib.Path, quick: bool) -> dict[str, Any]:
    reps = 12 if quick else 20
    calibrated = coverage_study(
        ishigami_spec(), ishigami, ishigami_analytic(),
        n_samples=256, degree=8, n_boot=200, replications=reps,
    )
    starved = coverage_study(
        ishigami_spec(), ishigami, ishigami_analytic(),
        n_samples=128, degree=8, n_boot=100, replications=max(8, reps // 2),
    )
    payload = {
        "calibrated_regime": calibrated.as_dict(),
        "starved_regime": starved.as_dict(),
        "calibrated_at_n256": calibrated.calibrated,
        "under_covered_at_n128": bool(not starved.calibrated),
    }
    (evidence_dir / "c7_ci_coverage.json").write_text(json.dumps(_jsonable(payload), indent=1))
    passed = bool(calibrated.calibrated and not starved.calibrated)
    return _criterion(
        "C7",
        "95 % bootstrap intervals cover the analytic indices at N=256, and the "
        "starved regime is detected at N=128",
        passed,
        payload,
        artifacts=["artifacts/in145_n08/evidence/c7_ci_coverage.json"],
        notes=[
            "with 12-20 replications only gross under-coverage is detectable; the "
            "starved regime is included as the positive control of the detector",
        ],
    )


# ---------------------------------------------------------------------------
# C8: shift invariance of the sampled estimators
# ---------------------------------------------------------------------------
def criterion_c8(evidence_dir: pathlib.Path) -> dict[str, Any]:
    spec = ishigami_spec()

    def evaluate(shift: float):
        def fn(unit: np.ndarray) -> np.ndarray:
            physical, _ = spec.transform(unit)
            return shift + ishigami(physical)

        return fn

    base = saltelli_indices(evaluate(0.0), 3, n_base=1 << 14, seed=20240, second_order=True)
    shifted = saltelli_indices(evaluate(1.0e4), 3, n_base=1 << 14, seed=20240, second_order=True)
    diffs = {
        "first": float(np.max(np.abs(base.first - shifted.first))),
        "first_saltelli": float(np.max(np.abs(base.first_saltelli - shifted.first_saltelli))),
        "total": float(np.max(np.abs(base.total - shifted.total))),
        "second": float(max(abs(base.second[k] - shifted.second[k]) for k in base.second)),
    }
    analytic = ishigami_analytic()
    rel_second = abs(shifted.second[(0, 2)] - analytic["second"][(0, 2)]) / analytic["second"][(0, 2)]
    payload = {
        "shift": 1.0e4,
        "max_abs_difference": diffs,
        "shift_invariant": bool(all(v <= 1.0e-9 for v in diffs.values())),
        "shifted_first": shifted.first.tolist(),
        "shifted_first_saltelli": shifted.first_saltelli.tolist(),
        "shifted_total": shifted.total.tolist(),
        "shifted_second": {f"{k[0]}-{k[1]}": v for k, v in shifted.second.items()},
        "analytic_second_x1x3": analytic["second"][(0, 2)],
        "relative_error_second_x1x3": rel_second,
        "n_base": base.n_base,
    }
    (evidence_dir / "c8_shift_invariance.json").write_text(json.dumps(_jsonable(payload), indent=1))
    return _criterion(
        "C8",
        "sampled estimators are shift-invariant (f and f + 1e4 give identical indices)",
        payload["shift_invariant"] and rel_second <= 0.05,
        payload,
        artifacts=["artifacts/in145_n08/evidence/c8_shift_invariance.json"],
        notes=[
            "regression of the defect fixed in this node: the Saltelli first-order "
            "and closed second-order estimators multiplied by a raw model value, so "
            "an output offset of 1e4 moved the second-order index to 12.38 (exact "
            "value 0.2437).  Ship resistance is O(1e3) kN, i.e. exactly this regime",
        ],
    )


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _figures(records: dict[str, dict[str, Any]], fig_dir: pathlib.Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    made: list[str] = []

    # fig1: Ishigami indices, analytic vs PCE vs Monte-Carlo, first and total order
    c1 = records["C1"]["numbers"]
    names = ["x1", "x2", "x3"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), sharey=True)
    x = np.arange(3)
    for ax, kind, title in (
        (axes[0], "first", "first order S_i"),
        (axes[1], "total", "total order S_T,i"),
    ):
        analytic = c1[f"analytic_{kind}"]
        ax.bar(x - 0.27, analytic, 0.27, label="analytic", color="#444444")
        ax.bar(x, c1[f"pce_{kind}"], 0.27, label="PCE (coefficients)", color="#2b7bba")
        ax.bar(x + 0.27, c1[f"mc_{kind}"], 0.27, label="Saltelli/Jansen", color="#7fa8c9")
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Sobol' index")
    axes[0].legend(loc="upper right", fontsize=7)
    fig.suptitle(
        f"Ishigami: max |dS|(PCE) = {c1['pce_max_abs_error']:.2e} (gate 1e-3); "
        f"max rel err (MC) = {c1['mc_max_relative_error']:.2%} (gate 5 %)"
    )
    path = fig_dir / "fig1_ishigami_indices.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(str(path.relative_to(ROOT)))

    # fig2: refinement, both metrics, three axes
    c2 = records["C2"]["numbers"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    est = c2["estimation_axis"]
    axes[0].loglog(est["budgets"], est["eps_loo"], "o-", label="eps_LOO")
    axes[0].loglog(est["budgets"], est["holdout"], "s--", label="holdout (independent)")
    axes[0].set_xlabel("solver runs N (fixed total degree 4)")
    axes[0].set_ylabel("relative error")
    axes[0].set_title(
        f"fixed truncation: rho_LOO={est['spearman_loo']:.2f}, "
        f"rho_hold={est['spearman_holdout']:.2f}"
    )
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)
    ad = c2["adaptive_axis"]
    axes[1].loglog(ad["budgets"], ad["eps_loo"], "o-", label="eps_LOO (adaptive set)")
    axes[1].loglog(ad["budgets"], ad["holdout"], "s--", label="holdout (independent)")
    axes[1].set_xlabel("solver runs N (fixed total degree 8, sparse selection)")
    axes[1].set_ylabel("relative error")
    axes[1].set_title(
        f"adaptive selection: rho_LOO={ad['spearman_loo']:.2f} (rising), "
        f"rho_hold={ad['spearman_holdout']:.2f}"
    )
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.3)
    path = fig_dir / "fig2_refinement.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(str(path.relative_to(ROOT)))

    # fig3: CI coverage with Wilson intervals
    c7 = records["C7"]["numbers"]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for offset, key, colour in (
        (-0.18, "calibrated_regime", "#2b7bba"),
        (0.18, "starved_regime", "#c0504d"),
    ):
        block = c7[key]
        entries = block["entries"]
        xs = np.arange(len(entries)) + offset
        cov = [e["coverage"] for e in entries]
        lo = [e["coverage"] - e["wilson95"][0] for e in entries]
        hi = [e["wilson95"][1] - e["coverage"] for e in entries]
        ax.bar(xs, cov, 0.34, color=colour, label=f"N={block['n_samples']} (pooled {block['pooled_coverage']:.2f})")
        ax.errorbar(xs, cov, yerr=[lo, hi], fmt="none", ecolor="#333", capsize=2, lw=0.8)
    ax.axhline(0.95, color="#333", ls=":", lw=1.0, label="nominal 0.95")
    ax.axhline(c7["calibrated_regime"]["coverage_floor"], color="orange", ls="--", lw=1.0, label="floor 0.80")
    ax.set_xticks(np.arange(len(c7["calibrated_regime"]["entries"])))
    ax.set_xticklabels([f"{e['kind']}:{e['label']}" for e in c7["calibrated_regime"]["entries"]], rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("coverage of the analytic index")
    ax.set_title("95 % bootstrap interval coverage: calibrated vs starved budget")
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)
    path = fig_dir / "fig3_ci_coverage.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(str(path.relative_to(ROOT)))

    # fig4: gate map -- band relative width vs the 15 % threshold
    c6 = records["C6"]["numbers"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    names = ["narrow inputs\n(+-1 %)", "wide inputs\n(+-4 % T, Cb)"]
    widths = [c6["narrow"]["band_relative_width"], c6["wide"]["band_relative_width"]]
    colours = ["#4f9d69", "#c0504d"]
    bars = ax.bar(names, widths, color=colours, width=0.45)
    ax.axhline(0.15, color="#333", ls="--", lw=1.2, label="engineering threshold 15 %")
    for bar, width in zip(bars, widths):
        ax.text(bar.get_x() + bar.get_width() / 2, width, f"{width:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("95 % band / mean  [R_T]")
    ax.set_title(
        f"Holtrop-1984 R_T gate: PASS / FAIL ({c6['narrow']['status']} / {c6['wide']['status']})"
    )
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    path = fig_dir / "fig4_gate_map.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(str(path.relative_to(ROOT)))

    # fig5: injection matrix
    c4 = records["C4"]["numbers"]
    cases = c4["cases"]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    colour_of = {WITHHELD: "#d9a441", FAIL: "#c0504d", PASS: "#4f9d69"}
    ys = np.arange(len(cases))
    ax.barh(ys, [1] * len(cases), color=[colour_of[c["observed_status"]] for c in cases])
    for y, case in zip(ys, cases):
        ax.text(0.01, y, f"{case['case_id']}: {case['observed_status']}", va="center", fontsize=7.5, color="black")
    ax.set_yticks([])
    ax.set_xlim(0, 1.6)
    ax.set_xlabel("injected defect (green = control PASS, amber = WITHHELD, red = FAIL)")
    ax.set_title(f"{len(cases)} injections, none passes the gate (control: {c4['control_status']})")
    ax.invert_yaxis()
    path = fig_dir / "fig5_injection_matrix.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(str(path.relative_to(ROOT)))

    return made


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=OUT_DIR)
    parser.add_argument("--quick", action="store_true", help="reduce the coverage study")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--determinism-probe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.determinism_probe:  # pragma: no cover - used by C3 via subprocess
        payload = run_uq_study(
            ishigami, ishigami_spec(), n_samples=512, degree=8, n_boot=0,
            n_base_mc=1 << 12, bootstrap_mc=0, budget_n_mc=4000,
        )
        text = json.dumps(payload.pce_sobol.as_dict(ishigami_spec().names), sort_keys=True, separators=(",", ":"))
        print(json.dumps({
            "sample_hash": payload.samples.hash,
            "indices_digest": hashlib.sha256(text.encode()).hexdigest(),
            "status": payload.status,
        }))
        return 0

    out = args.out
    evidence_dir = out / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    records: dict[str, dict[str, Any]] = {}

    def run(cid: str, fn, *fnargs) -> None:
        t0 = time.perf_counter()
        rec = fn(*fnargs)
        rec["seconds"] = float(np.round(time.perf_counter() - t0, 2))
        records[cid] = rec
        print(f"[{cid}] {'PASS' if rec['passed'] else 'FAIL'}  ({rec['seconds']:.1f}s)  {rec['title']}",
              flush=True)

    t0 = time.perf_counter()
    c1, c5, _study = criterion_c1_c5(evidence_dir)
    elapsed = float(np.round(time.perf_counter() - t0, 2))
    records["C1"] = {**c1, "seconds": elapsed}
    print(f"[C1] {'PASS' if c1['passed'] else 'FAIL'}  {c1['title']}", flush=True)
    records["C5"] = {**c5, "seconds": 0.0}  # measured together with C1
    print(f"[C5] {'PASS' if c5['passed'] else 'FAIL'}  {c5['title']}", flush=True)

    run("C2", criterion_c2, evidence_dir, args.quick)
    run("C3", criterion_c3, evidence_dir)
    run("C4", criterion_c4, evidence_dir)
    run("C6", criterion_c6, evidence_dir)
    run("C7", criterion_c7, evidence_dir, args.quick)
    run("C8", criterion_c8, evidence_dir)

    figures: list[str] = []
    if not args.skip_figures:
        try:
            figures = _figures(records, out / "figures")
        except Exception as exc:  # pragma: no cover - figures are cosmetic
            print(f"figure rendering failed: {type(exc).__name__}: {exc}", flush=True)

    order = ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8")
    criteria = [records[k] for k in order if k in records]
    all_passed = all(rec["passed"] for rec in criteria)
    summary = {
        "mission": "IN145-N08",
        "node": "不确定性预算成熟化：非侵入式 PCE + Sobol 方差分解作为出证前验收门",
        "repository": "TensorPhysics",
        "package": "tensorphysics.uq",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "git": _git_info(),
        "command": "python scripts/in145_n08_uq_acceptance.py" + (" --quick" if args.quick else ""),
        "criteria": criteria,
        "all_passed": bool(all_passed),
        "figures": figures,
        "seconds": float(np.round(time.perf_counter() - t_start, 1)),
    }
    path = out / "acceptance.json"
    path.write_text(json.dumps(_jsonable(summary), indent=1))

    print("\n=== IN145-N08 acceptance ===", flush=True)
    for rec in criteria:
        print(f"{rec['id']}: {'PASS' if rec['passed'] else 'FAIL'}  {rec['title']}", flush=True)
    print(f"all_passed = {all_passed}", flush=True)
    print(f"evidence -> {path.relative_to(ROOT)}", flush=True)
    if figures:
        print(f"figures  -> {figures}", flush=True)
    return 0 if all_passed else 1


def _git_info() -> dict[str, str]:
    info: dict[str, str] = {}
    for key, command in (
        ("commit", ["git", "rev-parse", "HEAD"]),
        ("branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    ):
        try:
            out = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
            info[key] = out.stdout.strip()
        except Exception:  # pragma: no cover - git may be absent
            info[key] = "unknown"
    return info


if __name__ == "__main__":
    raise SystemExit(main())