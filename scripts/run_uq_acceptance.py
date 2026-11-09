#!/usr/bin/env python3
"""Acceptance run of the uncertainty-budget contract (mission IN144-N11).

Executes every machine-checkable acceptance criterion of the node, writes
``artifacts/uq/acceptance_evidence.json`` (all numbers, no hand-copied digits)
and renders ``artifacts/uq/figures/*.png`` for the report.

Criteria
--------
A1  Ishigami PCE indices vs closed form: |dS| <= 1e-3 (S1, S2, S13, ST1, ST3)
A2  Ishigami Monte-Carlo (Saltelli/Jansen) indices: relative error <= 5 %
A3  eps_LOO decreases with the PCE degree (>= 4 grid points, Spearman <= -0.9)
A4  eps_LOO decreases with the sample count at fixed truncation (same test)
A5  same spec + rule + seed -> bit-identical sample hash and Sobol' indices
A6  NaN sample / too few samples / eps_LOO > 0.1 -> WITHHELD with reason codes
A7  PCE (analytic) vs Saltelli indices: relative deviation <= 5 %
A8  no solver kernel is touched: kernel file hashes unchanged, workflow module
    free of solver imports, cluster shards merge bit-identically by hash
A9  Sobol' G-function family regression: |dS| <= 5e-3 (d = 4)
A10 ship benchmark: narrow inputs -> PASS, wide inputs -> FAIL at 15 %

Usage::

    python scripts/run_uq_acceptance.py [--out artifacts/uq]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scipy.stats import spearmanr  # noqa: E402

import tensorphysics.uq as uq  # noqa: E402
from tensorphysics.uq import (  # noqa: E402
    FAIL,
    PASS,
    WITHHELD,
    fanout_evaluate,
    fit_pce,
    holtrop_type_benchmark,
    ishigami,
    ishigami_analytic,
    ishigami_spec,
    run_uq_study,
    sample,
    sobol_g,
    sobol_g_analytic,
    sobol_g_spec,
)

KERNEL_FILES = (
    "src/tensorphysics/physics_kernels_holtrop1984.py",
    "src/tensorphysics/physics_kernels_resistance.py",
)


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _round(value: float, digits: int = 6) -> float:
    return float(np.round(float(value), digits))


def _grid_note(values: Sequence[float]) -> dict[str, Any]:
    return {"values": [_round(v) for v in values]}


def _ci95(ci: tuple[np.ndarray, np.ndarray] | None, index: int) -> list[float] | None:
    """``[low, high]`` of the ``index``-th component of a percentile interval."""
    if ci is None:
        return None
    lo, hi = ci
    return [_round(lo[index]), _round(hi[index])]


# ---------------------------------------------------------------------------
# criteria
# ---------------------------------------------------------------------------
def criterion_a1_a2_a7(tmp: pathlib.Path) -> dict[str, Any]:
    """Ishigami: chaos vs analytic, Monte-Carlo vs analytic, chaos vs Monte-Carlo."""
    spec = ishigami_spec()
    ref = ishigami_analytic()
    study = run_uq_study(
        ishigami,
        spec,
        n_samples=1024,
        degree=8,
        n_boot=200,
        bootstrap_seed=777,
        n_base_mc=1 << 16,
        mc_seed=5150,
        bootstrap_mc=100,
        budget_n_mc=20000,
    )
    pce = study.pce_sobol
    mc = study.mc
    assert pce is not None and mc is not None

    labels = {
        "first": ["x1", "x2", "x3"],
        "total": ["x1", "x2", "x3"],
    }
    pce_table: dict[str, Any] = {}
    for kind, names in labels.items():
        pce_vals = getattr(pce, "first" if kind == "first" else "total")
        mc_vals = getattr(mc, "first" if kind == "first" else "total")
        ex_vals = ref["first"] if kind == "first" else ref["total"]
        pce_table[kind] = {
            name: {
                "analytic": _round(ex_vals[i]),
                "pce": _round(pce_vals[i]),
                "abs_error": _round(abs(pce_vals[i] - ex_vals[i])),
                "mc_jansen": _round(mc_vals[i]),
                "mc_saltelli": _round(
                    (mc.first_saltelli if kind == "first" else mc.total)[i]
                ),
                "mc_rel_error": _round(
                    abs(mc_vals[i] - ex_vals[i]) / ex_vals[i]
                    if ex_vals[i] != 0.0
                    else abs(mc_vals[i] - ex_vals[i])
                ),
            }
            for i, name in enumerate(names)
        }
    pce_table["second"] = {
        "x1*x3": {
            "analytic": _round(ref["second"][(0, 2)]),
            "pce": _round(pce.second[(0, 2)]),
            "abs_error": _round(abs(pce.second[(0, 2)] - ref["second"][(0, 2)])),
            "mc": _round(mc.second[(0, 2)]),
            "mc_rel_error": _round(
                abs(mc.second[(0, 2)] - ref["second"][(0, 2)]) / ref["second"][(0, 2)]
            ),
        },
        "x1*x2": {
            "analytic": 0.0,
            "pce": _round(pce.second[(0, 1)]),
            "abs_error": _round(abs(pce.second[(0, 1)])),
            "mc": _round(mc.second[(0, 1)]),
        },
        "x2*x3": {
            "analytic": 0.0,
            "pce": _round(pce.second[(1, 2)]),
            "abs_error": _round(abs(pce.second[(1, 2)])),
            "mc": _round(mc.second[(1, 2)]),
        },
    }

    worst_pce = max(
        abs(pce.first[0] - ref["first"][0]),
        abs(pce.first[1] - ref["first"][1]),
        abs(pce.total[0] - ref["total"][0]),
        abs(pce.total[2] - ref["total"][2]),
        abs(pce.second[(0, 2)] - ref["second"][(0, 2)]),
    )
    worst_mc = max(
        abs(mc.first[0] - ref["first"][0]) / ref["first"][0],
        abs(mc.first[1] - ref["first"][1]) / ref["first"][1],
        abs(mc.total[0] - ref["total"][0]) / ref["total"][0],
        abs(mc.total[1] - ref["total"][1]) / ref["total"][1],
        abs(mc.total[2] - ref["total"][2]) / ref["total"][2],
    )
    quoted = ref["quoted"]
    pce_table["quoted_reference_check"] = {
        "S1_quoted": quoted["S1"],
        "S1_pce_minus_quoted": _round(pce.first[0] - quoted["S1"]),
        "S2_quoted": quoted["S2"],
        "S2_pce_minus_quoted": _round(pce.first[1] - quoted["S2"]),
        "ST_quoted": quoted["ST"],
        "ST_pce_minus_quoted": _round(pce.total[0] - quoted["ST"]),
        "note": (
            "the closed-form reference gives S1=0.313905, S2=0.442411, ST=0.557589; "
            "the 4-decimal values quoted in the campaign node differ by < 2e-4"
        ),
    }
    payload = {
        "study": study.as_dict(),
        "indices": pce_table,
        "worst_pce_abs_error": _round(worst_pce),
        "worst_mc_rel_error": _round(worst_mc),
        "surrogate": {
            "eps_loo": _round(study.surrogate.eps_loo),
            "eps_loo_abs": _round(study.surrogate.eps_loo_abs),
            "r2": _round(study.surrogate.r2),
            "n_terms": study.surrogate.n_terms,
            "n_samples": study.surrogate.n_samples,
        },
        "pce_first_ci95": {
            n: _ci95(study.pce_sobol.first_ci, k) for k, n in ((0, "x1"), (1, "x2"), (2, "x3"))
        },
        "pce_total_ci95": {
            n: _ci95(study.pce_sobol.total_ci, k) for k, n in ((0, "x1"), (1, "x2"), (2, "x3"))
        },
        "mc": {
            "n_base": mc.n_base,
            "n_model_evaluations": mc.n_model_evaluations,
            "first_ci95": {n: _ci95(mc.first_ci, k) for k, n in ((0, "x1"), (1, "x2"), (2, "x3"))},
            "total_ci95": {n: _ci95(mc.total_ci, k) for k, n in ((0, "x1"), (1, "x2"), (2, "x3"))},
        },
    }
    (tmp / "ishigami_study.json").write_text(json.dumps(payload, indent=1))
    return {
        "A1_pce_vs_analytic": {
            "description": "|S_pce - S_exact| <= 1e-3 for S1, S2, S13, ST1, ST3",
            "measured": _round(worst_pce),
            "limit": 1e-3,
            "pass": bool(worst_pce <= 1.0e-3),
            "per_index": pce_table,
        },
        "A2_mc_vs_analytic": {
            "description": "Monte-Carlo (Saltelli/Jansen) relative error <= 5 %",
            "measured": _round(worst_mc),
            "limit": 0.05,
            "pass": bool(worst_mc <= 0.05),
            "n_base": mc.n_base,
        },
        "A7_pce_vs_saltelli": {
            "description": "PCE analytic vs Saltelli relative deviation <= 5 %",
            "measured": _round(study.cross_check.max_relative_deviation),
            "limit": 0.05,
            "pass": bool(study.cross_check.agreement and study.cross_check.max_relative_deviation <= 0.05),
            "cross_check": study.cross_check.as_dict(),
        },
        "artifacts": {"ishigami_study": str((tmp / "ishigami_study.json").relative_to(ROOT))},
        "_study": study,
    }


def criterion_a3_a4() -> dict[str, Any]:
    """LOO convergence with the polynomial degree and with the sample count."""
    spec = ishigami_spec()
    degrees = [2, 3, 4, 5, 6, 8, 10]
    design = sample(spec, 1024, seed=20240)
    coords = spec.from_latent(design.latent)[1]
    values = ishigami(design.points)
    loo_degree = [
        fit_pce(coords, values, degree=p, families=spec.families).eps_loo for p in degrees
    ]
    rho_degree = float(spearmanr(degrees, loo_degree).statistic)

    sizes = [512, 1024, 2048, 4096]
    loo_size = []
    for n in sizes:
        design_n = sample(spec, n, seed=20240)
        coords_n = spec.from_latent(design_n.latent)[1]
        pce = fit_pce(
            coords_n,
            ishigami(design_n.points),
            degree=8,
            families=spec.families,
            method="ols",  # truncation held fixed -> LOO measures estimation error only
        )
        loo_size.append(pce.eps_loo)
    rho_size = float(spearmanr(sizes, loo_size).statistic)

    return {
        "A3_loo_vs_degree": {
            "description": "eps_LOO decreases with PCE degree (>=4 points, Spearman <= -0.9)",
            "grid": degrees,
            "eps_loo": [_round(v) for v in loo_degree],
            "spearman": _round(rho_degree),
            "limit": -0.9,
            "n_points": len(degrees),
            "pass": bool(len(degrees) >= 4 and rho_degree <= -0.9),
            "method": "lars",
            "n_samples": 1024,
        },
        "A4_loo_vs_samples": {
            "description": (
                "eps_LOO decreases with the sample count at fixed truncation "
                "(>=4 points, Spearman <= -0.9)"
            ),
            "grid": sizes,
            "eps_loo": [_round(v) for v in loo_size],
            "spearman": _round(rho_size),
            "limit": -0.9,
            "n_points": len(sizes),
            "pass": bool(len(sizes) >= 4 and rho_size <= -0.9),
            "method": "ols",
            "degree": 8,
            "note": (
                "the truncation is held fixed (full total-degree basis) so that LOO "
                "measures estimation error; with the greedy selector the active set "
                "adapts to N and the two effects would be mixed"
            ),
        },
    }


def criterion_a5(tmp: pathlib.Path) -> dict[str, Any]:
    """Determinism: identical spec/rule/seed -> bit-identical indices."""
    kwargs = dict(
        n_samples=1024,
        degree=8,
        n_boot=0,
        n_base_mc=1 << 12,
        bootstrap_mc=0,
        budget_n_mc=4000,
        require_cross_check=False,
    )
    a = run_uq_study(ishigami, ishigami_spec(), **kwargs)
    b = run_uq_study(ishigami, ishigami_spec(), **kwargs)
    same_hash = a.samples.hash == b.samples.hash
    same_first = bool(np.array_equal(a.pce_sobol.first, b.pce_sobol.first))
    same_total = bool(np.array_equal(a.pce_sobol.total, b.pce_sobol.total))
    same_mc = bool(np.array_equal(a.mc.first, b.mc.first))
    same_budget = a.budget.as_dict() == b.budget.as_dict()
    payload = {
        "sample_hash": a.samples.hash,
        "spec_hash": a.spec.fingerprint(),
        "n_samples": a.samples.n,
        "rule": a.samples.rule,
        "seed": a.samples.seed,
        "second_run_sample_hash": b.samples.hash,
        "identical_first_order": same_first,
        "identical_total_order": same_total,
        "identical_mc": same_mc,
        "identical_budget": same_budget,
        "first_order": [_round(v) for v in a.pce_sobol.first],
        "total_order": [_round(v) for v in a.pce_sobol.total],
    }
    (tmp / "determinism.json").write_text(json.dumps(payload, indent=1))
    return {
        "description": "same spec+rule+seed -> bit-identical sample hash and Sobol indices",
        "pass": bool(same_hash and same_first and same_total and same_mc and same_budget),
        "measured": payload,
        "artifact": str((tmp / "determinism.json").relative_to(ROOT)),
    }


def criterion_a6() -> dict[str, Any]:
    """WITHHELD branches: NaN sample, too few samples, poor surrogate."""
    # 1. NaN injected into the solver output
    def poisoned(points: np.ndarray) -> np.ndarray:
        values = ishigami(points)
        values[[7, 19]] = np.nan
        return values

    nan_study = run_uq_study(
        poisoned,
        ishigami_spec(),
        n_samples=512,
        degree=6,
        n_boot=0,
        n_base_mc=1 << 10,
        bootstrap_mc=0,
        budget_n_mc=4000,
    )
    # 2. sample count below the floor
    few_study = run_uq_study(
        ishigami, ishigami_spec(), n_samples=16, degree=4, run_mc=False, budget_n_mc=2000
    )
    # 3. surrogate LOO above the floor
    poor_study = run_uq_study(
        ishigami, ishigami_spec(), n_samples=256, degree=2, run_mc=False, budget_n_mc=2000
    )
    cases = {
        "nan_sample": {
            "status": nan_study.status,
            "reason_codes": list(nan_study.verdict.reason_codes),
            "invalid_indices": list(nan_study.results.invalid_indices),
        },
        "too_few_samples": {
            "status": few_study.status,
            "reason_codes": list(few_study.verdict.reason_codes),
            "n_valid": few_study.results.n_valid,
        },
        "eps_loo_exceeded": {
            "status": poor_study.status,
            "reason_codes": list(poor_study.verdict.reason_codes),
            "eps_loo": _round(poor_study.surrogate.eps_loo),
        },
    }
    ok = (
        cases["nan_sample"]["status"] == WITHHELD
        and "UQ-W01-NONFINITE" in cases["nan_sample"]["reason_codes"]
        and cases["too_few_samples"]["status"] == WITHHELD
        and "UQ-W02-TOOFEW" in cases["too_few_samples"]["reason_codes"]
        and cases["eps_loo_exceeded"]["status"] == WITHHELD
        and "UQ-W03-LOO" in cases["eps_loo_exceeded"]["reason_codes"]
    )
    return {
        "description": "NaN sample / sample count below floor / eps_LOO > 0.1 -> WITHHELD + reason codes",
        "cases": cases,
        "eps_loo_limit": 0.1,
        "min_samples_floor": uq.min_samples_floor(3),
        "pass": bool(ok),
    }


def criterion_a8(tmp: pathlib.Path) -> dict[str, Any]:
    """Non-intrusiveness: kernels untouched, workflow solver-free, shards hash-pinned."""
    before = {path: _sha256(ROOT / path) for path in KERNEL_FILES}
    workflow_source = (ROOT / "src/tensorphysics/uq/workflow.py").read_text()
    forbidden = [token for token in ("physics_kernels", "holtrop1984", "resistance") if token in workflow_source]

    # cluster-style shard dispatch + hash-pinned merge (two subprocesses)
    bench = holtrop_type_benchmark()
    design = sample(bench.spec, 128, seed=20240)
    shard_dir = tmp / "shards"
    shard_dir.mkdir(exist_ok=True)
    shard_logs = []
    for index in range(2):
        out = shard_dir / f"shard_{index}.npz"
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/uq_shard_worker.py"),
                "--benchmark", "holtrop",
                "--n", str(design.n),
                "--seed", str(design.seed),
                "--shard", f"{index}/2",
                "--out", str(out),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        shard_logs.append(completed.stdout.strip())
    merged_path = shard_dir / "merged.npz"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/uq_shard_worker.py"),
            "--merge",
            str(shard_dir / "shard_0.npz"),
            str(shard_dir / "shard_1.npz"),
            "--out", str(merged_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    merged = np.load(merged_path)
    reference = fanout_evaluate(uq.CallableAdapter(bench.resistance_kn), design)
    shard_identical = bool(np.array_equal(merged["values"], reference.values))
    shard_hash_ok = str(merged["sample_hash"]) == design.hash

    # a stale shard must be refused by the merge
    stale = shard_dir / "stale.npz"
    np.savez(
        stale,
        values=np.zeros(64),
        indices=np.arange(64),
        sample_hash=np.array("0" * 64),
        spec_hash=np.array(design.spec.fingerprint()),
        n=np.array(design.n),
    )
    stale = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/uq_shard_worker.py"),
            "--merge",
            str(shard_dir / "shard_0.npz"),
            str(stale),
            "--out", str(shard_dir / "bad.npz"),
        ],
        capture_output=True,
        text=True,
    )
    after = {path: _sha256(ROOT / path) for path in KERNEL_FILES}
    kernels_untouched = before == after
    payload = {
        "kernel_sha256": after,
        "kernel_hashes_unchanged": kernels_untouched,
        "workflow_forbidden_tokens": forbidden,
        "cluster_command": uq.cluster_shard_command(
            repo="TensorPhysics", script="scripts/uq_shard_worker.py"
        ),
        "shard_outputs": shard_logs,
        "merged_sample_hash": str(merged["sample_hash"]),
        "design_sample_hash": design.hash,
        "shard_values_bitwise_equal_inprocess": shard_identical,
        "shard_hash_matches_design": shard_hash_ok,
        "stale_shard_merge_exit_code": stale.returncode,
        "stale_shard_error": stale.stderr.strip()[:400],
    }
    (tmp / "sharding.json").write_text(json.dumps(payload, indent=1))
    return {
        "description": (
            "solver kernels byte-identical after the run, UQ workflow contains no solver "
            "import, cluster shards merge bit-identically and a stale shard is refused"
        ),
        "pass": bool(
            kernels_untouched
            and not forbidden
            and shard_identical
            and shard_hash_ok
            and stale.returncode != 0
        ),
        "measured": payload,
        "artifact": str((tmp / "sharding.json").relative_to(ROOT)),
    }


def criterion_a9() -> dict[str, Any]:
    """G-function family regression (extensible dimension / interaction strength)."""
    cases = []
    for a in ([0.5, 1.0, 1.5, 2.0], [0.0, 0.5, 1.0, 1.5]):
        av = np.array(a)
        ref = sobol_g_analytic(av)
        study = run_uq_study(
            lambda points, av=av: sobol_g(points, av),
            sobol_g_spec(4),
            n_samples=2048,
            degree=6,
            n_boot=0,
            n_base_mc=1 << 14,
            bootstrap_mc=100,
            budget_n_mc=8000,
        )
        cases.append(
            {
                "a": a,
                "eps_loo": _round(study.surrogate.eps_loo),
                "worst_first_abs_error": _round(float(np.max(np.abs(study.pce_sobol.first - ref["first"])))),
                "worst_total_abs_error": _round(float(np.max(np.abs(study.pce_sobol.total - ref["total"])))),
                "worst_second_abs_error": _round(
                    float(max(abs(study.pce_sobol.second[k] - ref["second"][k]) for k in ref["second"]))
                ),
                "cross_check": study.cross_check.as_dict(),
                "status": study.status,
                "pce_first": {n: _round(v) for n, v in zip(uq.sobol_g_spec(4).names, study.pce_sobol.first)},
                "analytic_first": {f"x{i + 1}": _round(v) for i, v in enumerate(ref["first"])},
            }
        )
    worst = max(max(c["worst_first_abs_error"], c["worst_total_abs_error"]) for c in cases)
    return {
        "description": "G-function family (d=4, two interaction regimes): |dS| <= 5e-3",
        "cases": cases,
        "measured": _round(worst),
        "limit": 5e-3,
        "pass": bool(worst <= 5.0e-3),
    }


def criterion_a10(tmp: pathlib.Path) -> dict[str, Any]:
    """Ship benchmark: PASS with narrow inputs, FAIL with wide inputs, at 15 %."""
    narrow = holtrop_type_benchmark()
    narrow_study = run_uq_study(
        narrow.resistance_kn,
        narrow.spec,
        n_samples=256,
        degree=4,
        n_boot=200,
        n_base_mc=1 << 15,
        bootstrap_mc=100,
        budget_n_mc=20000,
        direct_budget_n_mc=4096,
    )
    wide = holtrop_type_benchmark(rel_uncertainty=0.04)
    wide_study = run_uq_study(
        wide.resistance_kn,
        wide.spec,
        n_samples=256,
        degree=4,
        n_boot=0,
        run_mc=False,
        require_cross_check=False,
        budget_n_mc=20000,
        direct_budget_n_mc=4096,
    )
    names = narrow.spec.names
    payload = {
        "benchmark": {
            "kernel": "tensorphysics.physics_kernels_holtrop1984.holtrop1984_resistance",
            "reference_case": {k: _round(v, 4) for k, v in narrow.reference_case.items()},
            "spec": narrow.spec.as_dict(),
            "spec_hash": narrow.spec.fingerprint(),
            "sample_hash": narrow_study.samples.hash,
            "qoi": "calm-water total resistance R_T [kN]",
            "notes": narrow.notes,
        },
        "narrow": {
            "status": narrow_study.status,
            "threshold": narrow_study.verdict.threshold,
            "relative_width": _round(narrow_study.budget.relative_width),
            "budget": narrow_study.budget.as_dict(),
            "budget_direct": narrow_study.budget_direct.as_dict() if narrow_study.budget_direct else None,
            "surrogate": {
                "eps_loo": _round(narrow_study.surrogate.eps_loo),
                "eps_loo_abs": _round(narrow_study.surrogate.eps_loo_abs),
                "n_terms": narrow_study.surrogate.n_terms,
            },
            "sobol_first": {n: _round(v) for n, v in zip(names, narrow_study.pce_sobol.first)},
            "sobol_total": {n: _round(v) for n, v in zip(names, narrow_study.pce_sobol.total)},
            "pce_first_ci95": {n: _ci95(narrow_study.pce_sobol.first_ci, k) for k, n in enumerate(names)},
            "pce_total_ci95": {n: _ci95(narrow_study.pce_sobol.total_ci, k) for k, n in enumerate(names)},
            "mc_first": {n: _round(v) for n, v in zip(names, narrow_study.mc.first)} if narrow_study.mc else None,
            "cross_check": narrow_study.cross_check.as_dict() if narrow_study.cross_check else None,
            "verdict": narrow_study.verdict.as_dict(),
        },
        "wide": {
            "status": wide_study.status,
            "relative_width": _round(wide_study.budget.relative_width),
            "budget": wide_study.budget.as_dict(),
            "verdict": wide_study.verdict.as_dict(),
        },
        "study_narrow": narrow_study.as_dict(),
    }
    (tmp / "ship_benchmark.json").write_text(json.dumps(payload, indent=1))
    return {
        "description": "narrow inputs -> PASS, wide inputs -> FAIL against the 15 % threshold",
        "threshold": 0.15,
        "pass": bool(
            narrow_study.status == PASS
            and narrow_study.budget.relative_width <= 0.15
            and wide_study.status == FAIL
            and wide_study.budget.relative_width > 0.15
        ),
        "narrow_relative_width": _round(narrow_study.budget.relative_width),
        "wide_relative_width": _round(wide_study.budget.relative_width),
        "narrow_status": narrow_study.status,
        "wide_status": wide_study.status,
        "artifact": str((tmp / "ship_benchmark.json").relative_to(ROOT)),
        "_narrow": narrow_study,
        "_wide": wide_study,
        "_bench": narrow,
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_figures(outdir: pathlib.Path, results: dict[str, Any]) -> list[pathlib.Path]:
    try:
        import matplotlib
    except ImportError:  # compute nodes may have no plotting stack
        print("matplotlib unavailable: figures skipped (numbers are still in the JSON)")
        return []

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    paths: list[pathlib.Path] = []

    # --- figure 1: Ishigami indices, three routes ---------------------------
    a1 = results["A1_pce_vs_analytic"]["per_index"]
    ref = ishigami_analytic()
    labels = ["S1", "S2", "S13", "ST1", "ST2", "ST3"]
    analytic = [ref["first"][0], ref["first"][1], ref["second"][(0, 2)],
                ref["total"][0], ref["total"][1], ref["total"][2]]
    pce = [a1["first"]["x1"]["pce"], a1["first"]["x2"]["pce"], a1["second"]["x1*x3"]["pce"],
           a1["total"]["x1"]["pce"], a1["total"]["x2"]["pce"], a1["total"]["x3"]["pce"]]
    mc = [a1["first"]["x1"]["mc_jansen"], a1["first"]["x2"]["mc_jansen"], a1["second"]["x1*x3"]["mc"],
          a1["total"]["x1"]["mc_jansen"], a1["total"]["x2"]["mc_jansen"], a1["total"]["x3"]["mc_jansen"]]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=160)
    ax.bar(x - 0.26, analytic, 0.26, label="closed form", color="#2b2b2b")
    ax.bar(x, pce, 0.26, label="PCE (degree 8, N=1024)", color="#1f77b4")
    ax.bar(x + 0.26, mc, 0.26, label="Saltelli/Jansen (n_base=65536)", color="#ff7f0e", alpha=0.85)
    for i, (p, m) in enumerate(zip(pce, mc)):
        ax.annotate(f"{p:.4f}", (i, p), ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Sobol' index [-]")
    ax.set_title("Ishigami anchor: chaos (analytic) vs sampled indices")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = outdir / "fig1_ishigami_indices.png"
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)

    # --- figure 2: LOO convergence ------------------------------------------
    a3 = results["A3_loo_vs_degree"]
    a4 = results["A4_loo_vs_samples"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), dpi=160)
    axes[0].semilogy(a3["grid"], a3["eps_loo"], "o-", color="#1f77b4")
    axes[0].set_xlabel("PCE total degree p (N = 1024, LARS)")
    axes[0].set_ylabel(r"$\varepsilon_{LOO}$ / std(y)")
    axes[0].set_title(f"degree refinement (Spearman = {a3['spearman']:.2f})")
    axes[0].grid(alpha=0.3, which="both")
    axes[1].semilogy(a4["grid"], a4["eps_loo"], "s-", color="#d62728")
    axes[1].set_xlabel("training samples N (p = 8, OLS: truncation fixed)")
    axes[1].set_ylabel(r"$\varepsilon_{LOO}$ / std(y)")
    axes[1].set_title(f"sample refinement (Spearman = {a4['spearman']:.2f})")
    axes[1].grid(alpha=0.3, which="both")
    fig.suptitle("Leave-one-out error of the chaos surrogate is monotone in both budget directions")
    fig.tight_layout()
    path = outdir / "fig2_loo_convergence.png"
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)

    # --- figure 3: ship benchmark sensitivity + band ------------------------
    study = results["A10_ship_benchmark"]["_narrow"]
    bench = results["A10_ship_benchmark"]["_bench"]
    names = list(bench.spec.names)
    first = study.pce_sobol.first
    total = study.pce_sobol.total
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), dpi=160,
                             gridspec_kw={"width_ratios": [1.15, 1.0]})
    y = np.arange(len(names))
    axes[0].barh(y - 0.2, first, 0.4, label="first order $S_i$", color="#1f77b4")
    axes[0].barh(y + 0.2, total, 0.4, label="total order $S_{T,i}$", color="#ff7f0e")
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Sobol' index [-]")
    axes[0].set_title("Holtrop-1984 R_T: input ranking")
    axes[0].legend(fontsize=7)
    axes[0].grid(axis="x", alpha=0.25)
    axes[1].set_title("95 % band of R_T vs 15 % gate")
    band_lo, band_hi, mean = study.budget.lower95, study.budget.upper95, study.budget.mean
    axes[1].axhline(0.15, color="#d62728", ls="--", label="threshold 15 %")
    for scenario, value, color in (
        ("Ishigami\n(anchor)", results["_ishigami_study"].budget.relative_width, "#7f7f7f"),
        ("R_T narrow", study.budget.relative_width, "#2ca02c"),
        ("R_T wide", results["A10_ship_benchmark"]["_wide"].budget.relative_width, "#d62728"),
    ):
        axes[1].bar([scenario], [value], color=color, alpha=0.85)
        axes[1].annotate(f"{value:.3f}", (scenario, value), ha="center", va="bottom", fontsize=7)
    axes[1].set_ylabel(r"$U / |\bar{Q}|$  (95 % width / mean)")
    axes[1].set_ylim(0, max(0.3, 1.15 * results["A10_ship_benchmark"]["_wide"].budget.relative_width))
    axes[1].legend(fontsize=7)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle(
        f"ship UQ: R_T = {mean:.1f} kN, 95 % band [{band_lo:.1f}, {band_hi:.1f}] kN "
        f"(U/|mean| = {study.budget.relative_width:.3f})"
    )
    fig.tight_layout()
    path = outdir / "fig3_ship_sensitivity_band.png"
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)

    # --- figure 4: LOO evidence and the three-state gate --------------------
    fig, ax = plt.subplots(figsize=(7.0, 3.6), dpi=160)
    cases = [
        ("Ishigami\np=8, N=1024", results["_ishigami_study"].surrogate.eps_loo, WITHHELD),
        ("R_T narrow\np=4, N=256", study.surrogate.eps_loo, PASS),
        ("G-function\nd=4, p=6", results["A9_g_function"]["cases"][0]["eps_loo"], WITHHELD),
        ("R_T wide\np=4, N=256", results["A10_ship_benchmark"]["_wide"].surrogate.eps_loo, FAIL),
    ]
    colors = {"PASS": "#2ca02c", "FAIL": "#d62728", "WITHHELD": "#7f7f7f"}
    values = [c[1] for c in cases]
    ax.bar([c[0] for c in cases], values, color=[colors[c[2]] for c in cases])
    ax.axhline(0.10, color="#1f77b4", ls="--", label=r"$\varepsilon_{LOO}$ floor 0.10")
    ax.set_yscale("log")
    ax.set_ylabel(r"$\varepsilon_{LOO}$ / std(y)")
    ax.set_title("surrogate-quality evidence behind each verdict (bar colour = verdict)")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.25, which="both")
    fig.tight_layout()
    path = outdir / "fig4_gate_evidence.png"
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="artifacts/uq")
    args = parser.parse_args(argv)
    outdir = (ROOT / args.out).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    scratch = outdir / "evidence"
    scratch.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    results: dict[str, Any] = {}
    ishigami_section = criterion_a1_a2_a7(scratch)
    results["A1_pce_vs_analytic"] = ishigami_section["A1_pce_vs_analytic"]
    results["A2_mc_vs_analytic"] = ishigami_section["A2_mc_vs_analytic"]
    results["A7_pce_vs_saltelli"] = ishigami_section["A7_pce_vs_saltelli"]
    results["_ishigami_study"] = ishigami_section["_study"]
    results["_ishigami_artifacts"] = ishigami_section["artifacts"]
    loo_section = criterion_a3_a4()
    results["A3_loo_vs_degree"] = loo_section["A3_loo_vs_degree"]
    results["A4_loo_vs_samples"] = loo_section["A4_loo_vs_samples"]
    results["A5_determinism"] = criterion_a5(scratch)
    results["A6_withheld"] = criterion_a6()
    results["A8_noninvasive"] = criterion_a8(scratch)
    results["A9_g_function"] = criterion_a9()
    results["A10_ship_benchmark"] = criterion_a10(scratch)

    figures = make_figures(outdir / "figures", results)

    order = (
        "A1_pce_vs_analytic",
        "A2_mc_vs_analytic",
        "A3_loo_vs_degree",
        "A4_loo_vs_samples",
        "A5_determinism",
        "A6_withheld",
        "A7_pce_vs_saltelli",
        "A8_noninvasive",
        "A9_g_function",
        "A10_ship_benchmark",
    )
    criteria = [
        {
            "id": key,
            "description": results[key].get("description", ""),
            "pass": bool(results[key].get("pass")),
        }
        for key in order
    ]
    all_pass = all(c["pass"] for c in criteria)

    def _clean(obj: Any) -> Any:
        """Drop private study handles and make everything JSON-serialisable."""
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items() if not k.startswith("_")}
        if isinstance(obj, (list, tuple)):
            return [_clean(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    evidence = {
        "mission": "IN144-N11",
        "capability": "uncertainty budget: non-intrusive PCE + Sobol variance decomposition as a pre-evidence gate",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_s": _round(time.perf_counter() - t0, 3),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "package": uq.__name__,
        },
        "all_criteria_pass": all_pass,
        "criteria": criteria,
        "details": _clean(results),
        "artifacts": {
            "figures": [str(p.relative_to(ROOT)) for p in figures],
            "evidence_dir": str(scratch.relative_to(ROOT)),
            "documentation": "docs/UQ_UNCERTAINTY_BUDGET.md",
            "end_to_end_example": "examples/uq_end_to_end.py",
            "shard_worker": "scripts/uq_shard_worker.py",
            "evidence_json": str((outdir / "acceptance_evidence.json").relative_to(ROOT)),
        },
    }
    out = outdir / "acceptance_evidence.json"
    out.write_text(json.dumps(evidence, indent=2))
    print(json.dumps({c["id"]: c["pass"] for c in criteria}, indent=1))
    print(f"all_criteria_pass = {all_pass}")
    print(f"evidence -> {out.relative_to(ROOT)}")
    print(f"figures  -> {[str(p.relative_to(ROOT)) for p in figures]}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())