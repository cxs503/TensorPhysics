"""Scaling-harness tests: grid sizing, aggregation arithmetic and JSON schema."""

from __future__ import annotations

import numpy as np

from tensorphysics.fem_parallel.scaling import (
    aggregate,
    grid_for_dofs,
    iteration_study,
    run_point,
    strong_scaling,
    weak_scaling,
)


def test_grid_for_dofs_hits_target_size():
    n = grid_for_dofs(1_000_000)
    assert n == (1000, 1000)
    n3 = grid_for_dofs(1_000_000, ndim=3)
    assert abs(int(np.prod(n3)) - 1_000_000) / 1_000_000 < 0.05
    n_aniso = grid_for_dofs(200_000, ndim=2, aspect=(4.0, 1.0))
    assert n_aniso[0] > 2 * n_aniso[1]


def test_aggregate_uses_slowest_rank_and_load_balance():
    ranks = [
        {"rank": 0, "t_assemble": 1.0, "t_setup": 0.5, "t_solve": 2.0, "t_comm": 0.5,
         "t_matvec_local": 1.0, "n_owned": 100, "iterations": 10, "converged": True,
         "n_global": 200, "nnz_local": 500},
        {"rank": 1, "t_assemble": 1.2, "t_setup": 0.5, "t_solve": 2.5, "t_comm": 0.6,
         "t_matvec_local": 1.2, "n_owned": 100, "iterations": 10, "converged": True,
         "n_global": 200, "nnz_local": 500},
    ]
    agg = aggregate(ranks)
    assert agg["t_solve"] == 2.5
    assert agg["t_total"] == 1.2 + 0.5 + 2.5
    assert agg["comm_fraction"] == 0.6 / 2.5
    assert 0.0 <= agg["load_balance_std_over_mean"] < 0.2
    assert agg["iterations"] == 10 and agg["converged"] is True


def test_run_point_reports_full_breakdown_serially():
    pt = run_point(n=(33, 31), nranks=1, tol=1e-6, maxiter=2000)
    for key in ("t_assemble", "t_setup", "t_solve", "t_total", "t_comm",
                "comm_fraction", "iterations", "load_balance_std_over_mean",
                "n_owned_min", "per_rank"):
        assert key in pt
    assert pt["t_solve"] > 0.0 and pt["converged"]
    assert pt["n_owned_min"] == 33 * 31
    assert pt["iterations"] > 5


def test_strong_scaling_records_speedup_and_efficiency():
    pts = strong_scaling(n=(49, 47), ranks=(1, 2), tol=1e-6, maxiter=2000)
    assert [p["nranks"] for p in pts] == [1, 2]
    base = pts[0]["t_solve"]
    assert abs(pts[0]["speedup"] - 1.0) < 1e-12
    for p in pts:
        assert p["speedup"] > 0.0
        assert abs(p["efficiency"] - p["speedup"] / p["nranks"]) < 1e-12
        assert p["speedup"] == base / p["t_solve"]
    assert pts[1]["iterations"] == pts[0]["iterations"]  # same tolerance, same system


def test_weak_scaling_keeps_work_per_rank_constant():
    pts = weak_scaling(n_per_rank=4000, ranks=(1, 2), tol=1e-6, maxiter=2000)
    assert [p["nranks"] for p in pts] == [1, 2]
    per_rank = [int(np.prod(p["grid"])) / p["nranks"] for p in pts]
    assert per_rank[1] > 0.6 * per_rank[0]
    for p in pts:
        assert "weak_ratio" in p and p["weak_ratio"] > 0.0


def test_iteration_study_reports_reduction_against_plain_cg():
    pts = iteration_study(n=(49, 47), nranks=1, variants=("none", "jacobi", "asm"),
                          tol=1e-6, maxiter=4000)
    by = {p["variant"]: p for p in pts}
    assert by["none"]["iteration_reduction"] == 0.0
    assert by["jacobi"]["iterations"] <= by["none"]["iterations"]
    assert by["asm"]["iteration_reduction"] >= 0.5
