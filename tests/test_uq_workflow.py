"""End-to-end tests of the UQ workflow and its chaining with the V&V gate."""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from tensorphysics.uq import (
    CallableAdapter,
    FAIL,
    PASS,
    WITHHELD,
    holtrop_type_benchmark,
    ishigami,
    ishigami_analytic,
    ishigami_spec,
    run_uq_study,
    sobol_g,
    sobol_g_analytic,
    sobol_g_spec,
)


def _fast_ishigami(**overrides):
    kwargs = dict(
        n_samples=1024,
        degree=8,
        n_boot=0,
        n_base_mc=1 << 14,
        bootstrap_mc=0,
        budget_n_mc=5000,
    )
    kwargs.update(overrides)
    return run_uq_study(ishigami, ishigami_spec(), **kwargs)


def test_ishigami_indices_and_verdict() -> None:
    study = _fast_ishigami()
    ref = ishigami_analytic()
    ind = study.pce_sobol
    assert abs(ind.first[0] - ref["first"][0]) <= 1.0e-3
    assert abs(ind.total[0] - ref["total"][0]) <= 1.0e-3
    assert abs(ind.second[(0, 2)] - ref["second"][(0, 2)]) <= 1.0e-3
    # the Monte-Carlo route agrees within 5 %
    assert study.cross_check is not None and study.cross_check.agreement
    assert study.cross_check.max_relative_deviation <= 0.05
    # the Ishigami 95 % band is 4x its mean -> the engineering gate must FAIL,
    # but on the band alone (no evidence is withheld)
    assert study.status == FAIL
    assert study.verdict.blocking_codes == ()
    assert study.budget.relative_width > 1.0
    assert study.verdict.ratio > 1.0


def test_study_is_deterministic_and_serialisable() -> None:
    a = _fast_ishigami(n_base_mc=1 << 11)
    b = _fast_ishigami(n_base_mc=1 << 11)
    assert a.samples.hash == b.samples.hash
    assert np.array_equal(a.pce_sobol.first, b.pce_sobol.first)
    assert np.array_equal(a.pce_sobol.total, b.pce_sobol.total)
    assert np.array_equal(a.mc.first, b.mc.first)
    payload = json.loads(json.dumps(a.as_dict()))
    assert payload["status"] == a.status
    assert payload["samples"]["hash"] == a.samples.hash
    assert payload["surrogate"]["sobol"]["first"]["x1"] == pytest.approx(a.pce_sobol.first[0])
    assert payload["metadata"]["workflow_version"] == "tpuq-workflow-v1"


def test_nan_sample_withholds_the_verdict() -> None:
    def poisoned(points: np.ndarray) -> np.ndarray:
        values = ishigami(points)
        values[7] = np.nan
        return values

    study = run_uq_study(
        poisoned,
        ishigami_spec(),
        n_samples=512,
        degree=6,
        n_boot=0,
        n_base_mc=1 << 10,
        bootstrap_mc=0,
        budget_n_mc=3000,
    )
    assert study.status == WITHHELD
    assert "UQ-W01-NONFINITE" in study.verdict.reason_codes
    assert 7 in study.results.invalid_indices
    assert study.results.n_valid == 511


def test_too_few_samples_withholds_the_verdict() -> None:
    study = run_uq_study(
        ishigami,
        ishigami_spec(),
        n_samples=16,
        degree=4,
        run_mc=False,
        budget_n_mc=2000,
    )
    assert study.status == WITHHELD
    assert "UQ-W02-TOOFEW" in study.verdict.reason_codes


def test_poor_surrogate_withholds_the_verdict() -> None:
    study = run_uq_study(
        ishigami,
        ishigami_spec(),
        n_samples=256,
        degree=2,
        run_mc=False,
        budget_n_mc=2000,
    )
    assert study.surrogate.eps_loo > 0.1
    assert study.status == WITHHELD
    assert "UQ-W03-LOO" in study.verdict.reason_codes


def test_missing_cross_check_withholds_by_default() -> None:
    study = run_uq_study(
        ishigami, ishigami_spec(), n_samples=512, degree=4, run_mc=False, budget_n_mc=2000
    )
    assert study.status == WITHHELD
    assert "UQ-W07-NOMC" in study.verdict.reason_codes
    # disabling the cross-check explicitly is a *note*, not a silent pass
    assert "UQ-N04-NO-CROSSCHECK" in study.verdict.reason_codes
    relaxed = run_uq_study(
        ishigami,
        ishigami_spec(),
        n_samples=512,
        degree=4,
        run_mc=False,
        budget_n_mc=2000,
        require_cross_check=False,
    )
    assert "UQ-W07-NOMC" not in relaxed.verdict.reason_codes


def test_g_function_family_regression() -> None:
    a = np.array([0.5, 1.0, 1.5, 2.0])
    ref = sobol_g_analytic(a)
    study = run_uq_study(
        lambda pts: sobol_g(pts, a),
        sobol_g_spec(4),
        n_samples=2048,
        degree=6,
        n_boot=0,
        n_base_mc=1 << 14,
        bootstrap_mc=0,
        budget_n_mc=5000,
    )
    assert study.cross_check.agreement
    assert np.max(np.abs(study.pce_sobol.first - ref["first"])) <= 5.0e-3
    assert np.max(np.abs(study.pce_sobol.total - ref["total"])) <= 5.0e-3
    # a coarser truncation must trip the surrogate-quality gate
    coarse = run_uq_study(
        lambda pts: sobol_g(pts, a),
        sobol_g_spec(4),
        n_samples=512,
        degree=3,
        run_mc=False,
        budget_n_mc=3000,
    )
    assert coarse.surrogate.eps_loo > 0.1
    assert coarse.status == WITHHELD


def test_holtrop_benchmark_pass_and_fail_branches() -> None:
    bench = holtrop_type_benchmark()
    study = run_uq_study(
        bench.resistance_kn,
        bench.spec,
        n_samples=256,
        degree=4,
        n_boot=0,
        n_base_mc=1 << 13,
        bootstrap_mc=100,
        budget_n_mc=8000,
    )
    assert study.status == PASS
    assert 0.0 < study.budget.relative_width <= 0.15
    names = bench.spec.names
    dominant = names[int(np.argmax(study.pce_sobol.total))]
    assert dominant == "Vs"  # speed dominates calm-water resistance
    assert study.surrogate.eps_loo < 1.0e-3
    assert study.budget.mean == pytest.approx(bench.reference_case["RT_kN"], rel=0.02)

    wide = holtrop_type_benchmark(rel_uncertainty=0.04)
    wide_study = run_uq_study(
        wide.resistance_kn,
        wide.spec,
        n_samples=256,
        degree=4,
        run_mc=False,
        budget_n_mc=8000,
        require_cross_check=False,
        n_boot=0,
    )
    assert wide_study.status == FAIL
    assert wide_study.budget.relative_width > 0.15
    assert wide_study.verdict.blocking_codes == ()


def test_chaining_with_the_asme_validation_gate() -> None:
    """The UQ band becomes ``U_input`` of ``tensorphysics.vv.validation``."""
    from tensorphysics.vv.validation import validate

    bench = holtrop_type_benchmark()
    study = run_uq_study(
        bench.resistance_kn,
        bench.spec,
        n_samples=256,
        degree=4,
        n_boot=0,
        n_base_mc=1 << 11,
        bootstrap_mc=0,
        budget_n_mc=6000,
    )
    # surrogate prediction at the nominal design vs the solver itself
    nominal = np.array([[bench.reference_case[k] for k in bench.spec.names]])
    _, coords = bench.spec.from_latent(bench.spec.physical_to_latent(nominal))
    f_sim = float(study.surrogate.evaluate(coords)[0])
    f_exp = bench.reference_case["RT_kN"]
    result = validate(
        f_sim,
        f_exp,
        u_num=study.surrogate.eps_loo_abs,
        u_input=0.5 * study.budget.width,
        label="holtrop-uq-chain",
    )
    assert result.passed
    assert result.u_val > 0.0
    assert abs(f_sim - f_exp) / f_exp < 1.0e-3


def test_workflow_does_not_import_solver_kernels() -> None:
    """The UQ workflow must stay a consumer of the black-box contract."""
    source = pathlib.Path("src/tensorphysics/uq/workflow.py").read_text()
    for forbidden in ("physics_kernels", "holtrop", "resistance"):
        assert forbidden not in source
    # the Holtrop benchmark module is the only place that touches the kernel,
    # and it does so through a local import
    bench_source = pathlib.Path("src/tensorphysics/uq/benchmarks.py").read_text()
    assert "from ..physics_kernels_holtrop1984 import" in bench_source
    assert "holtrop1984_resistance(" in bench_source


def test_adapter_objects_are_accepted() -> None:
    adapter = CallableAdapter(ishigami, names=ishigami_spec().names, name="ishigami")
    study = run_uq_study(
        adapter,
        ishigami_spec(),
        n_samples=512,
        degree=6,
        n_boot=0,
        n_base_mc=1 << 11,
        bootstrap_mc=0,
        budget_n_mc=3000,
    )
    assert study.results.n_valid == 512
    assert study.metadata["n_model_evaluations"] > 512
    with pytest.raises(TypeError):
        run_uq_study(42, ishigami_spec())  # type: ignore[arg-type]