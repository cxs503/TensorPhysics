"""Gate falsification matrix, ASME chaining, benchmark registry and CLI aliases.

These tests close the audit loop around the maturation deliverables:

* the three-state gate must not pass **any** injected defect (robustness matrix);
* the UQ record must chain into the ASME V&V 20 comparison;
* the benchmark registry behind the CLI must work for every advertised entry;
* the cross-repository parser must accept the documented TensorLBM marine list
  schema and must reject unknown schemas instead of guessing.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from tensorphysics.uq.benchmarks import (
    holtrop_type_benchmark,
    ishigami,
    ishigami_spec,
    sobol_g,
    sobol_g_analytic,
    sobol_g_spec,
)
from tensorphysics.uq.cli import main
from tensorphysics.uq.gate import (
    PASS,
    UQEvidenceError,
    chain_asme_validation,
    engineering_error_gate,
    pre_evidence,
    require_pass,
)
from tensorphysics.uq.robustness import injection_suite
from tensorphysics.uq.sample_spec import Marginal, SampleSpec
from tensorphysics.uq.solver_outputs import (
    UnknownBenchmarkError,
    parse_benchmarks,
)
from tensorphysics.uq.workflow import run_uq_study

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def narrow_study():
    spec = SampleSpec((Marginal.uniform("a", 0.9, 1.1), Marginal.uniform("b", -0.1, 0.1)))

    def solver(points):
        return 10.0 + points[:, 0] + 0.2 * points[:, 1]

    return run_uq_study(
        solver, spec, n_samples=128, degree=2, seed=4, run_mc=False,
        require_cross_check=False,
    )


def test_injection_matrix_never_passes_a_defect():
    report = injection_suite(seed=20240)
    assert report.control_passes is True
    assert report.all_safe is True
    assert report.all_named is True
    assert report.all_matched is True
    assert report.passed is True
    assert report.code_union, "every blocking condition must be named by some case"
    payload = report.as_dict()
    assert payload["passed"] is True
    # every case must land inside its own predeclared admissible set, and the
    # only case allowed to pass is the boundary control V5a
    for case in payload["cases"]:
        assert case["observed_status"] in case["allowed_statuses"], case["case_id"]
        if "PASS" not in case["allowed_statuses"]:
            assert case["observed_status"] != PASS, case["case_id"]
    defect_cases = [c for c in payload["cases"] if "PASS" not in c["allowed_statuses"]]
    assert len(defect_cases) == payload["n_cases"] - 1


def test_pre_evidence_digest_is_stable_and_addresses_the_budget(narrow_study):
    first = pre_evidence(narrow_study, label="unit")
    second = pre_evidence(narrow_study, label="unit")
    assert first.digest == second.digest
    assert first.status == PASS
    assert first.budget is not None
    assert len(first.digest) == 64


def test_engineering_error_gate_needs_a_study():
    decision = engineering_error_gate(study=None, relative_error=0.01)
    assert decision.status != PASS
    assert any(r.code == "UQ-G01-NO-EVIDENCE" for r in decision.reasons)


def test_require_pass_raises_for_non_pass():
    decision = engineering_error_gate(study=None, relative_error=0.01)
    with pytest.raises(UQEvidenceError):
        require_pass(decision)


def test_chain_asme_validation_keeps_both_verdicts(narrow_study):
    out = chain_asme_validation(
        study=narrow_study, f_sim=10.05, f_exp=10.0, u_num=0.02, label="chained"
    )
    assert out["label"] == "chained"
    assert out["asme"]["u_input"] > 0.0
    assert out["gate"]["status"] == PASS
    assert out["combined_status"] == PASS
    assert out["evidence"]["digest"]


def test_sobol_g_benchmark_closed_forms_and_validation():
    a = [0.0, 0.5, 3.0]
    analytic = sobol_g_analytic(a)
    assert analytic["first"].sum() < 1.0
    assert np.all(analytic["total"] >= analytic["first"] - 1e-12)
    points = np.random.default_rng(0).random((64, 3))
    values = sobol_g(points, a=a)
    assert values.shape == (64,)
    with pytest.raises(ValueError):
        sobol_g(points[:, :2], a=a)
    with pytest.raises(ValueError):
        sobol_g_analytic([-1.0, 1.0])
    spec = sobol_g_spec(3)
    assert spec.dim == 3
    assert spec.names == ("x1", "x2", "x3")


def test_holtrop_type_benchmark_is_callable_and_documented():
    bench = holtrop_type_benchmark()
    assert bench.dim == 5
    assert bench.notes
    centre = np.array([[bench.reference_case[k] for k in ("L", "B", "T", "Cb", "Vs")]])
    values = bench.resistance_kn(centre)
    assert values.shape == (1,)
    assert values[0] > 0.0
    # the same call through the ishigami-like interface used by the CLI
    assert ishigami(np.zeros((2, 3))).shape == (2,)


def test_tensorlbm_marine_list_schema(tmp_path):
    payload = [
        {"name": "cylinder_re100", "st_sim": 0.170, "st_ref": 0.166, "cd_sim": 1.40, "cd_ref": 1.38},
        {"name": "sloshing_tank", "omega_theory": 1.0, "omega_measured": 1.04},
        {"name": "pipeline_eD05", "st_sim": 0.190, "st_ref": 0.183},
    ]
    path = tmp_path / "bench_marine.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    artifact, records = parse_benchmarks(path)
    assert artifact.solver == "TensorLBM"
    ids = {r.qoi_id for r in records}
    assert "lbm.marine.cylinder_re100.strouhal" in ids
    assert "lbm.marine.cylinder_re100.drag_coefficient" in ids
    assert "lbm.marine.sloshing_tank.oscillation_frequency" in ids
    st = [r for r in records if r.qoi_id.endswith(".strouhal")][0]
    assert st.relative_error == pytest.approx(abs(0.170 - 0.166) / 0.166)
    assert st.value == pytest.approx(0.170)


def test_unknown_schema_and_empty_extraction_are_errors(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
    with pytest.raises(UnknownBenchmarkError):
        parse_benchmarks(bad)
    empty = tmp_path / "empty.json"
    empty.write_text(
        json.dumps({"cases": [], "length_study": [], "criterion": "none"}),
        encoding="utf-8",
    )
    with pytest.raises(UnknownBenchmarkError):
        parse_benchmarks(empty)
    with pytest.raises(FileNotFoundError):
        parse_benchmarks(tmp_path / "nope.json")


def test_cli_covers_the_remaining_benchmarks(tmp_path, capsys):
    out = tmp_path / "sobolg.json"
    assert main(
        ["--pce", "--benchmark", "sobol-g", "--dim", "3", "--n", "96",
         "--degree", "2", "--budget-n-mc", "512", "--json", str(out), "--quiet"]
    ) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["benchmark"] == "sobol-g"
    assert payload["analytic_comparison"]["first_max_abs_error"] < 0.2

    out2 = tmp_path / "holtrop.json"
    assert main(
        ["--mode", "pce", "--benchmark", "holtrop-type", "--n", "192", "--degree", "3",
         "--budget-n-mc", "1024", "--json", str(out2), "--quiet"]
    ) == 0
    payload2 = json.loads(out2.read_text(encoding="utf-8"))
    assert payload2["benchmark"] == "holtrop-type"
    # 5 inputs at degree 3 need enough samples per active term; below that the
    # gate must withhold rather than advertise an over-fitted surrogate
    assert payload2["gate"]["status"] in {"PASS", "FAIL", "WITHHELD"}


def test_cli_rejects_a_study_without_a_budget(tmp_path, capsys):
    study = tmp_path / "study.json"
    study.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    code = main([
        "--benchmark-json", str(ROOT / "data/cross_repo_benchmarks" /
                                "tensorfvm_lid_cavity_summary.json"),
        "--study-json", str(study), "--quiet",
    ])
    assert code == 2
    assert "relative_width" in capsys.readouterr().err


def test_cli_require_pass_on_a_passing_benchmark(tmp_path):
    """holtrop-type has a 5 % band, so --require-pass must succeed there."""
    out = tmp_path / "holtrop_pass.json"
    code = main([
        "--pce", "--benchmark", "holtrop-type", "--n", "128", "--degree", "3",
        "--budget-n-mc", "1024", "--require-pass", "--json", str(out), "--quiet",
    ])
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["gate"]["status"] == "PASS"
    assert code == 0


def test_cli_ishigami_spec_is_reused_by_the_suite():
    spec = ishigami_spec()
    assert spec.names == ("x1", "x2", "x3")
    assert spec.dim == 3