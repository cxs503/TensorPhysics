"""Cross-repository benchmark adapters (TensorLBM / TensorFVM / TensorFEM).

Acceptance item F6: benchmark JSON shipped by at least two solver repositories
must be parsed into machine-usable QoI records, with digest provenance, and the
comparison error must be able to enter the uncertainty budget and the
pre-evidence gate.

The fixtures in ``data/cross_repo_benchmarks`` are byte-identical copies of real
artefacts; the ``PROVENANCE.json`` digests are re-verified here.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from tensorphysics.uq.gate import PASS, WITHHELD
from tensorphysics.uq.solver_outputs import (
    SUPPORTED_SOLVERS,
    BenchmarkArtifact,
    BenchmarkQoI,
    UnknownBenchmarkError,
    combine_budget,
    detect_solver,
    gate_with_benchmark,
    parse_artifact,
    parse_benchmarks,
    relative_error,
    select,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cross_repo_benchmarks"

LBM = DATA / "tensorlbm_sdaa_wigley_hydrostatics.json"
FVM_PROFILE = DATA / "tensorfvm_lid_cavity_summary.json"
FVM_MESH = DATA / "tensorfvm_mesh_quality_results.json"
FEM = DATA / "tensorfem_finite_width_hole_audit.json"


@pytest.fixture(scope="module")
def provenance() -> dict:
    return json.loads((DATA / "PROVENANCE.json").read_text(encoding="utf-8"))


def test_at_least_two_solver_repositories_are_supported():
    assert len(SUPPORTED_SOLVERS) >= 2
    assert {"TensorLBM", "TensorFVM"} <= set(SUPPORTED_SOLVERS)


def test_detect_solver_on_every_fixture():
    assert BenchmarkArtifact.load(LBM).solver == "TensorLBM"
    assert BenchmarkArtifact.load(FVM_PROFILE).solver == "TensorFVM"
    assert BenchmarkArtifact.load(FVM_MESH).solver == "TensorFVM"
    assert BenchmarkArtifact.load(FEM).solver == "TensorFEM"
    with pytest.raises(UnknownBenchmarkError):
        detect_solver({"totally": "different"})


def test_digests_match_the_recorded_provenance(provenance):
    by_local = {item["local"]: item for item in provenance["artifacts"]}
    for path in (LBM, FVM_PROFILE, FVM_MESH, FEM):
        art = BenchmarkArtifact.load(path)
        assert art.sha256 == by_local[path.name]["sha256"], path.name
        assert art.solver == by_local[path.name]["solver"]


def test_tensorlbm_hydrostatics_records():
    art, records = parse_benchmarks(LBM)
    assert art.solver == "TensorLBM"
    assert len(records) == 14
    cb = [r for r in records if r.qoi_id == "lbm.wigley_hydrostatics.C_B@640"]
    assert len(cb) == 1
    assert cb[0].value == pytest.approx(0.4532183837890625)
    assert cb[0].reference == pytest.approx(0.4444444444)
    assert cb[0].relative_error == pytest.approx(0.019741, rel=1e-3)
    assert cb[0].passed is True  # taken from overall_PASS, not re-typed


def test_tensorfvm_velocity_profile_records():
    art, records = parse_benchmarks(FVM_PROFILE)
    assert art.solver == "TensorFVM"
    assert len(records) == 2
    rel = {r.qoi_id.split(".")[-1]: r.relative_error for r in records}
    assert rel["u_profile_l2"] == pytest.approx(0.004736, rel=1e-2)
    assert rel["v_profile_l2"] == pytest.approx(0.026813, rel=1e-2)
    assert all(r.detail["reynolds"] == 100.0 for r in records)


def test_tensorfvm_mesh_quality_keeps_absolute_error_when_reference_is_zero():
    _, records = parse_benchmarks(FVM_MESH)
    assert len(records) == 12
    first = records[0]
    assert first.reference == 0.0
    assert first.relative_error is None
    assert first.absolute_error == pytest.approx(8.537736462515939e-07)
    shear = [r for r in records if "shear_closed_form" in r.qoi_id]
    non_zero_ref = [r for r in shear if r.reference not in (0.0, None)]
    assert len(non_zero_ref) == 7, "the shear sweep must contain seven non-zero references"
    assert all(r.relative_error < 1e-10 for r in non_zero_ref)


def test_tensorfem_hole_audit_records():
    _, records = parse_benchmarks(FEM)
    assert len(records) == 9
    finest = [r for r in records if r.qoi_id.endswith("kt_net_finest")]
    assert len(finest) == 5
    assert max(r.relative_error for r in finest) < 0.03
    assert all(r.passed is True for r in finest)
    length = [r for r in records if "length_study" in r.qoi_id]
    assert len(length) == 4
    assert length[0].relative_error == pytest.approx(0.0018817, rel=1e-3)
    assert length[0].detail["half_length_over_half_width"] == pytest.approx(3.0)


def test_select_and_relative_error_filters():
    _, records = parse_benchmarks(FEM)
    c3 = select(records, qoi_id="fem.finite_width_hole.rho0.3.kt_net_finest")
    assert len(c3) == 1
    worst = relative_error(records)
    assert worst >= max(r.relative_error for r in records if r.relative_error is not None)
    with pytest.raises(ValueError):
        relative_error(records, qoi_id="does.not.exist")


def test_relative_error_is_not_reported_against_a_numerically_zero_reference():
    _, records = parse_benchmarks(FVM_MESH)
    rotation = [r for r in records if "rotation_invariance" in r.qoi_id]
    assert rotation, "the mesh-quality artefact must contain rotation-invariance rows"
    # the reference there is an orthogonal-mesh residual of order 1e-14
    assert all(r.reference is not None and abs(r.reference) < 1e-10 for r in rotation)
    assert all(r.relative_error is None for r in rotation)
    assert all(r.absolute_error is not None and r.absolute_error >= 0.0 for r in rotation)


def test_parse_artifact_is_pure(monkeypatch):
    art = BenchmarkArtifact.load(FEM)
    before = len(parse_artifact(art))
    again = len(parse_artifact(art))
    assert before == again


def test_combine_budget_quadrature_and_gate():
    _, records = parse_benchmarks(FEM)
    budget = combine_budget(0.05, records, threshold=0.15)
    uq_term = 0.05 / (2 * 1.959963984540054)
    bench_terms = [
        (abs(r.relative_error) / (3 ** 0.5)) for r in records if r.relative_error is not None
    ]
    expected = (uq_term**2 + sum(t**2 for t in bench_terms)) ** 0.5
    assert budget.combined_standard == pytest.approx(expected, rel=1e-12)
    assert budget.expanded_relative == pytest.approx(2.0 * expected, rel=1e-12)
    assert budget.status == "PASS"
    assert budget.components[0].name == "uq_band"
    assert "sha256=" in budget.components[1].source


def test_combine_budget_rejects_empty_or_invalid_input():
    _, records = parse_benchmarks(FEM)
    with pytest.raises(ValueError):
        combine_budget(-0.1, records)
    only_absolute = tuple(
        BenchmarkQoI(
            solver="TensorFVM", artifact="x", sha256="0" * 64, qoi_id="a", quantity="q",
            value=1.0, reference=0.0, absolute_error=1.0, relative_error=None, passed=None,
        )
        for _ in range(1)
    )
    with pytest.raises(ValueError):
        combine_budget(0.1, only_absolute)


@pytest.fixture(scope="module")
def narrow_study():
    """A real study whose 95 % band is comfortably inside the 15 % threshold."""
    from tensorphysics.uq.sample_spec import Marginal, SampleSpec
    from tensorphysics.uq.workflow import run_uq_study

    spec = SampleSpec((Marginal.uniform("a", 0.9, 1.1), Marginal.uniform("b", -0.1, 0.1)))

    def solver(points):
        return 10.0 + points[:, 0] + 0.2 * points[:, 1]

    return run_uq_study(
        solver, spec, n_samples=128, degree=2, seed=4, run_mc=False,
        require_cross_check=False,
    )


def test_gate_with_benchmark_passes_on_real_evidence(narrow_study):
    _, records = parse_benchmarks(FEM)
    assert narrow_study.status == PASS
    out = gate_with_benchmark(study=narrow_study, records=records, label="hole-kt")
    assert out["status"] == PASS
    assert out["relative_error"] == pytest.approx(relative_error(records))
    assert out["combined_budget"]["status"] == "PASS"
    assert {p["solver"] for p in out["provenance"]} == {"TensorFEM"}
    assert out["gate"]["evidence_digest"]


def test_gate_with_benchmark_withholds_without_a_study():
    _, records = parse_benchmarks(FEM)
    out = gate_with_benchmark(study=None, records=records, label="no-study")
    assert out["status"] == WITHHELD
    assert out["combined_budget"] is None