"""Tests for the cross-repository benchmark adapters and the gate CLI.

The adapters must be able to read the benchmark dialects of the Tensor* solver
repositories without any change to those repositories, and they must carry the
numbers through unchanged (field conservation ``diff < 1e-12``).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorphysics.vv import adapters
from tensorphysics.vv.cli import main as cli_main

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "vv_gate"

LBM_DOC = {
    "schema": "tensorlbm.convergence/1.0",
    "quantity": "rel_l2_error",
    "note": "poiseuille analytic oracle",
    "levels": [
        {"resolution": 16, "value": 2.394666e-02, "label": "n=16"},
        {"resolution": 32, "value": 1.383948e-02, "label": "n=32"},
        {"resolution": 64, "value": 7.834480e-03, "label": "n=64"},
    ],
}

FEM_DOC = {
    "schema": "tensorfem.h-convergence-gci/1.0",
    "bending": {
        "oracle": "pure bending (Navier)",
        "levels": [
            {"divisions": 1, "mesh_size": 1.0, "relative_strain_energy_error": 2.602884615384507},
            {"divisions": 2, "mesh_size": 0.5, "relative_strain_energy_error": 0.6507211538458297},
            {"divisions": 4, "mesh_size": 0.25, "relative_strain_energy_error": 0.16268028846081853},
        ],
    },
    "manufactured": {
        "oracle": "smooth trigonometric manufactured solution",
        "levels": [
            {"divisions": 1, "mesh_size": 1.0, "relative_l2_displacement_error": 1.6e-01},
            {"divisions": 2, "mesh_size": 0.5, "relative_l2_displacement_error": 4.0e-02},
            {"divisions": 4, "mesh_size": 0.25, "relative_l2_displacement_error": 1.0e-02},
        ],
    },
}

FVM_DOC = {
    "schema": "tensorfvm.convergence/1.0",
    "grids": [
        {"nx": 32, "h": 0.03125, "l2_error": 4.0e-3},
        {"nx": 64, "h": 0.015625, "l2_error": 1.0e-3},
        {"nx": 128, "h": 0.0078125, "l2_error": 2.5e-4},
    ],
}

DEM_DOC = {
    "schema": "tensordem.timestep-refinement/1.0",
    "timesteps": [
        {"dt": 4.0e-3, "max_error": 3.2e-2},
        {"dt": 2.0e-3, "max_error": 1.6e-2},
        {"dt": 1.0e-3, "max_error": 8.0e-3},
    ],
}

BEM_DOC = {
    "schema": "tensorbem.convergence/1.0",
    "meshes": [
        {"n_panels": 256, "relative_error": 8.0e-2},
        {"n_panels": 512, "relative_error": 4.0e-2},
        {"n_panels": 1024, "relative_error": 2.0e-2},
    ],
}


def _write(tmp_path: Path, name: str, payload: object) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# adapters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "doc, expected_repo",
    [
        (LBM_DOC, "tensorlbm"),
        (FEM_DOC, "tensorfem"),
        (FVM_DOC, "tensorfvm"),
        (DEM_DOC, "tensordem"),
        (BEM_DOC, "tensorbem"),
    ],
)
def test_profiles_are_detected_from_the_document(doc, expected_repo):
    assert adapters.detect_repo(doc) == expected_repo


def test_lbm_levels_use_inverse_resolution_as_grid_length():
    series = adapters.extract_series(LBM_DOC)
    assert series.repo == "tensorlbm"
    assert [lvl.h for lvl in series.levels] == [1.0 / 16, 1.0 / 32, 1.0 / 64]  # coarse -> fine
    assert series.levels[-1].f == pytest.approx(7.834480e-03, rel=0)
    assert series.quantity == "rel_l2_error"


def test_fem_evidence_selects_the_manufactured_study_by_default():
    series = adapters.extract_series(FEM_DOC)
    assert series.repo == "tensorfem"
    assert series.study == "manufactured"
    assert [lvl.h for lvl in series.levels] == [1.0, 0.5, 0.25]
    assert series.metadata["oracle"].startswith("smooth trigonometric")
    bent = adapters.extract_series(FEM_DOC, study="bending")
    assert bent.levels[-1].f == pytest.approx(0.16268028846081853, rel=0)


def test_explicit_value_key_overrides_the_profile_order():
    series = adapters.extract_series(FEM_DOC, study="manufactured",
                                     value_key="relative_l2_displacement_error")
    assert series.metadata["value_key"] == "relative_l2_displacement_error"
    assert series.levels[-1].f == pytest.approx(0.01, rel=0)


def test_unknown_documents_fall_back_to_the_generic_profile():
    doc = {"report": {"whatever": [{"h": 1.0, "value": 5.0}, {"h": 0.5, "value": 4.25}]}}
    assert adapters.detect_repo(doc) == "generic"
    series = adapters.extract_series(doc)
    assert [lvl.f for lvl in series.levels] == [5.0, 4.25]  # coarse -> fine


def test_adapter_errors_are_explicit(tmp_path):
    with pytest.raises(adapters.AdapterError):
        adapters.extract_series({"levels": [{"value": 1.0}, {"value": 2.0}]})
    with pytest.raises(adapters.AdapterError):
        adapters.extract_series(FEM_DOC, study="does-not-exist")
    with pytest.raises(adapters.AdapterError):
        adapters.extract_series({"levels": []})


@pytest.mark.parametrize("doc, value_key", [
    (LBM_DOC, "value"),
    (FEM_DOC, "relative_l2_displacement_error"),
    (FVM_DOC, "l2_error"),
    (DEM_DOC, "max_error"),
    (BEM_DOC, "relative_error"),
])
def test_field_conservation_through_the_gate_payload(doc, value_key):
    """Every number survives the adapter round trip to better than 1e-12."""
    series = adapters.extract_series(doc)
    payload = series.to_gate_payload(expected_order=2.0)
    text = json.dumps(payload)
    reloaded = json.loads(text)

    raw = series.raw_levels
    by_value = {}
    for item in raw:
        length_key = next((k for k in ("h", "mesh_size", "dx", "dt") if k in item), None)
        if length_key is not None:
            key = float(item[length_key])
        else:
            count_key = next((k for k in ("resolution", "nx", "n_panels") if k in item), None)
            key = 1.0 / float(item[count_key])
        by_value[key] = float(item[value_key])

    assert len(reloaded["levels"]) == len(by_value)
    for level in reloaded["levels"]:
        h, f = level["h"], level["f"]
        ref_f = by_value[h]
        assert abs(f - ref_f) <= 1e-12 * max(1.0, abs(ref_f))
    assert reloaded["schema"] == "tensorphysics.vv.observations/1"


def test_streaming_equivalent_output_for_two_different_repositories(tmp_path):
    """The same CLI consumes LBM and FEM documents and yields comparable studies."""
    lbm = _write(tmp_path, "lbm.json", LBM_DOC)
    fem = _write(tmp_path, "fem.json", FEM_DOC)
    lbm_series = adapters.extract_series(adapters.load_payload(lbm), source_path=str(lbm))
    fem_series = adapters.extract_series(adapters.load_payload(fem), source_path=str(fem))
    assert lbm_series.repo != fem_series.repo
    for series in (lbm_series, fem_series):
        payload = series.to_gate_payload()
        out = tmp_path / f"obs_{series.repo}.json"
        out.write_text(json.dumps(payload), encoding="utf-8")
        assert json.loads(out.read_text(encoding="utf-8"))["levels"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _obs_payload(levels, expected_order=None, validation=None):
    payload = {
        "schema": "tensorphysics.vv.observations/1",
        "quantity": "Cd",
        "levels": [{"h": h, "f": f} for h, f in levels],
    }
    if expected_order is not None:
        payload["expected_order"] = expected_order
    if validation:
        payload["validation"] = validation
    return payload


def test_cli_passes_a_clean_study(tmp_path, capsys):
    path = _write(tmp_path, "obs.json", _obs_payload(
        [(1.0, 3.5), (0.5, 3.125), (0.25, 3.03125)], expected_order=2.0))
    code = cli_main(["--obs", str(path), "--json"])
    record = json.loads(capsys.readouterr().out)
    assert code == 0
    assert record["status"] == "PASS"
    assert record["study"]["order_p"] == pytest.approx(2.0, abs=1e-9)


def test_cli_fails_a_non_asymptotic_study(tmp_path, capsys):
    path = _write(tmp_path, "obs.json", _obs_payload([(1.0, 1.5), (0.5, 1.2), (0.25, 1.35)]))
    code = cli_main(["--obs", str(path), "--json"])
    record = json.loads(capsys.readouterr().out)
    assert code == 2
    assert record["status"] == "FAIL"


def test_cli_reports_warn_with_exit_code_one(tmp_path, capsys):
    path = _write(tmp_path, "obs.json", _obs_payload(
        [(1.0, 3.5), (0.5, 3.125), (0.25, 3.03125)], expected_order=2.3))
    code = cli_main(["--obs", str(path)])
    assert code == 1
    assert "WARN" in capsys.readouterr().out


def test_cli_validation_blocks_are_honoured(tmp_path, capsys):
    payload = _obs_payload([(1.0, 3.5), (0.5, 3.125), (0.25, 3.03125)],
                           validation={"f_exp": 2.5, "u_input": 0.05})
    path = _write(tmp_path, "obs.json", payload)
    code = cli_main(["--obs", str(path), "--json"])
    record = json.loads(capsys.readouterr().out)
    assert code == 2
    assert record["validation"]["status"] == "FAIL"


def test_cli_reads_native_benchmark_documents(tmp_path, capsys):
    fem = _write(tmp_path, "evidence.json", FEM_DOC)
    code = cli_main(["--benchmark", str(fem), "--study", "manufactured", "--json"])
    record = json.loads(capsys.readouterr().out)
    assert code == 0
    assert record["adapter"]["repo"] == "tensorfem"
    assert record["study"]["order_p"] == pytest.approx(2.0, abs=1e-6)


def test_cli_table_mode(tmp_path, capsys):
    table = _write(tmp_path, "table.json", {"rows": [
        {"label": "a", "f_sim": 1.0, "f_exp": 1.02, "u_num": 0.02, "u_input": 0.01},
        {"label": "b", "f_sim": 1.0, "f_exp": 1.05, "u_num": 0.03, "u_input": 0.03},
    ]})
    code = cli_main(["--table", str(table), "--json"])
    record = json.loads(capsys.readouterr().out)
    assert code == 2
    assert (record["n_pass"], record["n_fail"]) == (1, 1)


def test_cli_returns_usage_error_for_bad_input(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert cli_main(["--obs", str(bad)]) == 64
    assert cli_main(["--obs", str(tmp_path / "missing.json")]) == 64
    assert cli_main(["--benchmark", str(_write(tmp_path, "empty.json", {"foo": 1}))]) == 64
    capsys.readouterr()


@pytest.mark.skipif(not (ARTIFACTS / "obs_tensorlbm_poiseuille.json").exists(),
                    reason="cross-repo artifacts not generated in this checkout")
def test_cli_on_the_real_tensorlbm_artifact(capsys):
    code = cli_main(["--obs", str(ARTIFACTS / "obs_tensorlbm_poiseuille.json"), "--json"])
    record = json.loads(capsys.readouterr().out)
    assert code in (0, 1, 2)
    assert record["study"]["n_levels"] == 3
    assert record["study"]["order_p"] is not None


@pytest.mark.skipif(not (ARTIFACTS / "obs_tensorfem_manufactured.json").exists(),
                    reason="cross-repo artifacts not generated in this checkout")
def test_cli_on_the_real_tensorfem_artifact(capsys):
    code = cli_main(["--obs", str(ARTIFACTS / "obs_tensorfem_manufactured.json"),
                     "--expected-order", "2", "--json"])
    record = json.loads(capsys.readouterr().out)
    assert code == 0
    assert abs(record["study"]["order_p"] - 2.0) < 0.2