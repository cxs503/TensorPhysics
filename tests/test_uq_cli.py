"""CLI: the three predeclared modes (``--pce`` / ``--sobol`` / ``--montecarlo``).

Acceptance item F5(c): all three modes are usable from the command line, the
modes are mutually exclusive, the fail-safe flag is available, and the
cross-repository artefact can be parsed from the same entry point.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from tensorphysics.uq.cli import BENCHMARKS, build_parser, main

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cross_repo_benchmarks"

ISHIGAMI_FAST = (
    "--benchmark", "ishigami", "--n", "128", "--degree", "8",
    "--n-base", "512", "--budget-n-mc", "512",
)


def test_parser_declares_three_modes():
    parser = build_parser()
    args = parser.parse_args(["--pce"])
    assert args.mode == "pce"
    assert parser.parse_args(["--sobol"]).mode == "sobol"
    assert parser.parse_args(["--montecarlo"]).mode == "montecarlo"
    assert set(BENCHMARKS) >= {"ishigami", "holton", "sobol-g", "holtrop-type"}


def test_modes_are_mutually_exclusive():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--pce", "--montecarlo"])


def test_pce_mode_reports_back_full_report(tmp_path, capsys):
    out = tmp_path / "pce.json"
    code = main(["--pce", *ISHIGAMI_FAST, "--json", str(out), "--quiet"])
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["mode"] == "pce"
    assert payload["indices_source"] == "pce"
    assert payload["q2"] > 0.95
    assert "montecarlo" not in payload
    comp = payload["analytic_comparison"]
    assert comp["first_max_abs_error"] <= 0.02
    assert comp["total_max_abs_error"] <= 0.02
    assert capsys.readouterr().out == ""


def test_sobol_mode_includes_the_cross_check(tmp_path):
    out = tmp_path / "sobol.json"
    assert main(["--sobol", *ISHIGAMI_FAST, "--json", str(out), "--quiet"]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["mode"] == "sobol"
    assert "montecarlo" in payload
    assert payload["pce_vs_mc"]["first_max_abs_diff"] < 0.1


def test_montecarlo_mode_has_no_surrogate(tmp_path):
    out = tmp_path / "mc.json"
    assert main(["--montecarlo", *ISHIGAMI_FAST, "--json", str(out), "--quiet"]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["mode"] == "montecarlo"
    assert payload["indices_source"] == "montecarlo"
    assert "q2" not in payload
    assert "first_jansen" in payload["indices"]
    assert payload["gate"]["status"] in {"PASS", "FAIL"}
    codes = [r["code"] for r in payload["gate"]["reasons"]]
    assert "UQ-N05-MC-FAILSAFE" in codes


def test_holton_benchmark_is_exact_at_degree_two(tmp_path):
    out = tmp_path / "holton.json"
    assert main(
        ["--pce", "--benchmark", "holton", "--dim", "6", "--n", "128",
         "--degree", "2", "--budget-n-mc", "512", "--json", str(out), "--quiet"]
    ) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["benchmark"] == "holton"
    assert payload["analytic_comparison"]["total_max_abs_error"] < 1e-8


def test_failsafe_flag_degrades_on_a_starved_design(tmp_path):
    out = tmp_path / "failsafe.json"
    assert main(
        ["--pce", "--benchmark", "ishigami", "--n", "8", "--degree", "8",
         "--budget-n-mc", "256", "--failsafe", "--json", str(out), "--quiet"]
    ) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["failsafe"]["mode"] == "montecarlo"
    assert payload["failsafe"]["decision"]["triggered"] is True


def test_require_pass_exit_code(tmp_path):
    out = tmp_path / "gate.json"
    # Ishigami's 95 % band is several times its mean: the gate cannot pass.
    assert main(["--pce", *ISHIGAMI_FAST, "--require-pass", "--json", str(out), "--quiet"]) == 3


def test_benchmark_json_listing(capsys):
    code = main([
        "--benchmark-json", str(DATA / "tensorlbm_sdaa_wigley_hydrostatics.json"),
        "--list-qoi",
    ])
    assert code == 0
    text = capsys.readouterr().out
    assert "lbm.wigley_hydrostatics.C_B@640" in text
    assert "QoI ids" in text


def test_benchmark_json_budget_combination(tmp_path, capsys):
    study = tmp_path / "study.json"
    assert main(["--pce", *ISHIGAMI_FAST, "--json", str(study), "--quiet"]) == 0
    out = tmp_path / "combined.json"
    code = main([
        "--benchmark-json", str(DATA / "tensorfem_finite_width_hole_audit.json"),
        "--study-json", str(study), "--json", str(out), "--quiet",
    ])
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["combined_budget"]["combined_standard"] > 0.0
    assert payload["artifact"]["solver"] == "TensorFEM"
    assert len(payload["records"]) == 9


def test_bad_inputs_return_two(tmp_path, capsys):
    assert main(["--benchmark-json", str(tmp_path / "missing.json")]) == 2
    assert main(["--benchmark-json", str(DATA / "PROVENANCE.json")]) == 2
    assert "error:" in capsys.readouterr().err


def test_summary_is_printed_by_default(capsys):
    assert main(["--pce", *ISHIGAMI_FAST]) == 0
    text = capsys.readouterr().out
    assert "gate:" in text
    assert "S_Ti" in text
    assert "vs analytic" in text