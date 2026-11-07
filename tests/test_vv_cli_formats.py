"""CLI output formats (text / json / table / plot) and adapter edge paths.

Mission IN57-N07 maturity item: the gate CLI must offer three machine-usable
output formats besides the human report, and the adapter layer must fail closed
on malformed benchmark documents instead of guessing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorphysics.vv import adapters, cli
from tensorphysics.vv.gci import GatePolicy, Observation, evaluate

ROOT = Path(__file__).resolve().parents[1]


def _obs_payload(levels, expected_order=None, sigma=None):
    payload = {
        "schema": "tensorphysics.vv.observations/1",
        "quantity": "Cd",
        "levels": [{"h": h, "f": f, "sigma": sigma} if sigma else {"h": h, "f": f}
                   for h, f in levels],
    }
    if expected_order is not None:
        payload["expected_order"] = expected_order
    return payload


def _clean(tmp_path: Path, sigma=None) -> Path:
    path = tmp_path / "obs.json"
    path.write_text(json.dumps(_obs_payload(
        [(1.0, 3.5), (0.5, 3.125), (0.25, 3.03125)], expected_order=2.0, sigma=sigma)),
        encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# output formats
# ---------------------------------------------------------------------------


def test_format_table_renders_metrics_and_checks(tmp_path, capsys):
    path = _clean(tmp_path)
    code = cli.main(["--obs", str(path), "--format", "table"])
    out = capsys.readouterr().out
    assert code == 1
    assert "| metric" in out and "| value" in out
    assert "observed order p" in out
    assert "| status" in out and "noise-certificate" in out
    assert out.rstrip().endswith("VERDICT: WARN")


def test_format_table_includes_the_validation_block(tmp_path, capsys):
    payload = _obs_payload([(1.0, 3.5), (0.5, 3.125), (0.25, 3.03125)],
                           expected_order=2.0)
    payload["validation"] = {"f_exp": 3.03, "u_input": 0.05}
    path = tmp_path / "obs.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    code = cli.main(["--obs", str(path), "--format", "table"])
    out = capsys.readouterr().out
    assert code in (0, 1, 2)
    assert "ASME V&V 20 validation" in out
    assert "|E|/U_val" in out


def test_format_plot_writes_a_png(tmp_path, capsys):
    path = _clean(tmp_path)
    png = tmp_path / "figs" / "gci.png"
    code = cli.main(["--obs", str(path), "--format", "plot", "--plot-out", str(png)])
    out = capsys.readouterr().out
    assert code in (0, 1)
    assert png.exists() and png.stat().st_size > 1024
    assert "plot written to" in out
    assert "VERDICT:" in out


def test_format_plot_default_path(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = _clean(tmp_path)
    code = cli.main(["--obs", str(path), "--format", "plot"])
    capsys.readouterr()
    assert code in (0, 1)
    assert (tmp_path / "vv_gate.png").exists()


def test_plot_without_matplotlib_is_a_usage_error(tmp_path, capsys, monkeypatch):
    path = _clean(tmp_path)

    def boom():
        raise RuntimeError("plot output needs matplotlib")

    monkeypatch.setattr(cli, "_pyplot", boom)
    code = cli.main(["--obs", str(path), "--format", "plot"])
    assert code == cli.EXIT_USAGE
    assert "matplotlib" in capsys.readouterr().err


def test_json_alias_and_out_file(tmp_path, capsys):
    path = _clean(tmp_path)
    out = tmp_path / "verdict.json"
    code = cli.main(["--obs", str(path), "--json", "--out", str(out),
                     "--allow-undeclared-noise"])
    record = json.loads(capsys.readouterr().out)
    assert code == 0 and record["status"] == "PASS"
    on_disk = json.loads(out.read_text())
    assert on_disk["status"] == record["status"]
    assert on_disk["study"]["order_p"] == pytest.approx(2.0, abs=1e-9)


def test_allow_undeclared_noise_restores_the_unqualified_pass(tmp_path, capsys):
    path = _clean(tmp_path)
    strict_code = cli.main(["--obs", str(path), "--json"])
    strict = json.loads(capsys.readouterr().out)
    relaxed_code = cli.main(["--obs", str(path), "--json", "--allow-undeclared-noise"])
    relaxed = json.loads(capsys.readouterr().out)
    assert (strict_code, strict["status"]) == (1, "WARN")
    assert (relaxed_code, relaxed["status"]) == (0, "PASS")


# ---------------------------------------------------------------------------
# ASME V&V 20 table: text, json and plot renderers
# ---------------------------------------------------------------------------

TABLE_ROWS = [
    {"label": "inside", "f_sim": 1.0, "f_exp": 1.02, "u_num": 0.02, "u_input": 0.01},
    {"label": "outside", "f_sim": 1.0, "f_exp": 1.05, "u_num": 0.03, "u_input": 0.03},
]


def _table(tmp_path: Path) -> Path:
    path = tmp_path / "table.json"
    path.write_text(json.dumps({"rows": TABLE_ROWS}), encoding="utf-8")
    return path


def test_vv20_table_text_renderer(tmp_path, capsys):
    code = cli.main(["--table", str(_table(tmp_path)), "--format", "table"])
    out = capsys.readouterr().out
    assert code == 2  # one row fails by construction
    assert "ASME V&V 20 table" in out and "PASS=1 FAIL=1" in out


def test_vv20_table_plot_renderer(tmp_path, capsys):
    png = tmp_path / "vv20.png"
    code = cli.main(["--table", str(_table(tmp_path)), "--format", "plot",
                     "--plot-out", str(png)])
    out = capsys.readouterr().out
    assert code == 2
    assert png.exists() and png.stat().st_size > 1024
    assert "PASS=1 FAIL=1" in out


def test_vv20_table_accepts_the_documented_alias(tmp_path, capsys):
    code = cli.main(["--vv20-table", str(_table(tmp_path)), "--json"])
    record = json.loads(capsys.readouterr().out)
    assert code == 2 and record["n_fail"] == 1


def test_cli_usage_errors_are_reported(tmp_path, capsys):
    assert cli.main(["--obs", str(tmp_path / "nope.json")]) == cli.EXIT_USAGE
    capsys.readouterr()
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"levels": [{"h": 1.0, "f": 1.0}], "quantity": "x"}),
                   encoding="utf-8")
    native = tmp_path / "native.json"
    native.write_text(json.dumps({"levels": [{"h": 1.0, "f": 1.0}]}), encoding="utf-8")
    assert cli.main(["--benchmark", str(native), "--repo", "tensornotreal"]) == cli.EXIT_USAGE
    assert "unknown repo profile" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# adapters: fail-closed paths
# ---------------------------------------------------------------------------


def test_unknown_repo_profile_is_rejected():
    with pytest.raises(adapters.AdapterError, match="unknown repo profile"):
        adapters.extract_series({"levels": []}, repo="tensorxyz")


def test_study_selection_errors():
    doc = {
        "manufactured": {"levels": [{"h": 1.0, "value": 1.0}, {"h": 0.5, "value": 0.9}]},
    }
    with pytest.raises(adapters.AdapterError, match="not present"):
        adapters.extract_series(doc, repo="auto", study="bending")


def test_nearest_study_is_chosen_when_value_key_matches():
    doc = {
        "bending": {"levels": [{"h": 1.0, "value": 1.0}, {"h": 0.5, "value": 0.9}]},
        "manufactured": {"levels": [{"mesh_size": 1.0, "strain_energy": 1.0},
                                    {"mesh_size": 0.5, "strain_energy": 0.9}]},
    }
    series = adapters.extract_series(doc, value_key="strain_energy")
    assert series.study == "manufactured"
    assert series.levels[0].h == 1.0


def test_level_without_length_or_count_is_rejected():
    doc = {"levels": [{"value": 1.0}, {"value": 0.9}]}
    with pytest.raises(adapters.AdapterError, match="no list of level mappings"):
        adapters.extract_series(doc)


def test_grid_count_is_converted_and_bad_counts_are_rejected():
    doc = {"levels": [{"resolution": 16, "value": 1.0}, {"resolution": 32, "value": 0.9}]}
    series = adapters.extract_series(doc, repo="tensorlbm")
    assert [lvl.h for lvl in series.levels] == [1 / 16, 1 / 32]  # coarse -> fine
    with pytest.raises(adapters.AdapterError, match="cannot turn"):
        adapters.extract_series({"levels": [{"resolution": "x", "value": 1.0},
                                            {"resolution": 8, "value": 0.9}]})


def test_explicit_value_key_must_exist():
    with pytest.raises(adapters.AdapterError, match="has no 'missing' field"):
        adapters.extract_series(
            {"levels": [{"h": 1.0, "value": 1.0}, {"h": 0.5, "value": 0.9}]},
            value_key="missing")
    with pytest.raises(adapters.AdapterError, match="no list of level mappings"):
        adapters.extract_series(
            {"levels": [{"h": 1.0, "note": "x"}, {"h": 0.5, "note": "y"}]},
            repo="generic")
    # a non-numeric field selected as the observable fails closed (ValueError)
    with pytest.raises(ValueError):
        adapters.extract_series(
            {"levels": [{"h": 1.0, "value": 1.0, "note": "x"},
                        {"h": 0.5, "l2_error": 0.9, "note": "y"}]},
            repo="generic", value_key="note")


def test_deeply_nested_document_is_rejected():
    node: dict = {"levels": "not-a-list"}
    for _ in range(8):
        node = {"wrapper": node}
    with pytest.raises(adapters.AdapterError):
        adapters.extract_series(node)


def test_sigma_fields_are_carried_through():
    doc = {"levels": [{"h": 1.0, "value": 1.0, "std": 0.01},
                      {"h": 0.5, "value": 0.9, "stderr": 0.02}]}
    series = adapters.extract_series(doc)
    assert [lvl.sigma for lvl in series.levels] == [0.01, 0.02]  # coarse -> fine


def test_quantity_selects_the_value_field():
    doc = {"levels": [{"h": 1.0, "value": 1.0, "drag": 2.0},
                      {"h": 0.5, "value": 0.9, "drag": 1.9}]}
    series = adapters.extract_series(doc, quantity="drag")
    assert series.quantity == "drag"
    assert [lvl.f for lvl in series.levels] == [2.0, 1.9]  # coarse -> fine


def test_write_gate_payload_roundtrip(tmp_path):
    doc = {"repo": "tensorlbm", "quantity": "l2_error",
           "levels": [{"resolution": 8, "value": 1e-2}, {"resolution": 16, "value": 2.5e-3}]}
    series = adapters.extract_series(doc)
    out = tmp_path / "payload.json"
    payload = adapters.write_gate_payload(series, out, expected_order=2.0)
    assert payload == json.loads(out.read_text())
    assert payload["expected_order"] == 2.0
    assert payload["source"]["repo"] == "tensorlbm"
    assert payload["levels"][0]["h"] == pytest.approx(1 / 8)


def test_load_payload_reads_a_file(tmp_path):
    path = tmp_path / "doc.json"
    path.write_text(json.dumps({"levels": []}), encoding="utf-8")
    assert adapters.load_payload(str(path)) == {"levels": []}


# ---------------------------------------------------------------------------
# policy plumbing
# ---------------------------------------------------------------------------


def test_gci_absolute_mode_and_cli_flag(tmp_path, capsys):
    from tensorphysics.vv.gci import grid_convergence_study

    levels = [Observation(h=h, f=0.5 * h ** 2) for h in (1.0, 0.5, 0.25)]
    relative = grid_convergence_study(levels)
    absolute = grid_convergence_study(levels, policy=GatePolicy(gci_absolute=True))
    assert relative.gci_is_relative is True
    assert absolute.gci_is_relative is False
    assert absolute.gci_fine == pytest.approx(
        relative.gci_fine * abs(levels[-1].f), rel=1e-12)

    path = tmp_path / "obs.json"
    path.write_text(json.dumps(_obs_payload(
        [(1.0, 0.5), (0.5, 0.125), (0.25, 0.03125)], expected_order=2.0)), encoding="utf-8")
    code = cli.main(["--obs", str(path), "--json", "--gci-absolute"])
    record = json.loads(capsys.readouterr().out)
    assert code in (0, 1, 2)
    assert record["study"]["gci_is_relative"] is False


def test_policy_carries_the_noise_certificate_switch():
    policy = GatePolicy()
    assert policy.pass_requires_noise_certificate is True
    relaxed = policy.merged(pass_requires_noise_certificate=False)
    assert relaxed.pass_requires_noise_certificate is False
    verdict = evaluate([Observation(h=h, f=3.0 + 0.5 * h * h) for h in (1.0, 0.5, 0.25)])
    assert verdict.status == "WARN"

# ---------------------------------------------------------------------------
# remaining adapter branches and payload plumbing
# ---------------------------------------------------------------------------


def test_nested_document_under_a_non_container_key():
    doc = {"meta": {"note": "wrapped"}, "results": {"inner": {
        "grids": [{"dx": 1.0, "l2_error": 1.0}, {"dx": 0.5, "l2_error": 0.25}]}}}
    series = adapters.extract_series(doc)
    assert [lvl.h for lvl in series.levels] == [1.0, 0.5]
    assert series.metadata["container"] == "grids"


def test_mms_study_is_preferred_over_alphabetical_order():
    doc = {
        "alpha": {"levels": [{"h": 1.0, "value": 1.0}, {"h": 0.5, "value": 0.9}]},
        "zeta_mms": {"levels": [{"h": 1.0, "value": 2.0}, {"h": 0.5, "value": 1.8}]},
    }
    series = adapters.extract_series(doc)
    assert series.study == "zeta_mms"


def test_value_key_scan_skips_studies_without_that_field():
    doc = {
        "bending": {"levels": [{"h": 1.0, "value": 1.0}, {"h": 0.5, "value": 0.9}]},
        "manufactured": {"levels": [{"mesh_size": 1.0, "strain_energy": 1.0},
                                    {"mesh_size": 0.5, "strain_energy": 0.9}]},
    }
    series = adapters.extract_series(doc, value_key="strain_energy")
    assert series.study == "manufactured"
    assert series.metadata["value_key"] == "strain_energy"


def test_to_gate_payload_carries_a_validation_block():
    series = adapters.extract_series(
        {"levels": [{"h": 1.0, "value": 1.0}, {"h": 0.5, "value": 0.9}]})
    payload = adapters.to_gate_payload(series, validation={"f_exp": 1.0, "u_input": 0.01})
    assert payload["validation"] == {"f_exp": 1.0, "u_input": 0.01}
    assert payload["schema"] == "tensorphysics.vv.observations/1"


def test_vv_module_entry_point_delegates_to_the_cli():
    import tensorphysics.vv.__main__ as entry

    assert entry.main is cli.main


def test_cli_reads_observations_from_stdin(monkeypatch, capsys):
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(_obs_payload(
        [(1.0, 3.5), (0.5, 3.125), (0.25, 3.03125)], expected_order=2.0))))
    code = cli.main(["--obs", "-", "--json", "--allow-undeclared-noise"])
    record = json.loads(capsys.readouterr().out)
    assert code == 0 and record["study"]["order_p"] == pytest.approx(2.0, abs=1e-9)


def test_vv20_plot_without_matplotlib_is_a_usage_error(tmp_path, capsys, monkeypatch):
    def boom():
        raise RuntimeError("plot output needs matplotlib")

    monkeypatch.setattr(cli, "_pyplot", boom)
    code = cli.main(["--table", str(_table(tmp_path)), "--format", "plot"])
    assert code == cli.EXIT_USAGE
    assert "matplotlib" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# observation mapping / degenerate formulas
# ---------------------------------------------------------------------------


def test_observation_from_mapping_aliases_and_errors():
    obs = Observation.from_mapping({"mesh_size": 0.5, "l2_error": 1e-3, "label": "n",
                                    "sigma": 1e-6})
    assert (obs.h, obs.f, obs.label, obs.sigma) == (0.5, 1e-3, "n", 1e-6)
    assert Observation.from_mapping({"dx": 0.25, "value": 1.0}).h == 0.25
    with pytest.raises(ValueError, match="observation needs h and f"):
        Observation.from_mapping({"h": 1.0})
    with pytest.raises(ValueError, match="sigma must be finite"):
        Observation(h=1.0, f=1.0, sigma=-1.0)


def test_observation_as_dict_keeps_optional_fields():
    assert Observation(h=1.0, f=2.0).as_dict() == {"h": 1.0, "f": 2.0}
    assert Observation(h=1.0, f=2.0, label="a", sigma=0.1).as_dict() == {
        "h": 1.0, "f": 2.0, "label": "a", "sigma": 0.1}


def test_degenerate_richardson_and_gci_arguments_are_rejected():
    from tensorphysics.vv.gci import gci, richardson_extrapolated

    levels = [Observation(h=1.0, f=1.0), Observation(h=0.5, f=2.0)]
    with pytest.raises(ValueError, match="degenerate Richardson factor"):
        richardson_extrapolated(levels, 0.0)  # rho**0 - 1 == 0
    with pytest.raises(ValueError, match="degenerate GCI denominator"):
        gci(Observation(h=1.0, f=2.0), Observation(h=0.5, f=1.0), -1.0)


def test_non_uniform_study_reports_the_least_squares_difference():
    from tensorphysics.vv.gci import evaluate

    model = lambda h: 3.0 + h * h + 0.1 * h ** 3  # noqa: E731  (not a pure power law)
    levels = [Observation(h=h, f=model(h)) for h in (1.0, 0.5, 0.25, 0.1)]
    study = evaluate(levels).study
    assert study.uniform_refinement is False
    assert study.order_source == "least-squares"
    assert study.order_p is not None
    assert study.fit_residual_rel is not None
    assert any("least-squares" in note for note in study.notes)
