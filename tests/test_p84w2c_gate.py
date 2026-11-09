"""P84-W2C campaign gate tests (module: ``tensorphysics.campaign.p84w2c_gate``).

Three layers are exercised:

* unit: metric lookup, artefact classification, pytest normalisation, criterion
  comparators, lineage adjudication;
* bilateral contracts: a synthetic consumer module is parsed with ``ast`` and the
  numeric anchor is compared, including tampered-constant negative controls;
* end-to-end: a synthetic ten-node campaign (fixtures only, no domain numbers
  from any real repository) is driven through ``run_gate`` and must produce a
  schema-valid ``campaign_gate_report.v1``, a >=3000 character markdown and a
  PASS on C1..C4.  A second run with one empty node must fail honestly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorphysics.campaign import schema as S  # noqa: E402
from tensorphysics.campaign import schema_defs as SD  # noqa: E402
from tensorphysics.campaign.p84w2c_gate import collect as C  # noqa: E402
from tensorphysics.campaign.p84w2c_gate import contracts as K  # noqa: E402
from tensorphysics.campaign.p84w2c_gate import criteria as J  # noqa: E402
from tensorphysics.campaign.p84w2c_gate import lineage as L  # noqa: E402
from tensorphysics.campaign.p84w2c_gate import spec as P  # noqa: E402
from tensorphysics.campaign.p84w2c_gate.gate import run_gate  # noqa: E402
from tensorphysics.campaign.p84w2c_gate.report import SECTIONS  # noqa: E402


# ---------------------------------------------------------------------------
# specification integrity
# ---------------------------------------------------------------------------
def test_spec_declares_ten_nodes_and_contracts():
    assert len(P.NODES) == 10
    assert [n.node_key for n in P.NODES] == [f"P84W2C-N{i:02d}" for i in range(1, 11)]
    assert len({n.mission_id for n in P.NODES}) == 10
    assert len(P.CONTRACTS) == 3
    by_id = {c.contract_id: c for c in P.CONTRACTS}
    assert by_id["aero-wind"].required and by_id["amp-eexi"].required
    assert not by_id["sediment-ukc"].required  # advisory: consumer endpoint not built
    for contract in P.CONTRACTS:
        assert contract.producer_node in P.NODE_BY_KEY
        assert contract.producer_schema in SD.ARTIFACT_SCHEMAS
        for field in contract.producer_required:
            assert field in SD.schema_for(contract.producer_schema)["properties"]
    for node in P.NODES:
        assert node.tests and node.artifact_globs and node.criteria
        for crit in node.criteria:
            assert crit.comparator in P.COMPARATORS


def test_new_schemas_are_registered_and_closed():
    for name in ("campaign_gate_report.v1", "aero_polar.v1", "amp_emissions.v1"):
        schema = SD.schema_for(name)
        assert schema["type"] == "object"
        assert schema.get("additionalProperties") is False
    fp_a = SD.schema_fingerprint("campaign_gate_report.v1")
    fp_b = SD.schema_fingerprint("campaign_gate_report.v1")
    assert fp_a == fp_b and len(fp_a) == 64
    with pytest.raises(S.SchemaError):
        SD.validate_artifact("aero_polar.v1", {"schema_name": "aero_polar.v1"})


# ---------------------------------------------------------------------------
# collectors
# ---------------------------------------------------------------------------
def test_classify_artifact_json_and_png(tmp_path: Path):
    good = tmp_path / "ok.json"
    good.write_text('{"a": 1}', encoding="utf-8")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    png = tmp_path / "fig.png"
    png.write_bytes(C.PNG_MAGIC + b"data" + C.PNG_IEND)
    assert C.classify_artifact(good)["parseable"] is True
    assert C.classify_artifact(bad)["parseable"] is False
    assert C.classify_artifact(png)["parseable"] is True
    assert C.classify_artifact(tmp_path / "missing.json")["exists"] is False
    assert len(C.classify_artifact(good)["sha256"]) == 64


def test_find_metric_alias_nesting_and_types():
    payloads = [("a.json", {"outer": {"C_D0_rel_error": 0.07},
                            "list": [{"mass_imbalance": 1e-9}],
                            "flag": {"negative_control_flip": True}})]
    assert C.find_metric(payloads, ("cd0_rel_error",))[0] == pytest.approx(0.07)
    assert C.find_metric(payloads, ("mass_imbalance",))[0] == pytest.approx(1e-9)
    value, source = C.find_metric(payloads, ("negative_control_flip",))
    assert value is True and source.endswith("negative_control_flip")
    assert C.find_metric(payloads, ("absent_key",)) == (None, "")


def test_run_pytest_executes_and_normalises(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "tests" / "test_ok.py").write_text(
        "def test_one():\n    assert 1 + 1 == 2\n\ndef test_two():\n    assert True\n",
        encoding="utf-8")
    record = C.run_pytest(repo, ("tests/test_ok.py",), tmp_path / "ev" / "ok.pytest.json",
                          timeout=120)
    assert record["mode"] == "executed" and record["status"] == "PASS"
    assert record["collected"] == 2 and record["failed"] == 0
    assert record["parseable"] is True
    assert json.loads(Path(record["record_path"]).read_text())["passed"] == 2


def test_run_pytest_failing_and_missing_are_honest(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "tests" / "test_bad.py").write_text(
        "def test_fail():\n    assert False\n", encoding="utf-8")
    failed = C.run_pytest(repo, ("tests/test_bad.py",), tmp_path / "ev" / "bad.json",
                          timeout=120)
    assert failed["status"] == "FAIL" and failed["failed"] == 1

    empty = tmp_path / "empty"
    (empty / "tests").mkdir(parents=True)
    (empty / "tests" / "test_placeholder.py").write_text("# empty\n", encoding="utf-8")
    collected = C.run_pytest(empty, ("tests/test_needed.py",), tmp_path / "ev" / "ph.json",
                             timeout=120)
    assert collected["mode"] == "collect-only" and collected["status"] == "FAIL"
    assert "not delivered" in collected["detail"]

    absent = C.collect_node(tmp_path / "nope", P.NODE_BY_KEY["P84W2C-N01"],
                            tmp_path / "ev", run_suite=True)
    assert absent["root_present"] is False
    assert absent["pytest"]["status"] == "NOT_SUPPORTED"
    assert json.loads(Path(absent["pytest"]["record_path"]).read_text())["mode"] == "missing"


# ---------------------------------------------------------------------------
# judging
# ---------------------------------------------------------------------------
def test_criterion_comparators_and_withheld():
    crit = P.NODES[0].criteria[0]  # rel_dev_le vs 2*pi, 5%
    assert J.judge_criterion(crit, P.TWO_PI * 1.01)["status"] == "PASS"
    assert J.judge_criterion(crit, P.TWO_PI * 1.20)["status"] == "FAIL"
    assert J.judge_criterion(crit, None)["status"] == "WITHHELD"

    rng = next(c for n in P.NODES for c in n.criteria if c.comparator == "in_range")
    lo, hi = rng.threshold
    assert J.judge_criterion(rng, (lo + hi) / 2)["status"] == "PASS"
    assert J.judge_criterion(rng, hi * 10)["status"] == "FAIL"

    le = next(c for c in P.NODES[0].criteria if c.comparator == "le")
    assert J.judge_criterion(le, 0.0)["status"] == "PASS"


def test_node_verdict_requires_everything():
    base = {"root_present": True,
            "artifacts": [{"kind": "json", "parseable": True}],
            "pytest": {"status": "PASS", "mode": "executed"},
            "criteria": [{"status": "PASS"}]}
    assert J.node_verdict(base) == "PASS"
    assert J.node_verdict({**base, "criteria": [{"status": "WITHHELD"}]}) == "WITHHELD"
    assert J.node_verdict({**base, "pytest": {"status": "FAIL", "mode": "executed"}}) == "FAIL"
    assert J.node_verdict({**base, "artifacts": []}) == "WITHHELD"
    assert J.node_verdict({**base, "root_present": False}) == "NOT_SUPPORTED"


def test_lineage_adjudication_verified_unverified_new():
    parents = {"1": {"name": "N01: A 面", "status": "done", "mission_id": 101},
               "2": {"name": "N02: B 面", "status": "in-flight", "mission_id": 102}}
    nodes = [{"node_key": "X1", "title": "A 面"}, {"node_key": "X2", "title": "B 面"},
             {"node_key": "X3", "title": "C 面"}]
    titles = {"X1": "N01: A 面", "X2": "N02: B 面", "X3": ""}
    results, summary = L.adjudicate_all(nodes, parents, titles)
    assert [r["verdict"] for r in results] == [
        "inherited_verified", "inherited_unverified", "seeded_new"]
    assert summary["inherited_verified"] == 1 and summary["seeded_new"] == 1


# ---------------------------------------------------------------------------
# bilateral contracts
# ---------------------------------------------------------------------------
CONSUMER_STABILITY = '''
RHO_AIR = 1.225
WIND_PRESSURE_IS_CODE = 504.0

class DynamicStabilityInput:
    wind_speed: float = 0.0
    wind_pressure: float | None = None
    wind_area: float = 0.0
    wind_lever: float = 0.0
    wind_pressure_model: str = "dynamic_pressure"

class Solver:
    rho_air = RHO_AIR
    def resolve_wind_pressure(self, input_data):
        if input_data.wind_pressure is not None:
            return float(input_data.wind_pressure)
        return (0.5 * self.rho_air * input_data.wind_speed ** 2, "dynamic_pressure")
'''

CONSUMER_GREEN = '''
class EediRequest:
    capacity_t: float = 1.0
    v_ref_kn: float = 12.0
    p_me_kw: float = 1000.0
    p_eff_kw: float = 0.0

class EexiRequest:
    capacity_t: float = 1.0
    v_ref_kn: float = 12.0
    p_me_kw: float = 1000.0
    p_eff_kw: float = 0.0
'''


def _make_consumer(tmp_path: Path, stability: str = CONSUMER_STABILITY) -> Path:
    root = tmp_path / "ship-platform"
    (root / "backend" / "api").mkdir(parents=True)
    (root / "backend" / "api" / "dynamic_stability.py").write_text(stability, encoding="utf-8")
    (root / "backend" / "api" / "green_shipping.py").write_text(CONSUMER_GREEN, encoding="utf-8")
    (root / "backend" / "api" / "calculations.py").write_text("def margin(ukc):\n    return ukc\n",
                                                             encoding="utf-8")
    return root


def _polar() -> dict:
    return {"schema_name": "aero_polar.v1",
            "alpha_deg": [-2.0, 0.0, 2.0, 4.0],
            "C_L": [-0.219, 0.0, 0.219, 0.438],
            "C_D": [0.008, 0.007, 0.008, 0.011],
            "rho_air_kg_m3": 1.225, "chord_m": 1.0}


def test_aero_wind_contract_passes_and_detects_tampering(tmp_path: Path):
    consumer = _make_consumer(tmp_path)
    payloads = [("polar.json", _polar())]
    records = K.evaluate_all_contracts({"ship-performance-platform": consumer},
                                       {"P84W2C-N01": payloads})
    by_id = {r["contract_id"]: r for r in records}
    aero = by_id["aero-wind"]
    assert aero["status"] == "PASS", aero["gaps"]
    assert aero["data_conformance"] == "PASS"
    names = {c["name"]: c["status"] for c in aero["checks"]}
    assert names["consumer_law_shape"] == "PASS"
    assert names["constant:RHO_AIR"] == "PASS"
    assert names["numeric_anchor"] == "PASS"
    assert names["is_code_cross_anchor"] == "PASS"

    tampered = _make_consumer(tmp_path / "tampered", CONSUMER_STABILITY.replace(
        "RHO_AIR = 1.225", "RHO_AIR = 1.2"))
    records = K.evaluate_all_contracts({"ship-performance-platform": tampered},
                                       {"P84W2C-N01": payloads})
    aero_bad = {r["contract_id"]: r for r in records}["aero-wind"]
    assert aero_bad["status"] == "FAIL"
    assert any(c["name"] == "constant:RHO_AIR" and c["status"] == "FAIL"
               for c in aero_bad["checks"])


def test_aero_contract_without_polar_artifact_is_withheld_data(tmp_path: Path):
    consumer = _make_consumer(tmp_path)
    records = K.evaluate_all_contracts({"ship-performance-platform": consumer},
                                       {"P84W2C-N01": []})
    aero = {r["contract_id"]: r for r in records}["aero-wind"]
    assert aero["status"] == "PASS"          # contract-level is verifiable
    assert aero["data_conformance"] == "WITHHELD"  # data-level is not invented


def test_polar_data_conformance_negative_controls():
    assert K._polar_conformance(_polar())[0] == "PASS"
    bad_alpha = _polar()
    bad_alpha["alpha_deg"] = [0.0, 2.0, 2.0, 4.0]
    assert K._polar_conformance(bad_alpha)[0] == "FAIL"
    asymmetric = _polar()
    asymmetric["C_L"] = [-0.219, 0.05, 0.219, 0.438]
    assert K._polar_conformance(asymmetric)[0] == "FAIL"
    negative_cd = _polar()
    negative_cd["C_D"] = [0.008, -0.007, 0.008, 0.011]
    assert K._polar_conformance(negative_cd)[0] == "FAIL"


def test_amp_contract_and_conformance():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        consumer = _make_consumer(Path(td))
        good = {"schema_name": "amp_emissions.v1", "aux_power_kw": 1200.0, "fuel_t": 5.0,
                "co2_t": 15.57, "co2_reduction_t": 12.0, "p_eff_kw": 100.0,
                "grid_factor_kg_per_kwh": 0.581}
        records = K.evaluate_all_contracts({"ship-performance-platform": consumer},
                                           {"P84W2C-N06": [("amp.json", good)]})
        amp = {r["contract_id"]: r for r in records}["amp-eexi"]
        assert amp["status"] == "PASS" and amp["data_conformance"] == "PASS"
        bad = dict(good, co2_t=99.0)  # implied carbon factor 19.8 -> impossible
        assert K._amp_conformance(bad)[0] == "FAIL"


# ---------------------------------------------------------------------------
# end-to-end campaign
# ---------------------------------------------------------------------------
_METRICS = {
    "P84W2C-N01": {"lift_slope": P.TWO_PI * 1.01, "cl_at_zero": 1e-6,
                   "grid_convergence_index": 0.005, "mass_imbalance": 1e-8,
                   "cd0_rel_error": 0.08, "stall_angle_dev_deg": 1.0},
    "P84W2C-N02": {"log_law_max_dev": 0.02, "re_tau_rel_error": 0.01,
                   "grid_convergence_index": 0.02, "sgs_stress_fraction": 0.2,
                   "mass_imbalance": 1e-10},
    "P84W2C-N03": {"linear_residual": 1e-12, "current_conservation_error": 1e-10,
                   "protection_ratio": 0.98, "grid_convergence_index": 0.01},
    "P84W2C-N04": {"shields_rel_error": 0.10, "rouse_z_rel_error": 0.02,
                   "mass_drift": 1e-8, "scour_ratio": 1.7},
    "P84W2C-N05": {"repose_angle_dev_deg": 5.0, "beverloo_r2": 0.97,
                   "beverloo_c_rel_error": 0.10, "mass_conservation_error": 1e-8,
                   "energy_residual": 1e-10},
    "P84W2C-N06": {"aux_load_balance_error": 1e-12, "co2_conservation_error": 1e-12,
                   "co2_reduction_t": 100.0, "grid_factor_rel_error": 0.02},
    "P84W2C-N07": {"energy_residual": 1e-8, "eta_over_carnot": 0.5, "min_pinch_k": 15.0,
                   "eedi_dual_rel_diff": 1e-9},
    "P84W2C-N08": {"acceleration_dual_rel_error": 1e-14, "force_moment_residual": 1e-11,
                   "lashing_monotonic_r2": 0.995},
    "P84W2C-N09": {"static_equilibrium_error": 1e-9, "two_model_mmax_rel_diff": 0.005,
                   "degradation_rel_error": 0.005},
    "P84W2C-N10": {"unit_chain_rel_error": 1e-14, "symmetry_rel_error": 1e-12,
                   "heeling_monotonic_r2": 0.995},
}

_ARTIFACT_DIR = {
    "P84W2C-N01": "artifacts/naca0012_aero", "P84W2C-N02": "artifacts/channel_les",
    "P84W2C-N03": "artifacts/cathodic_protection", "P84W2C-N04": "artifacts/p84_w1",
    "P84W2C-N05": "artifacts/hopper_discharge", "P84W2C-N06": "artifacts/shore_power",
    "P84W2C-N07": "artifacts/whr_orc", "P84W2C-N08": "artifacts/cargo_securing",
    "P84W2C-N09": "artifacts/drydock", "P84W2C-N10": "artifacts/grain_stability",
}


def _build_node(root: Path, node_key: str, with_polar: bool = False,
                with_amp: bool = False, deliver: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    declared = P.NODE_BY_KEY[node_key].tests[0]
    (root / declared).parent.mkdir(parents=True, exist_ok=True)
    (root / declared).write_text("def test_contract():\n    assert True\n", encoding="utf-8")
    if not deliver:
        return root
    art = root / _ARTIFACT_DIR[node_key]
    art.mkdir(parents=True, exist_ok=True)
    (art / "case.json").write_text(json.dumps(_METRICS[node_key]), encoding="utf-8")
    if with_polar:
        (art / "polar.json").write_text(json.dumps(_polar()), encoding="utf-8")
    if with_amp:
        (art / "amp.json").write_text(json.dumps({
            "schema_name": "amp_emissions.v1", "aux_power_kw": 1200.0, "fuel_t": 5.0,
            "co2_t": 15.57, "co2_reduction_t": 12.0, "p_eff_kw": 100.0,
            "grid_factor_kg_per_kwh": 0.581}), encoding="utf-8")
    return root


@pytest.fixture()
def campaign(tmp_path: Path) -> dict:
    node_parent = tmp_path / "nodes"
    roots = {}
    for index, node in enumerate(P.NODES):
        roots[node.node_key] = _build_node(node_parent / f"m{node.mission_id}", node.node_key,
                                           with_polar=(node.node_key == "P84W2C-N01"),
                                           with_amp=(node.node_key == "P84W2C-N06"))
    consumer = _make_consumer(tmp_path / "consumer")
    parents = {str(i): {"name": node.lineage_title, "status": "done" if i % 2 else "in-flight",
                        "mission_id": 1000 + i}
               for i, node in enumerate(P.NODES)}
    return {"roots": roots, "consumer": consumer, "parents": parents, "tmp": tmp_path}


def test_end_to_end_gate_passes_with_full_evidence(campaign):
    out = campaign["tmp"] / "artifacts" / "gate"
    doc = campaign["tmp"] / "docs" / "P84W2C_CAMPAIGN_GATE.md"
    report = run_gate(campaign["roots"], out, consumer_roots={
        "ship-performance-platform": campaign["consumer"]},
        platform={"available": True, "plan_id": 385, "nodes": {}, "missions": {}},
        parent_nodes=campaign["parents"], doc_path=doc, pytest_timeout=180,
        make_figure=False, reproduce=("python -m tensorphysics.campaign.p84w2c_gate",))

    SD.validate_artifact("campaign_gate_report.v1", report)  # explicit schema check
    assert report["overall_status"] == "PASS"
    assert [c["status"] for c in report["criteria"]] == ["PASS"] * 4
    assert report["blocker_count"] == 0
    assert len(report["fingerprint"]) == 64
    assert doc.is_file() and len(doc.read_text(encoding="utf-8")) >= P.MIN_REPORT_CHARS
    for section in SECTIONS:
        assert section in doc.read_text(encoding="utf-8")
    written = json.loads((out / "campaign_gate_report.json").read_text(encoding="utf-8"))
    assert written["schema_name"] == "campaign_gate_report.v1"
    assert written["fingerprint"] == report["fingerprint"]
    assert sum(len(n["criteria"]) for n in written["nodes"]) > 30
    assert all(n["verdict"] == "PASS" for n in written["nodes"])
    aero = {c["contract_id"]: c for c in written["contracts"]}["aero-wind"]
    assert aero["status"] == "PASS" and aero["data_conformance"] == "PASS"
    assert written["lineage_summary"]["inherited_verified"] == 5


def test_end_to_end_gate_fails_honestly_on_missing_node(campaign):
    # declared suite is delivered and green, but no artifact JSON is -> WITHHELD
    campaign["roots"]["P84W2C-N07"] = _build_node(
        campaign["tmp"] / "nodes" / "empty", "P84W2C-N07", deliver=False)
    out = campaign["tmp"] / "artifacts" / "gate_fail"
    doc = campaign["tmp"] / "docs_fail" / "P84W2C_CAMPAIGN_GATE.md"
    report = run_gate(campaign["roots"], out, consumer_roots={
        "ship-performance-platform": campaign["consumer"]},
        parent_nodes=campaign["parents"], doc_path=doc, pytest_timeout=180,
        make_figure=False)

    assert report["overall_status"] == "FAIL"
    node = next(n for n in report["nodes"] if n["node_key"] == "P84W2C-N07")
    assert node["verdict"] == "WITHHELD"
    assert node["gaps"]
    assert any(gap["node_key"] == "P84W2C-N07" for gap in report["gap_analysis"])
    # C1 (pytest parseability) is green -- the suite *was* delivered and run --
    # yet the campaign still fails: node acceptance is part of the conjunction.
    c1 = next(c for c in report["criteria"] if c["id"] == "C1")
    assert c1["status"] == "PASS" and c1["metrics"]["pytest_green"] == 10
    assert any("未落盘任何 JSON 产物" in gap["reason"] for gap in report["gap_analysis"]
               if gap["node_key"] == "P84W2C-N07")
    assert len(doc.read_text(encoding="utf-8")) >= P.MIN_REPORT_CHARS
    SD.validate_artifact("campaign_gate_report.v1", report)


def test_gate_figure_is_written_when_matplotlib_available(campaign):
    pytest.importorskip("matplotlib")
    out = campaign["tmp"] / "artifacts" / "gate_fig"
    doc = campaign["tmp"] / "docs_fig" / "P84W2C_CAMPAIGN_GATE.md"
    report = run_gate(campaign["roots"], out, consumer_roots={
        "ship-performance-platform": campaign["consumer"]},
        parent_nodes=campaign["parents"], doc_path=doc, pytest_timeout=180,
        make_figure=True)
    figure = report["artifacts_index"]["figure_png"]
    assert figure and Path(figure).is_file()
    assert Path(figure).read_bytes()[:8] == C.PNG_MAGIC
    assert report["markdown"]["figure"]

# ---------------------------------------------------------------------------
# criterion-number resolver (candidate precedence + conservative aggregation)
# ---------------------------------------------------------------------------
def test_candidates_canonical_block_list_form():
    crit = P.NODE_BY_KEY["P84W2C-N06"].criteria[0]
    payload = [("r.json", {"criteria": [
        {"id": "C1", "value": 0.0, "pass": True},
        {"id": "C2", "value": 1e-13, "pass": True}]})]
    cands = C.criterion_candidates(payload, crit)
    assert [c["value"] for c in cands] == [0.0]
    assert J.judge_criterion(crit, cands)["status"] == "PASS"


def test_candidates_scoped_block_and_percent_normalisation():
    crit = next(c for c in P.NODE_BY_KEY["P84W2C-N06"].criteria if c.cid == "C5")
    payload = [("r.json", {"criteria": [{"id": "C5", "value": {
        "n_ports": 5, "worst_rel_err_pct": 1.9867, "all_within_tol": True}}]})]
    cands = C.criterion_candidates(payload, crit)
    assert cands and abs(cands[0]["value"] - 0.019867) < 1e-12  # percent -> fraction
    assert J.judge_criterion(crit, cands)["status"] == "PASS"

    scoped = P.NODE_BY_KEY["P84W2C-N08"].criteria[0]
    payload = [("m.json", {"C1_double_implementation": {
        "a": {"max_relative_error": 2.2e-16}, "b": {"max_relative_error": 5.0e-16}}})]
    cands = C.criterion_candidates(payload, scoped)
    worst = J.judge_criterion(scoped, cands)
    assert worst["status"] == "PASS" and abs(worst["value"] - 5.0e-16) < 1e-28


def test_worst_case_aggregation_is_conservative():
    crit = next(c for c in P.NODE_BY_KEY["P84W2C-N08"].criteria if c.comparator == "ge")
    good = J.judge_criterion(crit, [{"value": 1.0, "source": "a"},
                                    {"value": 0.5, "source": "b"}])
    assert good["status"] == "FAIL" and good["value"] == pytest.approx(0.5)
    err = next(c for c in P.NODE_BY_KEY["P84W2C-N01"].criteria if c.comparator == "le")
    worst = J.judge_criterion(err, [{"value": 1e-8}, {"value": 3e-4}])
    assert worst["status"] == "FAIL" and worst["value"] == pytest.approx(3e-4)
    rng = next(c for n in P.NODES for c in n.criteria if c.comparator == "in_range")
    lo, hi = rng.threshold
    assert J.judge_criterion(rng, [{"value": (lo + hi) / 2}, {"value": hi * 2}])["status"] == "FAIL"


def test_no_pytest_mode_reuses_stored_record(tmp_path: Path):
    spec = P.NODE_BY_KEY["P84W2C-N01"]
    repo = tmp_path / "repo"
    declared = repo / spec.tests[0]
    declared.parent.mkdir(parents=True)
    declared.write_text("def test_a():\n    assert True\n", encoding="utf-8")
    evidence = tmp_path / "ev"
    first = C.collect_node(repo, spec, evidence, run_suite=True, timeout=120)
    assert first["pytest"]["mode"] == "executed"
    # a synthetic node whose declared suite does not exist: record stays honest
    other = tmp_path / "other"
    other.mkdir()
    reused = C.collect_node(other, spec, evidence, run_suite=False)
    assert reused["pytest"]["mode"] == "executed"  # stored record re-read, not laundered
    assert reused["pytest"].get("reused") is True
