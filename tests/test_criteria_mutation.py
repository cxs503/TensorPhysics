"""Acceptance tests for the criteria falsifiability ledger (criteria_mutation).

Each test pins one machine-checkable claim of the ledger:

* every clean case still reproduces the shipped gate's expected verdict
  (a criterion that cannot be instantiated would be an accounting ghost);
* every mutant really changes the case it mutates (no vacuous kills);
* the whole-ledger kill rate clears the >= 90 % KPI, and the five tautological
  negative-control criteria score 0/5 and are flagged unfalsifiable;
* the M1 eps-ladder (1 % -> 3 % -> 15 %) never loses kills;
* the merge gate rejects a new criterion that ships without a killer test and
  accepts the same criterion once a mutant-killing declaration exists;
* the domain-free engine region contains no domain vocabulary;
* the external PAPER-TRACE judge is consumed from its own per-mutant records and
  recomputes 92/186 = 49.5 % (pre-hardening suite) and 172/186 = 92.5 %
  (hardened suite) exactly;
* the full ledger fits the 5 min budget.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "audit" / "criteria_mutation.py"
EXTERNAL = ROOT / "artifacts" / "criteria_mutation" / "paper_trace_audit.json"


def _load_tool():
    spec = importlib.util.spec_from_file_location("criteria_mutation", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules["criteria_mutation"] = module      # tautology_factory is referenced by name
    spec.loader.exec_module(module)
    return module


cm = _load_tool()


@pytest.fixture(scope="module")
def external():
    assert EXTERNAL.is_file(), f"external audit record missing: {EXTERNAL}"
    return cm.load_external(EXTERNAL)


@pytest.fixture(scope="module")
def summary(external):
    return cm.run_ledger(external=external)


# ---------------------------------------------------------------------------
# 1. instantiation: clean cases reproduce the shipped gate verdict
# ---------------------------------------------------------------------------
def test_clean_cases_reproduce_expected_verdict(summary):
    not_instantiated = [c["id"] for c in summary["criteria"] if not c["instantiated"]]
    assert not_instantiated == []
    assert summary["instantiation_rate"] == 1.0
    assert summary["n_criteria"] >= 15


def test_criteria_expose_the_ledger_contract(summary):
    for criterion in summary["criteria"]:
        assert set(criterion) >= {"id", "gate", "callable", "expected_verdict",
                                  "evidence_ref", "case" if False else "check"}
        assert ":" in criterion["callable"]           # module:function of a shipped gate
        assert criterion["expected_verdict"] == "PASS"
        assert criterion["evidence_ref"]


# ---------------------------------------------------------------------------
# 2. mutation sanity: no vacuous mutants
# ---------------------------------------------------------------------------
def test_every_mutant_changes_its_case(summary):
    """A mutant that does not change the case would fake a kill."""
    by_id = {c["id"]: c for c in summary["criteria"]}
    exercises = {}
    for criterion in cm.GATE_ADAPTERS.values():
        for crit in criterion():
            exercises[crit.id] = crit
    checked = 0
    for record in summary["mutants"]:
        crit = exercises.get(record["criterion_id"])
        if crit is None:                              # negative control
            continue
        mutants = {m.label: m for m in cm.generate_mutants(crit)}
        mutant = mutants[record["label"]]
        assert mutant.case != crit.case, record["label"]
        checked += 1
    assert checked >= 20


def test_m1_perturbations_have_the_declared_magnitude():
    crit = next(c for c in cm.criteria_p20_hydrostatic() if c.id == "P20H-H6")
    clean = crit.case["displacement_m3"]
    seen = {}
    for mutant in cm.generate_mutants(crit):
        if mutant.klass != "M1":
            continue
        ratio = mutant.case["displacement_m3"] / clean - 1.0
        assert abs(ratio - mutant.epsilon) < 1e-12
        seen[mutant.epsilon] = ratio
    assert sorted(seen) == [0.01, 0.03, 0.15]


# ---------------------------------------------------------------------------
# 3. KPI
# ---------------------------------------------------------------------------
def test_whole_ledger_kill_rate_clears_the_kpi(summary):
    assert summary["kill_rate"] >= 0.90
    assert summary["meets_target"] is True
    assert summary["n_gates"] >= 4
    assert summary["n_killed"] > 0


def test_per_gate_kill_rate_is_recorded(summary):
    gates = {row["gate"]: row for row in summary["gates"]}
    assert {"p20_hydrostatic", "p20_seakeeping", "vv_gci",
            "vv_validation", "paper_trace"} <= set(gates)
    for row in summary["gates"]:
        assert row["kill_rate"] is not None
    # the external judge contributes its whole 186-mutant population
    assert gates["paper_trace"]["n_in_scope"] == 186
    assert summary["gates_below_target"] == ["vv_validation"], summary["gates_below_target"]


def test_weakest_mutants_are_listed_and_not_hidden(summary):
    """The one surviving in-band mutant must be named, with its reason."""
    equiv = summary["equivalent_survivors"]
    assert [row["criterion_id"] for row in equiv] == ["VVVAL-ratio"]
    assert equiv[0]["epsilon"] == 0.01          # inside the declared U_val = 2 %
    assert summary["blind_spots"] == []          # no unexplained survivor


def test_gate_only_measurement_exposes_provenance_dependence(summary):
    """M5 survives the shipped gate alone: the provenance guard is what kills it."""
    assert summary["kill_rate_gate_only"] < summary["kill_rate"]
    m5_blind = [row for row in summary["gate_only_blind_spots"] if row["klass"] == "M5"]
    assert {row["criterion_id"] for row in m5_blind} >= {"P20H-H6", "VVVAL-ratio"}


# ---------------------------------------------------------------------------
# 4. M1 eps-ladder monotonicity
# ---------------------------------------------------------------------------
def test_epsilon_ladder_is_monotone(summary):
    ladder = summary["epsilon_ladder"]
    kills = [ladder[key]["killed"] for key in ("0.01", "0.03", "0.15")]
    assert kills == sorted(kills)
    assert summary["epsilon_ladder_violations"] == 0
    assert kills[-1] >= kills[0]


# ---------------------------------------------------------------------------
# 5. negative control
# ---------------------------------------------------------------------------
def test_tautological_criteria_score_zero_out_of_five(summary):
    control = summary["negative_control"]
    assert control["n_criteria"] == 5
    assert control["n_mutants"] == 5
    assert control["n_killed"] == 0
    assert control["kill_rate"] == 0.0
    assert sorted(control["flagged_unfalsifiable"]) == [f"NC-TAUT-{i:02d}" for i in range(1, 6)]
    # the control is reported as a row but excluded from the measured KPI
    control_rows = [row for row in summary["gates"] if row["gate"] == "negative_control"]
    assert len(control_rows) == 1 and control_rows[0]["control"] is True
    assert summary["n_in_scope"] == sum(row["n_in_scope"] for row in summary["gates"]
                                        if not row["control"])


# ---------------------------------------------------------------------------
# 6. merge gate: red / green
# ---------------------------------------------------------------------------
def test_pr_gate_rejects_new_criterion_without_killer(summary):
    clean = {k: v for k, v in summary.items() if k != "mutants"}
    green = cm.pr_gate(clean, None)
    assert green["verdict"] == "ACCEPT"
    assert green["rejected"] == []

    red = json.loads(json.dumps(clean))
    red["criteria"] = list(red["criteria"]) + [{
        "id": "NEW-NO-KILLER", "gate": "p20_hydrostatic", "callable": "x:y",
        "expected_verdict": "PASS", "evidence_ref": "test", "check": None,
        "mutant_classes": [], "tolerance": 0.0, "note": "",
        "n_mutants": 0, "n_killed": 0}]
    verdict = cm.pr_gate(red, clean)
    assert verdict["verdict"] == "REJECT"
    assert [row["id"] for row in verdict["rejected"]] == ["NEW-NO-KILLER"]

    killed = json.loads(json.dumps(red))
    killed["criteria"][-1]["n_mutants"] = 3
    killed["criteria"][-1]["n_killed"] = 1
    assert cm.pr_gate(killed, clean)["verdict"] == "ACCEPT"


def test_pr_demo_writes_red_and_green_evidence(summary, tmp_path):
    clean = {k: v for k, v in summary.items() if k != "mutants"}
    demo = cm.pr_demo(clean, None, tmp_path)
    assert demo["green"]["verdict"] == "ACCEPT"
    assert demo["red"]["verdict"] == "REJECT"
    assert demo["exit_codes"] == {"green": 0, "red": 2}
    for name in ("green_ledger.json", "green_verdict.json",
                 "red_ledger.json", "red_verdict.json"):
        assert (tmp_path / name).is_file()


# ---------------------------------------------------------------------------
# 7. zero domain semantics in the mutant engine
# ---------------------------------------------------------------------------
def test_engine_has_no_domain_vocabulary():
    source = TOOL.read_text(encoding="utf-8")
    assert cm.scan_engine_lexicon(source) == []
    # the scan must be able to fail: feed it a poisoned engine region
    poisoned = source.replace("UNIT_FACTOR: dict[str, tuple[float, str]] = {",
                              "UNIT_FACTOR: dict[str, tuple[float, str]] = {  # propeller", 1)
    assert "propeller" in cm.scan_engine_lexicon(poisoned)


# ---------------------------------------------------------------------------
# 8. external judge consumed from its own records
# ---------------------------------------------------------------------------
def test_paper_trace_baseline_recomputed_to_92_of_186(external):
    run = external["runs"]["baseline"]
    assert run["total"] == 186
    assert run["killed"] == 92
    assert run["kill_rate"] == pytest.approx(92 / 186, rel=0, abs=1e-12)
    assert run["recomputed_matches_declared"] is True
    assert run["baseline_green"] is True


def test_paper_trace_hardened_suite_clears_168_of_186(external):
    run = external["runs"]["hardened"]
    assert run["total"] == 186
    assert run["killed"] >= 168
    assert run["kill_rate"] >= 0.90
    assert run["killed"] == 172          # recorded round-2 result, recomputed here
    assert run["recomputed_matches_declared"] is True


def test_external_records_are_per_mutant_and_complete(external):
    for name, run in external["runs"].items():
        assert run["records_complete"] is True
        assert run["killed"] + run["survived"] == run["total"], name
    assert external["operative_run"] == "hardened"
    assert len(external["source_sha256"]) == 64


# ---------------------------------------------------------------------------
# 9. budget
# ---------------------------------------------------------------------------
def test_ledger_runs_inside_the_budget(summary):
    assert summary["elapsed_s"] < 300.0
    assert summary["budget_ok"] is True


# ---------------------------------------------------------------------------
# 10. artifact round-trip
# ---------------------------------------------------------------------------
def test_ledger_artifact_carries_the_numbers(summary, tmp_path):
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    for key in ("kill_rate", "n_killed", "n_in_scope", "instantiation_rate",
                "epsilon_ladder_violations", "negative_control", "gates",
                "weakest_mutants", "criteria", "external", "engine_lexicon"):
        assert key in loaded
    assert loaded["engine_lexicon"] == []
    md = cm.render_markdown(loaded)
    assert "kill rate" in md.lower() or "kills" in md.lower()