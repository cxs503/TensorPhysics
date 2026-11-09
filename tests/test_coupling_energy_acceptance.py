"""Acceptance-layer tests for the P40W3-N09 coupling-energy ledger.

These tests target the *gate*, not the physics (the physics is covered by
``test_coupling_energy_contract.py``).  Two things are checked here:

* the frozen criterion set behaves as a contract -- unique ids, one check per
  criterion, a digest that changes when a threshold changes;
* the three-valued semantics cannot be laundered: an absent input is
  ``WITHHELD``, a violated threshold is ``FAIL``, a vacuous run (no contact, no
  heat input) cannot be quoted as evidence, and a complete record evaluates to
  ``REAL``.

Most cases run on synthetic records built in this file so that a broken gate
fails here rather than after twenty minutes of solver time; one integration test
runs the real LBM<->FEM solver end to end.
"""
from __future__ import annotations

import dataclasses
import math

import pytest

from tensorphysics.coupling_energy import acceptance as A
from tensorphysics.coupling_energy import scenarios as S
from tensorphysics.coupling_energy.exchange import TemporalScheme


# ---------------------------------------------------------------------------
# synthetic records
# ---------------------------------------------------------------------------
def _run(name: str, ratio: int, scheme: str, mode: str, residual: float,
         momentum: float | None = None, overlap: float = 1.1e-4,
         metrics: dict | None = None) -> dict:
    series = {"overlap": [0.0, overlap]}
    if momentum is not None:
        series["momentum_residual"] = [0.0, momentum]
    return {
        "name": name, "ratio": ratio, "scheme": scheme, "exchange_mode": mode,
        "residual_max": residual, "residual_final": residual,
        "series": series, "metrics": dict(metrics or {}),
    }


#: energy drift of the shared linear mechanical sweep (measured, see docs)
DRIFT = {1: 0.157, 2: 0.073, 4: 0.035, 8: 0.017}


def _complete_record() -> dict:
    """A 48-run record with realistic measured values but no solver calls."""
    runs: dict[str, list[dict]] = {"thermal": [], "mechanical": [], "three_field": []}
    for ratio in (1, 2, 4, 8):
        for scheme in ("zoh", "linear", "quadratic"):
            runs["thermal"].append(_run("thermal", ratio, scheme, "shared", 2.2e-13,
                                        metrics={"profile_rel_l2_error": 0.019}))
            runs["mechanical"].append(_run("mechanical", ratio, scheme, "shared",
                                           DRIFT[ratio], momentum=1.4e-13))
            runs["three_field"].append(_run("three_field", ratio, scheme, "shared",
                                            4.1e-06, momentum=1.4e-13,
                                            metrics={"thermal_residual_max": 1.0e-15,
                                                     "q_external_total_J": 2.4e5,
                                                     "max_contact_energy_J": 0.09}))
        runs["thermal"].append(_run("thermal", ratio, "linear", "independent", 0.362,
                                    metrics={"profile_rel_l2_error": 0.02}))
        runs["mechanical"].append(_run("mechanical", ratio, "linear", "independent",
                                       DRIFT[ratio] * 3, momentum=1.42))
        runs["three_field"].append(_run("three_field", ratio, "linear", "independent",
                                        0.488, momentum=1.4,
                                        metrics={"thermal_residual_max": 1.0e-3,
                                                 "q_external_total_J": 2.4e5,
                                                 "max_contact_energy_J": 0.09}))
    return runs


EXTRA = {
    "order_scan": {"zoh": 0.998, "linear": 1.968, "quadratic": 2.815},
    "plate_frequency": {"rel_error": 0.0025, "frequency_hz": 1868.07,
                        "analytic_hz": 1863.39},
}


def _statuses(report: A.AcceptanceReport) -> dict[str, str]:
    return {cid: o.status for cid, o in report.outcomes.items()}


# ---------------------------------------------------------------------------
# the frozen criterion set
# ---------------------------------------------------------------------------
def test_frozen_set_has_unique_ids_and_one_check_each():
    ids = [c.cid for c in A.CRITERIA]
    assert len(ids) == len(set(ids)), "duplicate criterion id"
    for crit in A.CRITERIA:
        assert crit.cid in A._CHECKS, f"{crit.cid} has no check"
        assert crit.gate in ("mandatory", "evidence"), crit.gate
        assert crit.requirement.strip(), f"{crit.cid} has no frozen requirement"
        assert crit.baseline.strip(), f"{crit.cid} states no baseline"
    assert set(A._CHECKS) <= set(ids), "check without a criterion"


def test_frozen_thresholds_are_the_enforced_ones():
    """The digest must attest the numbers that are actually applied."""
    runs = _complete_record()
    for r in runs["thermal"]:
        if r["exchange_mode"] == "shared":
            r["residual_max"] = 5.0e-13
    loose = [dataclasses.replace(c, threshold=1e-9)
             if c.cid == "thermal_energy_conservation" else c for c in A.CRITERIA]
    strict = [dataclasses.replace(c, threshold=1e-13)
              if c.cid == "thermal_energy_conservation" else c for c in A.CRITERIA]
    assert A.evaluate(runs, EXTRA, criteria=loose).outcomes[
        "thermal_energy_conservation"].status == "PASS"
    assert A.evaluate(runs, EXTRA, criteria=strict).outcomes[
        "thermal_energy_conservation"].status == "FAIL"
    assert A.criteria_digest(loose) != A.criteria_digest(strict)


def test_digest_is_stable_and_sensitive_to_the_thresholds():
    assert A.criteria_digest() == A.CRITERIA_DIGEST
    assert A.CRITERIA_DIGEST == A.criteria_digest(list(A.CRITERIA))   # order-free
    edited = list(A.CRITERIA)
    i = next(k for k, c in enumerate(edited)
             if c.cid == "thermal_energy_conservation")
    edited[i] = dataclasses.replace(edited[i],
                                    requirement="eps_max < 1e-6 over every shared run")
    assert A.criteria_digest(edited) != A.CRITERIA_DIGEST


def test_unknown_criterion_is_refused_instead_of_skipped():
    ghost = A.Criterion(cid="not_implemented", coupling="thermal", quantity="x",
                        requirement="x < 1", gate="mandatory", rationale="none")
    with pytest.raises(A.AcceptanceError):
        A.evaluate({"thermal": [], "mechanical": [], "three_field": []}, None,
                   criteria=list(A.CRITERIA) + [ghost])


# ---------------------------------------------------------------------------
# three-valued semantics
# ---------------------------------------------------------------------------
def test_empty_record_is_withheld_and_never_passes():
    report = A.evaluate({"thermal": [], "mechanical": [], "three_field": []})
    assert report.verdict == "WITHHELD"
    assert not any(o.status == "PASS"
                   for o in report.outcomes.values() if o.gate == "mandatory")
    assert {o.cid for o in report.withheld} >= {"thermal_energy_conservation",
                                                "matrix_completeness"}


def test_missing_control_arm_withholds_falsification_criteria():
    runs = _complete_record()
    for name in runs:
        runs[name] = [r for r in runs[name] if r["exchange_mode"] == "shared"]
    report = A.evaluate(runs, EXTRA)
    statuses = _statuses(report)
    for cid in ("thermal_falsification_ratio", "mechanical_falsification_ratio",
                "three_field_falsification_ratio", "falsification_arm_coverage"):
        assert statuses[cid] == "WITHHELD", cid
    assert "PASS" not in [statuses["thermal_falsification_ratio"]]


def test_partial_matrix_is_withheld_not_failed():
    runs = _complete_record()
    keep = {1, 2}
    runs = {k: [r for r in v if r["ratio"] in keep] for k, v in runs.items()}
    report = A.evaluate(runs, EXTRA)
    assert report.outcomes["matrix_completeness"].status == "WITHHELD"
    assert report.outcomes["mechanical_energy_convergence"].status == "WITHHELD"
    assert report.verdict == "WITHHELD"


def test_threshold_violation_fails_and_names_the_criterion():
    runs = _complete_record()
    for r in runs["thermal"]:
        if r["exchange_mode"] == "shared":
            r["residual_max"] = 3.0e-9           # booked numbers disagree
    report = A.evaluate(runs, EXTRA)
    assert report.verdict == "FAIL"
    assert [o.cid for o in report.failures] == ["thermal_energy_conservation"]
    with pytest.raises(A.AcceptanceError) as exc:
        A.assert_accepted(report)
    assert "thermal_energy_conservation" in str(exc.value)


def test_accuracy_gate_uses_the_three_percent_standard():
    runs = _complete_record()
    for r in runs["thermal"]:
        if r["exchange_mode"] == "shared":
            r["metrics"]["profile_rel_l2_error"] = 0.0345      # measured --quick mode
    report = A.evaluate(runs, EXTRA)
    outcome = report.outcomes["thermal_profile_accuracy"]
    assert outcome.status == "FAIL"
    assert outcome.actual == pytest.approx(0.0345)
    assert report.verdict == "FAIL"


def test_vacuous_mechanical_run_is_caught():
    """No contact -> the momentum residual is a meaningless zero."""
    runs = _complete_record()
    for r in runs["mechanical"]:
        if r["exchange_mode"] == "shared":
            r["series"]["overlap"] = [0.0, 0.0]
            r["series"]["momentum_residual"] = [0.0, 0.0]
    report = A.evaluate(runs, EXTRA)
    statuses = _statuses(report)
    # vacuity is missing evidence, not a violated threshold ...
    assert statuses["mechanical_contact_engaged"] == "WITHHELD"
    assert statuses["mechanical_momentum_conservation"] == "PASS"   # misleading ...
    assert report.verdict == "WITHHELD"                             # ... but never REAL


def test_vacuous_three_field_chain_is_caught():
    runs = _complete_record()
    for r in runs["three_field"]:
        if r["exchange_mode"] == "shared":
            r["metrics"]["q_external_total_J"] = 0.0
    report = A.evaluate(runs, EXTRA)
    assert report.outcomes["three_field_physics_engaged"].status == "WITHHELD"
    assert report.verdict == "WITHHELD"


def test_energy_drift_must_be_monotone_and_reach_five_percent():
    runs = _complete_record()
    for r in runs["mechanical"]:
        if r["exchange_mode"] == "shared" and r["scheme"] == "linear":
            r["residual_max"] = {1: 0.05, 2: 0.04, 4: 0.06, 8: 0.03}[r["ratio"]]
    report = A.evaluate(runs, EXTRA)
    outcome = report.outcomes["mechanical_energy_convergence"]
    assert outcome.status == "FAIL"
    assert outcome.gate == "evidence"
    # an evidence-only failure is reported but cannot overturn the verdict
    assert report.verdict == "REAL"


# ---------------------------------------------------------------------------
# measurements / re-gating / drift audit
# ---------------------------------------------------------------------------
def test_measurements_extract_the_drift_sweep_by_ratio():
    m = A.measurements(_complete_record(), EXTRA)
    assert sorted(m["mechanical_energy_drift"]) == [1, 2, 4, 8]
    assert m["mechanical_energy_drift"][8] == pytest.approx(0.017)
    assert m["n_runs"] == A.EXPECTED_MATRIX_RUNS == 48
    assert m["thermal_profile_rel_l2_r1"] == pytest.approx(0.019)


def test_complete_record_evaluates_to_real():
    report = A.evaluate(_complete_record(), EXTRA)
    assert report.verdict == "REAL", A.format_table(report)
    assert report.digest == A.CRITERIA_DIGEST
    assert report.failures == [] and report.withheld == []
    assert A.assert_accepted(report) is report


def test_evaluate_record_regates_a_stored_artifact_shape():
    record = {
        "mission_id": 7607, "node": "P40W3-N09",
        "runs": _complete_record(),
        "order_scan": EXTRA["order_scan"],
        "plate_first_frequency": EXTRA["plate_frequency"],
    }
    report = A.evaluate_record(record)
    assert report.verdict == "REAL"
    assert report.digest == A.CRITERIA_DIGEST
    # a bare suite mapping is accepted too
    assert A.evaluate_record(record["runs"], ).verdict in ("REAL", "WITHHELD")


def test_audit_drift_detects_status_flips_and_threshold_edits():
    reference = A.evaluate(_complete_record(), EXTRA)
    runs = _complete_record()
    for r in runs["three_field"]:
        if r["exchange_mode"] == "shared":
            r["metrics"]["thermal_residual_max"] = 1.0e-9
    candidate = A.evaluate(runs, EXTRA)
    kinds = {(d["kind"], d.get("cid")) for d in A.audit_drift(reference, candidate)}
    assert ("status", "three_field_thermal_subchain") in kinds
    # the numeric drift is reported under the same criterion id
    assert ("value", "three_field_thermal_subchain") in kinds

    edited = list(A.CRITERIA)
    i = next(k for k, c in enumerate(edited) if c.cid == "plate_first_frequency")
    edited[i] = dataclasses.replace(edited[i], threshold=1.0e-3,
                                    requirement="< 0.1 % rel. error")
    drifted = A.evaluate(_complete_record(), EXTRA, criteria=edited)
    assert A.audit_drift(reference, drifted)[0]["kind"] == "criteria_digest"
    # the threshold lives in the criterion, so tightening it really does fail
    assert drifted.verdict == "FAIL"
    assert drifted.outcomes["plate_first_frequency"].status == "FAIL"


def test_format_table_covers_every_criterion():
    table = A.format_table(A.evaluate(_complete_record(), EXTRA))
    for crit in A.CRITERIA:
        assert crit.cid in table
    assert "verdict: REAL" in table
    assert A.CRITERIA_DIGEST[:16] in table


# ---------------------------------------------------------------------------
# integration with the real solvers (LBM <-> FEM, one ratio)
# ---------------------------------------------------------------------------
def test_real_thermal_run_passes_conservation_and_accuracy():
    res = S.run_thermal_lbm_fem(ratio=1, scheme=TemporalScheme.LINEAR, t_end=800.0,
                                nx=40, ny=20, relax=0.2,
                                sample_times=[400.0, 800.0]).as_dict()
    m = A.measurements({"thermal": [res]})
    assert m["thermal_eps_shared_max"] < 1e-12
    assert "profile_rel_l2_error" in res["metrics"]
    assert m["thermal_profile_rel_l2_r1"] < 0.03
    report = A.evaluate({"thermal": [res]})
    assert report.outcomes["thermal_energy_conservation"].status == "PASS"
    assert report.outcomes["thermal_profile_accuracy"].status == "PASS"
    # one run is not the pre-registered matrix: the record is withheld
    assert report.verdict == "WITHHELD"
    assert math.isclose(report.outcomes["matrix_completeness"].actual, 1.0)