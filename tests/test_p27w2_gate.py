"""Tests for the P27-W2 campaign integration gate (node N10).

Covers the six machine-checkable acceptance criteria:

* (a) DAG acyclicity + predecessor terminal-state convergence
* (b) cross-repo interface contracts (RAO / granular flow / GCI-UQ)
* (c) maturation-node coverage (N04 spectral fatigue, N09 AMR)
* (d) evidence ledger completeness
* (e) capability ledger evidence levels
* (f) pytest count gate

All fixtures are synthetic; no network or I/O is required.
"""
from __future__ import annotations

import math

import pytest

from tensorphysics.campaign.p27w2_gate import (
    CampaignDAG,
    CycleError,
    DAGNode,
    build_dag_from_plan,
    build_evidence_ledger,
    build_maturity_report,
    entry_from_mission,
    inject_cycle,
    is_terminal,
    run_gate,
    validate_all,
    validate_gci_uq_handoff,
    validate_granular_flow_handoff,
    validate_rao_handoff,
)
from tensorphysics.campaign.p27w2_gate import contracts as C
from tensorphysics.campaign.p27w2_gate import evidence as E
from tensorphysics.campaign.p27w2_gate import gate as G
from tensorphysics.campaign.p27w2_gate import maturity as M

# ---------------------------------------------------------------------------
# Fixtures: the real P27-W2 DAG shape (plan 153)
# ---------------------------------------------------------------------------


def _p27w2_nodes(statuses: dict[int, str] | None = None) -> list[dict]:
    """The ten P27-W2 nodes with their real dependency structure."""
    statuses = statuses or {}
    base = {
        961: [], 962: [961], 963: [961, 962], 964: [], 965: [],
        966: [], 967: [966], 968: [], 969: [],
        970: [961, 962, 963, 964, 965, 966, 967, 968, 969],
    }
    names = {
        961: "N01 BEM radiation/diffraction",
        962: "N02 seakeeping RAO",
        963: "N03 roll damping three-channel",
        964: "N04 spectral fatigue",
        965: "N05 buoyancy-coupled plate",
        966: "N06 DEM sphere-sphere rolling friction",
        967: "N07 DEM non-spherical",
        968: "N08 turbulence u+ decomposition",
        969: "N09 AMR backflow correction",
        970: "N10 integration acceptance",
    }
    return [
        {"id": nid, "name": names[nid],
         "status": statuses.get(nid, "in-flight"),
         "dependencies": base[nid], "mission_id": 6655 + nid - 961}
        for nid in sorted(base)
    ]


def _dag(statuses: dict[int, str] | None = None) -> CampaignDAG:
    return build_dag_from_plan(_p27w2_nodes(statuses))


def _all_done() -> dict[int, str]:
    return {nid: "done" for nid in range(961, 971)}


# ---------------------------------------------------------------------------
# (a) DAG convergence
# ---------------------------------------------------------------------------


def test_dag_acyclic_and_topological_order():
    dag = _dag()
    order = dag.topological_sort()
    assert len(order) == 10
    pos = {nid: i for i, nid in enumerate(order)}
    # every dependency precedes its dependent
    for node in dag.nodes.values():
        for dep in node.depends_on:
            assert pos[dep] < pos[node.node_id]
    # the integration node is last
    assert order[-1] == 970


def test_dag_cycle_detected():
    dag = _dag()
    cyclic = inject_cycle(dag, 961)
    with pytest.raises(CycleError):
        cyclic.topological_sort()
    result = _verify(cyclic, 970)
    assert not result.acyclic
    assert not result.converged


def _verify(dag, integration=970):
    from tensorphysics.campaign.p27w2_gate import verify_convergence
    return verify_convergence(dag, integration)


def test_dag_converged_when_all_predecessors_done():
    dag = _dag(_all_done())
    result = _verify(dag)
    assert result.acyclic
    assert result.all_deps_done
    assert result.converged
    assert result.done_count == 10
    assert result.blocked_nodes == []


def test_dag_not_converged_while_predecessor_in_flight():
    statuses = _all_done()
    statuses[962] = "in-flight"
    dag = _dag(statuses)
    result = _verify(dag)
    assert result.acyclic
    assert not result.all_deps_done
    assert result.blocked_nodes == [962]
    assert not result.converged


def test_dag_honest_fail_is_terminal():
    statuses = _all_done()
    statuses[964] = "honest-FAIL"
    dag = _dag(statuses)
    result = _verify(dag)
    assert result.converged, "an honest FAIL must count as terminal"
    assert is_terminal("honest-FAIL")
    assert is_terminal("HONEST_FAIL")
    assert not is_terminal("running")


def test_dag_unknown_dependency_rejected():
    nodes = _p27w2_nodes()
    nodes[0]["dependencies"] = [999]
    with pytest.raises(ValueError):
        build_dag_from_plan(nodes)


def test_dag_self_loop_rejected():
    nodes = _p27w2_nodes()
    nodes[0]["dependencies"] = [961]
    with pytest.raises(ValueError):
        build_dag_from_plan(nodes)


# ---------------------------------------------------------------------------
# (b) cross-repo interface contracts
# ---------------------------------------------------------------------------


def _rao_payload() -> dict:
    """Legacy flat layout as produced by the ship-platform seakeeping
    module (m6657 backend/physics/rao_verification.py)."""
    return {
        "omega_e": [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0],
        "heave_rao": [0.95, 1.08, 1.05, 0.92, 0.55, 0.25, 0.15],
        "pitch_rao": [0.85, 1.02, 1.08, 1.00, 0.62, 0.32, 0.18],
        "roll_rao": [0.30, 0.42, 0.38, 0.31, 0.20, 0.12, 0.08],
        "heave_phase_deg": [-8, -22, -42, -68, -118, -168, -177],
        "pitch_phase_deg": [8, -12, -38, -72, -138, -177, -180],
        "roll_phase_deg": [5, -10, -30, -60, -120, -170, -179],
        "ship": {
            "L": 3.0, "B": 0.3, "T": 0.1875, "Cb": 0.446,
            "displacement": 0.07515, "rho": 1000.0, "Fn": 0.2,
        },
        "wave_direction_deg": 180,
        "kochin_energy_consistency": {"max_rel_dev": 0.012},
    }


def _rao_payload_canonical() -> dict:
    """Canonical layout with explicit schema + rao blocks."""
    return {
        "schema": C.RAO_SCHEMA,
        "hull_id": "wigley-60m",
        "ship": {"L": 3.0, "B": 0.3, "T": 0.1875, "Cb": 0.446,
                 "Fn": 0.2},
        "wave_direction_deg": 180,
        "frequencies_rad_s": [0.5, 1.0, 1.5, 2.0],
        "rao": {
            "heave": {"amplitude": [0.9, 1.1, 1.0, 0.6],
                     "phase": [0.1, 0.3, 0.5, 0.9]},
            "pitch": {"amplitude": [0.8, 1.0, 0.9, 0.5],
                      "phase": [0.0, 0.2, 0.4, 0.7]},
            "roll": {"amplitude": [0.3, 0.5, 0.4, 0.2],
                     "phase": [0.2, 0.4, 0.6, 1.1]},
        },
        "encounter_convention": "deep_water_encounter",
        "units": {"frequency": "rad/s", "phase": "rad"},
        "kochin_energy_consistency": {"max_rel_dev": 0.012},
    }


def test_rao_contract_valid():
    res = validate_rao_handoff(_rao_payload())
    assert res.ok, [c.detail for c in res.checks if not c.ok]
    assert res.coverage == 1.0


def test_rao_contract_rejects_non_monotonic_frequencies():
    p = _rao_payload()
    p["frequencies_rad_s"] = [0.5, 1.0, 0.9, 2.0]
    res = validate_rao_handoff(p)
    assert not res.ok
    bad = [c for c in res.checks if c.check_id == "C1.freq_monotonic"]
    assert bad and not bad[0].ok


def test_rao_contract_rejects_missing_dof():
    p = _rao_payload()
    del p["roll_rao"]
    res = validate_rao_handoff(p)
    assert not res.ok


def test_rao_contract_accepts_canonical_layout():
    res = validate_rao_handoff(_rao_payload_canonical())
    assert res.ok, [c.detail for c in res.checks if not c.ok]


def test_rao_contract_rejects_cb_closure_violation():
    p = _rao_payload()
    p["ship"]["Cb"] = 0.9  # inconsistent with displacement
    res = validate_rao_handoff(p)
    assert not res.ok
    bad = [c for c in res.checks if c.check_id == "C1.cb_closure"]
    assert bad and not bad[0].ok


def _granular_payload() -> dict:
    r = 0.005
    rho = 917.0
    m = rho * 4.0 / 3.0 * math.pi * r ** 3
    return {
        "schema": C.GRANULAR_SCHEMA,
        "particle": {"radius_m": r, "mass_kg": m, "density_kg_m3": rho},
        "contact": {"k_n": 1.0e5, "k_t": 3.0e4, "mu_s": 0.4,
                     "mu_r": 0.05, "c_rot": 1.0e-3},
        "angle_of_repose": {
            "target_deg": 34.0, "achieved_deg": 34.6,
            "calibration_source": "heap_repose_case_v1",
        },
    }


def test_granular_contract_valid():
    res = validate_granular_flow_handoff(_granular_payload())
    assert res.ok, [c.detail for c in res.checks if not c.ok]


def test_granular_contract_rejects_density_closure_violation():
    p = _granular_payload()
    p["particle"]["density_kg_m3"] = 1500.0
    res = validate_granular_flow_handoff(p)
    assert not res.ok
    bad = [c for c in res.checks if c.check_id == "C2.particle_closure"]
    assert bad and not bad[0].ok


def test_granular_contract_rejects_kt_above_kn():
    p = _granular_payload()
    p["contact"]["k_t"] = 2.0e5
    res = validate_granular_flow_handoff(p)
    assert not res.ok


def _gci_uq_payload() -> dict:
    """Richardson-consistent synthetic grid study: v = 1 + 2*h^2."""
    h1, h2, h3 = 0.1, 0.05, 0.025
    v1, v2, v3 = 1 + 2 * h1 ** 2, 1 + 2 * h2 ** 2, 1 + 2 * h3 ** 2
    p = 2.0
    n = 8
    mean, std = 1.02, 0.01
    ci95 = C.CI95_COEF * std / math.sqrt(n)
    return {
        "schema": C.GCI_UQ_SCHEMA,
        "gci": {
            "grids": [{"h": h1, "value": v1}, {"h": h2, "value": v2},
                      {"h": h3, "value": v3}],
            "observed_order": p,
            "gci_fine": 0.004,
            "gci_upper": 0.008,
            "gci_lower": 0.002,
        },
        "uq": {
            "quantity": "u_plus",
            "mean": mean, "std": std, "n_blocks": n, "ci95": ci95,
            "dual_domain": {"domain_a": "channelA", "domain_b": "channelB",
                            "cross_check_error": 0.03},
            "sgs_negative_control": {"enabled": True,
                                      "control_error": 0.01},
        },
        "units": {"quantity": "u_plus [dimensionless]"},
    }


def _gci_uq_payload_lbm_native() -> dict:
    """Real TensorLBM cylinder grid-convergence evidence layout."""
    return {
        "schema": C.GCI_UQ_SCHEMA,
        "gci": {
            "diameters_cells": [18.0, 24.0, 30.0],
            "cd_control_volume": [1.4472501372001216, 1.4461494896785114,
                                 1.435706587575468],
            "observed_order": 0.1,
            "fit_rms_pct": 0.20956699850023336,
            "extrapolated_value": 1.227122710564919,
            "gci_upper": 0.05,
            "gci_lower": 0.0,
        },
        "uq": {
            "quantity": "u_plus",
            "mean": 15.19, "std": 0.78, "n_blocks": 30,
            "ci95": C.CI95_COEF * 0.78 / math.sqrt(30),
            "dual_domain": {"domain_a": "2pi_delta", "domain_b": "4pi_delta",
                            "cross_check_error": 0.028},
            "sgs_negative_control": {"enabled": True,
                                      "control_error": 3.5},
        },
        "units": {"quantity": "u_plus [dimensionless]"},
    }


def test_gci_uq_contract_valid():
    res = validate_gci_uq_handoff(_gci_uq_payload())
    assert res.ok, [c.detail for c in res.checks if not c.ok]


def test_gci_uq_contract_accepts_fit_backed_order():
    res = validate_gci_uq_handoff(_gci_uq_payload_lbm_native())
    assert res.ok, [c.detail for c in res.checks if not c.ok]


def test_gci_uq_contract_rejects_inconsistent_order():
    p = _gci_uq_payload()
    p["gci"]["observed_order"] = 0.5  # inconsistent with the grids
    res = validate_gci_uq_handoff(p)
    assert not res.ok
    bad = [c for c in res.checks if c.check_id == "C3.order_consistency"]
    assert bad and not bad[0].ok


def test_validate_all_reports_missing_payloads():
    results = validate_all({"rao": _rao_payload()})
    assert len(results) == 3
    assert results[0].ok
    assert not results[1].ok and not results[2].ok


# ---------------------------------------------------------------------------
# (c) maturity aggregation
# ---------------------------------------------------------------------------


def test_maturity_report_covers_all_maturation_nodes():
    artifacts = {
        964: {"verdict": "done",
               "summary": "Dirlik vs Rayleigh ladder closed at 1.2%",
               "key_numbers": {"max_rel_dev": 0.012},
               "artifacts": ["runs/n04/result.json"]},
        969: {"verdict": "done",
               "summary": "AMR flux balance residual 3e-6",
               "key_numbers": {"flux_residual": 3e-6},
               "artifacts": ["runs/n09/result.json"]},
    }
    report = build_maturity_report(artifacts)
    cov = M.maturity_coverage(report)
    assert cov["complete"]
    assert cov["missing_nodes"] == []
    assert cov["coverage"] == 1.0


def test_maturity_report_flags_missing_node():
    report = build_maturity_report({964: {"verdict": "done",
                                          "summary": "ok"}})
    cov = M.maturity_coverage(report)
    assert cov["missing_nodes"] == [969]
    assert not cov["complete"]


# ---------------------------------------------------------------------------
# (d) evidence ledger
# ---------------------------------------------------------------------------


def _full_entry(nid: int = 961) -> E.EvidenceEntry:
    return E.EvidenceEntry(
        node_id=nid, mission_id=6655 + nid - 961,
        name=f"node {nid}", status="done",
        result_json=f"runs/n{nid}/result.json",
        artifacts=[f"runs/n{nid}/artifacts/field.png"],
        commit="abc123" * 5,
    )


def test_evidence_ledger_complete():
    ledger = build_evidence_ledger([_full_entry(n) for n in range(961, 971)])
    assert ledger.complete
    assert ledger.completeness_rate == 1.0
    assert ledger.incomplete() == []


def test_evidence_ledger_incomplete_when_commit_missing():
    entries = [_full_entry(n) for n in range(961, 971)]
    entries[0].commit = None
    ledger = build_evidence_ledger(entries)
    assert not ledger.complete
    assert ledger.completeness_rate == pytest.approx(9 / 10)
    missing = ledger.incomplete()
    assert [e.node_id for e in missing] == [961]
    assert "commit" in missing[0].missing_fields()


def test_entry_from_mission_running_has_no_result():
    node = {"id": 961, "name": "N01", "status": "in-flight",
             "mission_id": 6656}
    mission = {"id": 6656, "status": "running", "result": {}}
    entry = entry_from_mission(node, mission)
    assert not entry.complete
    assert "result_json" in entry.missing_fields()


# ---------------------------------------------------------------------------
# (e) capability ledger
# ---------------------------------------------------------------------------


def test_capability_criterion_real_and_withheld_pass():
    caps = [
        {"capability": "bem_rao_seakeeping", "evidence_level": "REAL",
         "previous_level": "MOCK"},
        {"capability": "dem_granular_flow", "evidence_level": "WITHHELD",
         "previous_level": "STUB"},
    ]
    crit = G.criterion_capability(caps)
    assert crit.status == "PASS"


def test_capability_criterion_mock_fails():
    caps = [{"capability": "lbm_gci_uq", "evidence_level": "MOCK"}]
    crit = G.criterion_capability(caps)
    assert crit.status == "FAIL"


# ---------------------------------------------------------------------------
# (f) pytest gate + overall aggregation
# ---------------------------------------------------------------------------


def test_pytest_criterion_counts():
    assert G.criterion_pytest({"passed": 12, "failed": 0,
                               "errors": 0}).status == "PASS"
    assert G.criterion_pytest({"passed": 5, "failed": 0,
                               "errors": 0}).status == "WITHHELD"
    assert G.criterion_pytest({"passed": 12, "failed": 1,
                               "errors": 0}).status == "FAIL"


def _green_inputs(statuses=None):
    dag = _dag(statuses or _all_done())
    contracts = validate_all({
        "rao": _rao_payload(),
        "granular": _granular_payload(),
        "gci_uq": _gci_uq_payload(),
    })
    maturity = build_maturity_report({
        964: {"verdict": "done", "summary": "s04",
               "key_numbers": {"a": 1.0}, "artifacts": ["a"]},
        969: {"verdict": "done", "summary": "s09",
               "key_numbers": {"b": 2.0}, "artifacts": ["b"]},
    })
    ledger = build_evidence_ledger([_full_entry(n) for n in range(961, 971)])
    caps = [{"capability": "c1", "evidence_level": "REAL"}]
    tests = {"passed": 18, "failed": 0, "errors": 0}
    return dag, contracts, maturity, ledger, caps, tests


def test_gate_pass_when_all_criteria_green():
    dag, contracts, maturity, ledger, caps, tests = _green_inputs()
    verdict = G.run_gate(dag, 970, contracts, maturity, ledger, caps,
                            tests, plan_id=153)
    assert verdict.overall == "PASS"
    assert all(c.status == "PASS" for c in verdict.criteria)


def test_gate_withheld_when_dag_not_converged():
    statuses = _all_done()
    statuses[968] = "in-flight"
    dag, contracts, maturity, ledger, caps, tests = _green_inputs(statuses)
    verdict = G.run_gate(dag, 970, contracts, maturity, ledger, caps,
                          tests, plan_id=153)
    assert verdict.overall == "WITHHELD"
    by_id = {c.criterion_id: c for c in verdict.criteria}
    assert by_id["a"].status == "WITHHELD"
    assert by_id["a"].evidence["blocked_nodes"] == [968]


def test_gate_fail_on_cycle():
    dag = inject_cycle(_dag(_all_done()), 961)
    _, contracts, maturity, ledger, caps, tests = _green_inputs()
    verdict = G.run_gate(dag, 970, contracts, maturity, ledger, caps,
                          tests, plan_id=153)
    assert verdict.overall == "FAIL"
    by_id = {c.criterion_id: c for c in verdict.criteria}
    assert by_id["a"].status == "FAIL"
