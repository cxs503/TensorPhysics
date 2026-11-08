"""End-to-end chain: the scenario produces chain_readiness.v1 and every claim holds."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tensorphysics.chain_ledger.readiness import CRITERIA, GATE_ID, SCHEMA
from tensorphysics.chain_ledger.runner import ChainConfig, run_chain_case


@pytest.fixture(scope="module")
def case_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("chain")
    detail = run_chain_case(ChainConfig(), outdir=out)
    return detail, out


def test_chain_verdict_is_pass_and_machine_readable(case_run):
    detail, out = case_run
    verdict = detail["readiness"]
    assert verdict["schema"] == SCHEMA
    assert verdict["gate_id"] == GATE_ID
    assert verdict["overall_status"] == "PASS", verdict["blocking_criteria"]
    assert verdict["chain_ready"] is True
    assert verdict["missing_products"] == []
    for face in ("A", "B", "C", "E", "T", "N"):
        assert verdict["faces"][face]["verdict"] == "PASS"
        assert verdict["faces"][face]["evidence_sha256"]

    on_disk = json.loads((out / "chain_readiness.json").read_text(encoding="utf-8"))
    assert on_disk["overall_status"] == "PASS"
    assert json.loads((out / "chain_ledger.json").read_text(encoding="utf-8"))["schema"] == (
        "chain_ledger_detail.v1"
    )


def test_every_criterion_is_measured_and_passes(case_run):
    detail, _ = case_run
    products = detail["products"]
    for crit in CRITERIA:
        value = products[crit.face][crit.key]
        assert crit.check(value), f"{crit.face}:{crit.key} = {value} fails {crit.op} {crit.threshold}"


def test_face_a_convergence_is_first_order_or_better(case_run):
    detail, _ = case_run
    rows = detail["faces"]["A"]["rows"]
    err = [r["hold_rel_err"] for r in rows]
    assert err == sorted(err, reverse=True)          # monotone under refinement
    assert rows[-1]["n_edge"] > rows[0]["n_edge"]
    assert detail["faces"]["A"]["observed_order"] >= 1.0


def test_face_b_negative_control_shows_the_defect(case_run):
    detail, _ = case_run
    face_b = detail["faces"]["B"]
    assert face_b["closed_flux_rel"] < face_b["closed_flux_negative_control_rel"]
    assert face_b["closed_flux_negative_control_rel"] > 1e-3


def test_face_c_peak_stress_is_physically_plausible(case_run):
    detail, _ = case_run
    face_c = detail["faces"]["C"]
    peak = face_c["peak_whipping_stress_MPa"]
    assert 0.05 < peak < 250.0                       # not a numerical artefact
    assert face_c["impulse_fraction_of_hull_momentum"] < 1.0
    assert face_c["impact_duration_s"] < 0.2         # short compared with the wave period
    # modal truncation: the midship stress must be insensitive to the last modes
    trunc = face_c["peak_stress_by_mode_count_Pa"]
    assert abs(trunc["8"] - trunc["6"]) / trunc["8"] < 0.15


def test_energy_ledger_closes_over_one_wave_period(case_run):
    detail, _ = case_run
    energy = detail["faces"]["E"]
    assert energy["wave"]["W_in_J"] > 0.0
    assert abs(energy["wave"]["dKE_plus_dPE_J"]) < 1e-6 * abs(energy["wave"]["W_in_J"])
    assert energy["wave_ledger_rel_err"] < 0.03
    assert energy["body_ledger_rel_err"] < 0.03
    # the marched 6-DOF reproduces the frequency-domain steady state
    assert energy["marched"]["amplitude_rel_diff"] < 0.02


def test_time_ledger_detects_both_negative_cases(case_run):
    detail, _ = case_run
    time_led = detail["faces"]["T"]
    assert time_led["zero_offset_rel_err"] < 0.03
    assert time_led["offset_residual"] > 0.10
    assert time_led["zero_order_force_err"] > 0.10
    assert time_led["offset_detected"] == 1.0
    assert time_led["zero_order_detected"] == 1.0
    # the aligned cases must stay below the detection threshold at the same rate
    fine = detail["config"]["fine_samples_per_period"]
    aligned = [r for r in time_led["wave_rows"]
               if r["samples_per_period"] == fine and r["offset_fraction"] == 0.0 and r["order"] == 2]
    assert aligned and aligned[0]["wave_ledger_rel_err"] < 0.03


def test_negative_controls_for_stabilisation(case_run):
    detail, _ = case_run
    negatives = detail["faces"]["N"]
    assert negatives["divergence_detection_rate"] == 1.0
    assert negatives["stabilised_converged_rate"] == 1.0
    assert negatives["spectral_radius"] > 1.0
    assert negatives["mass_ratio"] > 0.5             # floating equilibrium: m = rho*nabla
    assert negatives["added_mass_ratio"] < 1.0       # structural mass below the added mass


def test_readiness_withheld_when_an_interface_product_is_dropped(case_run):
    """Criterion (f): half a chain must never read as PASS."""
    from tensorphysics.chain_ledger.readiness import evaluate_chain

    detail, _ = case_run
    for face in ("A", "B", "C"):
        products = dict(detail["products"])
        products[face] = None
        verdict = evaluate_chain(products)
        assert verdict["overall_status"] == "WITHHELD"
        assert face in verdict["missing_products"]


def test_readiness_fails_when_a_face_does_not_close(case_run):
    """Criterion (g): a measured but non-closing face is an honest FAIL."""
    from tensorphysics.chain_ledger.readiness import evaluate_chain

    detail, _ = case_run
    products = json.loads(json.dumps(detail["products"]))
    products["E"]["wave_ledger_rel_err"] = 0.5
    verdict = evaluate_chain(products)
    assert verdict["overall_status"] == "FAIL"
    assert "E:wave_ledger_rel_err" in verdict["blocking_criteria"]


def test_water_module_quantities_are_self_consistent(case_run):
    detail, _ = case_run
    wave = detail["wave"]
    assert wave["dispersion_residual"] < 1e-12
    assert wave["numerical_energy_flux_W_m"] == pytest.approx(wave["energy_flux_W_m"], rel=1e-2)
    assert detail["hydrodynamics"]["natural_omega_rad_s"]["heave"] > 0.0
    np.testing.assert_allclose(
        np.asarray(detail["hydrodynamics"]["effective_mass_matrix"]),
        np.asarray(detail["hydrodynamics"]["effective_mass_matrix"]).T,
        rtol=0, atol=0,
    )