"""End-to-end acceptance-criteria test of the ice-impact campaign (quick mode)."""
import json
from pathlib import Path

import numpy as np
import pytest

from tensorphysics.ice.campaign import run_campaign


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    out = tmp_path_factory.mktemp("ice_campaign")
    bundle = run_campaign(out, quick=True)
    return bundle, out


def test_criteria_all_pass(campaign):
    bundle, _ = campaign
    crit = bundle["criteria"]
    assert crit["a_dispersion"]["pass"], crit["a_dispersion"]
    assert crit["b_static_deflection"]["pass"], crit["b_static_deflection"]
    assert crit["c_impact_load"]["pass_analytic_anchor"], crit["c_impact_load"]
    assert crit["d_fracture"]["pass"], crit["d_fracture"]
    assert crit["e_energy"]["pass"], crit["e_energy"]


def test_dispersion_numbers(campaign):
    bundle, _ = campaign
    disp = bundle["dispersion"]
    assert disp["wavelength_error_pct"] <= 3.0
    assert disp["omega_error_pct"] <= 3.0
    assert disp["group_speed_error_pct"] <= 3.0
    assert disp["energy_drift_rel"] < 1e-9


def test_static_numbers(campaign):
    bundle, _ = campaign
    st = bundle["static"]
    assert st["w_centre_error_pct"] <= 5.0
    assert st["max_profile_error_pct"] <= 5.0
    assert st["w_centre_analytic_m"] > 0.0


def test_impact_history_and_energy(campaign):
    bundle, _ = campaign
    res = bundle["impact_elastic"]["result"]
    hh = res["history"]
    t = np.asarray(hh["time_s"])
    assert t.size > 10
    assert np.all(np.diff(t) > 0)
    assert res["peak_force_N"] > 0.0
    # the ship decelerates against the ice
    assert hh["speed_m_s"][-1] < hh["speed_m_s"][0]
    led = res["ledger"]
    assert led["residual_total_pct"] <= 5.0
    assert led["residual_plate_side_pct"] <= 5.0
    assert led["impulse_error_pct"] <= 5.0


def test_fracture_case(campaign):
    bundle, _ = campaign
    frac = bundle["impact_fracture"]["result"]["fracture"]
    assert frac is not None and frac["triggered"]
    assert frac["time_s"] > 0.0
    assert frac["radius_over_lc"] < 1.0            # radial cracks at the contact
    assert 0.3 <= frac["ring"]["ring_radius_over_lc"] <= 3.5


def test_artifacts_written(campaign):
    bundle, out = campaign
    out = Path(out)
    data = json.loads((out / "criteria.json").read_text())
    assert "criteria" in data and "cases" in data
    for name in ("fig1_dispersion.png", "fig2_static_hetenyi.png",
                 "fig3_impact_history.png", "fig4_fracture_fields.png",
                 "fig5_energy_ledger.png", "fig6_supersonic_waves.png"):
        p = out / name
        assert p.exists() and p.stat().st_size > 5000
    assert (out / "history_elastic.csv").exists()
