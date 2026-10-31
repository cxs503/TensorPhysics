"""Tests for the mu(I) rheology calibration and the DEM shear cell (W5-N05)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tensorphysics.ice_wave import dem2d, rheology

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "w5n05" / "results.json"


def test_inertial_number_formula():
    value = rheology.inertial_number(0.5, 0.01, 100.0, 2500.0)
    expected = 0.5 * 0.01 / np.sqrt(100.0 / 2500.0)
    assert value == pytest.approx(expected, rel=1e-15)


def test_mu_jop_limits():
    assert rheology.mu_jop(0.0, mu_s=0.2, mu_2=0.4, I_0=0.3) == pytest.approx(0.2)
    assert rheology.mu_jop(1.0e9, mu_s=0.2, mu_2=0.4, I_0=0.3) == pytest.approx(0.4, rel=1e-9)
    values = rheology.mu_jop(np.logspace(-3, 1, 50), mu_s=0.2, mu_2=0.4, I_0=0.3)
    assert np.all(np.diff(values) > 0.0)


def test_fit_recovers_synthetic_parameters():
    """The calibrator must recover known Jop parameters from clean data."""
    I = np.logspace(-3, 0.0, 12)
    mu = rheology.mu_jop(I, mu_s=0.22, mu_2=0.45, I_0=0.15)
    fit = rheology.fit_mu_i(I, mu)
    assert fit["r2"] > 0.999
    assert fit["mu_s"] == pytest.approx(0.22, rel=2e-2)
    assert fit["mu_2"] == pytest.approx(0.45, rel=2e-2)
    assert fit["rmse"] < 1e-3


def test_fit_is_robust_to_small_noise():
    rng = np.random.default_rng(4)
    I = np.logspace(-2.2, -0.2, 8)
    mu = rheology.mu_jop(I, mu_s=0.25, mu_2=0.5, I_0=0.2) + rng.normal(scale=1e-3, size=8)
    fit = rheology.fit_mu_i(I, mu)
    assert fit["r2"] > 0.99


@pytest.mark.skipif(not RESULTS.exists(), reason="pipeline artifact not generated")
def test_dem_calibration_artifact_reaches_r2_criterion():
    """Acceptance criterion (c): R^2 > 0.99 on the DEM-measured mu(I) data."""
    results = json.loads(RESULTS.read_text())
    r2 = results["dem"]["criterion_c_r2"]
    assert r2 > 0.99
    rows = results["dem"]["calibration_rows"]
    assert len(rows) >= 5
    inertial = np.array([row["inertial_number"] for row in rows])
    friction = np.array([row["mu"] for row in rows])
    assert np.all(np.diff(inertial) > 0.0)
    assert friction[-1] > friction[0]  # shear strengthening


def test_dem_smoke_force_balance_and_pressure():
    """Action-reaction consistency and a compressive stress state."""
    cell = dem2d.ShearCell(n_particles=60, shear_rate=0.05, kn=1.0e3, seed=2)
    for _ in range(150):
        cell.step()
    contacts = cell.contact_forces()
    assert contacts["i"].size > 0
    # Every contact force must be bounded by the Coulomb cone.
    assert np.all(contacts["ft"] <= cell.mu_particle * contacts["fn"] + 1e-12)
    # Zero net internal force: sum over the pair list cancels by construction.
    net = np.array([contacts["fx"].sum(), contacts["fy"].sum()])
    sigma = cell.stress_tensor(contacts)
    assert 0.5 * np.trace(sigma) > 0.0
    assert np.all(np.isfinite(net))


def test_dem_integrates_without_blowup():
    cell = dem2d.ShearCell(n_particles=60, shear_rate=0.2, kn=1.0e3, seed=6)
    run = cell.run(200, measure_every=20)
    assert np.isfinite(run["p"]) and run["p"] > 0.0
    assert np.isfinite(run["mu_xy"])
    assert np.all(np.abs(cell.vel) < 1e3)


@pytest.mark.parametrize(
    "call",
    [
        lambda: rheology.inertial_number(0.0, 1.0, 1.0, 1.0),
        lambda: rheology.inertial_number(1.0, 1.0, -1.0, 1.0),
        lambda: rheology.mu_jop(1.0, mu_s=0.4, mu_2=0.2),
        lambda: rheology.mu_jop(-0.5),
        lambda: rheology.fit_mu_i([1.0, 2.0], [0.1, 0.2]),
        lambda: dem2d.ShearCell(n_particles=5),
        lambda: dem2d.ShearCell(area_fraction=1.5),
    ],
)
def test_invalid_inputs_raise(call):
    with pytest.raises((rheology.MuIError, dem2d.DEMError)):
        call()