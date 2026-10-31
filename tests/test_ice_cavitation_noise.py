"""Verification tests for the ice-water cavitation / radiated-noise domain module.

Physical anchors used here are analytic, not tuned:

* Rayleigh-Plesset collapse time ``0.91468 R0 sqrt(rho/dp)``
* inertial growth asymptote ``sqrt(2 dp / (3 rho))``
* Stokes drag ``3 pi mu d U`` in the creeping-flow limit
* Bernoulli inception number of a 1-D duct, ``(A_in/A_t)^2 - 1``
* compact monopole far field ``p = rho a^2 / r * du_s/dt`` and its ``1/r`` law
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# the domain kernels need numpy/scipy; skip (not fail) in a minimal CI image
pytest.importorskip("numpy")
pytest.importorskip("scipy")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.domains.ice_cavitation_noise.acoustics import (  # noqa: E402
    amplitude_spectrum,
    blade_pass_frequency,
    compact_source_far_field,
    kirchhoff_sphere_extrapolation,
    peak_frequency,
)
from tensorphysics.domains.ice_cavitation_noise.cavitation import (  # noqa: E402
    CavitationParams,
    P_V,
    RHO_L,
    RHO_V,
    SIGMA_L,
    alpha_from_radius,
    bubble_radius_forced,
    radius_from_alpha,
    rayleigh_collapse_time,
    rayleigh_growth_velocity,
    schnerr_sauer_source,
    zwart_gerber_belamri_source,
)
from tensorphysics.domains.ice_cavitation_noise.flow1d import (  # noqa: E402
    MixtureFlow1D,
    Quasi1DNozzle,
    analytic_inception_sigma,
)
from tensorphysics.domains.ice_cavitation_noise.particles import (  # noqa: E402
    IcePhaseParams,
    drag_force_per_particle,
    number_density,
    particle_response_time,
    pressure_fluctuation_rms,
)
from tensorphysics.domains.ice_cavitation_noise.steady1d import (  # noqa: E402
    SteadyNozzleFlow,
)

CAV = CavitationParams()


# ---------------------------------------------------------------------------
# Cavitation model
# ---------------------------------------------------------------------------
def test_radius_alpha_roundtrip():
    for alpha in (5.0e-4, 1.0e-2, 0.3):
        radius = radius_from_alpha(alpha, CAV.n0)
        assert alpha_from_radius(radius, CAV.n0) == pytest.approx(alpha, rel=1e-10)


def test_schnerr_sauer_vanishes_at_vapour_pressure():
    mdot = schnerr_sauer_source(P_V, 1.0e-3, CAV)
    assert abs(float(mdot)) < 1.0e-12


def test_cavitation_source_sign_follows_superheat():
    growth = float(schnerr_sauer_source(P_V - 5000.0, 1.0e-3, CAV))
    collapse = float(schnerr_sauer_source(P_V + 5000.0, 1.0e-3, CAV))
    assert growth > 0.0 > collapse
    # the nuclei background cannot condense away
    assert abs(float(schnerr_sauer_source(P_V + 5.0e4, CAV.alpha_nuc, CAV))) < 1.0e-12


def test_zgb_destruction_is_faster_than_production():
    growth = float(zwart_gerber_belamri_source(P_V - 5000.0, 1.0e-3, CAV))
    collapse = float(zwart_gerber_belamri_source(P_V + 5000.0, 1.0e-3, CAV))
    assert growth > 0.0 > collapse
    assert abs(collapse) / growth > 100.0


def test_rayleigh_plesset_collapse_matches_analytic():
    r0 = 1.0e-4
    dp = 1.0e4
    t_stop = rayleigh_collapse_time(r0, dp)
    t, r = __import__(
        "tensorphysics.domains.ice_cavitation_noise.cavitation",
        fromlist=["bubble_radius_history"],
    ).bubble_radius_history(
        r0, P_V + dp, t_end=1.5 * t_stop, dt=1.0e-9, r_stop=0.02 * r0
    )
    assert r[-1] <= 0.02 * r0 * 1.05
    assert t[-1] == pytest.approx(t_stop, rel=0.02)


def test_rayleigh_growth_asymptote():
    r0 = 1.0e-6
    dp = 2.0e4
    t = np.arange(0.0, 400.0e-6, 2.0e-8)
    from tensorphysics.domains.ice_cavitation_noise.cavitation import bubble_radius_history

    tt, rr = bubble_radius_history(r0, P_V - dp, t_end=t[-1], dt=2.0e-8)
    late = tt > 0.8 * tt[-1]
    slope = np.polyfit(tt[late], rr[late], 1)[0]
    assert slope == pytest.approx(rayleigh_growth_velocity(dp), rel=0.03)


def test_forced_rayleigh_plesset_is_step_converged():
    f = 80.0
    drive = P_V + 2500.0 * np.sin(2 * np.pi * f * np.arange(0, 0.02, 2.0e-6))
    coarse = bubble_radius_forced(np.arange(0, 0.02, 2.0e-6), drive, 2.286e-5)
    fine_t = np.arange(0, 0.02, 5.0e-7)
    drive_fine = P_V + 2500.0 * np.sin(2 * np.pi * f * fine_t)
    fine = bubble_radius_forced(fine_t, drive_fine, 2.286e-5)
    assert coarse.max() == pytest.approx(fine.max(), rel=0.02)
    assert coarse.min() == pytest.approx(fine.min(), rel=0.02)


# ---------------------------------------------------------------------------
# Ice phase
# ---------------------------------------------------------------------------
def test_particle_drag_reproduces_stokes_law():
    d = 1.0e-5
    u = 1.0e-4                       # Re ~ 1e-3, deep Stokes regime
    force = drag_force_per_particle(u, d)
    stokes = 3.0 * np.pi * 1.002e-3 * d * u
    assert force == pytest.approx(stokes, rel=0.02)


def test_particle_number_density_and_response_time():
    d = 5.0e-4
    c = 0.1
    n_p = number_density(c, d)
    assert n_p * np.pi * d ** 3 / 6.0 == pytest.approx(c, rel=1e-12)
    tau = particle_response_time(d)
    assert tau == pytest.approx(d ** 2 * (916.7 + 0.5 * RHO_L) / (18.0 * 1.002e-3), rel=1e-12)


def test_pressure_fluctuation_is_linear_in_concentration():
    ice = IcePhaseParams()
    p1 = pressure_fluctuation_rms(0.3, 0.05, ice)
    p2 = pressure_fluctuation_rms(0.3, 0.10, ice)
    assert p2 == pytest.approx(2.0 * p1, rel=1e-12)
    assert pressure_fluctuation_rms(0.3, 0.0, ice) == 0.0


# ---------------------------------------------------------------------------
# Steady flow solver
# ---------------------------------------------------------------------------
def test_steady_nozzle_recovers_bernoulli_throat_pressure():
    nozzle = Quasi1DNozzle(n_cells=100)
    flow = SteadyNozzleFlow(nozzle=nozzle, u_in=2.0)
    result = flow.march(sigma_of_inlet=5.0, freeze_vapour=True)
    u_throat = float(np.max(result.u))
    assert u_throat == pytest.approx(2.0 * nozzle.area_ratio, rel=0.01)
    drop = 0.5 * RHO_L * (u_throat ** 2 - 2.0 ** 2)
    assert (result.p_in - result.p_min) == pytest.approx(drop, rel=0.01)


def test_steady_inception_matches_bernoulli_analytic():
    """The single-phase pressure-minimum inception criterion must reproduce the
    Bernoulli value ``(A_in/A_t)^2 - 1`` of the duct."""
    nozzle = Quasi1DNozzle(n_cells=80)
    flow = SteadyNozzleFlow(nozzle=nozzle, u_in=2.0)
    sigma = analytic_inception_sigma(nozzle.area_ratio)
    result = flow.march(sigma_of_inlet=sigma, freeze_vapour=True)
    q = 0.5 * RHO_L * flow.u_in ** 2
    assert (result.p_in - result.p_min) / q == pytest.approx(sigma, rel=0.01)
    assert result.p_min == pytest.approx(P_V, rel=0.01)


def test_ice_lowers_the_local_pressure_field():
    nozzle = Quasi1DNozzle(n_cells=80)
    clean = SteadyNozzleFlow(nozzle=nozzle, u_in=2.0, ice_concentration=0.0)
    laden = SteadyNozzleFlow(nozzle=nozzle, u_in=2.0, ice_concentration=0.10)
    p_clean = clean.march(sigma_of_inlet=4.0, freeze_vapour=True).p_min
    p_laden = laden.march(sigma_of_inlet=4.0, freeze_vapour=True).p_min
    assert p_laden < p_clean
    assert laden.march(sigma_of_inlet=4.0, freeze_vapour=True).pressure_fluctuation_rms_max > 0.0


def test_unsteady_solver_matches_steady_bernoulli_limit():
    nozzle = Quasi1DNozzle(n_cells=80)
    solver = MixtureFlow1D(nozzle=nozzle, u_in=2.0, p_out=1.0e5)
    solver.run_to_steady(max_steps=3000)
    result = SteadyNozzleFlow(nozzle=nozzle, u_in=2.0).march(
        sigma_of_inlet=solver.sigma(), freeze_vapour=True
    )
    assert np.max(solver.p) == pytest.approx(np.max(result.p), rel=0.05)
    assert np.min(solver.p) == pytest.approx(result.p_min, rel=0.02)
    assert float(np.max(np.abs(solver.divergence()))) < 1.0


def test_unsteady_solver_keeps_vapour_fraction_bounded():
    nozzle = Quasi1DNozzle(n_cells=60)
    solver = MixtureFlow1D(nozzle=nozzle, u_in=2.0, p_out=9000.0)
    for _ in range(400):
        solver.advance(solver.stable_dt(0.3))
    assert np.all(np.isfinite(solver.p))
    assert np.all(solver.alpha >= CAV.alpha_floor)
    assert np.all(solver.alpha <= 1.0)


# ---------------------------------------------------------------------------
# Acoustics
# ---------------------------------------------------------------------------
def test_blade_pass_frequency():
    assert blade_pass_frequency(1200.0, 4) == pytest.approx(80.0)


def test_tone_amplitude_is_calibrated():
    t = np.arange(0.0, 1.0, 1.0e-4)
    amp = 3.0
    signal = amp * np.sin(2 * np.pi * 50.0 * t)
    freq, spec = amplitude_spectrum(t, signal)
    peak = peak_frequency(t, signal, f_min=10.0, f_max=200.0)
    assert peak["f_peak"] == pytest.approx(50.0, rel=0.01)
    assert peak["amplitude"] == pytest.approx(amp, rel=0.02)


def test_compact_monopole_matches_analytic_pulsating_sphere():
    radius, freq, u_amp = 2.0e-3, 200.0, 0.05
    t = np.arange(0.0, 20.0 / freq, 1.0e-6)
    volume = 4.0 / 3.0 * np.pi * (
        radius + u_amp / (2 * np.pi * freq) * np.sin(2 * np.pi * freq * t)
    ) ** 3
    far = compact_source_far_field(t, volume, distance=1.0)
    analytic = RHO_L * radius ** 2 * u_amp * 2 * np.pi * freq / 1.0
    peak = peak_frequency(t, far["p"], f_min=20.0, f_max=2000.0)
    assert peak["f_peak"] == pytest.approx(freq, rel=0.02)
    assert peak["amplitude"] == pytest.approx(analytic, rel=0.05)


def test_acoustic_inverse_square_law_and_kirchhoff():
    freq = 150.0
    t = np.arange(0.0, 10.0 / freq, 2.0e-6)
    a = 3.0e-3
    volume = 4.0 / 3.0 * np.pi * a ** 3 * (1.0 + 0.1 * np.sin(2 * np.pi * freq * t))
    r1 = compact_source_far_field(t, volume, distance=1.0)
    r2 = compact_source_far_field(t, volume, distance=2.0)
    assert r1["p_rms"] / r2["p_rms"] == pytest.approx(2.0, rel=0.02)
    # Kirchhoff extrapolation from a compact spherical surface at r=1 cm to the
    # far field must reproduce the direct monopole solution at the same radius
    surf = 1.0e-2
    near = compact_source_far_field(t, volume, distance=surf)
    kirch = kirchhoff_sphere_extrapolation(t, near["p"], surf, 1.0)
    assert kirch["p_rms"] == pytest.approx(r1["p_rms"], rel=0.03)


# ---------------------------------------------------------------------------
# Case-level integration (artifacts are machine-checkable)
# ---------------------------------------------------------------------------
def test_case_criteria_and_artifacts(tmp_path):
    from tensorphysics.domains.ice_cavitation_noise.case_runner import (
        CaseConfig,
        run_case,
        write_artifacts,
    )

    config = CaseConfig(n_cells=60, ice_concentrations=(0.0, 0.10))
    result, spectra, series = run_case(config)
    paths = write_artifacts(result, spectra, series, tmp_path)
    summary = json.loads((tmp_path / "case_summary.json").read_text(encoding="utf-8"))

    assert summary["criteria"]["a_inception_vs_clear_water"]["passed"] is True
    assert summary["criteria"]["c_tone_frequency"]["passed"] is True
    assert summary["criteria"]["acoustic_solver_verification"]["amplitude_error"] < 0.05
    assert summary["geometry"]["contraction_monotone"] is True
    assert summary["geometry"]["diffuser_monotone"] is True
    assert len(summary["ice"]) == 2
    # the ice-laden passage must reach the vapour pressure earlier (higher sigma_i)
    assert summary["ice"][1]["sigma_inception_fluctuation"] > summary["ice"][0]["sigma_inception_fluctuation"]
    assert any("fig_noise_spectra.png" in f for f in paths["files"])


def test_noise_increases_with_ice_concentration():
    """Broadband radiated pressure rises with ice concentration at fixed drive."""
    from tensorphysics.domains.ice_cavitation_noise.case_runner import (
        CaseConfig,
        run_case,
    )

    config = CaseConfig(n_cells=60, ice_concentrations=(0.0, 0.15), n_cycles=4)
    result, _, _ = run_case(config)
    noise = {row["ice_concentration"]: row for row in result.noise}
    assert noise[0.15]["broadband_db"] > noise[0.0]["broadband_db"]
    assert noise[0.15]["radius_ratio"] >= noise[0.0]["radius_ratio"]