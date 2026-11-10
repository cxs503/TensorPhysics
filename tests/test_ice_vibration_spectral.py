"""Tests for the ice-induced vibration spectral module (node P233W2-N07).

Acceptance criteria covered:

* AC1 SDOF natural frequency <= 1 %, MDOF (>=3 DOF) <= 5 %
* AC2 FRF amplitude/phase <= 1 % (<= 5 % near resonance)
* AC3 random-excitation RMS vs analytic <= 5 %
* AC4 resonance criterion identical to the analytic criterion over >= 20 points
* AC5 DEM-style ice load -> response PSD peak vs literature <= 20 %
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tensorphysics.ice_vibration_spectral import (
    ChainSystem,
    LighthouseCase,
    SDOFSystem,
    amplification_ratio,
    chain_analytic_frequencies,
    classification_consistency,
    crushing_frequency,
    half_power_edges_hz,
    intermittent_crushing_series,
    mdof_response_psd,
    narrowband_ice_psd,
    refine_psd,
    modal_response_psd,
    resonance_risk,
    rms_from_psd,
    run_acceptance,
    run_lighthouse_case,
    sdof_response_psd,
    state_space_frf_sdof,
    welch_psd,
    white_noise_rms_analytic,
    zero_order_hold_response,
)


@pytest.fixture(scope="module")
def acceptance() -> dict:
    return run_acceptance()


# ---------------------------------------------------------------- AC1
def test_ac1_sdof_natural_frequency() -> None:
    system = SDOFSystem(mass_kg=3.0e5, stiffness_n_per_m=1.4e7, damping_ratio=0.02)
    analytic = np.sqrt(1.4e7 / 3.0e5) / (2.0 * np.pi)
    assert system.frequency_hz == pytest.approx(analytic, rel=1e-12)
    grid = np.linspace(0.6 * analytic, 1.4 * analytic, 40001)
    identified = grid[int(np.argmax(np.abs(system.frf_displacement(grid))))]
    assert abs(identified - analytic) / analytic <= 0.01


def test_ac1_mdof_chain_analytic_frequencies() -> None:
    for n in (3, 5, 8):
        chain = ChainSystem(n_dof=n, mass_kg=1.0e4, stiffness_n_per_m=1.0e6, zeta_first=0.01, zeta_last=0.01)
        analytic = chain_analytic_frequencies(n, 1.0e4, 1.0e6)
        assert np.max(np.abs(chain.eigenfrequencies_hz - analytic) / analytic) <= 0.05
    # explicit 3-DOF closed form: omega_r = 2 sqrt(k/m) sin((2r-1)pi/(2(2n+1)))
    n, k, m = 3, 1.0e6, 1.0e4
    expected = 2.0 * np.sqrt(k / m) * np.sin(
        (2.0 * np.arange(1, n + 1) - 1.0) * np.pi / (2.0 * (2.0 * n + 1.0))
    ) / (2.0 * np.pi)
    assert np.allclose(chain_analytic_frequencies(n, m, k), expected, rtol=1e-12)


# ---------------------------------------------------------------- AC2
def test_ac2_sdof_frf_against_state_space() -> None:
    system = SDOFSystem(mass_kg=2.0e5, stiffness_n_per_m=9.5e6, damping_ratio=0.03)
    f = np.linspace(0.05, 5.0, 500)
    h1 = system.frf_displacement(f)
    h2 = state_space_frf_sdof(system.mass_kg, system.stiffness_n_per_m, system.damping_ratio, f)
    assert np.max(np.abs(np.abs(h1) - np.abs(h2)) / np.abs(h1)) <= 0.01
    phase = np.abs((np.angle(h1) - np.angle(h2) + np.pi) % (2.0 * np.pi) - np.pi)
    assert np.degrees(np.max(phase)) <= 1.0


def test_ac2_sdof_peak_amplitude_and_half_power_band() -> None:
    system = SDOFSystem(mass_kg=3.0e5, stiffness_n_per_m=1.4e7, damping_ratio=0.02)
    grid = np.linspace(0.95 * system.frequency_hz, 1.05 * system.frequency_hz, 200001)
    peak = np.max(np.abs(system.frf_displacement(grid)))
    assert abs(peak - system.peak_amplitude_m_per_n) / system.peak_amplitude_m_per_n <= 0.01
    lo, hi = half_power_edges_hz(system.frequency_hz, system.damping_ratio)
    # half-power points: |H| = peak / sqrt(2) within 1 %
    for edge in (lo, hi):
        amp = abs(system.frf_displacement(edge))
        assert amp == pytest.approx(peak / np.sqrt(2.0), rel=0.01)
    # small-damping limit of the half-bandwidth -> zeta f_n
    assert (hi - lo) / 2.0 == pytest.approx(system.damping_ratio * system.frequency_hz, rel=0.02)


def test_modal_basis_interface_matches_chain_system() -> None:
    """FEM-style modal input (f_n, phi_n, zeta_n) reproduces the chain response."""
    chain = ChainSystem(n_dof=5, mass_kg=5.0e6, stiffness_n_per_m=2.0e9, zeta_first=0.02, zeta_last=0.02)
    f = np.linspace(0.2, 8.0, 300)
    s_ff = np.ones_like(f)
    from_system = mdof_response_psd(chain, f, s_ff, 4, 4)
    from_modes = modal_response_psd(
        f, chain.eigenfrequencies_hz, chain.mode_shapes, chain.zeta, s_ff, 4, 4
    )
    assert np.allclose(from_system, from_modes, rtol=1e-12)
    # acceleration response differs by omega^4, as declared
    acc = modal_response_psd(
        f, chain.eigenfrequencies_hz, chain.mode_shapes, chain.zeta, s_ff, 4, 4, response="acceleration"
    )
    assert np.allclose(acc, from_system * (2.0 * np.pi * f) ** 4, rtol=1e-10)


def test_ac2_mdof_modal_vs_direct() -> None:
    chain = ChainSystem(n_dof=5, mass_kg=5.0e6, stiffness_n_per_m=2.0e9, zeta_first=0.02, zeta_last=0.02)
    f = np.linspace(0.2, 8.0, 200)
    direct = chain.frf_direct(f, 4, 4)
    modal = chain.frf_modal(f, 4, 4)
    assert np.max(np.abs(direct - modal) / np.abs(direct)) <= 0.01


def test_ac2_time_domain_harmonic_amplitude() -> None:
    """Steady-state harmonic amplitude from the exact ZOH integrator vs |H|."""
    system = SDOFSystem(mass_kg=3.0e5, stiffness_n_per_m=1.4e7, damping_ratio=0.02)
    f0 = 0.9 * system.frequency_hz
    dt = 5.0e-4
    n = 200000  # 100 s: the initial transient e^{-zeta omega t} has decayed to 1e-6
    t = np.arange(n) * dt
    hist = zero_order_hold_response(system, np.sin(2.0 * np.pi * f0 * t), dt)
    late_t = t[n // 2 :]
    late_u = hist.displacement_m[n // 2 :]
    # least-squares projection on the drive frequency (no max/min transient bias)
    c = 2.0 * np.mean(late_u * np.exp(-2j * np.pi * f0 * late_t))
    amplitude = abs(c)
    assert amplitude == pytest.approx(abs(system.frf_displacement(f0)), rel=0.02)


# ---------------------------------------------------------------- AC3
def test_ac3_white_noise_rms_matches_analytic() -> None:
    system = SDOFSystem(mass_kg=3.0e5, stiffness_n_per_m=1.4e7, damping_ratio=0.02)
    s0 = 1.0e8
    analytic = white_noise_rms_analytic(system, s0)
    f = np.linspace(1.0e-4, 40.0 * system.frequency_hz, 200000)
    spectral = rms_from_psd(f, sdof_response_psd(system, f, np.full_like(f, s0)))
    assert abs(spectral - analytic) / analytic <= 0.01
    # independent time-domain check (small ensemble, coarse tolerance)
    dt = 0.004
    n = int(300.0 / dt)
    sigma_f = np.sqrt(s0 / (2.0 * dt))
    rms = []
    for seed in (3, 9, 17, 29):
        rng = np.random.default_rng(seed)
        rms.append(zero_order_hold_response(system, rng.normal(0.0, sigma_f, n), dt).rms_displacement_m)
    assert abs(float(np.mean(rms)) - analytic) / analytic <= 0.10


# ---------------------------------------------------------------- AC4
def test_ac4_criterion_matches_analytic_predicate() -> None:
    result = classification_consistency()
    assert result["n_points"] >= 20
    assert result["agreement"] == 1.0
    # exact half-power band == amplification >= 1/sqrt(2) predicate
    for zeta in (0.01, 0.02, 0.05):
        lo, hi = half_power_edges_hz(1.0, zeta)
        assert amplification_ratio(lo, 1.0, zeta) == pytest.approx(1.0 / np.sqrt(2.0), rel=1e-9)
        assert amplification_ratio(hi, 1.0, zeta) == pytest.approx(1.0 / np.sqrt(2.0), rel=1e-9)


def test_ac4_tiers() -> None:
    within = resonance_risk(1.10, 1.10, 0.02)
    assert within.tier == "lock_in" and within.is_resonance_risk
    near = resonance_risk(1.07, 1.10, 0.02)
    assert near.tier == "near_resonance" and not near.is_resonance_risk
    far = resonance_risk(2.0, 1.10, 0.02)
    assert far.tier == "clear"


# ---------------------------------------------------------------- excitation
def test_crushing_frequency_relation() -> None:
    assert crushing_frequency(0.16, 0.15) == pytest.approx(1.0666667, rel=1e-6)
    assert crushing_frequency(0.0, 0.15) == 0.0
    with pytest.raises(ValueError):
        crushing_frequency(0.1, 0.0)


def test_welch_normalisation_of_crushing_series() -> None:
    ice = intermittent_crushing_series(duration_s=60.0, dt_s=0.004, nperseg=8192)
    assert rms_from_psd(ice.freqs_hz, ice.psd_n2_per_hz) ** 2 == pytest.approx(
        np.var(ice.force_n), rel=0.02
    )
    # the spectral peak tracks the crushing frequency within 10 %
    assert abs(ice.psd_peak_frequency_hz - ice.f_ice_hz) / ice.f_ice_hz <= 0.10


def test_narrowband_ice_psd_normalisation() -> None:
    f = np.linspace(0.0, 5.0, 200001)
    s = narrowband_ice_psd(f, 1.1, variance_n2=4.0e8, half_power_bandwidth_hz=0.05)
    assert rms_from_psd(f, s) ** 2 == pytest.approx(4.0e8, rel=1e-3)
    # full half-power bandwidth of the Gaussian shape is exactly the parameter
    half = 0.5 * s.max()
    idx = np.where(s >= half)[0]
    assert (f[idx[-1]] - f[idx[0]]) == pytest.approx(0.05, rel=0.01)


def test_refine_psd_preserves_integral() -> None:
    ice = intermittent_crushing_series(duration_s=60.0, dt_s=0.004, nperseg=8192)
    f_max = 20.0
    df = ice.freqs_hz[1] - ice.freqs_hz[0]
    n_band = int(f_max / df) + 1
    coarse = np.trapezoid(ice.psd_n2_per_hz[:n_band], ice.freqs_hz[:n_band])
    f, s = refine_psd(ice.freqs_hz, ice.psd_n2_per_hz, factor=16, f_max=f_max)
    fine = rms_from_psd(f, s) ** 2
    assert fine == pytest.approx(coarse, rel=0.05)


# ---------------------------------------------------------------- AC5 / cases
def test_ac5_lighthouse_literature_frequency() -> None:
    case = LighthouseCase(duration_s=120.0)
    result = run_lighthouse_case(case)
    f_resp = result["response"]["response_peak_frequency_hz"]
    assert abs(f_resp - 1.10) / 1.10 <= 0.20
    assert 1.00 <= f_resp <= 1.20
    assert result["literature_anchor"]["dominant_frequency_hz"] == 1.10


def test_lighthouse_spectral_rms_matches_time_domain() -> None:
    case = LighthouseCase(duration_s=120.0)
    result = run_lighthouse_case(case)
    assert result["response"]["spectral_vs_time_rms_relative_difference"] <= 0.15
    assert result["response"]["rms_displacement_m"] > 0.0


def test_lock_in_increases_response() -> None:
    case = LighthouseCase(duration_s=120.0)
    result = run_lighthouse_case(case)
    lock = result["lock_in_demonstration"]
    assert lock["amplification_of_rms"] > 1.3
    assert lock["verdict_locked"]["tier"] == "lock_in"
    assert lock["verdict_unlocked"]["tier"] in {"near_resonance", "lock_in"}


# ---------------------------------------------------------------- payload
def test_acceptance_payload_passes_and_is_json_serialisable(acceptance: dict) -> None:
    public = {k: v for k, v in acceptance.items() if k != "_series"}
    assert acceptance["all_passed"] is True
    for name in ("AC1", "AC2", "AC3", "AC4", "AC5"):
        assert acceptance["criteria"][name]["passed"] is True, name
    text = json.dumps(public)
    assert len(text) > 1000
    assert json.loads(text)["node"] == "P233W2-N07"