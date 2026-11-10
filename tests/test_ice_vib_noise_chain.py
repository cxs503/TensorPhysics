"""Verification tests for the coupled ice-vibration / radiated-noise chain.

Every check is anchored either on a closed-form result (piston radiation,
rectangular-aperture directivity, steady-state amplifier, Parseval) or on the
node's acceptance criteria C1-C5.  Nothing here is a tolerance on a
self-consistency only.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.ice_vibration.modal import HullParameters
from tensorphysics.ice_vib_noise import (
    CoupledCase,
    IceExcitationCase,
    HullGirderModes,
    RadiatorSurface,
    band_margins,
    iso19906_informed_envelope,
    one_third_octave_bands,
    piston_directivity,
    piston_radiation_efficiency,
    run_coupled_chain,
    run_with_idempotency,
    solve_modal_exact,
    solve_modal_newmark,
    steady_state_amplitude,
    thorp_absorption_db_per_km,
    transmission_loss_db,
)
from tensorphysics.ice_vib_noise.acoustics import range_to_threshold_m
from tensorphysics.ice_vib_noise.bands import band_mean_square
from tensorphysics.ice_vib_noise.chain import _amplitude_spectrum, _psd_from_amplitude
from tensorphysics.ice_vib_noise.response import energy_balance
from tensorphysics.ice_vib_noise.source import crushing_energy_j

RHO = 1025.0
C = 1500.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _hull() -> HullParameters:
    return HullParameters(
        length_m=130.0,
        breadth_m=28.0,
        draught_m=8.5,
        block_coefficient=0.725,
        displacement_kg=2.3e7,
        name="p210w2-test-hull",
    )


def _short_case(**kwargs) -> CoupledCase:
    exc = IceExcitationCase(duration_s=20.0, dt_s=0.005)
    return CoupledCase(hull=_hull(), excitation=exc, **kwargs)


def _disc_patches(radius: float, n: int) -> tuple[RadiatorSurface, float]:
    d = 2.0 * radius / n
    xs = (np.arange(n) + 0.5) * d - radius
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    inside = X ** 2 + Y ** 2 <= radius ** 2
    surf = RadiatorSurface(
        x_m=X[inside] + radius,
        y_m=Y[inside],
        area_m2=np.full(int(inside.sum()), d * d),
        fluid_density_kg_m3=RHO,
        sound_speed_m_s=C,
    )
    return surf, d * d * float(inside.sum())


# ---------------------------------------------------------------------------
# 1/3-octave band algebra
# ---------------------------------------------------------------------------


def test_band_series_is_iec_base10():
    bands = one_third_octave_bands(1.0, 1000.0)
    assert bands.n_bands == 31  # 10 bands per decade, three decades
    assert bands.centre_hz[0] == pytest.approx(1.0)
    ratios = np.diff(np.log10(np.array(bands.centre_hz))) * 10.0
    assert np.allclose(ratios, 1.0, atol=1e-12)
    edge_ratio = np.array(bands.upper_hz) / np.array(bands.lower_hz)
    assert np.allclose(edge_ratio, 10.0 ** 0.1, rtol=1e-12)


def test_band_sum_reproduces_psd_integral():
    rng = np.random.default_rng(3)
    dt = 0.005
    n = 12001
    v = rng.standard_normal(n)
    freqs, amp = _amplitude_spectrum(v, dt)
    psd = _psd_from_amplitude(freqs, amp)
    bands = one_third_octave_bands(0.5, 80.0)
    ms = band_mean_square(freqs, psd, bands)
    cum = np.concatenate(([0.0], np.cumsum(0.5 * (psd[1:] + psd[:-1]) * np.diff(freqs))))
    lo = np.interp(bands.lower_hz[0], freqs, cum)
    hi = np.interp(bands.upper_hz[-1], freqs, cum)
    assert ms.sum() == pytest.approx(hi - lo, rel=1e-12)
    # the band sum must also recover the time-domain mean square of the covered
    # frequency range (bands do not cover DC..f_lo nor f_hi..Nyquist)
    covered_share = ms.sum() / np.mean(v ** 2)
    expected = (bands.upper_hz[-1] - bands.lower_hz[0]) / (freqs[-1])
    assert covered_share == pytest.approx(expected, rel=0.02)


def test_single_tone_lands_in_its_band():
    dt = 0.002
    t = np.arange(int(40.0 / dt)) * dt
    f0 = 5.0
    v = np.sin(2 * math.pi * f0 * t)
    freqs, amp = _amplitude_spectrum(v, dt)
    bands = one_third_octave_bands(1.0, 50.0)
    ms = band_mean_square(freqs, _psd_from_amplitude(freqs, amp), bands)
    k = int(np.argmax(ms))
    assert abs(bands.centre_hz[k] / f0 - 1.0) < 0.06
    assert ms[k] == pytest.approx(np.mean(v ** 2), rel=2e-3)


# ---------------------------------------------------------------------------
# Rayleigh radiator against closed forms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ka", [0.5, 1.0, 2.0])
def test_piston_radiation_efficiency_matches_closed_form(ka):
    radius = 1.0
    surf, _ = _disc_patches(radius, 60)
    v = np.ones(surf.area_m2.size)
    omega = ka / radius * C
    sigma = surf.radiation_efficiency(v, omega, n_theta=64, n_phi=48)
    assert sigma == pytest.approx(piston_radiation_efficiency(ka), rel=2e-3)


def test_piston_directivity_matches_closed_form():
    radius = 1.0
    surf, area = _disc_patches(radius, 80)
    v = np.ones(surf.area_m2.size)
    ka = 2.0
    omega = ka / radius * C
    r = 1000.0
    thetas = np.array([0.0, 0.2, 0.4, 0.6])
    p = np.array([float(np.abs(surf.far_field_pressure(v, omega, r, th))) for th in thetas])
    directivity = p / p[0]
    assert np.allclose(directivity, piston_directivity(ka, thetas), rtol=2e-3)
    # on-axis monopole limit: p = rho c k v A / (2 pi r)
    on_axis = RHO * C * (omega / C) * 1.0 * area / (2.0 * math.pi * r)
    assert p[0] == pytest.approx(on_axis, rel=1e-9)


def _sinc(u: float) -> float:
    return 1.0 if abs(u) < 1e-12 else math.sin(u) / u


def test_rectangular_aperture_directivity_matches_sinc():
    length, width = 20.0, 6.0
    surf = RadiatorSurface.from_grid(length, width, n_x=60, n_y=18, fluid_density_kg_m3=RHO, sound_speed_m_s=C)
    v = np.ones(surf.area_m2.size)
    freq = 40.0
    omega = 2 * math.pi * freq
    k = omega / C
    r = 1000.0
    for theta_deg, phi_deg in ((20.0, 0.0), (35.0, 40.0), (60.0, 15.0)):
        th = math.radians(theta_deg)
        ph = math.radians(phi_deg)
        p = complex(surf.far_field_pressure(v, omega, r, th, ph))
        ux = k * math.sin(th) * math.cos(ph) * length / 2.0
        uy = k * math.sin(th) * math.sin(ph) * width / 2.0
        s_analytic = length * width * _sinc(ux) * _sinc(uy)
        # invert Eq. (1): S = p 2 pi r exp(-i k r) / (i rho omega)
        s_computed = p * 2.0 * math.pi * r * np.exp(-1j * k * r) / (1j * RHO * omega)
        # the patch grid starts at the bow (x = 0), so the aperture phase is
        # referenced to x = L/2: S = exp(-i k_x L/2) * S_centred
        phase_ref = np.exp(-1j * (k * math.sin(th) * math.cos(ph) * length / 2.0))
        assert abs(s_computed / (s_analytic * phase_ref) - 1.0) < 5e-3


def test_multi_frequency_power_equals_scalar_path():
    surf = RadiatorSurface.from_grid(130.0, 28.0, 48, 4, fluid_density_kg_m3=RHO, sound_speed_m_s=C)
    pattern = np.cos(2 * math.pi * surf.x_m / 130.0)
    freqs = np.array([2.0, 6.0, 18.0])
    multi = surf.radiated_power(np.tile(pattern, (freqs.size, 1)), 2 * math.pi * freqs, n_theta=16, n_phi=12)
    single = np.array(
        [float(surf.radiated_power(pattern, 2 * math.pi * f, n_theta=16, n_phi=12)) for f in freqs]
    )
    assert np.allclose(multi, single, rtol=1e-12)


def test_radiation_efficiency_grows_towards_unity():
    surf = RadiatorSurface.from_grid(4.0, 4.0, 40, 40, fluid_density_kg_m3=RHO, sound_speed_m_s=C)
    v = np.ones(surf.area_m2.size)
    low = surf.radiation_efficiency(v, 2 * math.pi * 30.0, n_theta=24, n_phi=16)
    high = surf.radiation_efficiency(v, 2 * math.pi * 3000.0, n_theta=64, n_phi=48)
    assert 0.0 < low < high
    assert high == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------------------
# Modal basis and integrator
# ---------------------------------------------------------------------------


def test_modal_basis_is_orthogonal_and_consistent():
    modes = HullGirderModes(_hull(), 2.0e12, n_modes=3)
    audit = modes.orthogonality_audit()
    assert audit["max_offdiag_mass_ratio"] < 1e-6
    assert audit["mass_orthogonality_pass"] and audit["stiffness_orthogonality_pass"]
    assert audit["max_rel_error_omega_from_kn"] < 1e-5
    # Euler-Bernoulli scaling: f ~ beta^2
    freqs = np.array(modes.frequencies_hz)
    beta2 = np.array([4.7300407448627040, 7.8532046240958376, 10.995607838001671]) ** 2
    assert np.allclose(freqs / freqs[0], beta2 / beta2[0], rtol=1e-10)


def test_exact_integrator_matches_steady_state_amplifier():
    m, fn, zeta = 1.0e5, 1.5, 0.02
    omega = 2 * math.pi * fn
    force_amp = 1.0e4
    excitation = 0.8 * omega
    dt = 0.002
    t = np.arange(int(120.0 / dt)) * dt
    Q = (force_amp * np.sin(2 * math.pi * excitation / (2 * math.pi) * t))[None, :]
    q, _, _ = solve_modal_exact(Q, dt, [omega], [zeta], [m])
    tail = t >= t[-1] - 10.0
    amplitude = (q[0][tail].max() - q[0][tail].min()) / 2.0
    analytic = steady_state_amplitude(excitation, omega, zeta, force_amp, m)
    assert amplitude == pytest.approx(analytic, rel=2e-3)


def test_newmark_agrees_with_exact_integrator():
    m, fn, zeta = 5.0e6, 0.7, 0.015
    omega = 2 * math.pi * fn
    dt = 0.004
    t = np.arange(int(30.0 / dt)) * dt
    Q = (1.0e6 * np.sin(2 * math.pi * 0.9 * fn * t))[None, :]
    q_exact, _, _ = solve_modal_exact(Q, dt, [omega], [zeta], [m])
    q_nm, _, _ = solve_modal_newmark(Q, dt, [omega], [zeta], [m])
    tail = t >= t[-1] - 5.0
    a_exact = (q_exact[0][tail].max() - q_exact[0][tail].min()) / 2.0
    a_nm = (q_nm[0][tail].max() - q_nm[0][tail].min()) / 2.0
    assert a_nm == pytest.approx(a_exact, rel=1e-2)


def test_zero_force_gives_zero_response():
    dt = 0.005
    t = np.arange(2000) * dt
    Q = np.zeros((2, t.size))
    q, qdot, qddot = solve_modal_exact(Q, dt, [4.45, 12.3], [0.015, 0.02], [5.0e6, 5.0e6])
    assert np.max(np.abs(q)) == 0.0
    assert np.max(np.abs(qdot)) == 0.0
    assert np.max(np.abs(qddot)) == 0.0


def test_energy_identity_closes_for_free_vibration_and_forcing():
    m, omega, zeta = 2.0e6, 5.0, 0.02
    c = 2 * zeta * omega * m
    k = m * omega ** 2
    dt = 0.001
    t = np.arange(4000) * dt
    Q = (3.0e5 * np.sin(2 * math.pi * 1.1 * t) + 1.0e5)[None, :]
    q, qdot, _ = solve_modal_exact(Q, dt, [omega], [zeta], [m])
    ledger = energy_balance(t, q, qdot, Q, [m], [k], [c], [0.0])
    assert ledger["residual_over_scale"] < 1e-4
    # the radiated branch must appear in the ledger and be non-negative
    q2, qd2, _ = solve_modal_exact(Q, dt, [omega], [zeta], [m], c_rad=[0.1 * c])
    ledger2 = energy_balance(t, q2, qd2, Q, [m], [k], [c], [0.1 * c])
    assert ledger2["per_mode"][0]["radiated_acoustic_j"] > 0.0
    assert ledger2["residual_over_scale"] < 1e-4


# ---------------------------------------------------------------------------
# Source, propagation and envelope helpers
# ---------------------------------------------------------------------------


def test_crushing_energy_is_drift_work():
    t = np.linspace(0.0, 10.0, 1001)
    force = np.full_like(t, 2.0e6)
    assert crushing_energy_j(force, t, 1.0) == pytest.approx(2.0e7, rel=1e-12)
    assert crushing_energy_j(force, t, 0.0) == 0.0


def test_thorp_and_transmission_loss():
    assert thorp_absorption_db_per_km(1000.0) > thorp_absorption_db_per_km(100.0)
    tl1 = float(transmission_loss_db(100.0, 1.0))
    tl2 = float(transmission_loss_db(100.0, 1000.0))
    assert abs(tl1) < 1e-3
    assert tl2 == pytest.approx(60.0 + float(thorp_absorption_db_per_km(100.0)), rel=1e-6)


def test_range_to_threshold_matches_received_level():
    sl, f, thr = 180.0, 100.0, 120.0
    r = range_to_threshold_m(sl, f, thr)
    assert r is not None and r > 1.0
    rl = sl - float(transmission_loss_db(f, r))
    assert rl == pytest.approx(thr, abs=1e-3)
    assert range_to_threshold_m(100.0, f, 120.0) is None


def test_envelope_shape_and_margins():
    env = iso19906_informed_envelope(0.05)
    f = np.array([0.25, 1.0, 8.0, 32.0])
    lim = env.limit(f)
    assert lim[1] == pytest.approx(0.05) and lim[2] == pytest.approx(0.05)
    assert lim[0] == pytest.approx(0.025)      # constant displacement branch below f_lo
    assert lim[3] == pytest.approx(0.0125)     # constant velocity branch above f_hi
    margins = band_margins([1.0, 2.0], [0.01, 0.2], [0.05, 0.05])
    assert margins["ratio"] == pytest.approx([0.2, 4.0])
    assert margins["governing_band_hz"] == 2.0
    assert margins["pass"] is False


# ---------------------------------------------------------------------------
# Chain criteria C1-C5 (short case) and physical sanity
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def chain_result():
    case = _short_case()
    result, idem = run_with_idempotency(case)
    return result, idem


def test_criterion_c1_zero_load(chain_result):
    c1 = chain_result[0]["criteria"]["C1_zero_load_control"]
    assert c1["max_abs_q"] == 0.0
    assert c1["relative_residual"] < 1e-9
    assert c1["pass"] is True


def test_criterion_c2_orthogonality(chain_result):
    c2 = chain_result[0]["criteria"]["C2_modal_orthogonality"]
    assert c2["max_offdiag_mass_ratio"] < 1e-6
    assert c2["pass"] is True


def test_criterion_c3_energy(chain_result):
    c3 = chain_result[0]["criteria"]["C3_energy_conservation"]
    assert c3["residual_over_work"] < 0.01
    assert c3["pass"] is True
    assert c3["parseval_band_closure_rel_error"] < 1e-6
    assert c3["parseval_pass"] is True
    # the spectral radiated energy must be positive and below the input work
    assert 0.0 < c3["radiated_acoustic_j_spectral"] < c3["work_in_j"]


def test_criterion_c4_linearity(chain_result):
    c4 = chain_result[0]["criteria"]["C4_envelope_consistency"]
    assert c4["max_gain_deviation_db"] < 1e-9
    assert c4["pass"] is True
    assert len(c4["equivalent_spl_envelope_db"]) == len(chain_result[0]["bands"]["centre_hz"])


def test_criterion_c5_idempotency(chain_result):
    _, idem = chain_result
    assert idem["identical"] is True
    assert idem["pass"] is True
    assert idem["digest_run_1"] == idem["digest_run_2"]


def test_chain_physical_sanity(chain_result):
    res, _ = chain_result
    geom = res["geometry"]
    assert geom["cb_consistent_within_0p02"] is True
    assert geom["block_coefficient_abs_error"] < 0.02
    # radiation damping of a hull girder must be far below structural damping
    zeta_rad = np.array(res["radiation"]["modal_radiation_damping_ratio"])
    assert np.all(zeta_rad < 5e-3)
    assert np.all(zeta_rad > 0.0)
    # source levels must be physically plausible for icebreaking noise
    sl = np.array(res["bands"]["source_level_db_1m"])
    assert 120.0 < sl.max() < 220.0
    # acoustic conversion efficiency of crushing is tiny but non-zero
    eta = res["energy"]["acoustic_conversion_efficiency"]
    assert 1e-12 < eta < 1e-2
    # mass conservation of the energy chain
    assert res["energy"]["work_in_j"] < res["energy"]["crushing_energy_j"]
    # regression guard (see docs/KNOWLEDGE_MODAL_RADIATION_BAND_LIMIT.md):
    # the acoustic band must contain the modal-velocity energy, otherwise a
    # truncated modal basis is being used as a physical surface velocity.
    coverage = np.array(
        res["criteria"]["C3_energy_conservation"]["band_coverage_share_of_modal_velocity_ms"]
    )
    assert np.all(coverage > 0.9)
    # and the radiated energy must stay far below the input work
    assert res["energy"]["radiated_acoustic_j_spectral"] < 0.01 * res["energy"]["work_in_j"]
    # the source spectrum must be above the equivalent envelope where the
    # acceleration exceeds the envelope (same sign of the margin in both domains)
    over = np.array(res["criteria"]["C4_envelope_consistency"]["physical_finding_C4b"][
        "spl_over_equivalent_envelope_db"
    ])
    accel_ratio = np.array(res["bands"]["acceleration_margin"]["midship"]["ratio"])
    assert np.allclose(over, 20.0 * np.log10(np.maximum(accel_ratio, 1e-30)), atol=1e-9)


def test_chain_rejects_bad_input():
    with pytest.raises(ValueError):
        CoupledCase(hull=_hull(), acoustic_validity_factor=0.5)
    with pytest.raises(ValueError):
        CoupledCase(hull=_hull(), linearity_scale=1.0)
    with pytest.raises(ValueError):
        CoupledCase(hull=_hull(), assessment_theta_deg=(90.0,))