"""Tests for the ice-induced vibration transfer chain (campaign node p228w1-N08).

The tests are deliberately layered: analytic identities first (transmissibility
limits, static pipe limit, Rayleigh narrowband damage, S-N inversion), then the
chain physics (wet below dry, attenuation band, fatigue monotonicity), then the
payload contract (every criterion carries a number and a pass flag).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.fatigue.sn_curve import DNCurve
from tensorphysics.hull_hydroelastic.presets import KCS, hull_cases
from tensorphysics.ice_vibration_transfer import (
    GOVERNING_PATH,
    ChainConfig,
    HullModalBasis,
    MountIsolation,
    PipeSpan,
    acceleration_transfer,
    fatigue_life,
    geometry_gate,
    milling_torque_excitation,
    narrowband_damage_rate,
    replace_scaled,
    run_chain,
    synthesise_stress_history,
    welch_psd,
    wet_dry_modal_basis,
)


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def basis() -> HullModalBasis:
    return wet_dry_modal_basis(KCS, n_modes=10, n_elements=100)


@pytest.fixture(scope="module")
def design_excitation():
    return milling_torque_excitation("design", n_rps=2.5, duration_s=60.0, dt_s=1.0e-3)


@pytest.fixture(scope="module")
def fast_chain():
    cfg = ChainConfig(
        n_modes=10,
        n_elements=100,
        duration_s=60.0,
        # 60 s of rainflow counting keeps the damage estimate stable to ~20 %
        # (a 20 s window is dominated by Monte-Carlo scatter and breaks C3a)
        fatigue_duration_s=60.0,
        fatigue_dt_s=2.0e-3,
        scale_sweep=(0.5, 1.0, 2.0),
    )
    return run_chain(cfg)


# ------------------------------------------------------- analytic identities
def test_transmissibility_limits():
    m = MountIsolation(f_n_hz=4.0, zeta=0.15)
    assert m.magnitude(np.array([0.0]))[0] == pytest.approx(1.0, rel=1e-12)
    assert m.isolation_frequency_hz == pytest.approx(math.sqrt(2.0) * 4.0)
    # |H| = 1 exactly at r = sqrt(2)
    assert m.magnitude(np.array([m.isolation_frequency_hz]))[0] == pytest.approx(1.0, rel=1e-9)
    # attenuation everywhere above sqrt(2) f_n
    f = np.linspace(m.isolation_frequency_hz * 1.001, 500.0, 4000)
    assert np.all(m.magnitude(f) < 1.0)
    # the peak of the base-excited transmissibility sits just below r = 1
    u_star = (math.sqrt(1.0 + 8.0 * m.zeta**2) - 1.0) / (4.0 * m.zeta**2)
    f_peak = m.f_n_hz * math.sqrt(u_star)
    assert f_peak < m.f_n_hz
    assert m.magnitude(np.array([f_peak]))[0] == pytest.approx(m.peak_amplification, rel=1e-9)
    assert m.magnitude(np.array([m.f_n_hz]))[0] < m.peak_amplification
    # static deflection of the mount under gravity
    assert m.static_deflection_m == pytest.approx(9.81 / (2 * math.pi * 4.0) ** 2)


def test_mount_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        MountIsolation(f_n_hz=0.0, zeta=0.1)
    with pytest.raises(ValueError):
        MountIsolation(f_n_hz=4.0, zeta=0.0)
    with pytest.raises(ValueError):
        MountIsolation(f_n_hz=4.0, zeta=1.2)


def test_pipe_static_limit_matches_closed_form():
    """The modal stress transfer must collapse to m L^2/(8 Z) at f -> 0."""
    span = PipeSpan(n_modes=25)
    t0 = float(np.real(span.stress_transfer(np.array([0.0]))[0]))
    assert abs(t0) == pytest.approx(span.static_stress_per_acceleration, rel=1e-3)
    # one mode is already within 5 % of the converged value, and the sum converges upwards
    t1 = abs(float(np.real(PipeSpan(n_modes=1).stress_transfer(np.array([0.0]))[0])))
    assert t1 == pytest.approx(span.static_stress_per_acceleration, rel=0.05)


def test_pipe_first_frequency_is_simply_supported():
    span = PipeSpan()
    expected = (np.pi / 2.0) / span.length_m**2 * math.sqrt(span.stiffness_ei_nm2 / span.mass_per_length_kg_m)
    assert span.first_frequency_hz == pytest.approx(expected, rel=1e-12)


def test_dnv_sn_curve_inversion_and_damage_scaling():
    sn = DNCurve("D", "seawater_cp")
    s = 30.0
    n = sn.n_to_failure(s)
    assert sn.stress_for_life(n) == pytest.approx(s, rel=1e-8)
    # high-cycle branch is the m = 5 segment: damage scales as stress^5
    d1 = 1.0 / sn.n_to_failure(10.0)
    d2 = 1.0 / sn.n_to_failure(20.0)
    assert d2 / d1 == pytest.approx(2.0**5, rel=1e-6)


def test_narrowband_damage_rate_matches_monte_carlo():
    """Validate the closed form against a Monte-Carlo of Rayleigh stress ranges."""
    sn = DNCurve("D", "seawater_cp")
    sigma_rms = 2.0e6
    fz = 5.0
    scf = 1.8
    analytic = narrowband_damage_rate(sigma_rms, fz, sn, scf_factor=scf)
    rng = np.random.default_rng(11)
    # stress range of a narrowband process = 2 x Rayleigh(scale=sigma) peak
    ranges = 2.0 * rng.rayleigh(scale=sigma_rms, size=400_000)
    vector = np.vectorize(lambda v: sn.n_to_failure(float(v)))
    n_fail = vector(ranges * scf / 1.0e6)
    mc = fz * float(np.mean(1.0 / n_fail))
    assert analytic == pytest.approx(mc, rel=0.05)
    # a Rayleigh range distribution gives more damage than one deterministic
    # sinusoid of the same RMS (spread of the ranges above the mean) - the
    # narrowband form must therefore be the conservative (larger) number
    deterministic = fz / sn.n_to_failure(2.0 * math.sqrt(2.0) * sigma_rms * scf / 1.0e6)
    assert analytic > deterministic


def test_stress_synthesis_reproduces_psd_variance():
    f = np.linspace(0.0, 200.0, 801)
    target = 1.0e10 / (1.0 + (f / 15.0) ** 2)
    sigma = synthesise_stress_history(f, target, duration_s=120.0, dt_s=2.0e-3, seed=3)
    expected = math.sqrt(float(np.trapezoid(target, f)))
    assert np.std(sigma) == pytest.approx(expected, rel=0.15)
    # determinism
    sigma2 = synthesise_stress_history(f, target, duration_s=120.0, dt_s=2.0e-3, seed=3)
    assert np.array_equal(sigma, sigma2)


def test_welch_psd_white_noise_level():
    rng = np.random.default_rng(0)
    dt = 1.0e-3
    sigma2 = 4.0
    x = rng.normal(scale=math.sqrt(sigma2), size=200_000)
    f, s = welch_psd(x, dt, nperseg=4096)
    one_sided = float(np.trapezoid(s, f))
    assert one_sided == pytest.approx(sigma2, rel=0.05)
    # flat spectrum
    band = (f > 20.0) & (f < 400.0)
    assert np.std(s[band]) / np.mean(s[band]) < 0.35


# ------------------------------------------------------------- physics gates
def test_geometry_gate_accepts_reference_hulls():
    for case in hull_cases():
        gate = geometry_gate(case)
        assert gate.passed, gate.to_json_dict()
        assert gate.displacement_rel_error is not None and gate.displacement_rel_error < 0.09


def test_geometry_gate_rejects_wrong_block_coefficient():
    bad = type(KCS)(
        name="KCS",
        length=KCS.length,
        breadth=KCS.breadth,
        draught=KCS.draught,
        depth=KCS.depth,
        block=0.55,
        design_speed_kn=KCS.design_speed_kn,
    )
    gate = geometry_gate(bad)
    assert not gate.passed
    assert gate.checks["cb_relative"] is False
    assert gate.checks["displacement_relative"] is False


def test_wet_modes_below_dry(basis):
    assert basis.n_modes == 10
    ratio = basis.frequency_ratio
    assert np.all(ratio < 1.0)
    assert np.all(basis.added_mass_ratio > 0.0)
    # the frequency drop is consistent with the generalised added mass of mode 1:
    # the reported ratio is referred to the *wet* generalised mass, so the dry
    # mass ratio is mu/(1-mu); buoyancy stiffening pushes the wet frequency back
    # up by a few percent (mode-1 buoyancy share is reported in the payload)
    mu_dry = basis.added_mass_ratio[0] / (1.0 - basis.added_mass_ratio[0])
    expected = 1.0 / math.sqrt(1.0 + mu_dry)
    assert ratio[0] == pytest.approx(expected, rel=0.08)
    assert ratio[0] > expected          # buoyancy restoring stiffness raises f_wet
    assert basis.buoyancy_share[0] > 0.0


def test_acceleration_transfer_scales_with_force_and_is_self_equilibrated(basis):
    f = np.array([0.0, 0.5, 2.0])
    h = acceleration_transfer(basis, f)
    assert h[0] == pytest.approx(0.0, abs=1e-30)          # a rigid-body translation gives no acceleration
    # swapping the couple stations flips the sign of the response
    h_swap = acceleration_transfer(basis, f, x_excitation=basis.x_reaction_m, x_reaction=basis.x_excitation_m)
    assert np.allclose(h, -h_swap)


def test_excitation_torque_matches_milling_level(design_excitation):
    exc = design_excitation
    assert exc.q_mill_nm > 0.0
    assert 0.5 < exc.q_mill_nm / exc.q_design_nm < 3.0     # milling level O(design torque)
    assert exc.q_envelope_nm > exc.q_mill_nm
    assert exc.force_rms_n > 0.0
    lo, hi = exc.energy_band(0.90)
    assert 0.0 <= lo < hi <= exc.freqs_hz[-1]


def test_scaling_is_exact_for_the_intensity_sweep(design_excitation):
    scaled = replace_scaled(design_excitation, 2.0)
    assert scaled.force_rms_n == pytest.approx(2.0 * design_excitation.force_rms_n, rel=1e-12)
    assert scaled.scale == 2.0


# ----------------------------------------------------------------- the chain
def test_chain_criteria_all_carry_numbers_and_pass(fast_chain):
    payload = fast_chain.to_json_dict()
    ids = [c["id"] for c in payload["criteria"]]
    assert ids == ["G1", "C1", "C2a", "C2b", "C2c", "C3a", "C3b", "C4", "C5", "C6"]
    for c in payload["criteria"]:
        assert c["passed"], f"{c['id']} failed: {c['detail']}"
        assert c["value"] is not None
        assert c["threshold"] is not None
    assert payload["all_criteria_pass"] is True


def test_chain_fatigue_monotonic_and_power_law(fast_chain):
    sweep = fast_chain.scale_sweep
    lives = [row["life_years"] for row in sweep]
    assert lives == sorted(lives, reverse=True)
    # with a common random realisation the damage must follow the S-N slope exactly
    d = {row["scale"]: row["damage_per_year"] for row in sweep}
    m = fast_chain.sn_curve.params.m2
    assert d[2.0] / d[1.0] == pytest.approx(2.0**m, rel=0.02)
    assert d[1.0] / d[0.5] == pytest.approx(2.0**m, rel=0.02)


def test_chain_disclosures_present(fast_chain):
    ids = [d["id"] for d in fast_chain.disclosures]
    assert set(ids) == {"D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"}
    # D1 must be the honest full-band ratio, which is not below one for a passive mount
    d1 = next(d for d in fast_chain.disclosures if d["id"] == "D1")
    assert d1["value"] > 1.0


def test_chain_fatigue_life_is_finite_and_positive(fast_chain):
    for severity in ("light", "design", "heavy"):
        for path in ("hull_girder", GOVERNING_PATH):
            block = fast_chain.severities[severity]["paths"][path]
            if block["pipe"].get("fatigue") is None:
                continue    # global path fatigue is only integrated for the design severity
            life = block["pipe"]["fatigue"]["life_years"]
            assert math.isfinite(life) and life > 0.0
            assert block["pipe"]["fatigue"]["damage_per_year"] > 0.0
            assert block["pipe"]["fatigue"]["rainflow_cycles"] > 0.0
            assert block["pipe"]["fatigue"]["max_stress_range_MPa"] < 100.0


def test_two_load_paths_are_reported_and_local_path_governs(fast_chain):
    design = fast_chain.severities["design"]["paths"]
    hull = design["hull_girder"]["hull_girder_acceleration"]["rms_m_s2"]
    local = design[GOVERNING_PATH]["support_acceleration"]["rms_all_m_s2"]
    # the local shaft-line reaction shakes the foundation far harder than the
    # global hull-girder response (the physical reason path B governs)
    assert local > 10.0 * hull
    rigid = design[GOVERNING_PATH]["pipe"]["fatigue_rigid_reference"]
    assert rigid["life_years"] < design[GOVERNING_PATH]["pipe"]["fatigue"]["life_years"]


def test_tuning_map_reports_the_isolation_limits(fast_chain):
    assert len(fast_chain.tuning_map) == 9 * 5
    row = fast_chain.tuning_map[0]
    assert {"f_n_hz", "zeta", "rms_ratio_base_over_hull", "static_deflection_m"} <= set(row)
    # extremely soft mount -> large static deflection (the physical reason the
    # full-band criterion cannot be met by a practical mount)
    soft = [r for r in fast_chain.tuning_map if r["f_n_hz"] == 0.5]
    assert min(r["static_deflection_m"] for r in soft) > 0.9


def test_bode_payload_shape(fast_chain):
    bode = fast_chain.bode
    assert len(bode["freqs_hz"]) == len(bode["magnitude"]) == len(bode["phase_deg"])
    assert min(bode["magnitude"]) < 1.0
    # the 240-point log grid does not hit the stationary frequency exactly
    assert max(bode["magnitude"]) == pytest.approx(bode["peak_amplification"], rel=5e-3)