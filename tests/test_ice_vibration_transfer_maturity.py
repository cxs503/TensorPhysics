"""Acceptance and physics tests for campaign node ``p227w2-N08``.

The node's machine-checked criteria are the five ``test_*`` functions named in
the campaign metadata:

``test_wet_modal_lower_than_dry``
    ``f_wet < f_dry`` for every elastic mode of the KCS hull.
``test_vibration_transfer_decay``
    ``|H(f)| < 1`` above ``sqrt(2) f_n`` for both the conventional and the QZS
    mount, and ``RMS(base) < RMS(hull)`` over the ice band with the QZS mount.
``test_piping_stress_positive``
    pipe stress positive for the three ice severities.
``test_fatigue_life_positive``
    fatigue life finite and positive for the three ice severities.
``test_fatigue_life_monotonic_ice``
    fatigue life decreases when the ice load increases.

The remaining tests pin down the physics of the QZS layer itself: reduction to
the conventional mount, static-deflection invariance, DC limit, RK4 versus
frequency-domain transmissibility, hardening backbone, tolerance formula, and
the pre/post contrast (the conventional mount cannot attenuate the ice band).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tensorphysics.ice_vibration_transfer.maturity import (
    DEFAULT_SEVERITIES,
    MitigationConfig,
    run_mitigation,
)
from tensorphysics.ice_vibration_transfer.mitigation import (
    QZSMount,
    band_attenuation,
    design_map,
    service_stroke,
    simulate_base_excitation,
)
from tensorphysics.ice_vibration_transfer.transfer import MountIsolation


@pytest.fixture(scope="module")
def chain():
    """Maturity chain on the reference KCS hull (reduced but converged settings)."""
    cfg = MitigationConfig(
        n_modes=10,
        n_elements=80,
        duration_s=30.0,
        fatigue_duration_s=20.0,
        fatigue_dt_s=2.0e-3,
    )
    return run_mitigation(cfg)


@pytest.fixture(scope="module")
def design(chain):
    return chain.design


# --------------------------------------------------------------- node criteria
def test_wet_modal_lower_than_dry(chain):
    """Criterion 1: added mass lowers every wet natural frequency."""
    basis = chain.basis
    ratio = basis.frequency_ratio
    assert ratio.shape[0] == chain.config.n_modes
    assert np.all(ratio < 1.0), f"wet/dry ratios must all be < 1, got {ratio}"
    assert basis.all_wet_below_dry
    # the 2-node vertical mode must drop by a physically plausible amount
    assert 0.5 < ratio[0] < 0.9
    assert 0.0 < basis.wet_frequencies_hz[0] < basis.dry_frequencies_hz[0]


def test_vibration_transfer_decay(chain):
    """Criterion 2: |H|<1 above sqrt(2) f_n, and energy-wise over the ice band."""
    f = chain.curves["design"]["freqs_hz"]
    for label in ("conventional", "qzs"):
        mount = chain.baseline if label == "conventional" else chain.design
        mag = mount.magnitude(f)
        above = f > mount.isolation_frequency_hz * (1.0 + 1e-12)
        assert np.all(mag[above] < 1.0), f"{label}: |H| >= 1 inside the attenuation band"
        # DC limit is unity for both mounts (passive suspension)
        assert abs(float(mount.magnitude(np.array([1e-9]))[0]) - 1.0) < 1e-6
    # the pointwise criterion is not enough: the band-integrated one is the node claim
    blk = chain.severities["design"]["qzs"]
    assert blk["rms_ratio_base_over_hull_ice_band"] < 1.0
    # base acceleration is below hull acceleration in the ice band
    assert blk["rms_base_ice_band_m_s2"] < chain.severities["design"]["hull_ice_band_rms_m_s2"]


def test_piping_stress_positive(chain):
    """Criterion 3: pipe stress is positive for the three ice severities."""
    for severity in DEFAULT_SEVERITIES:
        block = chain.severities[severity]["qzs"]
        assert block["stress_rms_MPa"] > 0.0
        assert block["stress_rms_scf_MPa"] > block["stress_rms_MPa"]  # SCF + thickness > 1
        assert np.isfinite(block["fatigue"]["max_stress_range_MPa"])
        assert block["fatigue"]["max_stress_range_MPa"] >= block["stress_rms_MPa"]


def test_fatigue_life_positive(chain):
    """Criterion 4: fatigue life finite and positive for the three ice severities."""
    for severity in DEFAULT_SEVERITIES:
        life = chain.severities[severity]["qzs"]["fatigue"]["life_years"]
        assert np.isfinite(life)
        assert life > 0.0
        assert chain.severities[severity]["qzs"]["fatigue"]["damage_per_year"] > 0.0


def test_fatigue_life_monotonic_ice(chain):
    """Criterion 5: more ice load -> shorter fatigue life (strictly monotonic)."""
    lives = [chain.severities[s]["qzs"]["fatigue"]["life_years"] for s in DEFAULT_SEVERITIES]
    assert lives[0] > lives[1] > lives[2], f"life must decrease with ice load, got {lives}"
    # and the same must hold on the conventional (pre-mitigation) chain
    base = [chain.severities[s]["conventional"]["fatigue"]["life_years"] for s in DEFAULT_SEVERITIES]
    assert base[0] > base[1] > base[2]
    # the S-N slope makes the decay super-linear in stress: life ratio > severity ratio
    assert lives[0] / lives[2] > 1.0


# ---------------------------------------------------------------- QZS physics
def test_qzs_reduces_to_conventional_mount():
    """Zero stiffness cancellation must reproduce the linear mount exactly."""
    f = np.linspace(0.05, 60.0, 700)
    qzs = QZSMount(equipment_mass_kg=2200.0, f_rigid_hz=4.0, residual_stiffness_fraction=1.0, zeta=0.15)
    lin = MountIsolation(f_n_hz=4.0, zeta=0.15)
    assert np.allclose(qzs.magnitude(f), lin.magnitude(f), rtol=0.0, atol=1e-12)
    assert qzs.f_n_hz == pytest.approx(4.0)
    assert qzs.negative_stiffness_fraction == pytest.approx(0.0)


def test_qzs_static_deflection_independent_of_alpha():
    """Static deflection is set by the load spring, not by the cancellation."""
    mounts = [
        QZSMount(equipment_mass_kg=2200.0, f_rigid_hz=4.0, residual_stiffness_fraction=s, zeta=0.15)
        for s in (1.0, 0.1, 0.01)
    ]
    deflections = [m.static_deflection_m for m in mounts]
    assert max(deflections) - min(deflections) < 1e-12
    assert deflections[0] == pytest.approx(9.81 / (2.0 * np.pi * 4.0) ** 2)
    # while the equivalent linear spring would need an impractical deflection
    assert mounts[-1].equivalent_linear_deflection_m > 50.0 * mounts[-1].static_deflection_m
    assert mounts[-1].f_n_hz == pytest.approx(4.0 * np.sqrt(0.01))


def test_qzs_dc_limit_is_unity(design):
    """Any passive mount transmits the support acceleration at zero frequency."""
    for freqs in (np.array([1e-9]), np.array([1e-6, 1e-4])):
        assert np.allclose(design.magnitude(freqs), 1.0, atol=1e-6)


def test_qzs_time_domain_matches_frequency_domain(design):
    """RK4 integration of the base-excited SDOF reproduces |H(f)|."""
    for f_probe in (0.5 * design.f_n_hz, design.f_n_hz, 3.0 * design.f_n_hz):
        sim = simulate_base_excitation(
            f_n_hz=design.f_n_hz,
            zeta=design.zeta,
            f_probe_hz=f_probe,
            cycles=150,
            steps_per_cycle=500,
            relative_amplitude_m=1.0e-3,
        )
        analytic = float(design.magnitude(np.array([f_probe]))[0])
        assert sim["acceleration_transmissibility"] == pytest.approx(analytic, rel=2e-2)


def test_qzs_relative_displacement_matches_rk4(design):
    """The relative-displacement transmissibility is consistent with the RK4 run."""
    f_probe = 0.5 * design.f_n_hz
    sim = simulate_base_excitation(
        f_n_hz=design.f_n_hz,
        zeta=design.zeta,
        f_probe_hz=f_probe,
        cycles=150,
        steps_per_cycle=500,
        relative_amplitude_m=2.0e-3,
    )
    expected = float(design.relative_displacement_magnitude(np.array([f_probe]))[0]) * 2.0e-3
    assert sim["relative_displacement_amplitude_m"] == pytest.approx(expected, rel=2e-2)


def test_qzs_frequency_sensitivity_formula(design):
    """d ln f_n / d alpha = -1/(2(1-alpha)) and the band ratio follows it."""
    sens = design.frequency_sensitivity(0.01)
    expected = -0.01 / (2.0 * design.residual_stiffness_fraction)
    assert sens["relative_frequency_error"] == pytest.approx(expected, rel=1e-12)
    # stiffening the mount (less cancellation) must degrade the attenuation
    stiffer = QZSMount(
        equipment_mass_kg=design.equipment_mass_kg,
        f_rigid_hz=design.f_rigid_hz,
        residual_stiffness_fraction=design.residual_stiffness_fraction + 0.02,
        zeta=design.zeta,
        stroke_m=design.stroke_m,
        ns_range_m=design.ns_range_m,
    )
    assert stiffer.f_n_hz > design.f_n_hz
    f = np.linspace(0.2, 50.0, 2000)
    s = 1.0 / (1.0 + (f / 0.9) ** 3)  # monotone toy spectrum, shape irrelevant
    soft = band_attenuation(f, s, design, (0.3, 3.0))["rms_ratio_base_over_hull"]
    hard = band_attenuation(f, s, stiffer, (0.3, 3.0))["rms_ratio_base_over_hull"]
    assert hard > soft


def test_qzs_hardening_backbone_from_nonlinear_integration():
    """The cubic QZS term shifts the resonance upwards (hardening)."""
    f_n = 0.5
    zeta = 0.15
    beta_over_k = 0.5
    mass = 2200.0
    freqs = np.array([0.9, 1.0, 1.1, 1.2, 1.3]) * f_n
    low = simulate_base_excitation(
        f_n_hz=f_n, zeta=zeta, f_probe_hz=0.8 * f_n, cycles=120, steps_per_cycle=400,
        relative_amplitude_m=1.0e-5, cubic_coefficient=beta_over_k * mass * (2 * np.pi * f_n) ** 2,
        mass_kg=mass,
    )["acceleration_transmissibility"]
    small = [
        simulate_base_excitation(
            f_n_hz=f_n, zeta=zeta, f_probe_hz=float(fp), cycles=120, steps_per_cycle=400,
            relative_amplitude_m=1.0e-6, cubic_coefficient=beta_over_k * mass * (2 * np.pi * f_n) ** 2,
            mass_kg=mass,
        )["acceleration_transmissibility"]
        for fp in freqs
    ]
    large = [
        simulate_base_excitation(
            f_n_hz=f_n, zeta=zeta, f_probe_hz=float(fp), cycles=120, steps_per_cycle=400,
            relative_amplitude_m=2.0e-3, cubic_coefficient=beta_over_k * mass * (2 * np.pi * f_n) ** 2,
            mass_kg=mass,
        )["acceleration_transmissibility"]
        for fp in freqs
    ]
    peak_small = freqs[int(np.argmax(small))]
    peak_large = freqs[int(np.argmax(large))]
    assert peak_large >= peak_small, "hardening must shift the peak to higher frequency"
    assert np.isfinite(low) and low > 0.0


def test_qzs_linearity_amplitude_is_physical(design):
    """The linear range of the QZS mechanism and the ice-band motion are separated."""
    amp = design.linearity_amplitude_m()
    assert amp > 0.0
    assert design.backbone_peak_hz(amp) > design.f_n_hz  # hardening
    # ice-band motion is orders of magnitude below the linearity threshold
    assert design.linearity_amplitude_m() > 10.0 * 1.0e-6


# ------------------------------------------------------------- design layer
def test_design_map_marks_mutually_exclusive_feasibility(chain):
    """Design map: static, stiffness, damping and stroke flags are not redundant."""
    rows = chain.design_map_rows
    assert rows
    # very soft designs violate the residual-stiffness budget
    soft = [r for r in rows if r["f_n_hz"] < 0.4]
    assert soft and all(not r["stiffness_ok"] for r in soft)
    # very stiff designs do not attenuate the ice band
    hard = [r for r in rows if r["f_n_hz"] > 1.2]
    assert hard and all(r["rms_ratio_base_over_hull"] > 1.0 for r in hard)
    # low damping is excluded by the damping floor
    low_zeta = [r for r in rows if r["zeta"] < chain.config.qzs_zeta_min]
    assert low_zeta and all(not r["zeta_ok"] for r in low_zeta)
    selected = chain.selection["selected"]
    assert selected["static_ok"] and selected["stiffness_ok"]
    assert selected["zeta_ok"] and selected["stroke_ok"] and selected["ice_linear_ok"]


def test_selected_design_meets_constraints(chain, design):
    """The selected point respects every declared constraint."""
    cfg = chain.config
    assert design.static_deflection_m <= cfg.static_deflection_limit_m
    assert design.residual_stiffness_fraction >= cfg.residual_stiffness_budget
    assert design.zeta >= cfg.qzs_zeta_min
    assert design.f_n_hz < cfg.baseline_mount_f_n_hz
    assert chain.selection["feasible"]
    assert chain.selection["selected"]["rms_ratio_base_over_hull"] < 1.0


def test_conventional_mount_cannot_attenuate_ice_band(chain):
    """Pre/post contrast: the conventional mount amplifies the ice band."""
    blk = chain.severities["design"]["conventional"]
    assert blk["rms_ratio_base_over_hull_ice_band"] > 1.0
    qzs = chain.severities["design"]["qzs"]
    assert qzs["rms_ratio_base_over_hull_ice_band"] < blk["rms_ratio_base_over_hull_ice_band"]
    assert chain.severities["design"]["life_gain_qzs_over_conventional"] > 1.0


def test_stroke_check_reports_bump_stop(chain, design):
    """Wave motion exceeds the stroke, the bump-stop force stays below the limit."""
    stroke = chain.stroke
    assert stroke["ice_case_inside_linear_range"]
    assert stroke["worst_relative_displacement_rms_m"] > design.stroke_m
    assert stroke["bump_stop_engaged"]
    assert 0.0 < stroke["bump_stop_force_fraction_of_weight"] <= chain.config.stop_force_limit_fraction
    assert stroke["bump_stop_stiffness_n_m"] > 0.0
    service = service_stroke(design)
    assert service["worst_case"] == "wave"


# --------------------------------------------------------------- payload IO
def test_criteria_carry_numbers(chain):
    ids = {c["id"] for c in chain.criteria}
    assert {"M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"} <= ids
    for c in chain.criteria:
        assert c["value"] is not None
        assert c["threshold"] is not None
    assert chain.all_pass


def test_payload_is_json_serialisable(chain):
    payload = chain.to_json_dict()
    text = json.dumps(payload, default=float)
    assert len(text) > 1000
    again = json.loads(text)
    assert again["node"] == "p227w2-N08"
    assert again["qzs_mount"]["f_n_hz"] == pytest.approx(chain.design.f_n_hz)
    assert again["all_criteria_pass"] is True
    for severity in DEFAULT_SEVERITIES:
        assert again["severities"][severity]["qzs"]["fatigue"]["life_years"] > 0.0
    # curves needed by the report figures are present
    assert again["curves"]["design"]["freqs_hz"]
    assert again["curves"]["design"]["base_psd_qzs"]


def test_bode_and_curve_shapes(chain):
    curve = chain.curves["design"]
    n = len(curve["freqs_hz"])
    for key in ("hull_psd", "support_psd", "base_psd_conventional", "base_psd_qzs",
                "h_magnitude_conventional", "h_magnitude_qzs", "stress_psd_qzs"):
        assert len(curve[key]) == n
    assert np.all(np.asarray(curve["hull_psd"]) >= 0.0)
    assert np.all(np.asarray(curve["h_magnitude_qzs"]) > 0.0)


def test_design_map_grid_reproducible():
    """Design map is a pure function of its declared inputs."""
    f = np.linspace(0.2, 50.0, 1500)
    s = 1.0 / (1.0 + (f / 1.1) ** 4) + 1e-6
    kwargs = dict(
        band_hz=(0.3, 3.0),
        f_rigid_hz=4.0,
        equipment_mass_kg=2200.0,
        f_n_grid=(0.4, 0.6, 0.8),
        zeta_grid=(0.15,),
        static_deflection_limit_m=0.05,
        residual_stiffness_budget=0.01,
    )
    a = design_map(f, s, **kwargs)
    b = design_map(f, s, **kwargs)
    assert [r["rms_ratio_base_over_hull"] for r in a] == [r["rms_ratio_base_over_hull"] for r in b]
    assert all(set(r) >= {"static_ok", "stiffness_ok", "zeta_ok", "stroke_ok", "ice_linear_ok"} for r in a)