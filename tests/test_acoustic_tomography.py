"""Tests for the under-ice acoustic tomography module.

Coverage
--------
* sound-speed relations: campaign form vs nine-term Mackenzie, analytic
  inversion ``c -> T``, sensitivity, three hard negative controls;
* ray tracing: closed-form linear-profile reference (range and travel time),
  reciprocity, straight line in a homogeneous medium, Snell invariant, ice
  reflection, and the path/slowness quadrature consistency that ties a traced
  arrival time to the tomographic path matrix;
* tomography: exact recovery of a known slowness perturbation, ART and SIRT
  behaviour, uncertainty that grows with the noise level;
* ice thickness: draft-mode echo inversion, closed-form uncertainty budget;
* the campaign case (reduced configuration): the acceptance criteria structure
  and the improvement of the retrieval over the background prior.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.acoustic_tomography import ice_thickness as ice
from tensorphysics.acoustic_tomography.case_under_ice import (
    CaseConfig,
    background_slowness,
    background_sound_speed,
    background_state,
    instrument_positions,
    pair_indices,
    run_case,
    truth_field,
    truth_sound_speed,
    truth_state,
)
from tensorphysics.acoustic_tomography.ray_tracing import (
    SoundSpeedField,
    eigenray_paths,
    find_eigenray,
    find_eigenrays,
    find_eigenrays_multipath,
    grid_sound_speed_field,
    linear_profile_arc,
    specular_reflection_time,
    straight_leg_time,
    trace_rays,
)
from tensorphysics.acoustic_tomography.sound_speed import (
    SOUND_SPEED_LOWER_BOUND,
    engineering_dc_dt,
    engineering_sound_speed,
    ice_sound_speed,
    invert_temperature_from_sound_speed,
    mackenzie_sound_speed,
    sound_speed_valid_mask,
    temperature_uncertainty,
)
from tensorphysics.acoustic_tomography.tomography import (
    NodeGrid,
    art,
    interpolate_nodes,
    invert_sound_speed,
    laplacian,
    monte_carlo_uncertainty,
    path_matrix,
    sirt,
)


# ---------------------------------------------------------------------------
# Sound speed and its inversion
# ---------------------------------------------------------------------------
def test_engineering_sound_speed_reference_points():
    # Campaign relation: 1449.2 + 4.6T - 5.5e-2 T^2 + 2.9e-4 T^3 + 1.34(S-35) + 1.6e-2 P
    assert engineering_sound_speed(0.0, 35.0, 0.0) == pytest.approx(1449.2, abs=1e-9)
    # the salinity term is linear and the pressure term is 0.016 per dbar
    assert engineering_sound_speed(0.0, 36.0, 0.0) == pytest.approx(1450.54, abs=1e-9)
    assert engineering_sound_speed(0.0, 35.0, 100.0) == pytest.approx(1450.8, abs=1e-9)


def test_mackenzie_and_campaign_relation_agree_within_1_m_s():
    t = np.linspace(-1.8, 4.0, 25)
    s = np.linspace(30.5, 35.0, 25)
    d = np.linspace(0.0, 300.0, 25)
    c_campaign = engineering_sound_speed(t, s, d)
    c_mack = mackenzie_sound_speed(t, s, d)
    assert np.max(np.abs(c_campaign - c_mack)) < 1.0      # independent relations
    # literature anchor: Chen-Millero/Mackenzie at T=0, S=35, D=0 ~ 1449.2 m/s
    assert 1448.0 < float(mackenzie_sound_speed(0.0, 35.0, 0.0)) < 1450.0


def test_temperature_inversion_is_exact_round_trip():
    rng = np.random.default_rng(0)
    t = rng.uniform(-1.9, 20.0, 200)
    s = rng.uniform(28.0, 38.0, 200)
    d = rng.uniform(0.0, 2000.0, 200)
    c = engineering_sound_speed(t, s, d)
    t_back = invert_temperature_from_sound_speed(c, s, d)
    assert np.all(np.isfinite(t_back))
    assert np.max(np.abs(t_back - t)) < 1e-6


def test_temperature_inversion_flags_out_of_range():
    # Fresh water at 0 degC is ~1402 m/s; a target of 1390 m/s is unreachable
    t_back = invert_temperature_from_sound_speed(1390.0, 0.0, 0.0)
    assert not np.isfinite(t_back)


def test_temperature_sensitivity_and_uncertainty():
    slope = engineering_dc_dt(np.array([-1.5, 0.0, 10.0]))
    assert np.all(slope > 3.5)                 # 4.77 / 4.60 / 3.59 m/s per degC
    assert 4.5 < slope[0] < 5.0                # cold water is the sensitive end
    sigma_t = temperature_uncertainty(2.3, 0.0)
    assert sigma_t == pytest.approx(2.3 / engineering_dc_dt(0.0), rel=1e-12)


def test_negative_control_temperature_zero_sound_speed():
    """Criterion (d3): at 0 degC the sound speed stays above 1400 m/s."""
    c = float(engineering_sound_speed(0.0, 0.0, 0.0))
    assert c == pytest.approx(1402.3, abs=1e-6)
    assert c >= SOUND_SPEED_LOWER_BOUND
    assert float(mackenzie_sound_speed(0.0, 0.0, 0.0)) >= SOUND_SPEED_LOWER_BOUND


def test_validity_mask():
    assert bool(sound_speed_valid_mask(0.0, 35.0, 100.0))
    assert not bool(sound_speed_valid_mask(-5.0, 35.0, 100.0))
    assert not bool(sound_speed_valid_mask(0.0, 10.0, 100.0))


def test_ice_sound_speed_decreases_with_warming():
    assert ice_sound_speed(0.0) == pytest.approx(3800.0)
    assert ice_sound_speed(-10.0) > ice_sound_speed(0.0)


# ---------------------------------------------------------------------------
# Ray tracing
# ---------------------------------------------------------------------------
def _linear_field(c0=1450.0, grad=0.06):
    return SoundSpeedField(lambda r, z: c0 + grad * z)


def test_tracer_matches_closed_form_linear_profile():
    """RK4 tracer vs the exact solution for c(z) = c0 + g z (<1e-6 relative)."""
    c0, grad = 1450.0, 0.06
    fld = _linear_field(c0, grad)
    z0, theta0, r_goal = 50.0, 0.05, 600.0
    p = np.cos(theta0) / fld(0.0, z0)
    # analytic landing depth for this range, then the exact travel time.
    # The bracket must stay below the turning depth z_turn where p*c = 1.
    from scipy.optimize import brentq

    z_turn = (1.0 / p - c0) / grad
    assert z_turn > z0
    z_end = brentq(lambda zb: linear_profile_arc(c0, grad, p, z0, zb)[0] - r_goal,
                   z0, 0.98 * z_turn)
    r_exact, t_exact = linear_profile_arc(c0, grad, p, z0, z_end)
    res = trace_rays(fld, 0.0, z0, np.array([theta0]), r_goal, step_len=10.0, n_steps=200)
    assert res["valid"][0]
    assert abs(res["z"][0] - z_end) < 1e-6                    # metres
    assert abs(res["time"][0] - t_exact) / t_exact < 1e-6      # seconds
    assert r_exact == pytest.approx(r_goal, abs=1e-6)


def test_snell_invariant_is_conserved_in_a_stratified_medium():
    fld = _linear_field(1445.0, 0.05)
    theta0 = 0.08
    res = trace_rays(fld, 0.0, 30.0, np.array([theta0]), 1500.0, step_len=10.0,
                     n_steps=400, keep_path=True)
    theta = np.arctan2(np.gradient(res["z_path"][0][np.isfinite(res["z_path"][0])]),
                       np.gradient(res["r_path"][0][np.isfinite(res["r_path"][0])]))
    z = res["z_path"][0][np.isfinite(res["z_path"][0])]
    c = fld(res["r_path"][0][np.isfinite(res["r_path"][0])], z)
    p = np.cos(theta) / c
    assert np.allclose(p, p[0], rtol=2e-3)


def test_homogeneous_medium_gives_straight_line_time():
    fld = SoundSpeedField(lambda r, z: 1500.0 + 0.0 * r)
    res = trace_rays(fld, 0.0, 20.0, np.array([0.10]), 1000.0, step_len=20.0, n_steps=100)
    path_len = np.hypot(1000.0, res["z"][0] - 20.0)
    assert res["time"][0] == pytest.approx(path_len / 1500.0, rel=1e-9)


def test_batch_eigenray_search_matches_straight_line():
    """The zero-bounce batch solver reproduces straight-line times when c is constant."""
    fld = SoundSpeedField(lambda r, z: 1480.0 + 0.0 * r)
    r0 = np.array([0.0, 100.0])
    z0 = np.array([50.0, 120.0])
    r1 = np.array([2000.0, 1500.0])
    z1 = np.array([80.0, 60.0])
    res = find_eigenrays(fld, r0, z0, r1, z1, step_len=20.0)
    assert np.all(res["valid"])
    for i in range(2):
        straight = np.hypot(r1[i] - r0[i], z1[i] - z0[i]) / 1480.0
        assert res["time"][i] == pytest.approx(straight, rel=1e-8)
        assert abs(res["eigenrays"][i].residual_z) < 1e-6
    assert len(eigenray_paths(res)) == 2


def test_reciprocity_of_travel_time():
    fld = _linear_field(1447.0, 0.07)
    a = find_eigenray(fld, 0.0, 50.0, 1200.0, 140.0, step_len=20.0)
    b = find_eigenray(fld, 1200.0, 140.0, 0.0, 50.0, step_len=20.0)
    assert a is not None and b is not None
    assert a.time == pytest.approx(b.time, rel=1e-10)


def test_ice_reflection_is_specular_and_adds_path():
    fld = SoundSpeedField(lambda r, z: 1450.0 + 0.0 * r)
    res = trace_rays(fld, 0.0, 100.0, np.array([-0.20]), 1000.0, step_len=10.0,
                     n_steps=300, z_top=0.0, keep_path=True)
    assert res["valid"][0]
    assert int(res["n_reflections"][0]) == 1
    assert np.nanmin(res["z_path"][0]) >= -1e-9    # never rises above the ice underside
    # in a homogeneous medium the path is two straight legs and t = L_arc / c
    path = res["polylines"][0]
    legs = np.sum(np.hypot(np.diff(path[:, 0]), np.diff(path[:, 1])))
    assert res["time"][0] == pytest.approx(legs / 1450.0, rel=1e-6)
    # the mirror construction: reflection at depth/|sin| gives a symmetric path
    r_bounce = float(path[:, 0][np.argmin(path[:, 1])])
    assert 200.0 < r_bounce < 800.0


def test_specular_reflection_time_matches_mirror_formula():
    """In a homogeneous medium Fermat minimisation reproduces the mirror formula."""
    c = 1450.0
    fld = SoundSpeedField(lambda r, z: c + 0.0 * r)
    L, depth = 3000.0, 50.0
    t_pred, r_b = specular_reflection_time(fld, 0.0, depth, L, depth, z_interface=0.0)
    t_mirror = 2.0 * np.hypot(L / 2.0, depth) / c
    assert t_pred == pytest.approx(t_mirror, rel=1e-9)
    assert r_b == pytest.approx(L / 2.0, rel=1e-3)


def test_multipath_finds_several_ducted_arrivals():
    cfg = CaseConfig()
    fld = truth_field(cfg)
    src = np.array([[0.0, 40.0]])
    rcv = np.array([[2000.0, 160.0]])
    res = find_eigenrays_multipath(fld, src[:, 0], src[:, 1], rcv[:, 0], rcv[:, 1],
                                   step_len=40.0, n_scan=121, n_refine=12, z_max=cfg.z_max)
    times = np.asarray(res["times"][0])
    assert times.size >= 1
    assert np.all(np.diff(times) > 0.0)         # sorted by arrival time
    assert res["n_roots"][0] >= 1
    # a ducted ray that reaches 160 m depth from 40 m must bounce off the ice
    assert any(er.n_reflections >= 1 for er in res["eigenrays"][0])


def test_path_matrix_quadrature_reproduces_traced_travel_time():
    """The path-integral matrix must reproduce the traced time (regression test).

    This is the check that exposed a path-length error at the ice reflections:
    the polyline used to cut the reflection corner, shortening the path by up to
    one step per bounce and biasing travel times by several milliseconds.
    """
    cfg = CaseConfig()
    fld = truth_field(cfg)                     # 2-D, range dependent
    bg = SoundSpeedField(lambda r, z: background_sound_speed(z), dr=25.0, dz=5.0)
    src = np.array([[0.0, 40.0], [0.0, 160.0]])
    rcv = np.array([[3000.0, 160.0], [3000.0, 40.0]])
    res = find_eigenrays_multipath(bg, src[:, 0], src[:, 1], rcv[:, 0], rcv[:, 1],
                                   step_len=30.0, n_scan=161, n_refine=12, z_max=cfg.z_max)
    nodes = NodeGrid(np.linspace(0.0, 4000.0, 9), np.linspace(0.0, 200.0, 9))
    s_bg = background_slowness(nodes)
    checked = 0
    for pair in res["eigenrays"]:
        for er in pair:
            A, length = path_matrix([er.path], nodes, ds=5.0)
            t_quad = float((A @ s_bg.reshape(-1))[0])
            assert abs(t_quad - er.time) < 1e-3        # < 1 ms quadrature error
            assert length[0] == pytest.approx(np.sum(np.hypot(np.diff(er.r), np.diff(er.z))), rel=1e-9)
            checked += 1
    assert checked >= 2
    # and the 2-D field must actually differ from the background along the path
    shift = truth_field(cfg)(0.5 * 3000.0, 100.0) - bg(0.5 * 3000.0, 100.0)
    assert abs(shift) > 0.1


# ---------------------------------------------------------------------------
# Tomography
# ---------------------------------------------------------------------------
def _synthetic_tomography_problem(seed=0, noise=0.0, n_nodes=(6, 4)):
    nodes = NodeGrid(np.linspace(0.0, 4000.0, n_nodes[0]), np.linspace(0.0, 200.0, n_nodes[1]))
    r_mesh, z_mesh = nodes.mesh()
    truth_slowness = 1.0 / (1450.0 + 6.0 * np.exp(-((r_mesh - 2000.0) / 900.0) ** 2)
                            * np.exp(-((z_mesh - 120.0) / 70.0) ** 2))
    rng = np.random.default_rng(seed)
    paths = []
    for _ in range(60):
        r0 = rng.uniform(0.0, 3000.0)
        z0 = rng.uniform(20.0, 180.0)
        r1 = r0 + rng.uniform(500.0, 4000.0 - r0)
        z1 = rng.uniform(20.0, 180.0)
        paths.append(np.array([[r0, z0], [r1, z1]]))
    A, _ = path_matrix(paths, nodes, ds=25.0)
    t = A @ truth_slowness.reshape(-1)
    t = t + rng.normal(0.0, noise, size=t.size) if noise > 0 else t
    return nodes, paths, truth_slowness, A, t


def test_sirt_and_art_recover_a_known_field():
    nodes, paths, truth, A, t = _synthetic_tomography_problem(noise=0.0)
    bg = np.full(nodes.size, 1.0 / 1450.0)
    for method in ("sirt", "art"):
        inv = invert_sound_speed(paths, t, nodes, bg, n_iter=400, relaxation=1.0,
                                 smooth=0.05, method=method)
        err = np.max(np.abs(inv["sound_speed"] - 1.0 / truth))
        assert err < 1.0, (method, err)      # < 1 m/s on a 6 m/s anomaly
        assert inv["rms_residual_s"] < 1e-5


def test_sirt_early_stop_by_discrepancy_principle():
    nodes, paths, truth, A, t = _synthetic_tomography_problem(noise=1e-3)
    bg = np.full(nodes.size, 1.0 / 1450.0)
    inv_free = invert_sound_speed(paths, t, nodes, bg, n_iter=600, smooth=0.1)
    inv_stop = invert_sound_speed(paths, t, nodes, bg, n_iter=600, smooth=0.1,
                                  target_residual=1e-3, check_every=25)
    assert inv_stop["n_iter"] <= inv_free["n_iter"]
    assert inv_stop["rms_residual_s"] <= 1.5e-3


def test_negative_control_no_data_is_nan():
    """Criterion (d1): without acoustic data the inversion is identically NaN."""
    nodes, paths, truth, A, t = _synthetic_tomography_problem()
    bg = np.full(nodes.size, 1.0 / 1450.0)
    inv = invert_sound_speed([], [], nodes, bg)
    assert np.all(np.isnan(inv["sound_speed"]))
    assert inv["n_observations"] == 0
    inv_empty = invert_sound_speed(paths, np.full(len(paths), np.nan), nodes, bg)
    assert np.all(np.isnan(inv_empty["sound_speed"]))


def test_monte_carlo_uncertainty_grows_with_noise():
    nodes, paths, truth, A, t = _synthetic_tomography_problem(noise=0.0)
    bg = np.full(nodes.size, 1.0 / 1450.0)
    low = monte_carlo_uncertainty(paths, t, nodes, bg, 0.5e-3, n_realisations=12, seed=1,
                                  smooth=0.05, 
                                  n_iter=200)
    high = monte_carlo_uncertainty(paths, t, nodes, bg, 5.0e-3, n_realisations=12, seed=1,
                                   smooth=0.05, n_iter=200)
    assert high["sigma_mean"] > 3.0 * low["sigma_mean"]
    assert np.all(np.isfinite(high["sigma_sound_speed"]))


def test_bilinear_interpolation_and_laplacian():
    nodes = NodeGrid(np.array([0.0, 1000.0]), np.array([0.0, 100.0]))
    values = np.array([[1.0, 3.0], [5.0, 7.0]])
    assert interpolate_nodes(values, nodes, 0.0, 0.0) == pytest.approx(1.0)
    assert interpolate_nodes(values, nodes, 500.0, 50.0) == pytest.approx(4.0)
    ramp = nodes.mesh()[0] * 0.001
    assert np.allclose(laplacian(nodes.flatten(ramp), nodes, physical=True), 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# Ice thickness
# ---------------------------------------------------------------------------
def test_ice_thickness_echo_inversion_round_trip():
    h = np.array([0.5, 1.0, 2.0, 3.5])
    c_w = 1440.0
    draft = ice.draft_from_thickness(h)
    assert np.allclose(draft, h * 917.0 / 1027.0)
    t = ice.echo_time(draft, c_w)
    assert np.allclose(ice.thickness_from_echo_time(t, c_w), h, rtol=1e-12)


def test_negative_control_ice_thickness_to_zero():
    """Criterion (d2): h -> 0 makes the reflection travel time vanish."""
    h = np.array([2.0, 1.0, 0.1, 1e-2, 1e-3, 0.0])
    t = ice.echo_time(ice.draft_from_thickness(h), 1440.0)
    assert t[-1] == 0.0
    assert np.all(np.diff(t) < 0.0)
    # linearity in h, no offset
    assert t[0] == pytest.approx(2.0 * ice.draft_from_thickness(2.0) / 1440.0, rel=1e-12)
    assert t[1] == pytest.approx(t[0] / 2.0, rel=1e-12)


def test_through_ice_mode():
    h = ice.thickness_from_ice_echo_time(0.001, 3800.0)
    assert h == pytest.approx(0.5 * 3800.0 * 0.001, rel=1e-12)


def test_thickness_uncertainty_budget():
    h = 2.0
    t = float(ice.echo_time(ice.draft_from_thickness(h), 1440.0))
    sigma_t_only = float(ice.thickness_uncertainty(t, 5e-5, 1440.0))
    assert sigma_t_only == pytest.approx(0.5 * 1440.0 * 1027.0 / 917.0 * 5e-5, rel=1e-9)
    # a density-ratio error is the dominant systematic for a 2 % ratio error
    sigma_ratio = float(ice.thickness_uncertainty(t, 5e-5, 1440.0, sigma_density_ratio=0.02))
    assert sigma_ratio > sigma_t_only
    assert sigma_ratio < 0.1


# ---------------------------------------------------------------------------
# Case level (reduced configuration)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def reduced_case():
    cfg = CaseConfig(
        instrument_ranges=(0.0, 2500.0, 4000.0),
        instrument_depths=(40.0, 160.0),
        n_scan=41,
        theta_max_deg=12.0,
        n_refine=8,
        step_len=60.0,
        n_r_nodes=5,
        n_z_nodes=4,
        n_iter=150,
        n_mc=4,
        n_bending=0,
        n_reciprocity_pairs=1,
        ice_stations=20,
        ctd_r=(500.0, 1500.0, 2500.0, 3500.0),
        ctd_z=(40.0, 100.0, 160.0),
    )
    return cfg, run_case(cfg)


def test_case_criteria_structure(reduced_case):
    cfg, res = reduced_case
    crit = res["criteria"]
    assert set(crit) >= {
        "a_sound_speed", "b_ice_thickness", "c_temperature",
        "d_negative_controls", "e_falsifiability", "all_pass",
    }
    assert crit["b_ice_thickness"]["n_samples"] >= 20
    assert crit["a_sound_speed"]["n_samples"] >= 30 or cfg.ctd_r is not None
    assert crit["d_negative_controls"]["all_pass"] is True


def test_case_observations_and_paths_are_consistent(reduced_case):
    """The case wires identified arrivals to inversion paths correctly.

    (The accuracy of the reconstruction is established by the full campaign run
    and by ``test_sirt_and_art_recover_a_known_field``; this smoke
    configuration is deliberately too coarse for a performance claim.)
    """
    cfg, res = reduced_case
    diag = res["diagnostics"]
    assert diag["n_observations"] > 0
    assert diag["n_pairs_used"] > 0
    # the discrepancy-principle stop must leave the residual at the noise level
    assert diag["rms_residual_stage1_s"] <= cfg.sigma_time
    obs = res["observations"]
    assert obs["paths"] and len(obs["paths"]) == obs["t_obs"].size
    pos = instrument_positions(cfg)
    for path, geom in zip(obs["paths"], obs["geometry"]):
        assert path.shape[1] == 2
        assert path[:, 1].min() >= -1e-6               # stays below the ice
        assert np.all(path[:, 1] <= cfg.z_max + 1e-6)  # and above the bottom
        # endpoints must coincide with an instrument position
        d0 = np.min(np.hypot(pos[:, 0] - path[0, 0], pos[:, 1] - path[0, 1]))
        d1 = np.min(np.hypot(pos[:, 0] - path[-1, 0], pos[:, 1] - path[-1, 1]))
        assert d0 < 1.5 and d1 < 1.5
    # travel times must be physically plausible for the array aperture
    t = obs["t_obs"]
    assert np.all(t > 0.0)
    assert np.all(t < 2.0 * cfg.r_max / 1400.0)
    # the truth arrival set is genuinely multi-arrival (ducted propagation)
    assert diag["n_arrivals_truth_model"] > diag["n_pairs_used"]
    assert len(diag["bounce_orders"]) >= 2


def test_case_temperature_and_ice_thickness_are_physical(reduced_case):
    cfg, res = reduced_case
    t_ret = res["fields"]["t_retrieved_samples"]
    assert np.all(np.isfinite(t_ret))
    assert t_ret.min() > -2.2 and t_ret.max() < 12.0
    h = res["fields"]["h_retrieved"]
    assert np.all(h > 0.0)
    sigma_h = res["fields"]["sigma_h"]
    assert np.all(sigma_h > 0.0)


def test_case_truth_state_is_physical():
    cfg = CaseConfig()
    r = np.linspace(0.0, cfg.r_max, 15)
    z = np.linspace(0.0, cfg.z_max, 15)
    t, s = truth_state(r, z, cfg)
    # above the freezing point of sea water at these salinities
    assert np.all(t > -1.85)
    assert np.all(s > 28.0) and np.all(s < 36.0)
    c = truth_sound_speed(r, z, cfg)
    assert np.all(c > 1425.0) and np.all(c < 1475.0)
    # the sound-speed minimum sits at the ice underside -> surface duct
    assert truth_sound_speed(0.0, 0.0, cfg) < truth_sound_speed(0.0, 200.0, cfg)