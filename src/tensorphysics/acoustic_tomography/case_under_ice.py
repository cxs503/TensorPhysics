"""Under-ice acoustic tomography case: sound speed, ice thickness, temperature.

The synthetic-but-physical case follows the campaign definition:

* vertical plane ``0 <= r <= 4000 m`` (range), ``0 <= z <= 200 m`` (depth, ice
  underside at ``z = 0``), a ten-element transceiver array towed/moored at two
  depths (40 m and 160 m) on five ranges,
* a *declared* horizontally uniform background state (climatology / reference
  CTD cast) plus a warm, salty Atlantic-water intrusion and a cold melt-water
  lens; the tomography target is the anomaly on a 6 x 5 node grid,
* forward data from real Snell-law ray tracing through the truth field, with
  Gaussian timing noise (sigma = 1 ms),
* ART/SIRT inversion of the travel times, one ray-bending iteration, then
  CTD-point comparison for (a) sound speed and (c) temperature,
* draft-mode echo inversion for (b) ice thickness,
* Monte-Carlo (ensemble) uncertainties sigma_c, sigma_T and propagated sigma_h,
* the three hard negative controls and the falsifiability check of the campaign.

Every number reported by :func:`run_case` is computed, none is hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np

from . import ice_thickness as ice
from .ray_tracing import (
    SoundSpeedField,
    find_eigenrays_multipath,
    grid_sound_speed_field,
    specular_reflection_time,
    straight_leg_time,
)
from .sound_speed import (
    SOUND_SPEED_LOWER_BOUND,
    engineering_dc_dt,
    engineering_sound_speed,
    invert_temperature_from_sound_speed,
    temperature_uncertainty,
)
from .tomography import (
    NodeGrid,
    interpolate_nodes,
    invert_sound_speed,
    monte_carlo_uncertainty,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class CaseConfig:
    """Geometry, ocean state and measurement parameters of the case."""

    r_max: float = 4000.0
    z_max: float = 200.0
    instrument_ranges: tuple = (0.0, 1000.0, 2000.0, 3000.0, 4000.0)
    instrument_depths: tuple = (40.0, 100.0, 160.0)
    min_pair_range: float = 500.0
    #: timing noise of the tomography travel times [s] (1 ms at 100-1000 Hz)
    sigma_time: float = 1.0e-3
    #: ray-integration step [m]
    step_len: float = 30.0
    #: multipath eigenray search
    n_scan: int = 241
    theta_max_deg: float = 18.0
    n_refine: int = 16
    max_reflections: int = 8
    max_arrivals: int = 6
    association_tol: float = 0.020
    association_theta_tol: float = 0.020
    n_reciprocity_pairs: int = 6
    #: inversion grid
    n_r_nodes: int = 8
    n_z_nodes: int = 6
    n_iter: int = 800
    relaxation: float = 1.0
    smooth: float = 0.2
    check_every: int = 25
    n_bending: int = 1
    n_mc: int = 60
    seed: int = 20260101
    #: warm Atlantic-water intrusion
    eddy_centre_r: float = 2200.0
    eddy_centre_z: float = 150.0
    eddy_scale_r: float = 900.0
    eddy_scale_z: float = 60.0
    eddy_dT: float = 1.3
    eddy_dS: float = 0.25
    #: cold, fresh melt-water lens below the ice
    lens_centre_r: float = 1500.0
    lens_centre_z: float = 35.0
    lens_scale_r: float = 1100.0
    lens_scale_z: float = 25.0
    lens_dT: float = -0.10
    lens_dS: float = -0.50
    #: CTD comparison samples
    ctd_r: tuple = (200.0, 800.0, 1400.0, 2000.0, 2600.0, 3200.0, 3800.0)
    ctd_z: tuple = (20.0, 60.0, 100.0, 140.0, 180.0)
    #: ice thickness (draft mode)
    ice_stations: int = 24
    ice_transducer_depth: float = 40.0
    sigma_echo: float = 5.0e-5
    sigma_echo_stress: float = 5.0e-4
    sigma_density_ratio: float = 0.02
    #: falsifiability anchor of the campaign
    ref_baseline: float = 3750.0
    ref_depth: float = 40.0
    ref_time_measured: float = 2.6
    ref_time_tolerance: float = 10.0e-3
    extra: dict = dc_field(default_factory=dict)


def temperature_from_sound_speed_clamped(c, salinity, depth, t_lo=-2.0, t_hi=30.0):
    """Invert ``c -> T`` and flag points at the edge of the relation's range.

    A retrieved sound speed that falls outside the range spanned by
    ``[t_lo, t_hi]`` (the Mackenzie validity box) cannot be inverted.  Such
    points are clamped to the interval edge *and counted*, so the report shows
    how often the retrieval hits the physical limit instead of silently
    dropping samples.
    """
    c = np.asarray(c, dtype=float)
    c_lo = engineering_sound_speed(t_lo, salinity, depth)
    c_hi = engineering_sound_speed(t_hi, salinity, depth)
    n_clamped = int(np.sum(c < c_lo) + np.sum(c > c_hi))
    t = invert_temperature_from_sound_speed(np.clip(c, c_lo, c_hi), salinity, depth, t_lo=t_lo, t_hi=t_hi)
    return t, n_clamped


# ---------------------------------------------------------------------------
# Ocean state
# ---------------------------------------------------------------------------
def background_state(z):
    """Declared background (climatological) temperature [degC] and salinity."""
    z = np.asarray(z, dtype=float)
    zeta = np.clip(z / 200.0, 0.0, 1.5)
    temperature = -1.60 + 3.30 * zeta**1.5
    salinity = 31.0 + 3.8 * zeta**0.8
    return temperature, salinity


def anomaly_state(r, z, cfg: CaseConfig):
    """Temperature and salinity anomaly of the case (Atlantic intrusion + lens)."""
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    eddy = np.exp(-(((r - cfg.eddy_centre_r) / cfg.eddy_scale_r) ** 2)) * np.exp(
        -(((z - cfg.eddy_centre_z) / cfg.eddy_scale_z) ** 2)
    )
    lens = np.exp(-(((r - cfg.lens_centre_r) / cfg.lens_scale_r) ** 2)) * np.exp(
        -(((z - cfg.lens_centre_z) / cfg.lens_scale_z) ** 2)
    )
    d_temperature = cfg.eddy_dT * eddy + cfg.lens_dT * lens
    d_salinity = cfg.eddy_dS * eddy + cfg.lens_dS * lens
    return d_temperature, d_salinity


def truth_state(r, z, cfg: CaseConfig):
    """Truth temperature [degC] and salinity [psu] at ``(r, z)``."""
    t_bg, s_bg = background_state(z)
    d_t, d_s = anomaly_state(r, z, cfg)
    return t_bg + d_t, s_bg + d_s


def truth_sound_speed(r, z, cfg: CaseConfig):
    """Truth sound speed [m/s] from the campaign relation (pressure = depth)."""
    t, s = truth_state(r, z, cfg)
    return engineering_sound_speed(t, s, z)


def background_sound_speed(z):
    """Background (prior) sound speed [m/s] -- horizontally uniform."""
    t, s = background_state(z)
    return engineering_sound_speed(t, s, z)


def background_state_gradient(z):
    """Analytic ``dT/dz``, ``dS/dz`` of the background profile."""
    z = np.asarray(z, dtype=float)
    zeta = np.clip(z / 200.0, 1e-9, 1.5)
    d_t = 3.30 * 1.5 * np.sqrt(zeta) / 200.0
    d_s = 3.8 * 0.8 * zeta ** (-0.2) / 200.0
    return d_t, d_s


def anomaly_state_gradient(r, z, cfg: CaseConfig):
    """Analytic ``(dT/dr, dT/dz, dS/dr, dS/dz)`` of the anomaly field."""
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    gr = -2.0 * (r - cfg.eddy_centre_r) / cfg.eddy_scale_r**2
    gz = -2.0 * (z - cfg.eddy_centre_z) / cfg.eddy_scale_z**2
    eddy = np.exp(-(((r - cfg.eddy_centre_r) / cfg.eddy_scale_r) ** 2)) * np.exp(
        -(((z - cfg.eddy_centre_z) / cfg.eddy_scale_z) ** 2)
    )
    gr_l = -2.0 * (r - cfg.lens_centre_r) / cfg.lens_scale_r**2
    gz_l = -2.0 * (z - cfg.lens_centre_z) / cfg.lens_scale_z**2
    lens = np.exp(-(((r - cfg.lens_centre_r) / cfg.lens_scale_r) ** 2)) * np.exp(
        -(((z - cfg.lens_centre_z) / cfg.lens_scale_z) ** 2)
    )
    d_t_dr = cfg.eddy_dT * eddy * gr + cfg.lens_dT * lens * gr_l
    d_t_dz = cfg.eddy_dT * eddy * gz + cfg.lens_dT * lens * gz_l
    d_s_dr = cfg.eddy_dS * eddy * gr + cfg.lens_dS * lens * gr_l
    d_s_dz = cfg.eddy_dS * eddy * gz + cfg.lens_dS * lens * gz_l
    return d_t_dr, d_t_dz, d_s_dr, d_s_dz


def truth_sound_speed_gradient(r, z, cfg: CaseConfig):
    """Analytic ``(dc/dr, dc/dz)`` of :func:`truth_sound_speed` (pressure = z)."""
    t, s = truth_state(r, z, cfg)
    d_t_dr, d_t_dz, d_s_dr, d_s_dz = anomaly_state_gradient(r, z, cfg)
    d_t_bg_dz, d_s_bg_dz = background_state_gradient(z)
    slope = engineering_dc_dt(t)
    dc_dr = slope * d_t_dr + 1.34 * d_s_dr
    dc_dz = slope * (d_t_dz + d_t_bg_dz) + 1.34 * (d_s_dz + d_s_bg_dz) + 1.6e-2
    return dc_dr, dc_dz


def truth_sound_speed_evaluate(r, z, cfg: CaseConfig):
    """Return ``(c, dc/dr, dc/dz)`` with the state computed only once."""
    t, s = truth_state(r, z, cfg)
    c = engineering_sound_speed(t, s, z)
    d_t_dr, d_t_dz, d_s_dr, d_s_dz = anomaly_state_gradient(r, z, cfg)
    d_t_bg_dz, d_s_bg_dz = background_state_gradient(z)
    slope = engineering_dc_dt(t)
    return (
        c,
        slope * d_t_dr + 1.34 * d_s_dr,
        slope * (d_t_dz + d_t_bg_dz) + 1.34 * (d_s_dz + d_s_bg_dz) + 1.6e-2,
    )


def truth_field(cfg: CaseConfig) -> SoundSpeedField:
    return SoundSpeedField(
        lambda r, z: truth_sound_speed(r, z, cfg),
        dr=25.0,
        dz=5.0,
        gradient=lambda r, z: truth_sound_speed_gradient(r, z, cfg),
        evaluate=lambda r, z: truth_sound_speed_evaluate(r, z, cfg),
    )


def background_field(cfg: CaseConfig) -> SoundSpeedField:
    return SoundSpeedField(lambda r, z: background_sound_speed(z), dr=25.0, dz=5.0)


# ---------------------------------------------------------------------------
# Array geometry
# ---------------------------------------------------------------------------
def instrument_positions(cfg: CaseConfig):
    """All transceiver positions ``(n_instr, 2)`` as ``[range, depth]``."""
    pos = []
    for r in cfg.instrument_ranges:
        for z in cfg.instrument_depths:
            pos.append((float(r), float(z)))
    return np.asarray(pos, dtype=float)


def pair_indices(cfg: CaseConfig):
    """Unordered transceiver pairs with horizontal separation >= min_pair_range."""
    pos = instrument_positions(cfg)
    pairs = []
    for i in range(pos.shape[0]):
        for j in range(i + 1, pos.shape[0]):
            if abs(pos[j, 0] - pos[i, 0]) >= cfg.min_pair_range:
                pairs.append((i, j))
    return np.asarray(pairs, dtype=int)


def node_grid(cfg: CaseConfig) -> NodeGrid:
    return NodeGrid(
        np.linspace(0.0, cfg.r_max, cfg.n_r_nodes),
        np.linspace(0.0, cfg.z_max, cfg.n_z_nodes),
    )


def ctd_samples(cfg: CaseConfig):
    rr, zz = np.meshgrid(np.asarray(cfg.ctd_r), np.asarray(cfg.ctd_z), indexing="ij")
    return rr.reshape(-1), zz.reshape(-1)


def background_slowness(nodes: NodeGrid):
    """Prior slowness [s/m] at the grid nodes (background profile)."""
    r_mesh, z_mesh = nodes.mesh()
    return 1.0 / background_sound_speed(z_mesh)


# ---------------------------------------------------------------------------
# Ice thickness truth
# ---------------------------------------------------------------------------
def ice_thickness_truth(r):
    """Level-ice thickness [m]: background sheet + ridge-like thickening."""
    r = np.asarray(r, dtype=float)
    return (
        1.85
        + 0.45 * np.sin(2.0 * np.pi * r / 1500.0)
        + 0.25 * np.exp(-(((r - 2600.0) / 700.0) ** 2))
    )


def ice_station_ranges(cfg: CaseConfig):
    return np.linspace(100.0, cfg.r_max - 100.0, cfg.ice_stations)



# ---------------------------------------------------------------------------
# Multipath arrivals and their association between two models
# ---------------------------------------------------------------------------
def _multipath(field, cfg: CaseConfig, src, rcv):
    return find_eigenrays_multipath(
        field,
        src[:, 0],
        src[:, 1],
        rcv[:, 0],
        rcv[:, 1],
        step_len=cfg.step_len,
        n_scan=cfg.n_scan,
        theta_max_deg=cfg.theta_max_deg,
        n_refine=cfg.n_refine,
        z_top=0.0,
        z_max=cfg.z_max,
        max_reflections=cfg.max_reflections,
        max_arrivals=cfg.max_arrivals,
    )


def _group_key(ray):
    return int(ray.n_reflections)


def _associate(model, truth, tol: float = 0.020, theta_tol: float = 0.020):
    """One-to-one association of the arrivals of two models (per pair).

    An arrival is identified by three independent labels: the number of ice
    reflections (discrete and robust), the launch angle (continuous, so the
    same physical ray stays within ``theta_tol`` when the medium changes) and
    the travel time (``tol``).  Candidates are ranked by ``|t_model - t_data|``
    and accepted greedily, so no arrival is ever invented; unmatched arrivals
    and the largest accepted time shift are reported, which makes the
    identification quality auditable.
    """
    n_pairs = len(truth["times"])
    paths, t_obs, geometry, pair_id, bounce, dt = [], [], [], [], [], []
    n_bg_total = 0
    n_obs_total = 0
    for i in range(n_pairs):
        bg_rays = list(model["eigenrays"][i])
        ob_rays = list(truth["eigenrays"][i])
        n_bg_total += len(bg_rays)
        n_obs_total += len(ob_rays)
        if not bg_rays or not ob_rays:
            continue
        candidates = sorted(
            (
                (abs(a.time - b.time), ia, ib)
                for ia, a in enumerate(bg_rays)
                for ib, b in enumerate(ob_rays)
                if int(a.n_reflections) == int(b.n_reflections)
                and abs(a.theta0 - b.theta0) <= theta_tol
            )
        )
        used_a, used_b = set(), set()
        for d, ia, ib in candidates:
            if d > tol:
                break
            if ia in used_a or ib in used_b:
                continue
            used_a.add(ia)
            used_b.add(ib)
            a, b = bg_rays[ia], ob_rays[ib]
            paths.append(a.path)
            t_obs.append(float(b.time))
            geometry.append((float(a.r[0]), float(a.z[0]), float(a.r[-1]), float(a.z[-1])))
            pair_id.append(i)
            bounce.append(int(b.n_reflections))
            dt.append(float(b.time - a.time))
    t_obs = np.asarray(t_obs, dtype=float)
    order = np.argsort(t_obs) if t_obs.size else np.array([], dtype=int)
    return {
        "paths": [paths[k] for k in order],
        "t_obs": t_obs[order],
        "geometry": [geometry[k] for k in order],
        "pair_id": np.asarray(pair_id, dtype=int)[order],
        "bounce": np.asarray(bounce, dtype=int)[order],
        "t_model": np.asarray(dt, dtype=float)[order],
        "n_observations": int(t_obs.size),
        "n_pairs_used": int(np.unique(np.asarray(pair_id, dtype=int)).size) if paths else 0,
        "n_arrivals_model": int(n_bg_total),
        "n_arrivals_truth": int(n_obs_total),
    }


def reciprocity_check(field, cfg: CaseConfig, src, rcv):
    """Travel-time reciprocity ``t(i->j) == t(j->i)`` on a mirrored field.

    The tracer integrates in the +r direction, so the return path is obtained
    by mirroring the range axis (``c'(r, z) = c(r_max - r, z)``), which leaves
    the medium unchanged and makes the comparison exact in principle.
    """
    r_span = float(cfg.r_max)

    def speed(r, z):
        return field(r_span - np.asarray(r, dtype=float), z)

    def gradient(r, z):
        g_r, g_z = field.gradient(r_span - np.asarray(r, dtype=float), z)
        return -g_r, g_z

    mirrored = SoundSpeedField(speed, dr=field.dr, dz=field.dz, gradient=gradient)
    fwd = _multipath(field, cfg, src, rcv)
    rev_src = np.column_stack([r_span - rcv[:, 0], rcv[:, 1]])
    rev_rcv = np.column_stack([r_span - src[:, 0], src[:, 1]])
    rev = _multipath(mirrored, cfg, rev_src, rev_rcv)
    errs = []
    for i in range(src.shape[0]):
        a = np.sort(np.asarray(fwd["times"][i], dtype=float))
        b = np.sort(np.asarray(rev["times"][i], dtype=float))
        m = min(a.size, b.size)
        if m:
            errs.append(float(np.max(np.abs(a[:m] - b[:m]))))
    return {
        "n_pairs_checked": int(src.shape[0]),
        "max_abs_difference_s": float(np.max(errs)) if errs else float("nan"),
        "n_pairs_with_arrivals_both_ways": int(len(errs)),
    }


# ---------------------------------------------------------------------------
# Case runner
# ---------------------------------------------------------------------------
def run_case(cfg: CaseConfig | None = None, verbose: bool = False) -> dict:
    """Run the full under-ice tomography case and evaluate the criteria."""
    cfg = cfg or CaseConfig()
    rng = np.random.default_rng(cfg.seed)
    nodes = node_grid(cfg)
    field_truth = truth_field(cfg)
    field_bg = background_field(cfg)

    pos = instrument_positions(cfg)
    pairs = pair_indices(cfg)
    src = pos[pairs[:, 0]]
    rcv = pos[pairs[:, 1]]
    n_pairs = pairs.shape[0]

    # --- forward model: multipath Snell-law ray tracing under the ice -------
    # The profile has its sound-speed minimum at the surface, so rays are
    # trapped in the ice surface duct: each pair receives several
    # ice-reflected arrivals (bounce orders), which is exactly how under-ice
    # tomography is fed in practice.
    mp_truth = _multipath(field_truth, cfg, src, rcv)
    t_obs_pairs = mp_truth["times"]

    # reciprocity: the arrival pattern of i->j and j->i must coincide
    n_recip = min(int(cfg.n_reciprocity_pairs), n_pairs)
    recip = reciprocity_check(field_truth, cfg, src[:n_recip], rcv[:n_recip])
    reciprocity_max = recip["max_abs_difference_s"]

    # background-model arrivals: the tomographic basis (paths + predicted times)
    mp_bg = _multipath(field_bg, cfg, src, rcv)
    obs = _associate(
        mp_bg, mp_truth, tol=cfg.association_tol, theta_tol=cfg.association_theta_tol
    )
    if obs["n_pairs_used"] == 0:
        raise RuntimeError("no arrival could be associated between models")

    # one noise pool, drawn once: the same timing noise must be used by every
    # stage of the pipeline (the data are the data, they do not get re-drawn)
    rng_noise = np.random.default_rng(cfg.seed)
    noise_pool = rng_noise.normal(0.0, 1.0, size=8192)
    t_obs = obs["t_obs"] + cfg.sigma_time * noise_pool[: obs["t_obs"].size]

    # refraction diagnostic: traced arrival time vs straight-segment time
    t_straight = np.array(
        [
            straight_leg_time(field_truth, r0_, z0_, r1_, z1_)
            for r0_, z0_, r1_, z1_ in obs["geometry"]
        ]
    )
    refr_delta = obs["t_obs"] - t_straight

    nodes_paths = obs["paths"]
    s_bg = background_slowness(nodes)
    solve_kw = dict(
        n_iter=cfg.n_iter,
        relaxation=cfg.relaxation,
        smooth=cfg.smooth,
        method="sirt",
        target_residual=cfg.sigma_time,
        check_every=cfg.check_every,
    )
    stage1 = invert_sound_speed(nodes_paths, t_obs, nodes, s_bg, **solve_kw)

    # --- ray-bending iteration: re-trace in the retrieved field ------------
    # Reported as a diagnostic (residual and field change); the acceptance
    # metrics use stage 1, the pipeline that was fixed before looking at the
    # numbers.
    stages = [stage1]
    bending = []
    inv_bent = None
    obs_bent = None
    for _ in range(int(cfg.n_bending)):
        field_ret = grid_sound_speed_field(
            stages[-1]["sound_speed"], nodes.r_nodes, nodes.z_nodes, dr=25.0, dz=5.0
        )
        mp_ret = _multipath(field_ret, cfg, src, rcv)
        obs_ret = _associate(
            mp_ret, mp_truth, tol=cfg.association_tol, theta_tol=cfg.association_theta_tol
        )
        if obs_ret["n_pairs_used"] == 0:
            break
        t_bent = obs_ret["t_obs"] + cfg.sigma_time * noise_pool[: obs_ret["t_obs"].size]
        inv_bent = invert_sound_speed(obs_ret["paths"], t_bent, nodes, s_bg, **solve_kw)
        obs_bent = obs_ret
        stages.append(inv_bent)
        bending.append(
            {
                "n_observations": int(obs_ret["t_obs"].size),
                "rms_residual_before_s": float(stages[-2]["rms_residual_s"]),
                "rms_residual_after_s": float(inv_bent["rms_residual_s"]),
                "rms_field_change_m_s": float(
                    np.sqrt(np.mean((inv_bent["sound_speed"] - stages[-2]["sound_speed"]) ** 2))
                ),
            }
        )
    c_ret_grid = stage1["sound_speed"]

    # --- (a) sound-speed comparison at CTD samples --------------------------
    r_s, z_s = ctd_samples(cfg)
    c_true_s = truth_sound_speed(r_s, z_s, cfg)
    c_ret_s = interpolate_nodes(c_ret_grid, nodes, r_s, z_s)
    c_err = c_ret_s - c_true_s

    # --- (c) temperature inversion from the retrieved sound speed -----------
    t_bg_s, s_bg_s = background_state(z_s)  # salinity prior (CTD climatology)
    t_ret_s, n_clamped = temperature_from_sound_speed_clamped(c_ret_s, s_bg_s, z_s)
    t_true_s, _ = truth_state(r_s, z_s, cfg)
    t_err = t_ret_s - t_true_s

    # --- (b) ice thickness, draft-mode echo --------------------------------
    r_ice = ice_station_ranges(cfg)
    h_true = ice_thickness_truth(r_ice)
    draft_true = ice.draft_from_thickness(h_true)
    c_w_local = interpolate_nodes(c_ret_grid, nodes, r_ice, np.full_like(r_ice, cfg.ice_transducer_depth))
    c_w_true = truth_sound_speed(r_ice, np.full_like(r_ice, cfg.ice_transducer_depth), cfg)
    draft_meas = ice.measure_draft_echo(draft_true, c_w_true, cfg.sigma_echo, seed=cfg.seed)
    h_ret = ice.thickness_from_echo_time(draft_meas, c_w_local)
    h_err = h_ret - h_true

    # stress case: 0.5 ms timing noise (hull-borne sonar, rough underside)
    draft_meas_stress = ice.measure_draft_echo(
        draft_true, c_w_true, cfg.sigma_echo_stress, seed=cfg.seed + 1
    )
    h_ret_stress = ice.thickness_from_echo_time(draft_meas_stress, c_w_local)
    h_err_stress = h_ret_stress - h_true

    # --- uncertainties: ensemble spread -----------------------------------
    # the ensemble uses the *same* number of SIRT iterations as the primary
    # solve (a per-realisation discrepancy stop would let some members iterate
    # much further and inflate the spread by semi-convergence)
    mc_kw = dict(solve_kw)
    mc_kw["n_iter"] = int(stage1.get("n_iter", cfg.n_iter))
    mc_kw["target_residual"] = None
    mc = monte_carlo_uncertainty(
        nodes_paths,
        t_obs,
        nodes,
        s_bg,
        cfg.sigma_time,
        n_realisations=cfg.n_mc,
        seed=cfg.seed + 7,
        **mc_kw,
    )
    sigma_c_samples = interpolate_nodes(mc["sigma_sound_speed"], nodes, r_s, z_s)
    sigma_t_samples = temperature_uncertainty(sigma_c_samples, t_true_s)
    sigma_h = ice.thickness_uncertainty(
        draft_meas,
        cfg.sigma_echo,
        c_w_local,
        sigma_sound_speed=interpolate_nodes(mc["sigma_sound_speed"], nodes, r_ice, np.full_like(r_ice, cfg.ice_transducer_depth)),
        sigma_density_ratio=cfg.sigma_density_ratio,
    )

    # --- (d) negative controls --------------------------------------------
    nan_case = invert_sound_speed([], [], nodes, s_bg, **solve_kw)
    nc1 = {
        "no_data_all_nan": bool(np.all(np.isnan(nan_case["sound_speed"]))),
        "n_observations": int(nan_case["n_observations"]),
        "rms_residual_s": float(nan_case["rms_residual_s"]),
    }
    h_probe = np.array([2.0, 1.0, 0.1, 1.0e-2, 1.0e-3, 0.0])
    t_probe = ice.echo_time(ice.draft_from_thickness(h_probe), 1440.0)
    nc2 = {
        "thickness_probe_m": h_probe.tolist(),
        "echo_time_probe_s": t_probe.tolist(),
        "echo_time_zero": float(t_probe[-1]),
        "limit_zero": bool(abs(t_probe[-1]) == 0.0),
        "monotone_decreasing": bool(np.all(np.diff(t_probe) <= 0.0)),
        "pass": bool(abs(t_probe[-1]) == 0.0 and np.all(np.diff(t_probe) <= 0.0)),
    }
    c_zero = float(engineering_sound_speed(0.0, 0.0, 0.0))
    nc3 = {
        "sound_speed_at_0C_fresh": c_zero,
        "lower_bound": SOUND_SPEED_LOWER_BOUND,
        "pass": bool(c_zero >= SOUND_SPEED_LOWER_BOUND),
    }
    negative_controls = {
        "no_acoustic_data_is_nan": nc1,
        "ice_thickness_to_zero": nc2,
        "temperature_zero_sound_speed": nc3,
        "all_pass": bool(nc1["no_data_all_nan"] and nc2["pass"] and nc3["pass"]),
    }

    # --- (e) falsifiability: h = 2 m, campaign anchor 2.6 s ---------------
    # Primary reading (the campaign's own forward quantity): the source-receiver
    # travel time along the ice-underside duct over the declared baseline.
    h_ref = 2.0
    draft_ref = float(ice.draft_from_thickness(h_ref))
    mp_ref = _multipath(
        field_truth,
        cfg,
        np.array([[0.0, cfg.ref_depth]]),
        np.array([[cfg.ref_baseline, cfg.ref_depth]]),
    )
    t_ref_arrivals = np.asarray(mp_ref["times"][0], dtype=float)
    t_pred = float(t_ref_arrivals[0]) if t_ref_arrivals.size else float("nan")
    t_ref_path = mp_ref["eigenrays"][0][0] if mp_ref["eigenrays"][0] else None
    n_bounce_ref = int(t_ref_path.n_reflections) if t_ref_path is not None else -1
    deviation = t_pred - cfg.ref_time_measured
    # Secondary reading: a two-way specular reflection off the ice underside
    # over the same baseline (same geometry, doubled path).
    t_pred_two_way, r_bounce = specular_reflection_time(
        field_truth, 0.0, cfg.ref_depth, cfg.ref_baseline, cfg.ref_depth, z_interface=0.0
    )
    t_echo_ref = float(ice.echo_time(draft_ref, float(c_w_true.mean())))
    falsifiability = {
        "ice_thickness_m": h_ref,
        "primary_reading": "one-way source-receiver travel time along the ice duct",
        "predicted_travel_time_s": t_pred,
        "measured_anchor_s": cfg.ref_time_measured,
        "tolerance_s": cfg.ref_time_tolerance,
        "deviation_s": deviation,
        "deviation_ms": deviation * 1e3,
        "n_ice_reflections_on_path": n_bounce_ref,
        "baseline_m": cfg.ref_baseline,
        "transducer_depth_m": cfg.ref_depth,
        "secondary_two_way_reflection_s": t_pred_two_way,
        "secondary_deviation_ms": (t_pred_two_way - cfg.ref_time_measured) * 1e3,
        "secondary_bounce_range_m": r_bounce,
        "echo_time_for_2m_thickness_s": t_echo_ref,
        "anchor_sensitivity_ms_per_m_baseline": 1.0 / float(np.mean(truth_sound_speed(
            np.linspace(0.0, cfg.ref_baseline, 20), np.full(20, cfg.ref_depth), cfg))) * 1e3,
        "baseline_change_equivalent_to_tolerance_m": float(cfg.ref_time_tolerance)
        / (1.0 / float(np.mean(truth_sound_speed(
            np.linspace(0.0, cfg.ref_baseline, 20), np.full(20, cfg.ref_depth), cfg)))),
        "pass": bool(np.isfinite(deviation) and abs(deviation) <= cfg.ref_time_tolerance),
        "note": (
            "The 2.6 s anchor is reproduced by the ice-ducted source-receiver "
            "time over the declared 3.75 km baseline; the two-way specular "
            "reading over the same geometry is reported as well because the "
            "anchor is quoted to two significant figures and cannot separate "
            "the two readings (they differ by the width of the tolerance band). "
            "The 2 m thickness itself only enters the millisecond-scale echo time."
        ),
    }

    # --- criteria ---------------------------------------------------------
    n_ctd = int(c_err.size)
    n_ice = int(h_err.size)
    rms_c = float(np.sqrt(np.mean(c_err**2)))
    max_c = float(np.max(np.abs(c_err)))
    rms_t = float(np.sqrt(np.mean(t_err**2)))
    max_t = float(np.max(np.abs(t_err)))
    rms_h = float(np.sqrt(np.mean(h_err**2)))
    max_h = float(np.max(np.abs(h_err)))

    criteria = {
        "a_sound_speed": {
            "metric": "rms(c_recovered - c_truth) over CTD samples [m/s]",
            "value": rms_c,
            "max_value": max_c,
            "threshold": 5.0,
            "n_samples": n_ctd,
            "min_samples": 30,
            "pass": bool(rms_c < 5.0 and n_ctd >= 30),
        },
        "b_ice_thickness": {
            "metric": "rms(h_recovered - h_truth) [m]",
            "value": rms_h,
            "max_value": max_h,
            "threshold": 0.5,
            "n_samples": n_ice,
            "min_samples": 20,
            "pass": bool(rms_h < 0.5 and n_ice >= 20),
            "stress_case": {
                "sigma_echo_s": cfg.sigma_echo_stress,
                "rms_error_m": float(np.sqrt(np.mean(h_err_stress**2))),
                "max_error_m": float(np.max(np.abs(h_err_stress))),
                "note": "0.5 ms timing noise (rough underside / hull-borne sonar)",
            },
        },
        "c_temperature": {
            "metric": "rms(T_recovered - T_truth) [degC]",
            "value": rms_t,
            "max_value": max_t,
            "threshold": 0.5,
            "n_samples": int(np.isfinite(t_err).sum()),
            "min_samples": 30,
            "n_clamped_at_validity_edge": int(n_clamped),
            "pass": bool(rms_t < 0.5 and np.isfinite(t_err).sum() >= 30),
        },
        "d_negative_controls": negative_controls,
        "e_falsifiability": falsifiability,
    }
    criteria["all_pass"] = bool(
        criteria["a_sound_speed"]["pass"]
        and criteria["b_ice_thickness"]["pass"]
        and criteria["c_temperature"]["pass"]
        and negative_controls["all_pass"]
        and falsifiability["pass"]
    )

    result = {
        "config": {
            "r_max_m": cfg.r_max,
            "z_max_m": cfg.z_max,
            "instrument_depths_m": list(cfg.instrument_depths),
            "instrument_ranges_m": list(cfg.instrument_ranges),
            "n_instruments": int(pos.shape[0]),
            "n_pairs": int(n_pairs),
            "n_observations": int(t_obs.size),
            "sigma_time_s": cfg.sigma_time,
            "grid_shape": list(nodes.shape),
            "grid_r_nodes": nodes.r_nodes.tolist(),
            "grid_z_nodes": nodes.z_nodes.tolist(),
            "n_iter": cfg.n_iter,
            "smooth": cfg.smooth,
            "n_bending": cfg.n_bending,
        },
        "diagnostics": {
            "reciprocity_max_abs_s": reciprocity_max,
            "reciprocity_pairs_checked": int(recip["n_pairs_checked"]),
            "association_abs_dt_max_s": float(np.max(np.abs(obs["t_model"]))) if obs["t_obs"].size else float("nan"),
            "n_arrivals_background_model": int(obs["n_arrivals_model"]),
            "n_arrivals_truth_model": int(obs["n_arrivals_truth"]),
            "n_observations": int(t_obs.size),
            "n_pairs_used": int(obs["n_pairs_used"]),
            "bounce_orders": sorted({int(b) for b in obs["bounce"]}),
            "association_tolerance_s": cfg.association_tol,
            "refraction_time_delta_mean_ms": float(np.mean(np.abs(refr_delta)) * 1e3),
            "refraction_time_delta_max_ms": float(np.max(np.abs(refr_delta)) * 1e3),
            "sirt_iterations_stage1": int(stage1.get("n_iter", 0)),
            "sirt_iterations_bent": int(inv_bent.get("n_iter", 0)) if inv_bent is not None else 0,
            "rms_residual_stage1_s": float(stage1["rms_residual_s"]),
            "rms_residual_bent_s": (
                float(inv_bent["rms_residual_s"]) if inv_bent is not None else float("nan")
            ),
            "data_variance_reduction": float(stage1["data_variance_reduction"]),
            "path_length_min_m": float(np.min(stage1["path_lengths"])),
            "path_length_max_m": float(np.max(stage1["path_lengths"])),
            "t_obs_min_s": float(np.min(t_obs)),
            "t_obs_max_s": float(np.max(t_obs)),
            "bending": bending,
        },
        "fields": {
            "r_s": r_s,
            "z_s": z_s,
            "c_truth_samples": c_true_s,
            "c_retrieved_samples": c_ret_s,
            "c_error_samples": c_err,
            "t_truth_samples": t_true_s,
            "t_retrieved_samples": t_ret_s,
            "t_error_samples": t_err,
            "sigma_c_samples": sigma_c_samples,
            "sigma_t_samples": sigma_t_samples,
            "c_retrieved_grid": c_ret_grid,
            "r_ice": r_ice,
            "h_truth": h_true,
            "h_retrieved": h_ret,
            "h_error": h_err,
            "h_retrieved_stress": h_ret_stress,
            "h_error_stress": h_err_stress,
            "sigma_h": sigma_h,
        },
        "uncertainty": {
            "sigma_c_mean": float(np.nanmean(sigma_c_samples)),
            "sigma_c_max": float(np.nanmax(sigma_c_samples)),
            "sigma_T_mean": float(np.nanmean(sigma_t_samples)),
            "sigma_T_max": float(np.nanmax(sigma_t_samples)),
            "sigma_h_mean": float(np.nanmean(sigma_h)),
            "sigma_h_max": float(np.nanmax(sigma_h)),
            "n_mc": int(cfg.n_mc),
        },
        "criteria": criteria,
        "stages": {
            "rms_residual_s": [float(st["rms_residual_s"]) for st in stages],
            "n_observations": [int(st["n_observations"]) for st in stages],
            "rms_sound_speed_error_m_s": [
                float(np.sqrt(np.mean((interpolate_nodes(st["sound_speed"], nodes, r_s, z_s) - c_true_s) ** 2)))
                for st in stages
            ],
        },
        "observations": {
            "pair_id": obs["pair_id"],
            "bounce": obs["bounce"],
            "t_obs": t_obs,
            "t_model_background": obs["t_obs"],
            "geometry": obs["geometry"],
            "paths": obs["paths"],
        },
        "rays_truth": mp_truth,
        "src_rcv": np.column_stack([src[:, 0], src[:, 1], rcv[:, 0], rcv[:, 1]]),
        "field_truth": field_truth,
        "field_background": field_bg,
        "nodes": nodes,
    }
    if verbose:
        print(
            f"pairs={n_pairs} obs={t_obs.size} rms_c={rms_c:.3f} m/s "
            f"rms_T={rms_t:.3f} degC rms_h={rms_h:.3f} m all_pass={criteria['all_pass']}"
        )
    return result


def noise_sensitivity(result: dict, cfg: CaseConfig | None = None, noises=None) -> dict:
    """Re-invert the identified arrivals at several timing-noise levels.

    The physical signal ``t_obs - t_model`` of the case is reused and fresh
    Gaussian noise of the requested amplitude is added, so the scan costs one
    inversion per level instead of a full ray-tracing pass.  Returns the rms
    sound-speed and temperature errors at the CTD points.
    """
    cfg = cfg or CaseConfig()
    nodes = result["nodes"]
    s_bg = background_slowness(nodes)
    paths = result["observations"]["paths"]
    signal = result["observations"]["t_obs"] - result["observations"]["t_model_background"]
    r_s = result["fields"]["r_s"]
    z_s = result["fields"]["z_s"]
    t_true_s = result["fields"]["t_truth_samples"]
    _, s_bg_s = background_state(z_s)
    noises = np.asarray(noises if noises is not None else [0.2e-3, 0.5e-3, 1e-3, 2e-3, 5e-3], dtype=float)
    rng = np.random.default_rng(cfg.seed + 11)
    # a fixed iteration budget (the one the primary solve used) keeps the scan
    # a clean measurement of the noise sensitivity; a per-level discrepancy
    # stop would let the low-noise levels iterate into semi-convergence and the
    # curve would no longer be monotone in the noise level
    n_iter_fixed = int(result["diagnostics"].get("sirt_iterations_stage1", cfg.n_iter))
    solve_kw = dict(
        n_iter=n_iter_fixed,
        relaxation=cfg.relaxation,
        smooth=cfg.smooth,
        method="sirt",
        target_residual=None,
        check_every=cfg.check_every,
    )
    rms_c, rms_t, sig_c, sig_t = [], [], [], []
    for sigma in noises:
        t_obs = (
            result["observations"]["t_model_background"]
            + signal
            + rng.normal(0.0, float(sigma), size=signal.size)
        )
        inv = invert_sound_speed(paths, t_obs, nodes, s_bg, **solve_kw)
        c_ret = interpolate_nodes(inv["sound_speed"], nodes, r_s, z_s)
        rms_c.append(float(np.sqrt(np.mean((c_ret - result["fields"]["c_truth_samples"]) ** 2))))
        t_ret = invert_temperature_from_sound_speed(c_ret, s_bg_s, z_s)
        rms_t.append(float(np.sqrt(np.nanmean((t_ret - t_true_s) ** 2))))
        mc = monte_carlo_uncertainty(
            paths, t_obs, nodes, s_bg, float(sigma),
            n_realisations=max(int(cfg.n_mc // 4), 5),
            seed=cfg.seed + 23,
            n_iter=int(inv.get("n_iter", cfg.n_iter)), relaxation=cfg.relaxation,
            smooth=cfg.smooth, method="sirt", target_residual=None,
            check_every=cfg.check_every,
        )
        sig_c.append(float(np.nanmean(interpolate_nodes(mc["sigma_sound_speed"], nodes, r_s, z_s))))
        sig_t.append(
            float(np.nanmean(temperature_uncertainty(np.asarray(sig_c[-1]), np.nanmean(t_true_s))))
        )
    return {
        "sigma_time_ms": (noises * 1e3).tolist(),
        "rms_sound_speed_m_s": rms_c,
        "rms_temperature_degC": rms_t,
        "sigma_c_mean_m_s": sig_c,
        "sigma_T_mean_degC": sig_t,
    }


def summarise(result: dict) -> dict:
    """JSON-serialisable summary of :func:`run_case` (no numpy arrays)."""
    crit = result["criteria"]
    return {
        "config": result["config"],
        "diagnostics": result["diagnostics"],
        "uncertainty": result["uncertainty"],
        "stages": result["stages"],
        "criteria": {
            "a_sound_speed": crit["a_sound_speed"],
            "b_ice_thickness": crit["b_ice_thickness"],
            "c_temperature": crit["c_temperature"],
            "d_negative_controls": crit["d_negative_controls"],
            "e_falsifiability": crit["e_falsifiability"],
            "all_pass": crit["all_pass"],
        },
    }