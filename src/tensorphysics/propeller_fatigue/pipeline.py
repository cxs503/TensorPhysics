"""End-to-end W7-N07 pipeline: ice + hydrodynamic coupled fatigue life.

Run as ``python -m tensorphysics.propeller_fatigue.pipeline --out artifacts/m1135``.

Everything written to the artifact JSON is computed here; no result number is
transcribed by hand.  Each acceptance criterion is evaluated and stored with its
number, the reference it was compared against, and the pass/fail decision.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np

from .beam_impact import AnalyticImpulse, Beam, finite_difference_beam, numerical_impulse
from .blade_stress import BladeGeometry, root_stress_components
from .hydro_unsteady import (
    WakeField,
    blade_element_hydro,
    harmonic_amplitude,
    harmonic_amplitude_angle,
    kussner_gust_response,
    sears_gust_function,
    sears_root_moment,
)
from .ice_spectrum import ICE_CASES, IceEnvelope, blade_impact_train, ice_milling_torque, impact_spectrum
from .materials_sn import MATERIALS, BasquinSN, SNReference, goodman_amplitude
from .rainflow_miner import (
    aggregate_realisations,
    initiation_life,
    miner_damage,
    rainflow_count,
    safety_factor,
    turning_points,
)

# --------------------------------------------------------------------------- #
#  Reference propeller / ship operating point
# --------------------------------------------------------------------------- #
DESIGN = {
    "P_D_W": 6.0e6,          # delivered power [W]
    "n_rps": 130.0 / 60.0,   # design shaft rate [rev/s]
    "J": 0.60,               # advance coefficient at design
    "eta_0": 0.55,           # open-water efficiency at design
    "n_rps_op2": 110.0 / 60.0,
    "J_op2": 0.45,
    "rho_water": 1025.0,
    "added_mass_flat": 1.40,  # flatwise hydrodynamic added mass factor
    "added_mass_edge": 1.25,  # edgewise hydrodynamic added mass factor
    "design_life_years": 20.0,
    "ice_hours_per_year": 4000.0,
    "n_revolutions_window": 200,
    "n_realisations": 4,
    "dt": 5.0e-4,
}


def design_point(D: float, n_rps: float, J: float) -> Dict[str, float]:
    Va = J * n_rps * D
    T = DESIGN["P_D_W"] * DESIGN["eta_0"] / Va
    Q = DESIGN["P_D_W"] / (2 * np.pi * n_rps)
    K_T = T / (DESIGN["rho_water"] * n_rps ** 2 * D ** 4)
    K_Q = Q / (DESIGN["rho_water"] * n_rps ** 2 * D ** 5)
    return {"Va_m_s": Va, "T_N": T, "Q_Nm": Q, "K_T": K_T, "K_Q": K_Q,
            "J": J, "n_rps": n_rps, "eta_open_water": DESIGN["eta_0"]}


# --------------------------------------------------------------------------- #
#  Criterion (a): single-cycle impact stress vs the beam impact closed form
# --------------------------------------------------------------------------- #
def beam_reference_case(geom: BladeGeometry, mat, env: IceEnvelope,
                        Q_max: float) -> Dict[str, object]:
    """Single-cycle ice impact stress: closed form vs numerical vs FD.

    The load is the axial component of the ice impact at the envelope maximum
    torque, applied as a half-sine pulse at the ice contact radius.
    """
    sec = geom.equivalent_beam_section(0.5, mat.rho)
    L = geom.span
    I = sec["I"]
    c_half = sec["c_half"]
    m_wet = sec["mass_per_length"] * DESIGN["added_mass_flat"]
    tau = env.impact_duration
    r_imp = env.r_impact_hat * geom.R
    F_t = Q_max / r_imp
    F_a = env.k_axial * F_t
    x_mid = L / 2.0
    dur_ss = max(6.0 * 2 * np.pi / (np.pi ** 2 * np.sqrt(mat.E * I / m_wet) / L ** 2), 3 * tau)

    # ---- anchor: simply supported, load and observation at midspan
    beam_ss = Beam(L, mat.E, I, m_wet, bc="simply_supported", n_modes=40,
                   outer_fibre_distance=c_half)
    sigma_ref_ss = AnalyticImpulse(beam_ss, F_a, tau, x_mid, x_mid).peak_stress(dur_ss)
    truncation = {}
    for n in (5, 10, 20, 40, 80):
        bb = Beam(L, mat.E, I, m_wet, bc="simply_supported", n_modes=n,
                  outer_fibre_distance=c_half)
        truncation[n] = AnalyticImpulse(bb, F_a, tau, x_mid, x_mid).peak_stress(dur_ss)

    # independent route 1: RK4 state-space integration (same truncation, 20 modes)
    beam_ss20 = Beam(L, mat.E, I, m_wet, bc="simply_supported", n_modes=20,
                     outer_fibre_distance=c_half)
    an_ss20 = AnalyticImpulse(beam_ss20, F_a, tau, x_mid, x_mid).peak_stress(dur_ss)
    _, sig_rk4 = numerical_impulse(beam_ss20, F_a, tau, x_load=x_mid, x_obs=x_mid,
                                   duration=dur_ss)
    sigma_rk4_ss = float(sig_rk4.max())

    # independent route 2: explicit finite differences of the beam PDE
    _, sig_fd = finite_difference_beam(L, mat.E, I, m_wet, F_a, tau, c_half,
                                      n_nodes=61, duration=dur_ss)
    sigma_fd_ss = float(sig_fd.max())

    # ---- in-service configuration: cantilever blade, load at the contact radius
    beam_c = Beam(L, mat.E, I, m_wet, bc="cantilever", n_modes=40,
                  outer_fibre_distance=c_half)
    x_load = r_imp - geom.r_hub
    dur_c = max(5.0 * beam_c.fundamental_period, 3 * tau)
    an_c = AnalyticImpulse(beam_c, F_a, tau, x_load, 0.0)
    sigma_analytic_c = an_c.peak_stress(dur_c)
    beam_c20 = Beam(L, mat.E, I, m_wet, bc="cantilever", n_modes=20,
                    outer_fibre_distance=c_half)
    an_c20 = AnalyticImpulse(beam_c20, F_a, tau, x_load, 0.0).peak_stress(dur_c)
    _, sig_rk4_c = numerical_impulse(beam_c20, F_a, tau, x_load=x_load, x_obs=0.0,
                                     duration=dur_c)
    sigma_rk4_c = float(sig_rk4_c.max())

    beam_1 = Beam(L, mat.E, I, m_wet, bc="simply_supported", n_modes=1,
                  outer_fibre_distance=c_half)
    dlf1 = AnalyticImpulse(beam_1, F_a, tau, x_mid, x_mid).dynamic_load_factor(dur_ss)
    dlf_ref = sigma_ref_ss / beam_ss.static_stress(0.0, F_a, x_mid, x_mid)

    err_rk4 = abs(sigma_rk4_ss - an_ss20) / an_ss20
    err_fd = abs(sigma_fd_ss - sigma_ref_ss) / sigma_ref_ss
    err_rk4_c = abs(sigma_rk4_c - an_c20) / an_c20
    return {
        "load_axial_N": F_a,
        "load_tangential_N": F_t,
        "pulse_duration_s": tau,
        "equivalent_I_m4": I,
        "mass_per_length_wet_kg_m": m_wet,
        "outer_fibre_distance_m": c_half,
        "f1_simply_supported_Hz": beam_ss.fundamental_frequency,
        "T1_simply_supported_s": beam_ss.fundamental_period,
        "tau_over_T1": tau / beam_ss.fundamental_period,
        "f1_cantilever_Hz": beam_c.fundamental_frequency,
        "T1_cantilever_s": beam_c.fundamental_period,
        "tau_over_T1_cantilever": tau / beam_c.fundamental_period,
        "sigma_analytic_ss_Pa": sigma_ref_ss,
        "sigma_analytic_ss_20modes_Pa": an_ss20,
        "sigma_rk4_ss_Pa": sigma_rk4_ss,
        "sigma_fd_ss_Pa": sigma_fd_ss,
        "rel_err_rk4_ss": float(err_rk4),
        "rel_err_fd_ss": float(err_fd),
        "truncation_study_ss_Pa": {str(k): v for k, v in truncation.items()},
        "truncation_rel_change_20_to_40": float(abs(truncation[20] - truncation[40]) / truncation[40]),
        "sigma_analytic_cantilever_Pa": sigma_analytic_c,
        "sigma_rk4_cantilever_Pa": sigma_rk4_c,
        "rel_err_rk4_cantilever": float(err_rk4_c),
        "dlf_multimode_ss": float(dlf_ref),
        "dlf_sdof_ss": float(dlf1),
        "dlf_sdof_error": float(abs(dlf1 - dlf_ref) / dlf_ref),
        "dlf_cantilever": float(sigma_analytic_c / beam_c.static_stress(0.0, F_a, x_load, 0.0)),
        "static_stress_cantilever_Pa": float(beam_c.static_stress(0.0, F_a, x_load, 0.0)),
        "criterion_a_pass": bool(max(err_rk4, err_fd) <= 0.05),
    }


def dlf_sweep(geom: BladeGeometry, mat, env: IceEnvelope, Q_max: float) -> Dict[str, object]:
    """Dynamic load factor vs pulse duration: multimode, SDOF and FD."""
    sec = geom.equivalent_beam_section(0.5, mat.rho)
    L = geom.span
    m_wet = sec["mass_per_length"] * DESIGN["added_mass_flat"]
    r_imp = env.r_impact_hat * geom.R
    F_a = env.k_axial * Q_max / r_imp
    ratios = np.array([0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0])
    beam_ss = Beam(L, mat.E, sec["I"], m_wet, bc="simply_supported", n_modes=40,
                   outer_fibre_distance=sec["c_half"])
    beam_1 = Beam(L, mat.E, sec["I"], m_wet, bc="simply_supported", n_modes=1,
                  outer_fibre_distance=sec["c_half"])
    T1 = beam_ss.fundamental_period
    x_mid = L / 2
    multi, sdof = [], []
    for r in ratios:
        tau = r * T1
        multi.append(AnalyticImpulse(beam_ss, F_a, tau, x_mid, x_mid).dynamic_load_factor())
        sdof.append(AnalyticImpulse(beam_1, F_a, tau, x_mid, x_mid).dynamic_load_factor())
    fd = []
    for r in (0.2, 0.5, 1.0):
        tau = r * T1
        _, sig = finite_difference_beam(L, mat.E, sec["I"], m_wet, F_a, tau,
                                        sec["c_half"], n_nodes=41, cfl=0.8)
        static = beam_ss.static_stress(0.0, F_a, x_mid, x_mid)
        fd.append((float(r), float(sig.max() / static)))
    return {
        "tau_over_T1": ratios.tolist(),
        "dlf_multimode": [float(v) for v in multi],
        "dlf_sdof": [float(v) for v in sdof],
        "fd_points": fd,
        "T1_s": T1,
    }


# --------------------------------------------------------------------------- #
#  Criterion (b): hydrodynamic alternating stress vs analytic unsteady theory
# --------------------------------------------------------------------------- #
def hydro_comparison(geom: BladeGeometry, wake: WakeField,
                     span_factor: float = 1.0) -> Dict[str, object]:
    """Pipeline unsteady BEMT vs the analytic 2D thin-airfoil reference.

    ``span_factor = 1`` keeps the blade element two-dimensional so that the
    comparison isolates the unsteady-response modelling (the 3D lift reduction
    used in the fatigue chain is reported separately).
    """
    st = geom.stations()
    st.span_factor = span_factor
    root = geom.root_properties()
    ops = []
    for key, n_rps, J in (("OP1_design", DESIGN["n_rps"], DESIGN["J"]),
                          ("OP2_light", DESIGN["n_rps_op2"], DESIGN["J_op2"])):
        dp = design_point(geom.D, n_rps, J)
        V = dp["Va_m_s"]
        qs = blade_element_hydro(st, V, n_rps, wake, unsteady=False)
        un = blade_element_hydro(st, V, n_rps, wake, unsteady=True)
        k_mean = float(np.mean(un["reduced_frequency"]["k_stations"]))
        gain = abs(sears_gust_function(k_mean))
        worst = int(qs["worst_blade"])
        amp_qs_1p = abs(harmonic_amplitude_angle(qs["moment"][worst], qs["theta"], 1))
        # pipeline route: station-wise unsteady correction inside the BEM
        amp_bem_1p = abs(harmonic_amplitude_angle(un["moment"][worst], un["theta"], 1))
        ref = sears_root_moment(st, V, n_rps, wake, harmonic=1)
        amp_ref = ref["moment_amplitude"]
        sigma_a_pipeline = amp_bem_1p / root["W_flat_m3"]
        sigma_a_reference = amp_ref / root["W_flat_m3"]
        sigma_a_qs = amp_qs_1p / root["W_flat_m3"]
        err = abs(sigma_a_pipeline - sigma_a_reference) / sigma_a_reference
        ops.append({
            "op": key,
            "n_rps": n_rps,
            "J": J,
            "Va_m_s": V,
            "reduced_frequency_k_mean": k_mean,
            "sears_gain_1P": gain,
            "moment_1p_quasisteady_Nm": float(amp_qs_1p),
            "moment_1p_pipeline_Nm": float(amp_bem_1p),
            "moment_1p_sears_reference_Nm": float(amp_ref),
            "sigma_a_pipeline_Pa": float(sigma_a_pipeline),
            "sigma_a_reference_Pa": float(sigma_a_reference),
            "sigma_a_quasisteady_Pa": float(sigma_a_qs),
            "rel_err": float(err),
            "rel_err_if_quasisteady": float(abs(sigma_a_qs - sigma_a_reference) / sigma_a_reference),
            "moment_amplitude_all_harmonics_Nm": float(un["moment_amplitude"][worst]),
            "pass": bool(err <= 0.15),
        })
    return {"ops": ops, "criterion_b_pass": bool(all(o["pass"] for o in ops)),
            "max_rel_err": float(max(o["rel_err"] for o in ops))}


def sears_verification() -> Dict[str, object]:
    """Frequency-domain Sears vs time-domain Kuessner gust response."""
    k_list = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0]
    V, c = 20.0, 1.0
    rows = []
    for k in k_list:
        f = k * V / (np.pi * c)          # k = pi f c / V
        t = np.arange(0, 40 * (1 / f), 1 / (f * 400))
        up = np.cos(2 * np.pi * f * t)
        n_settle = int(0.6 * t.size)
        CL = kussner_gust_response(t, up, V, c)
        amp_td = abs(harmonic_amplitude(CL[n_settle:], t[n_settle:], f)) * V
        amp_fd = abs(sears_gust_function(k))
        rows.append({"k": k, "f_Hz": f, "abs_S_freq": amp_fd, "abs_S_timedomain": float(amp_td),
                     "rel_err": float(abs(amp_td - amp_fd) / amp_fd)})
    valid = [r for r in rows if 0.1 <= r["k"] <= 0.8]
    return {"rows": rows,
            "validated_k_range": [0.1, 0.8],
            "max_rel_err": float(max(r["rel_err"] for r in valid)),
            "max_rel_err_all_k": float(max(r["rel_err"] for r in rows)),
            "pass": bool(max(r["rel_err"] for r in valid) <= 0.15),
            "note": "the two-exponential Kuessner approximation is used; its "
                    "accuracy degrades for k < 0.1 and k > 0.8 (see rows)"}


# --------------------------------------------------------------------------- #
#  Combined stress history and Miner damage
# --------------------------------------------------------------------------- #
def harmonic_moment_history(moment_1rev: np.ndarray, theta: np.ndarray, t: np.ndarray,
                            phase: np.ndarray, n_harmonics: int = 3) -> np.ndarray:
    """Smooth periodic reconstruction of a per-revolution moment history."""
    out = np.full_like(t, float(np.mean(moment_1rev)))
    for kk in range(1, n_harmonics + 1):
        amp = harmonic_amplitude_angle(moment_1rev, theta, kk)
        out = out + amp.real * np.cos(kk * phase) - amp.imag * np.sin(kk * phase)
    return out


def build_stress_history(geom: BladeGeometry, mat, env: IceEnvelope, wake: WakeField,
                         dp: Dict[str, float], n_real: int = 4,
                         dlf: float = 1.0,
                         n_revolutions: float = 200.0) -> Dict[str, object]:
    """Window stress history for the most heavily loaded blade and Miner damage."""
    st = geom.stations()
    root = geom.root_properties()
    n_rps = dp["n_rps"]
    V = dp["Va_m_s"]
    rpm = ice_milling_torque(geom, env, n_rps)
    spec = impact_spectrum(env, rpm["Q_mill_Nm"], dp["Q_Nm"], seed=11)
    hydro = blade_element_hydro(st, V, n_rps, wake, unsteady=True)
    worst = int(np.argmax(hydro["moment_amplitude"]))

    # steady (mean) stress: thrust + design torque + centrifugal
    r_imp = env.r_impact_hat * geom.R
    T_per_blade = dp["T_N"] / geom.Z
    M_thrust = T_per_blade * (0.7 * geom.R - geom.r_hub)      # centre of pressure ~0.7R
    M_torque = (dp["Q_Nm"] / geom.Z) / (0.7 * geom.R) * (0.7 * geom.R - geom.r_hub)
    steady = root_stress_components(geom, mat, n_rps, M_axial=M_thrust, M_tangential=M_torque)
    sigma_mean = steady["sigma_combined_outer_fibre"]

    T1 = n_revolutions / n_rps
    duration = T1
    hours_window = duration / 3600.0
    n_steps = int(np.ceil(duration / DESIGN["dt"]))
    t = np.arange(n_steps) * DESIGN["dt"]
    omega = 2 * np.pi * n_rps

    # hydrodynamic moment for the worst blade: smooth Fourier synthesis of the
    # 1P..3P content (avoids interpolation stair-steps that would add spurious
    # micro-cycles to the rainflow count)
    m_hydro = harmonic_moment_history(hydro["moment"][worst], hydro["theta"], t,
                                      omega * t + 2 * np.pi * worst / geom.Z)
    sigma_hydro = m_hydro / root["W_flat_m3"] - steady["sigma_axial_bending"]

    sn = BasquinSN.fit(SNReference("NAB"))
    damages = []
    last = None
    for r in range(n_real):
        train = blade_impact_train(geom, env, n_rps, duration, spec,
                                  dt=DESIGN["dt"], seed=100 + r)
        lever = train["r_impact_m"] - geom.r_hub
        sigma_ice = dlf * (train["force_axial"][worst] * lever / root["W_flat_m3"]
                           + train["force_tangential"][worst] * lever / root["W_edge_m3"])
        sigma = sigma_mean + sigma_hydro + sigma_ice
        cyc = rainflow_count(sigma, close_history=True)
        dmg = miner_damage(cyc, sn, mat.sigma_uts)
        damages.append(dmg["D"])
        last = {"sigma": sigma, "cycles": cyc, "damage": dmg, "train": train,
                "sigma_ice": sigma_ice}
    D_mean, D_sem = aggregate_realisations(damages)
    scale = DESIGN["design_life_years"] * DESIGN["ice_hours_per_year"] / hours_window
    dmg_life = miner_damage(last["cycles"], sn, mat.sigma_uts, scale=scale)

    # ---- statistically exact Miner integration (no Monte-Carlo tail noise)
    exposure_hours = DESIGN["design_life_years"] * DESIGN["ice_hours_per_year"]
    n_cycles_life = exposure_hours * 3600.0 * n_rps
    n_impacts_life = exposure_hours * env.impacts_per_hour
    hydro_sigma_a = float(0.5 * (sigma_hydro.max() - sigma_hydro.min()))
    d_hydro = 1.0 / sn.cycles(goodman_amplitude(hydro_sigma_a, sigma_mean, mat.sigma_uts)) \
        * n_cycles_life
    z = np.linspace(1.0e-9, 1.0 - 1.0e-9, 4000)
    from scipy.special import ndtri
    Q = spec["Q_median_Nm"] * np.exp(spec["sigma_ln"] * ndtri(z))
    Q = np.minimum(Q, 1.05 * spec["Q_max_envelope_Nm"])
    dsigma = dlf * (env.k_axial * Q / train["r_impact_m"] * lever / root["W_flat_m3"]
                    + Q / train["r_impact_m"] * lever / root["W_edge_m3"])
    s_peak = sigma_mean + dsigma
    s_amp = 0.5 * (s_peak - sigma_mean)
    s_mean = 0.5 * (s_peak + sigma_mean)
    N_imp = sn.cycles(goodman_amplitude(s_amp, s_mean, mat.sigma_uts))
    # expectation over the continuous log-normal spectrum (weights = quantiles)
    d_ice = float(n_impacts_life * np.mean(1.0 / N_imp))
    D_exact = float(d_hydro + d_ice)
    life_exact = exposure_hours / D_exact if D_exact > 0 else float("inf")
    return {
        "ice_case": env.key,
        "Q_mill_Nm": rpm["Q_mill_Nm"],
        "depth_of_cut_m": rpm["depth_of_cut_m"],
        "Q_design_Nm": dp["Q_Nm"],
        "Q_mill_over_Qdesign": rpm["Q_mill_Nm"] / dp["Q_Nm"],
        "Q_max_envelope_Nm": spec["Q_max_envelope_Nm"],
        "k_ice_envelope": spec["k_ice_envelope"],
        "impacts_per_hour": env.impacts_per_hour,
        "sigma_mean_Pa": sigma_mean,
        "sigma_hydro_amp_Pa": float(0.5 * (sigma_hydro.max() - sigma_hydro.min())),
        "sigma_ice_max_Pa": float(last["sigma_ice"].max()),
        "sigma_total_max_Pa": float(last["sigma"].max()),
        "sigma_total_min_Pa": float(last["sigma"].min()),
        "D_window": D_mean,
        "D_window_sem": D_sem,
        "D_life": dmg_life["D"],
        "n_cycles_window": dmg_life["n_cycles"] / scale,
        "n_cycles_life": dmg_life["n_cycles"],
        "life_hours": initiation_life(D_mean, hours_window),
        "SF": safety_factor(D_mean),
        "SF_life": safety_factor(dmg_life["D"]),
        "D_hydro_exact": float(d_hydro),
        "D_ice_exact": float(d_ice),
        "D_exact": D_exact,
        "life_hours_exact": life_exact,
        "SF_exact": safety_factor(D_exact),
        "hydro_sigma_a_Pa": hydro_sigma_a,
        "dlf_applied": dlf,
        "hours_window": hours_window,
        "exposure_hours": DESIGN["design_life_years"] * DESIGN["ice_hours_per_year"],
        "damage_bins": dmg_life["bins"],
        "sigma_a_eq_Pa": dmg_life["sigma_a_eq"],
        "damage_top_bin_fraction": dmg_life["damage_fraction_top_bin"],
        "n_impacts_window": int(last["train"]["n_impacts"]),
        "_hist": {"t": t, "sigma": last["sigma"], "sigma_hydro": sigma_hydro,
                  "sigma_ice": last["sigma_ice"]},
        "_cycles": last["cycles"],
    }


# --------------------------------------------------------------------------- #
#  Design margins: stress-magnification capacity and resonance check
# --------------------------------------------------------------------------- #
def margin_analysis(cases: Dict[str, object], geom: BladeGeometry, mat,
                    dp: Dict[str, float], env: IceEnvelope,
                    sweep: Dict[str, object]) -> Dict[str, object]:
    """Turn Miner damage into design statements.

    * ``kappa_capacity`` = ``SF^(1/m)`` - the stress magnification (stress
      concentration x marine/defect knock-down) the blade tolerates before
      ``D = 1`` at the 20-year exposure.
    * ``SF`` for two engineering penalty levels: ``kappa = 2`` (root fillet only)
      and ``kappa = 5`` (fillet x seawater/casting knock-down).
    * resonance check: amplification of the blade-rate harmonics near the
      in-water first bending frequency (``zeta = 0.02``).
    """
    sn = BasquinSN.fit(SNReference("NAB"))
    m = 1.0 / abs(sn.b)
    zeta = 0.02
    f1 = None
    out = {"exponent_m": m, "zeta": zeta, "cases": {}}
    # in-water first flatwise bending frequency of the cantilever blade
    sec = geom.equivalent_beam_section(0.5, mat.rho)
    m_wet = sec["mass_per_length"] * DESIGN["added_mass_flat"]
    beam = Beam(geom.span, mat.E, sec["I"], m_wet, bc="cantilever",
                n_modes=4, outer_fibre_distance=sec["c_half"])
    f1 = beam.fundamental_frequency
    n_rps = dp["n_rps"]
    f_blade_rate = geom.Z * n_rps
    harmonic_rows = []
    for kk in range(1, 5):
        f = kk * f_blade_rate
        ratio = f / f1
        H = 1.0 / abs(1.0 - ratio ** 2 + 2j * zeta * ratio)
        harmonic_rows.append({"harmonic": kk, "f_Hz": f, "f_over_f1": ratio,
                              "amplification": H, "damage_amplification": H ** m})
    out["f1_in_water_Hz"] = f1
    out["shaft_rate_Hz"] = n_rps
    out["blade_rate_Hz"] = f_blade_rate
    out["f1_over_blade_rate"] = f1 / f_blade_rate
    out["harmonic_amplification"] = harmonic_rows
    for key, c in cases.items():
        sf = c["SF_exact"]
        kappa = sf ** (1.0 / m)
        out["cases"][key] = {
            "SF_nominal": sf,
            "D_life": c["D_exact"],
            "life_hours": c["life_hours_exact"],
            "sigma_a_eq_Pa": c["sigma_a_eq_Pa"],
            "kappa_capacity": kappa,
            "SF_kappa2": sf / 2.0 ** m,
            "SF_kappa5": sf / 5.0 ** m,
            "life_hours_kappa5": c["life_hours_exact"] / 5.0 ** m,
        }
    return out


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def run(out_dir: str = "artifacts/m1135", make_figures: bool = True) -> Dict[str, object]:
    os.makedirs(out_dir, exist_ok=True)
    geom = BladeGeometry()
    mat = MATERIALS["NAB"]
    wake = WakeField()
    dp = design_point(geom.D, DESIGN["n_rps"], DESIGN["J"])
    dp2 = design_point(geom.D, DESIGN["n_rps_op2"], DESIGN["J_op2"])

    # ---------- geometry / data sanity
    blade_area = geom.planform_area()
    disc_area = np.pi * geom.R ** 2
    ear = geom.expanded_area_ratio()
    geometry_checks = {
        "D_m": geom.D,
        "R_m": geom.R,
        "r_hub_over_R": geom.r_hub / geom.R,
        "expanded_area_ratio": ear,
        "blade_planform_area_m2": blade_area,
        "disc_area_m2": disc_area,
        "thickness_ratio_root": float(geom.thickness(geom.root_radius()) / geom.chord(geom.root_radius())),
        "chord_monotonic_decreasing": bool(np.all(np.diff(geom.chord(geom.stations().r)) < 0)),
        "EAR_in_range_0.30_0.90": bool(0.30 <= ear <= 0.90),
    }

    # ---------- criterion (d): S-N fit
    sn_report = {}
    for key in ("NAB", "AL6082"):
        ref = SNReference(key)
        sn = BasquinSN.fit(ref)
        err = sn.fit_error(ref)
        sn_report[key] = {
            "fit": sn.as_dict(),
            "max_abs_rel_err": err["max_abs_rel"],
            "rms_rel_err": err["rms_rel"],
            "rel_err_per_point": err["residuals"],
            "reference_points": ref.points.tolist(),
            "provenance": ref.provenance,
            "pass": bool(err["max_abs_rel"] <= 0.10),
        }
    criterion_d = all(v["pass"] for v in sn_report.values())

    # ---------- criterion (a): beam impact anchor
    env_design = ICE_CASES["design"]
    Q_max_design = env_design.k_torque_max * dp["Q_Nm"]
    beam = beam_reference_case(geom, mat, env_design, Q_max_design)
    sweep = dlf_sweep(geom, mat, env_design, Q_max_design)

    # ---------- criterion (b): hydrodynamic alternating stress
    hyd = hydro_comparison(geom, wake)
    sears = sears_verification()

    # ---------- (c)(e): Miner damage per ice case
    cases = {}
    for key in ("light", "design", "heavy"):
        env = ICE_CASES[key]
        cases[key] = build_stress_history(geom, mat, env, wake, dp,
                                          n_real=DESIGN["n_realisations"],
                                          dlf=beam["dlf_cantilever"])
    # damage monotonicity in ice severity (physical direction gate for (c))
    keys = ["light", "design", "heavy"]
    D_seq = [cases[k]["D_window"] for k in keys]
    D_exact_seq = [cases[k]["D_exact"] for k in keys]
    monotone = bool(np.all(np.diff(D_seq) > 0) and np.all(np.diff(D_exact_seq) > 0))
    finite_life = all(np.isfinite(cases[k]["life_hours_exact"])
                      and cases[k]["life_hours_exact"] > 0 for k in keys)
    life_order = bool(np.all(np.diff([cases[k]["life_hours_exact"] for k in keys]) < 0))
    criterion_c = bool(monotone and finite_life and life_order)
    criterion_e = bool(cases["design"]["SF_exact"] >= 1.0)

    margins = margin_analysis(cases, geom, mat, dp, env_design, sweep)

    criteria = {
        "a_impact_stress_vs_beam_analytic": {
            "value": max(beam["rel_err_rk4_ss"], beam["rel_err_fd_ss"]),
            "threshold": 0.05, "pass": beam["criterion_a_pass"],
            "detail": "max rel. error of RK4-modal and FD routes vs closed-form "
                      "simply-supported beam impulse solution",
        },
        "b_hydro_alternating_stress": {
            "value": hyd["max_rel_err"], "threshold": 0.15,
            "pass": hyd["criterion_b_pass"],
            "detail": "pipeline unsteady-BEMT vs analytic Sears gust reference, 2 OPs",
        },
        "c_miner_crack_initiation": {
            "value": {"D_window": D_seq, "D_exact": D_exact_seq,
                      "life_hours_exact": [cases[k]["life_hours_exact"] for k in keys]},
            "threshold": "D monotone in severity, finite life at D=1",
            "pass": criterion_c,
        },
        "d_sn_fit": {
            "value": {k: v["max_abs_rel_err"] for k, v in sn_report.items()},
            "threshold": 0.10, "pass": criterion_d,
        },
        "e_fatigue_safety_factor": {
            "value": cases["design"]["SF_exact"], "threshold": 1.0,
            "pass": criterion_e,
            "detail": "SF = N_allowable/N_actual = 1/D at the 20-year design exposure",
        },
    }

    result = {
        "mission": "W7-N07",
        "title": "Propeller blade ice-hydrodynamic coupled fatigue life",
        "design_point": dp,
        "design_point_op2": dp2,
        "geometry_checks": geometry_checks,
        "root_section": geom.root_properties(),
        "equivalent_beam": geom.equivalent_beam_section(0.5, mat.rho),
        "material_key": "NAB",
        "materials": {k: {"E": m.E, "rho": m.rho, "sigma_uts": m.sigma_uts,
                          "sigma_y": m.sigma_y} for k, m in MATERIALS.items()},
        "sn_fits": sn_report,
        "beam_impact": beam,
        "dlf_sweep": sweep,
        "hydro": hyd,
        "sears_verification": sears,
        "ice_cases": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                      for k, v in cases.items()},
        "margins": margins,
        "criteria": criteria,
        "all_criteria_pass": bool(all(c["pass"] for c in criteria.values())),
        "model_notes": {
            "ice_envelope": "Q_ice,max = k_ice * Q_design (ice-class rule structure); "
                            "the W7-N06 torque envelope artifact is not present in this "
                            "work-tree, so the envelope is rebuilt here from the milling "
                            "crushing model documented in ice_spectrum.py",
            "hydro_reference": "analytic unsteady thin-airfoil (Sears + Kuessner); no "
                               "RANS/LES reference available in this work-tree",
            "added_mass": "flatwise 1.40, edgewise 1.25 (engineering values)",
            "damage_scale": "linear Miner extrapolation of a 200-revolution window",
        },
    }

    with open(os.path.join(out_dir, "w7n07_results.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    with open(os.path.join(out_dir, "w7n07_criteria.json"), "w") as fh:
        json.dump(criteria, fh, indent=2)

    if make_figures:
        from .figures import make_all_figures
        figs = make_all_figures(out_dir, result, cases, geom, mat, env_design, wake, dp)
        result["figures"] = figs
        with open(os.path.join(out_dir, "w7n07_results.json"), "w") as fh:
            json.dump(result, fh, indent=2)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/m1135")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    res = run(args.out, make_figures=not args.no_figures)
    print(json.dumps(res["criteria"], indent=2))
    print("all pass:", res["all_criteria_pass"])


if __name__ == "__main__":
    main()