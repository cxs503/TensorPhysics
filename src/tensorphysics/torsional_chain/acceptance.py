"""Machine-checkable acceptance computations C1..C6 for the torsional chain node.

Each function returns a plain ``dict`` with the numbers, the reference value they
are compared with, the relative error and a boolean ``pass`` flag, so the report
JSON can be generated without hand-copied numbers (the runner script serialises
exactly what these functions return).

Criteria
--------
C1  natural frequencies of 3 analytic configurations, error <= 3 %
C2  gear parametric response: quasi-static amplitude and peak position in the
    mesh cycle, amplitude error <= 5 %, peak-position error <= 5 % of a mesh period
C3  PMSM steady speed for 3 load cases, error <= 2 %
C4  torque-to-stress transfer function vs closed form, peak position <= 3 %
C5  sub-cycled interface work residual <= 2 % at N = 2 / 4 / 8
C6  IACS Polar Class stress + Miner fatigue life for 3 polar classes
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np

from .drivetrain import DrivetrainChain, ShaftSegment, standard_chain, uniform_segment_stiffness
from .excitation import IceMillingTorque
from .gear import MeshStiffness, ParametricMeshOscillator, mathieu_growth_rate
from .iacs import iacs_torsion_check
from .motor import PMSM, simulate_motor
from .theory import (sdof_amplitude, sdof_peak_frequency, three_inertia_chain_frequencies,
                     two_inertia_fixed_frequency, two_inertia_free_frequency)
from .timestep import (AdaptiveTimeStep, SolverRate, TimeStepCoordinator, cfl_step,
                       circuit_step, exchange_step, exchange_work_residual, mesh_step,
                       structural_accuracy_step)

__all__ = ["c1_natural_frequencies", "c2_gear_parametric", "c3_motor_steady_state",
           "c4_ice_torque_transfer", "c5_time_step_coordination", "c6_iacs_check",
           "run_acceptance", "torsional_reference_chain"]

G_SHEAR = 79.3e9


def torsional_reference_chain(inertia: float = 1.2e3, damping_ratio: float = 0.05,
                              length: float = 6.0, outer_diameter: float = 0.42,
                              inner_diameter: float = 0.18) -> Dict[str, object]:
    """Single-mode reference chain: propeller inertia grounded through one shaft.

    Closed forms: ``omega_n = sqrt(k/J)``, ``zeta = c / (2 sqrt(k J))``.
    """
    k = uniform_segment_stiffness(length, outer_diameter, inner_diameter, G_SHEAR)
    c = 2.0 * damping_ratio * math.sqrt(k * inertia)
    chain = DrivetrainChain()
    prop = chain.add_node("propeller", inertia)
    ground = chain.add_node("gearbox_ground", 8.0e2, fixed=True)
    segment = chain.add_shaft(ShaftSegment("shaft1", k, c, prop, ground,
                                           outer_diameter, inner_diameter))
    return {"chain": chain, "k": k, "c": c, "inertia": inertia,
            "omega_n": math.sqrt(k / inertia), "damping_ratio": damping_ratio,
            "segment": segment}


# ---------------------------------------------------------------------------
# C1 - natural frequencies against closed forms
# ---------------------------------------------------------------------------
def c1_natural_frequencies() -> Dict[str, object]:
    """Three analytic configurations plus the reflected-inertia consistency case."""
    k1 = uniform_segment_stiffness(6.0, 0.42, 0.18, G_SHEAR)
    k2 = uniform_segment_stiffness(5.0, 0.40, 0.16, G_SHEAR)
    k3 = uniform_segment_stiffness(2.0, 0.25, 0.0, G_SHEAR)
    cases: List[Dict[str, object]] = []

    # (a) two inertias, far end grounded
    j1 = 1.2e3
    chain = DrivetrainChain()
    a = chain.add_node("propeller", j1)
    b = chain.add_node("motor", 8.0e2, fixed=True)
    chain.add_shaft(ShaftSegment("shaft1", k1, 0.0, a, b, 0.42, 0.18))
    sim = chain.modal_frequencies()[0]
    ref = two_inertia_fixed_frequency(k1, j1) / (2.0 * math.pi)
    cases.append({"configuration": "two_inertia_grounded", "simulated_hz": sim,
                  "reference_hz": ref, "relative_error": abs(sim - ref) / ref})

    # (b) two inertias, free-free
    j2 = 8.0e2
    chain = DrivetrainChain()
    a = chain.add_node("propeller", j1)
    b = chain.add_node("motor", j2)
    chain.add_shaft(ShaftSegment("shaft1", k1, 0.0, a, b, 0.42, 0.18))
    sim = chain.modal_frequencies()[0]
    ref = two_inertia_free_frequency(k1, j1, j2) / (2.0 * math.pi)
    cases.append({"configuration": "two_inertia_free_free", "simulated_hz": sim,
                  "reference_hz": ref, "relative_error": abs(sim - ref) / ref})

    # (c) three inertias, free-free: exact characteristic polynomial
    jj = (9.0e2, 5.0e2, 1.5e3)
    chain = DrivetrainChain()
    na = chain.add_node("propeller", jj[0])
    nb = chain.add_node("gear", jj[1])
    nc = chain.add_node("motor", jj[2])
    chain.add_shaft(ShaftSegment("shaft1", k2, 0.0, na, nb, 0.40, 0.16))
    chain.add_shaft(ShaftSegment("shaft2", k3, 0.0, nb, nc, 0.25, 0.0))
    sim_modes = np.sort(chain.modal_frequencies())
    ref_modes = np.sort(np.array(three_inertia_chain_frequencies(k2, k3, *jj)) / (2.0 * math.pi))
    for idx, (sim, ref) in enumerate(zip(sim_modes, ref_modes)):
        cases.append({"configuration": f"three_inertia_free_free_mode{idx + 1}",
                      "simulated_hz": float(sim), "reference_hz": float(ref),
                      "relative_error": float(abs(sim - ref) / ref)})

    # (d) reflected-inertia law: stiff mesh + stiff shaft2 -> wheel side sees J_motor i**2
    ratio = 4.0
    chain = standard_chain(gear_stiffness=1.0e9, mesh_k0=1.0e12, mesh_modulation=0.0)
    k_shaft = chain.shafts[0].stiffness
    effective = chain.nodes[1].inertia + chain.nodes[3].inertia * ratio ** 2
    ref = math.sqrt(k_shaft * (1.0 / chain.nodes[0].inertia + 1.0 / effective)) / (2.0 * math.pi)
    sim = chain.modal_frequencies()[0]
    cases.append({"configuration": "reflected_inertia_low_speed_mode",
                  "simulated_hz": float(sim), "reference_hz": float(ref),
                  "relative_error": float(abs(sim - ref) / ref)})

    worst = max(c["relative_error"] for c in cases)
    return {"criterion": "C1", "threshold": 0.03, "worst_relative_error": worst,
            "cases": cases, "pass": bool(worst <= 0.03)}


# ---------------------------------------------------------------------------
# C2 - gear periodic stiffness response
# ---------------------------------------------------------------------------
def c2_gear_parametric(inertia: float = 4.0e1, k0: float = 4.0e7,
                       modulation: float = 0.2, teeth: int = 21,
                       damping_ratio: float = 0.02, ripple: float = 2.0e4,
                       speed_ratio: float = 0.1) -> Dict[str, object]:
    """Quasi-static ripple response and the Mathieu growth-rate check."""
    omega_n = math.sqrt(k0 / inertia)
    damping = 2.0 * damping_ratio * math.sqrt(k0 * inertia)
    mesh = MeshStiffness(k0, modulation, teeth)
    omega_mean = speed_ratio * omega_n / teeth
    osc = ParametricMeshOscillator(inertia, damping, mesh, omega_mean)
    mesh_omega = osc.mesh_frequency
    t_end = 4.0 * 2.0 * math.pi / mesh_omega
    dt = (2.0 * math.pi / omega_n) / 200.0
    history = osc.integrate(t_end, dt, lambda t: ripple * math.cos(mesh_omega * t))
    t, phi, k_t = history["t"], history["phi"], history["stiffness"]
    window = t >= 0.75 * t_end
    tw, phiw, kw = t[window], phi[window], k_t[window]

    # closed-form extremes of the quasi-static solution T1 cos(psi)/k(psi)
    amp_theory = 0.5 * (ripple / (k0 * (1.0 - modulation)) + ripple / (k0 * (1.0 + modulation)))
    amp_sim = 0.5 * (phiw.max() - phiw.min())
    amp_error = abs(amp_sim - amp_theory) / amp_theory

    quasi_static = ripple * np.cos(mesh_omega * tw) / kw
    l2_error = float(np.sqrt(np.mean((phiw - quasi_static) ** 2)) / amp_theory)

    # peak position inside the mesh cycle: theory is exactly half a mesh period
    peak_index = int(np.argmax(np.abs(phiw)))
    peak_phase = (mesh_omega * tw[peak_index]) % (2.0 * math.pi) / (2.0 * math.pi)
    phase_error = abs(peak_phase - 0.5)

    # Mathieu growth at tooth passing = 2 omega_n
    eps_m, zeta_m = 0.2, 0.005
    omega_n_m = 2.0 * math.pi * 30.0
    osc_m = ParametricMeshOscillator(1.0, 2.0 * zeta_m * omega_n_m,
                                     MeshStiffness(omega_n_m ** 2, eps_m, teeth),
                                     2.0 * omega_n_m / teeth)
    hist_m = osc_m.integrate(0.6, (2.0 * math.pi / omega_n_m) / 200.0,
                             lambda t: 0.0, phi0=1.0e-6)
    env = np.abs(hist_m["phi"])
    tm = hist_m["t"]
    a1 = env[(tm > 0.2) & (tm < 0.3)].max()
    a2 = env[(tm > 0.5) & (tm < 0.6)].max()
    sigma_sim = math.log(a2 / a1) / 0.3
    sigma_theory = mathieu_growth_rate(omega_n_m, eps_m, zeta_m)
    sigma_error = abs(sigma_sim - sigma_theory) / sigma_theory

    passed = amp_error <= 0.05 and phase_error <= 0.05
    return {"criterion": "C2", "threshold_amplitude": 0.05, "threshold_phase_fraction": 0.05,
            "amplitude_sim": amp_sim, "amplitude_theory": amp_theory,
            "amplitude_relative_error": amp_error,
            "quasistatic_l2_error": l2_error,
            "peak_phase_fraction_sim": peak_phase, "peak_phase_fraction_theory": 0.5,
            "peak_position_error_fraction": phase_error,
            "mathieu_growth_sim": sigma_sim, "mathieu_growth_theory": sigma_theory,
            "mathieu_growth_error": sigma_error,
            "mesh_frequency_hz": mesh_omega / (2.0 * math.pi),
            "natural_frequency_hz": omega_n / (2.0 * math.pi),
            "pass": bool(passed)}


# ---------------------------------------------------------------------------
# C3 - PMSM steady state
# ---------------------------------------------------------------------------
def c3_motor_steady_state() -> Dict[str, object]:
    motor = PMSM(resistance=0.05, inductance=5.0e-3, torque_constant=1.2,
                 back_emf_constant=1.15, supply_voltage=400.0)
    inertia = 0.5
    rows = []
    for load in (0.0, 20.0, 40.0):
        result = simulate_motor(motor, inertia, load, t_end=2.0, dt=1.0e-4)
        omega_end = float(result["omega"][-1])
        reference = motor.steady_state_speed(load)
        rows.append({"load_torque_nm": load, "omega_sim": omega_end,
                     "omega_analytic": reference,
                     "relative_error": abs(omega_end - reference) / reference})
    # electrical and mechanical time constants
    # effectively locked rotor: very large inertia, load far beyond the torque
    locked = simulate_motor(motor, 1.0e9, 1.0e9, t_end=0.4, dt=1.0e-4)
    tau_e = motor.electrical_time_constant
    idx = int(np.argmin(np.abs(locked["t"] - tau_e)))
    current_ratio = float(locked["current"][idx] * motor.resistance / motor.supply_voltage)
    worst = max(r["relative_error"] for r in rows)
    return {"criterion": "C3", "threshold": 0.02, "cases": rows,
            "worst_relative_error": worst,
            "electrical_time_constant_s": tau_e,
            "mechanical_time_constant_s": motor.mechanical_time_constant(inertia),
            "locked_rotor_current_fraction_at_tau_e": current_ratio,
            "pass": bool(worst <= 0.02)}


# ---------------------------------------------------------------------------
# C4 - ice torque -> shaft stress transfer
# ---------------------------------------------------------------------------
def _project_amplitude(t: np.ndarray, signal: np.ndarray, omega: float) -> float:
    """Least-squares amplitude of ``cos(omega t)``/``sin(omega t)`` in a signal."""
    cols = np.column_stack([np.cos(omega * t), np.sin(omega * t), np.ones_like(t)])
    coef, *_ = np.linalg.lstsq(cols, signal, rcond=None)
    return float(math.hypot(coef[0], coef[1]))


def c4_ice_torque_transfer(ripple: float = 1.0e5,
                           frequencies_hz: Sequence[float] | None = None) -> Dict[str, object]:
    ref = torsional_reference_chain()
    chain = ref["chain"]
    k, c, inertia = ref["k"], ref["c"], ref["inertia"]
    omega_n = ref["omega_n"]
    wp = ref["segment"].section_modulus
    if frequencies_hz is None:
        frequencies_hz = np.linspace(0.55, 1.45, 13) * omega_n / (2.0 * math.pi)

    rows = []
    t_end = 2.0
    dt = 2.5e-4
    for f in frequencies_hz:
        omega = 2.0 * math.pi * f
        hist = chain.integrate(t_end, dt, lambda t, th, om, w=omega: np.array([ripple * math.cos(w * t), 0.0]))
        window = hist["t"] >= 0.5 * t_end
        amp = _project_amplitude(hist["t"][window], hist["theta"][window, 0], omega)
        analytic = ripple * sdof_amplitude(omega, inertia, k, c)
        rows.append({"frequency_hz": float(f), "amplitude_rad": amp,
                     "amplitude_analytic": analytic,
                     "relative_error": abs(amp - analytic) / analytic,
                     "stress_amplitude_pa": amp * k / wp})
    worst = max(r["relative_error"] for r in rows)
    peak = max(rows, key=lambda r: r["amplitude_rad"])
    peak_theory = sdof_peak_frequency(inertia, k, c) / (2.0 * math.pi)
    peak_error = abs(peak["frequency_hz"] - peak_theory) / peak_theory

    # ice-milling line spectrum: every line must follow the closed-form transfer
    torque_model = IceMillingTorque(mean_torque=0.0, blade_passing_hz=8.0,
                                    first_harmonic=2.0e4, n_harmonics=3, decay=1.0)
    hist = chain.integrate(t_end, dt,
                           lambda t, th, om: np.array([torque_model.torque(t), 0.0]))
    window = hist["t"] >= 0.5 * t_end
    lines = []
    for amp_in, freq in zip(torque_model.amplitudes, torque_model.frequencies):
        omega = 2.0 * math.pi * freq
        amp = _project_amplitude(hist["t"][window], hist["theta"][window, 0], omega)
        analytic = amp_in * sdof_amplitude(omega, inertia, k, c)
        lines.append({"frequency_hz": freq, "torque_amplitude": amp_in,
                      "amplitude_rad": amp, "amplitude_analytic": analytic,
                      "relative_error": abs(amp - analytic) / analytic})
    line_worst = max(item["relative_error"] for item in lines)
    passed = worst <= 0.03 and peak_error <= 0.03 and line_worst <= 0.03
    return {"criterion": "C4", "threshold": 0.03,
            "natural_frequency_hz": omega_n / (2.0 * math.pi),
            "frf_cases": rows, "worst_frf_error": worst,
            "peak_frequency_hz": peak["frequency_hz"], "peak_frequency_theory_hz": peak_theory,
            "peak_position_error": peak_error,
            "ice_lines": lines, "worst_ice_line_error": line_worst,
            "ripple_torque_nm": ripple, "section_modulus_m3": wp,
            "pass": bool(passed)}


# ---------------------------------------------------------------------------
# C5 - time-step coordination and interface conservation
# ---------------------------------------------------------------------------
def c5_time_step_coordination(dt_macro: float | None = None,
                              ratios: Sequence[int] = (2, 4, 8),
                              bandwidth_points: int = 25) -> Dict[str, object]:
    """Sub-cycled interface conservation + coordinator plans.

    ``dt_macro`` defaults to the interface-bandwidth limit
    (``exchange_step(f_max, 25)``) where ``f_max`` is the highest ice-torque
    harmonic, which is the step a coordination-aware run would actually choose.
    A convergence study of the residual versus ``dt_macro`` is reported with it,
    because the residual is a property of the *pair* (macro step, input bandwidth)
    and not of the sub-cycle ratio alone.
    """
    ref = torsional_reference_chain()
    chain = ref["chain"]
    torque_model = IceMillingTorque(mean_torque=0.0, blade_passing_hz=7.0,
                                    first_harmonic=3.0e4, n_harmonics=3, decay=1.0)
    f_max_interface = torque_model.frequencies[-1]
    if dt_macro is None:
        dt_macro = exchange_step(f_max_interface, bandwidth_points)
    t_end = 0.4

    def work_of(hist, torque_used):
        omega = hist["omega"][:-1, 0]
        t = hist["t"][:-1]
        dt = float(np.diff(hist["t"])[0])
        torque = np.asarray([float(torque_used(tk)) for tk in t])
        return (float(np.sum(torque * omega) * dt),
                float(np.sum(torque) * dt),
                float(np.sum(np.abs(torque)) * dt))

    n_ref = 64
    dt_ref = dt_macro / n_ref
    hist_ref = chain.integrate(t_end, dt_ref,
                               lambda t, th, om: np.array([torque_model.torque(t), 0.0]))
    work_ref, imp_ref, imp_scale_ref = work_of(hist_ref, torque_model.torque)

    def reconstructed(t: float) -> float:
        macro = math.floor(t / dt_macro)
        t0 = macro * dt_macro
        v0 = float(torque_model.torque(t0))
        v1 = float(torque_model.torque(t0 + dt_macro))
        frac = min(max((t - t0) / dt_macro, 0.0), 1.0)
        return v0 + (v1 - v0) * frac

    rows = []
    for n_sub in ratios:
        dt_sub = dt_macro / n_sub
        hist = chain.integrate(t_end, dt_sub,
                               lambda t, th, om: np.array([reconstructed(t), 0.0]))
        work, impulse, _ = work_of(hist, reconstructed)
        rows.append({"subcycle_ratio": n_sub, "dt_sub": dt_sub,
                     "interface_work": work, "interface_impulse": impulse,
                     "work_residual": abs(work - work_ref) / abs(work_ref),
                     "impulse_residual": abs(impulse - imp_ref) / imp_scale_ref})

    # convergence of the interface residual with the macro step (ratio held at max)
    convergence = []
    for factor in (2.0, 1.5, 1.0, 0.5):
        step = dt_macro * factor
        def recon_f(t: float, _h=step) -> float:
            macro = math.floor(t / _h)
            t0 = macro * _h
            v0 = float(torque_model.torque(t0))
            v1 = float(torque_model.torque(t0 + _h))
            frac = min(max((t - t0) / _h, 0.0), 1.0)
            return v0 + (v1 - v0) * frac
        h_ref = chain.integrate(t_end, step / n_ref,
                                lambda t, th, om: np.array([torque_model.torque(t), 0.0]))
        w_ref, _, _ = work_of(h_ref, torque_model.torque)
        h_scheme = chain.integrate(t_end, step / max(ratios),
                                   lambda t, th, om: np.array([recon_f(t), 0.0]))
        w_scheme, _, _ = work_of(h_scheme, recon_f)
        convergence.append({"dt_macro": step,
                            "work_residual": abs(w_scheme - w_ref) / abs(w_ref)})
    orders = []
    for a, b in zip(convergence[:-1], convergence[1:]):
        ratio = a["dt_macro"] / b["dt_macro"]
        if a["work_residual"] > 0.0 and b["work_residual"] > 0.0:
            orders.append(math.log(a["work_residual"] / b["work_residual"]) / math.log(ratio))
    observed_order = float(np.median(orders)) if orders else float("nan")

    # coordinator plans for the same physical situation
    state = {"u_max": 1.2, "f_max": 28.7, "omega_gear": 12.6, "current": 40.0,
             "f_interface": f_max_interface}
    rates = [
        SolverRate("lbm_flow", lambda s: cfl_step(dx=5.0e-3, speed=s.get("u_max", 1.0),
                                                 cfl=0.5)),
        SolverRate("torsion_fem", lambda s: structural_accuracy_step(s.get("f_max", 10.0), 20)),
        SolverRate("gear_mesh", lambda s: mesh_step(21, s.get("omega_gear", 5.0), 20)),
        SolverRate("motor_circuit", lambda s: circuit_step(5.0e-3, 0.05, 0.2)),
        SolverRate("ice_load", lambda s: exchange_step(s.get("f_interface", 21.0),
                                                       bandwidth_points)),
    ]
    plans = {}
    for strategy in ("synchronized", "subcycling", "supercycling"):
        plan = TimeStepCoordinator(rates, strategy=strategy, safety=0.9).plan(state)
        plans[strategy] = plan.as_dict()

    adaptive = AdaptiveTimeStep(TimeStepCoordinator(rates, "synchronized"), growth=1.5,
                                shrink=0.5)
    adaptive_rows = []
    previous = None
    for _ in range(6):
        previous = adaptive.update(state, previous_dt=previous)
        adaptive_rows.append(previous)
    limits = {r.name: r.dt_max(state) for r in rates}
    adaptive_ok = all(v <= min(limits.values()) * (1 + 1e-12) for v in adaptive_rows) and \
        all(b <= a * 1.5 + 1e-18 for a, b in zip(adaptive_rows[:-1], adaptive_rows[1:]))

    # contrast: hold reconstruction is first-order in the macro step
    hold = exchange_work_residual(torque_model.torque,
                                 lambda t: 12.0 + 0.5 * math.sin(2.0 * math.pi * 7.0 * t),
                                 dt_macro, max(ratios), mode="hold")
    linear = exchange_work_residual(torque_model.torque,
                                   lambda t: 12.0 + 0.5 * math.sin(2.0 * math.pi * 7.0 * t),
                                   dt_macro, max(ratios), mode="linear")

    worst = max(r["work_residual"] for r in rows)
    passed = worst <= 0.02 and adaptive_ok
    return {"criterion": "C5", "threshold": 0.02, "dt_macro": dt_macro,
            "interface_bandwidth_hz": f_max_interface,
            "interface_bandwidth_limit": exchange_step(f_max_interface, bandwidth_points),
            "reference_substeps": n_ref, "reference_work": work_ref,
            "cases": rows, "worst_work_residual": worst,
            "convergence": convergence, "observed_order": observed_order,
            "plans": plans, "stability_limits": limits,
            "adaptive_sequence": adaptive_rows, "adaptive_ok": bool(adaptive_ok),
            "hold_mode_residual": hold["work_residual"],
            "linear_mode_residual": linear["work_residual"],
            "pass": bool(passed)}


# ---------------------------------------------------------------------------
# C6 - IACS Polar Class screening and fatigue life
# ---------------------------------------------------------------------------
def c6_iacs_check(design_life_years: float = 20.0,
                  polar_classes: Sequence[str] = ("PC2", "PC4", "PC6")) -> Dict[str, object]:
    ref = torsional_reference_chain()
    chain = ref["chain"]
    k, wp = ref["k"], ref["segment"].section_modulus
    torque_model = IceMillingTorque.from_ice_condition(
        ice_thickness=1.0, advance_speed=2.0, rpm=150.0, n_blades=5,
        mean_coefficient=2.5e5, amplitude_coefficient=5.0e4)
    t_end, dt = 2.0, 2.0e-4
    hist = chain.integrate(t_end, dt,
                           lambda t, th, om: np.array([torque_model.torque(t), 0.0]))
    window = hist["t"] >= 0.5 * t_end
    fundamental = torque_model.frequencies[0]
    amp = _project_amplitude(hist["t"][window], hist["theta"][window, 0],
                             2.0 * math.pi * fundamental)
    stress_amp = amp * k / wp
    mean_stress = torque_model.mean_torque / wp
    yield_shear = 0.577 * 355.0e6  # von Mises shear yield, model assumption
    cycles_per_hour = fundamental * 3600.0

    rows = [iacs_torsion_check(pc, stress_amp, cycles_per_hour, design_life_years)
            for pc in polar_classes]

    # severe hypothetical: the same ripple moved onto the first torsional mode
    omega_n = ref["omega_n"]
    severe_hist = chain.integrate(t_end, dt,
                                  lambda t, th, om: np.array(
                                      [4.0 * torque_model.first_harmonic * math.cos(omega_n * t), 0.0]))
    severe_window = severe_hist["t"] >= 0.5 * t_end
    severe_amp = _project_amplitude(severe_hist["t"][severe_window],
                                    severe_hist["theta"][severe_window, 0], omega_n)
    severe_stress = severe_amp * k / wp
    severe = iacs_torsion_check("PC2", severe_stress, omega_n / (2.0 * math.pi) * 3600.0,
                                design_life_years)

    passed = all(r["stress_ok"] and r["life_ok"] for r in rows)
    return {"criterion": "C6", "design_life_years": design_life_years,
            "stress_amplitude_pa": stress_amp, "mean_stress_pa": mean_stress,
            "mean_stress_utilization_vs_yield": mean_stress / yield_shear,
            "mean_torque_nm": torque_model.mean_torque,
            "torque_ripple_nm": torque_model.first_harmonic,
            "cycles_per_hour": cycles_per_hour, "polar_class_checks": rows,
            "severe_resonance_case": severe,
            "severe_discriminates": bool(not (severe["stress_ok"] and severe["life_ok"])),
            "parameter_status": rows[0]["parameter_status"],
            "pass": bool(passed)}


def run_acceptance() -> Dict[str, object]:
    """Run C1..C6 and return the aggregate report dictionary."""
    checks = [c1_natural_frequencies(), c2_gear_parametric(), c3_motor_steady_state(),
              c4_ice_torque_transfer(), c5_time_step_coordination(), c6_iacs_check()]
    return {"node": "p348w3-n08-physics-torsional-timestep",
            "checks": checks,
            "all_pass": all(c["pass"] for c in checks)}