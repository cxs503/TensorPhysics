"""End-to-end three-stage chain: wave tank -> light 6-DOF body -> whipping.

One scenario, run by :func:`run_chain_case`:

1. **Stage 1 (fluid).** Second-order Stokes wave in a 10 m flume; the pressure
   field on the wetted hull is integrated into the 6-DOF generalised force
   (interface A).
2. **Stage 2 (body).** A *light* flexible barge model (``m / A33 = 0.24``, i.e.
   structural mass below the added mass) is marched with RK4 in 6 DOF; the
   partitioned interface is stabilised with Aitken/IQN-ILS because the naive
   Gauss-Seidel iteration diverges (interface B, negative control N).
3. **Stage 3 (structure).** The barge plunges, the Wagner-type slamming pressure
   field is projected on the wet bending modes (interface C) and the midship
   whipping stress follows from the Duhamel convolution.

Outputs: six products (A/B/C/E/T/N) -> ``chain_readiness.v1`` verdict.

Everything the gate reads is computed here; nothing is hard-coded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .beam import FlexibleBarge
from .body import BodyDynamics, BoxBody, RigidPose, force_moment
from .interfaces import (
    duhamel_response,
    face_a_convergence,
    face_a_force,
    face_b_closed_flux,
    face_b_power_reciprocity,
    face_b_volume_rate,
    face_c_generalised_force,
    rock4_modal,
)
from .readiness import evaluate_chain, write_readiness
from .slamming import WagnerSlamming
from .stabilization import InterfaceProblem, divergence_detection_rate, solve_interface, spectral_radius
from .time_ledger import detection, footprint, reconstruct, signal_error
from .waves import StokesWave

__all__ = ["ChainConfig", "run_chain_case", "main"]


@dataclass(frozen=True)
class ChainConfig:
    """Scenario definition (all SI; every number is a documented model input)."""

    wave_height: float = 1.0
    wave_period: float = 8.0
    depth: float = 10.0
    rho: float = 1025.0
    deadrise_deg: float = 10.0
    heave_damping_ratio: float = 0.15      # fraction of critical, model input
    pitch_damping_ratio: float = 0.15
    n_modes: int = 8
    plunge_velocity: float = 1.0           # downward velocity at first contact [m/s]
    plunge_offset: float = 0.05            # initial heave above the design draught [m]
    fine_samples_per_period: int = 64      # production coupling rate
    coarse_samples_per_period: int = 8     # cheap-campaign coupling rate (negatives)
    impact_samples: int = 8                # samples per impact duration
    march_periods: float = 5.0
    dt_march: float = 5.0e-3
    dt_eval: float = 1.25e-2
    n_edge_force: int = 12                 # panelling of the operating force integral
    barge: FlexibleBarge = field(default_factory=FlexibleBarge)


# --------------------------------------------------------------------------
# case assembly
# --------------------------------------------------------------------------
def _build_case(cfg: ChainConfig) -> dict:
    wave = StokesWave(height=cfg.wave_height, period=cfg.wave_period, depth=cfg.depth, rho=cfg.rho)
    barge = cfg.barge
    body = BoxBody(
        half_beam=0.5 * barge.length,
        draught=barge.draught,
        width=barge.breadth,
        mass=barge.mass,
        depth=barge.depth,
        rho=cfg.rho,
    )
    beam = barge.wet_beam(n_modes=cfg.n_modes)
    a33 = barge.sectional_added_mass()

    # 6-DOF added mass: strip integration of the 2-D sectional value over the
    # hull length (heave) and its second moment (pitch).  Model input; no claim
    # of 3-D diffraction accuracy.
    added = np.zeros((6, 6))
    added[2, 2] = a33 * barge.length
    added[4, 4] = a33 * barge.length ** 3 / 12.0

    mass = body.mass_matrix
    restoring = body.restoring_matrix
    damping = np.zeros((6, 6))
    natural = {}
    for dof, zeta, name in (
        (2, cfg.heave_damping_ratio, "heave"),
        (4, cfg.pitch_damping_ratio, "pitch"),
    ):
        omega_n = float(np.sqrt(restoring[dof, dof] / (mass[dof, dof] + added[dof, dof])))
        natural[name] = omega_n
        damping[dof, dof] = 2.0 * zeta * omega_n * (mass[dof, dof] + added[dof, dof])

    return {
        "wave": wave,
        "body": body,
        "panels_force": body.panels(RigidPose(), n_edge=cfg.n_edge_force),
        "beam": beam,
        "barge": barge,
        "added_mass": added,
        "damping": damping,
        "dynamics": BodyDynamics(body=body, added_mass=added, damping=damping),
        "a33_per_length": a33,
        "natural_omega": natural,
        "mass": mass,
        "restoring": restoring,
    }


def _pressure_at_case(case: dict, t: float):
    wave = case["wave"]

    def p_at(points: np.ndarray) -> np.ndarray:
        return wave.pressure(points[:, 0], points[:, 2], t)

    return p_at


def _force6(case: dict, t: float, n_edge: int | None = None) -> np.ndarray:
    """6-DOF generalised force from the wave pressure field (interface A).

    The panel set is built once and cached (``case["panels_force"]``); the pose
    is the design pose, so the geometry does not change with time.
    """
    panels = case["panels_force"] if n_edge is None else case["body"].panels(
        RigidPose(), n_edge=int(n_edge)
    )
    F, M = face_a_force(_pressure_at_case(case, t), panels, np.zeros(3), method="gauss", order=3)
    # the body's weight closes the static balance: without it the hydrostatic
    # buoyancy would be an unbalanced constant force and the marched body would
    # sit at F_buoy/K instead of its design draught.
    F = F - np.array([0.0, 0.0, case["barge"].mass * 9.80665])
    return np.concatenate([F, M])


def _force6_series(case: dict, times: np.ndarray, n_edge: int | None = None) -> np.ndarray:
    return np.array([_force6(case, float(t), n_edge) for t in np.asarray(times, dtype=float)])


def _harmonic_amplitude(t: np.ndarray, y: np.ndarray, omega: float) -> np.ndarray:
    """Least-squares complex amplitude of the ``omega`` harmonic of ``y(t)``."""
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    basis = np.column_stack([np.cos(omega * t), np.sin(omega * t), np.ones_like(t)])
    out = []
    for j in range(y.shape[1]):
        coef, *_ = np.linalg.lstsq(basis, y[:, j], rcond=None)
        out.append(complex(coef[0], -coef[1]))
    return np.array(out)


# --------------------------------------------------------------------------
# face A
# --------------------------------------------------------------------------
def _face_a(cfg: ChainConfig, case: dict) -> dict:
    wave, body = case["wave"], case["body"]
    t_eval = 0.25 * cfg.wave_period          # non-symmetric loading -> pitch moment
    conv = face_a_convergence(
        wave, body, t=t_eval, n_edges=(2, 4, 8, 16, 32), order=2, reference=(64, 6)
    )
    # the integrated pitch moment of this loading is nearly zero, so a plain
    # relative moment error is meaningless: normalise by force * length instead.
    panels_ref = body.panels(RigidPose(), n_edge=64)
    M_ref = face_a_force(
        lambda pts: wave.pressure(pts[:, 0], pts[:, 2], t_eval), panels_ref, np.zeros(3),
        method="gauss", order=6,
    )[1]
    M_scale = float(np.linalg.norm(np.asarray(conv["F_ref"])) * body.half_beam)
    for row in conv["rows"]:
        pn = body.panels(RigidPose(), n_edge=row["n_edge"])
        M = face_a_force(
            lambda pts: wave.pressure(pts[:, 0], pts[:, 2], t_eval), pn, np.zeros(3),
            method="gauss", order=2,
        )[1]
        row["moment_err_norm"] = float(np.linalg.norm(M - M_ref) / max(M_scale, 1e-30))

    pressure_scale = float(
        abs(case["wave"].dynamic_pressure(np.array([0.0]), np.array([-body.draught]), 0.0)[0])
    )
    return {
        "force_rel_err": float(conv["rel_err"]),
        "observed_order": float(conv["order"]),
        "reference_self_consistency": float(conv["reference"]["self_consistency"]),
        "gauss_rel_err": float(conv["rel_err_gauss"]),
        "gauss_order": float(conv["quadrature_rel_err_order"]),
        "t_eval_s": float(t_eval),
        "rows": conv["rows"],
        "F_ref_N": conv["F_ref"],
        "M_ref_Nm": np.asarray(M_ref).tolist(),
        "dynamic_pressure_scale_Pa": pressure_scale,
    }


# --------------------------------------------------------------------------
# face B
# --------------------------------------------------------------------------
def _face_b(cfg: ChainConfig, case: dict) -> dict:
    body, wave, dyn = case["body"], case["wave"], case["dynamics"]
    omega = wave.omega

    # (i) closed-surface zero-flux identity for a rigid motion at a general pose
    pose = RigidPose(r=np.array([0.0, 0.0, 0.02]), theta=np.array([0.0, 0.03, 0.0]))
    v = np.array([0.05, 0.0, 0.30])
    om = np.array([0.0, 0.20, 0.0])
    closed = face_b_closed_flux(body, pose, v, om, n_edges=(4, 8, 16, 32))
    neg = face_b_closed_flux(body, pose, v, om, n_edges=(16,), rotate_normals=False)

    # (ii) immersed-volume rate identity in heave
    vol = face_b_volume_rate(body, heave=0.02, heave_rate=0.30, n_edges=(4, 8, 16, 32))

    # (iii) normal velocity + power reciprocity across the delivered hand-off
    dt_h = wave.period / cfg.fine_samples_per_period
    t_c = np.arange(0.0, wave.period + 0.5 * dt_h, dt_h)
    t_fine = np.arange(0.0, wave.period + 0.5 * cfg.dt_eval, cfg.dt_eval)
    F_hat = _harmonic_amplitude(t_c, _force6_series(case, t_c), omega)
    x_hat, v_hat = dyn.harmonic_steady_state(F_hat, omega)
    v_fine = np.real(v_hat[None, :] * np.exp(1j * omega * t_fine)[:, None])
    v_src = np.real(v_hat[None, :] * np.exp(1j * omega * t_c)[:, None])
    normal_err = signal_error(t_c, v_src[:, 2], t_fine, v_fine[:, 2], order=2, offset=0.0)["max_rel_err"]

    t0 = 0.25 * wave.period + 0.7 * dt_h    # deliberately off the hand-off grid
    panels = body.panels(RigidPose(), n_edge=32)
    p_now = wave.pressure(panels.centroid[:, 0], panels.centroid[:, 2], t0)
    f_panel = -(p_now * panels.area)[:, None] * panels.normal
    u_analytic = np.column_stack(
        [np.full(len(panels), np.real(v_hat[j] * np.exp(1j * omega * t0))) for j in range(3)]
    )
    u_del = u_analytic.copy()
    u_del[:, 2] = reconstruct(t_c, v_src[:, 2], np.array([t0]), order=2, offset=0.0)[0]
    power = face_b_power_reciprocity(panels, f_panel, u_analytic, u_del)

    return {
        "closed_flux_rel": float(closed["rel_flux"]),
        "closed_flux_negative_control_rel": float(neg["rel_flux"]),
        "volume_rel_err": float(vol["rel_err"]),
        "normal_velocity_rel_err": float(normal_err),
        "power_rel_err": float(power["rel_err"]),
        "power_analytic_W": power["power_analytic_W"],
        "power_delivered_W": power["power_delivered_W"],
        "closed_rows": closed["rows"],
        "volume_rows": vol["rows"],
        "pose": {"r": pose.r.tolist(), "theta": pose.theta.tolist()},
        "velocity": v.tolist(),
        "omega": om.tolist(),
    }


# --------------------------------------------------------------------------
# face C + whipping transient
# --------------------------------------------------------------------------
def _impact_velocity_history(
    cfg: ChainConfig, case: dict, t: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Relative entry velocity history and the slamming station.

    The body is plunged from ``cfg.plunge_offset`` with ``cfg.plunge_velocity``;
    the wave orbital velocity at the bow keel and the body's own heave velocity
    (from the 6-DOF march) give the approach velocity.  The contact ends when
    the body has been decelerated by the impact impulse (momentum balance).
    """
    wave, body, dyn = case["wave"], case["body"], case["dynamics"]
    x_imp = 0.5 * body.half_beam            # bow region

    # free plunge of the light body: march the 6-DOF with wave force + impact-free
    state = np.zeros(12)
    state[2] = cfg.plunge_offset
    state[8] = -cfg.plunge_velocity         # heave velocity (dof index 2 -> slot 6+2)
    dt = cfg.dt_march
    n = int(np.ceil(t[-1] / dt)) + 1
    v_body = np.zeros(n)
    t_march = np.arange(n) * dt
    for k in range(1, n):
        def f(tt, s, _k=k):
            return _force6(case, float(t_march[_k - 1] + tt))

        state = dyn.march(state, dt, f)
        v_body[k] = state[8]
    v_body = np.interp(t, t_march, v_body)
    w_fluid = wave.velocity(x_imp, -body.draught, t)[1]
    approach = -(v_body - w_fluid)
    v_rel = np.clip(approach, 0.0, None)
    return v_rel, float(x_imp), float(np.max(v_rel))


def _impact_pulse_time(model: WagnerSlamming, v_entry: float) -> float:
    """How long the slamming pressure acts on a fixed station (Wagner sweep).

    The spray root moves outwards at ``d c / dt = (pi/2) V / tan(beta)``, so the
    contact sweeps the attainable wetted half-width ``c_max`` in

        ``t_pulse = c_max / (d c / dt) = c_max tan(beta) / ((pi/2) V)``.

    For the model barge this is ~50 ms - short compared with the first whipping
    period (~85 ms), which is exactly why the slamming response is dynamic.  The
    hull's own vertical momentum is *not* absorbed by this pulse (the added water
    mass is); :func:`_impact_impulse` reports the delivered impulse as a fraction
    of the hull momentum so the assumption is auditable.
    """
    rate = model.width_rate * max(v_entry, 1e-9)
    return float(model.max_half_width / rate)


def _impact_impulse(model: WagnerSlamming, x: np.ndarray, x_imp: float, t_c: float,
                    v_entry: float, n: int = 160) -> float:
    """Vertical impulse [N s] delivered by the idealised pulse."""
    xi = np.linspace(-model.max_half_width, model.max_half_width, 121)
    dx = float(x[1] - x[0])
    tt = np.linspace(0.0, t_c, n)
    c_xi = model.contact_half_width(tt, np.full(tt.size, v_entry))
    force = np.array(
        [
            float(np.sum(np.trapezoid(model.field(x, xi, t, v_entry, cc, x_imp), xi, axis=1)) * dx)
            for t, cc in zip(tt, c_xi)
        ]
    )
    return float(np.trapezoid(force, tt))


def _modal_time_grid(beam, t_end: float, samples_per_period: int = 60) -> np.ndarray:
    """Modal integration grid: resolves the highest retained wet mode."""
    f_max = float(np.max(beam.omega)) / (2.0 * np.pi)
    dt = 1.0 / (samples_per_period * f_max)
    n = int(np.ceil(t_end / dt)) + 1
    return np.linspace(0.0, t_end, n)


def _generalised_force_history(model: WagnerSlamming, t: np.ndarray, v_rel: np.ndarray,
                               c_xi: np.ndarray, x_imp: float, beam_x: np.ndarray,
                               shapes: np.ndarray, n_xi: int = 121, n_x_field: int = 401) -> dict:
    """Modal generalised force from the full 2-D slamming field at every instant."""
    rows = [
        face_c_generalised_force(
            lambda xx, xi, _t=tt, _v=vv, _c=cc: model.field(xx, xi, _t, _v, _c, x_imp),
            beam_x,
            model.max_half_width,
            shapes,
            n_xi=n_xi,
            n_x_field=n_x_field,
        )
        for tt, vv, cc in zip(t, v_rel, c_xi)
    ]
    return {
        "Q_field": np.array([r["Q_field_N"] for r in rows]).T,
        "Q_strip": np.array([r["Q_strip_N"] for r in rows]).T,
        "rel_err": np.array([r["rel_err"] for r in rows]),
        "line_load": np.array([r["line_load_N_m"] for r in rows]),
    }


def _face_c(cfg: ChainConfig, case: dict) -> dict:
    beam, barge = case["beam"], case["barge"]
    model = WagnerSlamming(rho=cfg.rho, deadrise_deg=cfg.deadrise_deg, keel_depth=barge.draught,
                           decay_length=0.9, max_half_width=0.45 * barge.breadth,
                           max_patch_half_length=0.25 * barge.length)
    t_imp = np.linspace(0.0, 0.4, 161, endpoint=True)
    v_rel_imp, x_imp, v_max = _impact_velocity_history(cfg, case, t_imp)
    x_axis = np.linspace(0.0, barge.length, 201)
    t_c = _impact_pulse_time(model, float(v_max))
    impulse = _impact_impulse(model, x_axis, x_imp, t_c, float(v_max))

    # hand-off sampling grid (what the coupling actually exchanges); the entry
    # velocity is held constant over the (short) pressure pulse
    t_coarse = np.linspace(0.0, max(4.0 * t_c, 0.05), 201)
    v = np.where(t_coarse <= t_c, float(v_max), 0.0)
    c_xi = model.contact_half_width(t_coarse, v)

    beam_x = beam.x + 0.5 * barge.length          # beam axis 0..L, stern to bow
    hist = _generalised_force_history(model, t_coarse, v, c_xi, x_imp, beam_x, beam.shapes)
    Q_field, Q_strip = hist["Q_field"], hist["Q_strip"]
    rel_q = np.abs(Q_strip - Q_field) / np.maximum(np.abs(Q_field), np.max(np.abs(Q_field)) * 1e-12)

    # modal integration on the fine grid, fed by a high-order reconstruction of
    # the handed-over generalised force (the reference route for the time ledger)
    t_mod = _modal_time_grid(beam, t_coarse[-1])
    from scipy.interpolate import CubicSpline

    Q_ref = np.array([CubicSpline(t_coarse, Q_field[j])(t_mod) for j in range(beam.n_modes)])
    q_duh = duhamel_response(t_mod, Q_ref, beam.omega, beam.zeta)
    q_rk4 = rock4_modal(t_mod, Q_ref, beam.omega, beam.zeta)
    duh_err = float(np.max(np.abs(q_duh - q_rk4)) / max(np.max(np.abs(q_rk4)), 1e-30))
    stress = beam.stress_history(q_duh)
    i_peak = int(np.argmax(np.abs(stress)))
    # modal truncation audit: how much of the midship stress needs which modes
    truncation = {}
    for keep in (2, 4, 6, beam.n_modes):
        nk = min(int(keep), beam.n_modes)
        q_k = duhamel_response(t_mod, Q_ref[:nk], beam.omega[:nk], beam.zeta[:nk])
        s_k = beam.stress_shape[:nk] @ q_k
        truncation[str(nk)] = float(np.max(np.abs(s_k)))

    return {
        "modal_force_rel_err": float(np.max(rel_q)),
        "duhamel_rel_err": duh_err,
        "modal_force_rel_err_by_mode": np.max(rel_q, axis=1).tolist(),
        "t_s": t_coarse.tolist(),
        "v_rel_m_s": v.tolist(),
        "contact_half_width_m": c_xi.tolist(),
        "impact_duration_s": float(t_c),
        "impact_impulse_Ns": float(impulse),
        "impulse_fraction_of_hull_momentum": float(impulse / (barge.mass * max(v_max, 1e-9))),
        "impact_velocity_max_m_s": float(v_max),
        "impact_station_x_m": float(x_imp),
        "peak_spray_root_pressure_Pa": float(np.max(model.peak_pressure(v))),
        "Q_ref_N": Q_field.tolist(),
        "q_duhamel": q_duh.tolist(),
        "t_modal_s": t_mod.tolist(),
        "stress_Pa": stress.tolist(),
        "peak_whipping_stress_Pa": float(np.max(np.abs(stress))),
        "peak_whipping_stress_MPa": float(np.max(np.abs(stress)) / 1e6),
        "static_stress_estimate_Pa": float(
            np.max(np.abs(np.trapezoid(hist["line_load"], beam_x, axis=1)))
            * barge.length / 8.0 / barge.section_modulus
        ),
        "t_peak_s": float(t_mod[i_peak]),
        "modal_dt_s": float(t_mod[1] - t_mod[0]),
        "peak_stress_by_mode_count_Pa": truncation,
        "model": {
            "deadrise_deg": cfg.deadrise_deg,
            "pressure_coefficient": model.pressure_coefficient,
            "keel_depth_m": model.keel_depth,
            "max_half_width_m": model.max_half_width,
            "max_patch_half_length_m": model.max_patch_half_length,
        },
    }


# --------------------------------------------------------------------------
# energy ledger (E)
# --------------------------------------------------------------------------
def _energy(cfg: ChainConfig, case: dict) -> dict:
    """One-wave-period energy ledger across the whole chain.

    ``W_in``  work of the *delivered* (handed-over) generalised force on the
              physical interface velocity,
    ``dKE+dPE`` change of the body's kinetic (incl. added mass) and hydrostatic
              potential energy over the period,
    ``W_diss``  dissipation by the body damping.

    The residual ``W_in - (dKE + dPE + W_diss)`` is exactly criterion (c).
    Because the hands-over force and velocity are *independently* known (the
    force from the fluid pressure field, the velocity from the coupled
    steady state), a synchronisation or interpolation defect in the hand-off
    shows up here as work that the body cannot account for.
    """
    wave, dyn = case["wave"], case["dynamics"]
    omega = wave.omega
    T = wave.period
    dt_h = T / cfg.fine_samples_per_period
    t_c = np.arange(0.0, T + 0.5 * dt_h, dt_h)
    t_fine = np.arange(0.0, T + 0.5 * cfg.dt_eval, cfg.dt_eval)

    F_src = _force6_series(case, t_c)
    F_hat = _harmonic_amplitude(t_c, F_src, omega)
    x_hat, v_hat = dyn.harmonic_steady_state(F_hat, omega)

    v_fine = np.real(v_hat[None, :] * np.exp(1j * omega * t_fine)[:, None])
    x_fine = np.real(x_hat[None, :] * np.exp(1j * omega * t_fine)[:, None])
    B = dyn.damping

    def ledger(order: int, offset: float, t_source: np.ndarray | None = None,
               F_source: np.ndarray | None = None) -> dict:
        ts = t_c if t_source is None else t_source
        Fs = F_src if F_source is None else F_source
        F_del = reconstruct(ts, Fs, t_fine, order=order, offset=offset)
        power = np.einsum("ij,ij->i", F_del, v_fine)
        W_in = float(np.trapezoid(power, t_fine))
        p_diss = np.einsum("ij,jk,ik->i", v_fine, B, v_fine)
        W_diss = float(np.trapezoid(p_diss, t_fine))
        e0 = dyn.energy(np.concatenate([x_fine[0], v_fine[0]]))
        e1 = dyn.energy(np.concatenate([x_fine[-1], v_fine[-1]]))
        dE = e1["total_J"] - e0["total_J"]
        resid = abs(W_in - (dE + W_diss)) / max(abs(W_in), 1e-30)
        return {
            "W_in_J": W_in,
            "W_diss_J": W_diss,
            "dKE_plus_dPE_J": dE,
            "kinetic_share_J": e1["kinetic_J"],
            "potential_share_J": e1["potential_J"],
            "rel_err": float(resid),
        }

    main_ledger = ledger(2, 0.0)

    # marched-body ledger: RK4 integration of the same 6-DOF driven by the
    # delivered force must book exactly the same energy (added-mass KE,
    # hydrostatic PE, damping dissipation)
    state = np.concatenate([np.real(x_hat), np.real(v_hat)])
    # the marched body must see the *same* force model that the bookkeeping
    # integrates: build the delivered force directly on the marching grid and
    # tile it periodically (a coarse ZOH inside the step would leave an O(dt)
    # quadrature residue of the same size as the quantity being checked).
    n_per = int(round(T / cfg.dt_march))
    t_one = np.arange(n_per) * cfg.dt_march
    F_march = np.tile(reconstruct(t_c, F_src, t_one, order=2, offset=0.0), (int(round(cfg.march_periods)), 1))
    e_start = dyn.energy(state)["total_J"]
    w_in_march = 0.0
    w_diss_march = 0.0
    t_cur = 0.0
    heave_hist = []
    last_period_from = n_per * (int(round(cfg.march_periods)) - 1)
    F_prev = F_march[0]
    v_prev = state[6:12]
    p_prev = float(F_prev @ v_prev)
    d_prev = float(v_prev @ B @ v_prev)
    for k in range(F_march.shape[0]):
        F_now = F_march[k]
        v_now = state[6:12]
        p_now = float(F_now @ v_now)
        d_now = float(v_now @ B @ v_now)
        # trapezoidal bookkeeping (a left rectangle would leave an O(dt)
        # quadrature residue that is larger than the quantity being checked)
        w_in_march += 0.5 * (p_prev + p_now) * cfg.dt_march
        w_diss_march += 0.5 * (d_prev + d_now) * cfg.dt_march
        F_next = F_march[(k + 1) % F_march.shape[0]]
        # linear-in-step force: consistent with the trapezoidal bookkeeping below
        state = dyn.march(
            state,
            cfg.dt_march,
            lambda tt, _s, _a=F_now, _b=F_next, _h=cfg.dt_march: _a + (_b - _a) * (tt / _h),
        )
        t_cur += cfg.dt_march
        F_prev, v_prev, p_prev, d_prev = F_now, state[6:12].copy(), p_now, d_now
        if k >= last_period_from:
            heave_hist.append(float(state[2]))
    e_end = dyn.energy(state)["total_J"]
    body_resid = abs(w_in_march - ((e_end - e_start) + w_diss_march)) / max(abs(w_in_march), 1e-30)
    # drift of the marched response away from the periodic steady state
    amp_marched = float(0.5 * (np.max(heave_hist) - np.min(heave_hist)))
    amp_analytic = float(np.max(np.abs(x_fine[:, 2])))

    return {
        "wave_ledger_rel_err": float(main_ledger["rel_err"]),
        "body_ledger_rel_err": float(body_resid),
        "wave": main_ledger,
        "marched": {
            "W_in_J": w_in_march,
            "W_diss_J": w_diss_march,
            "dE_J": e_end - e_start,
            "periods": float(cfg.march_periods),
            "amplitude_marched_m": amp_marched,
            "amplitude_analytic_m": amp_analytic,
            "amplitude_rel_diff": float(abs(amp_marched - amp_analytic) / max(amp_analytic, 1e-30)),
        },
        "energy_perturbation_J": dyn.energy(np.concatenate([np.real(x_hat), np.real(v_hat)]))["total_J"],
        "excitation_harmonic_N": [complex(z) for z in F_hat],
        "steady_state_heave_m": float(abs(x_hat[2])),
        "damping_ratio": {
            "heave": float(B[2, 2] / (2.0 * np.sqrt(dyn.restoring[2, 2] * dyn.effective_mass[2, 2]))),
            "pitch": float(B[4, 4] / (2.0 * np.sqrt(dyn.restoring[4, 4] * dyn.effective_mass[4, 4]))),
        },
    }


# --------------------------------------------------------------------------
# time-synchronisation ledger (T)
# --------------------------------------------------------------------------
def _time_ledger(cfg: ChainConfig, case: dict, face_c: dict) -> dict:
    wave, dyn = case["wave"], case["dynamics"]
    omega, T = wave.omega, wave.period
    t_fine = np.arange(0.0, T + 0.5 * cfg.dt_eval, cfg.dt_eval)

    F_true = _force6_series(case, t_fine)
    F_hat = _harmonic_amplitude(t_fine, F_true, omega)
    x_hat, v_hat = dyn.harmonic_steady_state(F_hat, omega)
    v_fine = np.real(v_hat[None, :] * np.exp(1j * omega * t_fine)[:, None])
    B = dyn.damping
    W_diss = float(np.trapezoid(np.einsum("ij,jk,ik->i", v_fine, B, v_fine), t_fine))

    def wave_metric(dt_h: float, order: int, offset: float) -> dict:
        t_c = np.arange(0.0, T + 0.5 * dt_h, dt_h)
        F_src = _force6_series(case, t_c)
        F_del = reconstruct(t_c, F_src, t_fine, order=order, offset=offset)
        err = np.max(np.abs(F_del - F_true)) / max(float(np.max(np.abs(F_true))), 1e-30)
        W_in = float(np.trapezoid(np.einsum("ij,ij->i", F_del, v_fine), t_fine))
        return {
            "force_max_rel_err": float(err),
            "wave_ledger_rel_err": float(abs(W_in - W_diss) / max(abs(W_in), 1e-30)),
            "W_in_J": W_in,
        }

    wave_rows = footprint(
        wave_metric,
        period=T,
        densities=(8, 16, 32, 64),
        orders=(0, 1, 2),
        offset_fractions=(0.0, 0.5),
    )["rows"]

    # impact / whipping channel: the same question on the impact time scale
    beam = case["beam"]
    model = WagnerSlamming(rho=cfg.rho, deadrise_deg=cfg.deadrise_deg, keel_depth=case["barge"].draught,
                           decay_length=0.9, max_half_width=0.45 * case["barge"].breadth,
                           max_patch_half_length=0.25 * case["barge"].length)
    t_imp = np.linspace(0.0, 4.0 * face_c["impact_duration_s"], 201)
    v_imp = np.where(t_imp <= face_c["impact_duration_s"], face_c["impact_velocity_max_m_s"], 0.0)
    c_imp = model.contact_half_width(t_imp, v_imp)
    beam_x = beam.x + 0.5 * case["barge"].length
    xi = np.linspace(-model.max_half_width, model.max_half_width, 121)
    line_ref = np.array(
        [
            np.trapezoid(model.field(beam_x, xi, t, v, c, face_c["impact_station_x_m"]), xi, axis=1)
            for t, v, c in zip(t_imp, v_imp, c_imp)
        ]
    )                                              # (nt, nx)
    Q_ref = np.array([beam.project_line_load(row) for row in line_ref]).T   # (n_modes, nt)
    q_ref = duhamel_response(t_imp, Q_ref, beam.omega, beam.zeta)
    stress_ref = beam.stress_history(q_ref)
    peak_ref = float(np.max(np.abs(stress_ref)))
    l2_ref = float(np.sqrt(np.trapezoid(stress_ref ** 2, t_imp)))
    d_imp = float(t_imp[1] - t_imp[0])

    def impact_metric(dt_h: float, order: int, offset: float) -> dict:
        t_c = np.arange(0.0, t_imp[-1] + 0.5 * dt_h, dt_h)
        line_src = np.array(
            [
                np.trapezoid(
                    model.field(beam_x, xi, t, float(np.interp(t, t_imp, v_imp)),
                                float(np.interp(t, t_imp, c_imp)), face_c["impact_station_x_m"]),
                    xi, axis=1,
                )
                for t in t_c
            ]
        )
        line_del = reconstruct(t_c, line_src, t_imp, order=order, offset=offset)
        Q = np.array([beam.project_line_load(row) for row in line_del]).T
        q = duhamel_response(t_imp, Q, beam.omega, beam.zeta)
        stress = beam.stress_history(q)
        peak = float(np.max(np.abs(stress)))
        l2 = float(np.sqrt(np.trapezoid(stress ** 2, t_imp)))
        return {
            "peak_stress_rel_err": float(abs(peak - peak_ref) / max(peak_ref, 1e-30)),
            "stress_l2_rel_err": float(abs(l2 - l2_ref) / max(l2_ref, 1e-30)),
            "impulse_rel_err": float(
                abs(np.trapezoid(np.trapezoid(line_del, beam_x, axis=1), t_imp)
                    - np.trapezoid(np.trapezoid(line_ref, beam_x, axis=1), t_imp))
                / max(abs(np.trapezoid(np.trapezoid(line_ref, beam_x, axis=1), t_imp)), 1e-30)
            ),
            "peak_stress_Pa": peak,
        }

    impact_rows = footprint(
        impact_metric,
        period=face_c["impact_duration_s"],
        densities=(cfg.impact_samples, 2 * cfg.impact_samples, 4 * cfg.impact_samples),
        orders=(0, 1, 2),
        offset_fractions=(0.0, 0.5),
    )["rows"]

    fine = cfg.fine_samples_per_period
    coarse = cfg.coarse_samples_per_period
    zero_off = next(
        r for r in wave_rows
        if r["samples_per_period"] == fine and r["order"] == 2 and r["offset_fraction"] == 0.0
    )
    # detector 1: half-step offset on the wave-band energy ledger at the
    # production coupling rate (the sharpest available meter: the net work of a
    # lightly damped system is a small residual of large oscillating powers, so
    # its sensitivity to a phase error is amplified by ~1/cos(phase lag))
    offset_case = next(
        r for r in wave_rows
        if r["samples_per_period"] == fine and r["order"] == 2 and r["offset_fraction"] == 0.5
    )
    # detector 2: zero-order (piecewise constant) hand-off at the cheap
    # campaign rate -- the force itself is already in error by >10 %
    zero_case = next(
        r for r in wave_rows
        if r["samples_per_period"] == 2 * coarse and r["order"] == 0 and r["offset_fraction"] == 0.0
    )
    impact_fine = next(
        r for r in impact_rows
        if r["samples_per_period"] == 4 * cfg.impact_samples and r["order"] == 2 and r["offset_fraction"] == 0.0
    )
    impact_offset = next(
        r for r in impact_rows
        if r["samples_per_period"] == cfg.impact_samples and r["order"] == 2 and r["offset_fraction"] == 0.5
    )
    det_offset = detection(wave_rows, "wave_ledger_rel_err", 0.10)
    det_zero = detection(wave_rows, "force_max_rel_err", 0.10)

    return {
        "zero_offset_rel_err": float(zero_off["wave_ledger_rel_err"]),
        "offset_residual": float(offset_case["wave_ledger_rel_err"]),
        "zero_order_force_err": float(zero_case["force_max_rel_err"]),
        "offset_detected": 1.0 if offset_case["wave_ledger_rel_err"] > 0.10 else 0.0,
        "zero_order_detected": 1.0 if zero_case["force_max_rel_err"] > 0.10 else 0.0,
        "offset_force_rel_err": float(offset_case["force_max_rel_err"]),
        "zero_order_ledger_rel_err": float(zero_case["wave_ledger_rel_err"]),
        "impact_band": {
            "l2_rel_err_operating": float(impact_fine["stress_l2_rel_err"]),
            "l2_rel_err_offset": float(impact_offset["stress_l2_rel_err"]),
            "peak_stress_rel_err_operating": float(impact_fine["peak_stress_rel_err"]),
            "samples_per_impact_operating": int(4 * cfg.impact_samples),
        },
        "wave_rows": wave_rows,
        "impact_rows": impact_rows,
        "detection": {"offset_wave_band": det_offset, "zero_order_wave_band": det_zero},
        "operating_points": {
            "production": {"samples_per_period": fine, "order": 2, "offset": 0.0},
            "offset_negative_control": {"samples_per_period": fine, "order": 2, "offset": "dt/2"},
            "zero_order_negative_control": {"samples_per_period": 2 * coarse, "order": 0},
            "impact_time_band": {"samples_per_impact": 4 * cfg.impact_samples, "order": 2},
        },
        "peak_stress_reference_Pa": peak_ref,
        "impact_dt_fine_s": d_imp,
    }


# --------------------------------------------------------------------------
# negative controls (N)
# --------------------------------------------------------------------------
def _negatives(cfg: ChainConfig, case: dict) -> dict:
    dyn = case["dynamics"]
    mass = dyn.body.mass_matrix
    added = dyn.added_mass

    problems = []
    for scale in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
        f = np.array([5000.0, 0.0, 0.0, 0.0, 1.0e5, 0.0]) * scale
        problems.append(
            InterfaceProblem(
                structural_mass=np.diag([mass[2, 2], mass[4, 4]]),
                added_mass=np.diag([added[2, 2], added[4, 4]]),
                external_force=np.array([f[2], f[4]]),
            )
        )

    none = divergence_detection_rate(problems, scheme="none", max_iterations=60)
    aitken = divergence_detection_rate(problems, scheme="aitken", max_iterations=60)
    iqn = divergence_detection_rate(problems, scheme="iqn-ils", max_iterations=60)

    stabilised = []
    for p in problems:
        ok = True
        for scheme in ("aitken", "iqn-ils"):
            res = solve_interface(p, scheme=scheme, max_iterations=60, tol=1e-6)
            ok = ok and res.converged and res.final_residual <= 1e-6
        stabilised.append(bool(ok))

    return {
        "divergence_detection_rate": float(none["detection_rate"]),
        "stabilised_converged_rate": float(np.mean(stabilised)),
        "spectral_radius": float(spectral_radius(problems[-1])),
        "mass_ratio": float(case["barge"].mass / dyn.body.displacement),
        "added_mass_ratio": float(mass[2, 2] / added[2, 2]),
        "unstable_spectrum": [float(spectral_radius(p)) for p in problems],
        "none": none,
        "aitken": aitken,
        "iqn_ils": iqn,
    }


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------
def run_chain_case(cfg: ChainConfig | None = None, outdir: str | Path | None = None) -> dict:
    """Run the whole chain and return ``(detail, readiness)``."""
    cfg = cfg or ChainConfig()
    case = _build_case(cfg)
    face_a = _face_a(cfg, case)
    face_b = _face_b(cfg, case)
    face_c = _face_c(cfg, case)
    energy = _energy(cfg, case)
    time_led = _time_ledger(cfg, case, face_c)
    negatives = _negatives(cfg, case)

    products = {
        "A": {
            "force_rel_err": face_a["force_rel_err"],
            "observed_order": face_a["observed_order"],
            "reference_self_consistency": face_a["reference_self_consistency"],
        },
        "B": {
            "closed_flux_rel": face_b["closed_flux_rel"],
            "volume_rel_err": face_b["volume_rel_err"],
            "normal_velocity_rel_err": face_b["normal_velocity_rel_err"],
            "power_rel_err": face_b["power_rel_err"],
        },
        "C": {
            "modal_force_rel_err": face_c["modal_force_rel_err"],
            "duhamel_rel_err": face_c["duhamel_rel_err"],
        },
        "E": {
            "wave_ledger_rel_err": energy["wave_ledger_rel_err"],
            "body_ledger_rel_err": energy["body_ledger_rel_err"],
        },
        "T": {
            "zero_offset_rel_err": time_led["zero_offset_rel_err"],
            "offset_residual": time_led["offset_residual"],
            "zero_order_force_err": time_led["zero_order_force_err"],
            "offset_detected": time_led["offset_detected"],
            "zero_order_detected": time_led["zero_order_detected"],
        },
        "N": {
            "divergence_detection_rate": negatives["divergence_detection_rate"],
            "stabilised_converged_rate": negatives["stabilised_converged_rate"],
        },
    }
    verdict = evaluate_chain(products)

    detail = {
        "schema": "chain_ledger_detail.v1",
        "config": {
            "wave_height_m": cfg.wave_height,
            "wave_period_s": cfg.wave_period,
            "depth_m": cfg.depth,
            "rho_kg_m3": cfg.rho,
            "deadrise_deg": cfg.deadrise_deg,
            "heave_damping_ratio": cfg.heave_damping_ratio,
            "pitch_damping_ratio": cfg.pitch_damping_ratio,
            "n_modes": cfg.n_modes,
            "plunge_velocity_m_s": cfg.plunge_velocity,
            "fine_samples_per_period": cfg.fine_samples_per_period,
            "coarse_samples_per_period": cfg.coarse_samples_per_period,
            "impact_samples": cfg.impact_samples,
            "dt_march_s": cfg.dt_march,
        },
        "geometry": {
            "length_m": case["barge"].length,
            "breadth_m": case["barge"].breadth,
            "draught_m": case["barge"].draught,
            "depth_m": case["barge"].depth,
            "mass_kg": case["barge"].mass,
            "displacement_kg": case["body"].displacement,
            "mass_ratio": float(case["barge"].mass / case["body"].displacement),
            "ei_Nm2": case["barge"].ei,
            "section_modulus_m3": case["barge"].section_modulus,
            "waterplane_area_m2": case["body"].waterplane_area,
        },
        "wave": {
            "wavenumber_1_m": case["wave"].wavenumber,
            "wavelength_m": case["wave"].wavelength,
            "celerity_m_s": case["wave"].celerity,
            "group_velocity_m_s": case["wave"].group_velocity,
            "steepness": case["wave"].steepness,
            "energy_flux_W_m": case["wave"].energy_flux,
            "numerical_energy_flux_W_m": case["wave"].numerical_energy_flux(0.0),
            "dispersion_residual": case["wave"].dispersion_residual(),
            "second_harmonic_amplitude_m": case["wave"].second_harmonic_amplitude,
        },
        "hydrodynamics": {
            "sectional_added_mass_kg_m": case["a33_per_length"],
            "added_mass_matrix": case["added_mass"].tolist(),
            "effective_mass_matrix": case["dynamics"].effective_mass.tolist(),
            "damping_matrix": case["damping"].tolist(),
            "restoring_matrix": case["restoring"].tolist(),
            "natural_omega_rad_s": case["natural_omega"],
        },
        "beam": case["beam"].summary(),
        "faces": {"A": face_a, "B": face_b, "C": face_c, "E": energy, "T": time_led, "N": negatives},
        "products": products,
    }
    detail["readiness"] = verdict
    if outdir is not None:
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "chain_ledger.json").write_text(
            json.dumps(detail, indent=2, default=_json_default, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        write_readiness(verdict, out / "chain_readiness.json")
        (out / "products.json").write_text(
            json.dumps(products, indent=2, default=_json_default) + "\n", encoding="utf-8"
        )
    return detail


def _json_default(obj):
    if isinstance(obj, complex):
        return {"re": obj.real, "im": obj.imag}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.complexfloating,)):
        return {"re": float(obj.real), "im": float(obj.imag)}
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="run the P35-N10 three-stage chain ledger")
    parser.add_argument("--out", default="artifacts/chain_ledger", help="output directory")
    args = parser.parse_args(argv)
    detail = run_chain_case(ChainConfig(), outdir=args.out)
    verdict = detail["readiness"]
    print(json.dumps(
        {
            "overall_status": verdict["overall_status"],
            "blocking": verdict["blocking_criteria"],
            "products": detail["products"],
        },
        indent=2,
        default=_json_default,
    ))
    return 0 if verdict["overall_status"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
