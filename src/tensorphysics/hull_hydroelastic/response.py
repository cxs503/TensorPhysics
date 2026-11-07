"""Wave-excited springing, slamming whipping and rigid/elastic coupling.

Excitation model (strip theory, Smith-corrected)
------------------------------------------------
The sectional vertical wave force per unit ocean-wave amplitude is taken in the
standard strip form used for hull-girder vibration work,

    f3(x, omega) = rho g exp(-k T) [ b(x) + k a33(x) ],     k = omega^2 / g,

i.e. Froude-Krylov on the local waterplane width plus the diffraction term from
the sectional infinite-frequency added mass, both Smith-corrected for the local
draught.  The generalised modal force is ``Q_n = Int f3 phi_n dx``.

This is a **strip** model: it ignores three-dimensional (forward-speed)
diffraction, the wave-induced unsteady lift and the phase lag around the hull.
Its evidence grade is therefore APPROXIMATE-STRIP, and the report quantifies the
strip-vs-3D question qualitatively (no 3-D hydroelastic solver was coupled).

Springing
---------
Stress response spectral density at a hot spot (deck or keel, chosen as the node
of maximum wet-mode curvature)::

    S_sigma(omega) = |Sum_n s_n h_n(omega) Q_n(omega)|^2 S_zeta(omega),
    h_n = 1 / (omega_n^2 - omega^2 + 2 i zeta_n omega_n omega),
    s_n = E (D/2) phi_n''(x_hot),

with ``phi_n`` mass normalised on the *wet* generalised mass and ``S_zeta`` the
ITTC-1978 two-parameter spectrum.  Modes are summed coherently (standard modal
superposition for proportional damping); cross-modal correlation of the
excitation is thereby retained.

Whipping
--------
A slamming event is a half-sine pressure pulse of peak ``p0`` acting on a bow
patch.  The modal response is obtained by exact convolution of the modal
impulse response with the modal force (no numerical damping of the transient).
The resulting peak stress is the whipping stress; fatigue is evaluated with the
DNV-RP-C203 curve D in air (``m = 3``, ``log10 a = 12.164``) using the decaying
narrow-band envelope of the transient.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import fftconvolve

from ..hullgirder.model import GRAVITY, RHO_WATER, ShipGirder
from ..hullgirder.ultimate import ultimate_bending_moment
from ..hullgirder.wave import ittc_spectrum, moment_rms_from_spectrum, shear_and_moment
from .wet import YOUNG_STEEL, WetModes

__all__ = [
    "encounter_frequency",
    "balanced_load",
    "rigid_body_moment_response",
    "SN_CURVE_D_LOG_A",
    "SN_CURVE_D_M",
    "EXTREME_FACTOR_3H",
    "sectional_excitation",
    "generalised_force",
    "springing_stress",
    "slamming_force",
    "whipping_response",
    "whipping_fatigue",
    "springing_fatigue_rate",
    "coupling_table",
    "utilisation_gate",
]

# DNV-RP-C203 curve D in air (m = 3, log10 a = 12.164 for N <= 1e7)
SN_CURVE_D_LOG_A = 12.164
SN_CURVE_D_M = 3.0

#: 3-hour extreme factor of a narrow-band Gaussian process (1080 cycles)
EXTREME_FACTOR_3H = 3.7


def sectional_excitation(x: np.ndarray, omega: float, a33: np.ndarray, ship: ShipGirder) -> np.ndarray:
    """Strip sectional vertical force per unit wave amplitude (N/m per m)."""
    if omega <= 0.0:
        raise ValueError("omega must be positive")
    k = omega ** 2 / GRAVITY
    b = ship.waterplane_width(x)
    a33_nd = np.asarray(a33, dtype=float) / RHO_WATER
    return RHO_WATER * GRAVITY * np.exp(-k * ship.draught) * (b + k * a33_nd)


def generalised_force(
    x: np.ndarray, omega: float, a33: np.ndarray, ship: ShipGirder, shapes: np.ndarray
) -> np.ndarray:
    f = sectional_excitation(x, omega, a33, ship)
    return np.array([float(np.trapezoid(f * phi, x)) for phi in shapes])


def encounter_frequency(omega: np.ndarray, speed: float, heading_deg: float = 180.0) -> tuple:
    """Deep-water encounter frequency and its Jacobian.

    ``omega_e = omega - k U cos(mu)`` with ``mu`` the angle between the ship's
    heading and the wave propagation direction (``mu = 180 deg`` = head seas, so
    ``omega_e = omega + k U``).  Springing is an encounter-frequency phenomenon:
    the hull-girder mode (0.3-1 Hz) is excited by *low* wave frequencies whose
    encounter frequency has been shifted up to the resonance by the forward
    speed, which is also why the effect grows with Froude number.
    """
    omega = np.asarray(omega, dtype=float)
    k = omega ** 2 / GRAVITY
    c = np.cos(np.deg2rad(heading_deg))
    omega_e = omega - k * speed * c
    jac = 1.0 - 2.0 * omega * speed * c / GRAVITY
    return omega_e, jac


def balanced_load(x: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Remove the rigid (heave/pitch) residual of a load distribution.

    A free-free beam only carries bending from the self-equilibrated part of a
    load, and double integration amplifies the residual force by ``L^2/8``.  The
    linear part ``a + b x`` that cancels the resultant force and moment is
    removed exactly (in the same trapezoidal sense used by the bending
    integration), which is what keeps the numerical RAO at the analytic
    long-wave limit instead of at a discretisation artefact.
    """
    q = np.asarray(q, dtype=float)
    w = np.gradient(x)
    s0 = float((w).sum())
    s1 = float((x * w).sum())
    s2 = float((x * x * w).sum())
    r0 = float((q * w).sum())
    r1 = float((q * x * w).sum())
    det = s0 * s2 - s1 * s1
    if abs(det) < 1e-30:  # pragma: no cover - guard
        return q
    a = (r0 * s2 - r1 * s1) / det
    b = (s0 * r1 - s1 * r0) / det
    return q - (a + b * x)


def rigid_body_moment_response(
    ship: ShipGirder,
    x: np.ndarray,
    a33: np.ndarray,
    hs: float,
    t1: float,
    omega: np.ndarray | None = None,
    speed: float = 0.0,
    heading_deg: float = 180.0,
) -> dict:
    """Rigid-hull (decoupled) vertical wave bending moment - same hydrodynamics.

    The 2-DOF heave/pitch system is solved with the *same* strip sectional
    excitation as the elastic model (Froude-Krylov + diffraction, Smith
    corrected, added mass from the 2-D boundary-value problem), so decoupled and
    coupled results differ only by hull-girder flexibility.  Radiation damping
    is set to zero (not modelled anywhere in this node - stated limitation), so
    the rigid response is real-part only and its resonances are unbounded; the
    moment RMS is dominated by the wave band and is insensitive to that.

    The distributed load uses the standard frequency-domain decomposition

        q(x) = omega^2 m d + (rho g b + omega^2 a33) (d - zeta_FK),

    with ``d = eta + theta x`` the rigid heave/pitch response and
    ``zeta_FK = exp(-k T) cos(k x)`` the Smith-corrected incident wave.
    """
    omega = np.linspace(0.05, 2.5, 200) if omega is None else np.asarray(omega, dtype=float)
    b = ship.waterplane_width(x)
    m = ship.mass_per_length(x)
    a33 = np.asarray(a33, dtype=float)
    w = np.gradient(x)
    m33 = float((m * w).sum())
    m35 = float((m * x * w).sum())
    m55 = float((m * x * x * w).sum())
    a33i = float((a33 * w).sum())
    a35 = float((a33 * x * w).sum())
    a55 = float((a33 * x * x * w).sum())
    c33 = RHO_WATER * GRAVITY * float((b * w).sum())
    c35 = RHO_WATER * GRAVITY * float((b * x * w).sum())
    c55 = RHO_WATER * GRAVITY * float((b * x * x * w).sum())
    m_mid = x.size // 2
    moments = np.zeros(omega.size)
    spectrum = ittc_spectrum(omega, hs, t1)
    omega_e, jac = encounter_frequency(omega, speed, heading_deg)
    for i, om in enumerate(omega):
        k = om ** 2 / GRAVITY
        ome = float(omega_e[i])
        zeta_fk = np.exp(-k * ship.draught) * np.cos(k * x)
        f3 = RHO_WATER * GRAVITY * np.exp(-k * ship.draught) * (b + k * a33 / RHO_WATER) * np.cos(k * x)
        f_exc = f3  # (rho g b + omega^2 a33) zeta_FK
        F3 = float((f_exc * w).sum())
        F5 = float((f_exc * x * w).sum())
        # [omega_e^2 (M + A) + C] d = (C + omega^2 A) zeta_FK (relative-motion form)
        K = np.array(
            [
                [ome ** 2 * (m33 + a33i) + c33, ome ** 2 * (m35 + a35) + c35],
                [ome ** 2 * (m35 + a35) + c35, ome ** 2 * (m55 + a55) + c55],
            ]
        )
        try:
            eta, theta = np.linalg.solve(K, np.array([F3, F5]))
        except np.linalg.LinAlgError:  # pragma: no cover - degenerate
            eta, theta = 0.0, 0.0
        d = eta + theta * x
        # net downward load per unit length (positive = hogging-type downward)
        q = ome ** 2 * m * d + (RHO_WATER * GRAVITY * b + om ** 2 * a33) * (d - zeta_fk)
        moments[i] = float(shear_and_moment(x, balanced_load(x, q), support="free")[1][m_mid])
    dens = np.abs(spectrum / np.where(np.abs(jac) < 1e-6, 1e-6, jac))
    var = float(np.trapezoid(moments ** 2 * dens, omega))
    return {
        "moment_rms_Nm": float(np.sqrt(max(var, 0.0))),
        "stress_rms_Pa": float(np.sqrt(max(var, 0.0)) / ship.section_modulus),
        "omega": omega.tolist(),
        "spectrum_m2s": spectrum.tolist(),
        "moment_rao_Nm_per_m": moments.tolist(),
        "hs_m": float(hs),
        "t1_s": float(t1),
        "speed_ms": float(speed),
        "heading_deg": float(heading_deg),
        "heave_pitch_stiffness_N_per_m": c33,
    }


def _omega_grid(wet_omega: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    """Composite frequency grid: dense at every resonance, coarse elsewhere."""
    base = np.linspace(0.02, 12.0, 700)
    parts = [base]
    for w, z in zip(wet_omega, zeta):
        width = max(40.0 * z * w, 0.05)
        parts.append(np.linspace(max(w - width, 0.02), w + width, 220))
    return np.unique(np.concatenate(parts))


def _excitation_matrix(
    ship: ShipGirder, omega: np.ndarray, a33: np.ndarray, shapes: np.ndarray, x: np.ndarray
) -> np.ndarray:
    """``Q[n, k] = Int f3(x, omega_k) phi_n dx`` (generalised force per wave amplitude)."""
    k = omega ** 2 / GRAVITY
    b = ship.waterplane_width(x)
    # a33 is a *mass* per unit length (kg/m); the strip formula works with the
    # non-dimensional added mass per unit length a33/rho (m^2) so that the
    # diffraction term equals the added-mass force a33 * (omega^2 zeta_a).
    a33_nd = np.asarray(a33, dtype=float) / RHO_WATER
    # the incident wave phase cos(k x) is essential: without it the short waves
    # that are shifted onto the hull-girder mode by the encounter frequency
    # would be applied coherently over the whole length and the resonant
    # excitation would be over-predicted by two orders of magnitude.
    phase = np.cos(np.outer(k, x))
    f3 = (
        RHO_WATER
        * GRAVITY
        * np.exp(-k[:, None] * ship.draught)
        * (b[None, :] + k[:, None] * a33_nd[None, :])
        * phase
    )
    weights = np.gradient(x)
    return (shapes * weights[None, :]) @ f3.T


def springing_stress(
    ship: ShipGirder,
    wet: WetModes,
    hs: float,
    t1: float,
    n_modes: int | None = None,
    zeta: float | np.ndarray | None = None,
    hot_spot: str = "deck",
    speed: float = 0.0,
    heading_deg: float = 180.0,
) -> dict:
    """RMS wave-induced + springing stress at the midship deck (Pa).

    The response is solved in the dry modal basis with the added mass kept as a
    matrix, so the static limit (``omega -> 0``) reduces to the dry static
    flexibility and the added mass only acts through ``omega^2``::

        [K + G - omega^2 (I + A) + 2 i omega diag(zeta_n omega_n)] q = Q(omega)

    with ``K = diag(omega_dry^2)``, ``G`` the buoyancy restoring stiffness
    ``Int rho g b phi_n phi_m dx`` (a free-floating hull has one) and ``A`` the
    added-mass matrix.  ``n_modes`` truncates the modal sum (convergence gate).
    """
    n_modes = int(n_modes or wet.n_modes)
    x = wet.x
    if zeta is None:
        zeta_arr = wet.zeta[:n_modes]
    elif np.isscalar(zeta):
        zeta_arr = np.full(n_modes, float(zeta))
    else:
        zeta_arr = np.asarray(zeta, dtype=float)[:n_modes]
    if hot_spot not in ("deck", "keel"):
        raise ValueError("hot_spot must be 'deck' or 'keel'")

    # hot spot: deck at midship - the section modulus is a midship property, so
    # the stress and the utilisation ledger refer to the same section.
    i_hot = int(np.argmin(np.abs(x)))
    lever = 0.5 * ship.depth
    dry_shapes = wet.dry.shapes[:n_modes]
    s_dry = lever * YOUNG_STEEL * wet.dry.curvature[:n_modes, i_hot]
    if hot_spot == "keel":
        s_dry = -s_dry
    K = wet.modal_stiffness[:n_modes, :n_modes]
    M = wet.modal_mass[:n_modes, :n_modes]
    omega_n = wet.wet_omega[:n_modes]

    omega = _omega_grid(omega_n, zeta_arr)
    spectrum = ittc_spectrum(omega, hs, t1)
    Q = _excitation_matrix(ship, omega, wet.a33, dry_shapes, x)
    omega_e, jac = encounter_frequency(omega, speed, heading_deg)
    dens = np.abs(spectrum / np.where(np.abs(jac) < 1e-6, 1e-6, jac))
    stress_tf = np.zeros(omega.size, dtype=complex)
    per_mode_spec = np.zeros((n_modes, omega.size))
    for i in range(omega.size):
        om = float(omega_e[i])
        dyn = K - om ** 2 * M + 1j * np.diag(2.0 * zeta_arr * omega_n * om)
        try:
            q = np.linalg.solve(dyn, Q[:, i])
        except np.linalg.LinAlgError:  # pragma: no cover - guard
            q = np.zeros(n_modes, dtype=complex)
        stress_tf[i] = s_dry @ q
        per_mode_spec[:, i] = np.abs(s_dry * q) ** 2 * dens[i]
    s_sigma = np.abs(stress_tf) ** 2 * dens
    sigma_rms = float(np.sqrt(max(float(np.trapezoid(s_sigma, omega)), 0.0)))
    per_mode = np.sqrt(np.trapezoid(per_mode_spec, omega, axis=1))
    rigid = rigid_body_moment_response(ship, x, wet.a33, hs, t1, speed=speed, heading_deg=heading_deg)
    return {
        "sigma_rms_Pa": sigma_rms,
        "sigma_rms_MPa": sigma_rms / 1.0e6,
        "hot_spot": hot_spot,
        "hot_spot_x_m": float(x[i_hot]),
        "n_modes": n_modes,
        "zeta": [float(z) for z in zeta_arr],
        "mode_stress_per_unit_q_Pa": [float(s) for s in s_dry],
        "per_mode_rms_Pa": [float(v) for v in per_mode],
        "dominant_mode": int(np.argmax(per_mode) + 1),
        "omega": omega.tolist(),
        "spectrum_m2s": spectrum.tolist(),
        "stress_spectrum_Pa2s": s_sigma.tolist(),
        "rigid_moment_rms_Nm": rigid["moment_rms_Nm"],
        "rigid_stress_rms_Pa": rigid["stress_rms_Pa"],
        "platform_baseline_stress_rms_Pa": float(
            moment_rms_from_spectrum(ship, hs, t1) / ship.section_modulus
        ),
        "hs_m": float(hs),
        "t1_s": float(t1),
        "speed_ms": float(speed),
        "heading_deg": float(heading_deg),
    }


def slamming_force(
    t: np.ndarray,
    peak_force: float,
    duration: float,
    rise_fraction: float = 0.4,
) -> np.ndarray:
    """Asymmetric half-sine-ish impact pulse (rise ``rise_fraction d``, fall rest)."""
    t = np.asarray(t, dtype=float)
    force = np.zeros_like(t)
    rise = rise_fraction * duration
    up = (t >= 0.0) & (t < rise)
    down = (t >= rise) & (t < duration)
    force[up] = peak_force * np.sin(0.5 * np.pi * t[up] / rise)
    force[down] = peak_force * np.sin(
        0.5 * np.pi * (duration - t[down]) / (duration - rise)
    )
    return force


def whipping_response(
    ship: ShipGirder,
    wet: WetModes,
    peak_force: float = 5.0e6,
    patch_fraction: float = 0.10,
    duration: float = 0.25,
    n_modes: int | None = None,
    dt: float = 1.0e-3,
    t_total: float = 12.0,
    effective_width_fraction: float = 0.5,
    hot_spot: str = "deck",
    min_wavelength_over_depth: float | None = 4.0,
) -> dict:
    """Transient whipping response to one slamming event (modal superposition).

    ``min_wavelength_over_depth`` keeps only the modes whose vertical bending
    wavelength ``2 L / n`` exceeds that multiple of the hull depth.  The
    hull-girder *beam* idealisation is only meaningful while the bending
    wavelength is large compared with the depth (``4 D`` is used); the retained
    band is reported so the truncation is auditable.  Passing ``None`` keeps all
    modes, which yields the (unphysical) upper bound of the idealisation.
    """
    n_modes = int(n_modes or wet.n_modes)
    if min_wavelength_over_depth is not None:
        keep = [
            i
            for i in range(n_modes)
            if 2.0 * ship.length / (i + 1) >= min_wavelength_over_depth * ship.depth
        ]
        if not keep:  # pragma: no cover - guard
            keep = [0]
        n_modes = len(keep)
    else:
        keep = list(range(n_modes))
    x = wet.x
    shapes = wet.wet_shapes[keep]
    omega_n = wet.wet_omega[keep]
    zeta = wet.zeta[keep]
    i_hot = int(np.argmin(np.abs(x)))
    lever = 0.5 * ship.depth
    s_n = YOUNG_STEEL * lever * wet.wet_curvature[keep, i_hot]

    # bow patch: from the forward perpendicular aft over ``patch_fraction * L``
    patch_length = patch_fraction * ship.length
    half = ship.length / 2.0
    x_aft = half - patch_length
    in_patch = x >= x_aft
    # triangular distribution, unit integral over the patch
    w = np.where(in_patch, (x - x_aft), 0.0)
    integral = float(np.trapezoid(w, x))
    if integral <= 0.0:  # pragma: no cover - guard
        raise ValueError("empty slamming patch")
    w = w / integral
    width = effective_width_fraction * ship.breadth
    area = patch_length * width
    peak_pressure = peak_force / area

    t = np.arange(0.0, t_total, dt)
    force = slamming_force(t, peak_force, duration)
    stress = np.zeros_like(t)
    modal_response = np.zeros((n_modes, t.size))
    weights = np.gradient(x)
    shape_factor = (shapes * weights[None, :] * w[None, :]).sum(axis=1)  # Int w phi_n dx
    for n in range(n_modes):
        q = shape_factor[n] * force
        wd = omega_n[n] * np.sqrt(max(1.0 - zeta[n] ** 2, 1.0e-12))
        tau = t  # impulse response sampled on the same grid
        h = np.exp(-zeta[n] * omega_n[n] * tau) * np.sin(wd * tau) / wd
        resp = fftconvolve(q, h)[: t.size] * dt
        modal_response[n] = resp
        stress += s_n[n] * resp

    peak = float(np.max(np.abs(stress)))
    i_peak = int(np.argmax(np.abs(stress)))
    return {
        "t_s": t.tolist(),
        "stress_Pa": stress.tolist(),
        "peak_stress_Pa": peak,
        "peak_stress_MPa": peak / 1.0e6,
        "t_peak_s": float(t[i_peak]),
        "peak_force_N": float(peak_force),
        "peak_pressure_MPa": float(peak_pressure / 1.0e6),
        "patch_length_m": float(patch_length),
        "patch_area_m2": float(area),
        "impulse_Ns": float(np.trapezoid(force, t)),
        "duration_s": float(duration),
        "n_modes": n_modes,
        "modes_kept": [int(i + 1) for i in keep],
        "min_wavelength_over_depth": min_wavelength_over_depth,
        "hot_spot_x_m": float(x[i_hot]),
        "modal_peak_stress_Pa": [float(np.max(np.abs(s_n[n] * modal_response[n]))) for n in range(n_modes)],
        "dominant_mode": int(np.argmax([np.max(np.abs(s_n[n] * modal_response[n])) for n in range(n_modes)]) + 1),
        "modal_response": modal_response.tolist(),
    }


def whipping_fatigue(
    peak_stress: float,
    omega_2node: float,
    zeta: float,
    duration_factor: float = 1.0,
) -> dict:
    """S-N damage of one whipping event (DNV curve D in air, narrow-band envelope).

    The decaying transient is treated as a narrow-band process with envelope
    ``sigma_peak exp(-zeta omega t)``; with ``omega/(2 pi)`` cycles per second the
    Miner sum over the envelope becomes

        D_event = (2 sigma_peak)^m / (2 pi a m zeta),

    i.e. independent of the frequency except through ``omega`` in the envelope
    (the cycle count and the amplitude decay cancel exactly).
    """
    if peak_stress <= 0.0 or zeta <= 0.0 or omega_2node <= 0.0:
        raise ValueError("positive peak stress, damping and frequency required")
    sigma_mpa = peak_stress / 1.0e6
    a = 10.0 ** SN_CURVE_D_LOG_A
    m = SN_CURVE_D_M
    delta_sigma = 2.0 * sigma_mpa
    damage = (delta_sigma ** m) / (2.0 * np.pi * a * m * zeta) * duration_factor
    return {
        "damage_per_event": float(damage),
        "delta_sigma_MPa": float(delta_sigma),
        "sn_log_a": SN_CURVE_D_LOG_A,
        "sn_m": SN_CURVE_D_M,
        "zeta_used": float(zeta),
        "omega_2node_rad_s": float(omega_2node),
    }


def springing_fatigue_rate(sigma_rms: float, mean_zero_crossing: float) -> dict:
    """Narrow-band Rayleigh damage rate of the wave + springing stress (1/s)."""
    if sigma_rms <= 0.0:
        raise ValueError("sigma_rms must be positive")
    sigma_mpa = sigma_rms / 1.0e6
    a = 10.0 ** SN_CURVE_D_LOG_A
    m = SN_CURVE_D_M
    from math import gamma

    expected = (2.0 * np.sqrt(2.0) * sigma_mpa) ** m * gamma(1.0 + 0.5 * m)
    return {
        "damage_rate_per_s": float(mean_zero_crossing / (2.0 * np.pi) * expected / a),
        "stress_rms_MPa": float(sigma_mpa),
        "zero_crossing_rad_s": float(mean_zero_crossing),
    }


def coupling_table(
    ship: ShipGirder,
    wet: WetModes,
    hs: float,
    t1: float,
    froude_numbers: tuple[float, ...],
    slamming_force_at_ref: float = 5.0e6,
    ref_froude: float | None = None,
    patch_fraction: float = 0.10,
    duration: float = 0.25,
    hot_spot: str = "deck",
    zeta: float | None = None,
) -> dict:
    """Rigid (decoupled) vs flexible + slamming (coupled) midship bending moment.

    Both responses use the *same* strip hydrodynamics and the same encounter
    frequency, so the difference is the hull-girder flexibility (springing) plus
    the slamming transient - not a change of hydrodynamic model.

    ``decoupled`` = 3-hour extreme (3.7 sigma) of the rigid-hull wave moment.
    ``coupled``   = 3-hour extreme of the flexible-hull stress RMS mapped
    through the midship section modulus, plus the deterministic whipping peak.
    The slamming force follows the standard relative-velocity-squared scaling
    ``F(Fn) = F_ref (U / U_ref)^2``.
    """
    zeta_use = float(zeta if zeta is not None else wet.zeta[0])
    if ref_froude is None:
        ref_froude = float(froude_numbers[0])
    ref_speed = ref_froude * float(np.sqrt(GRAVITY * ship.length))
    rows = []
    for fn in froude_numbers:
        speed = fn * float(np.sqrt(GRAVITY * ship.length))
        spring = springing_stress(ship, wet, hs, t1, zeta=zeta_use, speed=speed)
        m_rigid = float(spring["rigid_moment_rms_Nm"])
        m_flex = float(spring["sigma_rms_Pa"] * ship.section_modulus)
        whip = whipping_response(
            ship,
            wet,
            peak_force=slamming_force_at_ref * (speed / ref_speed) ** 2,
            patch_fraction=patch_fraction,
            duration=duration,
            hot_spot=hot_spot,
        )
        m_whip = float(whip["peak_stress_Pa"] * ship.section_modulus)
        m_dec = EXTREME_FACTOR_3H * m_rigid
        m_coup = EXTREME_FACTOR_3H * m_flex + m_whip
        rows.append(
            {
                "Fn": float(fn),
                "speed_kn": float(speed / 0.514444),
                "slamming_force_MN": float(
                    slamming_force_at_ref * (speed / ref_speed) ** 2 / 1e6
                ),
                "M_rigid_rms_MNm": m_rigid / 1e6,
                "M_flexible_rms_MNm": m_flex / 1e6,
                "M_whipping_peak_MNm": m_whip / 1e6,
                "M_decoupled_extreme_MNm": m_dec / 1e6,
                "M_coupled_extreme_MNm": m_coup / 1e6,
                "delta_pct": float(100.0 * (m_coup / m_dec - 1.0)),
                "elastic_amplification": float(m_flex / m_rigid),
                "whipping_share_pct": float(100.0 * (m_whip / m_dec)),
                "sigma_rigid_rms_MPa": spring["rigid_stress_rms_Pa"] / 1e6,
                "sigma_flexible_rms_MPa": spring["sigma_rms_MPa"],
            }
        )
    return {
        "hs_m": float(hs),
        "t1_s": float(t1),
        "zeta": zeta_use,
        "rows": rows,
        "trend_increasing": bool(
            all(rows[i + 1]["delta_pct"] >= rows[i]["delta_pct"] for i in range(len(rows) - 1))
        ),
    }


def utilisation_gate(
    ship: ShipGirder,
    m_coupled_extreme: float,
    m_decoupled_extreme: float,
    gamma_m: float = 1.1,
) -> dict:
    """Elastic correction propagated through the Smith/section-modulus ledger.

    The platform ledger is ``utilisation = M / (M_u / gamma_m)`` with
    ``M_u = phi(beta) S_M sigma_y`` (:mod:`tensorphysics.hullgirder.ultimate`),
    so a moment increment must produce the *same* relative increment of the
    utilisation.  That proportionality is the acceptance gate; the binary
    PASS/FAIL answer reports whether the elastic correction flips the verdict.
    """
    cap = ultimate_bending_moment(ship)
    m_u = float(cap["M_ultimate_Nm"])
    capacity = m_u / gamma_m
    u_dec = abs(m_decoupled_extreme) / capacity
    u_coup = abs(m_coupled_extreme) / capacity
    m_inc = abs(m_coupled_extreme) / abs(m_decoupled_extreme) - 1.0
    u_inc = u_coup / u_dec - 1.0
    return {
        "M_ultimate_Nm": m_u,
        "M_capacity_factored_Nm": capacity,
        "utilisation_decoupled": float(u_dec),
        "utilisation_coupled": float(u_coup),
        "moment_increment_pct": float(100.0 * m_inc),
        "utilisation_increment_pct": float(100.0 * u_inc),
        "proportionality_error_pct": float(100.0 * abs(u_inc / m_inc - 1.0)) if m_inc else 0.0,
        "pass_decoupled": bool(u_dec <= 1.0),
        "pass_coupled": bool(u_coup <= 1.0),
        "elastic_correction_flips_verdict": bool((u_dec <= 1.0) != (u_coup <= 1.0)),
        "sigma_y_MPa": ship.yield_stress / 1e6,
        "section_modulus_m3": ship.section_modulus,
        "stress_decoupled_MPa": float(abs(m_decoupled_extreme) / ship.section_modulus / 1e6),
        "stress_coupled_MPa": float(abs(m_coupled_extreme) / ship.section_modulus / 1e6),
    }