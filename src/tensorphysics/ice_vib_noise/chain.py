"""Coupled ice-load -> hull vibration -> underwater radiated noise chain.

Chain stages (each one a physical computation, none a fitted curve):

1. **Ice load** — intermittent crushing series ``F_ice(t)`` and the bow contact
   distribution ``w(x)``; crushing energy ``E_crush = int F U dt``
   (:mod:`tensorphysics.ice_vib_noise.source`).
2. **Structural modal response** — generalised force ``Q_i = g_i F_ice``
   projected on the free-free hull-girder modes, integrated with the exact
   piecewise-linear scheme (:mod:`tensorphysics.ice_vib_noise.response`) with
   modal damping ``c_i = 2 zeta_i omega_i m_i + R_i(omega_i)``.
3. **Radiation** — the hull surface is a baffled planar radiator; the radiation
   resistance of every mode is computed from the Rayleigh far field on a
   frequency grid (:mod:`tensorphysics.ice_vib_noise.radiator`).  It supplies
   the (very small) radiation damping in the equation of motion and, together
   with the surface-velocity spectrum, the acoustic power spectrum.
4. **Spectra** — 1/3-octave RMS acceleration at the bow and at midship, and the
   1/3-octave source level ``SL`` from the far-field pressure at 1 m over a fan
   of directions (the per-band maximum is reported together with the
   directivity index against the angle-integrated power); band powers give the
   acoustic conversion efficiency ``E_rad / E_crush``.
5. **Propagation** — ``RL(f, r) = SL(f) - 20 log10 r - alpha(f) r/1000`` applied
   *per spectral line*, cross-checked against the band-centre approximation
   ``RL = SL - 20 log10 r - alpha(f_c) r`` (the difference is the in-band
   variation of the absorption).
6. **Criteria** (machine-evaluable, every number reported):

   * **C1 zero-load control** — ``F_ice = 0`` gives a response ``< 1e-9`` of the
     loaded response (relative): no spurious excitation anywhere in the chain.
   * **C2 modal orthogonality** — off-diagonal mass matrix ``< 1e-6``.
   * **C3 energy conservation** — ``W_in = dE_strain + dE_kin + E_struct +
     E_rad`` residual ``< 1%`` with the frequency-dependent (spectral) radiated
     energy, plus exact band/FFT Parseval closure ``< 1e-6``.
   * **C4 envelope consistency** — the chain must be *linear*: a load scaled by
     ``lambda`` moves every 1/3-octave level by exactly ``20 log10 lambda``
     (``< 1e-9`` dB).  That linearity is what makes the vibration -> noise
     envelope translation legitimate; the translated SPL envelope is reported
     band by band.  Whether the case satisfies the ISO-19906-informed vibration
     envelope is reported separately as the physical finding C4b.
   * **C5 idempotent reproduction** — two independent runs give identical
     canonical digests (:func:`run_with_idempotency`).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..ice_vibration.modal import HullParameters
from . import acoustics
from .bands import OneThirdOctaveBands, band_mean_square, one_third_octave_bands
from .envelope import band_margins, iso19906_informed_envelope
from .modal import HullGirderModes
from .radiator import P_REF_PA, RadiatorSurface
from .response import ModalResponse, solve_modal_exact
from .source import IceExcitationCase, bow_load_distribution, build_ice_excitation, crushing_energy_j

__all__ = ["CoupledCase", "run_coupled_chain", "run_with_idempotency", "canonical_digest"]

#: Seawater density and sound speed used for all acoustic bookkeeping.
RHO_WATER = 1025.0
C_WATER = 1500.0

#: Number of logarithmically spaced frequencies for the radiation resistance.
N_RESISTANCE_FREQUENCIES = 40

#: Assessment fan: polar angle from the hull-down normal and azimuth measured
#: from the bow direction (phi = 0 deg fore/aft, phi = 90 deg broadside).
ASSESSMENT_THETA_DEG: Tuple[float, ...] = (30.0, 45.0, 60.0, 75.0)
ASSESSMENT_PHI_DEG: Tuple[float, ...] = (0.0, 30.0, 60.0, 90.0)

#: Load scale factor used by the linearity criterion.
LINEARITY_SCALE = 1.37


@dataclass(frozen=True)
class CoupledCase:
    """Complete input set of the coupled ice-vibration / radiated-noise case."""

    hull: HullParameters
    bending_stiffness_nm2: float = 2.0e12
    n_modes: int = 3
    structural_damping: Tuple[float, ...] = (0.015, 0.015, 0.02)
    excitation: IceExcitationCase = field(default_factory=IceExcitationCase)
    excitation_mode: str = "crushing"
    radiator_nx: int = 48
    radiator_ny: int = 4
    band_min_hz: float = 0.5
    band_max_hz: float = 80.0
    acoustic_validity_factor: float = 2.0
    ranges_m: Tuple[float, ...] = (100.0, 1000.0, 10000.0)
    envelope_plateau_rms_m_s2: float = 0.05
    envelope_sensitivity: Tuple[float, ...] = (0.02, 0.05, 0.1)
    sensor_bow_x_m: Optional[float] = None
    sensor_mid_x_m: Optional[float] = None
    assessment_theta_deg: Tuple[float, ...] = ASSESSMENT_THETA_DEG
    assessment_phi_deg: Tuple[float, ...] = ASSESSMENT_PHI_DEG
    linearity_scale: float = LINEARITY_SCALE

    def __post_init__(self) -> None:
        if self.n_modes < 1:
            raise ValueError("n_modes must be >= 1")
        if len(self.structural_damping) < self.n_modes:
            raise ValueError("not enough structural damping ratios")
        for ang in self.assessment_theta_deg:
            if not 0.0 < ang < 90.0:
                raise ValueError("assessment theta must be strictly inside the hemisphere")
        if self.band_min_hz <= 0.0 or self.band_max_hz <= self.band_min_hz:
            raise ValueError("invalid band range")
        if self.acoustic_validity_factor <= 1.0:
            raise ValueError("acoustic_validity_factor must be > 1 (truncated modal basis)")
        if self.linearity_scale <= 0.0 or abs(self.linearity_scale - 1.0) < 1e-6:
            raise ValueError("linearity_scale must differ from 1")


# ---------------------------------------------------------------------------
# Spectral helpers
# ---------------------------------------------------------------------------


def _amplitude_spectrum(signal: np.ndarray, dt_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """One-sided *peak-amplitude* spectrum and the matching frequency axis."""
    n = signal.size
    spec = np.fft.rfft(signal)
    amp = 2.0 * spec / n
    amp[0] = spec[0] / n
    if n % 2 == 0:
        amp[-1] = spec[-1] / n
    return np.fft.rfftfreq(n, dt_s), amp


def _psd_from_amplitude(freqs: np.ndarray, amp: np.ndarray) -> np.ndarray:
    """One-sided PSD with ``int S df == mean square`` of the time history."""
    df = float(freqs[1] - freqs[0])
    return np.abs(amp) ** 2 / (2.0 * df)


def _band_mean_square(freqs: np.ndarray, amp: np.ndarray, bands: OneThirdOctaveBands) -> np.ndarray:
    return band_mean_square(freqs, _psd_from_amplitude(freqs, amp), bands)


def _db_pressure(mean_square_pa2: np.ndarray) -> np.ndarray:
    """Band mean-square pressure -> dB re 1 uPa."""
    return 10.0 * np.log10(np.maximum(mean_square_pa2, 1e-300) / P_REF_PA ** 2)


def canonical_digest(payload: Any, ndigits: int = 12) -> str:
    """Stable SHA-256 over a JSON payload with sorted keys and rounded floats."""

    def _walk(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _walk(obj[k]) for k in sorted(obj)}
        if isinstance(obj, (list, tuple)):
            return [_walk(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return [_walk(v) for v in obj.tolist()]
        if isinstance(obj, (np.floating, float)):
            v = round(float(obj), ndigits)
            return 0.0 if v == 0.0 else v
        if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
            return int(obj)
        return obj

    blob = json.dumps(_walk(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Physical core
# ---------------------------------------------------------------------------


def _run_core(case: CoupledCase) -> Dict[str, Any]:
    """Run the physical chain once (no criteria, no cross-checks)."""
    hull = case.hull
    exc = case.excitation
    dt = float(exc.dt_s)

    modes = HullGirderModes(
        hull=hull,
        bending_stiffness_nm2=case.bending_stiffness_nm2,
        n_modes=case.n_modes,
        damping_ratios=case.structural_damping,
    )
    x_grid = modes.grid_x_m
    omegas = np.array(modes.omegas_rad_s)
    masses = np.array(modes.modal_masses_kg)
    zetas = np.array(modes.zetas)
    k_modal = masses * omegas ** 2

    load = build_ice_excitation(exc, mode=case.excitation_mode)
    t = np.asarray(load.time_s, dtype=float)
    force = np.asarray(load.force_n, dtype=float)
    if t.size < 32:
        raise ValueError("load series too short for spectral analysis")
    w_x = bow_load_distribution(x_grid, hull.length_m, exc.bow_contact_length_m)
    coupling = np.array(
        [float(np.trapezoid(w_x * modes.mode_shape(i), x_grid)) for i in range(case.n_modes)]
    )
    Q = coupling[:, None] * force[None, :]
    e_crush = crushing_energy_j(force, t, exc.drift_speed_m_s)
    duration = float(t[-1] - t[0]) + dt

    # --- radiation resistance on a frequency grid ---------------------------
    radiator = RadiatorSurface.from_grid(
        length_m=hull.length_m,
        width_m=hull.breadth_m,
        n_x=case.radiator_nx,
        n_y=case.radiator_ny,
        fluid_density_kg_m3=RHO_WATER,
        sound_speed_m_s=C_WATER,
    )
    patch_phi = np.array([modes.mode_shape(i, radiator.x_m) for i in range(case.n_modes)])
    f_highest_mode = float(max(modes.frequencies_hz))
    f_valid = min(case.band_max_hz, case.acoustic_validity_factor * f_highest_mode)
    if f_valid <= case.band_min_hz:
        raise ValueError("acoustic validity band is empty; raise acoustic_validity_factor")
    f_grid = np.logspace(
        math.log10(max(0.25 * case.band_min_hz, 0.05)),
        math.log10(1.25 * f_valid),
        N_RESISTANCE_FREQUENCIES,
    )
    omega_grid = 2.0 * math.pi * f_grid
    resistance_grid = np.empty((case.n_modes, f_grid.size))
    for i in range(case.n_modes):
        resistance_grid[i] = radiator.radiated_power(
            np.tile(patch_phi[i], (f_grid.size, 1)), omega_grid, n_theta=20, n_phi=12
        )

    def resistance_at(mode: int, freq_hz) -> np.ndarray:
        """Log-log interpolation of ``R_i(f)`` (no linear extrapolation)."""
        return np.exp(
            np.interp(
                np.log(np.maximum(np.asarray(freq_hz, dtype=float), 1e-9)),
                np.log(f_grid),
                np.log(np.maximum(resistance_grid[mode], 1e-30)),
            )
        )

    r_rad = np.array(
        [float(resistance_at(i, np.array([modes.frequencies_hz[i]]))[0]) for i in range(case.n_modes)]
    )
    c_struct = 2.0 * zetas * omegas * masses
    zeta_rad = r_rad / (2.0 * omegas * masses)

    # --- structural response ------------------------------------------------
    q, qdot, qddot = solve_modal_exact(Q, dt, omegas, zetas, masses, c_rad=r_rad)
    ledger = ModalResponse(
        time_s=t,
        q=q,
        qdot=qdot,
        qddot=qddot,
        force_generalized_n=Q,
        modal_masses_kg=tuple(masses),
        omegas_rad_s=tuple(omegas),
        damping_total=tuple(c_struct + r_rad),
        damping_structural=tuple(c_struct),
        damping_radiation=tuple(r_rad),
        stiffness_n_m=tuple(k_modal),
    ).energy_ledger()
    q_zero, _, _ = solve_modal_exact(np.zeros_like(Q), dt, omegas, zetas, masses, c_rad=r_rad)

    # --- 1/3-octave spectra -------------------------------------------------
    bands = one_third_octave_bands(case.band_min_hz, min(f_valid, 0.45 / dt))
    centres = np.array(bands.centre_hz)
    widths = np.array(bands.upper_hz) - np.array(bands.lower_hz)
    x_bow = float(hull.length_m if case.sensor_bow_x_m is None else case.sensor_bow_x_m)
    x_mid = float(0.5 * hull.length_m if case.sensor_mid_x_m is None else case.sensor_mid_x_m)
    sensors = {"bow": x_bow, "midship": x_mid}
    phi_sensors = {
        name: np.array([modes.mode_shape(i, [xx])[0] for i in range(case.n_modes)])
        for name, xx in sensors.items()
    }
    # Restrict the spectral work to the band support (the bands do not cover the
    # whole Nyquist range; slicing keeps the Rayleigh fan affordable).
    freqs_full = np.fft.rfftfreq(t.size, dt)
    sel = np.where(
        (freqs_full >= 0.95 * bands.lower_hz[0]) & (freqs_full <= 1.05 * bands.upper_hz[-1])
    )[0]
    if sel.size < 8:
        raise ValueError("band support does not resolve enough spectral lines")
    freqs = freqs_full[sel]

    accel_bands: Dict[str, np.ndarray] = {}
    for name in sensors:
        a_t = phi_sensors[name] @ qddot
        _, amp = _amplitude_spectrum(a_t, dt)
        accel_bands[name] = np.sqrt(np.maximum(_band_mean_square(freqs, amp[sel], bands), 0.0))

    vq_amp = np.array([_amplitude_spectrum(qdot[i], dt)[1][sel] for i in range(case.n_modes)])
    psd_q = np.array([_psd_from_amplitude(freqs, vq_amp[i]) for i in range(case.n_modes)])

    parseval_rel = 0.0
    coverage_share = []
    for i in range(case.n_modes):
        band_ms = band_mean_square(freqs, psd_q[i], bands)
        mean_sq = float(np.trapezoid(qdot[i] ** 2, t)) / duration
        coverage_share.append(float(band_ms.sum()) / max(mean_sq, 1e-300))
        cum = np.concatenate(
            ([0.0], np.cumsum(0.5 * (psd_q[i][1:] + psd_q[i][:-1]) * np.diff(freqs)))
        )
        lo_c, hi_c = max(bands.lower_hz[0], float(freqs[0])), min(bands.upper_hz[-1], float(freqs[-1]))
        ref = float(np.interp(hi_c, freqs, cum) - np.interp(lo_c, freqs, cum))
        parseval_rel = max(parseval_rel, abs(float(band_ms.sum()) - ref) / max(abs(ref), 1e-300))

    # --- acoustic power spectrum --------------------------------------------
    r_on_bins = np.array([resistance_at(i, freqs) for i in range(case.n_modes)])
    power_psd = r_on_bins * psd_q                              # W/Hz, one row per mode
    modal_power_band = np.array(
        [band_mean_square(freqs, power_psd[i], bands) for i in range(case.n_modes)]
    )
    p_band = modal_power_band.sum(axis=0)
    e_rad_band = p_band * duration
    e_rad_spectral = float(e_rad_band.sum())
    e_rad_ode = float(
        sum(r_rad[i] * np.trapezoid(qdot[i] ** 2, t) for i in range(case.n_modes))
    )

    # --- far-field pressure fan ---------------------------------------------
    v_surface = patch_phi.T @ vq_amp                   # (n_patch, n_freq)
    omega_bins = 2.0 * math.pi * freqs
    k_bins = omega_bins / C_WATER
    pref = 1j * RHO_WATER * omega_bins / (2.0 * math.pi)
    direction_sl: Dict[str, np.ndarray] = {}
    p1_store: Dict[str, np.ndarray] = {}
    for th_deg in case.assessment_theta_deg:
        for ph_deg in case.assessment_phi_deg:
            th = math.radians(th_deg)
            ph = math.radians(ph_deg)
            phase = k_bins[:, None] * (
                math.sin(th) * math.cos(ph) * radiator.x_m[None, :]
                + math.sin(th) * math.sin(ph) * radiator.y_m[None, :]
            )
            kernel = np.exp(-1j * phase) * radiator.area_m2[None, :]
            s_bins = np.einsum("fp,pf->f", kernel, v_surface)
            p1 = pref * s_bins
            p1[0] = 0.0
            key = f"theta_{th_deg:g}_phi_{ph_deg:g}"
            p1_store[key] = p1
            direction_sl[key] = _db_pressure(_band_mean_square(freqs, p1, bands))

    keys = list(direction_sl)
    sl_matrix = np.array([direction_sl[k] for k in keys])
    best = np.argmax(sl_matrix, axis=0)
    sl_directional_max = sl_matrix[best, np.arange(bands.n_bands)]

    # Omni-directional equivalent source level: the level of the monopole that
    # would radiate the same band power.  This is the source level used by the
    # spherical-spreading propagation model (a directional maximum combined
    # with a 20 log10 r law would be inconsistent).
    power_psd_total = power_psd.sum(axis=0)                        # W/Hz
    p2_density_omni = RHO_WATER * C_WATER / (4.0 * math.pi) * power_psd_total   # Pa^2/Hz at 1 m
    sl_omni = _db_pressure(band_mean_square(freqs, p2_density_omni, bands))
    directivity_index = sl_directional_max - sl_omni

    # --- propagation (per spectral line, omni source) -----------------------
    alpha_bins = np.asarray(acoustics.thorp_absorption_db_per_km(freqs))
    attenuation_per_m = 10.0 ** (-alpha_bins / 1000.0 / 20.0)
    propagation: Dict[str, Any] = {}
    band_centre_error = 0.0
    for r in case.ranges_m:
        p2_density_r = p2_density_omni / r ** 2 * attenuation_per_m ** (2.0 * r)
        spl_r = _db_pressure(band_mean_square(freqs, p2_density_r, bands))
        absorb_centre = np.asarray(acoustics.thorp_absorption_db_per_km(centres)) * r / 1000.0
        predicted = sl_omni - 20.0 * math.log10(r) - absorb_centre
        err = float(np.max(np.abs(spl_r - predicted)))
        band_centre_error = max(band_centre_error, err)
        propagation[f"range_{r:g}_m"] = {
            "range_m": r,
            "spl_bands_db": spl_r.tolist(),
            "absorption_db_at_band_centre": absorb_centre.tolist(),
            "spreading_db": 20.0 * math.log10(r),
            "band_centre_approximation_error_db": err,
        }

    # --- cross-modal radiation interference ---------------------------------
    independence = []
    for i in range(case.n_modes):
        k_peak = int(np.argmax(np.abs(vq_amp[i][1:])) + 1)
        v_snapshot = v_surface[:, k_peak]
        if np.max(np.abs(v_snapshot)) <= 0.0:
            continue
        om_pk = float(omega_bins[k_peak])
        p_full = float(radiator.radiated_power(v_snapshot, om_pk, n_theta=24, n_phi=20))
        p_modal = float(
            sum(
                float(resistance_at(j, np.array([freqs[k_peak]]))[0]) * abs(vq_amp[j, k_peak]) ** 2
                for j in range(case.n_modes)
            )
        )
        independence.append(
            {
                "mode": i + 1,
                "frequency_hz": float(freqs[k_peak]),
                "power_full_rayleigh_w": p_full,
                "power_modal_superposition_w": p_modal,
                "relative_deviation": abs(p_full - p_modal) / max(abs(p_full), 1e-300),
            }
        )

    return {
        "dt": dt,
        "t": t,
        "force": force,
        "duration": duration,
        "modes": modes,
        "omegas": omegas,
        "masses": masses,
        "zetas": zetas,
        "k_modal": k_modal,
        "coupling": coupling,
        "e_crush": e_crush,
        "radiator": radiator,
        "f_grid": f_grid,
        "f_valid": f_valid,
        "resistance_grid": resistance_grid,
        "r_rad": r_rad,
        "c_struct": c_struct,
        "zeta_rad": zeta_rad,
        "q": q,
        "qdot": qdot,
        "qddot": qddot,
        "q_zero": q_zero,
        "ledger": ledger,
        "bands": bands,
        "centres": centres,
        "widths": widths,
        "freqs": freqs,
        "accel_bands": accel_bands,
        "sensors": sensors,
        "parseval_rel": parseval_rel,
        "coverage_share": coverage_share,
        "modal_power_band": modal_power_band,
        "p_band": p_band,
        "e_rad_band": e_rad_band,
        "e_rad_spectral": e_rad_spectral,
        "e_rad_ode": e_rad_ode,
        "direction_sl": direction_sl,
        "sl_directional_max": sl_directional_max,
        "sl_omni": sl_omni,
        "directivity_index": directivity_index,
        "governing_direction": [keys[int(best[b])] for b in range(bands.n_bands)],
        "governing_direction_max_sl": keys[int(best[int(np.argmax(sl_directional_max))])],
        "propagation": propagation,
        "band_centre_error": band_centre_error,
        "modal_interference": independence,
    }


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def run_coupled_chain(case: CoupledCase) -> Dict[str, Any]:
    """Run the full chain, all cross-checks and criteria, and return the record."""
    core = _run_core(case)
    hull = case.hull
    modes: HullGirderModes = core["modes"]
    modal_audit = modes.orthogonality_audit()
    bands = core["bands"]
    centres = core["centres"]

    # ------------------------------------------------------- C1 negative control
    scale_ref = max(float(np.max(np.abs(core["q"]))), 1e-300)
    zero_residual = float(np.max(np.abs(core["q_zero"])) / scale_ref)
    zero_control = {
        "max_abs_q": float(np.max(np.abs(core["q_zero"]))),
        "loaded_max_abs_q": scale_ref,
        "relative_residual": zero_residual,
        "threshold": 1e-9,
        "pass": bool(zero_residual < 1e-9),
    }

    # ----------------------------------------------------------- C4 linearity
    scaled_case = replace(
        case, excitation=replace(case.excitation, force_scale=case.excitation.force_scale * case.linearity_scale)
    )
    scaled_core = _run_core(scaled_case)
    expected_gain_db = 20.0 * math.log10(case.linearity_scale)
    gain_acc = 20.0 * np.log10(
        np.maximum(scaled_core["accel_bands"]["midship"], 1e-300)
        / np.maximum(core["accel_bands"]["midship"], 1e-300)
    )
    gain_sl = scaled_core["sl_omni"] - core["sl_omni"]
    linearity_err = float(
        max(float(np.max(np.abs(gain_acc - expected_gain_db))), float(np.max(np.abs(gain_sl - expected_gain_db))))
    )

    # ------------------------------------------------------------- envelopes
    envelope = iso19906_informed_envelope(case.envelope_plateau_rms_m_s2)
    a_lim = envelope.limit(centres)
    env_mid = band_margins(centres, core["accel_bands"]["midship"], a_lim)
    env_bow = band_margins(centres, core["accel_bands"]["bow"], a_lim)
    sensitivity = []
    for plateau in case.envelope_sensitivity:
        env_s = iso19906_informed_envelope(plateau)
        m_s = band_margins(centres, core["accel_bands"]["midship"], env_s.limit(centres))
        sensitivity.append(
            {"plateau_rms_m_s2": plateau, "max_ratio": m_s["max_ratio"], "pass": m_s["pass"]}
        )
    a_rms_mid = np.maximum(core["accel_bands"]["midship"], 1e-30)
    spl_envelope = core["sl_omni"] + 20.0 * np.log10(a_lim / a_rms_mid)
    spl_over_envelope = core["sl_omni"] - spl_envelope

    # -------------------------------------------------------------- criteria
    ledger = core["ledger"]
    e_in = ledger["work_in"]
    e_rad_spectral = core["e_rad_spectral"]
    residual = e_in - (
        ledger["strain"] + ledger["kinetic"] + ledger["diss_struct"] + e_rad_spectral
    )
    residual_ode = e_in - (
        ledger["strain"] + ledger["kinetic"] + ledger["diss_struct"] + core["e_rad_ode"]
    )
    criteria = {
        "C1_zero_load_control": zero_control,
        "C2_modal_orthogonality": {
            "max_offdiag_mass_ratio": modal_audit["max_offdiag_mass_ratio"],
            "tolerance": modal_audit["tolerance"],
            "n_modes": modal_audit["n_modes"],
            "pass": bool(modal_audit["mass_orthogonality_pass"]),
        },
        "C3_energy_conservation": {
            "work_in_j": e_in,
            "dissipated_structural_j": ledger["diss_struct"],
            "radiated_acoustic_j_spectral": e_rad_spectral,
            "radiated_acoustic_j_ode_damping": core["e_rad_ode"],
            "strain_change_j": ledger["strain"],
            "kinetic_change_j": ledger["kinetic"],
            "residual_j": residual,
            "residual_over_work": abs(residual) / max(abs(e_in), 1e-300),
            "residual_with_ode_damping_j": residual_ode,
            "residual_with_ode_damping_over_work": abs(residual_ode) / max(abs(e_in), 1e-300),
            "residual_attribution": "residual_with_ode_damping is the quadrature/integration "
            "error of the ledger; residual additionally contains the difference between the "
            "frequency-dependent radiation resistance (spectral estimate) and the "
            "at-resonance resistance used in the equation of motion.",
            "tolerance": 0.01,
            "pass": bool(abs(residual) / max(abs(e_in), 1e-300) < 0.01),
            "parseval_band_closure_rel_error": core["parseval_rel"],
            "parseval_tolerance": 1e-6,
            "parseval_pass": bool(core["parseval_rel"] < 1e-6),
            "band_coverage_share_of_modal_velocity_ms": core["coverage_share"],
            "acoustic_validity_band_hz": [float(bands.lower_hz[0]), float(bands.upper_hz[-1])],
            "acoustic_validity_rule": "bands limited to [band_min, min(band_max, "
            "acoustic_validity_factor * f_highest_retained_mode)]: a truncated modal basis "
            "cannot represent the surface velocity above that band",
        },
        "C4_envelope_consistency": {
            "linearity_scale": case.linearity_scale,
            "expected_gain_db": expected_gain_db,
            "max_gain_deviation_db": linearity_err,
            "linearity_tolerance_db": 1e-9,
            "pass": bool(linearity_err < 1e-9),
            "equivalent_spl_envelope_db": spl_envelope.tolist(),
            "physical_finding_C4b": {
                "envelope_plateau_rms_m_s2": case.envelope_plateau_rms_m_s2,
                "midship_acceleration_max_ratio": env_mid["max_ratio"],
                "governing_band_hz": env_mid["governing_band_hz"],
                "midship_within_envelope": env_mid["pass"],
                "bow_acceleration_max_ratio": env_bow["max_ratio"],
                "n_bands_over_limit": env_mid["n_bands_over_limit"],
                "worst_margin_db": env_mid["worst_margin_db"],
                "spl_over_equivalent_envelope_db": spl_over_envelope.tolist(),
                "envelope_sensitivity": sensitivity,
                "note": "C4b is a physics finding (is this ice condition habitable), not a "
                "criterion on the chain; the machine criterion C4 is the linearity that makes "
                "the vibration->noise envelope translation valid.",
            },
        },
    }

    result: Dict[str, Any] = {
        "case": {
            "hull_name": hull.name,
            "bending_stiffness_nm2": case.bending_stiffness_nm2,
            "n_modes": case.n_modes,
            "excitation_mode": case.excitation_mode,
            "excitation": case.excitation.to_json_dict(),
            "structural_damping": list(case.structural_damping[: case.n_modes]),
            "radiator": core["radiator"].to_json_dict(),
            "assessment_directions": {
                "theta_from_hull_down_normal_deg": list(case.assessment_theta_deg),
                "phi_from_bow_deg": list(case.assessment_phi_deg),
                "governing_direction_at_max_sl": core["governing_direction_max_sl"],
            },
            "ranges_m": list(case.ranges_m),
            "band_range_hz": [case.band_min_hz, case.band_max_hz],
            "acoustic_validity_upper_hz": float(core["f_valid"]),
            "sensors_x_m": core["sensors"],
            "resistance_frequency_grid_hz": core["f_grid"].tolist(),
        },
        "geometry": {
            "length_m": hull.length_m,
            "breadth_m": hull.breadth_m,
            "draught_m": hull.draught_m,
            "declared_block_coefficient": hull.block_coefficient,
            "block_coefficient_from_displacement": hull.cb_from_displacement,
            "block_coefficient_abs_error": abs(hull.cb_from_displacement - hull.block_coefficient),
            "displacement_kg": hull.displacement_kg,
            "cb_consistent_within_0p02": bool(
                abs(hull.cb_from_displacement - hull.block_coefficient) <= 0.02
            ),
        },
        "modes": modes.to_json_dict(),
        "load": {
            "n_samples": int(core["t"].size),
            "duration_s": core["duration"],
            "peak_force_n": float(np.max(np.abs(core["force"]))),
            "mean_force_n": float(np.mean(core["force"])),
            "rms_force_n": float(np.sqrt(np.mean(core["force"] ** 2))),
            "crushing_energy_j": core["e_crush"],
            "coupling_coefficients": core["coupling"].tolist(),
            "crushing_frequency_hz": case.excitation.crushing_frequency_hz,
            "static_deflection_mode1_m": float(
                core["coupling"][0] * np.mean(core["force"]) / core["k_modal"][0]
            ),
            "peak_modal_displacement_m": float(np.max(np.abs(core["q"]), axis=1).max()),
        },
        "radiation": {
            "modal_radiation_resistance_n_s_m": core["r_rad"].tolist(),
            "modal_radiation_damping_ratio": core["zeta_rad"].tolist(),
            "structural_damping_ratio": core["zetas"].tolist(),
            "total_damping_ratio": (core["zetas"] + core["zeta_rad"]).tolist(),
            "directivity_index_db": core["directivity_index"].tolist(),
            "governing_direction_per_band": core["governing_direction"],
            "resistance_grid": {
                "frequencies_hz": core["f_grid"].tolist(),
                "per_mode_n_s_m": core["resistance_grid"].tolist(),
            },
        },
        "energy": {
            "crushing_energy_j": core["e_crush"],
            "work_in_j": e_in,
            "radiated_acoustic_j_spectral": e_rad_spectral,
            "radiated_acoustic_j_ode_damping": core["e_rad_ode"],
            "acoustic_conversion_efficiency": e_rad_spectral / max(core["e_crush"], 1e-300),
            "work_to_crushing_ratio": e_in / max(core["e_crush"], 1e-300),
            "ledger": ledger,
        },
        "bands": {
            "centre_hz": list(bands.centre_hz),
            "lower_hz": list(bands.lower_hz),
            "upper_hz": list(bands.upper_hz),
            "band_width_hz": core["widths"].tolist(),
            "acceleration_rms_m_s2": {k: v.tolist() for k, v in core["accel_bands"].items()},
            "acceleration_envelope_m_s2": a_lim.tolist(),
            "acceleration_margin": {
                k: band_margins(centres, v, a_lim) for k, v in core["accel_bands"].items()
            },
            "source_level_db_1m": core["sl_omni"].tolist(),
            "source_level_directional_max_db_1m": core["sl_directional_max"].tolist(),
            "source_level_by_direction_db_1m": {
                k: v.tolist() for k, v in core["direction_sl"].items()
            },
            "equivalent_omni_source_level_db": core["sl_omni"].tolist(),
            "radiated_power_band_w": core["p_band"].tolist(),
            "radiated_power_per_mode_band_w": core["modal_power_band"].tolist(),
            "radiated_energy_band_j": core["e_rad_band"].tolist(),
            "equivalent_spl_envelope_db": spl_envelope.tolist(),
        },
        "propagation": {
            "model": "omni-directional equivalent source level, per-line 20 log10 r "
            "spherical spreading + Thorp absorption (seawater 1025 kg/m3, c = 1500 m/s)",
            "band_centre_approximation_error_db": core["band_centre_error"],
            "by_range": core["propagation"],
            "marine_mammal_thresholds": {
                "continuous_pts_db": acoustics.NMFS_CONTINUOUS_PTS_DB,
                "impulsive_sel_db": acoustics.NMFS_IMPULSIVE_SEL_DB,
                "impulsive_peak_db": acoustics.NMFS_IMPULSIVE_PEAK_DB,
                "range_to_continuous_threshold_m": acoustics.range_to_threshold_m(
                    float(np.max(core["sl_omni"])),
                    float(centres[int(np.argmax(core["sl_omni"]))]),
                    acoustics.NMFS_CONTINUOUS_PTS_DB,
                ),
                "note": "NMFS (2018) thresholds are criteria, not model outputs; SEL/peak "
                "criteria need the impulsive time signature and are reported for context only.",
            },
        },
        "modal_radiation_interference": {
            "per_mode": core["modal_interference"],
            "max_relative_deviation": max(
                (row["relative_deviation"] for row in core["modal_interference"]), default=0.0
            ),
            "note": "deviation between the exact Rayleigh power of the full surface velocity and "
            "the modal-superposition estimate at the same frequency (cross-modal interference "
            "plus resistance-interpolation error).",
        },
        "criteria": criteria,
    }

    result["digest"] = canonical_digest(
        {
            "criteria": criteria,
            "bands": {
                "acceleration_rms_m_s2": {k: v.tolist() for k, v in core["accel_bands"].items()},
                "source_level_db_1m": core["sl_omni"].tolist(),
            "source_level_directional_max_db_1m": core["sl_directional_max"].tolist(),
            },
            "energy": {"work_in_j": e_in, "radiated": e_rad_spectral},
            "response_checksum": {
                "sum_q": float(np.sum(core["q"])),
                "sum_qdot2": float(np.sum(core["qdot"] ** 2)),
            },
        }
    )
    result["criteria"]["C5_idempotency"] = {"digest": result["digest"], "runs_verified": 1, "pass": None}
    return result


def run_with_idempotency(case: CoupledCase) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run the chain twice and verify that the canonical digests agree (C5)."""
    first = run_coupled_chain(case)
    second = run_coupled_chain(case)
    check = {
        "digest_run_1": first["digest"],
        "digest_run_2": second["digest"],
        "identical": first["digest"] == second["digest"],
        "n_runs": 2,
        "pass": bool(first["digest"] == second["digest"]),
    }
    first["criteria"]["C5_idempotency"] = check
    return first, check