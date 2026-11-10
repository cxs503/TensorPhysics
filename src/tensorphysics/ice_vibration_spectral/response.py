"""Frequency-response, response-PSD and RMS post-processing.

Chain implemented here::

    S_ff(f)  ->  H_a(omega)  ->  S_yy(f) = |H_a(omega)|^2 S_ff(f)  ->  sigma_y

The displacement, velocity and acceleration responses are all derived from the
same displacement FRF, so the spectral moments of the response can be formed
without leaving the frequency domain.  Two closed-form anchors are used:

* white-noise excitation of a SDOF oscillator (one-sided density ``S0``)::

      sigma_u^2 = S0 / (8 zeta k sqrt(k m)) = pi S0 f_n / (4 zeta k^2 ... )

  derived from ``integral_{-inf}^{inf} |H|^2 domega = pi/(c k)`` with
  ``c = 2 zeta sqrt(k m)``;

* the exact discrete-time simulation of an SDOF oscillator driven by a
  piecewise-constant force (state transition matrix of the 2x2 oscillator,
  particular solution ``[F/k, 0]``), which is used as an independent time-domain
  check of the PSD/RMS path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .structural import ChainSystem, SDOFSystem

__all__ = [
    "modal_response_psd",
    "refine_psd",
    "zero_order_hold_response",
    "sdof_response_psd",
    "sdof_acceleration_psd",
    "mdof_response_psd",
    "rms_from_psd",
    "spectral_moments",
    "white_noise_rms_analytic",
    "psd_peak_frequency",
]


# --------------------------------------------------------------------------
# SDOF
# --------------------------------------------------------------------------
def sdof_response_psd(
    system: SDOFSystem, frequency_hz: np.ndarray, force_psd: np.ndarray
) -> np.ndarray:
    """Displacement response PSD [m^2/Hz] = |H(omega)|^2 S_ff."""
    h = system.frf_displacement(frequency_hz)
    return np.abs(h) ** 2 * np.asarray(force_psd, dtype=float)


def sdof_acceleration_psd(
    system: SDOFSystem, frequency_hz: np.ndarray, force_psd: np.ndarray
) -> np.ndarray:
    """Acceleration response PSD [(m/s^2)^2/Hz] = omega^4 |H|^2 S_ff."""
    h = system.frf_acceleration(frequency_hz)
    return np.abs(h) ** 2 * np.asarray(force_psd, dtype=float)


def mdof_response_psd(
    system: ChainSystem,
    frequency_hz: np.ndarray,
    force_psd: np.ndarray,
    dof_out: int,
    dof_in: int,
    response: str = "displacement",
) -> np.ndarray:
    """Response PSD at ``dof_out`` for a point force PSD at ``dof_in``.

    ``response`` selects ``displacement`` [m^2/Hz], ``velocity`` or
    ``acceleration``; the transfer function is the modal-superposition FRF.
    """
    h = system.frf_modal(frequency_hz, dof_out, dof_in)
    w = 2.0 * np.pi * np.asarray(frequency_hz, dtype=float)
    if response == "displacement":
        factor = np.ones_like(w)
    elif response == "velocity":
        factor = w * w
    elif response == "acceleration":
        factor = w**4
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unknown response type {response!r}")
    return np.abs(h) ** 2 * factor * np.asarray(force_psd, dtype=float)


def modal_response_psd(
    frequency_hz: np.ndarray,
    mode_frequencies_hz: np.ndarray,
    mode_shapes: np.ndarray,
    damping_ratios: np.ndarray,
    force_psd: np.ndarray,
    dof_in: int,
    dof_out: int,
    response: str = "displacement",
) -> np.ndarray:
    """Response PSD from an arbitrary modal basis (FEM / test modes).

    Interface contract for an external modal extraction (TensorFEM, measured
    modal survey): given the modal frequencies ``f_n`` [Hz], the *mass
    normalised* mode shapes ``phi`` (``n_dof x n_modes``, so that
    ``phi^T M phi = I``) and the modal damping ratios ``zeta_r``, the transfer
    function for a point force at ``dof_in`` reads

        H_ji(omega) = sum_r phi_jr phi_ir / (omega_r^2 - omega^2 + 2 i zeta_r omega_r omega),

    and the response PSD is ``S_yy = |H_ji|^2 S_ff`` (times ``omega^4`` for
    acceleration).  No mass or stiffness matrix is needed, which is what makes
    this usable with a FEM eigenvalue export.
    """
    f = np.asarray(frequency_hz, dtype=float)
    w = 2.0 * np.pi * f
    wn = 2.0 * np.pi * np.asarray(mode_frequencies_hz, dtype=float)
    phi = np.asarray(mode_shapes, dtype=float)
    zeta = np.asarray(damping_ratios, dtype=float)
    if phi.shape[1] != wn.size or zeta.size != wn.size:
        raise ValueError("mode_frequencies_hz, mode_shapes and damping_ratios must agree in mode count")
    h = np.zeros(w.size, dtype=complex)
    for r in range(wn.size):
        denom = wn[r] ** 2 - w * w + 2j * zeta[r] * wn[r] * w
        h += phi[dof_out, r] * phi[dof_in, r] / denom
    if response == "displacement":
        factor = np.ones_like(w)
    elif response == "velocity":
        factor = w * w
    elif response == "acceleration":
        factor = w**4
    else:
        raise ValueError(f"unknown response type {response!r}")
    return np.abs(h) ** 2 * factor * np.asarray(force_psd, dtype=float)


# --------------------------------------------------------------------------
# Spectral statistics
# --------------------------------------------------------------------------
def rms_from_psd(frequency_hz: np.ndarray, psd: np.ndarray) -> float:
    """RMS of a one-sided PSD, ``sqrt(integral S df)`` (trapezoidal rule)."""
    f = np.asarray(frequency_hz, dtype=float)
    s = np.asarray(psd, dtype=float)
    if f.size < 2:
        raise ValueError("need at least 2 frequency samples")
    return float(np.sqrt(max(np.trapezoid(s, f), 0.0)))


def spectral_moments(frequency_hz: np.ndarray, psd: np.ndarray, n_max: int = 4) -> np.ndarray:
    """Spectral moments ``m_n = integral f^n S(f) df`` for n = 0..n_max."""
    f = np.asarray(frequency_hz, dtype=float)
    s = np.asarray(psd, dtype=float)
    return np.array([float(np.trapezoid(s * f**n, f)) for n in range(n_max + 1)])


def white_noise_rms_analytic(system: SDOFSystem, s0_n2_per_hz: float) -> float:
    """Closed-form RMS displacement under one-sided white force PSD ``S0``.

    ``sigma^2 = S0/(8 zeta k sqrt(k m))``, valid for a white (flat two-sided)
    excitation and all damping ratios of an underdamped oscillator.
    """
    k = system.stiffness_n_per_m
    m = system.mass_kg
    return float(np.sqrt(s0_n2_per_hz / (8.0 * system.damping_ratio * k * np.sqrt(k * m))))


def psd_peak_frequency(frequency_hz: np.ndarray, psd: np.ndarray, f_min: float = 0.0) -> float:
    """Frequency of the largest PSD value above ``f_min``."""
    f = np.asarray(frequency_hz, dtype=float)
    s = np.asarray(psd, dtype=float)
    mask = f >= f_min
    if not np.any(mask):
        raise ValueError("no spectral samples above f_min")
    idx = np.argmax(s[mask])
    return float(f[mask][idx])


# --------------------------------------------------------------------------
# Exact time-domain check of the PSD/RMS path
# --------------------------------------------------------------------------
@dataclass
class SDOFTimeHistory:
    time_s: np.ndarray
    displacement_m: np.ndarray
    velocity_m_per_s: np.ndarray
    acceleration_m_per_s2: np.ndarray
    rms_displacement_m: float
    rms_acceleration_m_per_s2: float


def zero_order_hold_response(
    system: SDOFSystem, force_n: np.ndarray, dt_s: float
) -> SDOFTimeHistory:
    """Exact response of an SDOF oscillator to a piecewise-constant force.

    Uses the closed-form 2x2 state transition matrix of the underdamped
    oscillator

        Ad = exp(-a dt) [[cos(wd dt) + (a/wd) sin(wd dt), (1/wd) sin(wd dt)],
                         [-(wn^2/wd) sin(wd dt), cos(wd dt) - (a/wd) sin(wd dt)]]

    with ``a = zeta wn``, ``wd = wn sqrt(1-zeta^2)``, and the exact particular
    solution of a constant force ``x_p = [F/k, 0]``::

        x_{n+1} = Ad (x_n - x_p) + x_p.

    This is not a numerical approximation of the ODE (no time-integration
    truncation error), so it can be used as an independent check of the
    frequency-domain RMS.
    """
    m = system.mass_kg
    k = system.stiffness_n_per_m
    z = system.damping_ratio
    wn = system.omega_n_rad_s
    a = z * wn
    wd = wn * np.sqrt(1.0 - z * z)
    cdt = np.cos(wd * dt_s)
    sdt = np.sin(wd * dt_s)
    e = np.exp(-a * dt_s)
    ad11 = e * (cdt + (a / wd) * sdt)
    ad12 = e * (sdt / wd)
    ad21 = -e * (wn * wn / wd) * sdt
    ad22 = e * (cdt - (a / wd) * sdt)
    f = np.asarray(force_n, dtype=float)
    n = f.size
    disp = np.empty(n)
    vel = np.empty(n)
    acc = np.empty(n)
    u = 0.0
    v = 0.0
    for i in range(n):
        fi = float(f[i])
        disp[i] = u
        vel[i] = v
        acc[i] = (fi - 2.0 * z * wn * m * v - k * u) / m
        xp = fi / k
        du = u - xp
        # x_{n+1} = Ad (x - xp) + xp   (exact for piecewise-constant force)
        u_new = ad11 * du + ad12 * v + xp
        v_new = ad21 * du + ad22 * v
        u, v = u_new, v_new
    return SDOFTimeHistory(
        time_s=np.arange(n) * dt_s,
        displacement_m=disp,
        velocity_m_per_s=vel,
        acceleration_m_per_s2=acc,
        rms_displacement_m=float(np.std(disp)),
        rms_acceleration_m_per_s2=float(np.std(acc)),
    )


def refine_psd(
    frequency_hz: np.ndarray,
    psd: np.ndarray,
    factor: int = 32,
    f_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate a PSD onto a grid refined by ``factor``.

    The Welch grid of a finite ice-load record has a resolution of
    ``fs/nperseg`` (typically 0.03 - 0.12 Hz) that is *coarser* than the
    half-power bandwidth of a lightly damped structural mode (``zeta f_n``,
    e.g. 0.022 Hz).  Integrating ``|H|^2 S_ff`` directly on the Welch grid
    therefore under-resolves the resonance and biases the response RMS by tens
    of percent (measured: 32 % for the lighthouse case).  The force PSD is
    smooth on that scale, so it is interpolated onto a fine grid (linear, with
    zero outside the estimated band) before multiplying by the structural
    transfer function.  The interpolation error is bounded by the curvature of
    the Welch spectrum and is checked against the time-domain RMS.
    """
    f = np.asarray(frequency_hz, dtype=float)
    s = np.asarray(psd, dtype=float)
    hi = float(f_max) if f_max is not None else float(f[-1])
    n = int(max(2, round((hi - f[0]) / ((f[1] - f[0]) / max(factor, 1)))))
    ff = np.linspace(f[0], hi, n)
    ss = np.interp(ff, f, s, left=0.0, right=0.0)
    return ff, ss
