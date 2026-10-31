"""Far-field acoustic extrapolation from the near-field cavitation source.

Two classical routes are implemented.

1. **Lighthill / Ffowcs Williams-Hawkings, compact-source form.**  The cavity is
   acoustically compact (k a << 1), so its monopole (thickness/volume) radiation
   is

       p_mono(r, t) = rho_l / (4 pi r) * d^2 V / dt^2 (t - r/c)

   and the force (loading) dipole radiation is

       p_dip(r, t) = -1 / (4 pi c r) * d/dt [ F . r_hat ] (t - r/c)

   with the retarded time ``t - r/c``.  ``V`` is the total vapour volume of the
   cavitation region and ``F`` the unsteady blade/ice loading force.

2. **Kirchhoff surface integral for a compact spherical control surface.**
   For a monopole field measured on a sphere of radius ``a`` the Kirchhoff
   extrapolation reduces to

       p(r, t) = (a / r) * p_s(a, t - (r - a)/c)

   which is the exact linear-acoustics result for ``k a << 1`` and is used here
   as an independent check of the retarded-time / geometric-spreading pipeline.

The spectra are computed with a Hann window and a one-sided amplitude
normalisation, so that a pure tone of amplitude ``A`` peaks at ``A`` (not at
``A/2``).
"""

from __future__ import annotations

import numpy as np

from .cavitation import C_L, RHO_L

P_REF: float = 1.0e-6   # reference pressure of underwater acoustics [Pa]


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------
def blade_pass_frequency(rpm: float, n_blades: int) -> float:
    """Blade-passing frequency [Hz] from shaft speed in revolutions per minute."""
    if rpm <= 0.0 or n_blades <= 0:
        raise ValueError("rpm and n_blades must be positive")
    return rpm / 60.0 * n_blades


# ---------------------------------------------------------------------------
# Source terms
# ---------------------------------------------------------------------------
def _bandlimit(t: np.ndarray, x: np.ndarray, f_max: float) -> np.ndarray:
    """Ideal low-pass of a uniformly sampled signal (FFT brick wall)."""
    dt = float(np.mean(np.diff(t)))
    spec = np.fft.rfft(x)
    freq = np.fft.rfftfreq(x.size, d=dt)
    spec[freq > f_max] = 0.0
    return np.fft.irfft(spec, n=x.size)


def cavity_volume_acceleration(
    t: np.ndarray, volume: np.ndarray, smooth: int = 0, f_max: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Second time derivative of the vapour volume (the monopole source).

    Central differences on a uniform time grid; an optional moving-average
    ``smooth`` window (>0) is applied to ``dV/dt`` before the final difference to
    suppress the discretisation noise of the flow solver.  Values on the
    boundary are set to zero (they are excluded from the FFT band of interest
    anyway).
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(volume, dtype=float)
    if t.size < 3:
        raise ValueError("need at least three samples")
    dt = float(np.mean(np.diff(t)))
    if f_max is not None:
        # Differentiating twice amplifies the out-of-band numerical noise of
        # the bubble dynamics by omega^2, which would swamp the physical
        # monopole.  The volume history is therefore band-limited to the
        # receiver band of interest before the second derivative: content above
        # ``f_max`` is not resolved by the (isothermal, rigid-core) bubble
        # model and carries no physical amplitude.
        v = _bandlimit(t, v, f_max)
    dv = np.gradient(v, dt, edge_order=1)
    if smooth and smooth > 1:
        kernel = np.ones(int(smooth)) / float(smooth)
        dv = np.convolve(dv, kernel, mode="same")
    d2v = np.gradient(dv, dt, edge_order=1)
    d2v[:2] = 0.0
    d2v[-2:] = 0.0
    return t, d2v


def monopole_far_field(
    t: np.ndarray,
    volume: np.ndarray,
    distance: float,
    c: float = C_L,
    rho: float = RHO_L,
    smooth: int = 0,
    f_max: float | None = None,
) -> dict:
    """Compact monopole far-field pressure from a vapour-volume history."""
    t, d2v = cavity_volume_acceleration(t, volume, smooth=smooth, f_max=f_max)
    p_near = rho / (4.0 * np.pi) * d2v          # source strength
    delay = (t[-1] - t[0])
    if distance / c >= delay:
        raise ValueError("time series shorter than the propagation delay")
    p_far = _retarded_interp(t, p_near, distance / c) / distance
    return {
        "t": t,
        "p": p_far,
        "p_rms": float(np.sqrt(np.mean(p_far ** 2))),
        "distance": distance,
    }


def dipole_far_field(
    t: np.ndarray,
    force: np.ndarray,
    distance: float,
    theta: float = 0.0,
    c: float = C_L,
    smooth: int = 0,
) -> dict:
    """Compact dipole far-field pressure; ``theta`` is the angle to the force axis."""
    t = np.asarray(t, dtype=float)
    f = np.asarray(force, dtype=float)
    dt = float(np.mean(np.diff(t)))
    df = np.gradient(f, dt, edge_order=1)
    if smooth and smooth > 1:
        kernel = np.ones(int(smooth)) / float(smooth)
        df = np.convolve(df, kernel, mode="same")
    p_near = -np.cos(theta) / (4.0 * np.pi * c) * df
    p_far = _retarded_interp(t, p_near, distance / c) / distance
    return {
        "t": t,
        "p": p_far,
        "p_rms": float(np.sqrt(np.mean(p_far ** 2))),
        "distance": distance,
        "theta": float(theta),
    }


def compact_source_far_field(
    t: np.ndarray,
    volume: np.ndarray,
    force: np.ndarray | None = None,
    distance: float = 1.0,
    theta: float = 0.0,
    c: float = C_L,
    rho: float = RHO_L,
    smooth: int = 0,
    f_max: float | None = None,
) -> dict:
    """Monopole + dipole far field (FW-H compact form)."""
    mono = monopole_far_field(t, volume, distance, c=c, rho=rho, smooth=smooth, f_max=f_max)
    p = mono["p"]
    out = {"t": mono["t"], "p_monopole": p.copy()}
    if force is not None:
        dip = dipole_far_field(t, force, distance, theta=theta, c=c, smooth=smooth)
        p = p + dip["p"]
        out["p_dipole"] = dip["p"]
        out["dipole_rms"] = dip["p_rms"]
    out["p"] = p
    out["p_rms"] = float(np.sqrt(np.mean(p ** 2)))
    out["monopole_rms"] = mono["p_rms"]
    out["distance"] = distance
    return out


def kirchhoff_sphere_extrapolation(
    t: np.ndarray,
    p_surface: np.ndarray,
    surface_radius: float,
    distance: float,
    c: float = C_L,
) -> dict:
    """Kirchhoff extrapolation of a compact spherical control surface."""
    if distance < surface_radius:
        raise ValueError("observer must be outside the control surface")
    lag = (distance - surface_radius) / c
    p_far = _retarded_interp(t, np.asarray(p_surface, dtype=float), lag) * (
        surface_radius / distance
    )
    return {
        "t": t,
        "p": p_far,
        "p_rms": float(np.sqrt(np.mean(p_far ** 2))),
        "distance": distance,
        "surface_radius": surface_radius,
    }


def _retarded_interp(t: np.ndarray, x: np.ndarray, lag: float) -> np.ndarray:
    """Evaluate ``x(t - lag)`` on the sample grid (no extrapolation)."""
    t = np.asarray(t, dtype=float)
    dt = float(t[1] - t[0])
    shift = lag / dt
    n = x.size
    idx = np.arange(n) - shift
    i0 = np.floor(idx).astype(int)
    frac = idx - i0
    out = np.zeros(n)
    valid = (i0 >= 0) & (i0 + 1 < n)
    out[valid] = (1.0 - frac[valid]) * x[i0[valid]] + frac[valid] * x[i0[valid] + 1]
    return out


# ---------------------------------------------------------------------------
# Spectral analysis
# ---------------------------------------------------------------------------
def amplitude_spectrum(
    t: np.ndarray,
    signal: np.ndarray,
    detrend: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """One-sided amplitude spectrum (Hann window, tone amplitude calibration)."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(signal, dtype=float)
    n = x.size
    dt = float(np.mean(np.diff(t)))
    if detrend:
        x = x - np.mean(x)
    window = np.hanning(n)
    norm = np.sum(window) / 2.0
    spec = np.fft.rfft(x * window)
    amp = 2.0 * np.abs(spec) / (norm * 2.0)
    freq = np.fft.rfftfreq(n, d=dt)
    return freq, amp


def rms_of_band(freq: np.ndarray, amp: np.ndarray, f_lo: float, f_hi: float) -> float:
    """RMS pressure in a frequency band from an amplitude spectrum."""
    mask = (freq >= f_lo) & (freq <= f_hi)
    if not np.any(mask):
        return 0.0
    return float(np.sqrt(0.5 * np.sum(amp[mask] ** 2)))


def peak_frequency(
    t: np.ndarray,
    signal: np.ndarray,
    f_min: float = 1.0,
    f_max: float | None = None,
) -> dict:
    """Dominant tone frequency with parabolic (log-magnitude) interpolation."""
    freq, amp = amplitude_spectrum(t, signal)
    f_max = float(freq[-1]) if f_max is None else float(f_max)
    band = (freq >= f_min) & (freq <= f_max)
    if not np.any(band):
        raise ValueError("empty search band")
    idx_band = np.flatnonzero(band)
    k = idx_band[int(np.argmax(amp[band]))]
    f_peak = float(freq[k])
    if 0 < k < len(amp) - 1:
        y0, y1, y2 = np.log(np.maximum(amp[k - 1:k + 2], 1.0e-30))
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1.0e-30:
            delta = 0.5 * (y0 - y2) / denom
            df = float(freq[1] - freq[0])
            f_peak = float(freq[k] + np.clip(delta, -0.5, 0.5) * df)
    return {
        "f_peak": f_peak,
        "amplitude": float(amp[k]),
        "f_peak_bin": float(freq[k]),
        "df": float(freq[1] - freq[0]),
        "freq": freq,
        "amp": amp,
    }


def sound_pressure_level(p_rms: float, p_ref: float = P_REF) -> float:
    """SPL in dB re 1 uPa (used only for reporting, never for the criteria)."""
    return float(20.0 * np.log10(max(p_rms, 1e-30) / p_ref))