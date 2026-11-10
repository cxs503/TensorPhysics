"""Munk deep-sea sound-speed profile and normal-mode propagation.

The Munk profile (Munk, 1974) is the canonical deep-ocean sound-speed
profile:

    c(z) = c_m [1 + ε(η - 1 + e^{-η})]

where
    η = 2(z - z_m) / B
    c_m = sound speed at the sound-channel axis (minimum)
    z_m = depth of the sound-channel axis
    B = scale depth (thermocline width)
    ε = perturbation strength (≈ 0.0074)

The profile has a minimum at z = z_m (the SOFAR channel axis), with
sound speed increasing both upward (toward the surface) and downward
(toward the deep ocean).

Normal-mode propagation in the Munk profile is solved numerically via
the WKB / shooting method for the modal eigenvalue equation:

    d²ψ/dz² + [k²(z) - k_n²] ψ = 0

with k(z) = ω/c(z), and boundary conditions ψ(0) = 0 (pressure-release
surface) and ψ → 0 as z → ∞ (deep ocean).

The convergence-zone (CZ) structure arises from the periodic refocusing
of rays that are trapped in the sound channel: rays launched at shallow
angles turn at the upper and lower turning points and refocus at
predictable ranges.

References
----------
- Munk, W.H. (1974). "Sound channel in an exponentially stratified ocean,
  with application to SOFAR." J. Acoust. Soc. Am. 55, 220-226.
- Jensen, Kuperman, Porter, Schmidt (2011). "Computational Ocean
  Acoustics", 2nd ed., Springer. Ch. 3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import RHO_WATER, C_WATER, P_REF, TL


# ---------------------------------------------------------------------------
# Munk profile
# ---------------------------------------------------------------------------
def munk_sound_speed(
    z: float | np.ndarray,
    c_m: float = 1500.0,
    z_m: float = 1000.0,
    B: float = 1300.0,
    eps: float = 0.0074,
) -> float | np.ndarray:
    """Munk sound-speed profile c(z).

    c(z) = c_m [1 + ε(η - 1 + e^{-η})],  η = 2(z - z_m)/B

    Parameters
    ----------
    z : float or array
        Depth [m] (0 = surface, positive downward).
    c_m : float
        Sound speed at the channel axis [m/s].
    z_m : float
        Channel-axis depth [m].
    B : float
        Scale depth [m].
    eps : float
        Perturbation strength (dimensionless).

    Returns
    -------
    float or array
        Sound speed [m/s].
    """
    z = np.asarray(z, dtype=float)
    eta = 2.0 * (z - z_m) / B
    return c_m * (1.0 + eps * (eta - 1.0 + np.exp(-eta)))


@dataclass
class MunkProfile:
    """Munk deep-sea sound-speed profile.

    Parameters
    ----------
    c_m : float
        Sound speed at the channel axis [m/s].
    z_m : float
        Channel-axis depth [m].
    B : float
        Scale depth [m].
    eps : float
        Perturbation strength (dimensionless).
    """

    c_m: float = 1500.0
    z_m: float = 1000.0
    B: float = 1300.0
    eps: float = 0.0074

    def __post_init__(self):
        if self.B <= 0:
            raise ValueError(f"Scale depth B must be positive, got {self.B}")
        if self.eps <= 0:
            raise ValueError(f"Perturbation eps must be positive, got {self.eps}")

    def c(self, z: float | np.ndarray) -> float | np.ndarray:
        """Sound speed at depth z [m/s]."""
        return munk_sound_speed(z, self.c_m, self.z_m, self.B, self.eps)

    def dc_dz(self, z: float | np.ndarray) -> float | np.ndarray:
        """Sound-speed gradient dc/dz [m/s per m].

        d/dz [c_m(1 + ε(η - 1 + e^{-η}))] = c_m ε (2/B)(1 - e^{-η})
        """
        z = np.asarray(z, dtype=float)
        eta = 2.0 * (z - self.z_m) / self.B
        return self.c_m * self.eps * (2.0 / self.B) * (1.0 - np.exp(-eta))


# ---------------------------------------------------------------------------
# Normal-mode eigenvalue solution (WKB / shooting method)
# ---------------------------------------------------------------------------
def _wkb_phase(
    k_trial: float,
    omega: float,
    profile: MunkProfile,
    z: np.ndarray,
) -> np.ndarray:
    """WKB phase integral: φ(z) = ∫_0^z √(k²(z') - k_trial²) dz'.

    Computed only in the oscillatory region (k(z) > k_trial).

    Parameters
    ----------
    k_trial : float
        Trial horizontal wavenumber [1/m].
    omega : float
        Angular frequency [rad/s].
    profile : MunkProfile
        Munk sound-speed profile.
    z : np.ndarray
        Depth grid [m].

    Returns
    -------
    np.ndarray
        WKB phase φ(z) [rad].
    """
    c = profile.c(z)
    k_z = omega / c
    q = np.sqrt(np.maximum(k_z**2 - k_trial**2, 0.0))
    return np.cumsum(q) * (z[1] - z[0])


def _wkb_mode_shape(
    k_trial: float,
    omega: float,
    profile: MunkProfile,
    z: np.ndarray,
) -> np.ndarray:
    """WKB mode shape: ψ(z) = sin(φ(z)) / √(q(z)).

    Parameters
    ----------
    k_trial : float
        Trial horizontal wavenumber [1/m].
    omega : float
        Angular frequency [rad/s].
    profile : MunkProfile
        Munk sound-speed profile.
    z : np.ndarray
        Depth grid [m].

    Returns
    -------
    np.ndarray
        Mode shape ψ(z).
    """
    c = profile.c(z)
    k_z = omega / c
    q = np.sqrt(np.maximum(k_z**2 - k_trial**2, 1e-12))
    phi = _wkb_phase(k_trial, omega, profile, z)
    return np.sin(phi) / np.sqrt(q)


def _count_zero_crossings(psi: np.ndarray) -> int:
    """Count zero crossings of a mode shape."""
    crossings = 0
    for i in range(1, len(psi)):
        if psi[i-1] * psi[i] < 0:
            crossings += 1
    return crossings


def munk_eigenvalue_shooting(
    mode_index: int,
    omega: float,
    profile: MunkProfile,
    z_max: float = 6000.0,
    n_z: int = 2000,
    n_iter: int = 100,
) -> float:
    """Find the horizontal wavenumber k_n of mode n in the Munk profile.

    Uses the WKB quantization condition: mode n has n+1 zero crossings
    of the WKB mode shape.  Bisection on the number of zero crossings
    finds the eigenvalue.

    Parameters
    ----------
    mode_index : int
        Mode number (0-based).
    omega : float
        Angular frequency [rad/s].
    profile : MunkProfile
        Munk sound-speed profile.
    z_max : float
        Maximum depth for the truncated domain [m].
    n_z : int
        Number of depth grid points.
    n_iter : int
        Number of bisection iterations.

    Returns
    -------
    float
        Horizontal wavenumber k_n [1/m].
    """
    z = np.linspace(0.0, z_max, n_z)
    c = profile.c(z)
    k_z = omega / c

    # The eigenvalue k_n lies in [min(k_z), max(k_z)].
    k_min = np.min(k_z)  # at the channel axis
    k_max = np.max(k_z)  # at the surface

    # Bisection: find k such that count_zero_crossings(k) = mode_index + 1
    # As k increases, the number of zero crossings decreases.
    k_lo = k_min
    k_hi = k_max

    for _ in range(n_iter):
        k_mid = 0.5 * (k_lo + k_hi)
        psi = _wkb_mode_shape(k_mid, omega, profile, z)
        n_cross = _count_zero_crossings(psi)
        if n_cross >= mode_index + 1:
            k_lo = k_mid  # need larger k to reduce crossings
        else:
            k_hi = k_mid  # need smaller k to increase crossings

    return 0.5 * (k_lo + k_hi)


def _normalize_mode(psi: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Normalize a mode shape: ∫|ψ|² dz = 1."""
    norm = np.sqrt(np.trapezoid(psi**2, z))
    if norm < 1e-30:
        return psi
    return psi / norm


def munk_normal_mode_tl(
    frequency: float,
    source_depth: float,
    r: float | np.ndarray,
    z: float,
    profile: MunkProfile,
    n_modes: int = 20,
    z_max: float = 6000.0,
) -> float | np.ndarray:
    """Transmission loss in the Munk profile via normal-mode summation.

    TL(r, z) = -20 log10(|Σ_n p_n(r, z)| / p_ref)

    with the modal sum

        p(r, z) = Σ_n A_n · ψ_n(z_s) · ψ_n(z) · e^{i k_n r} / √(k_n r)

    where ψ_n are the numerically-computed (normalized) WKB mode shapes
    and A_n the normalization factors.

    Parameters
    ----------
    frequency : float
        Source frequency [Hz].
    source_depth : float
        Source depth [m].
    r : float or array
        Horizontal range [m].
    z : float
        Receiver depth [m].
    profile : MunkProfile
        Munk sound-speed profile.
    n_modes : int
        Number of modes to include.
    z_max : float
        Maximum depth for the truncated domain [m].

    Returns
    -------
    float or array
        Transmission loss [dB re 1 μPa].
    """
    omega = 2.0 * np.pi * frequency
    r = np.asarray(r, dtype=float)

    # Depth grid for mode shapes
    z_grid = np.linspace(0.0, z_max, 2000)
    dz = z_grid[1] - z_grid[0]

    # Compute mode eigenvalues and normalized mode shapes
    modes = []
    for n in range(n_modes):
        k_n = munk_eigenvalue_shooting(n, omega, profile, z_max)
        psi = _wkb_mode_shape(k_n, omega, profile, z_grid)
        psi_norm = _normalize_mode(psi, z_grid)
        modes.append((k_n, psi_norm))

    # Interpolate mode shapes to source and receiver depths
    def mode_value(psi_norm: np.ndarray, z_target: float) -> float:
        idx = int(z_target / dz)
        idx = min(max(idx, 0), len(psi_norm) - 1)
        return float(psi_norm[idx])

    # Modal sum
    p_total = np.zeros_like(r, dtype=complex)
    for k_n, psi_norm in modes:
        psi_s = mode_value(psi_norm, source_depth)
        psi_z = mode_value(psi_norm, z)
        # Normalization: A_n = √(2/(ρ0 · z_max)) (approximate)
        A_n = np.sqrt(2.0 / (RHO_WATER * z_max))
        p_n = A_n * psi_s * psi_z * np.exp(1j * k_n * r) / np.sqrt(k_n * r)
        p_total += p_n

    p_abs = np.abs(p_total)
    p_abs = np.where(p_abs < 1e-30, 1e-30, p_abs)
    return 20.0 * np.log10(p_abs / P_REF)


def munk_convergence_zone(
    profile: MunkProfile,
    source_depth: float,
    n_cz: int = 3,
) -> list[float]:
    """Predict the convergence-zone (CZ) ranges for the Munk profile.

    The convergence-zone range is the horizontal distance at which rays
    launched from the source depth refocus after one full cycle (two
    turning points).

    For the Munk profile, the CZ range for a source at the channel axis
    is approximately:

        R_CZ ≈ 2 · B · √(2/ε)

    (standard result from the ray invariant for the Munk profile).

    Parameters
    ----------
    profile : MunkProfile
        Munk sound-speed profile.
    source_depth : float
        Source depth [m].
    n_cz : int
        Number of convergence zones to predict.

    Returns
    -------
    list[float]
        Convergence-zone ranges [km].
    """
    # Standard CZ range for the Munk profile (source at the axis)
    R_cz = 2.0 * profile.B * np.sqrt(2.0 / profile.eps)

    # Adjust for source depth: if the source is not at the axis,
    # the CZ range is slightly different.  Use a simple correction.
    z_offset = abs(source_depth - profile.z_m) / profile.B
    R_cz *= (1.0 + 0.5 * z_offset)

    # Return CZ ranges in km
    return [R_cz * (i + 1) / 1000.0 for i in range(n_cz)]
