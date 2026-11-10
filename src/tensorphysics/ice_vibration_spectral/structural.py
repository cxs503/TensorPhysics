"""Structural dynamics kernels for ice-induced vibration spectral analysis.

Two structural idealisations are provided:

``SDOFSystem``
    The single-degree-of-freedom oscillator

        m u'' + c u' + k u = F(t),      c = 2 zeta omega_n m

    with the *displacement* frequency response function (time convention
    ``e^{i omega t}``)

        H(omega) = 1 / (k - m omega^2 + i c omega),

    and its velocity / acceleration counterparts ``H_v = i omega H`` and
    ``H_a = -omega^2 H``.  Peak amplitude ``|H(omega_n)| = 1/(2 zeta k)`` and the
    half-power band used by the resonance criterion.

``ChainSystem``
    A fixed-free lumped chain of ``n`` masses connected by ``n`` springs.  Its
    eigenfrequencies have the classical closed form

        omega_r = 2 sqrt(k/m) sin( (2r-1) pi / (2 (2n+1)) ),  r = 1..n,

    which is the analytic anchor for the MDOF acceptance criterion.  Modal
    damping is Rayleigh damping ``C = alpha M + beta K`` with
    ``zeta_r = (alpha + beta omega_r^2) / (2 omega_r)``.

The MDOF frequency response is evaluated by two independent code paths: the
direct complex solve ``(K - omega^2 M + i omega C)^{-1}`` and modal
superposition with mass-normalised modes.  Their agreement is one of the
numerical self-checks of the module.

References
----------
* Clough, R.W., Penzien, J. (1993) Dynamics of Structures, 2nd ed.
* Karna, T., Turunen, R. (1990) A straightforward technique for analysing
  structural response to dynamic ice action, IAHR Ice Symposium.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "SDOFSystem",
    "ChainSystem",
    "chain_matrices",
    "chain_analytic_frequencies",
    "rayleigh_modal_damping",
    "sdof_frf",
    "mdof_frf_direct",
    "mdof_frf_modal",
    "state_space_frf_sdof",
]


# --------------------------------------------------------------------------
# SDOF
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SDOFSystem:
    """Modal SDOF oscillator (mass m, stiffness k, viscous damping ratio zeta)."""

    mass_kg: float
    stiffness_n_per_m: float
    damping_ratio: float
    label: str = "sdof"

    def __post_init__(self) -> None:
        if self.mass_kg <= 0.0 or self.stiffness_n_per_m <= 0.0:
            raise ValueError("mass and stiffness must be positive")
        if not 0.0 < self.damping_ratio < 1.0:
            raise ValueError("damping_ratio must lie in (0, 1) for underdamped systems")

    # ------------------------------------------------------------- properties
    @property
    def omega_n_rad_s(self) -> float:
        return float(np.sqrt(self.stiffness_n_per_m / self.mass_kg))

    @property
    def frequency_hz(self) -> float:
        return self.omega_n_rad_s / (2.0 * np.pi)

    @property
    def damping_n_s_per_m(self) -> float:
        return 2.0 * self.damping_ratio * self.omega_n_rad_s * self.mass_kg

    @property
    def peak_amplitude_m_per_n(self) -> float:
        """|H| at the damped resonance peak, exact = 1/(2 zeta k sqrt(1-zeta^2))."""
        z = self.damping_ratio
        return 1.0 / (2.0 * z * self.stiffness_n_per_m * np.sqrt(1.0 - z * z))

    # ------------------------------------------------------------------ FRFs
    def frf_displacement(self, frequency_hz: np.ndarray | float) -> np.ndarray:
        """H(omega) = 1/(k - m omega^2 + i c omega)  [m/N]."""
        w = 2.0 * np.pi * np.asarray(frequency_hz, dtype=float)
        m, k, c = self.mass_kg, self.stiffness_n_per_m, self.damping_n_s_per_m
        return 1.0 / (k - m * w * w + 1j * c * w)

    def frf_velocity(self, frequency_hz: np.ndarray | float) -> np.ndarray:
        """H_v(omega) = i omega H(omega)  [(m/s)/N]."""
        w = 2.0 * np.pi * np.asarray(frequency_hz, dtype=float)
        m, k, c = self.mass_kg, self.stiffness_n_per_m, self.damping_n_s_per_m
        return 1j * w / (k - m * w * w + 1j * c * w)

    def frf_acceleration(self, frequency_hz: np.ndarray | float) -> np.ndarray:
        """H_a(omega) = -omega^2 H(omega)  [(m/s^2)/N]."""
        w = 2.0 * np.pi * np.asarray(frequency_hz, dtype=float)
        m, k, c = self.mass_kg, self.stiffness_n_per_m, self.damping_n_s_per_m
        return -(w * w) / (k - m * w * w + 1j * c * w)

    def peak_frequency_hz(self) -> float:
        """Frequency of the undamped-natural-normalised amplitude peak."""
        return self.frequency_hz * np.sqrt(1.0 - 2.0 * self.damping_ratio**2)

    def half_power_band_hz(self) -> tuple[float, float]:
        """Exact half-power band edges of |H(omega)|.

        Solving ``(1-u)^2 + 4 zeta^2 u = 8 zeta^2 (1-zeta^2)`` for
        ``u = (omega/omega_n)^2`` gives

            omega_+- / omega_n = sqrt(1 - 2 zeta^2 +- 2 zeta sqrt(1 - zeta^2)).

        For small ``zeta`` the half bandwidth tends to ``zeta f_n``.
        """
        z = self.damping_ratio
        base = 1.0 - 2.0 * z * z
        rad = 2.0 * z * np.sqrt(1.0 - z * z)
        lo = np.sqrt(max(base - rad, 0.0))
        hi = np.sqrt(base + rad)
        return float(lo * self.frequency_hz), float(hi * self.frequency_hz)

    def payload(self) -> dict[str, Any]:
        lo, hi = self.half_power_band_hz()
        return {
            "label": self.label,
            "mass_kg": self.mass_kg,
            "stiffness_n_per_m": self.stiffness_n_per_m,
            "damping_ratio": self.damping_ratio,
            "frequency_hz": self.frequency_hz,
            "omega_n_rad_s": self.omega_n_rad_s,
            "damping_n_s_per_m": self.damping_n_s_per_m,
            "peak_amplitude_m_per_n": self.peak_amplitude_m_per_n,
            "peak_frequency_hz": self.peak_frequency_hz(),
            "half_power_band_hz": [lo, hi],
        }


def sdof_frf(
    mass_kg: float,
    stiffness_n_per_m: float,
    damping_ratio: float,
    frequency_hz: np.ndarray,
) -> np.ndarray:
    """Functional form of the SDOF displacement FRF [m/N]."""
    return SDOFSystem(mass_kg, stiffness_n_per_m, damping_ratio).frf_displacement(frequency_hz)


def state_space_frf_sdof(
    mass_kg: float,
    stiffness_n_per_m: float,
    damping_ratio: float,
    frequency_hz: np.ndarray,
) -> np.ndarray:
    """Independent evaluation of the SDOF FRF through the state-space resolvent.

    With ``A = [[0, 1], [-k/m, -c/m]]``, ``B = [0, 1/m]^T``, ``C = [1, 0]``

        H(omega) = C (i omega I - A)^{-1} B = 1 / (m det),

        det = i omega (i omega + c/m) + k/m = (k - m omega^2 + i c omega)/m,

    which is assembled here from the 2x2 resolvent entries rather than from the
    single-line closed form, i.e. an algebraically distinct code path used to
    cross-check the frequency-response acceptance criterion.
    """
    w = 2.0 * np.pi * np.asarray(frequency_hz, dtype=float)
    m, k = float(mass_kg), float(stiffness_n_per_m)
    c = 2.0 * float(damping_ratio) * np.sqrt(k * m)
    a11 = 1j * w
    a22 = 1j * w + c / m
    det = a11 * a22 + (k / m)
    inverse_12 = 1.0 / det          # entry (1,2) of the resolvent
    return (1.0 / m) * inverse_12


# --------------------------------------------------------------------------
# MDOF: fixed-free lumped chain
# --------------------------------------------------------------------------
def chain_matrices(n_dof: int, mass_kg: float, stiffness_n_per_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Mass and stiffness matrices of a fixed-free uniform chain."""
    if n_dof < 1:
        raise ValueError("n_dof must be >= 1")
    m = np.eye(n_dof) * float(mass_kg)
    k = np.zeros((n_dof, n_dof))
    for i in range(n_dof):
        k[i, i] = 2.0 * stiffness_n_per_m if i < n_dof - 1 else stiffness_n_per_m
        if i + 1 < n_dof:
            k[i, i + 1] = k[i + 1, i] = -stiffness_n_per_m
    return m, k


def chain_analytic_frequencies(
    n_dof: int, mass_kg: float, stiffness_n_per_m: float
) -> np.ndarray:
    """Closed-form eigenfrequencies [Hz] of the fixed-free uniform chain."""
    omega0 = np.sqrt(float(stiffness_n_per_m) / float(mass_kg))
    r = np.arange(1, n_dof + 1)
    omega = 2.0 * omega0 * np.sin((2.0 * r - 1.0) * np.pi / (2.0 * (2.0 * n_dof + 1.0)))
    return omega / (2.0 * np.pi)


def rayleigh_modal_damping(
    omega_rad_s: np.ndarray, alpha: float, beta: float
) -> np.ndarray:
    """Rayleigh modal damping ratios zeta_r = (alpha + beta omega_r^2)/(2 omega_r)."""
    w = np.asarray(omega_rad_s, dtype=float)
    return (alpha + beta * w * w) / (2.0 * w)


def _rayleigh_from_two_modes(
    omega_a: float, zeta_a: float, omega_b: float, zeta_b: float
) -> tuple[float, float]:
    """Solve ``alpha + beta omega_r^2 = 2 zeta_r omega_r`` at two modes."""
    a = np.array([[1.0, omega_a * omega_a], [1.0, omega_b * omega_b]])
    rhs = np.array([2.0 * zeta_a * omega_a, 2.0 * zeta_b * omega_b])
    alpha, beta = np.linalg.solve(a, rhs)
    return float(alpha), float(beta)


@dataclass
class ChainSystem:
    """Fixed-free uniform chain with Rayleigh damping and modal basis.

    Parameters
    ----------
    n_dof : int
        Number of lumped masses (>= 3 for the MDOF acceptance criterion).
    mass_kg, stiffness_n_per_m : float
        Lumped mass and inter-mass spring stiffness.
    zeta_first, zeta_last : float
        Target modal damping ratios carried by the first and last modes; the
        Rayleigh coefficients are solved from these two conditions.
    """

    n_dof: int
    mass_kg: float
    stiffness_n_per_m: float
    zeta_first: float = 0.02
    zeta_last: float = 0.02
    label: str = "chain"

    mass_matrix: np.ndarray = field(init=False, repr=False)
    stiffness_matrix: np.ndarray = field(init=False, repr=False)
    damping_matrix: np.ndarray = field(init=False, repr=False)
    eigenfrequencies_hz: np.ndarray = field(init=False)
    omega_rad_s: np.ndarray = field(init=False)
    mode_shapes: np.ndarray = field(init=False)
    zeta: np.ndarray = field(init=False)
    alpha: float = field(init=False)
    beta: float = field(init=False)

    def __post_init__(self) -> None:
        if self.n_dof < 2:
            raise ValueError("a chain needs at least 2 DOF")
        self.mass_matrix, self.stiffness_matrix = chain_matrices(
            self.n_dof, self.mass_kg, self.stiffness_n_per_m
        )
        m_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(self.mass_matrix)))
        a = m_inv_sqrt @ self.stiffness_matrix @ m_inv_sqrt
        eigval, eigvec = np.linalg.eigh(a)
        order = np.argsort(eigval)
        self.omega_rad_s = np.sqrt(np.maximum(eigval[order], 0.0))
        self.mode_shapes = m_inv_sqrt @ eigvec[:, order]  # mass-normalised
        # sign convention: first non-zero component positive
        for j in range(self.n_dof):
            if self.mode_shapes[0, j] < 0:
                self.mode_shapes[:, j] *= -1.0
        self.eigenfrequencies_hz = self.omega_rad_s / (2.0 * np.pi)
        self.alpha, self.beta = _rayleigh_from_two_modes(
            self.omega_rad_s[0], self.zeta_first, self.omega_rad_s[-1], self.zeta_last
        )
        self.damping_matrix = self.alpha * self.mass_matrix + self.beta * self.stiffness_matrix
        self.zeta = rayleigh_modal_damping(self.omega_rad_s, self.alpha, self.beta)

    # ------------------------------------------------------------------ FRFs
    def frf_direct(self, frequency_hz: np.ndarray, dof_out: int, dof_in: int) -> np.ndarray:
        """H_ji(omega) = [(K - omega^2 M + i omega C)^{-1}]_{ji} (direct solve)."""
        w = 2.0 * np.pi * np.asarray(frequency_hz, dtype=float)
        out = np.empty(w.size, dtype=complex)
        for idx, wi in enumerate(w):
            dyn = (
                self.stiffness_matrix
                - wi * wi * self.mass_matrix
                + 1j * wi * self.damping_matrix
            )
            out[idx] = np.linalg.solve(dyn, np.eye(self.n_dof)[:, dof_in])[dof_out]
        return out

    def frf_modal(self, frequency_hz: np.ndarray, dof_out: int, dof_in: int) -> np.ndarray:
        """Modal superposition with mass-normalised modes (independent path)."""
        w = 2.0 * np.pi * np.asarray(frequency_hz, dtype=float)
        out = np.zeros(w.size, dtype=complex)
        phi_o = self.mode_shapes[dof_out, :]
        phi_i = self.mode_shapes[dof_in, :]
        for r in range(self.n_dof):
            wr = self.omega_rad_s[r]
            zr = self.zeta[r]
            denom = wr * wr - w * w + 2j * zr * wr * w
            out += phi_o[r] * phi_i[r] / denom
        return out

    # ------------------------------------------------------------- properties
    def payload(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "n_dof": self.n_dof,
            "mass_kg": self.mass_kg,
            "stiffness_n_per_m": self.stiffness_n_per_m,
            "eigenfrequencies_hz": self.eigenfrequencies_hz.tolist(),
            "analytic_frequencies_hz": chain_analytic_frequencies(
                self.n_dof, self.mass_kg, self.stiffness_n_per_m
            ).tolist(),
            "rayleigh_alpha": self.alpha,
            "rayleigh_beta": self.beta,
            "modal_damping": self.zeta.tolist(),
            "mode_shapes": self.mode_shapes.tolist(),
        }


def mdof_frf_direct(system: ChainSystem, frequency_hz: np.ndarray, dof_out: int, dof_in: int) -> np.ndarray:
    return system.frf_direct(frequency_hz, dof_out, dof_in)


def mdof_frf_modal(system: ChainSystem, frequency_hz: np.ndarray, dof_out: int, dof_in: int) -> np.ndarray:
    return system.frf_modal(frequency_hz, dof_out, dof_in)