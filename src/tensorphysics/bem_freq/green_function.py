"""Frequency-domain free-surface Green function G(x,ξ;ω) and its gradient.

The Green function satisfies:

    ∇²G = δ(x - ξ)          in the fluid domain (z < 0)
    -ω²G + g ∂G/∂z = 0      on z = 0  (linearised free-surface condition)
    ∂G/∂z → 0               as z → -∞
    outgoing radiation condition at infinity

It is decomposed as:

    G(x,ξ;ω) = G_rankine + G_wave

where G_rankine = -1/(4πr) - 1/(4πr') is the Rankine source plus its image
(r = |x - ξ|, r' = |x - ξ'| with ξ' the mirror of ξ about z = 0), and
G_wave is the wave part that satisfies the homogeneous Laplace equation and
the free-surface condition.

The wave part is computed via the series expansion of the Havelock form:

    G_wave = (1/π) ∫₀^∞ [k/(k - K)] e^{k(z+ζ)} J₀(kR) dk

where K = ω²/g is the wave number, R = √[(x-ξ)² + (y-η)²], and the integral
is understood in the principal-value sense with the pole at k = K handled
by the radiation condition (outgoing waves).

For the far field (kR >> 1), the stationary-phase approximation gives the
Kochin function H(θ) which characterises the radiated wave amplitude.

References
----------
- Newman, J.N. (1985) "The evaluation of free-surface Green's functions"
- Wehausen & Laitone (1960) "Surface Waves" in Handbuch der Physik
- Faltinsen, O.M. (1990) "Sea Loads on Ships and Offshore Structures"
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy import integrate, special

__all__ = [
    "GreenFunction",
    "green_function_value",
    "green_function_gradient",
    "kochin_function",
    "kochin_energy_damping",
]

RHO_WATER = 1025.0
GRAVITY = 9.81


@dataclass
class GreenFunction:
    """Frequency-domain free-surface Green function evaluator.

    Parameters
    ----------
    omega : float
        Angular frequency (rad/s).
    g : float
        Gravitational acceleration (m/s²).
    """

    omega: float
    g: float = GRAVITY

    @property
    def K(self) -> float:
        """Wave number K = ω²/g."""
        return self.omega ** 2 / self.g

    def rankine(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                xi: np.ndarray, eta: np.ndarray, zeta: np.ndarray) -> np.ndarray:
        """Rankine source + image: -1/(4πr) - 1/(4πr')."""
        dx = x - xi
        dy = y - eta
        dz = z - zeta
        r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        r = np.maximum(r, 1e-15)
        dz_img = z + zeta  # image source at -zeta
        r_img = np.sqrt(dx ** 2 + dy ** 2 + dz_img ** 2)
        r_img = np.maximum(r_img, 1e-15)
        return -1.0 / (4.0 * np.pi * r) - 1.0 / (4.0 * np.pi * r_img)

    def _wave_integrand(self, k: float, R: float, z: float, zeta: float) -> float:
        """Integrand for the wave part of the Green function.

        f(k) = k/(k - K) * exp(k(z+zeta)) * J₀(kR)

        The pole at k = K is handled by principal-value integration.
        """
        K = self.K
        # Avoid the pole
        if abs(k - K) < 1e-12:
            return 0.0
        exp_term = np.exp(k * (z + zeta))
        # Clamp to avoid overflow
        if exp_term > 1e15:
            return 0.0
        bessel = special.j0(k * R)
        return k / (k - K) * exp_term * bessel

    def wave_part(self, R: float, z: float, zeta: float) -> float:
        """Compute the wave part of the Green function via numerical integration.

        Parameters
        ----------
        R : float
            Horizontal distance √[(x-ξ)² + (y-η)²].
        z, zeta : float
            Vertical coordinates of field and source points (both ≤ 0).

        Returns
        -------
        float
            The wave part G_wave.
        """
        K = self.K
        if R < 1e-15:
            R = 1e-15

        # Split the integral at the pole k = K
        # Integrate from 0 to K-ε and from K+ε to ∞
        eps = max(1e-8, K * 1e-6)

        # Upper limit for integration (exponential decay for z+zeta < 0)
        k_max = max(50.0 * K, 100.0)

        try:
            # Principal value: integrate on both sides of the pole
            if K > eps:
                part1, _ = integrate.quad(
                    lambda k: self._wave_integrand(k, R, z, zeta),
                    0, K - eps, limit=200
                )
            else:
                part1 = 0.0

            part2, _ = integrate.quad(
                lambda k: self._wave_integrand(k, R, z, zeta),
                K + eps, k_max, limit=200
            )

            # Add the residue contribution from the pole (radiation condition)
            # The residue at k=K is: K * exp(K(z+zeta)) * J₀(KR) * iπ
            # For the real part (standing wave), we take the principal value
            # The imaginary part corresponds to outgoing waves
            residue = K * np.exp(K * (z + zeta)) * special.j0(K * R)

            return (part1 + part2) / np.pi + 1j * residue
        except Exception:
            return 0.0 + 0j

    def value(self, x: float, y: float, z: float,
              xi: float, eta: float, zeta: float) -> complex:
        """Full Green function G(x,ξ;ω).

        Returns complex value: real part = standing wave, imag = outgoing wave.
        """
        R = np.sqrt((x - xi) ** 2 + (y - eta) ** 2)
        rank = self.rankine(
            np.array([x]), np.array([y]), np.array([z]),
            np.array([xi]), np.array([eta]), np.array([zeta])
        )[0]
        wave = self.wave_part(R, z, zeta)
        return rank + wave

    def far_field_asymptotic(self, R: float, theta: float, z: float, zeta: float) -> complex:
        """Far-field asymptotic form of the Green function.

        For kR >> 1:
            G ~ -i * sqrt(2K/(πR)) * exp(-iKR + iπ/4) * exp(K(z+zeta)) * exp(iK(x-ξ)cosθ)

        This is used to verify the Kochin function radiation condition.
        """
        K = self.K
        if R < 1.0:
            R = 1.0
        amp = np.sqrt(2.0 * K / (np.pi * R))
        phase = -K * R + np.pi / 4.0
        exp_z = np.exp(K * (z + zeta))
        exp_z = np.clip(exp_z, 0, 1e15)
        return -1j * amp * np.exp(1j * phase) * exp_z


def green_function_value(omega: float, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                         xi: np.ndarray, eta: np.ndarray, zeta: np.ndarray,
                         g: float = GRAVITY) -> np.ndarray:
    """Vectorised Green function evaluation.

    Parameters
    ----------
    omega : float
        Angular frequency.
    x, y, z : np.ndarray
        Field point coordinates.
    xi, eta, zeta : np.ndarray
        Source point coordinates.
    g : float
        Gravity.

    Returns
    -------
    np.ndarray
        Green function values (complex).
    """
    gf = GreenFunction(omega, g)
    result = np.zeros_like(x, dtype=complex)
    for i in range(len(x)):
        result[i] = gf.value(x[i], y[i], z[i], xi[i], eta[i], zeta[i])
    return result


def green_function_gradient(omega: float, x: float, y: float, z: float,
                            xi: float, eta: float, zeta: float,
                            g: float = GRAVITY, h: float = 1e-6) -> np.ndarray:
    """Numerical gradient of the Green function w.r.t. field point.

    Returns [∂G/∂x, ∂G/∂y, ∂G/∂z].
    """
    gf = GreenFunction(omega, g)
    G0 = gf.value(x, y, z, xi, eta, zeta)
    Gx = gf.value(x + h, y, z, xi, eta, zeta)
    Gy = gf.value(x, y + h, z, xi, eta, zeta)
    Gz = gf.value(x, y, z + h, xi, eta, zeta)
    return np.array([
        (Gx - G0) / h,
        (Gy - G0) / h,
        (Gz - G0) / h,
    ])


def kochin_function(omega: float, theta: np.ndarray, source_strengths: np.ndarray,
                    panel_x: np.ndarray, panel_y: np.ndarray, panel_z: np.ndarray,
                    g: float = GRAVITY) -> np.ndarray:
    """Compute the Kochin function H(θ) for the far-field wave amplitude.

    The Kochin function is defined as:

        H(θ) = (1/4π) ∮ σ(ξ) exp[K(ζ + iξ·e_θ)] dS

    where e_θ = (cos θ, sin θ) is the horizontal direction and K = ω²/g.

    Parameters
    ----------
    omega : float
        Angular frequency.
    theta : np.ndarray
        Array of angles (rad).
    source_strengths : np.ndarray
        Source strengths on each panel.
    panel_x, panel_y, panel_z : np.ndarray
        Panel centroid coordinates.
    g : float
        Gravity.

    Returns
    -------
    np.ndarray
        Kochin function H(θ) for each angle.
    """
    K = omega ** 2 / g
    H = np.zeros_like(theta, dtype=complex)

    for j in range(len(theta)):
        ct = np.cos(theta[j])
        st = np.sin(theta[j])
        for i in range(len(source_strengths)):
            phase = K * (panel_z[i] + 1j * (panel_x[i] * ct + panel_y[i] * st))
            H[j] += source_strengths[i] * np.exp(phase)

    H /= (4.0 * np.pi)
    return H


def kochin_energy_damping(omega: float, H: np.ndarray, theta: np.ndarray,
                          rho: float = RHO_WATER, g: float = GRAVITY) -> float:
    """Compute radiation damping from Kochin function via energy conservation.

    B_ij = ρg/(2ω) ∮ |H(θ)|² dθ

    Parameters
    ----------
    omega : float
        Angular frequency.
    H : np.ndarray
        Kochin function values.
    theta : np.ndarray
        Angles (must cover 0 to 2π).
    rho : float
        Water density.
    g : float
        Gravity.

    Returns
    -------
    float
        Radiation damping B from energy conservation.
    """
    dtheta = theta[1] - theta[0] if len(theta) > 1 else 2 * np.pi / len(theta)
    integral = np.sum(np.abs(H) ** 2) * dtheta
    return rho * g / (2.0 * omega) * integral
