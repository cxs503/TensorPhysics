"""Pekeris waveguide: uniform water layer over semi-infinite bottom.

The Pekeris model is the canonical shallow-water acoustic waveguide:
a homogeneous water layer of depth H with sound speed c0 and density ρ0,
over a semi-infinite fluid bottom with sound speed c1 and density ρ1.

For **trapped modes** (c1 < c0, i.e., the bottom is slower than the water),
the dispersion relation (Pekeris, 1948) is:

    tan(γ0 H) = (ρ1/ρ0) · (γ0 / |γ1|)

where
    γ0 = √(k² - k0²),   k0 = ω/c0   (vertical wavenumber in water, real)
    γ1 = √(k² - k1²),   k1 = ω/c1   (vertical wavenumber in bottom, imaginary)

For trapped modes (k0 < k < k1), γ0 is real and γ1 is imaginary,
so the dispersion relation becomes:

    tan(γ0 H) = (ρ1/ρ0) · (γ0 / |γ1|)

The horizontal wavenumber k_n of mode n is found by solving this
transcendental equation.  The mode shape in the water column is:

    p_n(z) = A_n · sin(γ0_n z)

with the normalization A_n chosen so that the modes are orthonormal
(unit power per mode).

References
----------
- Pekeris, C.L. (1948). "Theory of propagation of explosive sound in
  shallow water." Geol. Soc. Am. Mem. 27.
- Jensen, Kuperman, Porter, Schmidt (2011). "Computational Ocean
  Acoustics", 2nd ed., Springer. Ch. 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import RHO_WATER, C_WATER, P_REF, TL


# ---------------------------------------------------------------------------
# Pekeris dispersion relation
# ---------------------------------------------------------------------------
def pekeris_dispersion(
    k: float,
    omega: float,
    depth: float,
    c0: float,
    c1: float,
    rho0: float,
    rho1: float,
) -> float:
    """Evaluate the Pekeris dispersion relation residual.

    F(k) = tan(γ0 H) - (ρ1/ρ0) · (γ0 / |γ1|)

    A mode exists where F(k) = 0 with k0 < k < k1 (trapped modes,
    c1 < c0).

    Parameters
    ----------
    k : float
        Horizontal wavenumber [1/m].
    omega : float
        Angular frequency [rad/s].
    depth : float
        Water depth [m].
    c0 : float
        Water sound speed [m/s].
    c1 : float
        Bottom sound speed [m/s].
    rho0 : float
        Water density [kg/m³].
    rho1 : float
        Bottom density [kg/m³].

    Returns
    -------
    float
        Dispersion relation residual F(k).
    """
    k0 = omega / c0
    k1 = omega / c1

    if k <= k0:
        return np.inf
    if k >= k1:
        return -np.inf

    gamma0 = np.sqrt(k**2 - k0**2)
    gamma1_abs = np.sqrt(k1**2 - k**2)

    return np.tan(gamma0 * depth) - (rho1 / rho0) * (gamma0 / gamma1_abs)


def _pole_k(m: int, omega: float, depth: float, c0: float) -> float:
    """Wavenumber of the m-th pole of tan(γ0 H).

    Poles at γ0 H = (m + 1/2)π → k = √(k0² + ((m+1/2)π/H)²).
    """
    k0 = omega / c0
    gamma0_pole = (m + 0.5) * np.pi / depth
    return np.sqrt(k0**2 + gamma0_pole**2)


def _find_all_roots(
    omega: float,
    depth: float,
    c0: float,
    c1: float,
    rho0: float,
    rho1: float,
    n_bisect: int = 100,
) -> list[float]:
    """Find all roots of the Pekeris dispersion relation in (k0, k1).

    Uses the pole structure: the n-th mode has its root between pole n
    and pole n+1 of tan(γ0 H).  Each root is found by bisection on the
    interval between consecutive poles, with a small offset from the
    poles to avoid the tan() singularity.

    Parameters
    ----------
    omega : float
        Angular frequency [rad/s].
    depth : float
        Water depth [m].
    c0 : float
        Water sound speed [m/s].
    c1 : float
        Bottom sound speed [m/s].
    rho0 : float
        Water density [kg/m³].
    rho1 : float
        Bottom density [kg/m³].
    n_bisect : int
        Number of bisection iterations per root.

    Returns
    -------
    list[float]
        All roots k_n in (k0, k1), sorted ascending.
    """
    k0 = omega / c0
    k1 = omega / c1

    if k1 <= k0:
        return []

    roots = []
    m = 0
    while True:
        k_pole_m = _pole_k(m, omega, depth, c0)
        k_pole_m1 = _pole_k(m + 1, omega, depth, c0)

        # Small offset from the poles (avoid tan() singularity)
        delta = 1e-6 * (k1 - k0)
        k_lo = max(k_pole_m + delta, k0 + 1e-8 * (k1 - k0))
        k_hi = min(k_pole_m1 - delta, k1 - 1e-8 * (k1 - k0))

        if k_lo >= k_hi:
            break  # no more modes in the band

        # At k_lo (just above pole m), tan(γ0 H) → +∞, so F(k_lo) > 0.
        # At k_hi (just below pole m+1), tan(γ0 H) → -∞, so F(k_hi) < 0.
        f_lo = pekeris_dispersion(k_lo, omega, depth, c0, c1, rho0, rho1)
        f_hi = pekeris_dispersion(k_hi, omega, depth, c0, c1, rho0, rho1)

        if f_lo * f_hi > 0:
            # No sign change → no root in this interval
            m += 1
            continue

        # Bisection
        for _ in range(n_bisect):
            k_mid = 0.5 * (k_lo + k_hi)
            f_mid = pekeris_dispersion(k_mid, omega, depth, c0, c1, rho0, rho1)
            if f_lo * f_mid < 0:
                k_hi = k_mid
                f_hi = f_mid
            else:
                k_lo = k_mid
                f_lo = f_mid

        roots.append(0.5 * (k_lo + k_hi))
        m += 1

        if m > 1000:
            break

    return sorted(roots)


def pekeris_mode_eigenvalue(
    mode_index: int,
    omega: float,
    depth: float,
    c0: float,
    c1: float,
    rho0: float,
    rho1: float,
    n_bisect: int = 200,
) -> float:
    """Find the horizontal wavenumber k_n of mode n in the Pekeris waveguide.

    The dispersion relation tan(γ0 H) = (ρ1/ρ0)(γ0/|γ1|) has a discrete
    set of solutions k_n in (k0, k1).  Mode n corresponds to the n-th
    solution (n = 0, 1, 2, ...).

    Parameters
    ----------
    mode_index : int
        Mode number (0-based).
    omega : float
        Angular frequency [rad/s].
    depth : float
        Water depth [m].
    c0 : float
        Water sound speed [m/s].
    c1 : float
        Bottom sound speed [m/s].
    rho0 : float
        Water density [kg/m³].
    rho1 : float
        Bottom density [kg/m³].
    n_bisect : int
        Number of bisection iterations.

    Returns
    -------
    float
        Horizontal wavenumber k_n [1/m].
    """
    k0 = omega / c0
    k1 = omega / c1

    if k1 <= k0:
        raise ValueError(
            f"Bottom speed c1={c1} must be < water speed c0={c0} "
            "for trapped modes (k1 > k0 required)"
        )

    # Find all roots and pick the mode_index-th one
    all_roots = _find_all_roots(omega, depth, c0, c1, rho0, rho1)

    if mode_index >= len(all_roots):
        return k1  # mode is cut off

    return all_roots[mode_index]


def pekeris_mode_shape(
    k_n: float,
    z: float | np.ndarray,
    omega: float,
    depth: float,
    c0: float,
    rho0: float,
    rho1: float,
) -> float | np.ndarray:
    """Mode shape p_n(z) in the water column.

    p_n(z) = A_n · sin(γ0_n z)

    with γ0_n = √(k_n² - k0²).

    Parameters
    ----------
    k_n : float
        Horizontal wavenumber of mode n [1/m].
    z : float or array
        Depth coordinate [m] (0 = surface, positive downward).
    omega : float
        Angular frequency [rad/s].
    depth : float
        Water depth [m].
    c0 : float
        Water sound speed [m/s].
    rho0 : float
        Water density [kg/m³].
    rho1 : float
        Bottom density [kg/m³].

    Returns
    -------
    float or array
        Mode shape amplitude (unnormalized, sin(γ0 z)).
    """
    k0 = omega / c0
    gamma0 = np.sqrt(k_n**2 - k0**2)
    z = np.asarray(z, dtype=float)
    return np.sin(gamma0 * z)


@dataclass
class PekerisWaveguide:
    """Pekeris waveguide: uniform water layer over semi-infinite bottom.

    Parameters
    ----------
    depth : float
        Water depth [m].
    c0 : float
        Water sound speed [m/s].
    c1 : float
        Bottom sound speed [m/s].  Must be < c0 for trapped modes.
    rho0 : float
        Water density [kg/m³].
    rho1 : float
        Bottom density [kg/m³].
    """

    depth: float
    c0: float = C_WATER
    c1: float = 1450.0
    rho0: float = RHO_WATER
    rho1: float = 2000.0

    def __post_init__(self):
        if self.depth <= 0:
            raise ValueError(f"Water depth must be positive, got {self.depth}")
        if self.c1 >= self.c0:
            raise ValueError(
                f"Bottom speed c1={self.c1} must be < water speed c0={self.c0} "
                "for trapped modes (k1 > k0 required)"
            )

    def mode_eigenvalues(self, frequency: float, n_modes: int | None = None) -> list[float]:
        """Compute the horizontal wavenumbers k_n of all trapped modes.

        Parameters
        ----------
        frequency : float
            Source frequency [Hz].
        n_modes : int, optional
            Maximum number of modes to compute.  If None, compute all
            modes with k_n < k1 (trapped modes).

        Returns
        -------
        list[float]
            Horizontal wavenumbers k_n [1/m], sorted ascending.
        """
        omega = 2.0 * np.pi * frequency
        all_roots = _find_all_roots(omega, self.depth, self.c0, self.c1,
                                    self.rho0, self.rho1)

        if n_modes is not None:
            all_roots = all_roots[:n_modes]

        return all_roots

    def mode_amplitudes(self, frequency: float, n_modes: int | None = None) -> list[float]:
        """Compute the mode excitation amplitudes A_n.

        For a point source at depth z_s, the mode amplitude is

            A_n = p_n(z_s) / (ρ0 · √(2π k_n r))

        (in the cylindrical-spreading normalization).  We return the
        normalized amplitude factor p_n(z_s) · √(2/(ρ0 H_eff)).

        Parameters
        ----------
        frequency : float
            Source frequency [Hz].
        n_modes : int, optional
            Number of modes.

        Returns
        -------
        list[float]
            Mode amplitude factors.
        """
        omega = 2.0 * np.pi * frequency
        k0 = omega / self.c0
        eigenvalues = self.mode_eigenvalues(frequency, n_modes)

        amplitudes = []
        for k_n in eigenvalues:
            gamma0 = np.sqrt(k_n**2 - k0**2)
            # Effective depth: H + 1/|γ1| (bottom penetration length)
            k1 = omega / self.c1
            gamma1_abs = np.sqrt(k1**2 - k_n**2)
            H_eff = self.depth + 1.0 / gamma1_abs
            # Normalization: A_n = √(2/(ρ0 H_eff))
            A_n = np.sqrt(2.0 / (self.rho0 * H_eff))
            amplitudes.append(A_n)

        return amplitudes

    def tl(self, frequency: float, source_depth: float, r: float | np.ndarray,
           z: float, n_modes: int | None = None) -> float | np.ndarray:
        """Transmission loss in the Pekeris waveguide.

        TL(r, z) = -20 log10(|Σ_n p_n(r, z)| / p_ref)

        with the modal sum

            p(r, z) = Σ_n A_n · sin(γ0_n z_s) · sin(γ0_n z)
                      · e^{i k_n r} / √(k_n r)

        (cylindrical spreading).

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
        n_modes : int, optional
            Number of modes to include.

        Returns
        -------
        float or array
            Transmission loss [dB re 1 μPa].
        """
        omega = 2.0 * np.pi * frequency
        k0 = omega / self.c0
        r = np.asarray(r, dtype=float)

        eigenvalues = self.mode_eigenvalues(frequency, n_modes)
        amplitudes = self.mode_amplitudes(frequency, n_modes)

        # Modal sum
        p_total = np.zeros_like(r, dtype=complex)
        for k_n, A_n in zip(eigenvalues, amplitudes):
            gamma0 = np.sqrt(k_n**2 - k0**2)
            # Mode shape at source and receiver
            psi_s = np.sin(gamma0 * source_depth)
            psi_z = np.sin(gamma0 * z)
            # Modal contribution with cylindrical spreading
            p_n = A_n * psi_s * psi_z * np.exp(1j * k_n * r) / np.sqrt(k_n * r)
            p_total += p_n

        # TL = -20 log10(|p| / p_ref)
        p_abs = np.abs(p_total)
        p_abs = np.where(p_abs < 1e-30, 1e-30, p_abs)
        return 20.0 * np.log10(p_abs / P_REF)


def pekeris_tl(
    frequency: float,
    source_depth: float,
    r: float | np.ndarray,
    z: float,
    depth: float,
    c0: float = C_WATER,
    c1: float = 1450.0,
    rho0: float = RHO_WATER,
    rho1: float = 2000.0,
    n_modes: int | None = None,
) -> float | np.ndarray:
    """Convenience wrapper: Pekeris waveguide TL.

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
    depth : float
        Water depth [m].
    c0, c1 : float
        Water / bottom sound speeds [m/s].
    rho0, rho1 : float
        Water / bottom densities [kg/m³].
    n_modes : int, optional
        Number of modes.

    Returns
    -------
    float or array
        Transmission loss [dB re 1 μPa].
    """
    wg = PekerisWaveguide(depth=depth, c0=c0, c1=c1, rho0=rho0, rho1=rho1)
    return wg.tl(frequency, source_depth, r, z, n_modes)
