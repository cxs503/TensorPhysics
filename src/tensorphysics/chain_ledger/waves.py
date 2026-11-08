"""Second-order Stokes wave field used as the *upstream* stage of the chain.

Scope and evidence discipline
-----------------------------
* ``eta`` (free-surface elevation) is **second-order** Stokes in finite depth
  (the second harmonic ``(k a^2/4)(3 - sigma^2)/sigma^3`` reduces to the classic
  ``k a^2 / 2`` in deep water - this identity is a unit test).
* The dynamic pressure is the Bernoulli pressure of the **first-order** potential
  plus the leading ``-rho |grad phi|^2 / 2`` term.  The second-order potential
  ``phi_2`` is *not* solved here, so the pressure field is accurate to
  ``O((ka)^2)`` with a documented missing term.  Nothing downstream claims more.
* Everything here is analytic (REAL): no hard-coded tables, no fitted numbers.

The wave field is the input to chain interface A (pressure -> rigid-body force).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["StokesWave", "dispersion_wavenumber"]

G_ACC = 9.80665


def dispersion_wavenumber(omega: float, depth: float, g: float = G_ACC, tol: float = 1e-13) -> float:
    """Solve ``omega^2 = g k tanh(k h)`` for ``k`` by Newton iteration.

    The deep-water guess is used to start; Newton converges monotonically for
    the physical branch.
    """
    if omega <= 0.0 or depth <= 0.0:
        raise ValueError("omega and depth must be positive")
    k = omega ** 2 / g                       # deep-water limit
    if not np.isfinite(k):
        raise ValueError("non-finite wavenumber")
    for _ in range(200):
        th = np.tanh(k * depth)
        f = g * k * th - omega ** 2
        df = g * (th + k * depth * (1.0 - th * th))
        step = f / df
        k -= step
        if abs(step) <= tol * max(abs(k), 1.0e-30):
            break
    return float(k)


@dataclass(frozen=True)
class StokesWave:
    """Second-order Stokes wave in constant depth.

    Parameters
    ----------
    height : float
        Wave height ``H`` [m] (crest-to-trough of the first-order component).
    period : float
        Wave period ``T`` [s].
    depth : float
        Still-water depth ``h`` [m].
    rho : float
        Water density [kg/m^3].
    g : float
        Gravity [m/s^2].
    """

    height: float
    period: float
    depth: float
    rho: float = 1025.0
    g: float = G_ACC

    # ------------------------------------------------------------- derived
    @property
    def omega(self) -> float:
        return 2.0 * np.pi / self.period

    @property
    def amplitude(self) -> float:
        return 0.5 * self.height

    @property
    def wavenumber(self) -> float:
        return dispersion_wavenumber(self.omega, self.depth, self.g)

    @property
    def wavelength(self) -> float:
        return 2.0 * np.pi / self.wavenumber

    @property
    def sigma(self) -> float:
        return float(np.tanh(self.wavenumber * self.depth))

    @property
    def celerity(self) -> float:
        return self.omega / self.wavenumber

    @property
    def group_velocity(self) -> float:
        kh = self.wavenumber * self.depth
        return 0.5 * self.celerity * (1.0 + 2.0 * kh / np.sinh(2.0 * kh))

    @property
    def steepness(self) -> float:
        return self.wavenumber * self.amplitude

    @property
    def second_harmonic_amplitude(self) -> float:
        """Amplitude of the second-order free-surface component."""
        k, a, s = self.wavenumber, self.amplitude, self.sigma
        return float(0.25 * k * a * a * (3.0 - s * s) / s ** 3)

    # ---------------------------------------------------------- kinematics
    def _theta(self, x, t):
        return self.wavenumber * np.asarray(x, dtype=float) - self.omega * t

    def eta(self, x, t) -> np.ndarray:
        """Second-order free-surface elevation [m] (mean water level = 0)."""
        a2 = self.second_harmonic_amplitude
        th = self._theta(x, t)
        return self.amplitude * np.cos(th) + a2 * np.cos(2.0 * th)

    def eta_first_order(self, x, t) -> np.ndarray:
        return self.amplitude * np.cos(self._theta(x, t))

    def potential(self, x, z, t) -> np.ndarray:
        """First-order velocity potential [m^2/s]."""
        k, a, w = self.wavenumber, self.amplitude, self.omega
        z = np.asarray(z, dtype=float)
        return (a * self.g / w) * np.cosh(k * (z + self.depth)) / np.cosh(k * self.depth) * np.sin(
            self._theta(x, t)
        )

    def velocity(self, x, z, t) -> tuple[np.ndarray, np.ndarray]:
        """First-order particle velocity ``(u, w)`` [m/s], ``w`` positive up.

        ``u = a omega cosh(k(z+h)) / sinh(kh) cos(theta)`` follows from the
        potential above together with the dispersion relation; dividing by
        ``cosh(kh)`` instead of ``sinh(kh)`` is a classic error that shows up as
        a spurious ``tanh(kh)`` factor in the energy flux (the
        ``numerical_energy_flux`` test pins it down).
        """
        k, a, w = self.wavenumber, self.amplitude, self.omega
        z = np.asarray(z, dtype=float)
        th = self._theta(x, t)
        inv_sigma = 1.0 / self.sigma
        ch = np.cosh(k * (z + self.depth)) / np.cosh(k * self.depth) * inv_sigma
        sh = np.sinh(k * (z + self.depth)) / np.cosh(k * self.depth) * inv_sigma
        u = a * w * ch * np.cos(th)
        wx = a * w * sh * np.sin(th)
        return u, wx

    def acceleration(self, x, z, t) -> tuple[np.ndarray, np.ndarray]:
        """First-order particle acceleration ``(du/dt, dw/dt)`` [m/s^2]."""
        k, a, w = self.wavenumber, self.amplitude, self.omega
        z = np.asarray(z, dtype=float)
        th = self._theta(x, t)
        inv_sigma = 1.0 / self.sigma
        ch = np.cosh(k * (z + self.depth)) / np.cosh(k * self.depth) * inv_sigma
        sh = np.sinh(k * (z + self.depth)) / np.cosh(k * self.depth) * inv_sigma
        ax = a * w * w * ch * np.sin(th)
        az = -a * w * w * sh * np.cos(th)
        return ax, az

    # ------------------------------------------------------------ pressure
    def dynamic_pressure(self, x, z, t) -> np.ndarray:
        """Bernoulli dynamic pressure [Pa] (first order + ``-rho |grad phi|^2/2``)."""
        k, a, w, rho = self.wavenumber, self.amplitude, self.omega, self.rho
        z = np.asarray(z, dtype=float)
        th = self._theta(x, t)
        ch = np.cosh(k * (z + self.depth)) / np.cosh(k * self.depth)
        p_lin = rho * self.g * a * ch * np.cos(th)
        u, wx = self.velocity(x, z, t)
        return p_lin - 0.5 * rho * (u * u + wx * wx)

    def pressure(self, x, z, t, atmosphere: float = 0.0) -> np.ndarray:
        """Total pressure [Pa]: ``p_dyn - rho g z + p_atm`` (``z`` up from MWL)."""
        z = np.asarray(z, dtype=float)
        return self.dynamic_pressure(x, z, t) - self.rho * self.g * z + atmosphere

    # -------------------------------------------------------------- energy
    @property
    def energy_density(self) -> float:
        """Wave energy per unit horizontal area [J/m^2] (second order in H)."""
        k, a = self.wavenumber, self.amplitude
        e1 = 0.5 * self.rho * self.g * a * a
        # second-order correction to the mean energy of a Stokes wave (deep water
        # limit of the standard expansion); kept small and reported for audit.
        e2 = 0.5 * self.rho * self.g * a * a * (k * a) ** 2
        return float(e1 + e2)

    @property
    def energy_flux(self) -> float:
        """Mean energy flux ``E c_g`` [W/m]."""
        return float(self.energy_density * self.group_velocity)

    def numerical_energy_flux(self, x: float, n_z: int = 400, n_t: int = 256) -> float:
        """Time-averaged vertical integral of the energy flux at station ``x``.

        ``F = < int_{-h}^{eta} (p_dyn + 1/2 rho |u|^2) u dz >``  [W/m].

        This is an independent numerical evaluation of the same physical quantity
        as :attr:`energy_flux`; agreement between the two is the wave-module
        energy validation.
        """
        z = np.linspace(-self.depth, 0.0, n_z)
        t = np.linspace(0.0, self.period, n_t, endpoint=False)
        total = 0.0
        for tt in t:
            u, w = self.velocity(x, z, tt)
            p = self.dynamic_pressure(x, z, tt)
            integrand = (p + 0.5 * self.rho * (u * u + w * w)) * u
            # extend the integration to the instantaneous free surface (linear):
            integrand = np.append(integrand, 0.5 * self.rho * u[-1] ** 2 * u[-1])
            zz = np.append(z, self.eta(x, tt))
            total += float(np.trapezoid(integrand, zz))
        return total / n_t

    def dispersion_residual(self) -> float:
        """Relative residual of ``omega^2 = g k tanh(kh)``."""
        k = self.wavenumber
        return float(abs(self.g * k * np.tanh(k * self.depth) - self.omega ** 2) / self.omega ** 2)