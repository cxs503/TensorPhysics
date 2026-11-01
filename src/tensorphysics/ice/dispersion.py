"""Flexural-gravity dispersion of an elastic ice plate floating on water.

Theory
------
The ice cover is a Kirchhoff-Love plate of flexural rigidity
``D = E h^3 / (12 (1 - nu^2))`` resting on an inviscid, irrotational water
layer of depth ``H`` with a rigid bottom.  The velocity potential satisfies
Laplace's equation

    d^2(phi)/dz^2 - k^2 phi = 0,          (2-D Fourier mode of wavenumber k)

with the kinematic (ice-water) condition at the interface ``z = 0``

    d(phi)/dz = w_dot ,

and the impermeability condition at the bottom ``d(phi)/dz = 0`` at ``z = -H``.
The solution ``phi = A cosh(k (z + H))`` gives, through the unsteady Bernoulli
relation ``p = -rho_w d(phi)/dt``, a wavenumber dependent *added mass*

    m_a(k) = rho_w / (k tanh(k H))        (finite depth),   rho_w / k (H -> inf).

Free flexural-gravity waves then obey

    omega^2 = (D k^4 + rho_w g) / (rho_i h + m_a(k)) .

The static (omega -> 0) limit gives the classical characteristic length
``l_c = (D / (rho_w g))^(1/4)`` and the static deflection wavelength
``lambda_c = 2 pi l_c``.

Notes / validity
----------------
* Free-surface (gravity-wave) dynamics in the ice-free region is *not* solved:
  a continuous cover suppresses those waves, and the impact time scale
  (O(1 s)) is short compared with the gravity-wave period, so the fluid
  response is modelled as an instantaneous (frequency-independent) added-mass
  operator.  This is the standard "quasi-static fluid" assumption of
  ice-cover impact models and is the dominant limitation of the model.
* The uniform vertical translation (k = 0) mode is excluded: with a rigid
  bottom and an infinite cover, uniform vertical motion would require an
  infinite lateral flux, so the added mass diverges.  Deflections are taken
  zero-mean, exactly as in the classical infinite-plate solution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRAVITY = 9.81
RHO_WATER = 1025.0
RHO_ICE = 917.0


def plate_rigidity(E: float, h: float, nu: float = 0.3) -> float:
    """Flexural rigidity ``D = E h^3 / (12 (1 - nu^2))`` [N m]."""
    if E <= 0.0 or h <= 0.0:
        raise ValueError("E and h must be positive")
    if not -0.99 < nu < 0.5:
        raise ValueError("Poisson ratio out of range")
    return E * h**3 / (12.0 * (1.0 - nu**2))


def characteristic_length(D: float, rho_w: float = RHO_WATER, g: float = GRAVITY) -> float:
    """Hetenyi characteristic length ``l_c = (D / (rho_w g))^(1/4)`` [m]."""
    if D <= 0.0 or rho_w <= 0.0 or g <= 0.0:
        raise ValueError("D, rho_w and g must be positive")
    return (D / (rho_w * g)) ** 0.25


def added_mass_per_area(
    wavenumber,
    rho_w: float = RHO_WATER,
    depth: float | None = None,
):
    """Potential-flow added mass per unit plate area [kg/m^2].

    ``m_a = rho_w / (k tanh(k H))`` for finite depth, ``rho_w / k`` for
    infinite depth.  ``m_a -> inf`` for ``k -> 0`` (uniform translation mode,
    see module docstring).
    """
    k = np.asarray(wavenumber, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if depth is None:
            m = rho_w / k
        else:
            if depth <= 0.0:
                raise ValueError("depth must be positive")
            m = rho_w / (k * np.tanh(k * depth))
    m = np.where(np.abs(k) < 1e-12, np.inf, m)
    return m if m.ndim else float(m)


def depth_fem_added_mass(
    wavenumber: float,
    rho_w: float = RHO_WATER,
    depth: float = 50.0,
    n_elem: int = 200,
) -> float:
    """Independent verification channel for :func:`added_mass_per_area`.

    Solves ``-k^2 phi + phi'' = 0`` on ``[-H, 0]`` with linear finite elements,
    Neumann data ``phi'(0) = 1`` (unit ice vertical velocity) and
    ``phi'(-H) = 0``.  Weak form

        int (k^2 phi psi + phi' psi') dz = [phi' psi]_{-H}^0 = 1 * psi(0),

    so the load vector is one at the interface node and the added mass is
    ``rho_w * phi(0)`` for a unit velocity amplitude.  The depth mesh
    convergence is O(n_elem^-2); this is a genuinely independent discretisation
    of the same boundary-value problem (used in the test suite).
    """
    if depth <= 0.0:
        raise ValueError("depth must be positive")
    if n_elem < 2:
        raise ValueError("n_elem must be >= 2")
    k = abs(float(wavenumber))
    if k < 1e-12:
        return float("inf")
    n_nodes = n_elem + 1
    z = np.linspace(-depth, 0.0, n_nodes)
    le = depth / n_elem
    K = np.zeros((n_nodes, n_nodes))
    M = np.zeros((n_nodes, n_nodes))
    for e in range(n_elem):
        Ke = (1.0 / le) * np.array([[1.0, -1.0], [-1.0, 1.0]])
        Me = (le / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])
        idx = [e, e + 1]
        for i in range(2):
            for j in range(2):
                K[idx[i], idx[j]] += Ke[i, j]
                M[idx[i], idx[j]] += Me[i, j]
    A = K + k * k * M
    f = np.zeros(n_nodes)
    f[-1] = 1.0  # interface node z = 0 (natural Neumann data)
    phi = np.linalg.solve(A, f)
    return float(rho_w * phi[-1])


@dataclass(frozen=True)
class FlexuralDispersion:
    """Dispersion relation ``omega(k)`` of a floating elastic plate.

    All arguments are SI; ``depth=None`` means an infinitely deep ocean.
    """

    rigidity: float = 8.24e8
    ice_thickness: float = 1.0
    rho_ice: float = RHO_ICE
    rho_w: float = RHO_WATER
    g: float = GRAVITY
    depth: float | None = 50.0

    # ---------------------------------------------------------------- helpers
    @classmethod
    def from_geometry(
        cls,
        E: float,
        h: float,
        nu: float = 0.3,
        rho_ice: float = RHO_ICE,
        rho_w: float = RHO_WATER,
        g: float = GRAVITY,
        depth: float | None = 50.0,
    ) -> "FlexuralDispersion":
        return cls(
            rigidity=plate_rigidity(E, h, nu),
            ice_thickness=h,
            rho_ice=rho_ice,
            rho_w=rho_w,
            g=g,
            depth=depth,
        )

    def stiffness(self, k) -> np.ndarray:
        """Restoring stiffness per unit area ``D k^4 + rho_w g`` [N/m^3]."""
        k = np.asarray(k, dtype=float)
        return self.rigidity * k**4 + self.rho_w * self.g

    def mass(self, k) -> np.ndarray:
        """Total inertia per unit area ``rho_i h + m_a(k)`` [kg/m^2]."""
        k = np.asarray(k, dtype=float)
        return self.rho_ice * self.ice_thickness + added_mass_per_area(
            k, rho_w=self.rho_w, depth=self.depth
        )

    def omega(self, k) -> np.ndarray:
        """Angular frequency [rad/s] of the flexural-gravity wave ``k``."""
        k = np.asarray(k, dtype=float)
        K = self.stiffness(k)
        m = self.mass(k)
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.sqrt(K / m)
        w = np.where(np.isfinite(w), w, 0.0)
        kk = np.abs(k)
        # long-wave limit k -> 0: m_a -> inf imposes zero mean motion
        w = np.where(kk < 1e-9, 0.0, w)
        return w if np.ndim(w) else float(w)

    def phase_speed(self, k) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            c = self.omega(k) / k
        return c

    def group_speed(self, k, dk: float | None = None) -> np.ndarray:
        """``d omega / d k`` by central differencing of the exact relation."""
        k = np.asarray(k, dtype=float)
        h = dk if dk is not None else np.maximum(np.abs(k) * 1e-5, 1e-7)
        return (self.omega(k + h) - self.omega(k - h)) / (2.0 * h)

    def wavelength(self, k) -> np.ndarray:
        return 2.0 * np.pi / np.asarray(k, dtype=float)

    def characteristic_length(self) -> float:
        return characteristic_length(self.rigidity, self.rho_w, self.g)

    def static_wavelength(self) -> float:
        """Wavelength of the static deflection pattern ``2 pi l_c`` [m]."""
        return 2.0 * np.pi * self.characteristic_length()

    def k_from_omega(self, omega: float, k_lo: float = 1e-6, k_hi: float = 10.0) -> float:
        """Invert ``omega(k)`` (monotone branch, bisection to 1e-12 relative)."""
        from scipy.optimize import brentq

        target = float(omega)
        if target <= 0.0:
            raise ValueError("omega must be positive")

        def f(kk: float) -> float:
            return float(self.omega(kk)) - target

        return float(brentq(f, k_lo, k_hi, xtol=1e-14, rtol=1e-14, maxiter=200))

    def minimum_phase_speed(self, k_log_lo: float = -8.0, k_log_hi: float = 3.0, n: int = 4000):
        """Minimum phase speed of the ice-coupled wave system [m/s].

        Loads moving faster than this radiate flexural waves (``supersonic``
        regime); a ramming icebreaker at 1-3 m/s is deep in the subsonic
        (quasi-static) regime.
        """
        k = np.logspace(k_log_lo, k_log_hi, n)
        c = self.phase_speed(k)
        c = np.where(np.isfinite(c) & (c > 0.0), c, np.inf)
        i = int(np.argmin(c))
        return float(c[i]), float(k[i])