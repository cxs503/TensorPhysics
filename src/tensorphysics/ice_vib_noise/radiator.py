"""Baffled planar radiator: far-field pressure, radiated power, radiation damping.

Physical model
--------------
The part of the hull that radiates is idealised as a **plane baffle** in the
``z = 0`` plane carrying a normal surface-velocity distribution ``v(x, y)``.
For a monochromatic component the Rayleigh-Sommerfeld integral over the baffled
plane gives, in the far field at range ``r`` and direction unit vector ``n``
(time convention ``exp(-i w t)``),

    p(r, n) = (i rho w / (2 pi r)) exp(i k r) * S(n),
    S(n)    = int_S v(x, y) exp(-i k n.x) dS.                      (1)

``S`` is the plane-wave (spatial Fourier) content of the velocity field at the
radiating wavenumber, so cancellation between patches that move in antiphase is
captured exactly: a hull-girder 2-node mode radiates like a dipole (efficiency
falls as ``(k L)^2`` at low ``k L``), a 3-node mode like a quadrupole.  Nothing
in this file contains a tuned efficiency: the radiation efficiency is *computed*
from (1).

The radiated acoustic power follows from integrating the far-field intensity
over the half space

    P = int_{hemisphere} |p|^2 / (2 rho c) r^2 dOmega
      = (rho w^2 / (8 pi^2 c)) int |S(n)|^2 dOmega.                (2)

Two exact results are used as verification anchors (see tests):

* uniform circular piston radius ``a``: directivity ``2 J1(ka sin th)/(ka sin th)``
  and radiation efficiency ``sigma = 1 - J1(2ka)/(ka)``;
* small-source limit: ``S -> v A`` so (1) degenerates to a monopole of strength
  ``Q = 2 v A`` (the factor 2 is the baffle doubling the pressure).

Validity limits (declared, not hidden): the baffle ignores hull curvature,
sea-surface/ice-cover reflection and refraction; the model is a far-field
radiation model, so grazing angles are not meaningful and are excluded from
the assessment directions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "RadiatorSurface",
    "far_field_pressure",
    "radiated_power",
    "radiation_efficiency",
    "piston_radiation_efficiency",
    "piston_directivity",
]

#: Reference pressure for underwater acoustics (1 uPa).
P_REF_PA = 1.0e-6


def _angle_grid(n_theta: int, n_phi: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quadrature nodes/weights for the half-space ``theta in [0, pi/2)``.

    Gauss-Legendre in ``theta`` with the ``sin(theta)`` Jacobian folded into the
    weights, and the trapezoidal rule in ``phi`` (periodic).
    """
    x, w = np.polynomial.legendre.leggauss(n_theta)
    theta = 0.25 * math.pi * (x + 1.0)  # maps [-1, 1] -> [0, pi/2]
    w_theta = 0.25 * math.pi * w * np.sin(theta)
    phi = (np.arange(n_phi) + 0.5) * (2.0 * math.pi / n_phi)
    w_phi = np.full(n_phi, 2.0 * math.pi / n_phi)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    W = np.outer(w_theta, w_phi)
    return TH.ravel(), PH.ravel(), W.ravel()


@dataclass(frozen=True)
class RadiatorSurface:
    """Discretised baffled planar radiating surface.

    The surface is a rectangular patch grid ``nx x ny`` in the ``z = 0`` plane:

    * ``x`` runs along the ship length, ``x = 0`` at the bow (``x`` increasing
      aft), so the radiating plane is the hull bottom / side footprint;
    * ``y`` runs athwartships, ``y = 0`` at the centreline.

    Radiating area, patch areas and the fluid properties are fixed at
    construction so that every power calculation is auditable.
    """

    x_m: np.ndarray
    y_m: np.ndarray
    area_m2: np.ndarray
    fluid_density_kg_m3: float = 1025.0
    sound_speed_m_s: float = 1500.0

    def __post_init__(self) -> None:
        x = np.asarray(self.x_m, dtype=float)
        y = np.asarray(self.y_m, dtype=float)
        a = np.asarray(self.area_m2, dtype=float)
        if not (x.shape == y.shape == a.shape) or x.ndim != 1:
            raise ValueError("x_m, y_m, area_m2 must be 1-D arrays of equal length")
        if np.any(a <= 0.0):
            raise ValueError("all patch areas must be positive")
        if self.fluid_density_kg_m3 <= 0.0 or self.sound_speed_m_s <= 0.0:
            raise ValueError("fluid density and sound speed must be positive")
        for name in ("x_m", "y_m", "area_m2"):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=float))

    # -- construction helpers -------------------------------------------------
    @classmethod
    def from_grid(
        cls,
        length_m: float,
        width_m: float,
        n_x: int = 40,
        n_y: int = 4,
        fluid_density_kg_m3: float = 1025.0,
        sound_speed_m_s: float = 1500.0,
    ) -> "RadiatorSurface":
        """Rectangular footprint ``length_m x width_m`` split into ``n_x * n_y`` patches."""
        if min(length_m, width_m) <= 0.0:
            raise ValueError("length and width must be positive")
        if n_x < 1 or n_y < 1:
            raise ValueError("n_x and n_y must be >= 1")
        dx = length_m / n_x
        dy = width_m / n_y
        xs = (np.arange(n_x) + 0.5) * dx
        ys = (np.arange(n_y) + 0.5) * dy - 0.5 * width_m
        XX, YY = np.meshgrid(xs, ys, indexing="ij")
        AA = np.full(XX.size, dx * dy)
        return cls(
            x_m=XX.ravel(),
            y_m=YY.ravel(),
            area_m2=AA,
            fluid_density_kg_m3=fluid_density_kg_m3,
            sound_speed_m_s=sound_speed_m_s,
        )

    # -- properties -----------------------------------------------------------
    @property
    def total_area_m2(self) -> float:
        return float(np.sum(self.area_m2))

    @property
    def wavenumber(self) -> float:
        return 1.0 / self.sound_speed_m_s

    def mean_square_velocity(self, velocity_amplitude: np.ndarray) -> float:
        """Area-averaged ``<|v|^2>`` for a complex amplitude distribution."""
        v = np.asarray(velocity_amplitude, dtype=complex)
        if v.shape != self.area_m2.shape:
            raise ValueError("velocity array must match the patch set")
        return float(np.sum(np.abs(v) ** 2 * self.area_m2) / self.total_area_m2)

    # -- far field ------------------------------------------------------------
    def plane_wave_amplitude(
        self,
        velocity_amplitude: np.ndarray,
        omega: float,
        theta: np.ndarray,
        phi: np.ndarray,
    ) -> np.ndarray:
        """``S(n) = int_S v exp(-i k n.x) dS`` for a batch of directions.

        ``velocity_amplitude`` is either one vector of length ``n_patches``
        (single frequency) or an array ``(n_freq, n_patches)``; the return value
        is then ``(n_dir,)`` or ``(n_freq, n_dir)``.
        """
        v = np.asarray(velocity_amplitude, dtype=complex)
        single = v.ndim == 1
        if single:
            v = v[None, :]
        if v.shape[1] != self.area_m2.size:
            raise ValueError("velocity array must match the patch set")
        k = float(omega) / self.sound_speed_m_s
        s_th = np.sin(np.asarray(theta, dtype=float))
        phase = (
            np.outer(k * s_th * np.cos(np.asarray(phi, dtype=float)), self.x_m)
            + np.outer(k * s_th * np.sin(np.asarray(phi, dtype=float)), self.y_m)
        )
        kernels = np.exp(-1j * phase) * self.area_m2[None, :]  # (n_dir, n_patch)
        s = v @ kernels.T
        return s[0] if single else s

    def far_field_pressure(
        self,
        velocity_amplitude: np.ndarray,
        omega: float,
        range_m: float,
        theta_rad: float,
        phi_rad: float = 0.0,
    ) -> np.ndarray:
        """Far-field pressure amplitude spectrum at ``(range, theta, phi)``.

        ``velocity_amplitude`` may be a batch ``(n_freq, n_patch)``; the result
        then has shape ``(n_freq,)`` and is complex (phase referenced to ``r``).
        """
        if range_m <= 0.0:
            raise ValueError("range must be positive")
        k = float(omega) / self.sound_speed_m_s
        amp = self.plane_wave_amplitude(velocity_amplitude, omega, [theta_rad], [phi_rad])
        # Eq. (1): p = i rho w /(2 pi r) exp(ikr) S
        pref = 1j * self.fluid_density_kg_m3 * float(omega) / (2.0 * math.pi * range_m) * np.exp(
            1j * k * range_m
        )
        if np.ndim(amp) == 1:  # single frequency -> scalar pressure
            return complex(pref * amp[0])
        return pref * np.asarray(amp)[:, 0]

    def _direction_geometry(
        self, theta: np.ndarray, phi: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Per-direction projection distances ``n.x`` split into its x and y parts."""
        s_th = np.sin(theta)
        dx = np.outer(np.cos(phi), self.x_m) * s_th[:, None]
        dy = np.outer(np.sin(phi), self.y_m) * s_th[:, None]
        return dx, dy

    def radiated_power(
        self,
        velocity_amplitude: np.ndarray,
        omega: Union[float, np.ndarray],
        n_theta: int = 48,
        n_phi: int = 32,
        chunk: int = 16,
    ) -> np.ndarray:
        """Radiated acoustic power ``P(w)`` [W] from Eq. (2).

        ``velocity_amplitude`` is ``(n_patch,)`` for a single frequency or
        ``(n_freq, n_patch)``.  ``omega`` is a scalar (shared by all rows) or an
        array of length ``n_freq`` — the latter is what the frequency dependent
        radiation resistance needs.  The angular quadrature is verified against
        the piston closed form (tests/test_ice_vib_noise_chain.py).
        """
        v = np.asarray(velocity_amplitude, dtype=complex)
        single = v.ndim == 1
        if single:
            v = v[None, :]
        if v.shape[1] != self.area_m2.size:
            raise ValueError("velocity array must match the patch set")
        theta, phi, w = _angle_grid(n_theta, n_phi)
        dx, dy = self._direction_geometry(theta, phi)
        omega_arr = np.asarray(omega, dtype=float)
        if omega_arr.ndim == 0:
            omega_arr = np.full(v.shape[0], float(omega_arr))
        if omega_arr.size != v.shape[0]:
            raise ValueError("omega must be scalar or have one entry per frequency row")
        out = np.empty(v.shape[0], dtype=float)
        pref = self.fluid_density_kg_m3 / (8.0 * math.pi ** 2 * self.sound_speed_m_s)
        for start in range(0, v.shape[0], chunk):
            block = v[start : start + chunk]
            om_block = omega_arr[start : start + chunk]
            k_block = om_block / self.sound_speed_m_s
            # (n_block, n_dir): S(n) = sum_p v_p A_p exp(-i k (dx + dy))
            e_x = np.exp(-1j * (k_block[:, None, None] * dx[None, :, :])) * self.area_m2[None, None, :]
            e_y = np.exp(-1j * (k_block[:, None, None] * dy[None, :, :]))
            s = np.einsum("bfp,bfp,bp->bf", e_x, e_y, block)
            integral = (np.abs(s) ** 2) @ w
            out[start : start + chunk] = pref * om_block ** 2 * integral
        return out[0] if single else out

    def radiation_efficiency(
        self,
        velocity_amplitude: np.ndarray,
        omega: float,
        n_theta: int = 48,
        n_phi: int = 32,
    ) -> float:
        """``sigma = P / (0.5 rho c A <|v|^2>)`` from the computed power."""
        v = np.asarray(velocity_amplitude, dtype=complex)
        power = float(self.radiated_power(v, omega, n_theta, n_phi))
        ms = self.mean_square_velocity(v)
        if ms <= 0.0:
            raise ValueError("zero surface velocity has no defined efficiency")
        return power / (0.5 * self.fluid_density_kg_m3 * self.sound_speed_m_s * self.total_area_m2 * ms)

    def radiation_resistance(
        self,
        velocity_pattern: np.ndarray,
        omega: float,
        modal_velocity_amplitude: float = 1.0,
        n_theta: int = 48,
        n_phi: int = 32,
    ) -> float:
        """Modal radiation resistance ``R`` [N s/m] with ``P = R |qdot|^2``.

        Because the Rayleigh model is linear in the velocity field, the power
        scales with ``|qdot|^2`` for a *fixed* modal velocity pattern, so one
        power evaluation defines the equivalent viscous radiation damping of
        that mode at the frequency ``omega``.
        """
        if modal_velocity_amplitude == 0.0:
            raise ValueError("modal velocity amplitude must be non-zero")
        pattern = np.asarray(velocity_pattern, dtype=float)
        power = float(self.radiated_power(pattern * modal_velocity_amplitude, omega, n_theta, n_phi))
        return power / modal_velocity_amplitude ** 2

    def to_json_dict(self) -> Dict[str, float]:
        return {
            "n_patches": int(self.area_m2.size),
            "total_area_m2": self.total_area_m2,
            "fluid_density_kg_m3": self.fluid_density_kg_m3,
            "sound_speed_m_s": self.sound_speed_m_s,
            "x_min_m": float(np.min(self.x_m)) if self.x_m.size else 0.0,
            "x_max_m": float(np.max(self.x_m)) if self.x_m.size else 0.0,
        }


# ---------------------------------------------------------------------------
# Module-level convenience wrappers (also used by the verification tests)
# ---------------------------------------------------------------------------


def far_field_pressure(
    surface: RadiatorSurface,
    velocity_amplitude: np.ndarray,
    omega: float,
    range_m: float,
    theta_rad: float,
    phi_rad: float = 0.0,
) -> np.ndarray:
    """See :meth:`RadiatorSurface.far_field_pressure`."""
    return surface.far_field_pressure(velocity_amplitude, omega, range_m, theta_rad, phi_rad)


def radiated_power(
    surface: RadiatorSurface,
    velocity_amplitude: np.ndarray,
    omega: float,
    n_theta: int = 48,
    n_phi: int = 32,
) -> float:
    """See :meth:`RadiatorSurface.radiated_power`."""
    return float(surface.radiated_power(velocity_amplitude, omega, n_theta, n_phi))


def radiation_efficiency(
    surface: RadiatorSurface,
    velocity_amplitude: np.ndarray,
    omega: float,
    n_theta: int = 48,
    n_phi: int = 32,
) -> float:
    """See :meth:`RadiatorSurface.radiation_efficiency`."""
    return surface.radiation_efficiency(velocity_amplitude, omega, n_theta, n_phi)


# ---------------------------------------------------------------------------
# Closed-form references for the uniform circular piston (verification only)
# ---------------------------------------------------------------------------


def piston_radiation_efficiency(ka: float) -> float:
    """``sigma = 1 - J1(2ka)/(ka)`` for a uniform circular piston in a baffle."""
    ka = float(ka)
    if ka <= 0.0:
        raise ValueError("ka must be positive")
    from scipy.special import j1

    return float(1.0 - j1(2.0 * ka) / ka)


def piston_directivity(ka: float, theta_rad: Sequence[float]) -> np.ndarray:
    """``D(th) = 2 J1(ka sin th)/(ka sin th)`` (normalised to 1 on axis)."""
    from scipy.special import j1

    th = np.asarray(theta_rad, dtype=float)
    x = float(ka) * np.sin(th)
    out = np.ones_like(x)
    nz = x > 1e-12
    out[nz] = 2.0 * j1(x[nz]) / x[nz]
    return out