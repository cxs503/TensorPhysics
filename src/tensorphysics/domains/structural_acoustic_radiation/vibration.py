"""Structural vibration stage of the underwater noise radiation chain.

The prediction chain implemented here is

    harmonic force F0 e^{iωt}
        -> surface velocity amplitude v0 (single-degree-of-freedom oscillator)
        -> mean-square velocity <v²> = v0²/2
        -> radiated sound power W_rad = σ_rad ρ c S <v²>
        -> far-field pressure p(r) = √(ρc W_rad / (Ω r²))
        -> sound pressure level SPL = 20 log10(p/p_ref)

The vibration stage is a classical SDOF oscillator with mass m, stiffness k
and viscous damping ratio ζ.  With the time convention e^{iωt} the mechanical
impedance is

    Z(ω) = c_d + i(ω m - k/ω),      c_d = 2 ζ ω_n m

and the velocity amplitude for a harmonic force of amplitude F0 is

    v0(ω) = F0 / |Z(ω)| = F0 / √(c_d² + (ω m - k/ω)²).

At resonance (ω = ω_n) this reduces to the closed-form value v0 = F0 / c_d,
which is used as the analytic anchor for the vibration stage.

References
----------
Fahy & Gardonio, "Sound and Structural Vibration", 2nd ed. (2007).
Junger & Feit, "Sound, Structures, and Their Interaction" (1986).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from .radiation import (
    C_WATER,
    P_REF,
    RHO_WATER,
    semi_empirical_radiation_efficiency,
)


@dataclass
class HarmonicVibrator:
    """Single-degree-of-freedom structure under harmonic forcing.

    Parameters
    ----------
    mass : float
        Modal mass m [kg].
    stiffness : float
        Modal stiffness k [N/m].
    damping_ratio : float
        Viscous damping ratio ζ (dimensionless, 0 < ζ < 1).
    force_amplitude : float
        Harmonic force amplitude F_0 [N].
    """

    mass: float
    stiffness: float
    damping_ratio: float
    force_amplitude: float

    def __post_init__(self) -> None:
        if self.mass <= 0 or self.stiffness <= 0:
            raise ValueError("mass and stiffness must be positive")
        if not (0.0 < self.damping_ratio < 1.0):
            raise ValueError("damping_ratio must satisfy 0 < zeta < 1")
        if self.force_amplitude < 0:
            raise ValueError("force_amplitude must be non-negative")

    @property
    def omega_n(self) -> float:
        """Undamped natural angular frequency ω_n = √(k/m) [rad/s]."""
        return float(np.sqrt(self.stiffness / self.mass))

    @property
    def natural_frequency(self) -> float:
        """Undamped natural frequency f_n [Hz]."""
        return self.omega_n / (2.0 * np.pi)

    @property
    def damping_coefficient(self) -> float:
        """Viscous damping coefficient c_d = 2 ζ ω_n m [N·s/m]."""
        return 2.0 * self.damping_ratio * self.omega_n * self.mass

    @property
    def static_displacement(self) -> float:
        """Static deflection under F_0 [m]."""
        return self.force_amplitude / self.stiffness

    def mechanical_impedance(self, omega: float) -> complex:
        """Mechanical impedance Z(ω) = c_d + i(ω m - k/ω) [N·s/m]."""
        omega = float(omega)
        if omega <= 0:
            raise ValueError("omega must be positive")
        return complex(self.damping_coefficient,
                       omega * self.mass - self.stiffness / omega)

    def velocity_amplitude(self, omega: float) -> float:
        """Steady-state surface velocity amplitude v_0 [m/s]."""
        return float(self.force_amplitude / abs(self.mechanical_impedance(omega)))

    def displacement_amplitude(self, omega: float) -> float:
        """Steady-state displacement amplitude |x| = v_0/ω [m]."""
        return self.velocity_amplitude(omega) / float(omega)

    def acceleration_amplitude(self, omega: float) -> float:
        """Steady-state acceleration amplitude ω v_0 [m/s²]."""
        return self.velocity_amplitude(omega) * float(omega)

    def mean_square_velocity(self, omega: float) -> float:
        """Time-averaged mean-square velocity <v²> = v_0²/2 [m²/s²]."""
        return self.velocity_amplitude(omega) ** 2 / 2.0

    def resonance_velocity_amplitude(self) -> float:
        """Closed-form resonant velocity amplitude v_0(ω_n) = F_0/c_d [m/s]."""
        return self.force_amplitude / self.damping_coefficient


def vibration_radiation_chain(
    vibrator: HarmonicVibrator,
    frequency: float,
    *,
    surface_area: float,
    characteristic_length: float,
    model: str = "monopole",
    half_space: bool | None = None,
    distances: list[float] | np.ndarray | None = None,
    rho: float = RHO_WATER,
    c: float = C_WATER,
    p_ref: float = P_REF,
) -> dict:
    """Run the full force -> velocity -> sound power -> far-field SPL chain.

    Parameters
    ----------
    vibrator : HarmonicVibrator
        SDOF structure model providing the surface velocity.
    frequency : float
        Excitation frequency f [Hz].
    surface_area : float
        Radiating surface area S [m²].
    characteristic_length : float
        Characteristic length a [m] used in ka = ω a / c.
    model : {"monopole", "dipole", "piston"}
        Radiation-efficiency model (see
        :func:`semi_empirical_radiation_efficiency`).
    half_space : bool, optional
        If True the power is spread over 2π sr (baffled source), else 4π sr
        (free field).  Default: True for ``model="piston"``, else False.
    distances : array-like, optional
        Observation distances r [m].  Default: (1, 10, 100) m.
    rho, c, p_ref : float
        Fluid properties and SPL reference pressure.

    Returns
    -------
    dict
        Chain results including v_0, <v²>, ka, σ_rad, W_rad, distances,
        p_rms, SPL_dB and the solid angle exponent used in the spreading law.
    """
    if frequency <= 0:
        raise ValueError("frequency must be positive")
    if surface_area <= 0 or characteristic_length <= 0:
        raise ValueError("surface_area and characteristic_length must be positive")
    if distances is None:
        distances = np.array([1.0, 10.0, 100.0], dtype=float)
    distances = np.asarray(distances, dtype=float)
    if np.any(distances <= 0):
        raise ValueError("distances must be positive")

    if half_space is None:
        half_space = model.lower() in ("piston", "baffled_piston")

    omega = 2.0 * np.pi * frequency
    k = omega / c
    ka = k * characteristic_length

    v0 = vibrator.velocity_amplitude(omega)
    ms_velocity = v0**2 / 2.0
    sigma = float(semi_empirical_radiation_efficiency(ka, model))
    W_rad = sigma * rho * c * surface_area * ms_velocity

    # W = Ω r² p_rms² / (ρc) with Ω = 2π (half space) or 4π (free field)
    solid_angle = 2.0 * np.pi if half_space else 4.0 * np.pi
    p_rms = np.sqrt(rho * c * W_rad / (solid_angle * distances**2))
    p_amp = np.sqrt(2.0) * p_rms
    spl = 20.0 * np.log10(np.maximum(p_rms, 1e-300) / p_ref)

    return {
        "frequency": float(frequency),
        "omega": omega,
        "wavenumber": k,
        "ka": float(ka),
        "model": model,
        "solid_angle_sr": float(solid_angle),
        "v0": float(v0),
        "v_rms": float(np.sqrt(ms_velocity)),
        "mean_square_velocity": float(ms_velocity),
        "sigma_rad": sigma,
        "rho": float(rho),
        "c": float(c),
        "surface_area": float(surface_area),
        "W_rad": float(W_rad),
        "distances": distances,
        "p_rms": p_rms,
        "p_amplitude": p_amp,
        "SPL_dB": spl,
    }


def chain_provenance(result: dict) -> dict:
    """Attach a sha256 anchor over the numerically relevant chain outputs.

    The hash covers the parameter block *and* every computed scalar/array so
    that a stored JSON artifact can be re-verified bit-for-bit.
    """
    def _encode(value):
        if isinstance(value, np.ndarray):
            return [float(x) for x in value.ravel()]
        if isinstance(value, (np.floating, float)):
            return float(value)
        return value

    payload = {
        key: _encode(value)
        for key, value in result.items()
        if key not in ("distances", "p_rms", "p_amplitude", "SPL_dB")
    }
    payload["distances"] = [float(x) for x in np.asarray(result["distances"]).ravel()]
    payload["p_rms"] = [float(x) for x in np.asarray(result["p_rms"]).ravel()]
    payload["SPL_dB"] = [float(x) for x in np.asarray(result["SPL_dB"]).ravel()]
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {
        "sha256": hashlib.sha256(blob.encode()).hexdigest(),
        "payload_keys": sorted(payload),
        "algorithm": "sha256 over canonical JSON (sorted keys, compact separators)",
    }