"""Structural vibration → acoustic radiation chain.

Implements the classical vibration-to-sound radiation prediction chain with
analytical anchors for verification:

1. Pulsating sphere (monopole source):
   - Exact radiation efficiency: σ = (ka)² / (1 + (ka)²)
   - Exact radiated power: W = 4πa² ρc <v²> σ
   - Exact far-field pressure: |p(r)| = ρc k v_0 a² / (r √(1+(ka)²))

2. Circular piston in infinite baffle:
   - Radiation efficiency: σ = 1 - 2J₁(2ka)/(2ka)
   - Low-frequency limit: σ ≈ (ka)²/2 for ka << 1
   - High-frequency limit: σ → 1 for ka >> 1

3. Dipole-type (unbaffled plate) semi-empirical model:
   - Radiation efficiency: sigma = (ka)^4 / (1 + (ka)^4)
   - Low-frequency limit: sigma ~ (ka)^4 (dipole, acoustic short-circuit)

4. General radiation chain:
   - W_rad = σ_rad · ρ · c · S · <v²>
   - p_rms(r) = √(ρc W_rad / (4πr²))
   - SPL = 20 log₁₀(p_rms / p_ref)

All formulas assume linear acoustics, time-harmonic excitation, and far-field
observation (kr >> 1).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from scipy.special import j1  # Bessel function of first kind, order 1


# ---------------------------------------------------------------------------
# Physical constants (underwater acoustics)
# ---------------------------------------------------------------------------
RHO_WATER: float = 1025.0   # density of seawater [kg/m³]
C_WATER: float = 1500.0     # speed of sound in seawater [m/s]
P_REF: float = 1.0e-6       # reference pressure for underwater SPL [Pa] (1 μPa)


# ---------------------------------------------------------------------------
# Pulsating sphere (exact analytical solution)
# ---------------------------------------------------------------------------
def pulsating_sphere_radiation_efficiency(ka: float | np.ndarray) -> float | np.ndarray:
    """Radiation efficiency of a pulsating sphere.

    Exact closed-form result from the radiation impedance of a sphere:

        σ(ka) = (ka)² / (1 + (ka)²)

    Parameters
    ----------
    ka : float or array
        Non-dimensional wavenumber (wavenumber × radius).

    Returns
    -------
    float or array
        Radiation efficiency (dimensionless, 0 ≤ σ ≤ 1).

    Notes
    -----
    - Low-frequency limit (ka << 1): σ ≈ (ka)²
    - High-frequency limit (ka >> 1): σ → 1
    - This is the exact result for a uniformly pulsating sphere; no
      approximations are involved.

    References
    ----------
    Morse & Ingard, "Theoretical Acoustics", §5.3 (1968).
    """
    ka = np.asarray(ka, dtype=float)
    return ka**2 / (1.0 + ka**2)


@dataclass
class PulsatingSphere:
    """Pulsating sphere acoustic source (monopole).

    A sphere of radius `a` with uniform normal surface velocity v_s = v_0 e^{iωt}.

    Parameters
    ----------
    radius : float
        Sphere radius [m].
    frequency : float
        Excitation frequency [Hz].
    velocity_amplitude : float
        Surface velocity amplitude v_0 [m/s].
    rho : float, optional
        Fluid density [kg/m³]. Default: seawater (1025 kg/m³).
    c : float, optional
        Sound speed [m/s]. Default: seawater (1500 m/s).
    """

    radius: float
    frequency: float
    velocity_amplitude: float
    rho: float = RHO_WATER
    c: float = C_WATER

    @property
    def omega(self) -> float:
        """Angular frequency [rad/s]."""
        return 2.0 * np.pi * self.frequency

    @property
    def k(self) -> float:
        """Acoustic wavenumber [1/m]."""
        return self.omega / self.c

    @property
    def ka(self) -> float:
        """Non-dimensional wavenumber."""
        return self.k * self.radius

    @property
    def surface_area(self) -> float:
        """Sphere surface area [m²]."""
        return 4.0 * np.pi * self.radius**2

    @property
    def mean_square_velocity(self) -> float:
        """Time-averaged mean-square surface velocity [m²/s²]."""
        return self.velocity_amplitude**2 / 2.0

    def radiation_efficiency(self) -> float:
        """Exact radiation efficiency σ(ka) = (ka)² / (1 + (ka)²)."""
        return float(pulsating_sphere_radiation_efficiency(self.ka))

    def radiated_power(self) -> float:
        """Exact radiated sound power [W].

        W = 4πa² ρc <v²> σ = 4πa² ρc (v_0²/2) (ka)²/(1+(ka)²)
        """
        S = self.surface_area
        sigma = self.radiation_efficiency()
        return sigma * self.rho * self.c * S * self.mean_square_velocity

    def far_field_pressure_amplitude(self, r: float) -> float:
        """Exact far-field pressure amplitude |p(r)| [Pa].

        |p(r)| = ρc k v_0 a² / (r √(1+(ka)²))

        Parameters
        ----------
        r : float
            Observation distance from sphere center [m]. Must satisfy r >> a.
        """
        if r <= self.radius:
            raise ValueError(f"Observer distance r={r} must be > sphere radius a={self.radius}")
        ka = self.ka
        return (self.rho * self.c * self.k * self.velocity_amplitude * self.radius**2
                / (r * np.sqrt(1.0 + ka**2)))

    def far_field_pressure_rms(self, r: float) -> float:
        """RMS far-field pressure [Pa]."""
        return self.far_field_pressure_amplitude(r) / np.sqrt(2.0)

    def verify_power_consistency(self, r: float, tol: float = 1e-10) -> bool:
        """Verify W = 4πr² I = 4πr² p_rms²/(ρc) (energy conservation)."""
        W = self.radiated_power()
        p_rms = self.far_field_pressure_rms(r)
        intensity = p_rms**2 / (self.rho * self.c)
        W_from_intensity = 4.0 * np.pi * r**2 * intensity
        return abs(W - W_from_intensity) / W < tol


# ---------------------------------------------------------------------------
# Circular piston in infinite baffle (exact analytical solution)
# ---------------------------------------------------------------------------
def piston_radiation_efficiency(ka: float | np.ndarray) -> float | np.ndarray:
    """Radiation efficiency of a circular piston in an infinite rigid baffle.

    Exact closed-form result:

        σ(ka) = 1 - 2 J₁(2ka) / (2ka)

    where J₁ is the Bessel function of the first kind of order 1.

    Parameters
    ----------
    ka : float or array
        Non-dimensional wavenumber (wavenumber × piston radius).

    Returns
    -------
    float or array
        Radiation efficiency (dimensionless, 0 ≤ σ ≤ 1).

    Notes
    -----
    - Low-frequency limit (ka << 1): σ ≈ (ka)²/2
    - High-frequency limit (ka >> 1): σ → 1
    - This is the exact result for a baffled circular piston with uniform
      normal velocity.

    References
    ----------
    Morse & Ingard, "Theoretical Acoustics", §7.4 (1968).
    Junger & Feit, "Sound, Structures, and Their Interaction", §3.5 (1986).
    """
    ka = np.asarray(ka, dtype=float)
    # Handle ka → 0 limit to avoid division by zero
    result = np.where(
        ka < 1e-12,
        0.0,  # σ → 0 as ka → 0
        1.0 - 2.0 * j1(2.0 * ka) / (2.0 * ka)
    )
    # Ensure scalar output for scalar input
    if result.ndim == 0:
        return float(result)
    return result


def dipole_radiation_efficiency(ka: float | np.ndarray) -> float | np.ndarray:
    """Semi-empirical radiation efficiency of a dipole-type radiator.

    Small unbaffled plates (and any source with zero net volume displacement)
    radiate as an acoustic dipole, so the low-frequency radiation efficiency
    scales as (ka)^4 instead of the monopole (ka)^2.  The closed-form blend

        sigma(ka) = (ka)^4 / (1 + (ka)^4)

    reproduces both asymptotic limits (sigma -> (ka)^4 for ka << 1,
    sigma -> 1 for ka >> 1) with the correct monotonic transition.

    Parameters
    ----------
    ka : float or array
        Non-dimensional wavenumber (k * characteristic half-length).

    Returns
    -------
    float or array
        Radiation efficiency (dimensionless, 0 <= sigma <= 1).

    References
    ----------
    Fahy & Gardonio, "Sound and Structural Vibration", 2nd ed., Ch. 6 (2007).
    """
    ka = np.asarray(ka, dtype=float)
    x4 = ka**4
    return x4 / (1.0 + x4)


def semi_empirical_radiation_efficiency(
    ka: float | np.ndarray,
    model: str = "monopole",
) -> float | np.ndarray:
    """Dispatching helper for the sigma_rad(ka) model ladder.

    Parameters
    ----------
    ka : float or array
        Non-dimensional wavenumber.
    model : {"monopole", "dipole", "piston"}
        - ``monopole``: pulsating-sphere exact result (ka)^2/(1+(ka)^2)
        - ``dipole``:   unbaffled-plate semi-empirical (ka)^4/(1+(ka)^4)
        - ``piston``:   baffled circular piston 1 - 2J1(2ka)/(2ka)

    Returns
    -------
    float or array
        Radiation efficiency.

    Notes
    -----
    All three models share the physically required asymptotic behaviour
    sigma(ka -> 0) -> 0 and sigma(ka -> inf) -> 1, which is what the
    acceptance negative control (d) tests.
    """
    key = model.lower()
    if key in ("monopole", "sphere", "pulsating_sphere"):
        return pulsating_sphere_radiation_efficiency(ka)
    if key in ("dipole", "plate", "unbaffled"):
        return dipole_radiation_efficiency(ka)
    if key in ("piston", "baffled_piston"):
        return piston_radiation_efficiency(ka)
    raise ValueError(
        f"Unknown radiation-efficiency model {model!r}; "
        "expected 'monopole', 'dipole' or 'piston'"
    )


@dataclass
class CircularPiston:
    """Circular piston in an infinite rigid baffle.

    A circular piston of radius `a` with uniform normal velocity v_0 e^{iωt}
    mounted in an infinite rigid baffle.

    Parameters
    ----------
    radius : float
        Piston radius [m].
    frequency : float
        Excitation frequency [Hz].
    velocity_amplitude : float
        Surface velocity amplitude v_0 [m/s].
    rho : float, optional
        Fluid density [kg/m³]. Default: seawater (1025 kg/m³).
    c : float, optional
        Sound speed [m/s]. Default: seawater (1500 m/s).
    """

    radius: float
    frequency: float
    velocity_amplitude: float
    rho: float = RHO_WATER
    c: float = C_WATER

    @property
    def omega(self) -> float:
        """Angular frequency [rad/s]."""
        return 2.0 * np.pi * self.frequency

    @property
    def k(self) -> float:
        """Acoustic wavenumber [1/m]."""
        return self.omega / self.c

    @property
    def ka(self) -> float:
        """Non-dimensional wavenumber."""
        return self.k * self.radius

    @property
    def surface_area(self) -> float:
        """Piston surface area [m²]."""
        return np.pi * self.radius**2

    @property
    def mean_square_velocity(self) -> float:
        """Time-averaged mean-square surface velocity [m²/s²]."""
        return self.velocity_amplitude**2 / 2.0

    def radiation_efficiency(self) -> float:
        """Exact radiation efficiency σ(ka) = 1 - 2J₁(2ka)/(2ka)."""
        return float(piston_radiation_efficiency(self.ka))

    def radiated_power(self) -> float:
        """Radiated sound power [W].

        W = σ ρc S <v²>

        Note: For a baffled piston, only one side radiates, so S = πa².
        """
        S = self.surface_area
        sigma = self.radiation_efficiency()
        return sigma * self.rho * self.c * S * self.mean_square_velocity

    def far_field_pressure_rms(self, r: float) -> float:
        """Spatially-averaged RMS far-field pressure [Pa].

        Energy balance over the half-space (2π sr) radiated by a baffled
        source gives

            W = 2π r² · p_rms² / (ρc)   =>   p_rms = √(ρc W / (2π r²)).

        This is the directivity-averaged value, not the on-axis value.
        """
        if r <= self.radius:
            raise ValueError(f"Observer distance r={r} must be > piston radius a={self.radius}")
        W = self.radiated_power()
        return float(np.sqrt(self.rho * self.c * W / (2.0 * np.pi * r**2)))

    def far_field_pressure_amplitude(self, r: float) -> float:
        """Peak far-field pressure of the equivalent omni-directional field [Pa].

        Equal to √2 · :meth:`far_field_pressure_rms`; it coincides with the
        exact on-axis value in the low-frequency limit (ka << 1).
        """
        return float(np.sqrt(2.0) * self.far_field_pressure_rms(r))

    def directivity(self, theta: float) -> float:
        """Far-field directivity factor D(theta) = 2J1(ka sinθ)/(ka sinθ)."""
        x = self.ka * np.sin(theta)
        if abs(x) < 1e-12:
            return 1.0
        return float(2.0 * j1(x) / x)

    def on_axis_pressure_amplitude(self, r: float) -> float:
        """Exact on-axis far-field pressure amplitude [Pa] (directivity included).

            |p(r)| = ρc k a² v_0 / (2r) · D(0) = ρc k a² v_0 / (2r)

        See :meth:`on_axis_pressure_exact` for the un-approximated form.
        """
        if r <= self.radius:
            raise ValueError(f"Observer distance r={r} must be > piston radius a={self.radius}")
        return float(self.rho * self.c * self.k * self.radius**2
                     * self.velocity_amplitude / (2.0 * r))

    def on_axis_pressure_exact(self, r: float) -> float:
        """Exact on-axis pressure amplitude [Pa] from the Rayleigh integral.

        For a uniformly vibrating circular piston in an infinite baffle the
        Rayleigh integral on the axis reduces to the closed form

            p(r) = ρ c v_0 |exp(-ikr) - exp(-ik√(r² + a²))|

        valid at every distance r > 0 (near field included). In the far field
        r >> a²/λ it collapses to ρ c k a² v_0 / (2r).
        """
        if r <= 0:
            raise ValueError("r must be positive")
        ph1 = np.exp(-1j * self.k * r)
        ph2 = np.exp(-1j * self.k * np.hypot(r, self.radius))
        return float(self.rho * self.c * self.velocity_amplitude * np.abs(ph1 - ph2))


# ---------------------------------------------------------------------------
# General radiation chain
# ---------------------------------------------------------------------------
def radiated_power(
    sigma_rad: float,
    rho: float,
    c: float,
    S: float,
    v_rms: float,
) -> float:
    """General radiated sound power formula.

    W_rad = σ_rad · ρ · c · S · <v²> = σ_rad · ρ · c · S · v_rms²

    Parameters
    ----------
    sigma_rad : float
        Radiation efficiency (dimensionless).
    rho : float
        Fluid density [kg/m³].
    c : float
        Sound speed [m/s].
    S : float
        Radiating surface area [m²].
    v_rms : float
        RMS surface velocity [m/s].

    Returns
    -------
    float
        Radiated sound power [W].
    """
    return sigma_rad * rho * c * S * v_rms**2


def far_field_pressure(
    W_rad: float,
    rho: float,
    c: float,
    r: float,
) -> float:
    """Far-field RMS pressure from radiated power (full-space radiation).

    p_rms(r) = √(ρc W_rad / (4πr²))

    Parameters
    ----------
    W_rad : float
        Radiated sound power [W].
    rho : float
        Fluid density [kg/m³].
    c : float
        Sound speed [m/s].
    r : float
        Observation distance [m].

    Returns
    -------
    float
        RMS far-field pressure [Pa].

    Notes
    -----
    Assumes full-space (4π steradian) radiation. For half-space (baffled),
    replace 4π with 2π.
    """
    return np.sqrt(rho * c * W_rad / (4.0 * np.pi * r**2))


def sound_pressure_level(p_rms: float, p_ref: float = P_REF) -> float:
    """Sound pressure level in dB.

    SPL = 20 log₁₀(p_rms / p_ref)

    Parameters
    ----------
    p_rms : float
        RMS pressure [Pa].
    p_ref : float, optional
        Reference pressure [Pa]. Default: 1 μPa (underwater).

    Returns
    -------
    float
        Sound pressure level [dB re p_ref].
    """
    if p_rms <= 0:
        raise ValueError("p_rms must be positive")
    return 20.0 * np.log10(p_rms / p_ref)


# ---------------------------------------------------------------------------
# Radiation chain orchestrator
# ---------------------------------------------------------------------------
@dataclass
class RadiationChain:
    """Complete vibration → sound power → far-field pressure chain.

    Orchestrates the full prediction chain for a generic vibrating structure
    with known radiation efficiency.

    Parameters
    ----------
    frequency : float
        Excitation frequency [Hz].
    velocity_amplitude : float
        Surface velocity amplitude v_0 [m/s].
    surface_area : float
        Radiating surface area S [m²].
    sigma_rad : float
        Radiation efficiency (dimensionless).
    rho : float, optional
        Fluid density [kg/m³]. Default: seawater.
    c : float, optional
        Sound speed [m/s]. Default: seawater.
    p_ref : float, optional
        Reference pressure for SPL [Pa]. Default: 1 μPa.
    """

    frequency: float
    velocity_amplitude: float
    surface_area: float
    sigma_rad: float
    rho: float = RHO_WATER
    c: float = C_WATER
    p_ref: float = P_REF

    @property
    def omega(self) -> float:
        return 2.0 * np.pi * self.frequency

    @property
    def k(self) -> float:
        return self.omega / self.c

    @property
    def v_rms(self) -> float:
        """RMS surface velocity [m/s]."""
        return self.velocity_amplitude / np.sqrt(2.0)

    @property
    def mean_square_velocity(self) -> float:
        """Mean-square velocity <v²> [m²/s²]."""
        return self.velocity_amplitude**2 / 2.0

    def compute_radiated_power(self) -> float:
        """Radiated sound power [W]."""
        return radiated_power(
            self.sigma_rad, self.rho, self.c, self.surface_area, self.v_rms
        )

    def compute_far_field_pressure(self, r: float) -> dict:
        """Compute far-field pressure at distance r.

        Returns dict with p_rms, p_amplitude, SPL.
        """
        W = self.compute_radiated_power()
        p_rms = far_field_pressure(W, self.rho, self.c, r)
        p_amp = p_rms * np.sqrt(2.0)
        spl = sound_pressure_level(p_rms, self.p_ref)
        return {
            "W_rad": W,
            "p_rms": p_rms,
            "p_amplitude": p_amp,
            "SPL_dB": spl,
            "distance": r,
        }

    def compute_chain(self, distances: np.ndarray | list) -> dict:
        """Compute full chain for multiple observation distances.

        Returns dict with arrays of results.
        """
        distances = np.asarray(distances, dtype=float)
        W = self.compute_radiated_power()
        p_rms = far_field_pressure(W, self.rho, self.c, distances)
        p_amp = p_rms * np.sqrt(2.0)
        spl = 20.0 * np.log10(p_rms / self.p_ref)
        return {
            "W_rad": W,
            "sigma_rad": self.sigma_rad,
            "v_rms": self.v_rms,
            "distances": distances,
            "p_rms": p_rms,
            "p_amplitude": p_amp,
            "SPL_dB": spl,
        }

    def to_provenance_dict(self, distances: np.ndarray | list | None = None) -> dict:
        """Generate provenance-anchored result dictionary.

        Includes sha256 hash of key parameters for reproducibility.
        """
        if distances is None:
            distances = np.array([1.0, 10.0, 100.0])
        result = self.compute_chain(distances)

        # Create provenance hash
        params = {
            "frequency": self.frequency,
            "velocity_amplitude": self.velocity_amplitude,
            "surface_area": self.surface_area,
            "sigma_rad": self.sigma_rad,
            "rho": self.rho,
            "c": self.c,
        }
        param_str = json.dumps(params, sort_keys=True)
        sha256 = hashlib.sha256(param_str.encode()).hexdigest()

        return {
            "provenance": {
                "sha256": sha256,
                "parameters": params,
                "module": "tensorphysics.domains.structural_acoustic_radiation",
            },
            "results": {
                "W_rad": float(result["W_rad"]),
                "sigma_rad": float(result["sigma_rad"]),
                "v_rms": float(result["v_rms"]),
                "distances": result["distances"].tolist(),
                "p_rms": result["p_rms"].tolist(),
                "p_amplitude": result["p_amplitude"].tolist(),
                "SPL_dB": result["SPL_dB"].tolist(),
            },
        }
