"""Ice-milling torque excitation and its analytic spectrum.

Model (documented engineering assumption, not a rule)::

    T_ice(t) = T_mean + sum_{n=1..N} T_n sin(n omega_b t + phi_n)

``omega_b = 2 pi f_b`` is the blade-passing frequency ``f_b = n_blades * RPM / 60``.
The harmonic amplitudes decay as ``T_n = T_1 / n**decay`` up to a cut-off, which
reproduces the observed ice-milling spectrum shape (dominant blade-passing line
plus a quickly decaying tail).  The mean torque and the amplitude scale linearly
with ice thickness and advance speed through ``mean_coefficient`` and
``amplitude_coefficient``; those coefficients are *model parameters* that must be
calibrated against full-scale measurements for a specific propeller, and they are
flagged as such in ``MODEL_ASSUMPTIONS``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = ["IceMillingTorque", "MODEL_ASSUMPTIONS"]

MODEL_ASSUMPTIONS = {
    "spectrum": "T_n = T_1 / n**decay with an exponential cut-off above cutoff_harmonic",
    "scaling": "T_mean and T_1 scale linearly with ice thickness and advance speed",
    "phase": "phases are independent uniform draws seeded for reproducibility, not measured",
    "calibration": "mean_coefficient / amplitude_coefficient are NOT measured propeller data",
}


@dataclass
class IceMillingTorque:
    """Ice-milling torque with a deterministic blade-passing harmonic spectrum."""

    mean_torque: float
    blade_passing_hz: float
    first_harmonic: float
    n_harmonics: int = 3
    decay: float = 1.0
    phases: Optional[Sequence[float]] = None
    seed: int = 0

    def __post_init__(self) -> None:
        if not (self.mean_torque >= 0.0):
            raise ValueError("mean_torque must be non-negative")
        if not (self.blade_passing_hz > 0.0):
            raise ValueError("blade_passing_hz must be positive")
        if self.n_harmonics < 1:
            raise ValueError("need at least one harmonic")
        if self.first_harmonic < 0.0:
            raise ValueError("first_harmonic must be non-negative")
        if self.phases is None:
            rng = np.random.RandomState(self.seed)
            self.phases = tuple(float(v) for v in rng.uniform(0.0, 2.0 * math.pi, self.n_harmonics))
        if len(self.phases) < self.n_harmonics:
            raise ValueError("phases shorter than the harmonic count")

    @classmethod
    def from_ice_condition(cls, ice_thickness: float, advance_speed: float, rpm: float,
                           n_blades: int, mean_coefficient: float = 6.0e4,
                           amplitude_coefficient: float = 1.2e4,
                           n_harmonics: int = 3, decay: float = 1.0,
                           seed: int = 0) -> "IceMillingTorque":
        """Engineering scaling: ``T_mean = c_m h v`` and ``T_1 = c_a h v``."""
        if ice_thickness <= 0.0 or advance_speed <= 0.0:
            raise ValueError("ice thickness and advance speed must be positive")
        if rpm <= 0.0 or n_blades < 2:
            raise ValueError("rpm must be positive and n_blades >= 2")
        mean = mean_coefficient * ice_thickness * advance_speed
        first = amplitude_coefficient * ice_thickness * advance_speed
        return cls(mean, n_blades * rpm / 60.0, first, n_harmonics, decay, None, seed)

    @property
    def amplitudes(self) -> Tuple[float, ...]:
        return tuple(self.first_harmonic / (n + 1) ** self.decay for n in range(self.n_harmonics))

    @property
    def frequencies(self) -> Tuple[float, ...]:
        return tuple((n + 1) * self.blade_passing_hz for n in range(self.n_harmonics))

    def torque(self, t):
        """Torque time history [N*m] (scalar or array input)."""
        tt = np.asarray(t, dtype=float)
        out = np.full(tt.shape, self.mean_torque, dtype=float)
        for n, (amp, freq, phase) in enumerate(zip(self.amplitudes, self.frequencies, self.phases)):
            out = out + amp * np.sin(2.0 * math.pi * freq * tt + phase)
        return float(out) if np.ndim(out) == 0 else out

    def spectrum_amplitude(self, frequency: float) -> complex:
        """One-sided analytic Fourier amplitude at ``frequency`` [N*m]."""
        acc = complex(self.mean_torque if frequency == 0.0 else 0.0, 0.0)
        for amp, freq, phase in zip(self.amplitudes, self.frequencies, self.phases):
            if abs(frequency - freq) < 1e-9:
                # sin(wt + phi) -> one-sided amplitude amp * exp(i*(phi - pi/2))
                acc += amp * np.exp(1j * (phase - 0.5 * math.pi))
        return acc

    def rms_amplitude(self) -> float:
        """RMS of the oscillating part [N*m]."""
        return math.sqrt(0.5 * sum(a * a for a in self.amplitudes))