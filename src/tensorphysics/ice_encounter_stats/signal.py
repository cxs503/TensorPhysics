"""Synthesis of the ice-resistance record as a Poisson shot-noise process.

For encounter times ``t_i`` and marks (peak force ``F_i``, contact duration
``D_i``) the ice-resistance record is

    R(t) = sum_i F_i * g((t - t_i) / D_i),                                   (2)

with ``g`` the normalised pulse of unit peak on ``[0, 1]``.  Writing
``a1 = int g dtau`` and ``b1 = int g^2 dtau``, Campbell's theorem gives the
analytic anchors used throughout this study::

    mu    = lambda * E[F * D] * a1                                           (3)
    sigma2 = lambda * E[F^2 * D] * b1                                        (4)

Equation (4) is exact for a Poisson-driven shot noise with independent marks --
it does **not** require the pulses to be non-overlapping, because the cross
terms factor into ``mu^2``.  That property is what makes it a usable analytic
anchor for dense, overlapping ice-resistance records.

The record is returned together with the *injected* marks, so the Campbell
prediction is evaluated from the ground truth rather than from a
threshold-recovered (and possibly biased) event list.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CaseSpec, ConfigError
from .field import encounter_times


@dataclass(frozen=True)
class Record:
    """A sampled resistance record plus the marks that generated it."""

    t: np.ndarray
    force: np.ndarray
    dt: float
    marks: dict

    @property
    def duration_s(self) -> float:
        return float(self.t[-1] - self.t[0]) + self.dt

    @property
    def mean(self) -> float:
        return float(np.mean(self.force))

    @property
    def var(self) -> float:
        return float(np.var(self.force, ddof=0))

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var))

    @property
    def cv(self) -> float:
        m = self.mean
        return float(self.std / m) if m > 0.0 else float("nan")

    def generator(self):
        return _seeded(self.marks.get("seed"))


def _seeded(seed):
    return np.random.default_rng(seed)


def simulate_record(case: CaseSpec, duration_s: float, seed: int,
                    samples_per_pulse: int = 40) -> Record:
    """Draw one independent realisation of the resistance record."""
    if duration_s <= 0.0:
        raise ConfigError("duration must be positive")
    if samples_per_pulse < 4:
        raise ConfigError("samples_per_pulse must be >= 4")
    rng = np.random.default_rng(int(seed))

    times = encounter_times(rng, case, duration_s)
    n_events = times.size

    floe = case.floe_size.sample(rng, n_events)
    thick = case.thickness.sample(rng, n_events)
    amp = case.force.amplitude(rng, floe, thick)
    dur = case.pulse.duration_s(floe, case.speed_mps)

    a1, b1 = case.pulse.integrals()
    area = amp * dur * a1          # int h dt   [N s]
    sq_area = amp * amp * dur * b1  # int h^2 dt [N^2 s]

    dt = case.pulse_scale_s() / float(samples_per_pulse)
    n_samples = int(np.ceil(duration_s / dt)) + 2
    t = np.arange(n_samples) * dt
    force = np.zeros(n_samples, dtype=float)

    if n_events:
        # vectorised superposition: every pulse occupies round(D/dt) samples
        start_index = np.rint(times / dt).astype(np.int64)
        span = np.maximum(2, np.rint(dur / dt).astype(np.int64))
        keep = start_index < n_samples
        start_index, span, amp_k = start_index[keep], span[keep], amp[keep]
        if start_index.size:
            offsets = np.cumsum(span) - span
            total = int(span.sum())
            local = np.arange(total) - np.repeat(offsets, span)
            index = np.repeat(start_index, span) + local
            tau = (local + 0.5) / np.repeat(span, span)
            values = np.repeat(amp_k, span) * case.pulse.evaluate(tau)
            valid = index < n_samples
            np.add.at(force, index[valid], values[valid])

    if case.noise_std_ratio > 0.0:
        scale = float(np.std(force)) if np.std(force) > 0.0 else 1.0
        force = force + case.noise_std_ratio * scale * rng.standard_normal(n_samples)

    marks = {
        "seed": int(seed),
        "encounter_times_s": times,
        "floe_m": floe,
        "thickness_m": thick,
        "amplitude_n": amp,
        "duration_s": dur,
        "area_ns": area,
        "sq_area_n2s": sq_area,
        "pulse_integrals": (a1, b1),
    }
    return Record(t=t, force=force, dt=dt, marks=marks)


def population_marks(case: CaseSpec, n_marks: int = 200_000,
                     seed: int = 0) -> tuple:
    """Monte-Carlo population of pulse marks (no record is built).

    Returns ``(areas, sq_areas)`` -- ``int h dt`` and ``int h^2 dt`` of
    ``n_marks`` independent encounters drawn from the case's spectra.  These give
    accurate *population* Campbell anchors ``mu = lambda E[A]``,
    ``sigma^2 = lambda E[B]`` against which a single record's moments can be
    judged (the record's own marks give the *conditional* anchor).
    """
    if n_marks < 100:
        raise ConfigError("population_marks needs at least 100 marks")
    rng = np.random.default_rng(int(seed))
    floe = case.floe_size.sample(rng, n_marks)
    thick = case.thickness.sample(rng, n_marks)
    amp = case.force.amplitude(rng, floe, thick)
    dur = case.pulse.duration_s(floe, case.speed_mps)
    a1, b1 = case.pulse.integrals()
    return amp * dur * a1, amp * amp * dur * b1


def window_mean(record: Record, start_s: float, length_s: float) -> float:
    """Mean of the record over ``[start_s, start_s + length_s)``."""
    i0 = int(np.searchsorted(record.t, start_s))
    i1 = int(np.searchsorted(record.t, start_s + length_s))
    if i1 - i0 < 2:
        raise ConfigError("window too short for the record time step")
    return float(np.mean(record.force[i0:i1]))


__all__ = ["Record", "population_marks", "simulate_record", "window_mean"]