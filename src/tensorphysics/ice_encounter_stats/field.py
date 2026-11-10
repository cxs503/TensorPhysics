"""Encounter point processes on the ship track.

Geometry.  A ship of beam ``B`` advancing at speed ``V`` sweeps a corridor of
width ``B`` through the floe field.  For a homogeneous planar Poisson field of
floe centres with number density ``rho`` the centres inside the corridor form a
one-dimensional Poisson process in the along-track coordinate with rate
``rho * B`` per metre, hence in *time* a Poisson process with rate

    lambda = rho * B * V                                             (1)

This is an exact factorisation of the planar process onto the strip, so it is
the correct synthetic ground truth for "uniform floe field" (AC1).  Sampling the
along-track coordinate directly (rather than a 2-D rectangle) keeps the
realisation window exactly ``[0, T)`` with no edge leakage.

Negative control.  Replacing the Poisson driver by a Neyman--Scott cluster
process (Poisson cluster centres, Poisson(``mean_offspring``) offspring per
cluster, Gaussian along-track scatter ``sigma_s``) produces a *clustered* ice
field -- the shape of a wind-row / brash-ice / floe-chain field.  Its encounter
process is a Cox (doubly stochastic Poisson) process: the same mean rate, but
over-dispersed intervals that an exponential model must reject (AC2).
"""

from __future__ import annotations

import numpy as np

from .config import CaseSpec, ConfigError


def poisson_encounter_times(rng: np.random.Generator, rate_per_s: float,
                            duration_s: float) -> np.ndarray:
    """Sorted encounter times of a Poisson process on ``[0, duration_s)``."""
    if rate_per_s < 0.0 or duration_s <= 0.0:
        raise ConfigError("poisson encounter times need rate >= 0 and duration > 0")
    n = int(rng.poisson(rate_per_s * duration_s))
    if n == 0:
        return np.empty(0, dtype=float)
    return np.sort(rng.uniform(0.0, duration_s, size=n))


def neyman_scott_encounter_times(rng: np.random.Generator, cluster_rate_per_s: float,
                                 mean_offspring: float, sigma_s: float,
                                 duration_s: float) -> np.ndarray:
    """Sorted encounter times of a 1-D Neyman--Scott cluster process."""
    if cluster_rate_per_s < 0.0 or mean_offspring <= 0.0 or sigma_s <= 0.0:
        raise ConfigError("Neyman-Scott needs rate >= 0, mean_offspring > 0, sigma_s > 0")
    n_clusters = int(rng.poisson(cluster_rate_per_s * duration_s))
    if n_clusters == 0:
        return np.empty(0, dtype=float)
    centres = rng.uniform(0.0, duration_s, size=n_clusters)
    counts = rng.poisson(mean_offspring, size=n_clusters)
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=float)
    owner = np.repeat(np.arange(n_clusters), counts)
    times = centres[owner] + sigma_s * rng.standard_normal(total)
    times = times[(times >= 0.0) & (times < duration_s)]
    return np.sort(times)


def encounter_times(rng: np.random.Generator, case: CaseSpec,
                    duration_s: float) -> np.ndarray:
    """Encounter times for ``case``; cluster driver when ``case.clustering`` set."""
    if case.clustering is None:
        return poisson_encounter_times(rng, case.encounter_rate, duration_s)
    return neyman_scott_encounter_times(
        rng, case.driver_rate, case.clustering.mean_offspring,
        case.clustering.sigma_s, duration_s)


def interval_dispersion(intervals: np.ndarray) -> tuple:
    """(mean, std, coefficient of variation) of an interval sample."""
    iv = np.asarray(intervals, dtype=float)
    if iv.size < 2:
        return float("nan"), float("nan"), float("nan")
    return float(iv.mean()), float(iv.std(ddof=1)), float(iv.std(ddof=1) / iv.mean())


__all__ = [
    "encounter_times",
    "interval_dispersion",
    "neyman_scott_encounter_times",
    "poisson_encounter_times",
]