"""Threshold extraction of encounter events from a resistance record.

A raw "ice resistance vs time" trace does not come with encounter labels.  We
recover them with a level-crossing detector with a dead time (merge) parameter:

1. mark samples above ``level``;
2. group them into contiguous runs;
3. merge runs separated by less than ``min_gap_s`` (the same physical contact
   stays one event while the signal dips through the level);
4. the event start is the first up-crossing of each merged run, its peak the
   maximum inside the run, its duration the run width, its impulse the
   trapezoidal integral over the run.

Two consequences are used explicitly by the acceptance criteria:

* **Thinning invariance.**  For a Poisson encounter process, dropping events by
  a rule that depends only on the individual mark leaves a Poisson process (of
  lower rate).  A detection threshold is (to the extent that neighbouring pulses
  do not mask each other) exactly such a marking, so Poisson-ness survives
  thresholding -- the KS test is not an artefact of the detector.
* **Clustering is not thinned away.**  Thinning a Cox (clustered) process keeps
  the clustering; the interval coefficient of variation stays well above 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EncounterEvents:
    """Recovered encounter events of one record."""

    level: float
    start_times: np.ndarray
    durations: np.ndarray
    peaks: np.ndarray
    impulses: np.ndarray

    @property
    def n_events(self) -> int:
        return int(self.start_times.size)

    @property
    def intervals(self) -> np.ndarray:
        if self.start_times.size < 2:
            return np.empty(0, dtype=float)
        return np.diff(self.start_times)

    @property
    def rate(self) -> float:
        if self.start_times.size < 2:
            return float("nan")
        span = float(self.start_times[-1] - self.start_times[0])
        return float((self.start_times.size - 1) / span) if span > 0.0 else float("nan")

    def to_dict(self) -> dict:
        return {
            "level": float(self.level),
            "n_events": self.n_events,
            "intervals_s": [float(x) for x in self.intervals],
            "peaks_n": [float(x) for x in self.peaks],
            "durations_s": [float(x) for x in self.durations],
            "impulses_ns": [float(x) for x in self.impulses],
        }


def _runs(mask: np.ndarray) -> tuple:
    """Start (inclusive) and end (exclusive) indices of True-runs of ``mask``."""
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    delta = np.diff(padded)
    starts = np.flatnonzero(delta == 1)
    ends = np.flatnonzero(delta == -1)
    return starts, ends


def auto_level(force: np.ndarray, quantile: float = 0.95,
               fraction: float = 0.30) -> float:
    """Detection level = ``fraction`` x a high quantile of the positive signal.

    Deliberately conservative (low) so that weak pulses are still detected; the
    quantile/fraction pair is injected from the manifest, not hard-coded.
    """
    if not (0.0 < quantile < 1.0) or fraction <= 0.0:
        raise ValueError("auto_level needs 0 < quantile < 1 and fraction > 0")
    positive = np.asarray(force, dtype=float)
    reference = float(np.quantile(positive, quantile))
    if reference <= 0.0:
        return float(np.max(positive) * fraction)
    return reference * fraction


def extract_events(t: np.ndarray, force: np.ndarray, level: float,
                   min_gap_s: float = 0.0) -> EncounterEvents:
    """Recover encounter events from a sampled record (see module docstring)."""
    t = np.asarray(t, dtype=float)
    force = np.asarray(force, dtype=float)
    if t.size != force.size or t.size < 3:
        raise ValueError("t and force must be equal-length with at least 3 samples")
    dt = float(t[1] - t[0])
    above = force >= level
    starts, ends = _runs(above)
    if starts.size == 0:
        return EncounterEvents(level=float(level), start_times=np.empty(0),
                               durations=np.empty(0), peaks=np.empty(0),
                               impulses=np.empty(0))

    # merge runs closer than min_gap_s
    merged_start = [int(starts[0])]
    merged_end = [int(ends[0])]
    gap_samples = int(round(min_gap_s / dt))
    for s, e in zip(starts[1:], ends[1:]):
        if int(s) - merged_end[-1] <= max(gap_samples, 0):
            merged_end[-1] = int(e)
        else:
            merged_start.append(int(s))
            merged_end.append(int(e))

    n_ev = len(merged_start)
    start_times = np.empty(n_ev)
    durations = np.empty(n_ev)
    peaks = np.empty(n_ev)
    impulses = np.empty(n_ev)
    for k, (s, e) in enumerate(zip(merged_start, merged_end)):
        start_times[k] = t[s]
        durations[k] = (e - s) * dt
        segment = force[s:e]
        peaks[k] = float(np.max(segment))
        impulses[k] = float(np.sum(segment) * dt)
    return EncounterEvents(level=float(level), start_times=start_times,
                           durations=durations, peaks=peaks, impulses=impulses)


__all__ = ["EncounterEvents", "auto_level", "extract_events"]