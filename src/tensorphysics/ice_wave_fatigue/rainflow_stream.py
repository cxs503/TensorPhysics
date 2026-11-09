"""ASTM E1049-85 four-point rainflow counting with a *streaming* (block-wise) mode.

Why a second rainflow implementation in this package
----------------------------------------------------
The combination law of this node is tested on concatenations of stress blocks
(ice block + wave block) and criterion ``(d)`` demands that segmenting a history
and re-ordering the segments leaves the damage *bit-for-bit* unchanged (tolerance
1e-9).  Both requirements need the residue stack of the four-point algorithm to
survive across block boundaries; the in-place list implementation of
``tensorphysics.fatigue.rainflow`` restarts from scratch per call.  The cycle
definitions (turning-point extraction, ``r2 <= r1 and r2 <= r3`` extraction rule,
half-cycle residue accounting, half-cycle ``count = 0.5``) are identical, and
``tests/test_ice_wave_fatigue.py`` asserts agreement with that implementation on
a synthetic signal.

Algorithm (m5528 / ASTM E1049-85 four-point method)
---------------------------------------------------
Turning points are pushed onto a stack; whenever the last four stack entries
``y1 y2 y3 y4`` satisfy ``|y3-y2| <= |y2-y1|`` and ``|y3-y2| <= |y4-y3|`` the
inner pair ``(y2, y3)`` is a closed cycle (count 1.0) and is removed.  Whatever
remains on the stack when the record ends yields half cycles (count 0.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

__all__ = [
    "turning_points",
    "rainflow_cycles",
    "StreamingRainflow",
    "damage_from_cycles",
    "equivalent_stress_range",
]


def turning_points(series: Sequence[float]) -> np.ndarray:
    """Alternating peaks/valleys of ``series`` (equal neighbours folded).

    The first and last samples are always returned, as in the m5528 reference.
    """
    x = np.asarray(series, dtype=float).ravel()
    if x.size == 0:
        return x
    # fold consecutive duplicates
    keep = np.ones(x.size, dtype=bool)
    keep[1:] = np.diff(x) != 0.0
    y = x[keep]
    if y.size < 3:
        return y
    d = np.sign(np.diff(y))
    inner = np.where(d[:-1] * d[1:] < 0.0)[0] + 1
    return np.concatenate(([y[0]], y[inner], [y[-1]]))


def rainflow_cycles(points: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Four-point rainflow counting of a turning-point sequence.

    Returns ``(ranges, means, counts)``; half cycles carry ``count = 0.5``.
    Zero-range cycles are dropped (they carry no damage and only add noise).
    """
    pts = np.asarray(points, dtype=float).ravel()
    if pts.size < 3:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    stack: List[float] = []
    ranges: List[float] = []
    means: List[float] = []
    counts: List[float] = []
    for p in pts:
        stack.append(float(p))
        while len(stack) >= 4:
            y1, y2, y3, y4 = stack[-4], stack[-3], stack[-2], stack[-1]
            r1 = abs(y2 - y1)
            r2 = abs(y3 - y2)
            r3 = abs(y4 - y3)
            if r2 <= r1 and r2 <= r3:
                ranges.append(r2)
                means.append(0.5 * (y2 + y3))
                counts.append(1.0)
                del stack[-3:-1]
            else:
                break
    for i in range(len(stack) - 1):
        ranges.append(abs(stack[i + 1] - stack[i]))
        means.append(0.5 * (stack[i] + stack[i + 1]))
        counts.append(0.5)
    r = np.asarray(ranges, dtype=float)
    m = np.asarray(means, dtype=float)
    c = np.asarray(counts, dtype=float)
    keep = r > 0.0
    return r[keep], m[keep], c[keep]


@dataclass
class StreamingRainflow:
    """Block-wise rainflow counter with a persistent residue stack.

    Feeding ``[A, B]`` as one call and feeding ``A`` then ``B`` produce exactly
    the same cycle list in exactly the same order, therefore the same damage
    (criterion (d)).  ``finalize`` flushes the residue as half cycles; call it
    once, at the end of the record.
    """

    _carry: List[float] = field(default_factory=list, repr=False)
    _stack: List[float] = field(default_factory=list, repr=False)
    ranges: List[float] = field(default_factory=list, repr=False)
    means: List[float] = field(default_factory=list, repr=False)
    counts: List[float] = field(default_factory=list, repr=False)
    _started: bool = field(default=False, repr=False)
    _finalized: bool = field(default=False, repr=False)

    # -- internal -----------------------------------------------------------
    def _push(self, point: float) -> None:
        self._stack.append(float(point))
        while len(self._stack) >= 4:
            y1, y2, y3, y4 = self._stack[-4], self._stack[-3], self._stack[-2], self._stack[-1]
            r1 = abs(y2 - y1)
            r2 = abs(y3 - y2)
            r3 = abs(y4 - y3)
            if r2 <= r1 and r2 <= r3:
                self.ranges.append(r2)
                self.means.append(0.5 * (y2 + y3))
                self.counts.append(1.0)
                del self._stack[-3:-1]
            else:
                break

    # -- public -------------------------------------------------------------
    def feed(self, block: Sequence[float]) -> "StreamingRainflow":
        """Consume the next contiguous block of the *same* history."""
        if self._finalized:
            raise RuntimeError("this counter is finalised; create a new one")
        x = np.asarray(block, dtype=float).ravel()
        if x.size == 0:
            return self
        if not np.all(np.isfinite(x)):
            raise ValueError("history contains non-finite values")
        y = np.concatenate((self._carry, x)) if self._carry else x.copy()
        # fold consecutive duplicates inside the concatenated carry+block
        keep = np.ones(y.size, dtype=bool)
        keep[1:] = np.diff(y) != 0.0
        y = y[keep]
        if y.size < 2:
            self._carry = [float(y[-1])] if y.size else []
            return self
        if not self._started:
            self._push(float(y[0]))
            self._started = True
        d = np.sign(np.diff(y))
        inner = np.where(d[:-1] * d[1:] < 0.0)[0] + 1
        for idx in inner:
            self._push(float(y[idx]))
        # keep the last two samples as carry: the last one may still be
        # superseded by the next block, the one before it may become an extremum
        self._carry = [float(y[-2]), float(y[-1])]
        return self

    def finalize(self) -> "StreamingRainflow":
        """Close the record: append the last sample, then flush the residue."""
        if self._finalized:
            return self
        if self._carry:
            last = self._carry[-1]
            if not self._stack or self._stack[-1] != last:
                self._push(last)
        for i in range(len(self._stack) - 1):
            self.ranges.append(abs(self._stack[i + 1] - self._stack[i]))
            self.means.append(0.5 * (self._stack[i] + self._stack[i + 1]))
            self.counts.append(0.5)
        self._stack.clear()
        self._finalized = True
        return self

    def cycles(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._finalized:
            raise RuntimeError("call finalize() before reading the cycles")
        r = np.asarray(self.ranges, dtype=float)
        m = np.asarray(self.means, dtype=float)
        c = np.asarray(self.counts, dtype=float)
        keep = r > 0.0
        return r[keep], m[keep], c[keep]


def damage_from_cycles(ranges, counts, m: float = 3.0, C: float = 1.0) -> float:
    """Single-slope Miner sum ``sum n_i S_i^m / C`` (MPa-based)."""
    r = np.asarray(ranges, dtype=float)
    c = np.asarray(counts, dtype=float)
    if r.size == 0:
        return 0.0
    return float(np.sum(c * r ** m) / C)


def equivalent_stress_range(ranges, counts, m: float = 3.0) -> float:
    """``Delta sigma_eq = (sum n_i S_i^m / sum n_i)^(1/m)`` (m5528 definition)."""
    r = np.asarray(ranges, dtype=float)
    c = np.asarray(counts, dtype=float)
    if r.size == 0 or np.sum(c) <= 0:
        return 0.0
    return float((np.sum(c * r ** m) / np.sum(c)) ** (1.0 / m))