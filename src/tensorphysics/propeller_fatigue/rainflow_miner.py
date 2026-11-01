"""Rainflow cycle counting and Miner linear cumulative damage.

* :func:`turning_points` - reduce a sampled history to its extrema.
* :func:`rainflow_count` - ASTM E1049 three-point rainflow (half cycles from the
  residues are retained).
* :func:`miner_damage` - ``D = sum(n_i / N_i)`` with Basquin + Goodman
  equivalent amplitude.
* :func:`initiation_life` - exposure hours at ``D = 1``.
* :func:`safety_factor` - ``N_allowable / N_actual`` (identical to ``1/D`` for a
  linear S-N law, which is checked in the tests).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .materials_sn import BasquinSN, goodman_amplitude


def turning_points(x: np.ndarray) -> np.ndarray:
    """Extract turning points (local extrema + endpoints) from a history."""
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return x.copy()
    d = np.diff(x)
    # indices where the sign of the slope changes
    sign = np.sign(d)
    keep = np.where(np.diff(sign) != 0)[0] + 1
    out = np.concatenate(([x[0]], x[keep], [x[-1]]))
    # remove consecutive duplicates
    mask = np.ones(out.size, dtype=bool)
    mask[1:] = np.diff(out) != 0
    return out[mask]


def rainflow_count(history: np.ndarray, close_history: bool = False) -> Dict[str, np.ndarray]:
    """ASTM E1049 three-point rainflow count of a stress history.

    With ``close_history`` the record is rotated to start at its largest
    deviation from the mean, which is the standard convention for open records
    (it prevents the leading half cycle from being closed into a full cycle).

    Returns ``{"range", "mean", "count", "amplitude"}`` arrays.
    """
    tp = turning_points(history)
    if close_history and tp.size >= 3:
        idx = int(np.argmax(np.abs(tp - tp.mean())))
        tp = np.concatenate([tp[idx:], tp[:idx]])
    stack = []
    ranges = []
    means = []
    counts = []
    for p in tp:
        stack.append(float(p))
        while len(stack) >= 3:
            x_rng = abs(stack[-2] - stack[-3])
            y_rng = abs(stack[-1] - stack[-2])
            if x_rng <= y_rng:
                ranges.append(x_rng)
                means.append(0.5 * (stack[-3] + stack[-2]))
                counts.append(1.0)
                del stack[-3:-1]
            else:
                break
    # residues -> half cycles
    for i in range(len(stack) - 1):
        ranges.append(abs(stack[i + 1] - stack[i]))
        means.append(0.5 * (stack[i + 1] + stack[i]))
        counts.append(0.5)
    return {
        "range": np.asarray(ranges, dtype=float),
        "mean": np.asarray(means, dtype=float),
        "count": np.asarray(counts, dtype=float),
        "amplitude": np.asarray(ranges, dtype=float) / 2.0,
    }


def miner_damage(cycles: Dict[str, np.ndarray], sn: BasquinSN, sigma_uts: float,
                 scale: float = 1.0, n_bins: int = 24) -> Dict[str, object]:
    """Miner damage for a counted history, scaled by ``scale`` (exposure ratio)."""
    amp = np.asarray(cycles["amplitude"], dtype=float)
    mean = np.asarray(cycles["mean"], dtype=float)
    cnt = np.asarray(cycles["count"], dtype=float) * float(scale)
    keep = amp > 1.0                      # ignore numerical dust (<1 Pa)
    amp, mean, cnt = amp[keep], mean[keep], cnt[keep]
    if amp.size == 0:
        return {"D": 0.0, "bins": [], "n_cycles": 0.0,
                "sigma_a_eq": 0.0, "sigma_a_max": 0.0}
    amp_eff = goodman_amplitude(amp, mean, sigma_uts)
    N_i = sn.cycles(amp_eff)
    d_i = cnt / N_i
    m = 1.0 / abs(sn.b)
    sigma_eq = float((np.sum(cnt * amp_eff ** m) / np.sum(cnt)) ** (1.0 / m))
    bins = []
    edges = np.linspace(0.0, float(amp_eff.max()) * 1.0001, n_bins + 1)
    idx = np.clip(np.digitize(amp_eff, edges) - 1, 0, n_bins - 1)
    for b in range(n_bins):
        sel = idx == b
        if not np.any(sel):
            continue
        bins.append({
            "amp_lo_Pa": float(edges[b]),
            "amp_hi_Pa": float(edges[b + 1]),
            "cycles": float(np.sum(cnt[sel])),
            "damage": float(np.sum(d_i[sel])),
        })
    return {
        "D": float(np.sum(d_i)),
        "n_cycles": float(np.sum(cnt)),
        "sigma_a_eq": sigma_eq,
        "sigma_a_max": float(amp_eff.max()),
        "sigma_a_max_raw": float(amp.max()),
        "bins": bins,
        "damage_fraction_top_bin": float(max(b["damage"] for b in bins) / max(np.sum(d_i), 1e-300)),
    }


def initiation_life(damage_window: float, hours_window: float) -> float:
    """Exposure hours at which ``D = 1`` (linear extrapolation of Miner)."""
    if damage_window <= 0:
        return float("inf")
    return float(hours_window / damage_window)


def safety_factor(damage: float) -> float:
    """``SF = N_allowable / N_actual = 1 / D`` (Linear damage identity)."""
    if damage <= 0:
        return float("inf")
    return float(1.0 / damage)


def aggregate_realisations(damages) -> Tuple[float, float]:
    """Mean and standard error of Monte-Carlo damage realisations."""
    d = np.asarray(list(damages), dtype=float)
    mean = float(d.mean())
    sem = float(d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1 else 0.0
    return mean, sem