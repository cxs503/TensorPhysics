"""Time-synchronisation ledger: interpolation order and half-step offset.

The chain exchanges data at a coupling interval ``dt_h`` while the structural
integrator sub-steps.  Two error sources live in that gap and nowhere else:

* the **interpolation order** used to reconstruct the handed-over signal
  (0 = zero-order hold of the last delivered field, 1 = linear, 2 = quadratic),
* a **synchronisation offset**: a staggered scheme that labels a sample taken at
  ``t`` with ``t + dt_h/2`` shifts the whole hand-off by half a step.

This module quantifies the footprint of both on the three quantities the chain
cares about (force amplitude, wave-period work, impact/whipping energy).  It is
deliberately generic: it only reconstructs sampled signals and calls a metric
callback, so the same table can be produced for interface A and interface C.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

__all__ = ["reconstruct", "handoff_grid", "signal_error", "footprint"]


def reconstruct(
    t_src: np.ndarray,
    y_src: np.ndarray,
    t_dst: np.ndarray,
    order: int = 1,
    offset: float = 0.0,
) -> np.ndarray:
    """Reconstruct a sampled signal at ``t_dst`` with an interpolation order.

    Parameters
    ----------
    t_src, y_src : np.ndarray
        Sample times and values as produced by the sender (``y`` may be 1-D or
        2-D with the sample axis first).
    t_dst : np.ndarray
        Times at which the receiver needs the signal.
    order : {0, 1, 2}
        Reconstruction order.
    offset : float
        Synchronisation bias [s]: the sender *labels* the sample taken at ``t``
        with ``t + offset``.  ``offset = dt_h/2`` is the classic staggered-scheme
        half-step lag, ``offset = 0`` is a correctly synchronised exchange.

    Notes
    -----
    Outside the sampled window the end values are held (no extrapolation) so
    that the error at the window edges cannot flatter the interior accuracy.
    """
    t_src = np.asarray(t_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    t_dst = np.asarray(t_dst, dtype=float)
    if t_src.size < 2:
        raise ValueError("at least two samples are required")
    if order not in (0, 1, 2):
        raise ValueError("order must be 0, 1 or 2")

    # the receiver believes sample i is at (t_i + offset); invert that bias by
    # evaluating the received curve on the receiver's own time axis
    t_lab = t_src + float(offset)
    flat = y_src.ndim == 1
    y2 = y_src[:, None] if flat else y_src

    if order == 0:
        idx = np.clip(np.searchsorted(t_lab, t_dst, side="right") - 1, 0, t_lab.size - 1)
        out = y2[idx]
    elif order == 1:
        out = np.column_stack(
            [np.interp(t_dst, t_lab, y2[:, j]) for j in range(y2.shape[1])]
        )
    else:  # quadratic Lagrange on the three nearest samples
        out = np.empty((t_dst.size, y2.shape[1]))
        for j in range(y2.shape[1]):
            out[:, j] = _lagrange2(t_lab, y2[:, j], t_dst)
    return out[:, 0] if flat else out


def _lagrange2(t: np.ndarray, y: np.ndarray, tq: np.ndarray) -> np.ndarray:
    n = t.size
    i = np.clip(np.searchsorted(t, tq), 1, n - 1)
    i0 = np.clip(i - 1, 0, n - 3)
    out = np.empty(tq.size)
    for k in range(tq.size):
        a = i0[k]
        idx = np.array([a, a + 1, a + 2])
        x, f = t[idx], y[idx]
        # forward form of the three-point Lagrange polynomial
        out[k] = (
            f[0] * (tq[k] - x[1]) * (tq[k] - x[2]) / ((x[0] - x[1]) * (x[0] - x[2]))
            + f[1] * (tq[k] - x[0]) * (tq[k] - x[2]) / ((x[1] - x[0]) * (x[1] - x[2]))
            + f[2] * (tq[k] - x[0]) * (tq[k] - x[1]) / ((x[2] - x[0]) * (x[2] - x[1]))
        )
    return out


def handoff_grid(period: float, samples_per_period: int, start: float = 0.0) -> np.ndarray:
    """Uniform coupling instants covering ``period`` with the given density."""
    if samples_per_period < 2:
        raise ValueError("samples_per_period must be >= 2")
    dt = period / samples_per_period
    n = int(round(period / dt))
    return start + np.arange(n + 1) * dt


def signal_error(
    t_src: np.ndarray,
    y_src: np.ndarray,
    t_ref: np.ndarray,
    y_ref: np.ndarray,
    order: int,
    offset: float = 0.0,
) -> dict:
    """Max/RMS relative error of the reconstructed signal against a reference."""
    rec = reconstruct(t_src, y_src, t_ref, order=order, offset=offset)
    scale = max(float(np.max(np.abs(y_ref))), 1e-30)
    err = np.asarray(rec) - np.asarray(y_ref)
    return {
        "max_rel_err": float(np.max(np.abs(err)) / scale),
        "rms_rel_err": float(np.sqrt(np.mean(err ** 2)) / scale),
        "scale": scale,
    }


@dataclass(frozen=True)
class FootprintCase:
    """One row of the time ledger: a coupling interval, an order and an offset."""

    samples_per_period: int
    order: int
    offset_fraction: float       # offset in units of dt_h

    @property
    def label(self) -> str:
        return f"dt=T/{self.samples_per_period},{self.order}ord,off={self.offset_fraction:g}dt"


def footprint(
    metric: Callable[[float, int, float], dict],
    period: float,
    densities: Iterable[int] = (8, 16, 32, 64),
    orders: Iterable[int] = (0, 1, 2),
    offset_fractions: Iterable[float] = (0.0, 0.5),
) -> dict:
    """Assemble the error-footprint table.

    ``metric(dt_h, order, offset)`` must return a dict of numbers; every key is
    tabulated so that the caller decides which one is the detection metric.
    """
    rows = []
    for n in densities:
        dt_h = period / float(n)
        for order in orders:
            for frac in offset_fractions:
                vals = metric(dt_h, int(order), float(frac) * dt_h)
                rows.append(
                    {
                        "samples_per_period": int(n),
                        "dt_h_s": float(dt_h),
                        "order": int(order),
                        "offset_fraction": float(frac),
                        "offset_s": float(frac) * dt_h,
                        **{k: (float(v) if np.isscalar(v) else v) for k, v in vals.items()},
                    }
                )
    return {"period_s": float(period), "rows": rows}


def detection(rows: list[dict], key: str, threshold: float, offset_fraction: float = 0.5) -> dict:
    """Was the half-step offset detected by ``key`` at each density/order?

    ``detected`` is True when the offset case exceeds ``threshold`` *and* the
    synchronised case with the same order stays below it - a detector that fires
    on both has no discriminating power and is not credited.
    """
    out = []
    for r in rows:
        if abs(r["offset_fraction"] - offset_fraction) > 1e-12:
            continue
        base = [
            q
            for q in rows
            if q["samples_per_period"] == r["samples_per_period"]
            and q["order"] == r["order"]
            and abs(q["offset_fraction"]) < 1e-12
        ]
        if not base:
            continue
        out.append(
            {
                "samples_per_period": r["samples_per_period"],
                "order": r["order"],
                f"{key}_offset": float(r[key]),
                f"{key}_aligned": float(base[0][key]),
                "detected": bool(r[key] > threshold and base[0][key] <= threshold),
            }
        )
    return {
        "key": key,
        "threshold": float(threshold),
        "cases": out,
        "all_detected": bool(all(c["detected"] for c in out)) if out else False,
    }