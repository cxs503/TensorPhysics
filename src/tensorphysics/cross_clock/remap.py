"""Conservative time-rebinning of interval aggregates (never resample point values).

A clock-mismatched exchange hands a partner an *integral* over a window it did not
resolve.  Regridding that integral onto another partition must not change the
total: the only permitted operation is redistribution of shares, never
reconstruct-and-resample of a pointwise signal.

Two schemes, both exactly total-conserving (asserted, not assumed):

``donor``
    Each source interval's share is split by overlap length (piecewise-constant
    sub-cell density).  First order: the *time centroid* of the share can be off
    by up to half a source interval.
``plm``
    Piecewise-linear sub-cell profile with a van Leer slope limiter.  The profile
    is symmetric about the cell centre, so the cell total is unchanged (total
    conservation stays exact by construction) while the time centroid is carried
    to second order, and the limiter cannot create a new extremum.

The first/second-order claims are measured against an analytic signal in
``tests/test_cross_clock_remap.py``; that measurement is why ``plm`` is the
default of the coupling manager.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .contract import ContractError

__all__ = [
    "overlap_fractions",
    "van_leer_limiter",
    "plm_slopes",
    "conservative_regrid",
    "first_moment",
    "regrid_report",
    "RebinResult",
]


def _edges(edges: Sequence[float]) -> np.ndarray:
    e = np.asarray(edges, dtype=float).ravel()
    if e.size < 2:
        raise ContractError("need at least two edges")
    if np.any(np.diff(e) <= 0.0):
        raise ContractError("edges must be strictly increasing")
    return e


def overlap_fractions(src_edges: Sequence[float], dst_edges: Sequence[float]) -> np.ndarray:
    """Overlap durations ``O[i, j] = |dst_i ∩ src_j|``.

    Every row sums to the destination duration and every column to the source
    duration; that is exactly what makes the regrid total-conserving.
    """
    src = _edges(src_edges)
    dst = _edges(dst_edges)
    lo = np.maximum(dst[:-1, None], src[None, :-1])
    hi = np.minimum(dst[1:, None], src[None, 1:])
    O = np.clip(hi - lo, 0.0, None)
    O[O < 1e-300] = 0.0
    atol = 1e-12 * max(abs(dst[-1]), abs(src[-1]), 1.0)
    if not np.allclose(O.sum(axis=1), np.diff(dst), rtol=0.0, atol=atol):
        raise ContractError("destination partition is not covered by the source partition")
    if not np.allclose(O.sum(axis=0), np.diff(src), rtol=0.0, atol=atol):
        raise ContractError("source partition is not fully distributed onto the destination")
    return O


def van_leer_limiter(backward: np.ndarray, forward: np.ndarray) -> np.ndarray:
    """van Leer limiter ``2ab/(a+b)`` on same-sign slopes, zero otherwise."""
    b = np.asarray(backward, dtype=float)
    f = np.asarray(forward, dtype=float)
    out = np.zeros(np.broadcast(b, f).shape)
    same = (b * f) > 0.0
    num = 2.0 * b * f
    den = b + f
    out = np.where(same & (np.abs(den) > 1e-300), num / np.where(np.abs(den) > 1e-300, den, 1.0), 0.0)
    return out


def plm_slopes(totals: Sequence[float], edges: Sequence[float]) -> np.ndarray:
    """Limited sub-cell slopes of the density, in density-per-time units."""
    e = _edges(edges)
    q = np.asarray(totals, dtype=float).ravel()
    if q.size != e.size - 1:
        raise ContractError(f"expected {e.size - 1} totals, got {q.size}")
    h = np.diff(e)
    dens = q / h
    n = dens.size
    if n < 3:
        if n == 2:
            d = (dens[1] - dens[0]) / (0.5 * (h[0] + h[1]))
            return np.array([0.0, 0.0]) if d == 0.0 else np.array([0.0, 0.0])
        return np.zeros(n)
    left = np.zeros(n)
    right = np.zeros(n)
    left[1:] = (dens[1:] - dens[:-1]) / (0.5 * (h[1:] + h[:-1]))
    right[:-1] = left[1:]
    # rebuild one-sided slopes with the *interval* widths (second-order centred)
    left[1:] = (dens[1:] - dens[:-1]) / (0.5 * (h[1:] + h[:-1]))
    right[:-1] = (dens[1:] - dens[:-1]) / (0.5 * (h[1:] + h[:-1]))
    left[0] = 0.0
    right[-1] = 0.0
    return van_leer_limiter(left, right)


@dataclass(frozen=True)
class RebinResult:
    """Outcome of a conservative regrid."""

    dst_totals: np.ndarray
    scheme: str
    src_totals: np.ndarray
    dst_edges: np.ndarray
    src_edges: np.ndarray

    @property
    def total_src(self) -> float:
        return float(np.sum(self.src_totals))

    @property
    def total_dst(self) -> float:
        return float(np.sum(self.dst_totals))

    @property
    def residual(self) -> float:
        return self.total_dst - self.total_src

    def as_dict(self) -> dict:
        scale = max(abs(self.total_src), 1e-300)
        return {
            "scheme": self.scheme,
            "total_src": self.total_src,
            "total_dst": self.total_dst,
            "residual": self.residual,
            "relative_drift": abs(self.residual) / scale,
        }


def conservative_regrid(
    src_edges: Sequence[float],
    src_totals: Sequence[float],
    dst_edges: Sequence[float],
    scheme: str = "plm",
) -> RebinResult:
    """Redistribute interval totals onto another partition, conserving the total.

    ``donor`` splits by overlap length; ``plm`` integrates a slope-limited linear
    sub-cell profile over each overlap.  Both reproduce the source total to
    floating-point round-off.
    """
    src = _edges(src_edges)
    dst = _edges(dst_edges)
    q = np.asarray(src_totals, dtype=float).ravel()
    if q.size != src.size - 1:
        raise ContractError(f"expected {src.size - 1} source totals, got {q.size}")
    O = overlap_fractions(src, dst)
    h = np.diff(src)
    dens = q / h
    if scheme == "donor":
        out = O @ dens
    elif scheme == "plm":
        slopes = plm_slopes(q, src)
        # keep the profile bounded by same-sign neighbour averages: |s| <= |dens| / (h/2)
        cap = 2.0 * np.abs(dens) / h
        slopes = np.clip(slopes, -cap, cap)
        lo = np.maximum(dst[:-1, None], src[None, :-1])
        hi = np.minimum(dst[1:, None], src[None, 1:])
        ov = np.clip(hi - lo, 0.0, None)
        centre = (0.5 * (src[:-1] + src[1:]))[None, :]
        # ∫_{lo}^{hi} [dens + s (x - centre)] dx = dens*ov + s*ov*(mid - centre).
        # The form matters: writing the first moment as (hi^2 - lo^2)/2 leaves a
        # spurious contribution for *non-overlapping* pairs (where clipped ov is
        # zero but hi < lo), which silently breaks total conservation.
        mid = 0.5 * (lo + hi)
        out = np.sum(ov * (dens[None, :] + slopes[None, :] * (mid - centre)), axis=1)
    else:
        raise ContractError(f"unknown regrid scheme {scheme!r}")
    res = RebinResult(np.asarray(out, dtype=float), scheme, q, dst, src)
    scale = max(abs(res.total_src), 1e-300)
    if abs(res.residual) > 1e-12 * max(scale, 1e-12):
        raise ContractError(
            f"{scheme} regrid lost {res.residual!r} of {res.total_src!r}: scheme is not conservative"
        )
    return res


def first_moment(totals: Sequence[float], edges: Sequence[float], density: str = "centroid") -> float:
    """First moment ``∫ t q(t) dt`` of the represented distribution.

    ``centroid`` puts every total at its interval centre -- the assumption a
    held-rate exchange makes.  ``uniform`` spreads it uniformly, which is the
    better reconstruction of a sub-cell average and is therefore used as the
    reference when judging what a clock mismatch threw away.
    """
    e = _edges(edges)
    q = np.asarray(totals, dtype=float).ravel()
    if q.size != e.size - 1:
        raise ContractError("totals/edges length mismatch")
    if density in ("centroid", "uniform"):
        return float(np.sum(q * 0.5 * (e[:-1] + e[1:])))
    raise ContractError(f"unknown density representation {density!r}")


def regrid_report(src_edges, src_totals, dst_edges, schemes: Sequence[str] = ("donor", "plm")) -> dict:
    return {s: conservative_regrid(src_edges, src_totals, dst_edges, s).as_dict() for s in schemes}