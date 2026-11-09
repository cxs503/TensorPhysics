"""S-N bookkeeping for the ice+wave combination study (m5528 parameter family).

Two damage models live side by side and are deliberately *not* mixed:

``single_slope``  ``N = C * S^-m``  (used for the spectral-method benchmark of
                  criterion (b), because Dirlik / Tovo-Benasciutti / Rayleigh
                  damage rates are derived for a single power law; the same
                  ``C``/``m`` is used for the rainflow truth, so the comparison
                  is like-for-like).
``bilinear DNV``  DNV-RP-C203 two-slope curve with thickness correction and
                  optional Goodman mean-stress correction (used for the physical
                  Level-B cases and for the m5528 reproduction of criterion (e)).

Provenance of the curve constants: node IN32-N07 (m5528)
``backend/modules/ice_fatigue/sn_curves.py``, which is the DNV-RP-C203 table
expressed as ``(m1, log10 a1, m2, log10 a2, N_knee)`` plus a ``log10(2.5)``
life shift for seawater with cathodic protection.  The two tabulated anchors
(``Delta sigma(N=1e7)`` = 52.63 MPa for D in air and 31.86 MPa for F3 in air)
are asserted in ``tests/test_ice_wave_fatigue.py``.

Units: stress in MPa, thickness in mm, damage in cycles/cycles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

__all__ = [
    "SNCurve",
    "DNV_CURVES",
    "get_curve",
    "goodman_amplitude",
    "miner_damage",
    "fatigue_life",
    "single_slope_damage",
]

#: air-environment bilinear constants: name -> (m1, log10 a1, m2, log10 a2, N_knee)
_DNV_AIR: Dict[str, tuple] = {
    "DNV-D-air": (3.0, 12.164, 5.0, 15.606, 1.0e7),
    "DNV-E-air": (3.0, 12.010, 5.0, 15.351, 1.0e7),
    "DNV-F-air": (3.0, 11.855, 5.0, 15.091, 1.0e7),
    "DNV-F3-air": (3.0, 11.510, 5.0, 14.516, 1.0e7),
    "DNV-W3-air": (3.0, 10.691, 5.0, 13.152, 1.0e7),
}
#: seawater with cathodic protection: life x 2.5  <=>  log10 a -= log10(2.5)
_CP_SHIFT = math.log10(2.5)

DNV_CURVES: Dict[str, tuple] = {}
for _k, (_m1, _a1, _m2, _a2, _nk) in _DNV_AIR.items():
    DNV_CURVES[_k] = (_m1, _a1, _m2, _a2, _nk)
    DNV_CURVES[_k.replace("-air", "-sw-cp")] = (_m1, _a1 - _CP_SHIFT, _m2, _a2 - _CP_SHIFT, _nk)


@dataclass
class SNCurve:
    """Bilinear DNV-RP-C203 curve with thickness correction."""

    name: str = "DNV-F3-sw-cp"
    m1: float = 3.0
    log_a1: float = 11.510 - _CP_SHIFT
    m2: float = 5.0
    log_a2: float = 14.516 - _CP_SHIFT
    n_knee: float = 1.0e7
    t_ref: float = 25.0
    thickness_exp: float = 0.2

    def __post_init__(self) -> None:
        if self.m1 <= 0 or self.m2 <= 0:
            raise ValueError("slopes must be positive")

    # -- knee stress --------------------------------------------------------
    @property
    def dsigma_knee(self) -> float:
        return 10.0 ** ((self.log_a1 - math.log10(self.n_knee)) / self.m1)

    def thickness_factor(self, t: Optional[float] = None) -> float:
        if t is None:
            return 1.0
        return (float(t) / self.t_ref) ** self.thickness_exp

    # -- N(S) ---------------------------------------------------------------
    def life(self, dsigma: np.ndarray, thickness: Optional[float] = None) -> np.ndarray:
        ds = np.abs(np.asarray(dsigma, dtype=float)) / self.thickness_factor(thickness)
        with np.errstate(divide="ignore", invalid="ignore"):
            n1 = 10.0 ** (self.log_a1 - self.m1 * np.log10(ds))
            n2 = 10.0 ** (self.log_a2 - self.m2 * np.log10(ds))
        return np.where(ds > self.dsigma_knee, n1, n2)

    def dsigma(self, n: float, thickness: Optional[float] = None) -> float:
        if n <= self.n_knee:
            ds = 10.0 ** ((self.log_a1 - math.log10(n)) / self.m1)
        else:
            ds = 10.0 ** ((self.log_a2 - math.log10(n)) / self.m2)
        return float(ds * self.thickness_factor(thickness))


def get_curve(name: str = "DNV-F3-sw-cp") -> SNCurve:
    key = name.upper().replace("DNV", "DNV").replace(" ", "")
    lookup = {k.upper(): k for k in DNV_CURVES}
    if key not in lookup:
        raise KeyError(f"unknown S-N curve {name!r}; available: {sorted(DNV_CURVES)}")
    m1, a1, m2, a2, nk = DNV_CURVES[lookup[key]]
    return SNCurve(name=lookup[key], m1=m1, log_a1=a1, m2=m2, log_a2=a2, n_knee=nk)


def goodman_amplitude(stress_amplitude, mean_stress, s_uts: float):
    """``sigma_a,eff = sigma_a / (1 - sigma_m / sigma_UTS)`` (m5528 form)."""
    if s_uts <= 0:
        raise ValueError("s_uts must be positive")
    sa = np.abs(np.asarray(stress_amplitude, dtype=float))
    sm = np.asarray(mean_stress, dtype=float)
    denom = 1.0 - sm / s_uts
    with np.errstate(divide="ignore", invalid="ignore"):
        out = sa / denom
    return np.where(denom > 0.0, out, np.inf)


def miner_damage(
    ranges: np.ndarray,
    means: np.ndarray,
    counts: np.ndarray,
    curve: SNCurve,
    s_uts: Optional[float] = None,
    thickness: Optional[float] = None,
    exposure_scale: float = 1.0,
) -> Dict[str, object]:
    """``D = sum_i n_i / N_i`` with optional Goodman + thickness correction."""
    r = np.asarray(ranges, dtype=float)
    c = np.asarray(counts, dtype=float)
    mn = np.asarray(means, dtype=float)
    if not (r.size == c.size == mn.size):
        raise ValueError("ranges, means and counts must have equal length")
    ds = r.copy()
    if s_uts is not None:
        ds = 2.0 * goodman_amplitude(0.5 * r, mn, s_uts)
    keep = np.isfinite(ds) & (ds > 0.0) & (c > 0.0)
    if not np.any(keep):
        return {"damage": 0.0, "n_cycles": 0.0, "n_infinite": 0.0}
    n_allow = curve.life(ds[keep], thickness)
    d_i = (c[keep] * exposure_scale) / n_allow
    n_inf = float(np.sum(c[~np.isfinite(ds)] * exposure_scale)) if np.any(~np.isfinite(ds)) else 0.0
    return {
        "damage": float(np.sum(d_i)),
        "n_cycles": float(np.sum(c[keep]) * exposure_scale),
        "n_infinite": n_inf,
        "worst_dsigma": float(np.max(ds[keep])),
    }


def single_slope_damage(ranges: np.ndarray, counts: np.ndarray,
                        m: float = 3.0, C: float = 1.0) -> float:
    """``D = sum_i n_i S_i^m / C`` -- the damage model behind the spectral methods.

    Using ``C = 1`` makes ``D`` carry units of MPa^m; only *ratios* of this
    quantity are ever compared (rainflow truth vs Dirlik vs Tovo-Benasciutti),
    so the constant is irrelevant as long as it is the same everywhere.
    """
    r = np.asarray(ranges, dtype=float)
    c = np.asarray(counts, dtype=float)
    if r.size == 0:
        return 0.0
    return float(np.sum(c * r ** m) / C)


def fatigue_life(damage: float, exposure: float) -> float:
    if damage <= 0.0:
        return math.inf
    return float(exposure / damage)