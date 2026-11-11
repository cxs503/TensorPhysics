"""Adaptive coupling clock: move the integer clock ratio so a declared bound fits a tolerance.

The controller acts on the one knob a partitioned code really has: the number of
fine steps per coupling window, an *integer* ratio ``r``.  Its inputs are the
computable bound of :mod:`tensorphysics.cross_clock.interpolate` and the
tolerance declared by the coupling contract; its output is the next ``r``.

Two safety properties are enforced rather than hoped for:

* **never accept a window that violates the tolerance** -- if the measured bound
  is above it, the ratio is reduced (window refined) before the step is taken;
* **no ratio churn** -- coarsening is allowed only when the bound is below
  ``tolerance / hysteresis`` and only by a bounded growth factor, so the
  controller cannot oscillate between two ratios around the tolerance.

The controller is deliberately stateless with respect to physics: it only sees
bounds and tolerances, so the same object drives the FEM-LBM and the DEM-LBM
pairs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

from .contract import ContractError

__all__ = ["AdaptiveClock", "ratio_scan", "AdaptiveRun", "MIN"]

MIN = 1


@dataclass
class AdaptiveClock:
    """Integer-ratio controller obeying a bound tolerance.

    ``order`` is the convergence order claimed for the exchange scheme: the
    bound is assumed to scale as ``h**order``, so the proposed ratio scales as
    ``(tolerance / bound) ** (1 / order)``.  A conservative factor ``safety``
    (<1) is applied to the predicted *ratio increase* only, which keeps the
    controller on the safe side without paying a full refinement whenever the
    estimate is nearly right.
    """

    ratio: int = 1
    tolerance: float = 1e-3
    order: int = 2
    min_ratio: int = MIN
    max_ratio: int = 4096
    safety: float = 0.8
    growth: float = 2.0
    hysteresis: float = 1.5
    history: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if not np.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ContractError(f"tolerance must be positive, got {self.tolerance!r}")
        if self.order < 1:
            raise ContractError("order must be >= 1")
        if not self.min_ratio <= self.ratio <= self.max_ratio:
            raise ContractError(f"initial ratio {self.ratio} outside [{self.min_ratio}, {self.max_ratio}]")
        if not 0.0 < self.safety <= 1.0:
            raise ContractError("safety must lie in (0, 1]")
        if self.growth <= 1.0:
            raise ContractError("growth must exceed 1")
        if self.hysteresis <= 1.0:
            raise ContractError("hysteresis must exceed 1")

    # ------------------------------------------------------------------ #
    def update(self, bound: float, measured: float | None = None) -> int:
        """Return the ratio to use for the next window given this window's bound.

        Direction convention: the *ratio* is the number of fine steps per coupling
        window, so a larger ratio means a coarser (worse) coupling.  The bound is
        modelled as ``bound ~ ratio**order``; violating the tolerance therefore
        *decreases* the ratio, and a comfortable margin allows increasing it by at
        most ``growth``.
        """
        if not np.isfinite(bound) or bound < 0.0:
            raise ContractError(f"bound must be finite and >= 0, got {bound!r}")
        old = self.ratio
        action = "hold"
        if bound > self.tolerance:
            need = (self.safety * self.tolerance / bound) ** (1.0 / self.order)
            new = int(np.floor(old * need))
            if new >= old:            # never fail to move when the bound is violated
                new = old - 1
            new = max(self.min_ratio, new)
            action = "refine"
        elif bound * self.hysteresis < self.tolerance:
            allow = int(
                np.floor(old * min(self.growth, (self.tolerance * self.safety / max(bound, 1e-300)) ** (1.0 / self.order)))
            )
            new = min(self.max_ratio, max(old, allow))
            action = "hold" if new == old else "coarsen"
        else:
            new = old
        new = int(np.clip(new, self.min_ratio, self.max_ratio))
        self.ratio = new
        self.history.append(
            {
                "ratio_before": old,
                "ratio_after": new,
                "bound": float(bound),
                "tolerance": self.tolerance,
                "measured": None if measured is None else float(measured),
                "action": action,
                "violated": bool(bound > self.tolerance),
            }
        )
        return new

    # ------------------------------------------------------------------ #
    def settle(
        self,
        bound_of: Callable[[int], float],
        max_iter: int = 40,
        min_iter: int = 1,
    ) -> dict:
        """Refine until the *measured* bound fits the tolerance.

        ``bound_of`` must evaluate a measured bound at that ratio (a fresh probe),
        never the controller's own model: a controller refining in the wrong
        direction then shows up as a non-converged result instead of being hidden
        behind its own assumption.  Coarsening is deliberately *not* attempted
        here -- settling is a feasibility search, not an efficiency search.
        """
        trail = []
        for i in range(max_iter):
            b = float(bound_of(self.ratio))
            trail.append({"iter": i, "ratio": self.ratio, "bound": b})
            if b <= self.tolerance and i + 1 >= min_iter:
                break
            prev = self.ratio
            nxt = self.update(b)
            if nxt == prev:      # the controller refused to move: infeasible in range
                break
        ok = trail[-1]["bound"] <= self.tolerance
        return {
            "converged": bool(ok),
            "iterations": len(trail),
            "final_ratio": self.ratio,
            "final_bound": trail[-1]["bound"],
            "tolerance": self.tolerance,
            "trail": trail,
            "history": list(self.history),
        }


def ratio_scan(bound_of: Callable[[int], float], ratios: Iterable[int], tolerance: float) -> dict:
    """Evaluate a measured bound over a ratio ladder; report the coarsest admissible ratio."""
    rows = []
    ok = []
    for r in ratios:
        b = float(bound_of(int(r)))
        rows.append({"ratio": int(r), "bound": b, "within_tolerance": bool(b <= tolerance)})
        if b <= tolerance:
            ok.append(int(r))
    return {
        "rows": rows,
        "ok_ratios": ok,
        "coarsest_admissible": max(ok) if ok else None,
        "tolerance": float(tolerance),
    }


@dataclass
class AdaptiveRun:
    """Bookkeeping of one adaptive coupled run."""

    contract_tolerance: float
    ratios: list = field(default_factory=list)
    bounds: list = field(default_factory=list)
    drifts: list = field(default_factory=list)
    actions: list = field(default_factory=list)

    def record(self, ratio: int, bound: float, drift: float | None, action: str = "hold") -> None:
        self.ratios.append(int(ratio))
        self.bounds.append(float(bound))
        self.drifts.append(None if drift is None else float(drift))
        self.actions.append(action)

    def summary(self) -> dict:
        b = np.asarray(self.bounds, dtype=float)
        d = np.asarray([v for v in self.drifts if v is not None], dtype=float)
        return {
            "tolerance": self.contract_tolerance,
            "windows": len(self.bounds),
            "ratio_first": self.ratios[0] if self.ratios else None,
            "ratio_last": self.ratios[-1] if self.ratios else None,
            "ratio_min": int(np.min(self.ratios)) if self.ratios else None,
            "ratio_max": int(np.max(self.ratios)) if self.ratios else None,
            "ratio_changes": int(sum(1 for a in self.actions if a != "hold")),
            "bound_max": float(np.max(b)) if b.size else None,
            "bound_median": float(np.median(b)) if b.size else None,
            "violations": int(np.sum(b > self.contract_tolerance)) if b.size else 0,
            "drift_max": float(np.max(d)) if d.size else None,
            "tolerance_met": bool(b.size and np.max(b) <= self.contract_tolerance),
            "drift_met": bool(d.size and np.max(d) < 1e-3),
        }


def decreasing_ratios(ratios: Sequence[int]) -> bool:
    return all(b < a for a, b in zip(ratios, ratios[1:]))