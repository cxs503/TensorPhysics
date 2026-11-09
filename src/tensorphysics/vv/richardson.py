"""High-level Richardson extrapolation / GCI engine.

This module provides a class-based, user-friendly wrapper around the
solver-neutral functional gate in :mod:`tensorphysics.vv.gci`.  It implements
the ASME V&V 20 / Roache (1994) grid-convergence methodology:

    * observed order of convergence ``p`` from three (or more) grids,
    * Richardson extrapolated continuum value ``f_ext``,
    * grid convergence index ``GCI`` of the finest and coarsest pairs,
    * asymptotic-range / convergence-status assessment.

The class API is deliberately thin: all numerical work is delegated to the
functional gate so that the two layers cannot drift apart.  A caller that
needs the full check ledger should use :func:`tensorphysics.vv.gci.evaluate`
directly; this module is for the common "three grids, one observable" case.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .gci import (
    FAIL,
    PASS,
    WARN,
    GatePolicy,
    Observation,
    gci,
    grid_convergence_study,
    observed_order,
    richardson_extrapolated,
)

__all__ = [
    "ConvergenceStatus",
    "GCIResult",
    "RichardsonExtrapolation",
]


class ConvergenceStatus(Enum):
    """Convergence determination status (high-level API)."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    UNRESOLVED = "UNRESOLVED"


@dataclass
class GCIResult:
    """Result of a Richardson / GCI study (high-level API).

    All fields are populated by :meth:`RichardsonExtrapolation.compute`.
    ``gci_fine`` / ``gci_coarse`` are **percentages** of the finest / coarsest
    value (Roache convention); ``p`` is the observed order of convergence.
    """

    f_coarse: float
    f_medium: float
    f_fine: float
    r_coarse: float
    r_fine: float
    p: float
    f_ext: float
    epsilon_fine: float
    epsilon_coarse: float
    gci_fine: float
    gci_coarse: float
    gci_ratio: float
    status: ConvergenceStatus
    asymptotic: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dictionary."""
        return {
            "f_coarse": self.f_coarse,
            "f_medium": self.f_medium,
            "f_fine": self.f_fine,
            "r_coarse": self.r_coarse,
            "r_fine": self.r_fine,
            "p": self.p,
            "f_ext": self.f_ext,
            "epsilon_fine": self.epsilon_fine,
            "epsilon_coarse": self.epsilon_coarse,
            "gci_fine": self.gci_fine,
            "gci_coarse": self.gci_coarse,
            "gci_ratio": self.gci_ratio,
            "status": self.status.value,
            "asymptotic": self.asymptotic,
        }


class RichardsonExtrapolation:
    """Richardson extrapolation with GCI (high-level API).

    Parameters
    ----------
    safety_factor : float
        Factor of safety ``Fs`` for the GCI (Roache 1994).  Default 1.25 for
        a three-grid study; pass 3.0 when only two grids are available.
    gci_threshold : float
        GCI threshold (percent) above which the study is reported FAIL.
        Default 5.0.
    """

    def __init__(self, safety_factor: float = 1.25, gci_threshold: float = 5.0):
        if not math.isfinite(safety_factor) or safety_factor <= 0.0:
            raise ValueError(f"safety_factor must be finite and positive, got {safety_factor!r}")
        if not math.isfinite(gci_threshold) or gci_threshold <= 0.0:
            raise ValueError(f"gci_threshold must be finite and positive, got {gci_threshold!r}")
        self.safety_factor = safety_factor
        self.gci_threshold = gci_threshold

    def compute(
        self,
        f_fine: float,
        f_medium: float,
        f_coarse: float,
        r_fine: float = 2.0,
        r_coarse: float = 2.0,
    ) -> GCIResult:
        """Run a three-grid Richardson / GCI study.

        Parameters
        ----------
        f_fine, f_medium, f_coarse : float
            Solution values on the finest / medium / coarsest grids.
        r_fine : float
            Refinement ratio ``h_medium / h_fine`` (> 1).
        r_coarse : float
            Refinement ratio ``h_coarse / h_medium`` (> 1).

        Returns
        -------
        GCIResult
            Complete study with ``p``, ``f_ext``, ``GCI`` and status.
        """
        for name, value in (
            ("f_fine", f_fine),
            ("f_medium", f_medium),
            ("f_coarse", f_coarse),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
        for name, value in (("r_fine", r_fine), ("r_coarse", r_coarse)):
            if not math.isfinite(value) or value <= 1.0:
                raise ValueError(f"{name} must be finite and > 1, got {value!r}")

        # Build the three-grid study (coarse -> fine).
        # h is normalised so that the finest grid has h=1.
        levels = (
            Observation(h=1.0, f=f_coarse),
            Observation(h=1.0 / r_coarse, f=f_medium),
            Observation(h=1.0 / (r_coarse * r_fine), f=f_fine),
        )

        # Observed order (closed form for uniform three-grid).
        p, source, _ = observed_order(levels)
        if p is None or not math.isfinite(p) or p <= 0.0:
            # Non-asymptotic (oscillatory / diverging / stagnant).
            return self._non_asymptotic(
                f_fine, f_medium, f_coarse, r_fine, r_coarse
            )

        # Richardson extrapolated value.
        f_ext = richardson_extrapolated(levels, p)

        # GCI of the finest pair (percent of the finest value).
        # gci() returns (value, is_relative); value is the relative GCI.
        gci_fine_val, _ = gci(levels[1], levels[2], p, self.safety_factor)
        gci_fine_pct = 100.0 * gci_fine_val if f_fine != 0.0 else float("inf")

        # GCI of the coarsest pair (percent of the coarsest value).
        gci_coarse_val, _ = gci(levels[0], levels[1], p, self.safety_factor)
        gci_coarse_pct = 100.0 * gci_coarse_val if f_coarse != 0.0 else float("inf")

        # Relative errors (percent).
        eps_fine = 100.0 * abs(f_fine - f_medium) / abs(f_fine) if f_fine != 0.0 else float("inf")
        eps_coarse = 100.0 * abs(f_coarse - f_medium) / abs(f_coarse) if f_coarse != 0.0 else float("inf")

        # GCI ratio (should be ~1.0 in the asymptotic range).
        gci_ratio = (
            gci_coarse_pct / (r_coarse ** p * gci_fine_pct)
            if gci_fine_pct > 0.0 and math.isfinite(gci_fine_pct)
            else float("inf")
        )

        # Asymptotic-range check.
        asymptotic = (
            abs(gci_ratio - 1.0) < 0.5
            and gci_fine_pct < self.gci_threshold
            and math.isfinite(f_ext)
        )

        # Status.
        if gci_fine_pct > self.gci_threshold:
            status = ConvergenceStatus.FAIL
        elif not asymptotic:
            status = ConvergenceStatus.WARN
        else:
            status = ConvergenceStatus.PASS

        return GCIResult(
            f_coarse=f_coarse,
            f_medium=f_medium,
            f_fine=f_fine,
            r_coarse=r_coarse,
            r_fine=r_fine,
            p=p,
            f_ext=f_ext,
            epsilon_fine=eps_fine,
            epsilon_coarse=eps_coarse,
            gci_fine=gci_fine_pct,
            gci_coarse=gci_coarse_pct,
            gci_ratio=gci_ratio,
            status=status,
            asymptotic=asymptotic,
        )

    def _non_asymptotic(
        self,
        f_fine: float,
        f_medium: float,
        f_coarse: float,
        r_fine: float,
        r_coarse: float,
    ) -> GCIResult:
        """Return a FAIL result for a non-asymptotic (oscillatory) study."""
        eps_fine = 100.0 * abs(f_fine - f_medium) / abs(f_fine) if f_fine != 0.0 else float("inf")
        eps_coarse = 100.0 * abs(f_coarse - f_medium) / abs(f_coarse) if f_coarse != 0.0 else float("inf")
        return GCIResult(
            f_coarse=f_coarse,
            f_medium=f_medium,
            f_fine=f_fine,
            r_coarse=r_coarse,
            r_fine=r_fine,
            p=0.0,
            f_ext=float("nan"),
            epsilon_fine=eps_fine,
            epsilon_coarse=eps_coarse,
            gci_fine=float("inf"),
            gci_coarse=float("inf"),
            gci_ratio=float("inf"),
            status=ConvergenceStatus.FAIL,
            asymptotic=False,
        )
