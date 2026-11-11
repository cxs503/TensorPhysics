"""Discretisation (grid) convergence and Richardson extrapolation.

The Smith method has exactly one discretisation parameter: the number of fibres
used across every element in the through-thickness direction (``n_sub``) plus,
for sections whose webs are idealised as axial strips, the number of web strips
along the depth (``n_web_strips``).  Both enter the analysis through the same
generic mechanism - they subdivide the section into axial load-carrying
elements - so a grid study is a sequence of sections built on successively
finer subdivisions with total area, centroid and boundary conditions unchanged.

The convergence driver follows the standard verification practice (Roache):
three grids with a constant refinement ratio ``r``, observed order of accuracy
``p = ln|f2-f1| / ln(r)`` (0th/1st/2nd differences), Richardson estimate
``f_ext = f3 + (f3 - f2)/(r^p - 1)``, and a discretisation uncertainty
``U = |f3 - f_ext| / f_ext`` reported as the residual.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

__all__ = ["richardson_extrapolation", "grid_convergence_study", "RichardsonResult"]


@dataclass
class RichardsonResult:
    values: List[float]
    levels: List[float]
    ratio: float
    order: float
    extrapolated: float
    residual: float
    monotone: bool
    monotone_direction: int
    gci_coarse: float
    gci_medium: float
    ok: bool
    note: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {
            "values": list(self.values),
            "levels": list(self.levels),
            "refinement_ratio": self.ratio,
            "observed_order": self.order,
            "extrapolated": self.extrapolated,
            "residual_finest_vs_extrapolated": self.residual,
            "monotone": self.monotone,
            "monotone_direction": self.monotone_direction,
            "gci_coarse": self.gci_coarse,
            "gci_medium": self.gci_medium,
            "ok": self.ok,
            "note": self.note,
        }


def richardson_extrapolation(
    values: Sequence[float], ratio: float = 2.0, tol: float = 1e-12
) -> RichardsonResult:
    """Richardson extrapolation from three grids ordered coarse -> fine.

    Monotonicity is assessed by the sign of the successive differences; the
    observed order requires ``|f2 - f1| / |f3 - f2| > 0`` and both differences
    larger than ``tol`` relative to the finest value.
    """
    vals = [float(v) for v in values]
    if len(vals) != 3:
        raise ValueError("Richardson extrapolation needs exactly three grid levels")
    if ratio <= 1.0:
        raise ValueError("refinement ratio must be > 1")
    f1, f2, f3 = vals
    d12, d23 = f2 - f1, f3 - f2
    scale = max(abs(f3), tol)
    monotone = abs(d12) > tol * scale and abs(d23) > tol * scale and (d12 * d23 > 0.0)
    direction = 0 if abs(d23) <= tol * scale else (1 if d23 > 0 else -1)
    note = ""
    if monotone:
        order = math.log(abs(d12 / d23)) / math.log(ratio)
        if not math.isfinite(order) or order <= 0.0:
            note = "non-physical observed order; first-order estimate used"
            order = 1.0
        extrapolated = f3 + (f3 - f2) / (ratio**order - 1.0)
    else:
        order = float("nan")
        extrapolated = f3
        note = ("differences not monotone (or too small to resolve): observed order "
                "undefined; the finest value is reported instead of an extrapolation")
    residual = abs(f3 - extrapolated) / abs(extrapolated) if extrapolated else float("inf")
    if monotone and math.isfinite(order):
        gci_medium = 1.25 * abs((f3 - f2) / f3) / (ratio**order - 1.0) if f3 else float("inf")
        gci_coarse = 1.25 * abs((f2 - f1) / f2) / (ratio**order - 1.0) if f2 else float("inf")
    else:
        gci_medium = gci_coarse = float("nan")
    return RichardsonResult(
        values=vals, levels=[1.0 / (ratio**i) for i in range(3)], ratio=ratio,
        order=order, extrapolated=float(extrapolated), residual=float(residual),
        monotone=bool(monotone), monotone_direction=int(direction),
        gci_coarse=float(gci_coarse), gci_medium=float(gci_medium),
        ok=bool(monotone), note=note,
    )


def grid_convergence_study(
    runner: Callable[[int], float],
    levels: Sequence[int] = (1, 2, 4),
    ratio: float = 2.0,
) -> Dict[str, object]:
    """Run ``runner(level)`` on a refined sequence and Richardson-extrapolate.

    ``levels`` are refinement indices of the discretisation (coarse -> fine);
    the refinement ratio must be constant for the extrapolation to be valid.
    """
    levels = list(levels)
    if len(levels) != 3:
        raise ValueError("three levels required for the convergence study")
    values = [float(runner(int(l))) for l in levels]
    rich = richardson_extrapolation(values, ratio=ratio)
    out = {
        "levels": levels,
        "values": values,
        "richardson": rich.as_dict(),
        "monotone": rich.monotone,
        "residual": rich.residual,
        "extrapolated": rich.extrapolated,
    }
    return out