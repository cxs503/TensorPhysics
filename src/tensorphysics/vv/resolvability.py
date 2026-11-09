"""Discriminability of a physical feature on a discretisation (the ``R`` law).

Motivation
----------
Richardson extrapolation answers *"does the discrete solution converge?"*.  It
does not answer *"is the feature I am reporting even present in the discrete
solution?"*  When the reported quantity is controlled by a physical length
:math:`\\delta` (boundary-layer thickness, vortex-core diameter, ice-floe or
contact-zone size, grating gap, ...) and the discretisation carries a spacing
:math:`\\Delta x`, the number of cells across the feature is

.. math::  R = \\delta / \\Delta x .

That single number is the discretisation-side discriminator of the W10-N07
"resolvability law": below ``R = 1`` the feature is **subgrid** and any
converged number is converging to the wrong physics; between ``1`` and the
resolution floor (default ``4`` cells, the conservative value for a
second-order accurate reconstruction of amplitude and gradient) the feature is
**marginal**; at or above the floor it is **resolved**.

Why a separate gate
-------------------
A study can be textbook-clean in Richardson terms -- monotone, order ``p = 2``,
small GCI -- and still be worthless because the boundary layer only spans half a
cell.  Therefore the machine gate must be able to return

``WITHHELD``
    the feature scale was not declared, so no resolvability statement is
    possible and the number must not be published as resolved;
``UNRESOLVED``
    the feature scale *was* declared and the grid does not resolve it.

Both statuses are *evidence statements*, not failures of the solver: they say
"the number is under-determined by this discretisation", which is exactly the
``WITHHELD``/``UNRESOLVED`` vocabulary of the platform evidence grades.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

__all__ = [
    "DiscriminabilityPolicy",
    "FeatureRecord",
    "FeatureSpec",
    "RESOLVED",
    "MARGINAL",
    "UNRESOLVED",
    "WITHHELD",
    "discriminability",
    "required_grid_spacing",
    "worst_ratio",
]

RESOLVED = "RESOLVED"
MARGINAL = "MARGINAL"
UNRESOLVED = "UNRESOLVED"
WITHHELD = "WITHHELD"

_SEVERITY = {RESOLVED: 0, MARGINAL: 1, WITHHELD: 2, UNRESOLVED: 3}


@dataclass(frozen=True)
class DiscriminabilityPolicy:
    """Resolution floor of the ``R = delta / dx`` law.

    ``min_cells_per_feature`` is the conservative default of the W10-N07 law:
    four cells across the feature give a second-order reconstruction of both the
    amplitude and the interior gradient (two cells is the Nyquist limit at which
    the amplitude of a sinusoidal feature is already off by ~10 %).
    ``subgrid_below`` separates "marginal" from "subgrid": a feature narrower
    than one cell cannot be represented at all.
    """

    min_cells_per_feature: float = 4.0
    subgrid_below: float = 1.0
    require_declaration: bool = True

    def __post_init__(self) -> None:
        if not (self.min_cells_per_feature > self.subgrid_below > 0.0):
            raise ValueError(
                "need 0 < subgrid_below < min_cells_per_feature, got "
                f"{self.subgrid_below} / {self.min_cells_per_feature}"
            )

    def merged(self, **overrides: Any) -> "DiscriminabilityPolicy":
        data = {k: getattr(self, k) for k in self.__dataclass_fields__}
        data.update({k: v for k, v in overrides.items() if v is not None})
        return DiscriminabilityPolicy(**data)

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass(frozen=True)
class FeatureSpec:
    """A physical feature whose resolution the gate must certify."""

    name: str
    delta: float
    unit: str = "m"
    source: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.delta, (int, float)) or not math.isfinite(self.delta) or self.delta <= 0.0:
            raise ValueError(f"feature {self.name!r}: delta must be finite and positive, got {self.delta!r}")

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "delta": self.delta, "unit": self.unit, "source": self.source}


@dataclass(frozen=True)
class FeatureRecord:
    """Resolution verdict of one feature on one grid."""

    name: str
    delta: float
    dx: float | None
    ratio: float | None
    status: str
    detail: str
    unit: str = "m"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "delta": self.delta,
            "dx": self.dx,
            "ratio": self.ratio,
            "status": self.status,
            "detail": self.detail,
            "unit": self.unit,
        }


@dataclass(frozen=True)
class ResolvabilityResult:
    """Gate-level resolvability statement over a list of features."""

    status: str
    dx: float | None
    features: tuple[FeatureRecord, ...] = field(default_factory=tuple)
    policy: DiscriminabilityPolicy = field(default_factory=DiscriminabilityPolicy)
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ratio(self) -> float | None:
        """Worst (smallest) resolution ratio of the study."""
        ratios = [r.ratio for r in self.features if r.ratio is not None]
        return min(ratios) if ratios else None

    @property
    def statuses(self) -> tuple[str, ...]:
        return tuple(r.status for r in self.features)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "tensorphysics.vv.resolvability/1",
            "status": self.status,
            "law": "R = delta / dx",
            "dx": self.dx,
            "ratio": self.ratio,
            "features": [f.as_dict() for f in self.features],
            "policy": self.policy.as_dict(),
            "reasons": list(self.reasons),
        }


def _classify(ratio: float, policy: DiscriminabilityPolicy) -> str:
    if ratio < policy.subgrid_below:
        return UNRESOLVED
    if ratio < policy.min_cells_per_feature:
        return MARGINAL
    return RESOLVED


def discriminability(
    features: Sequence[FeatureSpec | dict[str, Any]],
    dx: float | None,
    *,
    policy: DiscriminabilityPolicy | None = None,
) -> ResolvabilityResult:
    """Apply ``R = delta / dx`` to every feature at the finest spacing ``dx``.

    ``features`` may be :class:`FeatureSpec` objects or mappings with ``name``
    and ``delta`` (plus optional ``unit``/``source``).  An empty feature list
    yields ``WITHHELD`` when ``policy.require_declaration`` is set (the study
    made no claim that could be certified) and ``RESOLVED`` otherwise.
    """
    policy = policy or DiscriminabilityPolicy()

    if dx is not None:
        if not isinstance(dx, (int, float)) or not math.isfinite(dx) or dx <= 0.0:
            raise ValueError(f"dx must be finite and positive, got {dx!r}")
        dx = float(dx)

    records: list[FeatureRecord] = []
    reasons: list[str] = []
    for item in features:
        spec = item if isinstance(item, FeatureSpec) else FeatureSpec(**item)
        if dx is None:
            records.append(
                FeatureRecord(
                    name=spec.name,
                    delta=spec.delta,
                    dx=None,
                    ratio=None,
                    status=WITHHELD,
                    detail=f"{spec.name}: no grid spacing supplied, R is undefined",
                    unit=spec.unit,
                )
            )
            reasons.append(f"{spec.name}: grid spacing unknown")
            continue
        ratio = spec.delta / dx
        status = _classify(ratio, policy)
        if status == UNRESOLVED:
            detail = (
                f"{spec.name}: R = delta/dx = {spec.delta:g}/{dx:g} = {ratio:.4f} < "
                f"{policy.subgrid_below:g} -> feature is subgrid, the converged number is not the "
                "physics of this feature"
            )
            reasons.append(detail)
        elif status == MARGINAL:
            detail = (
                f"{spec.name}: R = {ratio:.4f} is in [{policy.subgrid_below:g}, "
                f"{policy.min_cells_per_feature:g}) -> feature is under-resolved, the gate cannot "
                f"certify it (needs dx <= {spec.delta / policy.min_cells_per_feature:g} {spec.unit})"
            )
            reasons.append(detail)
        else:
            detail = (
                f"{spec.name}: R = {ratio:.4f} >= {policy.min_cells_per_feature:g} -> feature resolved "
                f"on dx = {dx:g} {spec.unit}"
            )
        records.append(
            FeatureRecord(
                name=spec.name,
                delta=spec.delta,
                dx=dx,
                ratio=ratio,
                status=status,
                detail=detail,
                unit=spec.unit,
            )
        )

    if not records:
        status = WITHHELD if policy.require_declaration else RESOLVED
        if policy.require_declaration:
            reasons.append(
                "no feature scale declared: the gate cannot certify that the reported physics is "
                "resolved (declared WITHHELD)"
            )
        return ResolvabilityResult(status=status, dx=dx, features=(), policy=policy, reasons=tuple(reasons))

    status = max((r.status for r in records), key=lambda s: _SEVERITY[s])
    return ResolvabilityResult(status=status, dx=dx, features=tuple(records), policy=policy, reasons=tuple(reasons))


def required_grid_spacing(delta: float, *, policy: DiscriminabilityPolicy | None = None) -> float:
    """Largest spacing that still resolves a feature of size ``delta``."""
    policy = policy or DiscriminabilityPolicy()
    if not isinstance(delta, (int, float)) or not math.isfinite(delta) or delta <= 0.0:
        raise ValueError(f"delta must be finite and positive, got {delta!r}")
    return float(delta) / policy.min_cells_per_feature


def worst_ratio(results: Iterable[ResolvabilityResult]) -> float | None:
    """Smallest ratio over several results (``None`` when nothing was measurable)."""
    ratios = [r.ratio for r in results if r.ratio is not None]
    return min(ratios) if ratios else None