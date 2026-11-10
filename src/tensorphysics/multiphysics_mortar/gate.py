"""Machine gate for interface conservation residuals.

The gate is deliberately *fail-closed*: a measurement whose name has no declared
tolerance is an error, not a pass.  Tolerances are declared once, at module level,
so a run cannot silently relax them; :class:`ConservationGate` accepts an explicit
tolerance map only when a caller wants a different (documented) acceptance face.

Criterion tolerances (campaign node P162W1-N06):

===================  =========  =====================================================
key                  tol        meaning
===================  =========  =====================================================
``C1_1d_mass``       1e-12      1D scalar transport, mass residual (round-off level)
``C2_2d_momentum``   1e-8       2D FSI interface, momentum residual
``C2_2d_energy``     1e-6       2D FSI interface, energy residual
``C3_3d_mass``       1e-6       3D flow-ice patch, mass residual
``C3_3d_momentum``   1e-5       3D flow-ice patch, momentum residual
``C4_identity_1d``   1e-12      matching 1D meshes -> mortar is the identity map
``C4_identity_2d``   1e-12      matching 2D patch -> mortar is the identity map
===================  =========  =====================================================
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

__all__ = [
    "ConservationGate",
    "ConservationGateError",
    "GATE_TOLERANCES",
    "GateItem",
    "GateReport",
]

GATE_TOLERANCES: dict[str, float] = {
    "C1_1d_mass": 1e-12,
    "C2_2d_momentum": 1e-8,
    "C2_2d_energy": 1e-6,
    "C3_3d_mass": 1e-6,
    "C3_3d_momentum": 1e-5,
    "C4_identity_1d": 1e-12,
    "C4_identity_2d": 1e-12,
}


class ConservationGateError(RuntimeError):
    """Raised when a conservation measurement exceeds its declared tolerance."""


@dataclass(frozen=True)
class GateItem:
    name: str
    value: float
    tolerance: float

    @property
    def passed(self) -> bool:
        return bool(self.value <= self.tolerance)

    def to_dict(self) -> dict:
        return {"name": self.name, "value": float(self.value),
                "tolerance": float(self.tolerance), "passed": self.passed}


@dataclass(frozen=True)
class GateReport:
    items: tuple[GateItem, ...]
    distributions: dict[str, str] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.items)

    @property
    def failures(self) -> tuple[GateItem, ...]:
        return tuple(item for item in self.items if not item.passed)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "items": [item.to_dict() for item in self.items],
            "failures": [item.name for item in self.failures],
            "distributions": dict(self.distributions),
        }


@dataclass(frozen=True)
class ConservationGate:
    """Threshold set applied to a mapping of named conservation residuals."""

    tolerances: Mapping[str, float] = field(
        default_factory=lambda: dict(GATE_TOLERANCES))

    def evaluate(self, measured: Mapping[str, float],
                 distributions: Mapping[str, str] | None = None) -> GateReport:
        unknown = sorted(set(measured) - set(self.tolerances))
        if unknown:
            raise KeyError(f"no declared tolerance for measurement(s) {unknown}")
        items = tuple(GateItem(name=name, value=float(measured[name]),
                               tolerance=float(self.tolerances[name]))
                      for name in sorted(measured))
        return GateReport(items=items,
                          distributions=dict(distributions or {}))

    def require(self, measured: Mapping[str, float],
                distributions: Mapping[str, str] | None = None) -> GateReport:
        report = self.evaluate(measured, distributions)
        if not report.passed:
            detail = ", ".join(f"{item.name}={item.value:.3e} > {item.tolerance:.1e}"
                               for item in report.failures)
            raise ConservationGateError(f"conservation gate FAIL: {detail}")
        return report