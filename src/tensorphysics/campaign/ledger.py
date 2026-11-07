"""Cross-library dual-ledger reconciliation for mass, energy and impulse.

A coupled ice-region campaign moves three conserved (or conserved-by-accounting)
quantities between libraries: melt water / eroded ice mass, frictional and
crushing energy, and contact impulse.  Each library keeps its own book.  The two
books are *not* required to be bit-identical -- different summation orders are
legitimate -- but their difference must sit at the floating-point noise floor of
the bookkeeping, i.e. at machine precision of the ledger totals.

Tolerance model (per component ``i``)::

    scale_i  = max(|P_i|, |C_i|, sum_k |p_ki|, sum_k |c_ki|, tiny)
    tol_i    = atol_ulps * eps * scale_i

The ``scale`` term is a condition-number proxy: a book whose transfers of
opposite sign nearly cancel cannot be reconciled at the absolute level of its
total, so the L1 mass of its entries enters the tolerance.  Grades are
``exact`` (residual is literally zero), ``machine_precision`` (inside the
window) and ``out_of_tolerance`` (the books disagree beyond rounding).

``same_batch_digest`` is the SHA-256 of the deterministic reconciliation content;
re-running the same batch must reproduce it bit-for-bit, which is the
``diff = 0`` criterion of the campaign contract.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from .clock import CouplingClock, CouplingClockSpec
from .schema import SchemaError, payload_digest
from .schema_defs import validate_artifact, validate_quantity_book

EPS = sys.float_info.epsilon
TINY = sys.float_info.min
DEFAULT_ATOL_ULPS = 8.0

SIDES = ("producer", "consumer")


class LedgerError(ValueError):
    """A ledger entry, book, or reconciliation request is inconsistent."""


class QuantityKind(str, Enum):
    MASS = "mass"
    ENERGY = "energy"
    IMPULSE = "impulse"


#: Physical dimension (number of components) of each book.
DIMENSION: dict[QuantityKind, int] = {
    QuantityKind.MASS: 1,
    QuantityKind.ENERGY: 1,
    QuantityKind.IMPULSE: 3,
}

UNITS: dict[QuantityKind, str] = {
    QuantityKind.MASS: "kg",
    QuantityKind.ENERGY: "J",
    QuantityKind.IMPULSE: "N*s",
}


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LedgerError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise LedgerError(f"{name} must be finite")
    return number


@dataclass(frozen=True)
class LedgerEntry:
    """One transfer window on one side of the book."""

    label: str
    step: int
    components: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise LedgerError("entry label must be a non-empty string")
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0:
            raise LedgerError("entry step must be a non-negative integer")
        if not isinstance(self.components, tuple) or not self.components:
            raise LedgerError("entry components must be a non-empty tuple")
        if len(self.components) > 3:
            raise LedgerError("at most three vector components are supported")
        for index, value in enumerate(self.components):
            _finite(value, f"components[{index}]")

    def magnitude(self) -> float:
        """L1 norm of the entry (used for the cancellation scale)."""
        return math.fsum(abs(value) for value in self.components)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "step": self.step,
            "components": list(self.components),
        }


@dataclass(frozen=True)
class SideLedger:
    """One side (producer or consumer) of a quantity book."""

    quantity: QuantityKind
    side: str
    entries: tuple[LedgerEntry, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.quantity, QuantityKind):
            raise LedgerError("quantity must be a QuantityKind")
        if self.side not in SIDES:
            raise LedgerError(f"side must be one of {SIDES}")
        if not isinstance(self.entries, tuple):
            raise LedgerError("entries must be a tuple of LedgerEntry")
        width = DIMENSION[self.quantity]
        previous_step = -1
        for index, entry in enumerate(self.entries):
            if not isinstance(entry, LedgerEntry):
                raise LedgerError(f"entries[{index}] must be a LedgerEntry")
            if len(entry.components) != width:
                raise LedgerError(
                    f"{self.quantity.value} needs {width} component(s), "
                    f"entry {entry.label!r} has {len(entry.components)}"
                )
            if entry.step < previous_step:
                raise LedgerError("entries must be ordered by non-decreasing step")
            previous_step = entry.step

    @property
    def dimension(self) -> int:
        return DIMENSION[self.quantity]

    @property
    def units(self) -> str:
        return UNITS[self.quantity]

    def total(self) -> tuple[float, ...]:
        """Component-wise compensated sum of all transfers."""
        return tuple(
            math.fsum(entry.components[index] for entry in self.entries)
            for index in range(self.dimension)
        )

    def component_scale(self) -> tuple[float, ...]:
        """Sum of absolute transfers per component (cancellation proxy)."""
        return tuple(
            math.fsum(abs(entry.components[index]) for entry in self.entries)
            for index in range(self.dimension)
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_name": "side_ledger",
            "schema_version": 1,
            "quantity": self.quantity.value,
            "units": self.units,
            "side": self.side,
            "entries": [entry.to_dict() for entry in self.entries],
            "total": list(self.total()),
            "entry_count": len(self.entries),
        }
        validate_artifact("side_ledger.v1", payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Any) -> "SideLedger":
        try:
            validate_artifact("side_ledger.v1", payload)
        except SchemaError as exc:
            raise LedgerError(f"side ledger rejected: {exc}") from exc
        if payload["entry_count"] != len(payload["entries"]):
            raise LedgerError("entry_count does not match the entries array")
        ledger = cls(
            quantity=QuantityKind(payload["quantity"]),
            side=payload["side"],
            entries=tuple(
                LedgerEntry(
                    label=entry["label"],
                    step=int(entry["step"]),
                    components=tuple(float(value) for value in entry["components"]),
                )
                for entry in payload["entries"]
            ),
        )
        recorded = tuple(float(value) for value in payload["total"])
        recomputed = ledger.total()
        if len(recorded) != len(recomputed):
            raise LedgerError("recorded total has the wrong dimension")
        for index, (left, right) in enumerate(zip(recorded, recomputed)):
            if abs(left - right) > DEFAULT_ATOL_ULPS * EPS * max(abs(left), abs(right), TINY):
                raise LedgerError(
                    f"recorded total[{index}]={left!r} disagrees with the entries ({right!r})"
                )
        return ledger


@dataclass(frozen=True)
class ComponentResidual:
    """Reconciliation verdict for one vector component of one book."""

    component: int
    producer: float
    consumer: float
    residual: float
    abs_residual: float
    scale: float
    tolerance: float
    relative: float
    grade: str

    @property
    def passed(self) -> bool:
        return self.grade != "out_of_tolerance"

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "producer": self.producer,
            "consumer": self.consumer,
            "residual": self.residual,
            "abs_residual": self.abs_residual,
            "scale": self.scale,
            "tolerance": self.tolerance,
            "relative": self.relative,
            "grade": self.grade,
        }


@dataclass(frozen=True)
class QuantityReconciliation:
    """Reconciliation verdict for one quantity book pair."""

    quantity: QuantityKind
    residuals: tuple[ComponentResidual, ...]
    atol_ulps: float

    @property
    def units(self) -> str:
        return UNITS[self.quantity]

    @property
    def status(self) -> str:
        return "pass" if all(item.passed for item in self.residuals) else "fail"

    @property
    def max_abs_residual(self) -> float:
        return max(item.abs_residual for item in self.residuals)

    @property
    def max_relative_residual(self) -> float:
        return max(item.relative for item in self.residuals)

    @property
    def grades(self) -> tuple[str, ...]:
        return tuple(item.grade for item in self.residuals)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "quantity": self.quantity.value,
            "units": self.units,
            "component_count": len(self.residuals),
            "max_abs_residual": self.max_abs_residual,
            "max_relative_residual": self.max_relative_residual,
            "status": self.status,
            "residuals": [item.to_dict() for item in self.residuals],
        }
        validate_quantity_book(payload)
        return payload


def reconcile(
    producer: SideLedger,
    consumer: SideLedger,
    *,
    atol_ulps: float = DEFAULT_ATOL_ULPS,
) -> QuantityReconciliation:
    """Reconcile the two sides of one book at machine precision."""
    if not isinstance(producer, SideLedger) or not isinstance(consumer, SideLedger):
        raise LedgerError("reconcile() needs two SideLedger objects")
    if producer.quantity is not consumer.quantity:
        raise LedgerError("ledgers describe different quantities")
    if producer.side != "producer" or consumer.side != "consumer":
        raise LedgerError("argument order is (producer, consumer)")
    if _finite(atol_ulps, "atol_ulps") <= 0.0:
        raise LedgerError("atol_ulps must be > 0")
    if [entry.label for entry in producer.entries] != [entry.label for entry in consumer.entries]:
        raise LedgerError("both sides must book the same transfer windows")

    totals_p = producer.total()
    totals_c = consumer.total()
    scales_p = producer.component_scale()
    scales_c = consumer.component_scale()
    residuals: list[ComponentResidual] = []
    for index in range(producer.dimension):
        p_value = totals_p[index]
        c_value = totals_c[index]
        residual = p_value - c_value
        abs_residual = abs(residual)
        scale = max(abs(p_value), abs(c_value), scales_p[index], scales_c[index], TINY)
        tolerance = atol_ulps * EPS * scale
        if residual == 0.0:
            grade = "exact"
        elif abs_residual <= tolerance:
            grade = "machine_precision"
        else:
            grade = "out_of_tolerance"
        residuals.append(
            ComponentResidual(
                component=index,
                producer=p_value,
                consumer=c_value,
                residual=residual,
                abs_residual=abs_residual,
                scale=scale,
                tolerance=tolerance,
                relative=abs_residual / scale,
                grade=grade,
            )
        )
    return QuantityReconciliation(
        quantity=producer.quantity,
        residuals=tuple(residuals),
        atol_ulps=atol_ulps,
    )


def reconcile_books(
    books: Mapping[QuantityKind, tuple[SideLedger, SideLedger]],
    *,
    scenario: str,
    atol_ulps: float = DEFAULT_ATOL_ULPS,
) -> dict[str, Any]:
    """Reconcile a batch of books and emit the ``ledger_reconciliation.v1`` artifact."""
    if not isinstance(scenario, str) or not scenario.strip():
        raise LedgerError("scenario must be a non-empty string")
    if not books:
        raise LedgerError("at least one quantity book is required")
    ordered = [QuantityKind.MASS, QuantityKind.ENERGY, QuantityKind.IMPULSE]
    payload_books: list[dict[str, Any]] = []
    for quantity in ordered:
        if quantity not in books:
            continue
        producer, consumer = books[quantity]
        payload_books.append(reconcile(producer, consumer, atol_ulps=atol_ulps).to_dict())
    max_relative = max(book["max_relative_residual"] for book in payload_books)
    status = "pass" if all(book["status"] == "pass" for book in payload_books) else "fail"
    digest = payload_digest(
        {"scenario": scenario.strip(), "atol_ulps": float(atol_ulps), "quantity_books": payload_books}
    )
    payload = {
        "schema_name": "ledger_reconciliation",
        "schema_version": 1,
        "scenario": scenario.strip(),
        "atol_ulps": float(atol_ulps),
        "quantity_books": payload_books,
        "status": status,
        "max_relative_residual": max_relative,
        "same_batch_digest": digest,
    }
    validate_artifact("ledger_reconciliation.v1", payload)
    return payload


def verify_same_batch(payload: Mapping[str, Any]) -> str:
    """Recompute the same-batch digest of a reconciliation artifact.

    Returns ``"pass"`` when the artifact is a bit-faithful reproduction of its
    own deterministic content (the campaign ``diff = 0`` criterion) and
    ``"fail"`` when it is not.
    """
    try:
        validate_artifact("ledger_reconciliation.v1", payload)
    except SchemaError as exc:
        raise LedgerError(f"reconciliation artifact rejected: {exc}") from exc
    expected = payload_digest(
        {
            "scenario": payload["scenario"],
            "atol_ulps": payload["atol_ulps"],
            "quantity_books": payload["quantity_books"],
        }
    )
    return "pass" if expected == payload["same_batch_digest"] else "fail"


@dataclass(frozen=True)
class ReferenceFluxes:
    """Constant-in-time physical fluxes used to instantiate the ledger template."""

    mass_flow_kg_s: float
    power_w: float
    force_n: tuple[float, float, float]

    def __post_init__(self) -> None:
        _finite(self.mass_flow_kg_s, "mass_flow_kg_s")
        _finite(self.power_w, "power_w")
        if len(self.force_n) != 3:
            raise LedgerError("force_n needs three components")
        for index, value in enumerate(self.force_n):
            _finite(value, f"force_n[{index}]")


def build_reference_dual_ledgers(
    spec: CouplingClockSpec,
    master_ticks: int,
    fluxes: ReferenceFluxes,
) -> dict[QuantityKind, tuple[SideLedger, SideLedger]]:
    """Instantiate the mass/energy/impulse books for a known analytic exchange.

    Both sides book the same exchange windows produced by the coupling clock.
    The producer accumulates each window with :func:`math.fsum` (correctly
    rounded window total); the consumer accumulates the same window sequentially
    in floating point, which is the rounding pattern of an incremental
    integrator.  The window totals therefore differ only by summation rounding,
    and reconciliation must return to the machine-precision floor.
    """
    if isinstance(master_ticks, bool) or not isinstance(master_ticks, int) or master_ticks < 0:
        raise LedgerError("master_ticks must be a non-negative integer")
    if not isinstance(fluxes, ReferenceFluxes):
        raise LedgerError("fluxes must be ReferenceFluxes")
    clock = CouplingClock(spec)
    events = clock.run(master_ticks)
    sub_dt = spec.sub_dt_s
    sub_steps = spec.sub_steps_per_exchange

    books: dict[QuantityKind, tuple[SideLedger, SideLedger]] = {}
    for quantity in (QuantityKind.MASS, QuantityKind.ENERGY, QuantityKind.IMPULSE):
        producer_entries: list[LedgerEntry] = []
        consumer_entries: list[LedgerEntry] = []
        for event in events:
            terms = _window_terms(quantity, fluxes, sub_dt, sub_steps)
            producer_entries.append(
                LedgerEntry(
                    label=f"exchange_{event.index:04d}",
                    step=event.master_tick,
                    components=tuple(math.fsum(component) for component in terms),
                )
            )
            consumer_entries.append(
                LedgerEntry(
                    label=f"exchange_{event.index:04d}",
                    step=event.master_tick,
                    components=tuple(_sequential(component) for component in terms),
                )
            )
        books[quantity] = (
            SideLedger(quantity=quantity, side="producer", entries=tuple(producer_entries)),
            SideLedger(quantity=quantity, side="consumer", entries=tuple(consumer_entries)),
        )
    return books


def _window_terms(
    quantity: QuantityKind,
    fluxes: ReferenceFluxes,
    sub_dt: float,
    sub_steps: int,
) -> list[list[float]]:
    """Per-sub-step contributions inside one exchange window."""
    if quantity is QuantityKind.IMPULSE:
        return [
            [fluxes.force_n[index] * sub_dt for _ in range(sub_steps)]
            for index in range(3)
        ]
    rate = fluxes.mass_flow_kg_s if quantity is QuantityKind.MASS else fluxes.power_w
    return [[rate * sub_dt for _ in range(sub_steps)]]


def _sequential(values: Sequence[float]) -> float:
    """Naive left-to-right accumulation (an incremental integrator)."""
    total = 0.0
    for value in values:
        total += value
    return total


def ledger_totals(books: Mapping[QuantityKind, tuple[SideLedger, SideLedger]]) -> dict[str, Any]:
    """Convenience view of both sides' totals for reporting."""
    view: dict[str, Any] = {}
    for quantity, (producer, consumer) in books.items():
        view[quantity.value] = {
            "units": UNITS[quantity],
            "producer_total": list(producer.total()),
            "consumer_total": list(consumer.total()),
            "windows": len(producer.entries),
        }
    return view