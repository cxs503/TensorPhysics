"""Structured in-memory diagnostics without imposing a logging backend."""
from dataclasses import asdict, dataclass, field
from time import time
from typing import Any, Mapping

@dataclass(frozen=True)
class DiagnosticRecord:
    event: str
    step: int
    time: float
    values: Mapping[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time)

class Diagnostics:
    def __init__(self) -> None:
        self._records: list[DiagnosticRecord] = []

    def emit(self, event: str, step: int, time_value: float,
             values: Mapping[str, Any] | None = None) -> DiagnosticRecord:
        record = DiagnosticRecord(event, step, time_value, values or {})
        self._records.append(record)
        return record

    @property
    def records(self) -> tuple[DiagnosticRecord, ...]:
        return tuple(self._records)

    def to_dicts(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self._records]

    def clear(self) -> None:
        self._records.clear()
