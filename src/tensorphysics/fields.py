"""Solver-neutral field metadata and time-stamped data exchange."""
from dataclasses import dataclass
from typing import Any


class FieldExchangeError(RuntimeError):
    """A field publication or read violates the exchange contract."""


@dataclass(frozen=True)
class FieldDescriptor:
    """Metadata required to interpret a field without owning its numerical array."""

    name: str
    producer: str
    units: str
    location: str
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        for key in ("name", "producer", "units", "location"):
            if not getattr(self, key).strip():
                raise ValueError(f"{key} must be a non-empty string")
        if self.shape is not None and any(not isinstance(n, int) or n < 0 for n in self.shape):
            raise ValueError("shape dimensions must be non-negative integers")


@dataclass(frozen=True)
class FieldValue:
    """A published field reference; data ownership remains with the producer."""

    descriptor: FieldDescriptor
    data: Any
    step: int
    time: float
    version: int


class FieldStore:
    """Deterministic in-memory exchange registry, independent of NumPy/PyTorch.

    The store deliberately does not copy arrays or convert units. A producer must
    publish at monotonically non-decreasing step/time; consumers can enforce a
    maximum step age when reading.
    """

    def __init__(self) -> None:
        self._descriptors: dict[str, FieldDescriptor] = {}
        self._values: dict[str, FieldValue] = {}

    @property
    def descriptors(self) -> dict[str, FieldDescriptor]:
        return dict(self._descriptors)

    def register(self, descriptor: FieldDescriptor) -> None:
        previous = self._descriptors.get(descriptor.name)
        if previous is not None:
            if previous != descriptor:
                raise FieldExchangeError(
                    f"field {descriptor.name!r} already registered with different metadata"
                )
            return
        self._descriptors[descriptor.name] = descriptor

    def publish(self, name: str, data: Any, *, producer: str, step: int, time: float) -> FieldValue:
        if name not in self._descriptors:
            raise FieldExchangeError(f"unregistered field: {name}")
        descriptor = self._descriptors[name]
        if producer != descriptor.producer:
            raise FieldExchangeError(
                f"field {name!r} is owned by producer {descriptor.producer!r}, not {producer!r}"
            )
        if step < 0 or time < 0:
            raise ValueError("step and time must be non-negative")
        actual_shape = getattr(data, "shape", None)
        if actual_shape is not None:
            actual_shape = tuple(int(n) for n in actual_shape)
            if descriptor.shape is not None and actual_shape != descriptor.shape:
                raise FieldExchangeError(
                    f"field {name!r} shape mismatch: expected {descriptor.shape}, got {actual_shape}"
                )
        previous = self._values.get(name)
        if previous is not None:
            if step < previous.step or time < previous.time:
                raise FieldExchangeError("field step and time must not move backwards")
            if step == previous.step and time != previous.time:
                raise FieldExchangeError("a field cannot have two times for the same step")
            version = previous.version + 1
        else:
            version = 1
        value = FieldValue(descriptor, data, step, float(time), version)
        self._values[name] = value
        return value

    def read(self, name: str, *, current_step: int | None = None,
             max_age_steps: int | None = None) -> FieldValue:
        if name not in self._descriptors:
            raise FieldExchangeError(f"unregistered field: {name}")
        if name not in self._values:
            raise FieldExchangeError(f"field has not been published: {name}")
        if max_age_steps is not None:
            if max_age_steps < 0:
                raise ValueError("max_age_steps must be non-negative")
            if current_step is None:
                raise ValueError("current_step is required when max_age_steps is set")
            value = self._values[name]
            if current_step < value.step:
                raise FieldExchangeError("current_step precedes the published field")
            if current_step - value.step > max_age_steps:
                raise FieldExchangeError(
                    f"field {name!r} is stale: age {current_step - value.step} steps "
                    f"exceeds limit {max_age_steps}"
                )
        return self._values[name]

    def clear_values(self) -> None:
        """Drop published data while retaining field contracts."""
        self._values.clear()
