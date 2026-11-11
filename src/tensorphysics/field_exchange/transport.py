"""Field exchange *contract*: descriptors, units, copy ledger, expectations.

This module is the type/unit/bit-exactness half of the in-memory cross-solver
field exchange (GAP-CROSS-FIELD).  The transports live in
:mod:`tensorphysics.field_exchange.channels`; together they let two real solver
processes hand a field (velocity / density / traction) to each other in memory
with a machine-checkable contract:

* :class:`FieldDescriptor` -- self-describing frame header (name, shape, dtype,
  byte order, order, unit text, nbytes, sha256 of the payload),
* :class:`UnitSpec` / :func:`parse_unit` -- UDUNITS-2/CF unit strings resolved to
  exponents over the seven SI base dimensions, so ``Pa`` and ``N/m^2`` are
  compatible while ``m/s`` and ``Pa`` are rejected,
* :class:`ExchangeContract` -- the acceptance rule on the copy ledger
  (zero-copy / single-copy) and on bit-exactness,
* :class:`Expectation` -- what the receiver declares it will accept.

Bit-exactness is enforced structurally: the descriptor carries the sha256 of the
payload and every receive re-hashes the received bytes, so a single flipped bit
(including a NaN payload or a ``-0.0`` sign bit) raises
:class:`PayloadMismatchError` instead of passing an ``allclose`` tolerance.

External anchors
----------------
* IEEE 754-2019 -- bit patterns of ``NaN``/``-0.0`` are payload-relevant; NumPy
  dtype/byte-order semantics and PEP 3118 (buffer protocol) for zero-copy views.
* UDUNITS-2 unit grammar (Unidata) and CF conventions section 3.1 for the unit
  strings.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "FieldExchangeError", "FieldTypeError", "UnitMismatchError",
    "UnknownUnitError", "PayloadMismatchError", "CopyBudgetExceeded",
    "ChannelStateError", "FrameCorruptError",
    "UnitSpec", "parse_unit", "FieldDescriptor", "describe",
    "payload_memoryview", "CopyLedger", "ExchangeContract", "Expectation",
    "ReceivedField", "FieldStats", "FieldChannel",
]


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #
class FieldExchangeError(RuntimeError):
    """Base class for every contract violation raised by this module."""


class FieldTypeError(FieldExchangeError):
    """dtype / shape / byte order / contiguity assertion failed."""


class UnitMismatchError(FieldExchangeError):
    """Two unit strings are dimensionally incompatible."""


class UnknownUnitError(FieldExchangeError):
    """A unit string could not be resolved to SI base dimensions."""


class PayloadMismatchError(FieldExchangeError):
    """Payload bytes are not bit-identical to what the descriptor declares."""


class CopyBudgetExceeded(FieldExchangeError):
    """The transfer performed more copies than the contract allows."""


class ChannelStateError(FieldExchangeError):
    """Channel used out of order (closed, full, wrong peer, ...)."""


class FrameCorruptError(FieldExchangeError):
    """Frame header / magic / version is malformed or truncated."""


# --------------------------------------------------------------------------- #
# units: UDUNITS-2 / CF subset -> SI base-dimension exponents
# --------------------------------------------------------------------------- #
_DIM_KEYS = ("L", "M", "T", "I", "Theta", "Nmol", "Cd")

_UNIT_TABLE: Dict[str, Dict[str, int]] = {
    "1": {},
    "": {},
    "rad": {},
    "deg": {},
    "m": {"L": 1},
    "s": {"T": 1},
    "kg": {"M": 1},
    "A": {"I": 1},
    "K": {"Theta": 1},
    "mol": {"Nmol": 1},
    "cd": {"Cd": 1},
    "N": {"M": 1, "L": 1, "T": -2},
    "Pa": {"M": 1, "L": -1, "T": -2},
    "J": {"M": 1, "L": 2, "T": -2},
    "W": {"M": 1, "L": 2, "T": -3},
    "Hz": {"T": -1},
}


def _combine(dst: Dict[str, int], src: Dict[str, int], sign: int) -> None:
    for dim, exp in src.items():
        total = dst.get(dim, 0) + sign * exp
        if total:
            dst[dim] = total
        else:
            dst.pop(dim, None)


def _tokenize(expr: str) -> List[str]:
    expr = expr.replace("**", "^").replace("\u00b7", "*")
    tokens: List[str] = []
    i, n = 0, len(expr)
    while i < n:
        ch = expr[i]
        if ch.isspace():
            i += 1
        elif ch in "*/()^-":
            tokens.append(ch)
            i += 1
        elif ch.isalnum() or ch in "_.":
            j = i
            while j < n and (expr[j].isalnum() or expr[j] in "_."):
                j += 1
            tokens.append(expr[i:j])
            i = j
        else:
            raise UnknownUnitError(f"unsupported character {ch!r} in unit {expr!r}")
    return tokens


def _looks_numeric(tok: str) -> bool:
    try:
        float(tok)
    except (TypeError, ValueError):
        return False
    return True


class _UnitParser:
    """Recursive descent: expr := factor (('*' | '/') factor)*."""

    def __init__(self, tokens: Sequence[str], source: str) -> None:
        self._tokens = list(tokens)
        self._i = 0
        self._src = source

    def _peek(self) -> Optional[str]:
        return self._tokens[self._i] if self._i < len(self._tokens) else None

    def _next(self) -> str:
        tok = self._peek()
        if tok is None:
            raise UnknownUnitError(f"truncated unit {self._src!r}")
        self._i += 1
        return tok

    def parse(self) -> Dict[str, int]:
        dims = self._expr()
        if self._peek() is not None:
            raise UnknownUnitError(
                f"unexpected token {self._peek()!r} in unit {self._src!r}")
        return dims

    def _expr(self) -> Dict[str, int]:
        dims = self._factor()
        while self._peek() in ("*", "/"):
            sign = 1 if self._next() == "*" else -1
            _combine(dims, self._factor(), sign)
        return dims

    def _factor(self) -> Dict[str, int]:
        tok = self._next()
        if tok == "(":
            dims = self._expr()
            if self._next() != ")":
                raise UnknownUnitError(f"unbalanced parenthesis in {self._src!r}")
        else:
            if tok in ("*", "/", "^", ")", "-"):
                raise UnknownUnitError(f"unexpected {tok!r} in unit {self._src!r}")
            if tok in _UNIT_TABLE:
                dims = dict(_UNIT_TABLE[tok])
            elif _looks_numeric(tok):
                if float(tok) != 1.0:
                    raise UnknownUnitError(
                        f"unit scaling factor {tok!r} not supported in {self._src!r} "
                        f"(use SI values plus an explicit scale)")
                dims = {}
            else:
                raise UnknownUnitError(
                    f"unknown unit token {tok!r} in {self._src!r}; known: "
                    f"{sorted(k for k in _UNIT_TABLE if k)}")
        if self._peek() == "^":
            self._next()
            sign = 1
            if self._peek() == "-":
                self._next()
                sign = -1
            raw = self._next()
            try:
                exp = sign * int(raw)
            except ValueError as exc:
                raise UnknownUnitError(
                    f"non-integer exponent {raw!r} in {self._src!r}") from exc
            dims = {k: v * exp for k, v in dims.items() if v * exp}
        return dims


def parse_unit(text: str) -> Dict[str, int]:
    """Resolve a unit string to ``{base_dim: exponent}`` (empty == dimensionless)."""
    if not isinstance(text, str):
        raise UnknownUnitError(
            f"unit must be a string ('' or '1' for dimensionless), got {type(text).__name__}")
    if not text.strip():
        return {}
    return _UnitParser(_tokenize(text), text).parse()


@dataclass(frozen=True)
class UnitSpec:
    """Dimensional signature of a unit string."""

    text: str
    dims: Tuple[Tuple[str, int], ...]

    @classmethod
    def of(cls, text: str) -> "UnitSpec":
        return cls(text=text, dims=tuple(sorted(parse_unit(text).items())))

    def dimension_vector(self) -> Dict[str, int]:
        return dict(self.dims)

    def is_dimensionless(self) -> bool:
        return not self.dims

    def compatible_with(self, other: "UnitSpec") -> bool:
        return self.dims == other.dims

    def assert_compatible(self, other: "UnitSpec", *, label: str = "unit") -> None:
        if not self.compatible_with(other):
            raise UnitMismatchError(
                f"{label} mismatch: {self.text!r} {self.dimension_vector()} vs "
                f"{other.text!r} {other.dimension_vector()}")


# --------------------------------------------------------------------------- #
# descriptor
# --------------------------------------------------------------------------- #
def canonical_dtype(dtype: Any) -> np.dtype:
    """Canonical little-endian dtype; big-endian payloads are rejected."""
    dt = np.dtype(dtype)
    if dt.byteorder == ">":
        raise FieldTypeError(
            f"big-endian dtype {dt.str} is not accepted (payloads are canonical "
            f"little-endian); convert with astype('{dt.newbyteorder('<').str}')")
    return dt.newbyteorder("<")


def payload_memoryview(arr: np.ndarray) -> memoryview:
    """Byte view of ``arr`` **without copying** (works for C- and F-contiguous)."""
    return memoryview(arr.ravel(order="K")).cast("B")


def nbytes_of(shape: Iterable[int], itemsize: int) -> int:
    return int(np.prod(tuple(shape), dtype=np.int64)) * int(itemsize)


@dataclass(frozen=True)
class FieldDescriptor:
    """Self-describing frame header for one exchanged field."""

    name: str
    shape: Tuple[int, ...]
    dtype: str
    unit: str
    order: str = "C"
    checksum: str = ""
    nbytes: int = 0

    @property
    def np_dtype(self) -> np.dtype:
        return np.dtype(self.dtype)

    @property
    def unit_spec(self) -> UnitSpec:
        return UnitSpec.of(self.unit)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "unit": self.unit,
            "order": self.order,
            "checksum": self.checksum,
            "nbytes": self.nbytes,
        }

    def to_json(self) -> bytes:
        return json.dumps(self.to_dict(), sort_keys=True).encode()

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "FieldDescriptor":
        try:
            return cls(
                name=str(obj["name"]),
                shape=tuple(int(x) for x in obj["shape"]),
                dtype=str(obj["dtype"]),
                unit=str(obj["unit"]),
                order=str(obj.get("order", "C")),
                checksum=str(obj.get("checksum", "")),
                nbytes=int(obj.get("nbytes", 0)),
            )
        except KeyError as exc:
            raise FrameCorruptError(f"descriptor missing field {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise FrameCorruptError(f"descriptor malformed: {exc}") from exc


def describe(array: Any, *, name: str, unit: str = "1",
             order: Optional[str] = None) -> FieldDescriptor:
    """Build and validate a :class:`FieldDescriptor` for ``array`` (no copy)."""
    arr = np.asarray(array)
    if arr.dtype.hasobject:
        raise FieldTypeError("object dtype is not bit-stable and cannot be exchanged")
    if not (arr.flags.c_contiguous or arr.flags.f_contiguous):
        raise FieldTypeError(
            "non-contiguous field would need a hidden copy; pass "
            "np.ascontiguousarray(arr) explicitly so the copy is visible")
    dt = canonical_dtype(arr.dtype)
    if order is None:
        order = "C" if arr.flags.c_contiguous else "F"
    order = order.upper()
    if order not in ("C", "F"):
        raise FieldTypeError(f"order must be 'C' or 'F', got {order!r}")
    if order == "C" and not arr.flags.c_contiguous:
        raise FieldTypeError("declared C order but array is not C-contiguous")
    if order == "F" and not arr.flags.f_contiguous:
        raise FieldTypeError("declared F order but array is not F-contiguous")
    payload = payload_memoryview(arr)
    return FieldDescriptor(
        name=name,
        shape=tuple(int(s) for s in arr.shape),
        dtype=dt.str,
        unit=unit,
        order=order,
        checksum=hashlib.sha256(payload).hexdigest(),
        nbytes=len(payload),
    )


def verify_payload(desc: FieldDescriptor, payload: memoryview) -> None:
    """Raise unless ``payload`` is bit-identical to what ``desc`` declares."""
    if len(payload) != desc.nbytes:
        raise PayloadMismatchError(
            f"payload length {len(payload)} != descriptor nbytes {desc.nbytes}")
    got = hashlib.sha256(payload).hexdigest()
    if got != desc.checksum:
        raise PayloadMismatchError(
            f"sha256 mismatch for field {desc.name!r}: descriptor "
            f"{desc.checksum[:16]}... vs payload {got[:16]}... (not bit-identical)")


# --------------------------------------------------------------------------- #
# copy ledger / contract / expectation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CopyLedger:
    producer_copies: int
    receiver_copies: int
    channel: str = ""

    @property
    def total(self) -> int:
        return self.producer_copies + self.receiver_copies

    def to_dict(self) -> Dict[str, Any]:
        return {
            "producer_copies": self.producer_copies,
            "receiver_copies": self.receiver_copies,
            "total": self.total,
            "channel": self.channel,
        }


@dataclass(frozen=True)
class ExchangeContract:
    """Acceptance rule applied to a transfer's copy ledger and bit-exactness."""

    max_total_copies: int = 1
    require_zero_receiver_copy: bool = True
    require_bit_exact: bool = True

    def check(self, ledger: CopyLedger, *, label: str = "transfer") -> CopyLedger:
        if self.require_zero_receiver_copy and ledger.receiver_copies != 0:
            raise CopyBudgetExceeded(
                f"{label}: contract requires zero receiver-side copies, ledger "
                f"{ledger.to_dict()}")
        if ledger.total > self.max_total_copies:
            raise CopyBudgetExceeded(
                f"{label}: contract allows <= {self.max_total_copies} copies, "
                f"ledger {ledger.to_dict()}")
        return ledger

    def check_bit_exact(self, reference: Any, received: Any,
                        *, label: str = "roundtrip") -> bool:
        a, b = np.asarray(reference), np.asarray(received)
        if a.shape != b.shape or a.dtype != b.dtype:
            raise PayloadMismatchError(
                f"{label}: shape/dtype differ {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}")
        if not self.require_bit_exact:
            return bool(np.array_equal(a, b, equal_nan=True))
        if bytes(payload_memoryview(a)) != bytes(payload_memoryview(b)):
            raise PayloadMismatchError(f"{label}: payload bytes differ (not bit-exact)")
        return True


@dataclass(frozen=True)
class Expectation:
    """Receiver-side declaration of what it is willing to accept."""

    name: Optional[str] = None
    shape: Optional[Tuple[int, ...]] = None
    dtype: Optional[str] = None
    unit: Optional[str] = None

    def check(self, desc: FieldDescriptor) -> None:
        if self.name is not None and desc.name != self.name:
            raise FieldTypeError(f"field name {desc.name!r} != expected {self.name!r}")
        if self.shape is not None and tuple(desc.shape) != tuple(self.shape):
            raise FieldTypeError(f"field shape {tuple(desc.shape)} != expected {tuple(self.shape)}")
        if self.dtype is not None:
            want = canonical_dtype(self.dtype)
            if desc.np_dtype != want:
                raise FieldTypeError(
                    f"field dtype {desc.np_dtype.str} != expected {want.str} "
                    f"(itemsize {desc.np_dtype.itemsize} vs {want.itemsize})")
        if self.unit is not None:
            UnitSpec.of(self.unit).assert_compatible(desc.unit_spec, label="unit")


@dataclass
class ReceivedField:
    """A field delivered by a channel together with its audit trail."""

    descriptor: FieldDescriptor
    view: np.ndarray
    ledger: CopyLedger
    seq: int
    channel: str = ""

    @property
    def array(self) -> np.ndarray:
        return self.view

    def to_dict(self) -> Dict[str, Any]:
        return {
            "descriptor": self.descriptor.to_dict(),
            "ledger": self.ledger.to_dict(),
            "seq": self.seq,
            "channel": self.channel,
        }


@dataclass
class FieldStats:
    """Channel-level accounting (cumulative, both directions of every frame)."""

    published: int = 0
    received: int = 0
    bytes: int = 0
    producer_copies: int = 0
    receiver_copies: int = 0

    @property
    def transfers(self) -> int:
        return self.published

    def to_dict(self) -> Dict[str, Any]:
        return {
            "published": self.published,
            "received": self.received,
            "transfers": self.published,
            "bytes": self.bytes,
            "producer_copies": self.producer_copies,
            "receiver_copies": self.receiver_copies,
        }


class FieldChannel:
    """Common protocol: ``publish`` a field, ``receive`` it as a view."""

    kind = "abstract"
    zero_copy_receive = False
    filesystem_backed = False

    def __init__(self, *, payload_capacity: int, slots: int = 2) -> None:
        if payload_capacity <= 0:
            raise ChannelStateError("payload_capacity must be > 0")
        if slots < 2:
            raise ChannelStateError("slots must be >= 2 (single-producer ring)")
        self.payload_capacity = int(payload_capacity)
        self.slots = int(slots)
        self.stats = FieldStats()
        self._seq = 0
        self._closed = False
        self._producer_copies_by_seq: Dict[int, int] = {}
        self._last_published = 0

    # -- lifecycle -------------------------------------------------------- #
    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "FieldChannel":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- helpers ---------------------------------------------------------- #
    def _require_open(self) -> None:
        if self._closed:
            raise ChannelStateError(f"{self.kind} channel is closed")

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    @property
    def last_published_seq(self) -> int:
        return int(self._last_published)

    def _note_publish(self, seq: int, copies: int, nbytes: int) -> None:
        self._last_published = int(seq)
        self._producer_copies_by_seq[int(seq)] = int(copies)
        self.stats.published += 1
        self.stats.bytes += int(nbytes)
        self.stats.producer_copies += int(copies)

    def _note_receive(self, seq: int, receiver_copies: int, nbytes: int,
                      producer_copies: Optional[int] = None) -> CopyLedger:
        if producer_copies is None:
            producer_copies = self._producer_copies_by_seq.pop(int(seq), 0)
        else:
            self._producer_copies_by_seq.pop(int(seq), None)
        self.stats.received += 1
        self.stats.receiver_copies += int(receiver_copies)
        return CopyLedger(producer_copies=producer_copies,
                          receiver_copies=int(receiver_copies), channel=self.kind)

    # -- protocol --------------------------------------------------------- #
    def publish(self, array: Any, *, name: str = "field", unit: str = "1",
                order: Optional[str] = None,
                contract: Optional[ExchangeContract] = None) -> FieldDescriptor:
        raise NotImplementedError

    def receive(self, *, expect: Optional[Expectation] = None,
                contract: Optional[ExchangeContract] = None) -> ReceivedField:
        raise NotImplementedError