"""In-memory transports for the cross-solver field exchange (no file hand-off).

Every channel implements :class:`~tensorphysics.field_exchange.transport.FieldChannel`:
``publish`` a ``(descriptor, payload)`` frame and ``receive`` it on the peer side,
with an audited copy ledger and sha256-enforced bit-exactness.

Channels
--------
``AnonymousShmChannel``  (recommended, zero copy)
    ``mmap(MAP_SHARED | MAP_ANONYMOUS)`` arena created before ``fork`` and
    inherited by the child: pure kernel memory, **no path**, **no /dev/shm
    entry**, **no read()/write()** on any file.  Payload stays in the arena;
    both sides get ``np.ndarray`` views.  ``allocate()``/``publish_buffer()`` is
    the true zero-copy path (the producer writes the solver output straight into
    the shared slot); ``publish(array)`` costs exactly one producer-side copy
    (marshal) and zero receiver-side copies.
``PosixShmChannel``      (zero copy, unrelated processes)
    ``multiprocessing.shared_memory`` -- a kernel tmpfs object with a name, used
    only for peers that cannot inherit an anonymous mapping.  Never written
    through file APIs; unlinked in ``close()``.  :attr:`filesystem_backed` stays
    ``False`` but :attr:`kernel_object_name` is exposed for audit.
``SocketChannel``        (one copy per side)
    ``AF_UNIX`` ``SOCK_STREAM`` ``socketpair``: one copy into the kernel socket
    buffer, one ``recv_into`` into the receiver's numpy buffer.  Fallback for
    peers with no shared address space.
``LocalChannel``         (zero copy, same process)
    Module-to-module hand-off inside one interpreter: descriptor + reference.

Synchronisation
---------------
The shared-memory channels use ``multiprocessing.Value('q', lock=True)``
sequence counters (inherited across ``fork``) so the publish store is visible
before the receiver polls it.  ``wait_for`` spins with ``time.sleep(0)`` and a
timeout; it never touches the filesystem.
"""
from __future__ import annotations

import json
import mmap
import socket
import struct
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .transport import (ChannelStateError, CopyLedger, ExchangeContract,
                        Expectation, FieldChannel, FieldDescriptor,
                        FieldTypeError, FrameCorruptError, ReceivedField,
                        canonical_dtype, describe, nbytes_of, payload_memoryview,
                        verify_payload)

__all__ = [
    "FieldBuffer", "AnonymousShmChannel", "PosixShmChannel", "SocketChannel",
    "LocalChannel", "make_channel", "CHANNEL_KINDS",
]

_GLOBAL_BYTES = 64
_SLOT_HDR_BYTES = 1024
_SHM_MAGIC = b"TPFXSHM1"
_VERSION = 1
_HDR_GLOBAL = struct.Struct("<8sIIQII")   # magic, version, slots, capacity, hdr, stride
_SLOT_META = struct.Struct("<qII")        # seq, hdr_len, producer_copies

_SOCK_MAGIC = b"TPFXSCK1"
_SOCK_HDR = struct.Struct("<8sIIQI")      # magic, version, hdr_len, payload_len, producer_copies


@dataclass
class FieldBuffer:
    """A field slot *inside* the shared arena (write here for zero-copy publish)."""

    channel: FieldChannel
    array: np.ndarray
    name: str
    unit: str
    seq: int
    index: int
    published: bool = False


class _ShmChannelBase(FieldChannel):
    """Slot arena + descriptor/payload frames; the payload never leaves the arena."""

    kind = "shm"
    zero_copy_receive = True
    filesystem_backed = False

    def __init__(self, *, payload_capacity: int, slots: int = 2) -> None:
        super().__init__(payload_capacity=payload_capacity, slots=slots)
        self._stride = _SLOT_META.size + _SLOT_HDR_BYTES + self.payload_capacity
        self._size = _GLOBAL_BYTES + self.slots * self._stride
        # ``fork`` inherits these shared counters; they give the publish/consume
        # ordering (release/acquire) that a bare int64 store in the arena lacks.
        from multiprocessing import Value

        self._write_seq = Value("q", 0, lock=True)
        self._last_read = 0
        self._buf: Any = None
        self._mv: Optional[memoryview] = None
        self._views_issued = 0
        self._open_arena()
        self._init_header()

    # -- subclass hooks ---------------------------------------------------- #
    def _open_arena(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def _release_arena(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def _init_header(self) -> None:
        self._buf[0:_GLOBAL_BYTES] = _HDR_GLOBAL.pack(
            _SHM_MAGIC, _VERSION, self.slots, self.payload_capacity,
            _SLOT_HDR_BYTES, self._stride).ljust(_GLOBAL_BYTES, b"\0")
        self._mv = memoryview(self._buf)

    def arena_bytes(self) -> np.ndarray:
        """Writable byte view of the whole arena (used to prove sharing)."""
        if self._mv is None:
            raise ChannelStateError("arena is closed")
        return np.frombuffer(self._mv, dtype=np.uint8)

    # -- addressing -------------------------------------------------------- #
    def _slot_base(self, index: int) -> int:
        return _GLOBAL_BYTES + (index % self.slots) * self._stride

    def _payload_offset(self, index: int) -> int:
        return self._slot_base(index) + _SLOT_META.size + _SLOT_HDR_BYTES

    def _arena_bounds(self) -> Tuple[int, int]:
        import ctypes

        start = ctypes.addressof(ctypes.c_char.from_buffer(self._buf))
        return start, start + self._size

    def _owns(self, array: np.ndarray) -> bool:
        """True when ``array`` already lives in this arena (=> no marshal copy)."""
        try:
            base = int(array.__array_interface__["data"][0])
        except (AttributeError, KeyError, TypeError):
            return False
        lo, hi = self._arena_bounds()
        return lo <= base < hi

    # -- arena path: true zero copy ---------------------------------------- #
    def allocate(self, *, shape: Sequence[int], dtype: Any, name: str,
                 unit: str = "1") -> FieldBuffer:
        self._require_open()
        dt = canonical_dtype(dtype)
        shape = tuple(int(s) for s in shape)
        nbytes = nbytes_of(shape, dt.itemsize)
        if nbytes > self.payload_capacity:
            raise ChannelStateError(
                f"field needs {nbytes} B > slot capacity {self.payload_capacity} B")
        seq = self._next_seq()
        index = seq % self.slots
        if self._mv is None:
            raise ChannelStateError("arena is closed")
        view = np.ndarray(shape, dtype=dt, buffer=self._mv,
                          offset=self._payload_offset(index), order="C")
        self._views_issued += 1
        return FieldBuffer(channel=self, array=view, name=name, unit=unit,
                           seq=seq, index=index)

    def allocate_like(self, array: Any, *, name: str, unit: str = "1") -> FieldBuffer:
        arr = np.asarray(array)
        return self.allocate(shape=arr.shape, dtype=arr.dtype, name=name, unit=unit)

    def publish_buffer(self, buf: FieldBuffer, *,
                       contract: Optional[ExchangeContract] = None) -> FieldDescriptor:
        self._require_open()
        if buf.channel is not self:
            raise ChannelStateError("FieldBuffer belongs to a different channel")
        if buf.published:
            raise ChannelStateError(
                "FieldBuffer is single-use: allocate a new buffer for the next frame")
        arr = buf.array
        desc = describe(arr, name=buf.name, unit=buf.unit, order="C")
        self._write_frame(buf.seq, desc, payload=None, producer_copies=0)
        buf.published = True
        ledger = CopyLedger(producer_copies=0, receiver_copies=0, channel=self.kind)
        self._note_publish(buf.seq, 0, desc.nbytes)
        if contract is not None:
            contract.check(ledger, label=f"{self.kind}:publish_buffer")
        with self._write_seq.get_lock():
            self._write_seq.value = buf.seq
        return desc

    # -- marshal path: one producer copy ----------------------------------- #
    def publish(self, array: Any, *, name: str = "field", unit: str = "1",
                order: Optional[str] = None,
                contract: Optional[ExchangeContract] = None) -> FieldDescriptor:
        self._require_open()
        arr = np.asarray(array)
        desc = describe(arr, name=name, unit=unit, order=order)
        if desc.nbytes > self.payload_capacity:
            raise ChannelStateError(
                f"field {desc.nbytes} B > slot capacity {self.payload_capacity} B")
        seq = self._next_seq()
        in_arena = self._owns(arr)
        producer_copies = 0 if in_arena else 1
        self._write_frame(seq, desc,
                          payload=None if in_arena else payload_memoryview(arr),
                          producer_copies=producer_copies)
        ledger = CopyLedger(producer_copies=0 if in_arena else 1, receiver_copies=0,
                            channel=self.kind)
        self._note_publish(seq, ledger.producer_copies, desc.nbytes)
        if contract is not None:
            contract.check(ledger, label=f"{self.kind}:publish")
        with self._write_seq.get_lock():
            self._write_seq.value = seq
        return desc

    def _write_frame(self, seq: int, desc: FieldDescriptor,
                     payload: Optional[memoryview], producer_copies: int = 0) -> None:
        if self._mv is None:
            raise ChannelStateError("arena is closed")
        index = seq % self.slots
        base = self._slot_base(index)
        hdr = desc.to_json()
        if len(hdr) > _SLOT_HDR_BYTES:
            raise ChannelStateError(
                f"descriptor {len(hdr)} B > {_SLOT_HDR_BYTES} B header slot")
        head = _SLOT_META.pack(seq, len(hdr), int(producer_copies))
        self._mv[base:base + _SLOT_META.size] = head
        self._mv[base + _SLOT_META.size:base + _SLOT_META.size + len(hdr)] = hdr
        if payload is not None:
            poff = self._payload_offset(index)
            self._mv[poff:poff + len(payload)] = payload

    def payload_view(self, seq: int, *, shape: Sequence[int], dtype: Any) -> np.ndarray:
        """Zero-copy view on slot ``seq`` (peer-side inspection / write-back)."""
        if self._mv is None:
            raise ChannelStateError("arena is closed")
        dt = canonical_dtype(dtype)
        shape = tuple(int(s) for s in shape)
        nbytes = nbytes_of(shape, dt.itemsize)
        if nbytes > self.payload_capacity:
            raise ChannelStateError("requested view exceeds slot capacity")
        off = self._payload_offset(int(seq))
        self._views_issued += 1
        return np.ndarray(shape, dtype=dt, buffer=self._mv, offset=off, order="C")

    # -- receive ------------------------------------------------------------ #
    def _write_seq_value(self) -> int:
        with self._write_seq.get_lock():
            return int(self._write_seq.value)

    def wait_for(self, seq: int, *, timeout: float = 30.0) -> int:
        """Block until the producer published ``seq`` (spin, no filesystem use)."""
        deadline = time.monotonic() + timeout
        while True:
            cur = self._write_seq_value()
            if cur >= seq:
                return cur
            if time.monotonic() > deadline:
                raise ChannelStateError(
                    f"{self.kind}: timed out after {timeout}s waiting for seq {seq} "
                    f"(producer at {cur})")
            time.sleep(0.0005)

    def receive(self, *, expect: Optional[Expectation] = None,
                contract: Optional[ExchangeContract] = None,
                seq: Optional[int] = None, timeout: float = 30.0) -> ReceivedField:
        self._require_open()
        target = self._last_read + 1 if seq is None else int(seq)
        self.wait_for(target, timeout=timeout)
        if self._mv is None:
            raise ChannelStateError("arena is closed")
        index = target % self.slots
        base = self._slot_base(index)
        got_seq, hdr_len, frame_producer_copies = _SLOT_META.unpack_from(self._mv, base)
        if got_seq != target:
            raise FrameCorruptError(
                f"{self.kind}: slot {index} holds seq {got_seq}, expected {target} "
                f"(slow consumer overwritten by producer; use more slots)")
        if hdr_len == 0 or hdr_len > _SLOT_HDR_BYTES:
            raise FrameCorruptError(f"{self.kind}: bad descriptor length {hdr_len}")
        raw = bytes(self._mv[base + _SLOT_META.size:base + _SLOT_META.size + hdr_len])
        try:
            desc = FieldDescriptor.from_dict(json.loads(raw))
        except json.JSONDecodeError as exc:
            raise FrameCorruptError(f"{self.kind}: descriptor is not JSON: {exc}") from exc
        if expect is not None:
            expect.check(desc)
        poff = self._payload_offset(index)
        payload = self._mv[poff:poff + desc.nbytes]        # memoryview: no copy
        verify_payload(desc, payload)
        view = np.ndarray(desc.shape, dtype=desc.np_dtype, buffer=payload,
                          order=desc.order)
        self._views_issued += 1
        ledger = self._note_receive(target, 0, desc.nbytes,
                                    producer_copies=frame_producer_copies)
        self._last_read = target
        if contract is not None:
            contract.check(ledger, label=f"{self.kind}:receive")
        return ReceivedField(descriptor=desc, view=view, ledger=ledger, seq=target,
                             channel=self.kind)

    # -- lifecycle ---------------------------------------------------------- #
    def close(self) -> None:
        if self._closed:
            return
        super().close()
        try:
            self._release_arena()
        except BufferError:  # live views still exported - keep the mapping alive
            pass


class AnonymousShmChannel(_ShmChannelBase):
    """Zero-copy channel over ``mmap(MAP_SHARED|MAP_ANONYMOUS)`` (fork peers).

    No filesystem object is created: the segment has no name, is not visible in
    ``/dev/shm``, and is released when both processes exit.
    """

    kind = "anon-shm"

    def _open_arena(self) -> None:
        self._buf = mmap.mmap(-1, self._size, flags=mmap.MAP_SHARED)

    def _release_arena(self) -> None:
        arena, self._buf = self._buf, None
        if self._mv is not None:
            try:
                self._mv.release()
            except BufferError:
                pass
            self._mv = None
        if arena is not None and self._views_issued == 0:
            try:
                arena.close()
            except BufferError:  # pragma: no cover - defensive
                pass
        # else: numpy views still reference the mapping; the mmap object stays
        # alive through them and is unmapped by the GC once they die.


class PosixShmChannel(_ShmChannelBase):
    """Zero-copy channel over POSIX shared memory (unrelated processes).

    The segment is a kernel ``tmpfs`` object with a name: no user-space file is
    opened, written or read, and it is unlinked in :meth:`close`.  Use it only
    when the peer cannot inherit an :class:`AnonymousShmChannel` mapping.
    """

    kind = "posix-shm"

    def _open_arena(self) -> None:
        from multiprocessing import shared_memory

        self._shm = shared_memory.SharedMemory(create=True, size=self._size)
        self.kernel_object_name = self._shm.name  # audit only; never opened as a file
        self._buf = self._shm.buf

    def arena_bytes(self) -> np.ndarray:
        return np.frombuffer(self._buf, dtype=np.uint8)

    def _release_arena(self) -> None:
        if self._mv is not None:
            try:
                self._mv.release()
            except BufferError:
                pass
            self._mv = None
        try:
            self._shm.unlink()   # drop the name; memory lives while views hold it
            if self._views_issued == 0:
                self._shm.close()
        except (FileNotFoundError, BufferError):
            pass


class SocketChannel(FieldChannel):
    """One-copy transport over an ``AF_UNIX`` ``SOCK_STREAM`` ``socketpair``."""

    kind = "socket"
    zero_copy_receive = False
    filesystem_backed = False
    max_frame_nbytes_hint = 1 << 30

    def __init__(self, *, payload_capacity: int, slots: int = 2,
                 sock: Optional[socket.socket] = None,
                 peer: Optional[socket.socket] = None) -> None:
        super().__init__(payload_capacity=payload_capacity, slots=slots)
        if sock is None:
            sock, peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock = sock      # send endpoint
        self.peer = peer      # receive endpoint
        self._last_read = 0
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)

    def publish(self, array: Any, *, name: str = "field", unit: str = "1",
                order: Optional[str] = None,
                contract: Optional[ExchangeContract] = None) -> FieldDescriptor:
        self._require_open()
        arr = np.asarray(array)
        desc = describe(arr, name=name, unit=unit, order=order)
        if desc.nbytes > self.payload_capacity:
            raise ChannelStateError(
                f"field {desc.nbytes} B > declared capacity {self.payload_capacity} B")
        hdr = desc.to_json()
        self.sock.sendall(_SOCK_HDR.pack(_SOCK_MAGIC, _VERSION, len(hdr),
                                         desc.nbytes, 1))
        self.sock.sendall(hdr)
        self.sock.sendall(payload_memoryview(arr))
        ledger = CopyLedger(producer_copies=1, receiver_copies=1, channel=self.kind)
        self._note_publish(self._seq, 1, desc.nbytes)
        if contract is not None:
            contract.check(ledger, label=f"{self.kind}:publish")
        return desc

    def _recv_exact(self, n: int) -> bytearray:
        buf = bytearray(n)
        view = memoryview(buf)
        got = 0
        while got < n:
            k = self.peer.recv_into(view[got:], n - got)
            if k == 0:
                raise ChannelStateError(f"{self.kind}: peer closed during frame")
            got += k
        return buf

    def receive(self, *, expect: Optional[Expectation] = None,
                contract: Optional[ExchangeContract] = None,
                seq: Optional[int] = None, timeout: float = 30.0) -> ReceivedField:
        self._require_open()
        self.peer.settimeout(timeout)
        head = self._recv_exact(_SOCK_HDR.size)
        magic, version, hdr_len, payload_len, frame_producer_copies = _SOCK_HDR.unpack(head)
        if magic != _SOCK_MAGIC:
            raise FrameCorruptError(f"{self.kind}: bad magic {magic!r}")
        if version != _VERSION:
            raise FrameCorruptError(f"{self.kind}: unsupported version {version}")
        raw = bytes(self._recv_exact(hdr_len))
        try:
            desc = FieldDescriptor.from_dict(json.loads(raw))
        except json.JSONDecodeError as exc:
            raise FrameCorruptError(f"{self.kind}: descriptor is not JSON: {exc}") from exc
        if expect is not None:
            expect.check(desc)
        if payload_len != desc.nbytes:
            raise FrameCorruptError(
                f"{self.kind}: frame payload {payload_len} != descriptor {desc.nbytes}")
        # exactly one receiver-side copy: kernel buffer -> numpy buffer
        out = np.empty(desc.shape, dtype=desc.np_dtype, order=desc.order)
        mv = payload_memoryview(out)
        got = 0
        while got < desc.nbytes:
            k = self.peer.recv_into(mv[got:], desc.nbytes - got)
            if k == 0:
                raise ChannelStateError(f"{self.kind}: peer closed mid-payload")
            got += k
        verify_payload(desc, mv)
        self._last_read += 1
        ledger = self._note_receive(self._last_read, 1, desc.nbytes,
                                    producer_copies=frame_producer_copies)
        if contract is not None:
            contract.check(ledger, label=f"{self.kind}:receive")
        return ReceivedField(descriptor=desc, view=out, ledger=ledger,
                             seq=self._last_read, channel=self.kind)

    def close(self) -> None:
        if self._closed:
            return
        super().close()
        for s in (self.sock, self.peer):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass


class LocalChannel(FieldChannel):
    """Zero-copy hand-off between two modules of the *same* interpreter."""

    kind = "local"
    zero_copy_receive = True
    filesystem_backed = False

    def __init__(self) -> None:  # noqa: D401 - trivial
        super().__init__(payload_capacity=1 << 62, slots=2)
        self._pending: Dict[int, Tuple[FieldDescriptor, np.ndarray]] = {}

    def publish(self, array: Any, *, name: str = "field", unit: str = "1",
                order: Optional[str] = None,
                contract: Optional[ExchangeContract] = None) -> FieldDescriptor:
        self._require_open()
        arr = np.asarray(array)
        desc = describe(arr, name=name, unit=unit, order=order)
        seq = self._next_seq()
        self._pending[seq] = (desc, arr)
        ledger = CopyLedger(producer_copies=0, receiver_copies=0, channel=self.kind)
        self._note_publish(seq, 0, desc.nbytes)
        if contract is not None:
            contract.check(ledger, label="local:publish")
        return desc

    def receive(self, *, expect: Optional[Expectation] = None,
                contract: Optional[ExchangeContract] = None,
                seq: Optional[int] = None, timeout: float = 30.0) -> ReceivedField:
        self._require_open()
        if not self._pending:
            raise ChannelStateError("local: no pending field")
        target = min(self._pending) if seq is None else int(seq)
        if target not in self._pending:
            raise ChannelStateError(f"local: seq {target} was never published")
        desc, arr = self._pending.pop(target)
        if expect is not None:
            expect.check(desc)
        verify_payload(desc, payload_memoryview(arr))
        if arr.dtype != desc.np_dtype or tuple(arr.shape) != tuple(desc.shape):
            raise FieldTypeError("local: descriptor does not match the array")
        ledger = self._note_receive(target, 0, desc.nbytes)
        if contract is not None:
            contract.check(ledger, label="local:receive")
        return ReceivedField(descriptor=desc, view=arr, ledger=ledger, seq=target,
                             channel=self.kind)


CHANNEL_KINDS = {
    "anon-shm": AnonymousShmChannel,
    "posix-shm": PosixShmChannel,
    "socket": SocketChannel,
    "local": LocalChannel,
}


def make_channel(kind: str, *, payload_capacity: int, slots: int = 2) -> FieldChannel:
    """Factory: ``kind in {'anon-shm', 'posix-shm', 'socket', 'local'}``."""
    if kind == "local":
        return LocalChannel()
    try:
        cls = CHANNEL_KINDS[kind]
    except KeyError as exc:
        raise ChannelStateError(
            f"unknown channel {kind!r}; known: {sorted(CHANNEL_KINDS)}") from exc
    return cls(payload_capacity=payload_capacity, slots=slots)