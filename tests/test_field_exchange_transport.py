"""Acceptance tests for the in-memory cross-solver field exchange contract.

GAP-CROSS-FIELD: two real solver processes/modules exchange one field through an
in-memory interface; the round trip must be **bit-exact**, the copy semantics
must be **zero-copy or single-copy** with an audited ledger, type and unit
assertions must fire on mismatch, and **no file** may intermediate the payload.

Run: ``python -m pytest tests/test_field_exchange_transport.py -q``
"""
from __future__ import annotations

import builtins
import io
import multiprocessing as mp
import os
import tempfile

import numpy as np
import pytest

from tensorphysics.field_exchange.channels import (
    AnonymousShmChannel, LocalChannel, PosixShmChannel, SocketChannel, make_channel)
from tensorphysics.field_exchange.lbm import run_channel, wall_shear_from_profile
from tensorphysics.field_exchange.transport import (
    CopyBudgetExceeded, ExchangeContract, Expectation, FieldDescriptor,
    FieldTypeError, PayloadMismatchError, UnitMismatchError, UnitSpec,
    UnknownUnitError, ChannelStateError, describe, parse_unit)

FORK = mp.get_context("fork")


def adversarial_payload() -> np.ndarray:
    """f64 payload with the bit patterns that ``allclose`` cannot police."""
    bits = np.array([
        0x0000000000000000,  # +0
        0x8000000000000000,  # -0
        0x7FF8000000000001,  # NaN with a non-default payload
        0xFFF800000000ABCD,  # negative NaN, different payload
        0x7FF0000000000000,  # +inf
        0xFFF0000000000000,  # -inf
        0x0000000000000001,  # smallest subnormal
        0x3FF0000000000000,  # 1.0
    ], dtype=np.uint64)
    return bits.view(np.float64)


# --------------------------------------------------------------------------- #
# units
# --------------------------------------------------------------------------- #
def test_unit_parsing_resolves_si_dimensions():
    assert parse_unit("m/s") == {"L": 1, "T": -1}
    assert parse_unit("m*s^-1") == {"L": 1, "T": -1}
    assert parse_unit("m^2/s") == {"L": 2, "T": -1}
    assert parse_unit("kg/(m*s^2)") == {"M": 1, "L": -1, "T": -2}
    assert parse_unit("Pa") == parse_unit("N/m^2")
    assert parse_unit("1") == {} and parse_unit("") == {}


def test_unit_compatibility_and_mismatch():
    u = UnitSpec.of("m/s")
    u.assert_compatible(UnitSpec.of("m*s^-1"))
    assert UnitSpec.of("Pa").compatible_with(UnitSpec.of("N/m^2"))
    assert not UnitSpec.of("Pa").compatible_with(UnitSpec.of("J"))
    with pytest.raises(UnitMismatchError, match="unit mismatch"):
        UnitSpec.of("m/s").assert_compatible(UnitSpec.of("Pa"), label="unit")
    assert UnitSpec.of("1").is_dimensionless()


def test_unknown_and_scaled_units_are_rejected():
    with pytest.raises(UnknownUnitError, match="unknown unit token"):
        parse_unit("furlong/fortnight")
    with pytest.raises(UnknownUnitError, match="scaling factor"):
        parse_unit("2*m")
    with pytest.raises(UnknownUnitError, match="unbalanced|truncated"):
        parse_unit("kg/(m*s")


# --------------------------------------------------------------------------- #
# type assertions
# --------------------------------------------------------------------------- #
def test_descriptor_rejects_non_contiguous_and_big_endian():
    base = np.arange(24, dtype=np.float64).reshape(4, 6)
    with pytest.raises(FieldTypeError, match="non-contiguous"):
        describe(base[:, ::2], name="u", unit="m/s")
    with pytest.raises(FieldTypeError, match="big-endian"):
        describe(base.astype(">f8"), name="u", unit="m/s")
    with pytest.raises(FieldTypeError, match="object dtype"):
        describe(np.array([object()], dtype=object), name="u", unit="1")
    # F-order is allowed when declared
    fortran = np.asfortranarray(base)
    desc = describe(fortran, name="u", unit="m/s")
    assert desc.order == "F"


def test_expectation_enforces_type_shape_name_and_unit():
    arr = np.arange(12, dtype=np.float64).reshape(3, 4)
    ch = LocalChannel()
    ch.publish(arr, name="ux", unit="m/s")
    got = ch.receive(expect=Expectation(name="ux", shape=(3, 4), dtype="f8", unit="m/s"))
    assert got.view.dtype == np.float64 and got.view.shape == (3, 4)

    ch.publish(arr, name="ux", unit="m/s")
    with pytest.raises(FieldTypeError, match="dtype"):
        ch.receive(expect=Expectation(dtype="f4"))
    ch.publish(arr, name="ux", unit="m/s")
    with pytest.raises(FieldTypeError, match="shape"):
        ch.receive(expect=Expectation(shape=(4, 3)))
    ch.publish(arr, name="ux", unit="m/s")
    with pytest.raises(FieldTypeError, match="name"):
        ch.receive(expect=Expectation(name="rho"))
    ch.publish(arr, name="ux", unit="m/s")
    with pytest.raises(UnitMismatchError):
        ch.receive(expect=Expectation(unit="Pa"))


# --------------------------------------------------------------------------- #
# bit-exact round trips + copy semantics
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["local", "socket", "anon-shm", "posix-shm"])
def test_bit_exact_roundtrip_every_channel(kind):
    arr = adversarial_payload()
    ch = make_channel(kind, payload_capacity=1 << 12)
    try:
        desc = ch.publish(arr, name="u", unit="m/s")
        got = ch.receive(expect=Expectation(name="u", shape=arr.shape,
                                           dtype="f8", unit="m/s"))
        assert got.view.tobytes() == arr.tobytes(), "round trip is not bit-exact"
        assert desc.checksum == got.descriptor.checksum
        assert np.isnan(got.view[2]) and np.signbit(got.view[1])
    finally:
        ch.close()


@pytest.mark.parametrize("kind", ["local", "anon-shm", "posix-shm"])
def test_zero_copy_semantics(kind):
    arr = adversarial_payload()
    ch = make_channel(kind, payload_capacity=1 << 12)
    contract = ExchangeContract(max_total_copies=0, require_zero_receiver_copy=True)
    try:
        if kind == "local":
            ch.publish(arr, name="u", unit="m/s", contract=contract)
        else:
            # arena path: the producer writes straight into the shared slot
            buf = ch.allocate(shape=arr.shape, dtype=arr.dtype, name="u", unit="m/s")
            buf.array[:] = arr
            ch.publish_buffer(buf, contract=contract)
        got = ch.receive(expect=Expectation(name="u", unit="m/s"), contract=contract)
        assert got.ledger.receiver_copies == 0
        assert got.ledger.total == 0
        if kind == "local":
            assert np.shares_memory(got.view, arr)
        else:
            assert np.shares_memory(got.view, ch.arena_bytes())
        assert got.view.tobytes() == arr.tobytes()
    finally:
        ch.close()


def test_socket_single_copy_semantics():
    arr = adversarial_payload()
    ch = SocketChannel(payload_capacity=1 << 12)
    try:
        ch.publish(arr, name="u", unit="m/s",
                   contract=ExchangeContract(max_total_copies=2,
                                             require_zero_receiver_copy=False))
        got = ch.receive(expect=Expectation(unit="m/s"),
                         contract=ExchangeContract(max_total_copies=2,
                                                   require_zero_receiver_copy=False))
        assert (got.ledger.producer_copies, got.ledger.receiver_copies) == (1, 1)
        assert not np.shares_memory(got.view, arr)
        # a zero-copy contract must reject the socket transport
        ch.publish(arr, name="u", unit="m/s")
        with pytest.raises(CopyBudgetExceeded):
            ch.receive(contract=ExchangeContract(max_total_copies=1,
                                                require_zero_receiver_copy=True))
    finally:
        ch.close()


def test_arena_path_is_true_zero_copy_and_marshal_path_is_single_copy():
    ch = AnonymousShmChannel(payload_capacity=1 << 12)
    try:
        buf = ch.allocate(shape=(4, 3), dtype=np.float64, name="ux", unit="m/s")
        buf.array[:] = np.arange(12).reshape(4, 3)
        desc = ch.publish_buffer(buf, contract=ExchangeContract(max_total_copies=0))
        got = ch.receive(expect=Expectation(name="ux", shape=(4, 3), dtype="f8",
                                           unit="m/s"))
        assert desc.checksum == got.descriptor.checksum
        assert (got.ledger.producer_copies, got.ledger.receiver_copies) == (0, 0)
        assert np.shares_memory(got.view, buf.array)

        external = np.linspace(0.0, 1.0, 12).reshape(4, 3)
        ch.publish(external, name="ux", unit="m/s",
                   contract=ExchangeContract(max_total_copies=1))
        got2 = ch.receive(expect=Expectation(unit="m/s"))
        assert got2.ledger.to_dict() == {"producer_copies": 1, "receiver_copies": 0,
                                         "total": 1, "channel": "anon-shm"}
        assert got2.view.tobytes() == external.tobytes()
    finally:
        ch.close()


def test_corrupted_payload_is_detected():
    ch = AnonymousShmChannel(payload_capacity=1 << 10)
    try:
        arr = np.arange(6, dtype=np.float64)
        ch.publish(arr, name="u", unit="m/s")
        seq = ch.last_published_seq
        # flip one bit inside the slot payload after publication
        off = ch._payload_offset(seq % ch.slots)
        view = np.frombuffer(ch._mv, dtype=np.uint8)
        view[off + 3] ^= 0x01
        with pytest.raises(PayloadMismatchError, match="sha256 mismatch"):
            ch.receive(seq=seq)
    finally:
        ch.close()


def test_capacity_and_state_errors():
    with pytest.raises(ChannelStateError, match="capacity"):
        AnonymousShmChannel(payload_capacity=64).publish(np.zeros(100), name="u")
    with pytest.raises(ChannelStateError, match="slots"):
        SocketChannel(payload_capacity=128, slots=1)
    ch = LocalChannel()
    with pytest.raises(ChannelStateError, match="no pending field"):
        ch.receive()
    ch.publish(np.zeros(2), name="u")
    ch.close()
    with pytest.raises(ChannelStateError, match="closed"):
        ch.publish(np.zeros(2), name="u")


# --------------------------------------------------------------------------- #
# cross-process: real solver field, zero copy, type/unit assertions in flight
# --------------------------------------------------------------------------- #
def _consumer(chan, ctrl, expect_dtype="f8", mutate=False):
    try:
        got = chan.receive(expect=Expectation(name="ux", dtype="f8", unit="m/s"),
                           timeout=60)
        ctrl["child"].put({
            "shape": got.view.shape,
            "dtype": got.view.dtype.str,
            "unit": got.descriptor.unit,
            "checksum": got.descriptor.checksum,
            "copies": got.ledger.to_dict(),
            "shares_arena": bool(np.shares_memory(got.view, chan.arena_bytes())),
            "mean": float(got.view.mean()),
            "wall_shear": wall_shear_from_profile(got.view.mean(axis=0), 0.1),
        })
        if mutate:
            got.view.flat[0] = 42.5  # write through the shared pages
    except BaseException as exc:  # noqa: BLE001 - reported to the parent
        ctrl["child"].put({"error": f"{type(exc).__name__}: {exc}"})


def test_cross_process_real_solver_field_roundtrip_zero_copy():
    """Producer (parent) runs the LBM channel; consumer (child) verifies + writes back."""
    result = run_channel(nx=4, ny=16, lid_velocity=0.1, max_steps=2000,
                         tol=1e-9, check_every=100)
    ux = np.ascontiguousarray(result.ux)          # (nx, ny) real solver output
    ch = AnonymousShmChannel(payload_capacity=ux.nbytes, slots=4)
    ctrl = {"child": FORK.Queue()}
    proc = FORK.Process(target=_consumer, args=(ch, ctrl, "f8", True))
    proc.start()
    try:
        ch.publish(ux, name="ux", unit="m/s", contract=ExchangeContract(max_total_copies=1))
        proc.join(120)
        assert proc.exitcode == 0
        report = ctrl["child"].get(timeout=5)
        # read the shared pages back through the public zero-copy view
        shared_back = ch.payload_view(ch.last_published_seq, shape=ux.shape,
                                      dtype=ux.dtype).copy()
    finally:
        ch.close()
    assert "error" not in report, report
    assert report["shares_arena"] is True, "receiver must get a view, not a copy"
    assert report["copies"] == {"producer_copies": 1, "receiver_copies": 0,
                                "total": 1, "channel": "anon-shm"}
    ref_wall = wall_shear_from_profile(ux.mean(axis=0), 0.1)
    assert report["wall_shear"] == ref_wall, "derived scalar must be bit-identical"
    assert report["mean"] == float(ux.mean())
    assert report["checksum"] == describe(ux, name="ux", unit="m/s").checksum
    # child wrote into the shared pages => the parent's arena shows that value
    assert shared_back.flat[0] == 42.5
    assert shared_back.flat[1] == ux.flat[1]


def test_cross_process_type_mismatch_is_rejected_in_child():
    arr = np.arange(8, dtype=np.float64)
    ch = AnonymousShmChannel(payload_capacity=arr.nbytes, slots=2)
    ctrl = {"child": FORK.Queue()}

    def child(chan, c):
        try:
            chan.receive(expect=Expectation(dtype="f4", unit="Pa"), timeout=30)
            c["child"].put({"error": "NO_ERROR"})
        except BaseException as exc:  # noqa: BLE001
            c["child"].put({"error": f"{type(exc).__name__}: {exc}"})

    proc = FORK.Process(target=child, args=(ch, ctrl))
    proc.start()
    try:
        ch.publish(arr, name="ux", unit="m/s")
        proc.join(60)
        report = ctrl["child"].get(timeout=5)
    finally:
        ch.close()
    assert report["error"].startswith("FieldTypeError"), report
    assert "dtype" in report["error"]


# --------------------------------------------------------------------------- #
# no file intermediation
# --------------------------------------------------------------------------- #
def test_no_file_intermediation(monkeypatch, tmp_path):
    """No regular file may be opened, created or read for the payload."""
    opened = []
    real_os_open = os.open
    real_builtins_open = builtins.open
    real_io_open = io.open

    def spy_os_open(path, *a, **k):
        opened.append(str(path))
        return real_os_open(path, *a, **k)

    def spy_open(path, *a, **k):
        opened.append(str(path))
        return real_builtins_open(path, *a, **k)

    def spy_io_open(path, *a, **k):
        opened.append(str(path))
        return real_io_open(path, *a, **k)

    def shm_entries():
        try:
            return set(os.listdir("/dev/shm"))
        except OSError:
            return set()

    tmp_before = set(os.listdir(tmp_path))
    shm_before = shm_entries()
    monkeypatch.setattr(os, "open", spy_os_open)
    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(io, "open", spy_io_open)

    ch = AnonymousShmChannel(payload_capacity=1 << 12)
    try:
        arr = adversarial_payload()
        ch.publish(arr, name="ux", unit="m/s")
        got = ch.receive(expect=Expectation(unit="m/s"))
        assert got.view.tobytes() == arr.tobytes()
        for _ in range(3):
            ch.publish(arr * 2.0, name="ux", unit="m/s")
            ch.receive(expect=Expectation(unit="m/s"))
    finally:
        ch.close()

    assert opened == [], f"file APIs were called during the exchange: {opened}"
    assert set(os.listdir(tmp_path)) == tmp_before
    assert shm_entries() - shm_before == set(), "a shared-memory name leaked into /dev/shm"


def test_posix_shm_is_unlinked_on_close():
    before = set(os.listdir("/dev/shm"))
    ch = PosixShmChannel(payload_capacity=1 << 10)
    name = ch.kernel_object_name
    assert name in set(os.listdir("/dev/shm"))
    ch.publish(adversarial_payload()[:4], name="ux", unit="m/s")
    ch.receive(expect=Expectation(unit="m/s"))
    ch.close()
    after = set(os.listdir("/dev/shm"))
    assert name not in after
    assert after - before == set()


def test_throughput_guard_reuses_slots_without_copy_growth():
    ch = AnonymousShmChannel(payload_capacity=1 << 14, slots=2)
    try:
        for i in range(20):
            arr = np.full(64, float(i))
            ch.publish(arr, name="ux", unit="m/s")
            got = ch.receive(expect=Expectation(shape=(64,), dtype="f8", unit="m/s"))
            assert got.view.tobytes() == arr.tobytes()
        assert ch.stats.receiver_copies == 0
        assert ch.stats.published == 20 and ch.stats.received == 20
    finally:
        ch.close()