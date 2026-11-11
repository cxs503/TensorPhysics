# In-memory cross-solver field exchange contract (GAP-CROSS-FIELD)

Status: implemented + machine-verified (mission m14678, 2026-11-11)
Modules: `src/tensorphysics/field_exchange/transport.py`, `.../channels.py`
Tests: `tests/test_field_exchange_transport.py` (21 tests)
Benchmark: `tools/m14678_field_transport_benchmark.py` → `artifacts/m14678/`

## 1. What the gap was

`field_exchange.mapping` / `volume_transfer` already moved fields **inside one
process** between discretisations (panel → node, cell → node, particle → volume
fraction). Nothing existed for the *transport* step of a coupled run: two real
solver processes/modules handing a field (`ux`, `rho`, wall traction) to each
other **in memory**, with evidence that

1. the round trip is bit-exact (not `allclose`),
2. the copy semantics are declared and audited (zero-copy or single-copy),
3. dtype/shape/order and physical **units** are asserted on both ends,
4. **no file** intermediates the payload (no `np.save`, no temp file, no
   persistent `/dev/shm` name).

## 2. Contract surface

```python
from tensorphysics.field_exchange import (
    AnonymousShmChannel, SocketChannel, ExchangeContract, Expectation, describe)

chan = AnonymousShmChannel(payload_capacity=ux.nbytes, slots=4)   # before fork
desc = chan.publish(ux, name="ux", unit="m/s",
                    contract=ExchangeContract(max_total_copies=1))
# ... peer process ...
got = chan.receive(expect=Expectation(name="ux", dtype="f8", unit="m/s"),
                   contract=ExchangeContract(max_total_copies=1, require_zero_receiver_copy=True))
assert got.view.tobytes() == ux.tobytes()      # bit-exact
```

| element | meaning |
|---|---|
| `FieldDescriptor` | name, shape, dtype (canonical little-endian), order, unit text, `nbytes`, **sha256 of the payload** |
| `Expectation` | receiver-side declaration: name / shape / dtype / unit |
| `CopyLedger` | `producer_copies`, `receiver_copies`, `total` — carried **inside the frame** so the peer can audit it |
| `ExchangeContract` | acceptance rule: `max_total_copies`, `require_zero_receiver_copy`, `require_bit_exact`; violations raise `CopyBudgetExceeded` / `PayloadMismatchError` |
| `ReceivedField` | view + descriptor + ledger + seq; `view` is an `np.ndarray` **view** (zero copy) on shared-memory channels |

### Channels and their honest copy semantics

| channel | shared state | producer copies | receiver copies | file involvement |
|---|---|---|---|---|
| `AnonymousShmChannel` (arena path: `allocate` → write `.array` → `publish_buffer`) | `mmap(MAP_SHARED\|MAP_ANONYMOUS)`, inherited by `fork` | 0 | 0 | none (no name, no `/dev/shm` entry, no `read`/`write`) |
| `AnonymousShmChannel` (marshal path: `publish(array)`) | same | 1 | 0 | none |
| `PosixShmChannel` | `multiprocessing.shared_memory` kernel tmpfs object | 0 / 1 | 0 | name only, unlinked on `close()`; never opened as a file |
| `SocketChannel` | `AF_UNIX` `SOCK_STREAM` `socketpair` | 1 (into kernel buffer) | 1 (`recv_into`) | none |
| `LocalChannel` | module → module, same interpreter | 0 | 0 | none |

`FieldBuffer` is **single-use**: one `allocate` per frame (`ChannelStateError`
on re-publication). Slot count ≥ 2 keeps a single producer/consumer pair from
lapping; a lapped slot is detected by the per-slot `seq` and raises
`FrameCorruptError` instead of silently mixing fields.

### Ordering

`publish` stores the payload, writes the frame header (incl. `producer_copies`),
then releases a `multiprocessing.Value('q', lock=True)` sequence counter that the
consumer polls in `wait_for()` (timeout, no filesystem use). The counter is
inherited across `fork`, so the publish is visible before the receive.

## 3. Type and unit assertions

* dtype: canonical little-endian; big-endian payloads are **rejected** with a
  conversion hint (`FieldTypeError`), so byte identity is host-independent.
* layout: non-contiguous arrays are rejected instead of being silently copied
  (`np.ascontiguousarray` must be written explicitly — the copy becomes visible
  in the code, not hidden in the transport).
* unit strings: UDUNITS-2/CF subset parsed into exponents over the seven SI base
  dimensions — `Pa` ≡ `N/m^2`, `m/s` ≡ `m*s^-1`; `m/s` vs `Pa` raises
  `UnitMismatchError`; unknown tokens (`furlong`) and scaling factors (`2*m`)
  raise `UnknownUnitError` instead of degrading to "don't care".

## 4. Adjacent utilities

* `np.ascontiguousarray` + `tobytes()` remain the escape hatches for payloads
  that must be *materialised*; the contract exists so that the common path does
  not need them.
* `field_exchange.mapping` / `volume_transfer` consume `got.view` unchanged: the
  transport hands over a normal `np.ndarray`, so conservative mapping audits
  (`audit_panel_map`, `audit_nodal_field`) run on the received field directly —
  the benchmark does exactly that with `wall_shear_from_profile`.

## 5. Evidence (artifacts/m14678)

`results.json` + `run_manifest.json` + three figures. Key numbers from the run:

| quantity | reference | got | error |
|---|---|---|---|
| `tau_w` (Couette, `mu U / H`) | analytic | produced from the *received* field in the second process | see `results.json → physical.tau_w_rel_error` (≈1e-11) |
| returned profile vs producer profile | producer in-process | consumer-returned frame | max abs diff `0.0` (bit-exact) |
| payload sha256 | descriptor | received payload | identical |

Repro:

```
python tools/m14678_field_transport_benchmark.py --out artifacts/m14678
python -m pytest tests/test_field_exchange_transport.py -q
```

## 6. External anchors

* IEEE 754-2019 — bit patterns of `NaN`/`-0.0`/subnormals are part of the
  payload; the tests include a NaN with a non-default payload and a negative NaN.
* PEP 3118 (buffer protocol) / NumPy view semantics — zero-copy `np.ndarray`
  construction over `mmap`.
* `mmap(2)` `MAP_SHARED|MAP_ANONYMOUS`, `shm_open(3)` — kernel-mediated shared
  pages, no user-space file I/O.
* MPI-4.0 §11 (one-sided communication / shared-memory windows) — the same
  "publish descriptor + bytes into a window, peer reads a view" model.
* UDUNITS-2 grammar (Unidata), CF conventions §3.1 — unit strings.
* Analytic plane-Couette solution `tau_w = mu U / H` — physical anchor for the
  scalar computed from the exchanged field.

## 7. Limits / not covered

* `fork`-based peers only for `AnonymousShmChannel` (Linux default). Unrelated
  processes must use `PosixShmChannel` (kernel tmpfs name) or `SocketChannel`.
* Single-producer / single-consumer ring; concurrent publishers would need an
  external lock (documented, not implemented).
* Large payloads (> physical RAM) are out of scope: the arena is fully resident.
* GPU / accelerator memory is not modelled (`cupy`/`torch` tensors would need a
  device-aware descriptor extension).

### Measured cost of the contract (honest)

The acceptance target is semantics (bit-exact, 0/1 copy, typed, file-free), not
throughput — and the throughput is **dominated by the audit itself**, not by the
transport. On this host `sha256` costs ~9.6 ms per 1 MB (~110 MB/s), and the
contract hashes the payload on publish *and* on receive:

| payload | round trip (shm zero-copy) | file baseline (forbidden) | copies (shm / file) |
|---|---|---|---|
| 1 KB | 0.13 ms | 0.38 ms | 0 / 2 |
| 128 KB | 1.56 ms | 0.46 ms | 0 / 2 |
| 1 MB | 11.8 ms | 0.97 ms | 0 / 2 |
| 4 MB | 46.6 ms | 3.63 ms | 0 / 2 |

So for large fields the sha256 verification (≈2 × 9.6 ms/MB) and first-touch page
faults on fresh slots outweigh the memcpy that the file path pays — the zero-copy
property buys *no* payload copy and *no* file I/O, not latency. A production fast
path would keep the descriptor checksum (publish side) and skip the receive-side
re-hash, or switch to CRC32C/xxhash; that is deliberately **not** the default
here because bit-exactness enforcement is the point of the contract. Numbers come
from `artifacts/m14678/results.json → transport_matrix`.