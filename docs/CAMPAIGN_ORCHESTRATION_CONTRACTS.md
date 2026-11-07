# Campaign orchestration contracts (`tensorphysics.campaign`)

`tensorphysics.campaign` is the **contract layer** for multi-library ice-region
campaigns.  It contains no solver kernels: kernels live in TensorLBM / TensorFVM /
TensorFEM, while this package fixes the semantics that make a coupled campaign
reproducible and machine-checkable across libraries.

Everything exchanged between libraries is a versioned JSON artifact with a closed
schema (`additionalProperties: false`), a canonical SHA-256 digest and a schema
fingerprint that is pinned in tests.  Artifacts are validated on the producer side
before writing and again on the consumer side after reading.

## 1. Coupling clock (M12 disposition)

`CouplingClockSpec(master_name, sub_name, master_dt_s, sub_dt_s, exchange_every)`

* `ratio = dt_master / dt_sub` must be an integer within `RATIO_RTOL = 1e-12`;
  non-integer ratios are rejected instead of silently truncated.
* `exchange_every` (master ticks between exchanges) has **no default** — disposition
  M12 requires the cadence to be explicit.
* Time lives on an integer sub-step lattice: an exchange occurs at
  `master_tick * ratio` sub-steps and its physical time is one rounding of
  `master_tick * dt_master`, so a long campaign cannot drift.  `clock_drift()`
  quantifies the drift of naive `t += dt` accumulation for comparison.
* `max_information_age_s = exchange_every * dt_master` is the upper bound on how
  stale a coupled quantity may be.
* `CouplingClockSpec.from_dict` recomputes every derived field and the tick state
  and rejects internally inconsistent artifacts.

Artifact schema: `coupling_clock.v1`.

## 2. Dual ledger reconciliation (mass / energy / impulse)

* `SideLedger(quantity, side, entries)` holds one side of a book; `side` is
  `producer` or `consumer`, `QuantityKind` is `mass` (kg, 1 component),
  `energy` (J, 1 component) or `impulse` (N*s, 3 components).
* Both sides must book the same exchange windows (labels and steps); a missing or
  reordered window fails reconciliation.
* `reconcile(producer, consumer, atol_ulps=8)` grades every component:
  * `exact` — residual is literally `0.0`;
  * `machine_precision` — `|dP - dC| <= atol_ulps * eps * scale`;
  * `out_of_tolerance` — the books disagree beyond rounding.

  `scale = max(|P|, |C|, sum|p_k|, sum|c_k|)` is a condition-number proxy so that
  books with large opposite-sign transfers are not judged by their total alone.
* `reconcile_books(books, scenario=...)` emits the `ledger_reconciliation.v1`
  artifact with a `same_batch_digest`; `verify_same_batch()` recomputes it, which
  is the campaign `diff = 0` criterion.
* `build_reference_dual_ledgers(spec, master_ticks, fluxes)` instantiates the
  template for a known analytic flux: the producer sums each window with
  `math.fsum`, the consumer accumulates sequentially, so the two books differ only
  by summation rounding — the machine-precision floor the criterion asserts.

Artifacts: `side_ledger.v1`, `ledger_reconciliation.v1`.

## 3. Geometry contract and failure faces

* `GeometryContractRegistry.register_file("geom_pc6.json")` validates the closed
  schema, checks `Cb * Lpp * B * T = nabla` (relative closure `<= 1e-12`) and
  `Lpp > B > T`, enforces that `geometry_id` matches the file name, and resolves
  the referenced failure-face packet.
* `FailureFacePacket` must describe a closed control surface,
  `sum_i A_i n_i = 0` (relative closure `<= 1e-12`, discrete divergence theorem).
  `tetrahedron_face_packet()` builds such a packet from vertex coordinates; its
  irrational face areas mean the identity only cancels to round-off, so the check
  has real content.  Dropping a face or flipping a normal opens the surface and is
  rejected.
* Interface negotiation: `failure_faces` v2 extends v1 by optional `centroid_m`,
  `failure_mode`, `contact_region`, `load_share`, `redundancy_group`.  A v1 packet
  is accepted by a v2 consumer, a v2 packet downgrades lossily to v1, and versions
  outside the contract's `consumer_contract` window are rejected.

Artifacts: `geometry_contract.v1`, `failure_faces.v1`, `failure_faces.v2`.

## 4. Schema lock

`schema_fingerprint(name)` hashes the canonical JSON of a registered schema;
`schema_fingerprints()` returns the whole set.  `tests/test_campaign_contract_lock.py`
pins those fingerprints, so a schema edit that is not accompanied by an explicit
pin update breaks both the producer and the consumer test path.

## 5. Reference execution

```bash
python3 scripts/run_in12_p6_campaign.py   # artifacts/in12_p6/*.json
python3 scripts/in12_p6_audit.py          # consumer-side recomputation + criterion_report.json
python3 scripts/in12_p6_report.py         # docs/reports/in12_p6/REPORT.md + figures
python3 -m pytest tests/test_campaign_*.py -q
```

The driver script holds no numerics: on the IN12-P6 node it is 102 code lines
against 1791 library code lines (**5.7 %**), the thin-orchestration self-proof
(regenerated into `artifacts/in12_p6/criterion_report.json` on every run).