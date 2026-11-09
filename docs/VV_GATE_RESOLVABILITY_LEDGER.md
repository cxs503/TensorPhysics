# V&V machine gate, maturation P204-N10: `R = δ/Δx` discriminability + ASME V&V 20 ledger

Status: **delivered, regression-locked**. Module `src/tensorphysics/vv/`
(the mission brief says `tensor_physics/vv_gate/`; the established package path in
this repository is `src/tensorphysics/vv/`, so the maturity work is an *additive
extension of the existing module* instead of a second, competing package — see
§7 "Deviations").

- Gate entry point: `tensorphysics.vv.evaluate(...)`
- New stages: `tensorphysics.vv.resolvability`, `tensorphysics.vv.uncertainty`,
  `tensorphysics.vv.schema`
- Regression lock: `tests/test_vv_in30_regression.py` +
  `tests/data/vv_in30/golden_head_5de4f11.json`
- Numbers/figures: `artifacts/vv_gate_maturity/`

## 1. Physics problem the gate had to learn

Richardson extrapolation answers *"does the discrete solution converge?"*. It
cannot answer *"is the reported physics present in the discrete solution at all?"*

Four ship-domain observables are controlled by a length scale that can be
smaller than the cell:

| observable | controlling scale δ | typical failure |
|---|---|---|
| appendage / aft-body resistance | boundary-layer thickness at AP, δ ≈ 4 mm (model scale) | converged `C_T` that is 8 % low because the layer spans <1 cell |
| cavitation noise inception | vortex-core diameter, δ ≈ 0.1–1 mm | "clean" pressure signal with the cavity entirely subgrid |
| ice contact pressure | contact-zone width, δ ≈ 6 h (see `docs/HULLGIRDER_ICE_EXTREME_BENDING.md`) | contact force carried by one cell, mesh-insensitive by accident |
| green-ship EEDI corrections | added-resistance feature (bulb/paint roughness length) | subgrid always, gate must say so instead of PASS |

A study can therefore be textbook clean — monotone, `p = 2.000`, `GCI = 3.2e-5`
— and still be worthless. The gate now distinguishes the two statements.

## 2. The `R = δ/Δx` discriminability law (W10-N07)

For a declared feature scale δ and the finest grid spacing Δx:

```
R = δ / Δx
```

| R | status | meaning |
|---|---|---|
| `R < 1` | **UNRESOLVED** | feature is subgrid; a converged number is converging to the wrong physics |
| `1 ≤ R < 4` | **MARGINAL** | representable but under-resolved; verdict capped at WARN, the required Δx is reported |
| `R ≥ 4` | **RESOLVED** | 4 cells across the feature; the conservative floor for a 2nd-order reconstruction of amplitude *and* interior gradient |
| δ or Δx not declared | **WITHHELD** | no resolvability statement is possible; the number must not be published as resolved |

Rationale for the floor: 2 cells is the Nyquist limit at which a sinusoidal
feature already loses ≈ 10 % amplitude, and a 2nd-order stencil needs ≥ 2 cells
per side to reconstruct the gradient. 4 cells is the conservative choice; the
policy is a dataclass (`DiscriminabilityPolicy`) whose inequality
`0 < subgrid_below < min_cells_per_feature` is enforced at construction, and the
floor is exposed on the CLI (`--min-cells-per-feature`).

Planning inverse: `required_grid_spacing(δ) = δ / 4`. For the KCS panel study
(δ = 4.2 mm at model scale) that is `Δx ≤ 1.05 mm`, i.e. **≥ 953 panels**
(`artifacts/vv_gate_maturity/resolvability_sweep.json`):

| panels | Δx [m] | R | status |
|---|---|---|---|
| 75 | 1.333e-2 | 0.315 | UNRESOLVED |
| 150 | 6.667e-3 | 0.630 | UNRESOLVED |
| 300 | 3.333e-3 | 1.260 | MARGINAL |
| 600 | 1.667e-3 | 2.520 | MARGINAL |
| 1200 | 8.333e-4 | 5.040 | RESOLVED |
| 4800 | 2.083e-4 | 20.16 | RESOLVED |

Note that `p` and the GCI are *identical* for all of these runs (they are the
same sequence scaled); only the `R` law separates them. That is precisely the
blind spot this node closes.

Gate wiring: the resolution stage runs on `dx = args.dx` if given, otherwise on
the finest level of the study, and appends the check
`resolvability/R=delta/dx`. Status combination uses the ordering
`PASS < WARN < WITHHELD < UNRESOLVED < FAIL` — a hard convergence failure
(`FAIL`: stagnant / oscillatory / order out of band) is never masked by a
resolvability statement, and conversely a subgrid feature can no longer hide
behind a clean convergence pass.

## 3. ASME V&V 20 uncertainty ledger

```
U_val = sqrt(U_num² + U_input²)                 (ASME V&V 20-2009)
U_num = GCI_fine · |f_fine|  or  GCI_fine       (relative / absolute band)
E     = f_sim − f_ref
validated  ⇔  |E| ≤ U_val
```

The maturity gap was not the formula but the **comparison benchmark**: `U_val`
alone cannot say whether the reference could have separated a good model from a
bad one. The ledger therefore carries the reference term `U_ref` explicitly and
reports

```
U_val_total = sqrt(U_num² + U_input² + U_ref²)   (transparency only)
U_ref/U_val                                      (discriminability ratio)
```

Decision rule (`tensorphysics.vv.uncertainty.compare`):

| condition | status |
|---|---|
| `U_ref/U_val ≥ 1` (default factor) | **WITHHELD** — the benchmark is too blurry to decide; the raw `|E|/U_val` is still printed |
| `U_ref/U_val < 1` and `|E| ≤ U_val` | PASS |
| `U_ref/U_val < 1` and `|E| > U_val` | FAIL (honest, not overwritten) |

`U_num` is taken from the finest-grid GCI with the conversion recorded verbatim
in `u_num_source` (e.g. `GCI_fine=0.00568999 (relative, Fs=1.25) x |f_fine|=402.24`),
so the ledger is self-auditing; `UncertaintyLedger.as_dict()["digest"]` is the
sha256 of the canonical ledger.

Worked example (KCS panel refinement, `tests/data/vv_benchmark/ship_resistance_study.json`,
4800 panels, `artifacts/vv_gate_maturity/uncertainty_ledgers.json`):

| term | value [kN] | source |
|---|---|---|
| `U_num` | 2.2887 | GCI_fine = 5.690e-3 (relative) × 402.24 |
| `U_input` | 1.5000 | declared |
| `U_val` | 2.7365 | `sqrt(U_num² + U_input²)` |
| `U_ref` | 2.0000 | declared reference (model-test) uncertainty |
| `U_ref/U_val` | 0.7309 | discriminative |
| `E = f_sim − f_ref` | +4.2400 | 402.24 − 398.00 |
| `|E|/U_val` | 1.5494 | **FAIL** — the model is not validated against this reference |
| ledger digest | `46532f0b6d65…` | sha256 |

Had `U_ref` been declared as 5 kN (a 1.25 %-of-value test uncertainty), the same
study would return **WITHHELD** instead of a misleading FAIL/PASS: at
`U_ref/U_val = 1.83` the comparison cannot discriminate.

## 4. Cross-repository consumption

`tensorphysics.vv.schema` + `schemas/vv_benchmark.schema.json` define the single
document every solver repository must be able to emit; `adapters.py` (extended
with the `ship` dialect) translates the per-repo dialects. See
`docs/VV_BENCHMARK_CROSSREPO_ADAPTER.md`.

Adapted and gated in this worktree (numbers in
`artifacts/vv_gate_maturity/uncertainty_ledgers.json`):

| document | repo dialect | levels | observed p | GCI_fine | verdict |
|---|---|---|---|---|---|
| `ship_resistance_study.json` | `ship-performance.vv-benchmark/1` | 3 | 1.3634 | 5.690e-3 | PASS (convergence only) |
| `tensorlbm_channel_study.json` | `tensorlbm.vv-benchmark/1` | 3 | 2.0000 | 1.054e-4 | WARN (undeclared noise floor) |
| `tensorfem_bending_study.json` | `tensorfem.h-convergence-gci/1.0` | 4 | 2.0002 | 1.250 | PASS |

## 5. Regression against plan 48 (IN30-N01 … N08)

`tools/vv_in30_regression.py` re-expresses the eight IN30 cases as analytic
power-law studies (known `p`, known continuum value), and the golden file
`tests/data/vv_in30/golden_head_5de4f11.json` was captured **before** this node's
changes at gate commit `5de4f11`. The matured gate reproduces it bit-for-bit:

| case | status | p | note |
|---|---|---|---|
| IN30-N01 | WARN | 2.0000 | clean study, undeclared noise floor → capped |
| IN30-N02 | PASS | 2.0000 | round-off declared, `GCI = 3.194e-5` |
| IN30-N03 | PASS | 1.5000 | mixed order recovered |
| IN30-N04 | FAIL | – | stagnant sequence |
| IN30-N05 | FAIL | – | oscillatory (odd/even) |
| IN30-N06 | FAIL | 8.1997 | order outside `(0.05, 6]` |
| IN30-N07 | WARN | 2.0000 | `|E|/U_val = 0.099` (validated, capped by noise certificate) |
| IN30-N08 | FAIL | 2.0000 | `|E|/U_val = 1.872` → honest FAIL |

`digest = 202c700ae10573249d3a7fc9f940fa661bdcfdb6ef8c089a80c4b71133007c4d` for
both the historical and the current run: **the new stages are provably
additive**. `tests/test_vv_in30_regression.py` asserts this equality, the digest,
the historical status table and an independent closed-form recomputation of
`GCI = Fs·|Δf|/|f|/(ρ^p − 1)` for N02.

## 6. How to use

```bash
# convergence + resolvability + ledger in one call
python -m tensorphysics.vv --benchmark tests/data/vv_benchmark/ship_resistance_study.json \
    --format json --feature-scale boundary_layer=0.0042 --u-ref 2.0 --u-input 1.5

# gate-withheld because no feature scale was declared
python -m tensorphysics.vv --benchmark tests/data/vv_benchmark/tensorlbm_channel_study.json \
    --require-resolvability

# programme-level view
python tools/vv_gate_maturity_demo.py     # artifacts + figures
python tools/vv_in30_regression.py        # IN30 corpus, prints + writes goldens
```

Exit codes: `0` PASS, `1` WARN, `2` FAIL, `3` WITHHELD, `4` UNRESOLVED,
`64` usage/input error. The new codes are documented in `cli.py` so a CI job can
treat "cannot decide" differently from "decided negative".

```python
from tensorphysics.vv import evaluate
verdict = evaluate(levels, features=[{"name": "boundary_layer", "delta": 0.0042}],
                   f_exp=398.0, u_input=1.5, u_ref=2.0)
verdict.status, verdict.resolvability.ratio, verdict.uncertainty.u_val
```

## 7. Deviations, limitations, residual risk

**Deviations (declared).**
1. Package path is `src/tensorphysics/vv/` (established, with 94 legacy tests)
   rather than a new `tensor_physics/vv_gate/`; a parallel package would have
   duplicated 1 900 lines and split the gate's single source of truth. All new
   capability lives in new modules next to it.
2. `U_val` keeps the strict ASME form `sqrt(U_num² + U_input²)`; `U_ref` is **not**
   folded into it. It is a separate, explicitly reported term because ASME
   V&V 20 assigns the experimental uncertainty to `U_input` — folding `U_ref` in
   would silently loosen every historical comparison. `U_val_total` is reported
   for transparency only and never used for a decision.

**Limitations.**
- δ is *declared*, not measured. The law is a necessary condition, not a
  sufficient one: `R ≥ 4` does not certify that the feature is *correctly*
  modelled, only that it is representable.
- The 1-cell / 4-cell thresholds are engineering defaults (Nyquist + 2nd-order
  stencil). They are policy, not physics; both are configurable and the chosen
  policy is echoed in every verdict (`policy` block) for reproducibility.
- Uniform refinement is assumed. On an anisotropic/adaptive mesh the caller must
  pass the *local* Δx at the feature (`dx=`); the sweep over panel counts uses
  `Δx = 1/N`, valid only for a uniformly refined panel family.
- `U_ref` is not yet derived from experimental data files: it must be declared
  per benchmark (a `u_ref` field in the contract). Auto-propagation of
  instrument/scatter budgets is future work.
- The `ship` adapter dialect is validated against fixture documents in
  `tests/data/vv_benchmark/`; the live ship-performance API was unreachable from
  this worktree (`:8001/8002 均不可达`), so the end-to-end run against the live
  service is **not** claimed. The adapter is schema-driven, so an unreachable
  service cannot silently change the numbers, but the first live integration
  will need to confirm the emitted field names.
- 5+ grid studies still use the least-squares order fit; no automatic order
  selection study was added in this node.

## 8. Verification performed

```
python -m pytest tests/test_vv_gci_core.py tests/test_vv_validation.py \
    tests/test_vv_adapters_cli.py tests/test_vv_cli_formats.py \
    tests/test_vv_mms_crossrepo.py tests/test_vv_acceptance_n07.py \
    tests/test_vv_resolvability.py tests/test_vv_uncertainty_ledger.py \
    tests/test_vv_in30_regression.py tests/test_vv_schema_ship_adapter.py -q
# 154 passed, 17 skipped   (skips: LBM/TensorFEM artifacts not generated in this checkout)
```

Legacy suite before the change: `94 passed, 17 skipped`. The 60 new tests are
`test_vv_resolvability.py` (20), `test_vv_uncertainty_ledger.py` (16),
`test_vv_in30_regression.py` (9), `test_vv_schema_ship_adapter.py` (15).

Architecture cross-check on the loong64 cluster node (`n330`, same tree staged):

```
python -m pytest tests/test_vv_*.py -q          -> 151 passed, 17 skipped, 3 failed
python tools/vv_in30_regression.py --out …      -> digest 202c700a…  (identical to x86)
```

The 3 cluster failures are the `--format plot` tests and are an environment gap,
not a logic gap: `ModuleNotFoundError: No module named 'matplotlib'` on that
node makes the CLI return `64` (usage/dependency). The IN30 digest is bit-identical
on x86-64 and loong64, which is the check that matters for a numerical gate.

Not-run / residual checks (honest list):

- the **full** repository suite was not completed: a cluster collection run
  segfaults while importing a torch-based test module (pre-existing environment
  defect, unrelated to `tensorphysics.vv`), and locally the suite exceeds the
  harness per-call limit. Two pre-existing failures were observed in the partial
  full run and are caused by artifacts other nodes never generated in this
  checkout: `test_campaign_contract_lock::…[clock_decimal_clock.json]`
  (missing `artifacts/in12_p6/clock_decimal_clock.json`) and
  `test_criteria_mutation` (missing `artifacts/criteria_mutation/paper_trace_audit.json`).
  Neither touches the V&V gate.
- the live ship-performance API was unreachable, so the `ship` dialect is
  validated against fixtures only (§7).