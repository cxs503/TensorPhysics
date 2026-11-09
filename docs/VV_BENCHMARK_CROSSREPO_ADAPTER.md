# Cross-repository V&V benchmark output: adapter contract v1

Owner: TensorPhysics (`src/tensorphysics/vv/schema.py`, `src/tensorphysics/vv/adapters.py`).
Consumers: TensorLBM · TensorFVM · TensorFEM · TensorDEM · TensorBEM ·
ship-performance-platform · TensorPhysics itself.

The gate consumes exactly one shape — a systematically refined study of one
observable on ≥ 2 levels, plus optional feature scales and an optional ASME
V&V 20 comparison block.  Solver repositories keep their own output dialect; the
adapter translates.  **No solver repository has to change a single line** to be
gated, and no solver-specific field name is hard-coded into the gate logic
(names live in `REPO_PROFILES`).

## 1. The contract

Machine-readable schema: `src/tensorphysics/vv/schemas/vv_benchmark.schema.json`
(JSON Schema 2020-12; regenerate with
`python -c "from tensorphysics.vv.schema import write_schema; write_schema()"` —
`tests/test_vv_schema_ship_adapter.py::test_shipped_schema_matches_the_code`
asserts the file and the code agree).

```json
{
  "schema": "ship-performance.vv-benchmark/1",
  "quantity": "total_resistance",
  "unit": "kN",
  "levels": [
    {"panels": 1200, "total_resistance": 412.53, "sigma": 0.20},
    {"panels": 2400, "total_resistance": 405.12, "sigma": 0.10},
    {"panels": 4800, "total_resistance": 402.24, "sigma": 0.05}
  ],
  "features": [{"name": "boundary_layer_thickness_at_ap", "delta": 0.0042, "unit": "m"}],
  "validation": {"f_sim": 402.24, "f_exp": 398.0, "u_input": 1.50, "u_ref": 2.00}
}
```

Rules enforced by `schema.check_benchmark(payload) -> list[str]` (empty list =
conforms; a malformed document yields *reasons*, never a traceback):

| field | rule |
|---|---|
| `schema` | non-empty string; its prefix selects the dialect |
| `quantity` | non-empty string |
| `levels` | list, ≥ 2 objects, spacings strictly **decreasing** (coarse → fine) |
| level spacing | a positive length (`h`, `dx`, `dt`, `mesh_size`, `panel_size`, `cell_size`, `element_size`, `spacing`, `time_step`) **or** a positive count (`panels`, `n_panels`, `cells`, `n_cells`, `elements`, `n_elements`, `nodes`, `resolution`, `n`) |
| level observable | at least one finite number among the value keys (`value`, `f`, `total_resistance`, `ct`, `cw`, `thrust`, `power`, `added_resistance`, `rao`, `kinetic_energy`, `error`, `l2_error`, …) |
| level `sigma` | optional, finite, ≥ 0 (round-off / sampling noise; required for a non-WARN PASS on 3 grids) |
| `features[]` | `name` non-empty, `delta` finite > 0 → feeds the `R = δ/Δx` law |
| `validation` | `f_exp` finite; optional `f_sim`, `u_num`, `u_input`, `u_ref` finite ≥ 0 |

Counts are converted with `Δx = 1/count` because only *ratios* between levels
enter the Richardson/GCI algebra.

## 2. Dialects (version assumptions)

| `schema` prefix | profile | level keys preferred | value keys preferred | version assumption |
|---|---|---|---|---|
| `tensorlbm` | `tensorlbm` | `h`, `dx`, `cell_size`, `dt`, `lattice_spacing` / `resolution`, `n`, `nx`, `ny`, `nz`, `n_cells` | `value`, `f`, `l2_error`, `rel_l2_error`, `kinetic_energy`, `momentum_flux`, `observed` | `tensorlbm.vv-benchmark/1` |
| `tensorfvm` | `tensorfvm` | `h`, `dx`, `cell_size`, `mesh_size` / `n`, `nx`, `cells`, `n_cells` | `value`, `f`, `l1_error`, `l2_error`, `linf_error`, `residual` | `tensorfvm.vv-benchmark/1` |
| `tensorfem` | `tensorfem` | `mesh_size`, `h`, `element_size` / `divisions`, `nodes`, `elements`, `dofs` | `relative_strain_energy_error`, `relative_l2_displacement_error`, `strain_energy`, … | `tensorfem.h-convergence-gci/1.0` |
| `tensordem` | `tensordem` | `dt`, `time_step`, `h`, `spacing` / `n_particles`, `steps` | `value`, `position_error`, `l2_error`, `max_error` | `tensordem.vv-benchmark/1` |
| `tensorbem` | `tensorbem` | `h`, `panel_size`, `mesh_size` / `n_panels`, `panels`, `nodes` | `value`, `rao_error`, `added_mass_error`, `relative_error` | `tensorbem.vv-benchmark/1` |
| `ship-performance` | `ship` | `h`, `dx`, `dt`, `panel_size`, `mesh_size`, `cell_size`, `element_size` / `panels`, `cells`, `elements`, `nodes`, `resolution` | `value`, `total_resistance`, `resistance`, `rt`, `ct`, `cw`, `thrust`, `power`, `added_resistance`, `rao`, `amplitude`, errors | `ship-performance.vv-benchmark/1` (new in this node) |
| anything else | `tensorphysics` / `generic` | walks the document (depth ≤ 6) and takes the first list of ≥ 2 mappings carrying a length-or-count key **and** a value key | | schema-less documents |

Detection order: `schema` prefix → `repo`/`solver` field → generic profile.  A
document that keeps a `repo` field can therefore be schema-less and still be
routed to the right dialect.

## 3. Emitting from a solver repository (minimal)

TensorLBM (lattice refinement) — already the historical shape:

```json
{"schema": "tensorlbm.vv-benchmark/1", "quantity": "pressure_gradient",
 "levels": [{"resolution": 32, "value": 7.914e-3},
            {"resolution": 64, "value": 7.906e-3},
            {"resolution": 128, "value": 7.904e-3}]}
```

TensorFEM (h-convergence evidence, one sub-study):

```json
{"schema": "tensorfem.h-convergence-gci/1.0",
 "quantity": "relative_strain_energy_error",
 "levels": [{"mesh_size": 0.05, "relative_strain_energy_error": 4.12e-2}, …]}
```

ship-performance-platform (panel/cell refinement + validation against a model
test), see `tests/data/vv_benchmark/ship_resistance_study.json`:

```json
{"schema": "ship-performance.vv-benchmark/1", "quantity": "total_resistance", "unit": "kN",
 "levels": [{"panels": 1200, "total_resistance": 412.53}, …],
 "features": [{"name": "boundary_layer_thickness_at_ap", "delta": 0.0042, "unit": "m"}],
 "validation": {"f_sim": 402.24, "f_exp": 398.0, "u_input": 1.5, "u_ref": 2.0}}
```

## 4. Consuming

```python
from tensorphysics.vv import adapters, evaluate
from tensorphysics.vv.schema import check_benchmark, load_benchmark

payload, problems = load_benchmark("study.json")   # problems == absent contract violations
series = adapters.extract_series(payload)          # -> AdapterSeries(repo, quantity, levels)
verdict = evaluate(series.levels, quantity=series.quantity,
                   features=payload.get("features"),
                   f_exp=payload.get("validation", {}).get("f_exp"),
                   u_input=payload.get("validation", {}).get("u_input"),
                   u_ref=payload.get("validation", {}).get("u_ref"))
```

CLI (flags override the document, the document wins over nothing):

```bash
python -m tensorphysics.vv --benchmark study.json --format json
python -m tensorphysics.vv --benchmark study.json --feature-scale bl=0.0042 --u-ref 2.0
python -m tensorphysics.vv --benchmark study.json --require-resolvability
```

`features` / `validation` / `dx` present in a benchmark document are forwarded
by the CLI automatically, so the document reproduces its own V&V statement.

## 5. Verified in this node

| document | dialect resolution | contract | levels | observed p | GCI_fine | verdict |
|---|---|---|---|---|---|---|
| `ship_resistance_study.json` | `ship` (from schema prefix) | clean | 3 | 1.3634 | 5.690e-3 | PASS (convergence) / FAIL (vs 398 kN ref, `|E|/U_val = 1.55`) |
| `tensorlbm_channel_study.json` | `tensorlbm` | clean | 3 | 2.0000 | 1.054e-4 | WARN (noise floor undeclared) |
| `tensorfem_bending_study.json` | `tensorfem` | clean | 4 | 2.0002 | 1.250 | PASS |
| `invalid_no_ordering.json` | `tensorphysics` | **violation**: `levels are not ordered coarse->fine …` | 2 | – | – | rejected with a reason |

## 6. Caveats

- The `ship` dialect was validated against fixture documents only: the ship
  API (`:8001`/`:8002`) was unreachable from this worktree, so live-service field
  names are unverified.  Because the mapping is table-driven
  (`REPO_PROFILES`), a field rename is a one-line fix and cannot silently change
  gate arithmetic.
- Adding a repository = adding one entry to `REPO_PROFILES` (keys only); no gate
  logic changes.
- A study declared with a *dedicated* schema that the adapter does not know is
  read with the generic profile — reported in `series.repo == "generic"` so the
  consumer can see that the dialect was not recognised.
- Units are carried (`unit`) but never converted; mixing `kN` and `N` across
  levels is a caller error the contract cannot detect.