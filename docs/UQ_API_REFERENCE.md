# `tensorphysics.uq` — API reference (P142W3-N10)

Complete reference for the uncertainty-budget package, including the
capabilities matured in node **P142W3-N10**: the Holton-type high-dimensional
anchor, the fail-safe route policy, the cross-repository benchmark adapters and
the command-line front end.

* Package: `src/tensorphysics/uq/`
* CLI: `python -m tensorphysics.uq.cli` (or `python -m tensorphysics.uq`)
* Test suite: `tests/test_uq_*.py` (coverage measured with
  `python -m coverage run --source=src/tensorphysics/uq -m pytest tests/test_uq_*.py`)
* Evidence artefacts for this node: `artifacts/uq_maturity/`
  (`p142w3_n10_verification.json`, figures)

---

## 1. Module map

| module | role |
|---|---|
| `sample_spec.py` | uncertain inputs (`Marginal.uniform/normal/lognormal`, optional Gaussian copula), unit → physical → chaos-coordinate maps, spec fingerprint |
| `sampler.py` | scrambled Sobol' / LHS / MC designs, sha256 **sample-set hash** |
| `adapter.py` | black-box `points (n, d) -> values (n,)`, shard fan-out, hash-pinned merge, NaN bookkeeping |
| `pce.py` | orthonormal Legendre/Hermite bases, sparse total-degree PCE (OMP forward selection on the PRESS/LOO criterion), analytic Sobol' indices, bootstrap CIs |
| `sobol_mc.py` | independent Saltelli/Jansen pick-freeze estimators (1st / 2nd / total), quasi-Monte-Carlo design |
| `verdict.py` | 95 % output interval `U`, three-state gate (PASS / FAIL / WITHHELD), reason codes |
| `workflow.py` | `run_uq_study` — chains sampler → adapter → PCE → Sobol → cross-check → verdict |
| **`holton.py`** *(new)* | high-dimensional linear + pairwise-interaction anchor with **closed-form** Sobol' indices |
| **`failsafe.py`** *(new)* | predeclared `PCE` vs `MONTECARLO` route decision, `Q²` gate, direct band, degraded study |
| **`solver_outputs.py`** *(new)* | TensorLBM / TensorFVM / TensorFEM benchmark-JSON adapters, GUM budget combination, gate chaining |
| **`cli.py` / `__main__.py`** *(new)* | `--pce` / `--sobol` / `--montecarlo` modes, `--failsafe`, `--benchmark-json` |
| `gate.py` | pre-evidence gate, digest-addressed evidence, ASME V&V 20 chaining |
| `robustness.py` | falsification matrix: no injected defect may yield PASS |
| `refinement.py`, `ci_coverage.py` | refinement studies, interval calibration |

---

## 2. Quick start

```python
from tensorphysics.uq import (
    Marginal, SampleSpec, run_uq_study, run_with_failsafe,
    ishigami, ishigami_spec,
)

spec = SampleSpec((
    Marginal.uniform("L", 179.0, 181.0),
    Marginal.normal("Vs", 14.0, 0.05),
))

study = run_uq_study(my_solver, spec, n_samples=512, degree=4, run_mc=True)
study.status                 # "PASS" | "FAIL" | "WITHHELD"
study.pce_sobol.first        # S_i
study.verdict.reason_codes   # ("UQ-W03-LOO", ...)
study.as_dict()              # full JSON evidence
```

Fail-safe form (indices are certified by the chaos route *or* by quasi-MC):

```python
res = run_with_failsafe(my_solver, spec, n_samples=512, degree=4, q2_min=0.5)
res.mode                     # "pce" | "montecarlo"
res.indices.first            # primary S_i (route-appropriate)
res.decision.as_dict()       # q2, floor, trigger reasons
```

---

## 3. Holton-type anchor (`tensorphysics.uq.holton`)

High-dimensional model whose Sobol' decomposition is known in closed form:

```
Y = Σ_i a_i x_i + Σ_{i<j} b_ij x_i x_j,    x_i ~ U(0, 1) independent
```

With `x_i = 1/2 + u_i`, `Var u = 1/12`, `c_i = a_i + (1/2) Σ_j b_ij`:

```
Var(Y) = Σ_i c_i²/12 + Σ_{i<j} b_ij²/144
S_i    = (c_i²/12) / Var(Y)
S_Ti   = (c_i²/12 + Σ_{j≠i} b_ij²/144) / Var(Y)
S_ij   = (b_ij²/144) / Var(Y)
Σ_i S_Ti = 1 + Σ_{i<j} S_ij
```

| function | meaning |
|---|---|
| `make_holton(dim=8, interaction_band=2, interaction_scale=1.4)` | build the anchor (`HoltonBenchmark`: callable, `.spec()`, `.analytic()`, `.as_dict()`) |
| `holton(points, *, linear=None, interaction=None, dim=None)` | evaluate the physical model |
| `holton_spec(dim=8)` | independent `U(0,1)` marginals |
| `holton_analytic(dim=8, *, linear=None, interaction=None)` | closed-form `mean`, `variance`, `first`, `total`, `second`, variance shares |
| `holton_default_coefficients(dim, interaction_band=2, interaction_scale=1.4)` | `(a, b)` derived from centred coefficients `c_i = 0.5/(1+i)` |
| `holton_reference_indices(dim=8)` | analytic dict plus `names` and `sqrt(variance)` |

Defaults (dim 8): first-order indices span `0.537 … 0.008`, the interaction
share is ≈18 % of the variance, and `Σ S_Ti ≈ 1.18` — i.e. a first-order-only
analysis is provably insufficient on this anchor.

---

## 4. Fail-safe policy (`tensorphysics.uq.failsafe`)

Route is decided **before** any index is reported, from a predeclared floor:

| constant / function | meaning |
|---|---|
| `DEFAULT_Q2_MIN = 0.5` | quality floor on `Q²` |
| `loo_q2(surrogate)` | `Q² = 1 − ε_LOO²` (LOO coefficient of determination) |
| `assess(...)` → `FailsafeDecision` | `mode`, `triggered`, `q2`, `reasons` (`PCE-FIT-FAILED`, `PCE-Q2-BELOW-FLOOR`, `PCE-TOOFEW-SAMPLES`) |
| `direct_budget(solver, spec, n_mc=4096)` | 95 % prediction interval by **direct** quasi-MC solver sampling |
| `indices_from_mc(mc)` | promote a pick-freeze estimate to the primary index set |
| `run_with_failsafe(...)` → `FailsafeStudy` | `.mode`, `.decision`, `.indices`, `.budget`, `.verdict`, `.as_dict()` |

Degradation rules:

1. `Q² < q2_min`, or the fit raised, or `n_valid < min_samples_floor(dim)` →
   reject the surrogate;
2. the pick-freeze design (`n_base` default `max(4096, 64·dim)`) becomes the
   primary index route;
3. the output band is measured on solver output (`budget_certified_directly`),
   so the gate can still PASS/FAIL instead of withholding for a missing
   surrogate;
4. the verdict carries `UQ-N05-MC-FAILSAFE` (non-blocking) and the reason for
   the shift is recorded in `decision.reasons`.

`verdict.decide(..., budget_certified_directly=True)` is the only hook this adds
to the existing gate; its default `False` leaves every existing caller's
semantics unchanged.

---

## 5. Cross-repository benchmark adapters (`tensorphysics.uq.solver_outputs`)

Parses benchmark artefacts **without importing the solver repositories**.

```python
from tensorphysics.uq import parse_benchmarks, combine_budget, gate_with_benchmark

artifact, records = parse_benchmarks("data/cross_repo_benchmarks/tensorfvm_lid_cavity_summary.json")
record = records[0]
record.qoi_id, record.value, record.reference, record.relative_error, record.sha256
```

| type / function | meaning |
|---|---|
| `SUPPORTED_SOLVERS` | `("TensorLBM", "TensorFVM", "TensorFEM")` |
| `detect_solver(payload)` | schema-based repository identification |
| `BenchmarkArtifact.load(path, solver=None)` | digest-pinned (`sha256`) parsed artefact |
| `parse_benchmarks(path)` | `(artifact, records)`; raises `UnknownBenchmarkError` on unfamiliar schemas |
| `BenchmarkQoI` | `qoi_id`, `quantity`, `value`, `reference`, `absolute_error`, `relative_error`, `passed`, `detail`, `sha256` |
| `select(records, solver=/qoi_id=/quantity=)` | filtering |
| `relative_error(records, qoi_id=/solver=)` | worst selected relative error |
| `combine_budget(uq_relative_width, records, threshold=0.15)` | GUM quadrature: `u_UQ = U/(2·1.96)`, `u_bench = e/√3`, expanded with `k = 2` |
| `gate_with_benchmark(study=..., records=..., threshold=0.15)` | feeds the benchmark error into `engineering_error_gate` and returns gate + budget + provenance |

Supported schemas:

* **TensorLBM** — `bench_marine.py` result lists (`st_sim`/`st_ref`,
  `cd_sim`/`cd_ref`, `omega_measured`/`omega_theory`); `artifacts/sdaa_*.json`
  hydrostatics blocks (`results.<grid>.<coef>` vs `analytic.<coef>`,
  `passed` taken from `overall_PASS`).
* **TensorFVM** — verification summaries with `meta` + `cases` (velocity-profile
  L2 error against the Ghia reference); mesh-quality `results.json`
  (closed-form shear / non-orthogonality, absolute error kept when the
  reference is exactly zero).
* **TensorFEM** — `finite_width_hole_audit.json` (`kt_net` vs Howland per case,
  plus the `length_study` relative errors).

The fixtures used by the tests are byte-identical copies of real artefacts;
`data/cross_repo_benchmarks/PROVENANCE.json` records the source path and sha256
of each, and `tests/test_uq_solver_outputs.py` re-verifies them.

---

## 6. Command line (`tensorphysics.uq.cli`)

```
python -m tensorphysics.uq.cli --sobol --benchmark ishigami --json out.json
python -m tensorphysics.uq.cli --pce --benchmark holton --dim 8 --degree 2
python -m tensorphysics.uq.cli --montecarlo --benchmark ishigami --n-base 8192
python -m tensorphysics.uq.cli --pce --failsafe --benchmark ishigami --n 8
python -m tensorphysics.uq.cli --benchmark-json data/cross_repo_benchmarks/tensorlbm_sdaa_wigley_hydrostatics.json --list-qoi
python -m tensorphysics.uq.cli --benchmark-json <artifact> --study-json study.json
```

| flag | meaning |
|---|---|
| `--pce` | chaos route only (analytic indices of the surrogate) |
| `--sobol` | chaos route **plus** independent Saltelli cross-check (default) |
| `--montecarlo` | quasi-MC indices only, no surrogate |
| `--mode {pce,sobol,montecarlo}` | long form of the same three modes (mutually exclusive with the short flags) |
| `--benchmark {ishigami,holton,sobol-g,holtrop-type}` | test function / benchmark |
| `--benchmark-json PATH` | parse a solver-repository artefact instead |
| `--study-json PATH` | with `--benchmark-json`: combine the UQ band with the benchmark errors |
| `--list-qoi` | list extracted QoI ids |
| `--dim`, `--n`, `--degree`, `--method`, `--n-base`, `--budget-n-mc`, `--bootstrap`, `--seed`, `--rule` | design / truncation / sampling controls |
| `--threshold` | engineering threshold on the relative 95 % band (default 0.15) |
| `--q2-min` | fail-safe quality floor (default 0.5) |
| `--eps-loo-max` | surrogate LOO gate (default 0.1) |
| `--failsafe` | also run the fail-safe policy and attach its report |
| `--require-pass` | exit `3` unless the gate status is `PASS` |
| `--json PATH`, `--quiet` | write the full report, suppress the summary |

Exit codes: `0` normal, `2` usage / schema / missing-evidence error,
`3` `--require-pass` not satisfied.

---

## 7. Verdict reason codes

| code | blocking | meaning |
|---|---|---|
| `UQ-W01-NONFINITE` | yes | ≥1 non-finite solver sample |
| `UQ-W02-TOOFEW` | yes | too few valid samples (per input or per active term) |
| `UQ-W03-LOO` | yes | surrogate `ε_LOO` above `eps_loo_max` |
| `UQ-W04-CROSSCHECK` | yes | PCE and Saltelli indices disagree |
| `UQ-W05-DEGENERATE` | yes | zero / non-estimable QoI variance |
| `UQ-W06-HASH` | yes | results belong to a different sample set |
| `UQ-W07-NOMC` | yes | Monte-Carlo cross-check missing when required |
| `UQ-W08-NOPCE` | yes | no surrogate to certify the band |
| `UQ-N01-CORRELATED` | no | latent-coordinate indices (copula) |
| `UQ-N02-NOISY-MC` | no | Jansen vs Saltelli first-order spread > 0.05 |
| `UQ-N03-WIDE-INDEX-CI` | no | bootstrap index CI too wide |
| `UQ-N04-NO-CROSSCHECK` | no | cross-check explicitly disabled |
| `UQ-N05-MC-FAILSAFE` | no | band certified directly; indices from the quasi-MC route |

`FailsafeDecision.reasons` (route level, distinct from verdict codes):
`PCE-FIT-FAILED`, `PCE-Q2-BELOW-FLOOR`, `PCE-TOOFEW-SAMPLES`.

---

## 8. Evidence produced by node P142W3-N10

| acceptance item | artefact / test |
|---|---|
| F1 Ishigami indices ≤ 2 % | `tests/test_uq_cli.py::test_pce_mode_reports_back_full_report`, `artifacts/uq_maturity/p142w3_n10_verification.json` |
| F2 Holton indices ≤ 5 % | `tests/test_uq_holton.py::test_holton_sampled_indices_converge_within_five_percent` |
| F3 `Q² ≥ 0.95` | same CLI test (`q2` field), verification JSON |
| F4 fail-safe switch, error ≤ 10 % | `tests/test_uq_failsafe.py::test_run_with_failsafe_degrades_and_stays_within_ten_percent` |
| F5 coverage / docs / CLI modes | `verification JSON → f5_maturity_gate`, `tests/test_uq_cli.py` |
| F6 ≥2 repositories parsed | `tests/test_uq_solver_outputs.py`, `data/cross_repo_benchmarks/` |
| F7 ≥20 pytest items | `verification JSON → f7_pytest` |

---

## 9. Limits

* The chaos route assumes a smooth QoI at the chosen total degree; `Q²` is the
  machine-checked escape hatch, and the fail-safe policy is the automatic one.
* Sparse selection is greedy OMP with a PRESS stopping rule, not a full LARS
  path or an L1 solver.
* The GUM combination treats the benchmark comparison error as a uniform bound
  (`e/√3`) and the UQ band as approximately normal — both are stated
  assumptions, not measurements.
* Determinism is scoped to a NumPy/SciPy build; the sample-set hash makes a
  change visible instead of silent.
* Cross-repository parsing is schema-based: an unfamiliar artefact raises
  `UnknownBenchmarkError` rather than being guessed at.