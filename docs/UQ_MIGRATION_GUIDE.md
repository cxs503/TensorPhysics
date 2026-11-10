# Migrating from hand-rolled uncertainty analysis to `tensorphysics.uq`

Audience: an engineer or agent who currently produces uncertainty statements
inside a benchmark script — a loop that perturbs a few inputs, a hard-coded
Sobol' table from a paper, a spreadsheet of "±10 %", or a single-number "error
bar" typed into a report. The goal of the migration is that the number in the
report is **reproducible, digest-addressed and machine-adjudicated**, and that a
bad surrogate cannot be quoted as if it were evidence.

The whole migration is 8 steps; steps 0 and 8 are the ones people skip and
later regret.

---

## Step 0 — write down the decision the study supports

Before touching code, state (in the report or the PR description):

* **Decision**: e.g. "is the 95 % band of `R_T` inside the 15 % engineering
  threshold, so that the comparison error may be quoted?"
* **Alternatives**: PCE route, sampled route, or no statement at all.
* **Predeclared thresholds**: `threshold` (default 0.15), `eps_loo_max`
  (default 0.1), `q2_min` (default 0.5), sample floors.
* **Stopping rule**: stop when the gate returns PASS or the band FAILs; do not
  re-run with a different seed until it passes.

These are inputs to the code below, not afterthoughts: `run_with_failsafe` takes
`q2_min` **before** the fit is inspected.

---

## Step 1 — replace the perturbation loop with a `SampleSpec`

Before (typical hand-rolled code):

```python
rng = np.random.default_rng(0)
for _ in range(200):
    L = 180.0 + 1.0 * (2 * rng.random() - 1)      # "±1 m"
    Vs = 14.0 * (1 + 0.05 * rng.standard_normal()) # "5 %"
    rt.append(solver(L, Vs))
sd = np.std(rt); print("R_T = %.2f ± %.2f kN" % (np.mean(rt), 1.96 * sd))
```

After:

```python
from tensorphysics.uq import Marginal, SampleSpec

spec = SampleSpec((
    Marginal.uniform("L", 179.0, 181.0),     # ±1 m, stated explicitly
    Marginal.normal("Vs", 14.0, 0.70),       # 5 % of 14 kn, as a std dev
    Marginal.uniform("Cb", 0.615, 0.625),
))
```

Why: the distribution shape is now explicit and is what selects the chaos basis
(Legendre for uniform, Hermite for normal). The "±10 %" ambiguity disappears.

**Checkpoint**: `spec.dim`, `spec.names`, `spec.fingerprint()` (hex digest you
can put in the report).

---

## Step 2 — wrap the solver, do not modify it

```python
from tensorphysics.uq import run_uq_study

def solver(points: np.ndarray) -> np.ndarray:
    # points: (n, 3) physical parameters in spec.names order
    return np.array([resistance_kn(L, Cb, Vs) for L, Vs, Cb in points])
```

Rules:

* one array in, one array out, same length; **no** internal randomness;
* non-finite results are allowed (they are recorded as NaN and force
  `WITHHELD`) but a *wrong length* raises `AdapterContractError` — fix the
  adapter instead of returning a partial vector;
* if the solver runs in shards or on a cluster, pass `runner=` /
  `cluster_shard_command` rather than reordering results by hand.

**Checkpoint**: `solver(spec.transform(unit).points).shape == (n, 3)`.

---

## Step 3 — run the plain study and read the verdict

```python
study = run_uq_study(solver, spec, n_samples=512, degree=4, run_mc=True)
print(study.status, study.verdict.reason_codes)
print(study.pce_sobol.first, study.pce_sobol.total)
print(study.budget.relative_width)
```

Interpretation:

| status | meaning | action |
|---|---|---|
| `PASS` | band inside the threshold and evidence complete | quote it |
| `FAIL` | band outside the threshold | report the band, do **not** narrow the inputs to pass |
| `WITHHELD` | evidence incomplete | read the reason codes; do not quote the band |

The two numbers that decide most cases: `study.surrogate.eps_loo` (surrogate
quality) and `study.budget.relative_width` (physical band).

**Checkpoint**: `study.as_dict()` is JSON-serialisable and carries the
sample-set hash, spec fingerprint and timing.

---

## Step 4 — add the fail-safe policy for production runs

Hand-rolled analyses usually fail silently when the sample budget is too small:
a degree-8 fit on 20 runs still returns indices. Replace "hope" with a declared
route:

```python
from tensorphysics.uq import run_with_failsafe

res = run_with_failsafe(solver, spec, n_samples=512, degree=4, q2_min=0.5)
res.mode          # "pce" or "montecarlo"
res.decision.as_dict()
# {'mode': 'montecarlo', 'triggered': True,
#  'reasons': ['PCE-TOOFEW-SAMPLES'], 'q2': ..., 'q2_min': 0.5, ...}
```

If the route degraded, `res.indices` are the **quasi-Monte-Carlo** (scrambled
Sobol' sequence, Saltelli/Jansen) indices and `res.budget` is a direct solver
prediction interval. Both are honest about their cost
(`res.metadata["n_mc_evaluations"]`).

**Checkpoint**: `res.mode` is decided by `res.decision.reasons`, never by
looking at the indices and deciding they "look right".

---

## Step 5 — attach a benchmark reference instead of retyping the error

Hand-rolled reports usually contain a typed number like "error vs Howland: 0.9 %".
Replace it with a parsed artefact:

```python
from tensorphysics.uq import parse_benchmarks, combine_budget, gate_with_benchmark

artifact, records = parse_benchmarks(
    "data/cross_repo_benchmarks/tensorfem_finite_width_hole_audit.json"
)
print(artifact.solver, artifact.sha256)      # TensorFEM, digest of the file
print(records[0].qoi_id, records[0].relative_error)
```

Supported today: TensorLBM (marine list + hydrostatics blocks), TensorFVM
(profile verification + mesh quality), TensorFEM (finite-width hole). An
unfamiliar schema raises `UnknownBenchmarkError` — extend the parser rather than
pasting numbers.

**Checkpoint**: the report quotes `record.qoi_id` and `record.sha256[:12]`.

---

## Step 6 — combine the budget (GUM, first order)

```python
budget = combine_budget(study.budget.relative_width, records, threshold=0.15)
budget.as_dict()["components"]     # uq_band + one component per benchmark QoI
budget.expanded_relative           # k = 2 expanded combined uncertainty
budget.status                      # PASS / FAIL against the threshold
```

Assumptions (state them in the report):

* `u_UQ = U95 / (2 × 1.959964)` — the band is treated as approximately normal;
* `u_bench = e / √3` — the comparison error is a uniform bound, not a std dev;
* components are independent (different solvers / references).

---

## Step 7 — gate the claim, not just the numbers

```python
out = gate_with_benchmark(study=study, records=records, threshold=0.15,
                          label="net-section Kt, rho=0.3")
out["status"]              # PASS / FAIL / WITHHELD
out["gate"]["evidence_digest"]
```

Or chain it into ASME V&V 20 with the numerical uncertainty `u_num`:

```python
from tensorphysics.uq import chain_asme_validation
res = chain_asme_validation(study=study, f_sim=2.744, f_exp=2.721, u_num=0.02)
res["combined_status"], res["asme"]["ratio"], res["gate"]["status"]
```

A claim without a UQ record returns `WITHHELD` (`UQ-G01-NO-EVIDENCE`) by design —
absence of evidence is not a pass.

---

## Step 8 — make it runnable by someone else

Two options, both intended to be copy-paste:

```bash
# one-shot, reproducible, no Python
python -m tensorphysics.uq.cli --sobol --benchmark ishigami --json out.json
python -m tensorphysics.uq.cli --pce --failsafe --benchmark holton --dim 8 --degree 2
python -m tensorphysics.uq.cli --benchmark-json <artifact> --study-json out.json
```

```python
# in-process, as a test
from tensorphysics.uq import ishigami, ishigami_analytic, ishigami_spec, run_uq_study
study = run_uq_study(ishigami, ishigami_spec(), n_samples=256, degree=8, run_mc=False)
assert study.status in {"PASS", "FAIL", "WITHHELD"}
```

Finally record in the report: package version, `--seed`, `--n`, `--degree`,
`--n-base`, `--threshold`, `--q2-min`, spec fingerprint, sample-set hash,
artefact digests. All of them are already inside `study.as_dict()` and the
verification JSON, so the migration is finished when the report can be
regenerated without you.

---

## Migration checklist

- [ ] decision, alternatives and thresholds written **before** the run
- [ ] `SampleSpec` marginals replace ad-hoc perturbations
- [ ] solver wrapped as `points -> values` (no kernel modification)
- [ ] `run_uq_study` status read; `WITHHELD` never quoted as a number
- [ ] `run_with_failsafe` used for production runs; degraded mode recorded
- [ ] benchmark error parsed from a digest-pinned artefact, not retyped
- [ ] `combine_budget` assumptions stated (normal band, uniform bound, k = 2)
- [ ] claim adjudicated by `gate_with_benchmark` / `chain_asme_validation`
- [ ] CLI or a pytest reproduces the number
- [ ] hashes, seeds and versions in the report

## What this migration does *not* fix

* It does not make a coarse solver accurate — it makes the error band honest.
* It does not remove the discretisation uncertainty `u_num`; that is the ASME
  chaining input and stays the caller's responsibility.
* It does not license quoting indices from a `WITHHELD` study: reason codes
  `UQ-W0*` are blocking and mean the evidence is incomplete.