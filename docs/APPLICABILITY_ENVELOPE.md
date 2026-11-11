# Applicability envelopes: machine-checked validity ranges with fail-closed out-of-domain grading

**Module** `tensorphysics.applicability` | **campaign node** P155W1-N09 | **mission** 8238
| configuration `src/tensorphysics/applicability/configs/default_capabilities.v1.json` (digest `a3daea28b05f361a`)
| envelope artifact `artifacts/applicability/envelopes.json` (suite digest `56fe7015279cf087`)

Reproduce:

```bash
PYTHONPATH=src python tools/make_applicability_report.py            # artifacts + figures + this document
PYTHONPATH=src python -m tensorphysics.applicability --out /tmp/envelopes.json --figures /tmp/fig
PYTHONPATH=src python -m tensorphysics.applicability.benchmarks     # synthetic self-check table
PYTHONPATH=src python -m pytest tests/test_applicability_envelope.py tests/test_applicability_artifact.py -q
PYTHONPATH=src python -m tensorphysics.applicability --envelope artifacts/applicability/envelopes.json     --query '{"<capability id>": {"<axis>": 42.0}}'      # offline gate, no probe
```

---

## 1. Case introduction

Every capability of a scientific workflow has a **range in which its numbers mean
something**: a stationary solver is accurate only on grids below some size, an
iteration converges only below some relaxation factor, a transfer operator is
second order only while the feature it resolves is wide enough. Today that range
lives in a paragraph of a human-written report, and nothing stops a pipeline from
printing a number produced outside it as if it were validated.

This node turns the paragraph into a mechanism. A **capability** is registered as
data: its parameter axes (with the span to explore), an evaluator operator, an
acceptance criterion, the failure signals it must respect, and the human-written
declaration it must not exceed. The engine discovers the domain by probing along
each axis, refines the boundary to 0.1 % of the axis span, and stamps every result
with `domain_status` in {IN, OUT, UNKNOWN}. An `OUT` result forces the grading
field to `OUT_OF_DOMAIN` and refuses a `REAL` write.

Three real capabilities of this repository are registered by the shipped
configuration; all three are deliberately **domain-neutral numerics**, so the
mechanism can be validated without any physical assumption:

| capability id | kernel | axis | measured scalar |
|---|---|---|---|
| `stationary-solve-error-vs-grid` | `multigrid_poisson.core.jacobi_solver` | grid size (`grid_n`) | relative Linf error against `analytic_solution` at a fixed 200-sweep budget |
| `stationary-iteration-stability` | `multigrid_poisson.core.jacobi_sweeps` | relaxation factor (`relaxation`) | measured per-sweep growth factor (power iteration, homogeneous problem) |
| `multigrid-contraction-vs-damping` | `multigrid_poisson.core.MultigridSolver` | smoother damping (`damping`) | asymptotic per-V-cycle contraction factor |

Parameter table (every value is injected by the configuration, none hard-coded):

| capability id | axis | searched span | scale | samples | criterion | declared domain |
|---|---|---|---|---|---|---|
| `multigrid-contraction-vs-damping` | `damping` | [0.2, 1.8] | linear | 15 | range [0, 1] | [0.2, 1] (slack 0.02) |
| `stationary-iteration-stability` | `relaxation` | [0.2, 1.6] | linear | 15 | range [0, 1] | [0.2, 1] (slack 0.02) |
| `stationary-solve-error-vs-grid` | `grid_n` | [4, 256] | log | 12 | abs: |v - 0| <= 0.15 | [4, 256] (slack 0.0) |

## 2. Method

Four stages, one module each, so the stages can be tested and replaced
independently.

**Stage 1 - capability interface** (`capability.py`, `axis.py`). A capability is a
`CapabilitySpec`: axes (name, searched span, scale, sample count, monotonicity law,
round-off floor, quantisation), an `EvaluatorSpec` (`module:callable` plus a fixed
key-word block and an axis-to-key-word map), a `FailurePolicy` (criterion in
`{abs, rel, range}`, declared mechanical failure flags, a convergence-order
window, a sign law) and an optional human `declared` statement with its
documentary source. Configuration handling is fail-closed: unknown keys are
rejected, the evaluator target is imported at registration time, and every
declared axis must be driven by the evaluator (and vice versa), so a typo can
never silently switch a criterion off.

**Stage 2 - probing operator** (`probe.py`, `evaluation.py`). Each axis is scanned
with a *geometric gradient* lattice: Chebyshev-Lobatto spacing in scale
coordinates, `u_i = (1 - cos(pi i/(N-1)))/2`, which densifies both ends of the span
(where boundaries normally sit) while still covering the interior; `log` axes
sample multiplicatively. Every sample is judged by three independent criteria:
(a) the acceptance criterion against its reference/anchor; (b) the monotonicity
and sign laws (the sign law is a point property, checked on *every* evaluation;
the monotonicity law is a sequence property, checked across the scan); (c)
numerical pathology - non-finite value, a declared mechanical failure flag, a
convergence-order collapse outside the declared window, or an evaluator that
raises. Anything that raises, diverges or is unknown is never certified.

**Stage 3 - boundary refinement** (`refine.py`). Each adjacent in/out pair is
bisected in scale coordinates down to `rel_tol = 0.001` of the axis span
(0.1 %) and reported as `Boundary(axis, side, value, width, span_fraction, source,
iterations, converged, confidence)`. Sequence properties and quantised axes cannot
be bisected below their resolution: they are reported at sampling/quantum
resolution with the reduced confidence that this implies instead of a fake
precise number.

**Stage 4 - out-of-domain verdict** (`envelope.py`). Per axis the envelope holds
the in-domain intervals (open at an end where no failure was found inside the
span) plus the refined boundaries. `classify(point)` answers `UNKNOWN` outside the
searched span (the mechanism never extrapolates its own scan), `OUT` outside an
interval, and re-evaluates a point inside an interval so that a failing point
property overrides the interval with `OUT`. `assign_grade(verdict, "REAL")` raises
`GradeWriteError` for `OUT` (non-strict mode records `effective = OUT_OF_DOMAIN,
accepted = False`) and downgrades `REAL` to `WITHHELD` for `UNKNOWN`.
`declaration_compat()` compares the detected in-domain hull with the human
declaration and answers `NOT_WIDER`, `CONSERVATIVE`, `WIDER_THAN_DECLARED`,
`NO_DOMAIN_DETECTED` or `NO_DECLARATION`.

**Zero-domain-semantics discipline.** `backend/tools/tests` carry no capability
name, axis name or domain word: ids and axis names live only in the JSON
configuration, and a test enforces that no capability id or axis name of the
shipped configuration appears in the package sources. The evaluator bridge imports
its target by the path the configuration gives it, so the same engine probes an
analytic scalar in a test and a repository kernel in production.

## 3. Results

### 3.1 Envelopes of the three real capabilities

All numbers are read from `artifacts/applicability/envelopes.json`
(suite digest `56fe7015279cf087`); `*` marks an open interval end (no failure found
inside the searched span).

| capability | axis (scale) | searched span | in-domain interval | boundary +/- width | width / span | confidence | declared | compat |
|---|---|---|---|---|---|---|---|---|
| `multigrid-contraction-vs-damping` | `damping` (linear) | [0.2, 1.8] | [0.2, 1.01321]* | 1.01321 ± 0.00139 | 0.000869 | high | [0.2, 1] | NOT_WIDER |
| `stationary-iteration-stability` | `relaxation` (linear) | [0.2, 1.6] | [0.2, 1.0077]* | 1.0077 ± 0.00122 | 0.000869 | high | [0.2, 1] | NOT_WIDER |
| `stationary-solve-error-vs-grid` | `grid_n` (log) | [4, 256] | [4, 22]* | 22 ± 1 | 0.00397 | medium | [4, 256] | CONSERVATIVE |

Compatibility summary: **3 / 3** real capabilities have a
mechanism boundary that is *not wider* than the human declaration (`NOT_WIDER` or
`CONSERVATIVE`), and `wider_low_by = wider_high_by = 0` on every axis.

Physically the three boundaries say the same thing from three angles: the
undamped stationary scheme is a *smoother*, not a solver, and once its spectral
radius approaches one (large grids, over-relaxed or over-damped parameters) the
computed number stops being usable - well before any code raises an exception.

### 3.2 Synthetic self-check against closed-form truth

`artifacts/applicability/synthetic_selfcheck.json` - 13 /
13 cases pass; the largest relative error of a detected
boundary against its closed-form value is **0.333 %**.

| synthetic capability | detected vs truth | rel. error [%] | verdict |
|---|---|---|---|
| `syn-square-window` | high: detected=1.00192 truth=1 | 0.192 | PASS |
| `syn-reciprocal-limit` | low: detected=0.499825 truth=0.5 | 0.0349 | PASS |
| `syn-exponential-tolerance` | low: detected=1.4987 truth=1.49787 | 0.0554 | PASS |
| `syn-divergence-window` | high: detected=1.81754 truth=1.81818 | 0.0354 | PASS |
| `syn-mechanical-flag` | high: detected=3.00116 truth=3 | 0.0388 | PASS |
| `syn-order-window` | low: detected=0.498337 truth=0.5; high: detected=5.99723 truth=6 | 0.333 | PASS |
| `syn-no-failure` | open: detected=[True, True] truth=[True, True] | — | PASS |
| `syn-empty-domain` | intervals: detected=0 truth=0 | — | PASS |
| `syn-monotone-break` | samples_in: detected=1 truth=1; law_boundary: detected=1 truth=1 | — | PASS |
| `syn-sign-law` | samples_in: detected=0 truth=0 | — | PASS |
| `syn-two-axis-product` | x: detected=1.99939 truth=2; y: detected=1.99719 truth=2 | 0.14 | PASS |
| `syn-singular-exception` | pathology: detected=1 truth=ZeroDivisionError | — | PASS |
| `syn-quantised-axis` | high: detected=10 truth=10; quantum: detected=1 truth=1 | 0 | PASS |

### 3.3 Out-of-domain grading

Queries sent through the artifact builder (2 inside, 4 outside):

| query | point | domain status | effective grade | accepted |
|---|---|---|---|---|
| multigrid-contraction-vs-damping @ span high | `{"damping": 1.8}` | OUT | `OUT_OF_DOMAIN` | False |
| multigrid-contraction-vs-damping @ span midpoint | `{"damping": 1.0}` | IN | `REAL` | True |
| stationary-iteration-stability @ span high | `{"relaxation": 1.6}` | OUT | `OUT_OF_DOMAIN` | False |
| stationary-iteration-stability @ span midpoint | `{"relaxation": 0.9}` | IN | `REAL` | True |
| stationary-solve-error-vs-grid @ span high | `{"grid_n": 256.0}` | OUT | `OUT_OF_DOMAIN` | False |
| stationary-solve-error-vs-grid @ span midpoint | `{"grid_n": 32.0}` | OUT | `OUT_OF_DOMAIN` | False |

## 4. Figures

* `artifacts/applicability/fig1_envelopes.png` - per capability: the probed scalar along its axis (filled markers =
  in domain, crosses = out of domain), the refined boundary as a vertical line
  annotated with its value and confidence, and the human-written declared interval
  as a shaded band. The three panels share the same qualitative shape: a long
  in-domain plateau that ends abruptly at the stability limit.
* `artifacts/applicability/fig2_acceptance_window.png` - the same probe samples drawn against the acceptance window
  (red = criterion bounds, green = refined boundary), which makes the boundary
  traceable to the criterion rather than to the sampling.

## 5. Notes, traps and limitations

1. **Point properties vs sequence properties.** Only the acceptance criterion, the
   pathology check and the sign law can be evaluated at an arbitrary point; a
   monotonicity law is a property of a *sequence* and cannot be bisected. Envelope
   ends produced by a law break are reported at sampling resolution with
   `confidence <= 0.5` and an explicit note; a point query inside such an interval
   is certified by the interval plus a point evaluation, not by the law. This is
   recorded in `AxisEnvelope.notes` and `Boundary.source`.
2. **The mechanism never extrapolates.** A query outside the searched span is
   `UNKNOWN` even when the neighbouring interval is open. Widening a span is a
   deliberate, auditable configuration change, not an interpolation.
3. **Quantised axes.** A count-valued axis is snapped to its quantum and the
   boundary is reported within one quantum (`confidence = medium`); its
   `span_fraction` legitimately exceeds the 0.1 % bisection target. Reporting
   bisection precision inside an integer axis would be the dishonest alternative.
4. **One axis at a time.** Axes are probed coordinate-wise with the other axes
   pinned at their declared nominal value, so the envelope describes *slices* of
   the domain, not the joint boundary: a coupled boundary, where the limit of one
   axis moves with another, is not detected. The two-axis synthetic case documents
   this (independent slices).
5. **Declarations are human statements.** The compatibility verdict checks
   containment of the detected hull inside the declared interval with an explicit
   tolerance (`declared.slack`, a fraction of the span) meant to absorb estimator
   noise. `CONSERVATIVE` says the documentation was more conservative than the
   mechanism - information, not a pass/fail, reported per capability.
6. **Estimator noise vs failure.** The stationary growth factor is a finite-sample
   estimate (60 sweeps, ten-ratio tail average), which is why the stability limit
   is detected at about 1.01 rather than exactly at 1.0. The 2 % slack makes that
   explicit instead of tuning the criterion until the numbers look clean.
7. **Unresolved evaluations.** An evaluator failing for a *wiring* reason (outside
   the numeric exception family) yields status `error`, not `pathology`: the axis
   becomes `unresolved` and queries on it are `UNKNOWN`, so a broken declaration is
   withheld rather than interpreted as a physical limit.

## 6. Error judgement

| key quantity | benchmark | detected | error | < 3 % ? |
|---|---|---|---|---|
| synthetic boundary `syn-square-window` | closed-form value (high=1) | high=1.00192 | 0.192 % | yes |
| synthetic boundary `syn-reciprocal-limit` | closed-form value (low=0.5) | low=0.499825 | 0.0349 % | yes |
| synthetic boundary `syn-exponential-tolerance` | closed-form value (low=1.49787) | low=1.4987 | 0.0554 % | yes |
| synthetic boundary `syn-divergence-window` | closed-form value (high=1.81818) | high=1.81754 | 0.0354 % | yes |
| synthetic boundary `syn-mechanical-flag` | closed-form value (high=3) | high=3.00116 | 0.0388 % | yes |
| synthetic boundary `syn-order-window` | closed-form value (low=0.5; high=6) | low=0.498337; high=5.99723 | 0.333 % | yes |
| synthetic boundary `syn-no-failure` | closed-form value (open=[True, True]) | open=[True, True] | 0 % | yes |
| synthetic boundary `syn-empty-domain` | closed-form value (intervals=0) | intervals=0 | 0 % | yes |
| synthetic boundary `syn-monotone-break` | closed-form value (samples_in=1; law_boundary=1) | samples_in=1; law_boundary=1 | 0 % | yes |
| synthetic boundary `syn-sign-law` | closed-form value (samples_in=0) | samples_in=0 | 0 % | yes |
| synthetic boundary `syn-two-axis-product` | closed-form value (x=2; y=2) | x=1.99939; y=1.99719 | 0.14 % | yes |
| synthetic boundary `syn-singular-exception` | closed-form value (pathology=ZeroDivisionError) | pathology=1 | 0 % | yes |
| synthetic boundary `syn-quantised-axis` | closed-form value (high=10; quantum=1) | high=10; quantum=1 | 0 % | yes |

* **Synthetic boundaries vs closed-form truth**: worst relative error
  **0.333 % < 3 %**. The refinement target is 0.001 of the *span*, so
  the relative error on a boundary value scales with span/value; all
  13 cases stay below the 3 % bar.
* **Real capabilities vs human declaration**: containment margins are exactly zero
  on every axis (nothing certified beyond the documented claim). The detected
  boundaries are quantised by the grid size (one grid cell, 1 in units
  of `grid_n`), which is the honest resolution of that axis.
* **No external experimental benchmark applies** to this node: the deliverable is a
  *mechanism* (an out-of-domain detector), not a physical prediction. Its accuracy
  claim is therefore against (i) closed-form algebraic boundaries and (ii) the human
  declarations of the repository's own kernels - both reported above. What is *not*
  established: joint multi-axis boundaries, the physical domains of the
  domain-specific solvers, and any statement about a solver's accuracy *inside* its
  envelope (an orthogonal question).

## 7. Acceptance criteria and verdict

| criterion | evidence | verdict |
|---|---|---|
| mechanism lives in TensorPhysics with pytest coverage | `src/tensorphysics/applicability/`, `tests/test_applicability_envelope.py` (mechanism) and `tests/test_applicability_artifact.py` (round trip + offline gate) | PASS |
| >= 10 synthetic black-box capabilities | `artifacts/applicability/synthetic_selfcheck.json`: 13 / 13 pass | PASS |
| >= 3 real capabilities, mechanism boundary not wider than the declaration | section 3.1: 3 / 3 `NOT_WIDER` or `CONSERVATIVE`, containment margins 0 | PASS |
| boundary refined to 0.1 % of the axis span | `width/span` column of section 3.1 (0.000869, 0.000869) and the continuous-axis synthetic cases; the count axis is documented as quantum-limited | PASS (with documented quantum exception) |
| `domain_status` in {IN, OUT, UNKNOWN}; OUT forces `OUT_OF_DOMAIN` and refuses REAL | section 3.3, `assign_grade`, `GradeWriteError` tests | PASS |
| zero domain semantics in code/tools/tests | capability ids and axis names appear only in the JSON configuration; enforced by a test | PASS |
| no fabricated numbers in this document | every number injected by `tools/make_applicability_report.py` from the probe artifact | PASS |
| the persisted artifact is consumable | section 8: reloaded 3 envelopes, digest `56fe7015279cf087` recomputed and equal to the stored one (True), 11 offline queries | PASS |
| the suite digest identifies content, not the clock | `test_suite_digest_is_reproducible_across_runs` (two runs one second apart) and section 8 | PASS |
| the offline gate is fail-closed | section 8: 3 OUT -> `OUT_OF_DOMAIN`, 3 UNKNOWN -> `WITHHELD`, 2 mechanical errors reported | PASS |

## 8. Consumer side: the artifact is loadable, and the digest is reproducible

Producing an envelope artifact is only half of the mechanism; the other half is
that a **later process can consume it**. `ApplicabilityEnvelope.from_dict` and
`load_envelopes` rebuild the exact frozen dataclasses the probe produced (the
classification code is the same object, not a copy), so a publishing pipeline can
gate a grade without importing any evaluator and without repeating the scan. The
command line exposes it directly:

```bash
python -m tensorphysics.applicability --envelope <artifact.json> --query '<json>' --out <verdicts.json>
```

Two guarantees are enforced on load, both fail-closed:

* **schema** must be `tensorphysics.applicability.envelopes.v1` (anything else is
  an `ArtifactError`);
* **suite digest** is recomputed from the reloaded envelopes and compared with the
  stored value -- a hand-edited interval is rejected instead of certified
  (`ArtifactError: suite digest mismatch`). The escape hatch is explicit
  (`verify_digest=False`).

That check only means something if the digest is *reproducible*. It was not: the
digest hashed the whole serialized envelope including `generated_utc`, so two
identical probe runs one second apart hashed differently and the digest printed in
a report could never be reproduced -- so no consumer could ever verify an artifact.
The digest now hashes content only (`generated_utc` excluded,
`probe.CONTENT_VOLATILE_FIELDS`); the timestamp stays inside the artifact but is
not part of its identity. `test_suite_digest_is_reproducible_across_runs` pins this.

Reload of `artifacts/applicability/envelopes.json`:

| stored digest | recomputed after reload | equal |
|---|---|---|
| `56fe7015279cf087` | `56fe7015279cf087` | True |

Offline verdicts (query points derived from the envelopes themselves, no domain
knowledge in the generator; `artifacts/applicability/offline_gate.json`):

| query | point | domain_status | effective grade | accepted | note |
|---|---|---|---|---|---|
| `multigrid-contraction-vs-damping::inside` | `{"damping": 0.6066060902268281}` | IN | `REAL` | True |  |
| `multigrid-contraction-vs-damping::past-detected-boundary` | `{"damping": 1.4066060902268283}` | OUT | `OUT_OF_DOMAIN` | False |  |
| `multigrid-contraction-vs-damping::beyond-searched-span` | `{"damping": 3.4000000000000004}` | UNKNOWN | `WITHHELD` | False |  |
| `stationary-iteration-stability::inside` | `{"relaxation": 0.6038483275726315}` | IN | `REAL` | True |  |
| `stationary-iteration-stability::past-detected-boundary` | `{"relaxation": 1.3038483275726316}` | OUT | `OUT_OF_DOMAIN` | False |  |
| `stationary-iteration-stability::beyond-searched-span` | `{"relaxation": 3.0}` | UNKNOWN | `WITHHELD` | False |  |
| `stationary-solve-error-vs-grid::inside` | `{"grid_n": 13.0}` | IN | `REAL` | True |  |
| `stationary-solve-error-vs-grid::past-detected-boundary` | `{"grid_n": 139.0}` | OUT | `OUT_OF_DOMAIN` | False |  |
| `stationary-solve-error-vs-grid::beyond-searched-span` | `{"grid_n": 508.0}` | UNKNOWN | `WITHHELD` | False |  |
| `unknown-capability` | `{}` | — | `ERROR` | — | capability not in artifact (available: multigrid-contraction-vs-damping, stationary-iteration-stability, stationary-solve-error-vs-grid) |
| `missing-axis` | `{}` | — | `ERROR` | — | query is missing axes: damping |

The pattern is the one the live path implements: a point inside a detected
interval is `IN` and keeps the requested grade; a point past a detected boundary
but inside the searched span is `OUT` -> `OUT_OF_DOMAIN` (refused); a point
outside the searched span is `UNKNOWN` -> `WITHHELD` (the artifact never
extrapolates); a query for a capability or an axis that the artifact does not
contain is reported as an error instead of being answered. What is *not* provided:
verification of the *producer* identity (the digest is tamper evidence, not a
signature) and re-evaluation of a point property at query time -- the offline
consumer sees only the persisted intervals, whereas the live path additionally
re-evaluates the acceptance criterion at the queried point.

## 9. Joint (multi-axis) coverage: one axis at a time is not enough

Everything above probes **one axis at a time**, every other axis sitting at its
nominal value. That leaves a hole that is invisible in the numbers: a capability
whose criterion fails only when two parameters are unfavourable *together* keeps
both 1-D intervals full, and the offline gate of section 8 then writes `REAL` for
a point the capability fails. Closing that hole is the second stage of this node
(`tensorphysics.applicability.interaction`, report
`docs/reports/p155w1_n09/APPLICABILITY_INTERACTION.md`).

Method: for each configured axis pair, a joint lattice (the same sampling law,
midpoints included) is evaluated; a node is a **witness** when every 1-D interval
places it inside yet the evaluation fails; the lattice cell containing a witness
becomes a conservative out-of-domain box, so `classify` returns `OUT` there and
`REAL` is refused. Cells tile the plane, so boxes leave no gaps, and each box
provably contains a failure - the price is a boundary layer of at most one cell,
which shrinks with the lattice spacing.

| capability | baseline false certifications (61x61 truth lattice) | matured | boxes | over-refusal of the searched box | coarse -> fine lattice |
|---|---|---|---|---|---|
| `syn-joint-product-corner` | 2754 | 0 | 102 | 7.61 % | 11.96 % -> 3.28 % |
| `syn-joint-sum-diagonal` | 1830 | 0 | 78 | 6.32 % | 8.57 % -> 3.57 % |
| `syn-joint-ratio-hyperbola` | 1841 | 0 | 69 | 8.57 % | 14.59 % -> 5.62 % |

* baseline false certifications **6425**, matured **0**,
  criterion C6 `PASS` (per-axis envelopes of the three
  real capabilities are bit-identical before/after, so nothing above moves).
* the joint scan is configuration driven (`interaction: {enabled, samples,
  max_pairs, max_rays}`, defaults {'enabled': True, 'samples': 13, 'max_pairs': 3, 'max_rays': 24}) and single-axis
  capabilities get no joint region at all, hence no extra cost.
* what remains open: the conservative boxes certify only failures that were
  *sampled*; a failure region thinner than one joint cell can still be aliased, so
  the guarantee is resolution-limited rather than continuous. Dense
  `classify(..., verify=True)` on a critical point re-runs the evaluator and is the
  escape hatch.
