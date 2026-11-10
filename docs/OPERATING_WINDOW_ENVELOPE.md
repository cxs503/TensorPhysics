# Operating-window envelope engine

`tensorphysics.envelope` answers one question for a parameterised engineering
capability: **on which part of a declared parameter span do all declared
constraints hold, exactly where does that part end, and is the answer
believable?**

Before this module, a window in this repository was one of three things: a
paragraph of prose in a report, a hand-tuned script per capability, or a dense
grid scan whose answer moved with the grid. The engine replaces all three with a
stage-ordered mechanism where the numbers, the semantics and the verdicts are
produced together.

## 1. What enters, what leaves

```python
from tensorphysics.envelope import EnvelopeEngine, load_case

case = load_case("scenarios/envelope_cases/<case>.json")   # configuration only
result = EnvelopeEngine(case).run(condition="<label>")

result.status          # "feasible" | "infeasible"  (derived, never declared)
result.intervals       # 1-D: refined boundaries + the binding constraint per side
result.components      # 2-D: separate regions, each with its own area and drivers
result.discrete        # attainable settings on the declared lattice (or None)
result.nearest         # only when empty: point + residual + search resolution
result.contracts       # per-check verdicts with the numbers behind them
result.n_evals, result.elapsed_s
```

`result.to_dict()` is JSON-serialisable and is what the artifacts and the
acceptance script consume, so a report number can always be traced back to a
result object, a configuration digest and a code digest.

## 2. The five stages

| stage | module | question |
|---|---|---|
| 1 constraint model | `constraints`, `expressions` | what is the constraint network, as data? |
| 2 grid scan | `grid` | coarse feasible mask, active set, cell area |
| 3 boundary marching | `marching`, `roots` | where exactly is the boundary; how many regions? |
| 4 contract checks | `contracts` | is the answer believable? |
| 5 orchestration | `engine`, `artifact` | one result object plus a digest chain |

**Stage 1 — constraint model.** Three kinds (`inequality`, `equality`,
`discrete`) are all reduced to one scalar margin with a single sign convention,
`feasible ⟺ margin >= 0`:

```
inequality   margin = side * value                     side = +1 (value >= 0) or -1 (value <= 0)
equality     margin = tolerance - |value - target|
discrete     margin = tolerance - distance(value, lattice)
```

Everything downstream is therefore kind-agnostic. The margin is an expression
tree over named parameters (`expressions`); the operator table is closed
(`const`, `var`, `sum`, `product`, `sub`, `div`, `reciprocal`, `neg`, `scale`,
`abs`, `square`, `sqrt`, `max`, `min`, `table`) and `table` refuses to
extrapolate. Every operator except `table` and `sqrt` is written with plain
arithmetic, so one code path serves both scalar evaluation (production, standard
library only) and vectorised evaluation on array operands (used by the
brute-force cross-check without duplicating the constraint algebra).

**Stage 2 — grid scan.** A uniform feasibility mask on the plane, plus per-cell
binding constraint. Its area is a raw cell count and is quoted only as an
order-of-magnitude cross-check — it is first order in the cell size, so it can
never be the reported boundary.

**Stage 3 — boundary marching.** 1-D: sample the span, bracket every constraint
that changes sign, refine each bracket with Brent (bisection fallback) to
machine precision, then partition the span at the refined roots and decide each
sub-interval by one evaluation at its midpoint. Boundaries are therefore
*refined roots*, not grid nodes, and several separate intervals fall out of the
same partition. The detection is re-run on a grid twice as fine and only
accepted if the root set and the interval structure reproduce
(`verified: true`); otherwise the grid is refined again, up to a cap, and a
non-reproducible answer is reported as such.

2-D: the 1-D primitive runs line by line along one axis for a ladder of
positions along the other; intervals of neighbouring lines are linked into
components by overlap (union-find), so two disjoint regions stay two components
instead of fusing into one bounding box. A component's area is the analytic end
blocks (assuming `L²` linear near a tapered end, which locates the tip from two
knots) plus a trapezoid in between; a component that is clipped by the span, or
that has more than one interval on a line, is reported with `area_method:
trapezoid` and its degraded accuracy is visible rather than hidden.

**Stage 4 — contract checks.** Five falsifiable checks, each with its own
metrics dictionary:

* **monotonicity** — a constraint declared to move in one direction along an
  axis must not bend back (central differences at five stations; a declared but
  flat response is also a violation; stationary stations are not counted as
  folds).
* **convexity** — a constraint declared to define a convex feasible set is
  verified through a finite-difference Hessian and its eigenvalues. The check is
  the *sufficient* condition `margin Hessian negative semidefinite`; a monotone
  margin with a converged single root has a convex feasible set without meeting
  it, so such constraints simply are not declared `convex`.
* **boundary order** — the interpolated boundary error at chord midpoints under
  grid refinement must decay at the declared order (default 1.9). This is the
  check that distinguishes "the boundary moved because the physics moved" from
  "the boundary moved because the grid moved".
* **infeasible semantics** — an empty window must carry status `infeasible`, no
  interval, no component, no area, and a nearest-feasible report with a positive
  residual and the resolution of the search that produced it.
* **structure and active set** — the number of intervals/components and the
  identity of the binding constraint on each side can be declared in
  configuration and are then verified, which turns the active-set report into a
  falsifiable statement.

**Stage 5 — orchestration and artifacts.** `EnvelopeEngine.run` fixes the order
of the stages and the meaning of the status; `artifact.write_artifact` writes
byte-deterministic JSON with a code digest and an artifact digest, and
`read_artifact` recomputes the digest so a hand-edited artifact is rejected
instead of silently trusted.

## 3. Configuration schema (`tensorphysics.envelope.case.v1`)

```json
{
  "schema": "tensorphysics.envelope.case.v1",
  "case_id": "…", "title": "…", "description": "…",
  "mode": "interval | plane",
  "axes": [{"name": "…", "lo": 0.0, "hi": 1.0}],
  "constants": {"…": 0.0},
  "conditions": [{"label": "…", "params": {"…": 0.0},
                  "expect": {"intervals": 1, "infeasible": false}}],
  "constraints": [
    {"id": "…", "kind": "inequality", "side": "ge|le",
     "value": {"op": "…"},
     "active_tolerance": 1e-9,
     "contract": {"slope": {"<axis>": -1}, "convex": true}},
    {"id": "…", "kind": "discrete",
     "value": {"op": "var", "name": "<axis>"},
     "lattice": {"step": 250.0, "offset": 0.0, "tolerance": 1e-6}}
  ],
  "anchor": {"<axis>": 0.0},
  "contract": {"min_boundary_order": 1.9, "area_rel_tol": 1e-3,
               "monotone_rel_tol": 1e-2, "convexity_eig_tolerance": 1e-6,
               "expect": {"intervals": 1, "components": 2, "infeasible": false,
                          "active": {"lower": "…", "upper": "…"}}},
  "extensions": {}
}
```

Validation is fail-closed and located: unknown parameter names, a non-increasing
span, a condition that fails to provide a parameter an expression needs, a
duplicate id, an axis/constant name collision, an anchor outside its span, or a
`plane` case without exactly two axes all raise `EnvelopeConfigError` naming the
offending path. The case is hashed (SHA-256 of the canonical JSON), so a result
can be tied to the exact configuration that produced it.

A condition may carry its own `expect` block; it replaces the case-level
expectation for that condition only (an impossible loading condition can
therefore be *declared* infeasible instead of tripping the structure check that
the other conditions need). A policy passed explicitly to `EnvelopeEngine` wins
over both.

`extensions` is an opaque, digest-covered namespace for material that the engine
does not read (for example the closed-form coefficients used by the acceptance
script to cross-check the numerical path).

## 4. Zero domain semantics

Names, spans, coefficients, tolerances, monotonicity expectations and the
setting lattice are all data. The shipped configurations live in
`scenarios/envelope_cases/`, outside both the package and the test suite, and
the acceptance script counts domain-vocabulary hits in
`src/tensorphysics/envelope/*.py` and `tests/test_envelope*.py` (currently 0 over
the whole scan list).

## 5. Cost model

The cost unit is one constraint evaluation, counted by `EvalCounter` and reported
as `result.n_evals`. A 1-D window with four constraints and one condition costs
roughly `2 × 2 × n_scan × n_constraints` evaluations (the detection runs twice,
for the reproducibility check), i.e. of order 2·10³. A 2-D window at
`n_lines × n_scan` costs `n_lines × n_scan × n_constraints`. On the reference
machine 160 lines × 320 samples × 1 constraint = 65 842 evaluations in 0.43 s,
which is ~1.5·10⁵ evaluations per second on one core; the criterion of 5·10⁴
evaluations within 60 s is met with two orders of magnitude of margin.

## 6. Limitations

* **Vertical tangents.** A closed region whose boundary becomes vertical (the
  synthetic disc) has a chord error of order `h^{1/2}` near the tips, so the
  *pointwise maximum* boundary error is not a second-order quantity there. The
  order criterion is therefore stated on a window with smooth boundaries; the
  area of the same closed region is still second-order-ish (measured order
  1.2–1.5 with a small constant) and is reported separately.
* **First-order grid scan.** `grid.scan_plane` is a cell count; do not quote its
  area as a result, only as a cross-check.
* **Clipped components.** A region cut by the declared span has no analytic end
  block; its area degrades to a trapezoid (`area_method: "trapezoid"`,
  `clipped: true`), and `span_truncated` marks a 1-D window that ends on the
  span boundary rather than on a constraint.
* **Complex topology.** A component with more than one interval on the same line
  (a `C`-shaped region, for instance) is flagged `complex_topology: true` and its
  area falls back to the trapezoid.
* **Convexity is a sufficient test.** See stage 4 above.
* **Discrete feasibility is applied after the fact.** A lattice has no boundary
  to march, so it is snapped onto the continuous window and reported separately.
  The reported continuous boundary may therefore not itself be attainable — that
  is exactly why the two are reported as different fields.
* **Two axes.** The marcher is written for a plane; a 3-D window is reduced by
  slicing, not supported directly.

## 7. Running it

```bash
PYTHONPATH=src python -m pytest tests/test_envelope_engine.py tests/test_envelope_contracts.py -q
python scripts/envelope_cases_acceptance.py --out artifacts/envelope
python scripts/generate_envelope_report.py
```

The acceptance script writes `artifacts/envelope/criteria_evidence.json` (one
record per criterion, with the numbers), `artifacts/envelope/case_results.json`
(one record per case and condition) and the figures under
`artifacts/envelope/figures/`.