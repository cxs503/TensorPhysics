# Executable Research Methodology (v1)

## Purpose

TensorPhysics should not rely on a low-cost execution LLM to remember an entire
scientific methodology in its prompt. Stable, repeatable parts of the method
belong in versioned contracts, deterministic gates, and auditable protocols.
The LLM proposes plans and explains unexpected observations; the platform
enforces budgets, required evidence, retry limits, and promotion rules.

This is a policy kernel, not a claim that scientific discovery has been fully
automated. Passing software gates is not proof of physical correctness.

## Control loop

1. **Specify the question** — a human-owned `ResearchGoal` defines objective,
   acceptance criteria, constraints, compute budget, and approval boundaries.
2. **Plan one bounded experiment** — an `ExperimentPlan` declares solver,
   parameters, required independent checks, and maximum run count.
3. **Execute reproducibly** — assign a durable run ID; record normalized
   configuration, code revisions, environment, random seeds, logs, and outputs.
4. **Validate independently** — validators produce a `ValidationReport`.
   Solver exit status and finite output alone are not sufficient.
5. **Apply policy** — `decide_next_action` evaluates required check names and
   the declared budget. The policy fails closed when checks are absent or fail.
6. **Retain evidence** — save decision reasons with report/artifact references.
   Only explicitly qualified outputs may be proposed for dataset promotion.
7. **Escalate when uncertain** — ambiguous or novel failures go to a stronger
   reasoner or human review instead of unbounded retries.

## Decision semantics

| Condition | Action | Meaning |
|---|---|---|
| Report passed and all required checks passed; runs remain | `run_next_bounded_experiment` | A bounded follow-up is allowed, not automatically launched |
| Report passed and all required checks passed; budget exhausted | `accept_validated_result` | Accept only against the declared checks |
| Any required check missing/failed or report not passed | `stop_and_repair` | Do not promote; diagnose before retry |
| Failed/incomplete gate and run budget exhausted | `stop_budget_exhausted` | Stop; do not exceed the declared budget |
| Ambiguous or novel failure | `escalate_to_stronger_reasoner` | Request diagnosis before retry |

The policy returns a decision record and does not execute tools by itself.
A worker/orchestrator remains responsible for honoring the action, permissions,
timeouts, cancellation, and resource limits.

## What can be formalized vs. what cannot

**Good candidates for deterministic enforcement:** schema and unit checks;
finite values; monotonic clocks; conservation tolerances; required benchmark
metrics; artifact checksums; provenance completeness; resource budgets; retry
limits; duplicate-run handling; evidence status; stop/escalation conditions.

**Needs domain-specific validators or expert review:** whether a physical model
is adequate, whether a benchmark is appropriate, whether agreement is meaningful,
whether an observed effect supports a scientific claim, and whether a new
hypothesis is worth testing. The policy can require evidence for these decisions
but cannot manufacture the evidence.

## Example

```python
from tensorphysics.research_contracts import ValidationReport
from tensorphysics.research_policy import decide_next_action

report = ValidationReport(
    run_id="drop-impact-001",
    status="passed",
    checks=(
        {"name": "finite_values", "passed": True},
        {"name": "mass_conservation", "passed": True},
        {"name": "benchmark_error_within_tolerance", "passed": True},
    ),
    summary="Independent benchmark gates passed",
)
decision = decide_next_action(
    report,
    required_checks=(
        "finite_values",
        "mass_conservation",
        "benchmark_error_within_tolerance",
    ),
    runs_completed=3,
    max_runs=3,
)
assert decision.action == "accept_validated_result"
```

The names above are illustrative. A validator must compute each result from
actual run artifacts and a versioned benchmark definition; never set a check to
passed just to satisfy the policy.

## Acceptance tests for this increment

- Missing required check cannot be treated as success.
- A failed or incomplete report cannot be accepted.
- Retry permission cannot exceed the declared maximum number of runs.
- Ambiguous failures escalate rather than trigger blind retries.
- Every returned action contains reasons and the list of passed, missing, and
  failed required checks.

## Next integration step

Wire the policy into the persistent worker only after worker lifecycle and
failure semantics are covered by tests. Then add domain-specific gate packs for
the droplet-impact LBM benchmark and the bonded-particle DEM benchmark. Keep
scientific acceptance thresholds with the benchmark definitions and validators,
not in a generic LLM prompt.
