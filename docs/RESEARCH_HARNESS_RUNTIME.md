# Research Harness runtime

The first runtime slice is a vendor-neutral execution contract, a durable task ledger,
and a bounded single-attempt runner. It lives in TensorPhysics because it owns scientific
workflow semantics; TensorPlatform remains responsible for shared identity, catalogs,
data, artifacts, and reusable platform services.

## Implemented

- SQLite task ledger with append-only transition events.
- Explicit lifecycle transitions, optimistic expected-state checks, attempt accounting,
  and restart recovery for in-flight work.
- Vendor-neutral `AgentBackend`, `AgentRequest`, and `AgentResult` contracts.
- `ResearchTaskRunner.run_once()` to invoke one Agent attempt, persist the output,
  call a separate validator, and only mark the task succeeded when the validator returns
  `status: "passed"`.
- Human-input pause state (`waiting_input`) and bounded retry state (`retry_wait`).
- Strict JSON serialization checks rejecting NaN, infinity, and non-serializable payloads.

## Minimal usage

```python
from tensorphysics.research_runtime import ResearchTaskStore
from tensorphysics.research_runner import ResearchTaskRunner

with ResearchTaskStore("state/research.sqlite3") as store:
    store.create_task(
        "dem-benchmark-001",
        "dem-validation-goal",
        {"solver": "TensorDEM", "case": "3d-bending"},
        max_attempts=3,
    )
    runner = ResearchTaskRunner(store, agent=my_agent_backend, validator=my_independent_validator)
    task = runner.run_once(
        "dem-benchmark-001",
        goal={"objective": "validate the 3D bending benchmark"},
        workspace="/isolated/workspaces/dem-benchmark-001",
        allowed_tools=("read_files", "edit_workspace", "run_tests"),
    )
```

The adapter and validator are injected dependencies. The runtime does not itself launch
an arbitrary shell or claim that a generated plot is correct. A production adapter must
enforce workspace isolation, command/tool allowlists, timeout handling, and scoped
credentials. The validator should check numerical evidence independently of the Agent.

After a retryable failure, the scheduler may transition `retry_wait → queued` and call
`run_once()` again. The runner never retries in an unbounded loop. Tasks requiring human
input pause at `waiting_input`; an authorized operator can update the plan and transition
`waiting_input → queued`.

## Scope and next integration work

SQLite is a durable single-node starting point; multi-node deployments should replace the
store behind the same contract with a shared transactional database and queue. This slice
does not yet provide vendor-specific Codex/Claude CLI adapters, OS-level sandboxing,
leases/heartbeats, HPC scheduling, or a scientifically qualified TensorDEM campaign.
