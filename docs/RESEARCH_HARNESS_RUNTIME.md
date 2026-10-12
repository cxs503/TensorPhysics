# Research Harness runtime: first implementation slice

This document describes the durable task runtime in `tensorphysics.research_runtime`.
It is intentionally a small, vendor-neutral foundation rather than a new general-purpose
Agent framework.

## Implemented

- SQLite task ledger with durable task records and append-only state-transition events.
- Explicit state transitions: `queued → claimed → running → verifying → succeeded`, with
  bounded retry and cancellation paths.
- Expected-state checks inside a write transaction to reject stale worker transitions.
- Attempt accounting and restart recovery for tasks left claimed/running/verifying.
- JSON validation that rejects NaN, infinity, non-serializable plans, results, and event payloads.
- Vendor-neutral `AgentBackend`, `AgentRequest`, and `AgentResult` contracts. An adapter can
  target Codex, Claude Code, or another engine without changing the task ledger.

## Minimal usage

```python
from tensorphysics.research_runtime import ResearchTaskStore, TaskStatus

with ResearchTaskStore("state/research.sqlite3") as store:
    store.create_task(
        "dem-benchmark-001",
        "dem-validation-goal",
        {"solver": "TensorDEM", "case": "3d-bending", "acceptance": ["finite outputs"]},
        max_attempts=3,
    )
    store.transition(
        "dem-benchmark-001",
        expected_status=TaskStatus.QUEUED,
        new_status=TaskStatus.CLAIMED,
        event_type="worker_claimed",
    )
```

A worker increments the attempt counter when it transitions from `claimed` to `running`.
On process restart, call `recover_interrupted()`; in-flight tasks become `retry_wait`
if attempts remain, otherwise `failed`. A worker can then move `retry_wait → queued`
before a new claim.

## Deliberate boundaries

- The ledger records lifecycle state; it does not execute arbitrary shell commands.
- An Agent's `completed` result does not mean a scientific task passed verification.
  The task should enter `verifying`, and only a validator should decide whether it can
  transition to `succeeded`.
- SQLite is suitable for a single-node first slice. Multi-node workers should later use
  a shared transactional database/queue behind the same repository interface.
- CLI/SDK adapters, sandbox enforcement, leases/heartbeats, resource quotas, and the
  first fully autonomous Agent-to-TensorDEM campaign remain follow-up work.
