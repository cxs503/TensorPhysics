"""Revival scheduler: a landed blocker spawns at most one task per criterion.

Trigger contract
----------------
* a revival task is emitted on the **first tick after** its blocker reads
  ``done`` (latency ``<= 1`` scheduler cycle; a blocker that landed before the
  tick is emitted in that very tick, latency 0);
* a blocker that is not ``done`` emits nothing -- the false-trigger count of a
  state where nothing landed is exactly zero;
* task identity is ``sha256(criterion_digest | input_digest)``, so re-ticking
  the same state (or re-landing the same blocker) cannot duplicate work;
* the task body is digests only: no threshold, no tolerance, no verdict.

The scheduler is a pure function of (graph, state, already-emitted tasks); it
keeps no hidden mutable state between ticks beyond the emitted-task store, which
is part of the input of the next tick.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .blockers import BlockerGraph, BlockerState, blocker_done, blocker_of
from .model import FailRecord, RevivalTask, task_key

SCHEDULER_SCHEMA = "revival_scheduler.v1"


def baseline_input_digest(record: FailRecord) -> str:
    """Digest of the run context that produced the original measurement.

    It is *not* a data fingerprint of the raw inputs (the artifact does not
    contain them); it is the identity of the archived evidence
    (mission, report digest, criterion digest), which is what a revival must
    re-open.  Declared baselines may override it.
    """
    from .model import sha256_obj

    return sha256_obj({
        "mission_id": record.mission_id,
        "report_sha256": record.report_sha256,
        "criterion_sha256": record.criterion_sha256,
    })


@dataclass
class EmittedTasks:
    """Task store: the only state the scheduler carries between ticks."""

    tasks: dict[str, RevivalTask] = field(default_factory=dict)
    duplicates_blocked: int = 0
    ticks: int = 0

    def has(self, key: str) -> bool:
        return key in self.tasks

    def add(self, task: RevivalTask) -> bool:
        if task.task_key in self.tasks:
            self.duplicates_blocked += 1
            return False
        self.tasks[task.task_key] = task
        return True

    def ordered(self) -> tuple[RevivalTask, ...]:
        return tuple(self.tasks[k] for k in sorted(self.tasks))


@dataclass(frozen=True)
class TickReport:
    cycle: int
    emitted: tuple[RevivalTask, ...]
    skipped_done: tuple[str, ...]
    state: Mapping[str, Any]
    schema: str = SCHEDULER_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "cycle": self.cycle,
            "emitted": [t.to_dict() for t in self.emitted],
            "skipped_done": list(self.skipped_done),
            "state": dict(self.state),
        }


def tick(graph: BlockerGraph, state: BlockerState, store: EmittedTasks, *,
         cycle: int | None = None,
         input_digests: Mapping[str, str] | None = None) -> TickReport:
    """Advance the scheduler one cycle and emit the newly unblocked tasks."""
    store.ticks += 1
    if cycle is None:
        cycle = store.ticks
    emitted: list[RevivalTask] = []
    skipped: list[str] = []
    for rec in graph.records:
        blocker = blocker_of(rec.fail_id, graph.blockers)
        if blocker is None or blocker.kind == "undeclared":
            continue
        if not blocker_done(blocker, state):
            continue
        digest = (input_digests or {}).get(rec.fail_id) or baseline_input_digest(rec)
        key = task_key(rec.criterion_sha256, digest)
        task = RevivalTask(
            task_key=key,
            fail_key=rec.fail_id,
            criterion_sha256=rec.criterion_sha256,
            input_sha256=digest,
            blocker_id=blocker.blocker_id,
            blocker_kind=blocker.kind,
            mission_id=rec.mission_id,
            cycle=cycle,
        )
        if store.add(task):
            emitted.append(task)
        else:
            skipped.append(rec.fail_id)
    return TickReport(cycle=cycle, emitted=tuple(emitted), skipped_done=tuple(sorted(skipped)),
                      state=state.to_dict())


def audit_triggers(graph: BlockerGraph, states: Sequence[BlockerState],
                   reports: Sequence[TickReport]) -> dict[str, Any]:
    """Independently re-check every emitted task against the state of its cycle.

    A *false trigger* is an emitted task whose blocker is not ``done`` in the
    state that produced it.  A *missed trigger* is a FAIL whose blocker is done
    but which produced no task.  Both are counted; both must be zero.
    """
    state_by_cycle = {i + 1: s for i, s in enumerate(states)}
    false_triggers: list[dict[str, Any]] = []
    emitted_fails: set[str] = set()
    for rep in reports:
        state = state_by_cycle.get(rep.cycle)
        if state is None:
            continue
        for task in rep.emitted:
            emitted_fails.add(task.fail_key)
            blocker = graph.blockers.get(task.blocker_id)
            if blocker is None or not blocker_done(blocker, state):
                false_triggers.append({"cycle": rep.cycle, "fail_key": task.fail_key,
                                       "blocker_id": task.blocker_id})
    ever_done = set(unblocked_fails(graph, states[-1])) if states else set()
    missed = sorted(f for f in ever_done if f not in emitted_fails)
    return {
        "cycles": len(reports),
        "tasks": sum(len(r.emitted) for r in reports),
        "false_triggers": false_triggers,
        "false_trigger_count": len(false_triggers),
        "missed": missed,
        "missed_count": len(missed),
    }


def unblocked_fails(graph: BlockerGraph, state: BlockerState) -> tuple[str, ...]:
    out = []
    for rec in graph.records:
        blocker = blocker_of(rec.fail_id, graph.blockers)
        if blocker is not None and blocker_done(blocker, state):
            out.append(rec.fail_id)
    return tuple(sorted(out))


def run_cycles(graph: BlockerGraph, states: Sequence[BlockerState], *,
               input_digests: Mapping[str, str] | None = None) -> list[TickReport]:
    store = EmittedTasks()
    return [tick(graph, s, store, cycle=i + 1, input_digests=input_digests)
            for i, s in enumerate(states)]


def latency_cycles(graph: BlockerGraph, landed_at: Mapping[str, int], reports: Sequence[TickReport],
                   input_digests: Mapping[str, str] | None = None) -> dict[str, int]:
    """Cycle distance between 'blocker landed' and 'task emitted' per FAIL."""
    emitted_at = {}
    for rep in reports:
        for task in rep.emitted:
            emitted_at.setdefault(task.fail_key, rep.cycle)
    out: dict[str, int] = {}
    for fail_key, when in landed_at.items():
        if fail_key in emitted_at:
            out[fail_key] = emitted_at[fail_key] - when
    return out