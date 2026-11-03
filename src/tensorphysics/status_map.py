"""TPh task status -> TP mission status mapping.

The cross-repo mapping has exactly one definition:
``contracts/mission_task_status_map.v1.json`` (embedded copy of TensorPlatform
``backend/contracts/``).  Import-time validation guarantees the table covers
``TaskStatus`` exactly and every mapped target is a legal TP mission status,
so the two vocabularies cannot silently drift apart (migration_disposition §3
G-A).
"""
from __future__ import annotations

from .contracts import mission_status_contract, mission_task_status_map_contract
from .research_runtime import TaskStatus


def _build() -> dict[str, str]:
    rows = mission_task_status_map_contract()["mapping"]
    table = {row["tph"]: row["tp"] for row in rows}
    enum_values = {s.value for s in TaskStatus}
    if set(table) != enum_values:
        raise RuntimeError(
            "mission_task_status_map.v1.json does not cover TaskStatus exactly: "
            f"missing={sorted(enum_values - set(table))} "
            f"extra={sorted(set(table) - enum_values)}"
        )
    vocab = set(mission_status_contract()["statuses"])
    bad = {v for v in table.values() if v not in vocab}
    if bad:
        raise RuntimeError(f"mapped targets outside TP mission vocabulary: {sorted(bad)}")
    return table


_TABLE = _build()
_TERMINAL = frozenset(mission_status_contract()["terminal"])


def to_platform(status: "TaskStatus | str") -> str:
    """Map a TensorPhysics task status to its TensorPlatform mission status."""
    key = status.value if isinstance(status, TaskStatus) else str(status)
    try:
        return _TABLE[key]
    except KeyError:
        raise ValueError(f"unknown TensorPhysics task status: {key!r}") from None


def is_terminal_on_platform(status: "TaskStatus | str") -> bool:
    """True when the mapped TP status is terminal (registration-worthy).

    Mid-flight states (verifying / waiting_input / retry_wait) map to TP
    ``running`` and must not be registered as outcomes.
    """
    return to_platform(status) in _TERMINAL
