"""Regression tests for preserving parent-run lineage across the platform adapter."""
from __future__ import annotations

from pathlib import Path
import sys

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))
import research_loop_dem_platform as adapter  # noqa: E402


def test_child_invocation_forwards_all_parent_run_ids(tmp_path: Path) -> None:
    command = adapter._build_demo_command(
        EXAMPLES / "research_loop_dem.py",
        tmp_path,
        "campaign-r02",
        8,
        3,
        20,
        parent_run_ids=["campaign-r01", "external-seed-01"],
    )

    assert command.count("--parent-run-id") == 2
    first = command.index("--parent-run-id")
    second = command.index("--parent-run-id", first + 1)
    assert command[first + 1] == "campaign-r01"
    assert command[second + 1] == "external-seed-01"


def test_parent_run_id_rejects_blank_or_padded_values(tmp_path: Path) -> None:
    for parent in ("", " parent "):
        try:
            adapter._build_demo_command(
                EXAMPLES / "research_loop_dem.py",
                tmp_path,
                "child",
                7,
                3,
                20,
                parent_run_ids=[parent],
            )
        except ValueError as exc:
            assert "parent run IDs" in str(exc)
        else:
            raise AssertionError(f"expected invalid parent run ID to be rejected: {parent!r}")
