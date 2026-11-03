"""Embedded copies of TensorPlatform's backend/contracts (single source of truth).

The three ``*.v1.json`` files beside this module are byte-identical copies of
``cxs503/TensorPlatform@main:backend/contracts/``.  Upstream renaming a
contract, editing a vocabulary, or bumping ``version`` requires copying the
file here in the same change: TensorPlatform's
``backend/tests/test_status_contracts.py`` and this repo's
``tests/test_platform_status_contracts.py`` each lock their side, so a
one-sided edit fails one of the two suites (migration_disposition §3 G-A/G-C).
"""
from __future__ import annotations

import json
from pathlib import Path

_CONTRACTS_DIR = Path(__file__).resolve().parent


def _load(filename: str, expected: str) -> dict:
    doc = json.loads((_CONTRACTS_DIR / filename).read_text())
    if doc.get("contract") != expected or not isinstance(doc.get("version"), int):
        raise RuntimeError(f"contract file {filename} malformed (contract/version)")
    return doc


def mission_status_contract() -> dict:
    return _load("mission_status_vocab.v1.json", "mission_status_vocab")


def research_run_contract() -> dict:
    return _load("research_run_status_vocab.v1.json", "research_run_status_vocab")


def mission_task_status_map_contract() -> dict:
    return _load("mission_task_status_map.v1.json", "mission_task_status_map")
