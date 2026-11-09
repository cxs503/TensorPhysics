"""Lineage-inheritance debt ledger: prose inheritance turned into a recomputable gate.

The campaign used to inherit its limitations by *writing a paragraph* ("this wave
inherits the parent wave's honest FAILs and KPI gaps").  Nothing could check it:
a child wave could be seeded while silently dropping every gap of its parent.
This package makes the inheritance contract machine-checkable in five steps.

``extract``
    report error tables (``|err| > threshold``) and terminal nodes
    (``blocked``/``failed``) become debt items carrying
    ``domain / quantity / err / report_id / lineage``.
``criteria``
    a child node's acceptance criteria are read from structured metadata first
    and by regex fallback, canonicalised through the injected vocabulary.
``coverage``
    every debt is matched against the descendant criteria ->
    ``covered`` / ``covered_by_domain`` / ``deferred`` (with reason) /
    ``dropped``.
``gate``
    a dropped debt without a written reason blocks seeding (fail-closed) and
    enumerates the missing items; alternatively the plan metadata receives
    ``autonomy.debt_unpaid`` so the debt stays visible.
``replay`` / ``query``
    the ledger is a deterministic pure function of a campaign snapshot, and it
    is queryable per wave, plan, domain, state and coverage outcome.

Design rules
------------
* **zero domain semantics** -- domain names, quantity aliases, header tokens
  and thresholds all live in ``configs/lineage_debt.vocab.json``; ``selfcheck``
  proves the logic carries none of them.
* **determinism** -- canonical JSON everywhere, sorted collections, no clock,
  no network: the same snapshot replays to the same bytes.
* **honest accounting** -- a debt is closed only by a *later* in-threshold
  observation of the same key, never by a claim; an unknown quantity is kept
  with its raw label instead of being dropped.
"""
from __future__ import annotations

from .config import DEFAULT_CONFIG_PATH, VocabConfig, load_config
from .coverage import CoverageEntry, build_matrix, match_debt, resolve_deferral
from .criteria import Criterion, parse_node_criteria
from .extract import (ErrorObservation, extract_node_terminal_debts,
                      extract_report_observations, observations_to_debts,
                      parse_markdown_tables)
from .gate import SeedingBlocked, SeedingVerdict, apply_autonomy_patch, enforce, seeding_check
from .model import LEDGER_SCHEMA, DebtItem, canonical_json, dedupe_debts, sha256_text, summarize
from .query import coverage_table, gate_report, load_ledger, query
from .replay import (BUNDLE_SCHEMA, LedgerBundle, build_ledger, load_bundle, plan_chain,
                     plan_debts, write_ledger)
from .selfcheck import selfcheck

__all__ = [
    "BUNDLE_SCHEMA",
    "DEFAULT_CONFIG_PATH",
    "CoverageEntry",
    "Criterion",
    "DebtItem",
    "ErrorObservation",
    "LEDGER_SCHEMA",
    "LedgerBundle",
    "SeedingBlocked",
    "SeedingVerdict",
    "VocabConfig",
    "apply_autonomy_patch",
    "build_ledger",
    "build_matrix",
    "canonical_json",
    "coverage_table",
    "dedupe_debts",
    "enforce",
    "extract_node_terminal_debts",
    "extract_report_observations",
    "gate_report",
    "load_bundle",
    "load_config",
    "load_ledger",
    "match_debt",
    "observations_to_debts",
    "parse_markdown_tables",
    "parse_node_criteria",
    "plan_debts",
    "plan_chain",
    "query",
    "resolve_deferral",
    "seeding_check",
    "selfcheck",
    "sha256_text",
    "summarize",
    "write_ledger",
]