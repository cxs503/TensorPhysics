"""Criterion ledger adapter: the falsifiability ledger for acceptance gates.

A *criterion* is one machine-checkable acceptance statement owned by a gate::

    criterion = {id, gate, callable, expected_verdict, evidence_ref}

* ``id``               -- stable identifier inside the gate (e.g. "PG4", "C1")
* ``gate``             -- ``gate_id`` of the owning gate
* ``callable``         -- the function that evaluates the criterion
* ``expected_verdict`` -- verdict the criterion must produce on the gate's
                          reference evidence ("PASS" / "FAIL")
* ``evidence_ref``     -- path/pointer to the evidence the verdict rests on

Design rules
------------
* **data-driven** -- gates and criteria are registered as *data*; the
  evaluation logic stays in each gate's own module (consumed, never
  rewritten);
* **zero domain semantics** -- gate ids, paths, verdicts and mutant
  parameters are data, not logic; nothing in this package hard-codes a
  domain vocabulary;
* **honest accounting** -- a criterion that cannot be instantiated is
  reported as such (``instantiable: false`` + reason), never silently
  dropped.

Instantiability (machine definition)
------------------------------------
A criterion is *instantiable* iff its ``callable`` is (1) callable,
(2) has a resolvable signature, and (3) lives in an importable module.
That is the weakest property a falsifiability audit needs: the criterion
can be bound, mutated and re-evaluated.
"""
from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

PASS, FAIL, WITHHELD = "PASS", "FAIL", "WITHHELD"


@dataclass(frozen=True)
class Criterion:
    """One machine-checkable acceptance statement of a gate."""

    id: str
    gate: str
    callable: Callable[..., Any]
    expected_verdict: str
    evidence_ref: str

    def as_dict(self) -> dict:
        fn = self.callable
        where = (getattr(fn, "__module__", "?") + "."
                 + getattr(fn, "__qualname__", "?"))
        return {
            "id": self.id,
            "gate": self.gate,
            "callable": where,
            "expected_verdict": self.expected_verdict,
            "evidence_ref": self.evidence_ref,
        }


@dataclass(frozen=True)
class GateSpec:
    """Declarative description of one acceptance gate.

    ``module``/``tests`` are paths *relative to the gate's work root* (the
    directory that contains the repo-relative layout the test suite expects).
    ``upstream_repo``/``upstream_ref`` document an unmerged upstream branch
    the gate's files live on (interface contract, see ``interface_contract``).
    """

    gate_id: str
    title: str
    module: Path
    tests: tuple[Path, ...]
    mutant_classes: tuple[str, ...] = ("native",)
    evidence_ref: str = ""
    upstream_repo: str | None = None
    upstream_ref: str | None = None
    interface_contract: str = ""
    #: extra paths (relative to the work root) copied into the scratch tree
    #: (e.g. the package root + conftest for in-repo package imports)
    extra_copies: tuple = ()

    def as_dict(self) -> dict:
        return {
            "gate_id": self.gate_id,
            "title": self.title,
            "module": str(self.module),
            "tests": [str(t) for t in self.tests],
            "mutant_classes": list(self.mutant_classes),
            "evidence_ref": self.evidence_ref,
            "upstream_repo": self.upstream_repo,
            "upstream_ref": self.upstream_ref,
            "interface_contract": self.interface_contract,
            "extra_copies": [str(e) for e in self.extra_copies],
        }


class Ledger:
    """Registry of gates and their criteria (the falsifiability ledger)."""

    def __init__(self) -> None:
        self._gates: dict[str, GateSpec] = {}
        self._criteria: list[Criterion] = []

    # -- registration -----------------------------------------------------
    def register_gate(self, spec: GateSpec) -> None:
        if spec.gate_id in self._gates:
            raise ValueError(f"gate already registered: {spec.gate_id}")
        self._gates[spec.gate_id] = spec

    def register_criterion(self, c: Criterion) -> None:
        if c.gate not in self._gates:
            raise KeyError(f"unknown gate: {c.gate}")
        if any(x.id == c.id and x.gate == c.gate for x in self._criteria):
            raise ValueError(f"duplicate criterion {c.gate}/{c.id}")
        self._criteria.append(c)

    # -- queries -----------------------------------------------------------
    def gate(self, gate_id: str) -> GateSpec:
        return self._gates[gate_id]

    def gates(self) -> tuple[GateSpec, ...]:
        return tuple(self._gates.values())

    def criteria(self, gate_id: str | None = None) -> tuple[Criterion, ...]:
        return tuple(c for c in self._criteria
                     if gate_id is None or c.gate == gate_id)

    # -- KPI inputs ---------------------------------------------------------
    def instantiability(self) -> dict[str, dict]:
        """Per-criterion instantiability (see module docstring)."""
        out: dict[str, dict] = {}
        for c in self._criteria:
            ok, reason = probe_instantiable(c)
            out[f"{c.gate}/{c.id}"] = {"instantiable": ok, "reason": reason}
        return out

    def summary(self) -> dict:
        inst = self.instantiability()
        n_ok = sum(1 for v in inst.values() if v["instantiable"])
        total = len(self._criteria)
        return {
            "gates": len(self._gates),
            "criteria": total,
            "instantiable": n_ok,
            "instantiability_rate": round(n_ok / total, 4) if total else 0.0,
            "per_criterion": inst,
        }


def probe_instantiable(c: Criterion) -> tuple[bool, str]:
    """Machine-checkable instantiability probe for one criterion."""
    fn = c.callable
    if not callable(fn):
        return False, "not callable"
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        return False, f"signature not resolvable: {exc}"
    modname = getattr(fn, "__module__", None)
    if modname:
        try:
            importlib.import_module(modname)
        except Exception as exc:  # noqa: BLE001 - any import failure counts
            return False, f"module {modname!r} not importable: {exc}"
    return True, f"callable, {len(sig.parameters)} parameter(s)"
