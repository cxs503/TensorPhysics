"""Criterion falsifiability ledger (P96-N9).

Upgrades gate falsifiability from single-gate sampling to a full-ledger KPI +
gate:

* :mod:`~tensorphysics.campaign.criterion_ledger.adapter` -- the ledger
  (``criterion = {id, gate, callable, expected_verdict, evidence_ref}``);
* :mod:`~tensorphysics.campaign.criterion_ledger.mutants` -- mutant classes
  M1..M5 (numeric perturbation / unit mismatch / constant substitution /
  sample disguise / reference self-reference);
* :mod:`~tensorphysics.campaign.criterion_ledger.runner` -- scratch-tree
  mutation runner with cluster sharding (consumes the upstream mutation-audit
  tool, never rewrites it);
* :mod:`~tensorphysics.campaign.criterion_ledger.kpi` -- per-gate kill rate,
  weakest-mutant list, monotonic kill curve, falsifiability marking;
* :mod:`~tensorphysics.campaign.criterion_ledger.pr_gate` -- new-criterion PR
  check: a criterion without a killer test is rejected.

Zero domain semantics: gate ids, paths, verdicts and mutant parameters are
data; no domain vocabulary is hard-coded in this package's logic.
"""
from .adapter import Criterion, GateSpec, Ledger, probe_instantiable
from .gates import build_default_ledger
from .kpi import (
    falsifiability,
    kill_rate,
    ledger_kpi,
    monotonic_kill_curve,
    weakest_mutants,
)
from .pr_gate import check_new_criterion
from .runner import (
    aggregate,
    aggregate_full_ledger,
    enumerate_gate_mutants,
    load_audit_module,
    run_full_ledger_shard,
    run_gate,
    shard_slice,
)

__all__ = [
    "Criterion",
    "GateSpec",
    "Ledger",
    "probe_instantiable",
    "build_default_ledger",
    "falsifiability",
    "kill_rate",
    "ledger_kpi",
    "monotonic_kill_curve",
    "weakest_mutants",
    "check_new_criterion",
    "aggregate",
    "aggregate_full_ledger",
    "run_full_ledger_shard",
    "enumerate_gate_mutants",
    "load_audit_module",
    "run_gate",
    "shard_slice",
]
