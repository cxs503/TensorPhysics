"""Gate registry: the acceptance gates enrolled in the falsifiability ledger.

This module is *data*: it names paths, criterion ids and upstream refs.  No
evaluation logic lives here (that stays in each gate's own module), and no
domain vocabulary is hard-coded (zero domain semantics).

Gates
-----
* ``PAPER-TRACE``  -- manuscript traceability gate.  Files live on an
  **unmerged upstream branch** of the platform repository (interface
  contract below); the ledger consumes them by path, never rewrites them.
* ``P34-G1``       -- P34 campaign integration gate (this repo).
* ``P16-G1``       -- P16 campaign integration gate (this repo).
* ``WAVE3-GATE``   -- wave-3 closure gate (this repo).
* ``WAVE4-GATE``   -- wave-4 plan-acceptance gate (this repo).

PAPER-TRACE interface contract (unmerged upstream)
--------------------------------------------------
* repo: TensorPlatform (platform repo, not this one)
* ref:  commit ``3c51f337`` (round-2 mutation audit; branch ``agent/m1994``
  at the time of the audit -- the branch ref is live and may move, the
  commit object is the stable pin)
* files consumed (read-only):
  - ``tools/paper_gate/traceability_gate.py``          (gate implementation)
  - ``tools/audit/paper_gate_mutation_audit.py``       (mutation audit tool)
  - ``backend/tests/test_paper_traceability_gate.py``  (original suite, 21 tests)
  - ``backend/tests/test_paper_traceability_gate_kills.py``   (round-1 killers)
  - ``backend/tests/test_paper_traceability_gate_kills_v2.py`` (round-2 killers)
* consumed API: ``enumerate_mutants(src) -> list[Mutant]``,
  ``apply_mutant(src, mutant) -> str``, ``Mutant`` dataclass
  (op/index/line/snippet/patch/killed/detail/extra)
* audit history recorded in that tree:
  ``artifacts/paper_gate_mutation/audit_v{1,2,3}.json``
  (92/186 -> 111/186 -> 172/186)
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional

from .adapter import Criterion, GateSpec, Ledger

#: PAPER-TRACE upstream pin (see module docstring for the contract).
PAPER_TRACE_UPSTREAM_REPO = "TensorPlatform"
PAPER_TRACE_UPSTREAM_REF = "3c51f337"

PAPER_TRACE_MODULE = Path("tools/paper_gate/traceability_gate.py")
PAPER_TRACE_TESTS = (
    Path("backend/tests/test_paper_traceability_gate.py"),
    Path("backend/tests/test_paper_traceability_gate_kills.py"),
    Path("backend/tests/test_paper_traceability_gate_kills_v2.py"),
)
PAPER_TRACE_AUDIT_TOOL = Path("tools/audit/paper_gate_mutation_audit.py")

#: PAPER-TRACE criteria: PG code -> (check function name, one-line label).
PAPER_TRACE_CRITERIA = (
    ("PG1", "check_paths", "phantom path: backticked path must resolve"),
    ("PG2", "check_report_refs", "phantom report: cited report id in ledger"),
    ("PG3", "check_figures", "figure ref: image link resolves, no zero-citation"),
    ("PG4", "check_table_arithmetic", "table arithmetic: error column recomputed"),
    ("PG5", "check_inline_errors", "inline error: prose claim recomputed"),
    ("PG6", "check_anchors", "anchor mismatch: claim anchor matches value"),
    ("PG7", "artifact_provenance", "artifact provenance: results cite artifacts"),
    ("PG8", "check_report_values", "value in source: numbers present in source"),
)

ALL_CLASSES = ("native", "M1_numeric_perturb", "M2_unit_mismatch",
               "M3_constant_substitute", "M4_sample_disguise",
               "M5_reference_selfref")


def _load_script_module(path: Path):
    import sys as _sys
    name = "paper_trace_gate_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    _sys.modules[name] = mod  # so the module is importable by name
    spec.loader.exec_module(mod)
    return mod


def build_default_ledger(repo_root: Path,
                         tp_root: Optional[Path] = None) -> Ledger:
    """Build the default ledger.

    ``repo_root`` -- root of this repo's worktree (in-repo gates).
    ``tp_root``   -- root of the pinned upstream tree holding the
      PAPER-TRACE gate (None -> the gate is registered but its criteria
      are not instantiable: honest WITHHELD, never a silent pass).
    """
    repo_root = Path(repo_root)
    ledger = Ledger()

    # ------------------------------------------------------------------ P34
    from tensorphysics.campaign.p34_gate import criteria as p34c
    p34 = GateSpec(
        gate_id="P34-G1",
        title="P34 campaign integration gate",
        module=Path("src/tensorphysics/campaign/p34_gate/gate.py"),
        tests=(Path("tests/test_p34_integration.py"),),
        mutant_classes=ALL_CLASSES,
        evidence_ref="src/tensorphysics/campaign/p34_gate/spec.py",
        extra_copies=(Path("src"),),
    )
    ledger.register_gate(p34)
    for cid, fn in (("C1", p34c.evaluate_c1), ("C2", p34c.evaluate_c2),
                      ("C3", p34c.evaluate_c3), ("C4", p34c.evaluate_c4),
                      ("C5", p34c.evaluate_c5)):
        ledger.register_criterion(Criterion(
            id=cid, gate="P34-G1", callable=fn, expected_verdict="PASS",
            evidence_ref="src/tensorphysics/campaign/p34_gate/spec.py"))

    # ------------------------------------------------------------------ P16
    from tensorphysics.campaign.integration_gate import gate as p16g
    p16 = GateSpec(
        gate_id="P16-G1",
        title="P16 campaign integration gate",
        module=Path("src/tensorphysics/campaign/integration_gate/gate.py"),
        tests=(Path("tests/test_integration_gate.py"),),
        mutant_classes=ALL_CLASSES,
        evidence_ref="src/tensorphysics/campaign/integration_gate/gate.py",
        extra_copies=(Path("src"),),
    )
    ledger.register_gate(p16)
    for cid, label in (("a", "node acceptance criteria evaluated"),
                       ("b", "cross-domain interface contracts validated"),
                       ("c", "regression probe delta within threshold"),
                       ("d", "campaign verdict report with provenance")):
        ledger.register_criterion(Criterion(
            id=cid, gate="P16-G1", callable=p16g.run_gate,
            expected_verdict="PASS",
            evidence_ref=f"src/tensorphysics/campaign/integration_gate/gate.py::{label}"))

    # ---------------------------------------------------------------- WAVE3
    from tensorphysics.campaign import wave3_gate as w3
    w3g = GateSpec(
        gate_id="WAVE3-GATE",
        title="wave-3 closure gate",
        module=Path("src/tensorphysics/campaign/wave3_gate.py"),
        tests=(Path("tests/test_wave3_gate.py"),),
        mutant_classes=ALL_CLASSES,
        evidence_ref="src/tensorphysics/campaign/wave3_gate.py",
        extra_copies=(Path("src"),),
    )
    ledger.register_gate(w3g)
    for cid, label in (("C1", "matrix coverage"),
                       ("C2", "root-cause completeness"),
                       ("C3", "lineage explicitness"),
                       ("C4", "error table"),
                       ("C5", "peer review")):
        ledger.register_criterion(Criterion(
            id=cid, gate="WAVE3-GATE", callable=w3.build_report,
            expected_verdict="PASS",
            evidence_ref=f"src/tensorphysics/campaign/wave3_gate.py::{label}"))

    # ---------------------------------------------------------------- WAVE4
    from tensorphysics.campaign import wave4_gate as w4
    w4g = GateSpec(
        gate_id="WAVE4-GATE",
        title="wave-4 plan-acceptance gate",
        module=Path("src/tensorphysics/campaign/wave4_gate.py"),
        tests=(Path("tests/test_wave4_gate.py"),),
        mutant_classes=ALL_CLASSES,
        evidence_ref="src/tensorphysics/campaign/wave4_gate.py",
        extra_copies=(Path("src"),),
    )
    ledger.register_gate(w4g)
    for cid, fn, label in (("plan_acceptance", w4.decide_plan,
                           "plan acceptance decision"),
                          ("report_integrity", w4.verify_report_integrity,
                           "report integrity re-derivation"),
                          ("guide_gaps", w4.guide_gaps_from_rows,
                           "guide-gap detection")):
        ledger.register_criterion(Criterion(
            id=cid, gate="WAVE4-GATE", callable=fn,
            expected_verdict="PASS",
            evidence_ref=f"src/tensorphysics/campaign/wave4_gate.py::{label}"))

    # ------------------------------------------------------------- PAPER-TRACE
    pt = GateSpec(
        gate_id="PAPER-TRACE",
        title="manuscript traceability gate (upstream, unmerged)",
        module=PAPER_TRACE_MODULE,
        tests=PAPER_TRACE_TESTS,
        mutant_classes=ALL_CLASSES,
        evidence_ref="artifacts/criterion_ledger_m5618/paper_trace_c2_full.json",
        upstream_repo=PAPER_TRACE_UPSTREAM_REPO,
        upstream_ref=PAPER_TRACE_UPSTREAM_REF,
        interface_contract=(
            "TensorPlatform@3c51f337 (unmerged): tools/paper_gate/"
            "traceability_gate.py + tools/audit/paper_gate_mutation_audit.py "
            "+ 3 test files; consumed by path, never rewritten"),
    )
    ledger.register_gate(pt)
    if tp_root is not None:
        tp_root = Path(tp_root)
        gate_mod = _load_script_module(tp_root / PAPER_TRACE_MODULE)
        for cid, fn_name, label in PAPER_TRACE_CRITERIA:
            ledger.register_criterion(Criterion(
                id=cid, gate="PAPER-TRACE",
                callable=getattr(gate_mod, fn_name),
                expected_verdict="PASS",
                evidence_ref=(f"{PAPER_TRACE_UPSTREAM_REPO}@{PAPER_TRACE_UPSTREAM_REF}"
                              f":tools/paper_gate/traceability_gate.py::{label}")))
    else:
        for cid, _fn_name, label in PAPER_TRACE_CRITERIA:
            ledger.register_criterion(Criterion(
                id=cid, gate="PAPER-TRACE", callable=None,
                expected_verdict="PASS",
                evidence_ref=(f"{PAPER_TRACE_UPSTREAM_REPO}@{PAPER_TRACE_UPSTREAM_REF}"
                              f" (upstream tree not provided: WITHHELD)::{label}")))
    return ledger
