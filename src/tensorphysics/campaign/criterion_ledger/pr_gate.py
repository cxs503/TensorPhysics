"""New-criterion PR gate: no killer test, no merge.

Rule (P96-N9): *any new criterion added to an acceptance gate must carry
at least one mutant test that can kill it.*  A criterion whose check survives
every mutant is indistinguishable from a tautology -- the suite cannot
notice the gate stopping checking -- so the PR is rejected.

:func:`check_new_criterion` is the machine check:

1. enumerate the mutants of the criterion's module (all requested classes);
2. run *only* the proposed killer tests against each mutant;
3. accept iff at least one mutant is killed.

The check is deliberately strict in both directions:

* ``killer_tests`` empty            -> rejected (nothing can kill it);
* killer tests that never fail      -> rejected (they kill nothing);
* at least one mutant killed        -> accepted, with the kill count.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Sequence

from . import mutants as M
from .runner import (
    _common_root,
    _run_pytest,
    apply_gate_mutant,
    build_scratch_tree,
    enumerate_gate_mutants,
    load_audit_module,
)


def check_new_criterion(module_src: str, module_name: str,
                        killer_tests: Sequence[Path],
                        work_root: Path, *,
                        classes: Sequence[str] = (M.M1, M.M2, M.M3, M.M4, M.M5),
                        audit_tool: Optional[Path] = None,
                        jobs: int = 4, timeout: int = 120,
                        max_mutants: int = 60) -> dict:
    """PR gate for one new criterion.

    ``module_src`` is the source of the module implementing the criterion's
    check; ``killer_tests`` are the test files the PR proposes as the
    criterion's killer tests.  Returns::

        {"accepted": bool, "total": int, "killed": int,
         "kill_rate": float, "reason": str, "killed_mutants": [...]}
    """
    if not killer_tests:
        return {"accepted": False, "total": 0, "killed": 0, "kill_rate": 0.0,
                "reason": "no killer test provided with the new criterion",
                "killed_mutants": []}

    classes = tuple(classes)
    audit_mod = (load_audit_module(audit_tool)
                 if M.NATIVE in classes and audit_tool is not None else None)
    mutants = enumerate_gate_mutants(module_src, classes, audit_mod)
    if not mutants:
        return {"accepted": False, "total": 0, "killed": 0, "kill_rate": 0.0,
                "reason": "no mutable sites found in the criterion module",
                "killed_mutants": []}
    mutants = mutants[:max_mutants]

    # scratch layout: module + killer tests under a common root
    base = Path(tempfile.mkdtemp(prefix="critpr-"))
    try:
        mod_path = base / module_name
        mod_path.parent.mkdir(parents=True, exist_ok=True)
        mod_path.write_text(module_src, encoding="utf-8")
        test_paths: List[Path] = []
        for i, t in enumerate(killer_tests):
            dst = base / "tests" / f"test_killer_{i}.py"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(t, dst)
            test_paths.append(dst)

        # sanity: killer tests must be green on the pristine module
        rc, out = _run_pytest(base, test_paths, timeout)
        if rc != 0:
            return {"accepted": False, "total": 0, "killed": 0, "kill_rate": 0.0,
                    "reason": f"killer tests not green on pristine module (rc={rc}):\n{out}",
                    "killed_mutants": []}

        def work(m: M.Mutant) -> M.Mutant:
            local = Path(tempfile.mkdtemp(prefix="critpr-run-", dir=str(base)))
            shutil.copytree(base, local, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("critpr-run-*",
                                                         "__pycache__"))
            try:
                (local / module_name).write_text(
                    apply_gate_mutant(module_src, m, audit_mod), encoding="utf-8")
                rc, out = _run_pytest(local,
                                      [local / "tests" / t.name for t in test_paths],
                                      timeout)
                m.killed = rc != 0
                if rc == 124:
                    m.detail = "timeout"
                return m
            finally:
                shutil.rmtree(local, ignore_errors=True)

        if jobs > 1 and len(mutants) > 1:
            with ThreadPoolExecutor(max_workers=jobs) as ex:
                mutants = list(ex.map(work, mutants))
        else:
            mutants = [work(m) for m in mutants]
    finally:
        shutil.rmtree(base, ignore_errors=True)

    killed = [m for m in mutants if m.killed]
    accepted = len(killed) >= 1
    return {
        "accepted": accepted,
        "total": len(mutants),
        "killed": len(killed),
        "kill_rate": round(len(killed) / len(mutants), 4) if mutants else 0.0,
        "reason": ("killer test kills "
                   f"{len(killed)}/{len(mutants)} mutants"
                   if accepted else
                   "no killer test kills any mutant -- criterion is "
                   "unfalsifiable as tested"),
        "killed_mutants": [
            {"name": m.name, "op": m.op, "line": m.line, "patch": m.patch}
            for m in killed
        ],
    }


def ci_check(pr_payload: dict, work_root: Path,
             audit_tool: Optional[Path] = None) -> dict:
    """CI entry point: validate a PR payload that adds criteria.

    ``pr_payload`` = ``{"criteria": [{"id", "gate", "module_src",
    "module_name", "killer_tests": [paths]}], ...}``.  The PR is rejected if
    any new criterion fails :func:`check_new_criterion`.
    """
    results = []
    for c in pr_payload.get("criteria", []):
        r = check_new_criterion(
            c["module_src"], c["module_name"],
            [Path(p) for p in c.get("killer_tests", [])],
            work_root, audit_tool=audit_tool)
        r["criterion"] = f"{c['gate']}/{c['id']}"
        results.append(r)
    accepted = all(r["accepted"] for r in results)
    return {
        "accepted": accepted,
        "verdict": "PASS" if accepted else "FAIL",
        "results": results,
    }
