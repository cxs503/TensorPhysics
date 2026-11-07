"""Scratch-tree mutation runner with cluster sharding.

For one gate the runner:

1. builds a scratch tree that preserves the repo-relative layout the gate's
   test suite expects (module parent dir + test files + extras, copied
   verbatim);
2. verifies the pristine tree is green (else every mutant would look
   "killed" -- the audit is void);
3. for each assigned mutant: copies the scratch tree, applies the mutant to
   the module, runs the gate's test suite, records killed / survived;
4. writes ``shard_<i>.json``; :func:`aggregate` merges the shards.

The *native* mutant classes (operator swaps, silent drops, severity flips,
constant scaling) are enumerated and applied by the upstream mutation-audit
tool -- loaded by path (``audit_tool``) and consumed, never rewritten.
M1..M5 come from :mod:`...criterion_ledger.mutants`.

Sharding
--------
* per-gate: the gate's mutant list is split into ``n_shards`` contiguous
  slices; shard ``i`` runs slice ``i`` and writes ``shard_<i>.json``;
* full ledger: :func:`run_full_ledger_shard` concatenates the mutant lists
  of all gates (deterministic order), slices the *global* list, and runs
  each gate's slice in one scratch tree per gate.  Slices are disjoint and
  complete, so the aggregation is exact.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from . import mutants as M

SHARD_GLOB = "shard_*.json"


# ---------------------------------------------------------------------------
# upstream tool consumption
# ---------------------------------------------------------------------------

def load_audit_module(path: Path):
    """Load the upstream mutation-audit tool by file path (consume, not rewrite)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"audit tool not found: {path}")
    spec = importlib.util.spec_from_file_location(
            "upstream_mutation_audit_" + path.parent.name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _from_audit(m) -> M.Mutant:
    return M.Mutant(op=m.op, index=m.index, line=m.line, snippet=m.snippet,
                    patch=m.patch, killed=m.killed, detail=m.detail,
                    extra=dict(getattr(m, "extra", {}) or {}))


def _to_audit(m: M.Mutant, audit_mod):
    """Map a ledger Mutant onto the upstream tool's Mutant dataclass."""
    return audit_mod.Mutant(op=m.op, index=m.index, line=m.line,
                            snippet=m.snippet, patch=m.patch, killed=m.killed,
                            detail=m.detail, extra=dict(m.extra))


def enumerate_gate_mutants(src: str, classes: Sequence[str],
                           audit_mod=None) -> List[M.Mutant]:
    """Enumerate all mutants of ``src`` for the requested classes."""
    out: List[M.Mutant] = []
    for cls in classes:
        if cls == M.NATIVE:
            if audit_mod is None:
                raise ValueError("class 'native' requires the upstream audit module")
            out.extend(_from_audit(m) for m in audit_mod.enumerate_mutants(src))
        elif cls in M.ENUMERATORS:
            out.extend(M.ENUMERATORS[cls](src))
        else:
            raise ValueError(f"unknown mutant class: {cls}")
    return out


def apply_gate_mutant(src: str, m: M.Mutant, audit_mod=None) -> str:
    """Apply one mutant (native via the upstream tool, M1..M5 locally)."""
    if m.op in M.ENUMERATORS:
        return M.apply_mutant(src, m)
    if audit_mod is None:
        raise ValueError(f"native mutant {m.name} requires the audit module")
    return audit_mod.apply_mutant(src, _to_audit(m, audit_mod))


# ---------------------------------------------------------------------------
# scratch tree
# ---------------------------------------------------------------------------

def _common_root(paths: Sequence[Path]) -> Path:
    ps = [p.resolve() for p in paths]
    root = ps[0].parent
    while not all(p.is_relative_to(root) for p in ps):
        if root.parent == root:
            raise ValueError(f"no common root for {ps}")
        root = root.parent
    return root


def build_scratch_tree(module: Path, tests: Sequence[Path],
                        extra_copies: Sequence[Path] = ()) -> Path:
    """Copy module parent dir + test files (+ extras) into a scratch tree.

    The layout under the common root of module and tests is preserved, so
    test files that compute their import root from ``__file__`` keep
    working unchanged.  ``extra_copies`` are paths (relative to the common
    root) copied verbatim -- e.g. a package root for in-repo package
    imports.
    """
    base = Path(tempfile.mkdtemp(prefix="critmut-"))
    root = _common_root([module, *tests])
    mod_rel = module.resolve().relative_to(root)
    shutil.copytree(module.resolve().parent, base / mod_rel.parent,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__"))
    for t in tests:
        rel = t.resolve().relative_to(root)
        dst = base / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(t, dst)
    for e in extra_copies:
        src_p = root / e
        dst = base / e
        if src_p.is_dir():
            shutil.copytree(src_p, dst, dirs_exist_ok=True,
                           ignore=shutil.ignore_patterns("__pycache__"))
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_p, dst)
    return base


def _run_pytest(workdir: Path, tests: Sequence[Path], timeout: int):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    cmd = [sys.executable, "-m", "pytest", *[str(t) for t in tests], "-q",
           "--noconftest", "-p", "no:cacheprovider", "-x"]
    try:
        p = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True,
                           text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr)[-800:]
    except subprocess.TimeoutExpired:
        return 124, "timeout"


# ---------------------------------------------------------------------------
# core: run a given mutant list for one gate
# ---------------------------------------------------------------------------

def shard_slice(items: Sequence, shard_id: int, n_shards: int) -> List:
    """Contiguous slice ``shard_id`` of ``n_shards`` (disjoint, complete)."""
    if n_shards < 1:
        raise ValueError("n_shards must be >= 1")
    if not (0 <= shard_id < n_shards):
        raise ValueError(f"shard_id {shard_id} out of range [0, {n_shards})")
    n = len(items)
    lo = n * shard_id // n_shards
    hi = n * (shard_id + 1) // n_shards
    return list(items[lo:hi])


def _mutant_records(mutants: Sequence[M.Mutant]) -> List[dict]:
    return [
        {"name": m.name, "op": m.op, "line": m.line, "killed": m.killed,
         "snippet": m.snippet, "patch": m.patch, "detail": m.detail}
        for m in mutants
    ]


def run_gate_mutants(gate, work_root: Path, mutants: Sequence[M.Mutant],
                     *, jobs: int = 8, timeout: int = 180,
                     audit_mod=None) -> dict:
    """Run a pre-enumerated mutant list against one gate (scratch tree)."""
    module = (work_root / gate.module).resolve()
    tests = [(work_root / t).resolve() for t in gate.tests]
    if not module.is_file():
        raise FileNotFoundError(f"gate module missing: {module}")
    missing = [str(t) for t in tests if not t.is_file()]
    if missing:
        raise FileNotFoundError(f"gate tests missing: {missing}")

    base = build_scratch_tree(module, tests, gate.extra_copies)
    root = _common_root([module, *tests])
    mod_rel = module.relative_to(root)
    test_rels = [t.relative_to(root) for t in tests]
    mod_path = base / mod_rel
    pristine = mod_path.read_text(encoding="utf-8")
    try:
        rc, out = _run_pytest(base, [base / r for r in test_rels], timeout)
        if rc != 0:
            raise SystemExit(
                f"baseline suite is not green in scratch tree (rc={rc}):\n{out}")

        def work(m: M.Mutant) -> M.Mutant:
            local = Path(tempfile.mkdtemp(prefix="critmut-run-", dir=str(base)))
            shutil.copytree(base, local, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("critmut-run-*",
                                                         "__pycache__"))
            try:
                (local / mod_rel).write_text(
                    apply_gate_mutant(pristine, m, audit_mod), encoding="utf-8")
                rc, out = _run_pytest(local, [local / r for r in test_rels],
                                     timeout)
                m.killed = rc != 0
                if rc == 124:
                    m.detail = "timeout"
                elif not m.killed and out.strip():
                    m.detail = out.strip().splitlines()[-1][:200]
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

    return {
        "gate": gate.gate_id,
        "module": str(gate.module),
        "tests": [str(t) for t in gate.tests],
        "baseline_green": True,
        "total": len(mutants),
        "killed": sum(1 for m in mutants if m.killed),
        "mutants": _mutant_records(mutants),
    }


def run_gate(gate, work_root: Path, *, jobs: int = 8, timeout: int = 180,
               out_dir: Optional[Path] = None, shard_id: int = 0,
               n_shards: int = 1, classes: Optional[Sequence[str]] = None,
               audit_tool: Optional[Path] = None,
               per_class: Optional[int] = None,
               m1_eps: Optional[float] = None) -> dict:
    """Run the mutation audit of one gate (one shard) and write its JSON.

    ``gate`` is a :class:`~...adapter.GateSpec`; ``work_root`` is the
    directory containing the repo-relative layout (``gate.module`` and
    ``gate.tests`` are relative to it).  ``per_class=k`` keeps only the
    first ``k`` mutants of each class (negative-control mode).
    """
    module = (work_root / gate.module).resolve()
    tests = [(work_root / t).resolve() for t in gate.tests]
    if not module.is_file():
        raise FileNotFoundError(f"gate module missing: {module}")
    missing = [str(t) for t in tests if not t.is_file()]
    if missing:
        raise FileNotFoundError(f"gate tests missing: {missing}")

    classes = tuple(classes) if classes is not None else gate.mutant_classes
    audit_mod = (load_audit_module(audit_tool)
                 if M.NATIVE in classes and audit_tool is not None else None)

    src = module.read_text(encoding="utf-8")
    mutants = enumerate_gate_mutants(src, classes, audit_mod)
    if m1_eps is not None and M.M1 in classes:
        mutants = [m for m in mutants if m.op != M.M1]
        mutants.extend(M.enumerate_m1(src, eps=m1_eps))
    if per_class is not None:
        kept: dict[str, int] = {}
        picked: List[M.Mutant] = []
        for m in mutants:
            if kept.get(m.op, 0) < per_class:
                picked.append(m)
                kept[m.op] = kept.get(m.op, 0) + 1
        mutants = picked
    mutants = shard_slice(mutants, shard_id, n_shards)

    res = run_gate_mutants(gate, work_root, mutants, jobs=jobs,
                           timeout=timeout, audit_mod=audit_mod)
    res["shard_id"] = shard_id
    res["n_shards"] = n_shards
    res["classes"] = list(classes)
    res["survived"] = res["total"] - res["killed"]
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"shard_{shard_id}.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


# ---------------------------------------------------------------------------
# full-ledger sharding
# ---------------------------------------------------------------------------

def run_full_ledger_shard(ledger, work_roots: Dict[str, Path], *,
                          shard_id: int, n_shards: int,
                          out_dir: Path, jobs: int = 8, timeout: int = 180,
                          audit_tool: Optional[Path] = None) -> dict:
    """Run one shard of the *full* ledger (all gates, all configured classes).

    ``work_roots`` maps gate_id -> the root of the tree holding that gate's
    layout.  The global mutant list (gates in registration order, classes in
    spec order) is sliced; each gate's slice is executed in its own scratch
    tree.  Writes ``shard_<i>.json`` = ``{gate_id: result}``.
    """
    audit_mod = (load_audit_module(audit_tool)
                if audit_tool is not None else None)
    global_list: List[Tuple[str, M.Mutant]] = []
    for gate in ledger.gates():
        module = (work_roots[gate.gate_id] / gate.module)
        src = module.read_text(encoding="utf-8")
        ms = enumerate_gate_mutants(src, gate.mutant_classes, audit_mod)
        global_list.extend((gate.gate_id, m) for m in ms)

    picked = shard_slice(global_list, shard_id, n_shards)
    by_gate: Dict[str, List[M.Mutant]] = {}
    for gid, m in picked:
        by_gate.setdefault(gid, []).append(m)

    results: Dict[str, dict] = {}
    for gid, ms in by_gate.items():
        gate = ledger.gate(gid)
        results[gid] = run_gate_mutants(gate, work_roots[gid], ms,
                                       jobs=jobs, timeout=timeout,
                                       audit_mod=audit_mod)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"shard_{shard_id}.json").write_text(
        json.dumps({"shard_id": shard_id, "n_shards": n_shards,
                    "gates": results}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return results


def aggregate_full_ledger(out_dir: Path) -> dict:
    """Merge full-ledger shards into per-gate results + overall KPI."""
    from .kpi import ledger_kpi
    out_dir = Path(out_dir)
    shards = sorted(out_dir.glob(SHARD_GLOB),
                    key=lambda p: int(p.stem.split("_")[1]))
    if not shards:
        raise FileNotFoundError(f"no {SHARD_GLOB} under {out_dir}")
    per_gate: Dict[str, List[dict]] = {}
    meta: Dict[str, dict] = {}
    for s in shards:
        d = json.loads(s.read_text(encoding="utf-8"))
        for gid, res in d["gates"].items():
            per_gate.setdefault(gid, []).extend(res["mutants"])
            meta[gid] = res
    per_gate: Dict[str, dict] = {}
    for gid, ms in per_gate.items():
        by_class: dict[str, dict[str, int]] = {}
        for m in ms:
            d = by_class.setdefault(m["op"], {"total": 0, "killed": 0})
            d["total"] += 1
            d["killed"] += int(m["killed"])
        killed = sum(1 for m in ms if m["killed"])
        per_gate[gid] = {
            "gate": gid,
            "module": meta[gid].get("module"),
            "tests": meta[gid].get("tests"),
            "baseline_green": meta[gid].get("baseline_green", True),
            "total": len(ms),
            "killed": killed,
            "survived": len(ms) - killed,
            "kill_rate": round(killed / len(ms), 4) if ms else 0.0,
            "by_class": {k: {**v, "kill_rate": round(v["killed"] / v["total"], 4)}
                         for k, v in sorted(by_class.items())},
            "survivors": [
                {"name": m["name"], "op": m["op"], "line": m["line"],
                 "snippet": m["snippet"], "patch": m["patch"]}
                for m in ms if not m["killed"]
            ],
            "mutants": ms,
        }
    kpi = ledger_kpi(per_gate)
    full = {"per_gate": per_gate, "kpi": kpi}
    (out_dir / "full_ledger.json").write_text(
        json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8")
    return full


# ---------------------------------------------------------------------------
# per-gate aggregation
# ---------------------------------------------------------------------------

def aggregate(out_dir: Path) -> dict:
    """Merge all ``shard_*.json`` of one gate into the full result."""
    out_dir = Path(out_dir)
    shards = sorted(out_dir.glob(SHARD_GLOB),
                    key=lambda p: int(p.stem.split("_")[1]))
    if not shards:
        raise FileNotFoundError(f"no {SHARD_GLOB} under {out_dir}")
    mutants: List[dict] = []
    meta: dict = {}
    for s in shards:
        d = json.loads(s.read_text(encoding="utf-8"))
        meta = d
        mutants.extend(d["mutants"])
    by_class: dict[str, dict[str, int]] = {}
    for m in mutants:
        d = by_class.setdefault(m["op"], {"total": 0, "killed": 0})
        d["total"] += 1
        d["killed"] += int(m["killed"])
    killed = sum(1 for m in mutants if m["killed"])
    res = {
        "gate": meta.get("gate"),
        "module": meta.get("module"),
        "tests": meta.get("tests"),
        "classes": meta.get("classes"),
        "baseline_green": all(
            json.loads(s.read_text(encoding="utf-8"))["baseline_green"]
            for s in shards),
        "total": len(mutants),
        "killed": killed,
        "survived": len(mutants) - killed,
        "kill_rate": round(killed / len(mutants), 4) if mutants else 0.0,
        "by_class": {k: {**v, "kill_rate": round(v["killed"] / v["total"], 4)}
                     for k, v in sorted(by_class.items())},
        "survivors": [
            {"name": m["name"], "op": m["op"], "line": m["line"],
             "snippet": m["snippet"], "patch": m["patch"]}
            for m in mutants if not m["killed"]
        ],
        "mutants": mutants,
    }
    (out_dir / "aggregated.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res
