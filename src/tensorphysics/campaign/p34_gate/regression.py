"""Regression runner for criterion C3 (pre-existing capabilities still green).

The campaign is only acceptable if the capabilities that existed *before* P34 are
still green.  C3 therefore does not trust any node's claim about "existing tests
still pass"; it re-runs a pinned set of test files per capability family and
records, for every invocation, the repository head, archive architecture, exit
code and junit-derived counts.

Runs are resumable: results are appended to ``<out>/records.json`` after each
file, so a driver can execute the suite in small time slices without losing
evidence (the agent shell enforces a 120 s per-call ceiling).
"""
from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .collect import _parse_junit, _git


@dataclass
class SuiteSpec:
    name: str
    family: str
    repo: str
    root: Path
    tests: tuple[str, ...] = ()
    note: str = ""


#: Failure causes that are *not* a numerical/logic regression of the module; they
#: are recorded separately so that a red suite can be read for what it is.
RED_CLASSES = (
    ("environment", re.compile(
        r"(ModuleNotFoundError|No module named|ImportError|PermissionError|"
        r"libflexiblas|cannot open shared object)", re.I)),
    ("missing_artifact", re.compile(
        r"(does not exist|No such file|FileNotFoundError|exists\(\)|"
        r"artifact .* missing|missing (figures|artifact))", re.I)),
    ("live_service", re.compile(
        r"(URLError|Connection refused|ConnectionError|HTTPError|"
        r"RemoteDisconnected|Max retries|timed out|XPASS\(strict\))", re.I)),
)


def classify_red(junit_paths) -> list[dict]:
    """Read failure/error messages from junit and bucket them by cause.

    Buckets: ``missing_artifact`` (evidence files filtered out of the checkout),
    ``live_service`` (test needs a running API endpoint), ``logic_or_numeric``
    (neither pattern matched -> treated as a genuine regression until shown
    otherwise).  Nothing is dropped: every red test appears with a class.
    """
    from xml.etree import ElementTree as ET
    out: list[dict] = []
    for jp in junit_paths:
        jp = Path(jp)
        if not jp.is_file():
            continue
        try:
            root = ET.parse(jp).getroot()
        except ET.ParseError:
            continue
        for tc in root.iter("testcase"):
            bad = [c for c in tc if c.tag in ("failure", "error")]
            if not bad:
                continue
            msg = " ".join((bad[0].get("message") or "") + " " + (bad[0].text or "")
                           for _ in [0])[:600]
            klass = "logic_or_numeric"
            for name, pat in RED_CLASSES:
                if pat.search(msg):
                    klass = name
                    break
            out.append({"test": f"{tc.get('classname', '')}::{tc.get('name', '')}",
                        "class": klass, "message": msg[:200]})
    return out


def _head_sha(root: Path) -> str:
    rc, out = _git(root, "rev-parse", "HEAD")
    return out if rc == 0 and len(out) == 40 else ""


def _covered(records: list[dict]) -> set[str]:
    out: set[str] = set()
    for r in records:
        out.update(r.get("files") or [r["test"]])
    return out


def pending(spec: SuiteSpec, out_dir: Path) -> list[str]:
    rec_path = out_dir / spec.name / "records.json"
    records = json.loads(rec_path.read_text(encoding="utf-8")) if rec_path.is_file() else []
    done = _covered(records)
    return [t for t in spec.tests if t not in done]


def run_suite(spec: SuiteSpec, out_dir: Path, budget_s: float = 100.0,
              per_file_timeout: float = 240.0, python: str | None = None,
              group_size: int = 1) -> dict:
    """Run not-yet-recorded test files of ``spec`` until ``budget_s`` elapses.

    ``group_size`` batches several files into one pytest invocation to amortise
    interpreter start-up; each record keeps the file list it covers, so coverage
    accounting stays exact.
    """
    d = out_dir / spec.name
    (d / "junit").mkdir(parents=True, exist_ok=True)
    rec_path = d / "records.json"
    records = json.loads(rec_path.read_text(encoding="utf-8")) if rec_path.is_file() else []
    todo = pending(spec, out_dir)
    started = time.time()
    ran: list[str] = []
    groups = [todo[i:i + group_size] for i in range(0, len(todo), group_size)]
    for group in groups:
        if time.time() - started > budget_s:
            break
        files = list(group)
        missing = [f for f in files if not (spec.root / f).exists()]
        if missing:
            records.append({"test": ", ".join(files), "files": files,
                            "collected": 0, "passed": 0, "failed": 0,
                            "errors": len(missing), "skipped": 0, "exit_code": 4,
                            "duration_s": 0.0, "arch": platform.machine(),
                            "head_sha": _head_sha(spec.root), "junit": "",
                            "note": f"test file missing: {missing}", "family": spec.family})
            ran.extend(files)
            rec_path.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
            continue
        tag = "_".join(f.split("/")[-1].removesuffix(".py") for f in files)[:120]
        # absolute: pytest runs with cwd=<suite repo>, evidence must stay in out_dir
        junit = (d / "junit" / f"{tag}.xml").resolve()
        cmd = [python or sys.executable, "-m", "pytest", *files, "-q", "-p", "no:cacheprovider",
               f"--junitxml={junit}"]
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, cwd=str(spec.root), capture_output=True, text=True,
                                  timeout=per_file_timeout)
            exit_code, note = proc.returncode, (proc.stdout or "").strip().splitlines()[-1:] or [""]
            note = note[0][:200] if isinstance(note, list) else str(note)[:200]
        except subprocess.TimeoutExpired:
            exit_code, note = 124, f"timeout {per_file_timeout:.0f}s"
        counts = _parse_junit(junit)
        rec = {"test": ", ".join(files), "files": files, **counts, "exit_code": exit_code,
               "duration_s": round(time.time() - t0, 2), "arch": platform.machine(),
               "head_sha": _head_sha(spec.root), "junit": str(junit),
               "note": note, "family": spec.family}
        records.append(rec)
        ran.extend(files)
        rec_path.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"suite": spec.name, "family": spec.family, "ran": ran,
            "n_pending": len(todo), "n_remaining": len(pending(spec, out_dir)),
            "elapsed_s": round(time.time() - started, 2)}


def assemble(spec: SuiteSpec, out_dir: Path) -> dict:
    """Fold per-file records into one suite verdict row for C3."""
    d = out_dir / spec.name
    rec_path = d / "records.json"
    records = json.loads(rec_path.read_text(encoding="utf-8")) if rec_path.is_file() else []
    covered = _covered(records)
    missing = [t for t in spec.tests if t not in covered]
    green = [r for r in records if r["exit_code"] == 0 and r["failed"] == 0
             and r["errors"] == 0 and r["collected"] > 0]
    red = [r for r in records if r not in green]
    return {
        "name": spec.name,
        "family": spec.family,
        "repo": spec.repo,
        "target": f"{spec.repo}:{len(spec.tests)} test files ({spec.name})",
        "root": str(spec.root),
        "head_sha": _head_sha(spec.root),
        "arch": platform.machine(),
        "n_test_files": len(spec.tests),
        "n_covered": len(covered),
        "n_missing": len(missing),
        "missing": missing,
        "collected": sum(r["collected"] for r in records),
        "passed": sum(r["passed"] for r in records),
        "failed": sum(r["failed"] for r in records),
        "errors": sum(r["errors"] for r in records) + len(missing),
        "skipped": sum(r["skipped"] for r in records),
        "duration_s": round(sum(r["duration_s"] for r in records), 1),
        # exit_code is the aggregate: 0 only if every file ran green
        "exit_code": 0 if not red and not missing else 1,
        "red_files": ([{"test": r["test"], "exit_code": r["exit_code"],
                        "failed": r["failed"], "errors": r["errors"],
                        "note": r.get("note", "")} for r in red]
                      + [{"test": t, "exit_code": None, "failed": 0, "errors": 1,
                          "note": "not run (no record)"} for t in missing]),
        "junit_dir": str(d / "junit"),
        "notes": spec.note,
        "red_classes": classify_red(sorted((d / "junit").glob("*.xml"))),
    }


def write_regression(specs: list[SuiteSpec], out_dir: Path, out_path: Path) -> dict:
    rows = [assemble(s, out_dir) for s in specs]
    doc = {
        "schema": "p34_regression_report.v1",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arch": platform.machine(),
        "n_suites": len(rows),
        "n_green": sum(1 for r in rows if r["exit_code"] == 0),
        "suites": rows,
        "path": str(out_path),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return doc