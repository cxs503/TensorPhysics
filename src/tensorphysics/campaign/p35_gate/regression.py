"""Criterion (c): re-run the already-verified faces and compare to their baselines.

The gate never re-implements a solver kernel.  For every declared case it runs
the *producer's own published runner* inside the producer checkout, reads the
metric out of the JSON that runner writes, and compares it with the baseline the
same directory already records in ``result.json``.

Three verdicts, and no fourth:

``PASS``
    the probe executed and ``|measured - baseline| / max(|baseline|, floor)`` is
    below the campaign tolerance (1%).  ``floor`` is mandatory for
    machine-precision metrics: an l2 error of ``6e-14`` against a baseline of
    ``9.6e-14`` is 40% in relative terms and physically meaningless, so the
    comparison is made against a declared absolute floor and says so.
``FAIL``
    the probe executed and left the tolerance -- a real regression.
``WITHHELD``
    the probe could not execute (runner entry point absent from the checkout,
    baseline record absent, timeout, environment failure).  Withheld is never a
    pass; criterion (c) stays non-green and the reason is written to the verdict.

Results are appended to a records file as they are produced, so a long probe can
be run in a separate shell (the harness caps one call at 120 s) and the verdict
re-rendered later from the records.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import spec as S

PASS, FAIL, WITHHELD = "PASS", "FAIL", "WITHHELD"

#: How much of the child's stdout/stderr to keep in the verdict (bytes).
_TAIL = 1200


def _dotted(payload, path: str):
    cur = payload
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return cur


def _split_metric_pattern(pattern: str) -> tuple[str, str]:
    if ":" not in pattern:
        raise ValueError(f"metric_pattern must be 'file_glob:key_path', got {pattern!r}")
    glob, key = pattern.split(":", 1)
    return glob, key


def read_metric(out_dir: Path, pattern: str) -> tuple[float | None, str]:
    """Locate the probe output JSON and pull the metric out of it."""
    glob, key = _split_metric_pattern(pattern)
    matches = sorted(out_dir.glob(glob))
    if not matches:
        return None, f"probe output not found: {out_dir}/{glob}"
    try:
        payload = json.loads(matches[-1].read_text())
    except (OSError, ValueError) as exc:
        return None, f"probe output unreadable ({matches[-1].name}): {exc}"
    value = _dotted(payload, key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None, f"metric {key!r} missing/non-numeric in {matches[-1].name}"
    return float(value), ""


def compare(measured: float, baseline: float, threshold: float, floor: float) -> tuple[str, float]:
    """Return ``(status, relative_deviation)`` under the declared floor rule."""
    scale = max(abs(baseline), abs(floor), 1e-300)
    rel = abs(measured - baseline) / scale
    return (PASS if rel <= threshold else FAIL), rel


@dataclass
class ProbeResult:
    case: str
    repo: str
    status: str
    detail: str
    baseline: float | None = None
    measured: float | None = None
    rel_dev: float | None = None
    threshold: float = S.REGRESSION_THRESHOLD
    floor: float = 0.0
    elapsed_s: float | None = None
    repo_root: str | None = None
    repo_head: str | None = None
    command: list = field(default_factory=list)
    stdout_tail: str = ""

    def as_dict(self) -> dict:
        return {
            "case": self.case,
            "repo": self.repo,
            "status": self.status,
            "detail": self.detail,
            "baseline": self.baseline,
            "measured": self.measured,
            "rel_dev": self.rel_dev,
            "threshold": self.threshold,
            "floor": self.floor,
            "elapsed_s": self.elapsed_s,
            "repo_root": self.repo_root,
            "repo_head": self.repo_head,
            "command": list(self.command),
            "stdout_tail": self.stdout_tail,
        }


def git_head(root: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def run_probe(case: S.RegressionCase, repo_root: str | Path, workdir: str | Path, *,
              timeout: float = 3600.0, python: str | None = None,
              extra_args: tuple[str, ...] = ()) -> ProbeResult:
    """Execute one declared case and judge it against its recorded baseline."""
    root = Path(repo_root)
    out_dir = Path(workdir) / case.case
    out_dir.mkdir(parents=True, exist_ok=True)
    base = ProbeResult(case=case.case, repo=case.repo, status=WITHHELD, detail="",
                       threshold=case.threshold, floor=case.floor,
                       repo_root=str(root), repo_head=git_head(root) if root.is_dir() else None)

    baseline_doc_path = root / case.baseline_path
    if not baseline_doc_path.is_file():
        base.detail = f"baseline record absent: {case.baseline_path}"
        return base
    baseline = _dotted(json.loads(baseline_doc_path.read_text()), case.metric_path)
    if not isinstance(baseline, (int, float)) or isinstance(baseline, bool):
        base.detail = f"baseline metric {case.metric_path!r} missing in {case.baseline_path}"
        return base
    base.baseline = float(baseline)

    budget, estimate = case.budget_s, case.estimated_cost_s
    if budget is not None and estimate is not None and estimate > budget:
        base.detail = (f"案例未启动：测算成本 {estimate:.0f} s 超出本门预算 {budget:.0f} s "
                       f"(declared estimate; 见 spec 实测标定)")
        return base

    runner = case.command[1] if len(case.command) > 1 else ""
    if runner and not (root / runner).is_file():
        base.detail = (f"producer runner entry point absent from checkout: {runner} "
                       f"(repo {case.repo}, head {base.repo_head}); probe not executable")
        return base

    interpreter = python or sys.executable
    command = [interpreter if i == 0 and part in ("python", "python3") else part.replace(
        "{out}", str(out_dir))
        for i, part in enumerate(case.command)]
    base.command = command
    env = dict(os.environ, PYTHONPATH=str(root / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    t0 = time.time()
    try:
        proc = subprocess.run(command + list(extra_args), cwd=str(root), capture_output=True,
                              text=True, timeout=timeout, env=env)
        rc, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        base.elapsed_s = round(time.time() - t0, 1)
        base.detail = f"probe exceeded {timeout:.0f}s budget"
        base.stdout_tail = ((exc.stdout or b"") if isinstance(exc.stdout, bytes)
                            else (exc.stdout or ""))[-_TAIL:] if exc.stdout else ""
        return base
    except OSError as exc:
        base.detail = f"probe launch failed: {exc}"
        return base
    base.elapsed_s = round(time.time() - t0, 1)
    base.stdout_tail = ((stdout or "") + (stderr or ""))[-_TAIL:]

    if rc != 0:
        base.status = FAIL
        base.detail = f"producer runner exited {rc}"
        return base

    measured, why = read_metric(out_dir, case.metric_pattern)
    if measured is None:
        base.detail = why
        return base
    base.measured = measured
    status, rel = compare(measured, base.baseline, case.threshold, case.floor)
    base.status = status
    base.rel_dev = rel
    base.detail = (f"measured {measured:.6g} vs baseline {base.baseline:.6g} -> "
                   f"|delta|/max(|baseline|,floor) = {rel:.3e} "
                   f"(threshold {case.threshold:.1%}, floor {case.floor:g})")
    return base


def load_records(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def append_record(path: str | Path, result: ProbeResult) -> dict:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    records = load_records(p)
    records[result.case] = result.as_dict()
    p.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    return records


def evaluate_regression(records: dict, *, cases=S.REGRESSION_CASES) -> tuple[list[dict], str]:
    """Turn recorded probes into per-case records and one aggregate status."""
    out, statuses = [], []
    for case in cases:
        rec = records.get(case.case)
        if not rec:
            rec = {
                "case": case.case, "repo": case.repo, "status": WITHHELD,
                "detail": "no probe record: case not re-run in this gate environment",
                "baseline": None, "measured": None, "rel_dev": None,
                "threshold": case.threshold, "floor": case.floor,
                "elapsed_s": None, "repo_root": None, "repo_head": None,
                "command": list(case.command), "stdout_tail": "",
            }
        out.append(rec)
        statuses.append(rec["status"])
    if not statuses:
        return out, WITHHELD
    if FAIL in statuses:
        return out, FAIL
    if all(s == PASS for s in statuses):
        return out, PASS
    return out, WITHHELD


def median_deviation(cases: list[dict]) -> float | None:
    devs = [c["rel_dev"] for c in cases if isinstance(c.get("rel_dev"), (int, float))]
    if not devs:
        return None
    devs.sort()
    mid = len(devs) // 2
    return devs[mid] if len(devs) % 2 else 0.5 * (devs[mid - 1] + devs[mid])