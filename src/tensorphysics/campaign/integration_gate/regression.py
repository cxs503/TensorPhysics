"""P15 -> P16 regression baseline for the eleven delivered P15 nodes.

P15 (plan 15) closed eleven domain nodes ``IN14-D1 .. IN14-D11`` in the solver
libraries.  P16 must not degrade them.  What is actually comparable is what the
platform recorded for each closed P15 mission: the *probe* (pytest targets) and
the *result* (``19/19 passed``, ``16 passed``, ...).  The regression metric is
therefore the probe pass rate

    rate = passed / (passed + failed + errors)

and the gate criterion ``|delta| < 1%`` is evaluated against the recorded P15
rate (1.0 for an all-green P15 probe).  A probe that cannot be executed in the
gate environment yields ``WITHHELD`` -- never a silent pass.

Two guard rails keep this from being a vacuous "tests still pass" ritual:

* ``test-count preservation`` -- a repo may not silently delete/skip the P15
  tests in order to stay green (``n_passed_now >= n_passed_p15``);
* optional *numeric* metrics -- when a node recorded a number (e.g. a decay
  rate or a coefficient), a ``metric`` block gives an exact relative delta.

Probe execution happens where the code lives (the solver repo checkout, e.g. on
the compute cluster) and is fed to the gate as ``probe_results.json``, so the
gate itself stays offline and reproducible.
"""
from __future__ import annotations

import json
import os
import re
import subprocess  # nosec - the gate runs the project's own tests
from dataclasses import dataclass, field
from pathlib import Path

REGRESSION_THRESHOLD = 0.01          # |delta| < 1 %
PROBE_RESULTS_SCHEMA = "p15_probe_results.v1"

_COUNT_KEYS = {"passed": r"(\d+)\s+passed", "failed": r"(\d+)\s+failed",
               "errors": r"(\d+)\s+error", "skipped": r"(\d+)\s+skipped",
               "xfailed": r"(\d+)\s+xfailed", "deselected": r"(\d+)\s+deselected"}


@dataclass
class CaseVerdict:
    ext_id: str
    repo: str | None
    name: str = ""
    status: str = "WITHHELD"
    rate_p15: float | None = None
    rate_now: float | None = None
    delta: float | None = None
    n_passed_p15: int | None = None
    n_passed_now: int | None = None
    test_count_preserved: bool | None = None
    command: str | None = None
    detail: str = ""
    metric: dict = field(default_factory=dict)
    coverage: str = "full"          # full | partial | none
    unresolved_targets: list = field(default_factory=list)
    failing_tests: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "ext_id": self.ext_id,
            "repo": self.repo,
            "name": self.name,
            "status": self.status,
            "rate_p15": self.rate_p15,
            "rate_now": self.rate_now,
            "delta": self.delta,
            "delta_abs": None if self.delta is None else abs(self.delta),
            "n_passed_p15": self.n_passed_p15,
            "n_passed_now": self.n_passed_now,
            "test_count_preserved": self.test_count_preserved,
            "command": self.command,
            "detail": self.detail,
            "metric": self.metric,
            "coverage": self.coverage,
            "unresolved_targets": self.unresolved_targets,
            "failing_tests": self.failing_tests,
        }


def parse_pytest_counts(text: str) -> dict:
    """Extract pass/fail/error counts from pytest stdout."""
    counts = {}
    for key, pat in _COUNT_KEYS.items():
        vals = [int(m) for m in re.findall(pat, text or "")]
        if vals:
            counts[key] = max(vals) if key == "skipped" else sum(vals)
    if not counts:
        for m in re.finditer(r"(\d+)\s*/\s*(\d+)\s*passed", text or ""):
            counts["passed"] = int(m.group(1))
            counts["failed"] = int(m.group(2)) - int(m.group(1))
            break
    return counts


def pass_rate(counts: dict) -> float | None:
    p = counts.get("passed", 0)
    f = counts.get("failed", 0) + counts.get("errors", 0)
    total = p + f
    if total == 0:
        return None
    return p / total


def _baseline_case(p15_node: dict) -> tuple[float | None, int | None]:
    probe = p15_node.get("probe") or {}
    recorded_passed = probe.get("n_passed_recorded")
    recorded_total = probe.get("n_total_recorded")
    recorded_failed = probe.get("n_failed_recorded")
    if recorded_passed is not None and recorded_total is not None:
        return (recorded_passed / recorded_total if recorded_total else None), recorded_passed
    if recorded_passed is not None and recorded_failed is not None:
        tot = recorded_passed + recorded_failed
        return (recorded_passed / tot if tot else None), recorded_passed
    if recorded_passed is not None:
        return 1.0, recorded_passed      # P15 recorded "N passed" with no failures named
    return None, None


def evaluate_case(p15_node: dict, probe_result: dict | None,
                  threshold: float = REGRESSION_THRESHOLD,
                  fresh_absence: dict | None = None) -> CaseVerdict:
    rate_p15, n_p15 = _baseline_case(p15_node)
    v = CaseVerdict(
        ext_id=p15_node.get("ext_id") or "?",
        repo=p15_node.get("repo"),
        name=(p15_node.get("name") or "")[:80],
        rate_p15=rate_p15,
        n_passed_p15=n_p15,
    )
    targets = (p15_node.get("probe") or {}).get("test_targets") or []
    v.command = "python -m pytest " + " ".join(targets) + " -q" if targets else None

    # A verification case that is gone from a *fresh* head (GitHub main) is a
    # dropped P15 deliverable, not a stale checkout: that is a hard regression.
    absence = (fresh_absence or {}).get(v.ext_id) or {}
    gone = absence.get("targets_absent_at_fresh_head") or []
    if gone:
        v.status = "FAIL"
        v.coverage = "none"
        v.unresolved_targets = list(gone)
        v.detail = (f"P15 verification case(s) absent at fresh head "
                    f"{absence.get('head')} of {absence.get('repo')} "
                    f"(cluster job {absence.get('job_id')}): {gone}")
        return v

    if probe_result is None:
        v.status = "WITHHELD"
        v.detail = ("probe not executed in the gate environment (repo checkout unavailable); "
                    "P15 baseline recorded, nothing to compare")
        return v

    unresolved = [u.get("path") for u in (probe_result.get("unresolved_detail") or [])]
    v.unresolved_targets = unresolved
    v.coverage = "partial" if unresolved else "full"
    counts = probe_result.get("counts") or parse_pytest_counts(probe_result.get("stdout_tail", ""))
    rate_now = pass_rate(counts)
    v.rate_now = rate_now
    v.n_passed_now = counts.get("passed")
    v.command = probe_result.get("command", v.command)

    if rate_now is None and probe_result.get("resolvable") is False:
        v.status = "WITHHELD"
        v.detail = ("no P15 probe target resolvable at this checkout head (stale mirror or "
                    f"renamed tests); unresolved: {unresolved}")
        return v
    if rate_now is None:
        v.status = "WITHHELD"
        v.detail = (f"probe produced no parsable pytest summary (returncode="
                    f"{probe_result.get('returncode')}); stdout tail: "
                    f"{(probe_result.get('stdout_tail') or '')[-200:]!r}")
        return v

    if rate_p15 is None:
        v.status = "WITHHELD"
        v.detail = "P15 baseline has no recorded pass count for this probe"
        return v

    v.failing_tests = [ln.split("::", 1)[-1] if "::" in ln else ln
                       for ln in (probe_result.get("stdout_tail") or "").splitlines()
                       if ln.startswith("FAILED")]
    v.delta = rate_now - rate_p15
    v.test_count_preserved = (n_p15 is None) or (v.n_passed_now or 0) >= n_p15
    failures = counts.get("failed", 0) + counts.get("errors", 0)

    metric = p15_node.get("probe", {}).get("metric")
    if isinstance(metric, dict) and "baseline" in metric and "value" in metric:
        base = float(metric["baseline"])
        val = float(metric["value"])
        rel = abs(val - base) / abs(base) if base else float("inf")
        v.metric = {"name": metric.get("name"), "baseline": base, "value": val,
                    "rel_delta": rel, "within_threshold": rel < threshold}

    ok = (failures == 0 and abs(v.delta) < threshold and v.test_count_preserved
          and (not v.metric or v.metric["within_threshold"]))
    if ok and v.coverage == "partial":
        # measured subset is green, but part of the P15 probe set could not be
        # resolved at this checkout -> coverage is an evidence gap, not a pass.
        v.status = "WITHHELD"
        v.detail = (f"measured subset green ({counts}) but unresolved P15 targets "
                    f"{unresolved} block a full no-degradation statement")
        return v
    v.status = "PASS" if ok else "FAIL"
    v.detail = (f"pytest={counts}; pass rate {rate_p15:.4f} -> {rate_now:.4f} "
                f"(delta {v.delta:+.2e}, threshold {threshold:g}); "
                f"tests preserved={v.test_count_preserved}"
                + (f"; failing: {v.failing_tests}" if v.failing_tests else ""))
    return v


def evaluate_regression(p15_baseline: dict, probe_results: dict | None,
                        threshold: float = REGRESSION_THRESHOLD) -> list[CaseVerdict]:
    results = (probe_results or {}).get("results") or {}
    fresh = (probe_results or {}).get("fresh_head_absence") or {}
    out = []
    for node in p15_baseline.get("nodes") or []:
        ext = node.get("ext_id")
        out.append(evaluate_case(node, results.get(ext), threshold=threshold,
                                 fresh_absence=fresh))
    return out


# ---------------------------------------------------------------------------
# probe execution (runs where the solver repos live)
# ---------------------------------------------------------------------------
def run_probe(repo_root: str | os.PathLike, targets: list[str],
              timeout: float = 1800.0) -> dict:
    """Run the P15 pytest targets inside a repo checkout and summarise the result."""
    cmd = ["python", "-m", "pytest", *targets, "-q"]
    env = dict(os.environ)
    src = Path(repo_root) / "src"
    if src.is_dir():
        env["PYTHONPATH"] = str(src) + (os.pathsep + env["PYTHONPATH"]
                                        if env.get("PYTHONPATH") else "")
    try:
        proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True,  # nosec
                              timeout=timeout, env=env)
        stdout = (proc.stdout or "") + (proc.stderr or "")
        return {"command": " ".join(cmd), "returncode": proc.returncode,
                "counts": parse_pytest_counts(stdout), "stdout_tail": stdout[-4000:]}
    except subprocess.TimeoutExpired:
        return {"command": " ".join(cmd), "returncode": None, "counts": {},
                "stdout_tail": f"TIMEOUT after {timeout:g}s"}
    except FileNotFoundError as exc:  # pragma: no cover
        return {"command": " ".join(cmd), "returncode": None, "counts": {},
                "stdout_tail": f"repo checkout not available: {exc}"}


def run_probes(p15_baseline: dict, repo_roots: dict[str, str],
               timeout: float = 1800.0) -> dict:
    """Execute every P15 probe whose repo has a local checkout."""
    results, skipped = {}, {}
    for node in p15_baseline.get("nodes") or []:
        ext, repo = node.get("ext_id"), node.get("repo")
        targets = (node.get("probe") or {}).get("test_targets") or []
        root = repo_roots.get(repo) if repo else None
        if not root or not targets:
            skipped[ext] = f"repo {repo!r} not provided / no test targets recorded"
            continue
        results[ext] = run_probe(root, targets, timeout=timeout)
    return {"schema": PROBE_RESULTS_SCHEMA, "results": results, "skipped": skipped}


def ingest_probe_results(path: str | os.PathLike) -> dict:
    return json.loads(Path(path).read_text())