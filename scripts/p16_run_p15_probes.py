#!/usr/bin/env python3
"""Execute the P15 regression probes against local P16 code checkouts.

Writes ``probe_results.json`` (schema ``p15_probe_results.v1``) which the gate
consumes.  Test targets recorded for P15 are resolved at the *current* head of
each repository: exact path, else a recursive basename search.  Targets that
cannot be resolved are recorded as unresolved instead of being silently
dropped -- an unreproducible P15 probe is an evidence gap, not a pass.

Usage::

    python3 scripts/p16_run_p15_probes.py \
        --checkouts /root/tensor-ecosystem \
        --baseline artifacts/p16_gate/p15_baseline.json \
        --out artifacts/p16_gate/probe_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess  # nosec - runs the project's own test suite
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tensorphysics.campaign.integration_gate.regression import (  # noqa: E402
    PROBE_RESULTS_SCHEMA,
    parse_pytest_counts,
)


def resolve_targets(repo_dir: pathlib.Path, targets: list[str]) -> tuple[list[str], list[str]]:
    """Map P15 test paths onto the files present at the current repo head."""
    found, missing = [], []
    for t in targets:
        p = repo_dir / t
        if p.is_file():
            found.append(t)
            continue
        hits = sorted((repo_dir / "tests").rglob(pathlib.Path(t).name)) if (repo_dir / "tests").is_dir() else []
        if hits:
            found.append(str(hits[0].relative_to(repo_dir)))
        else:
            missing.append(t)
    return found, missing


def missing_target_detail(repo_dir: pathlib.Path, paths: list[str]) -> list[dict]:
    """Classify each missing P15 probe target against the current head.

    ``removed_from_baseline`` means the deliverable commit is an ancestor of
    HEAD but the file is gone: the verification case was dropped from the P16
    baseline.  ``not_in_baseline_history`` means the checkout predates the P15
    merge (stale mirror) -- inconclusive, never reported as a regression.
    """
    out: list[dict] = []
    for t in paths:
        last = subprocess.run(
            ["git", "-C", str(repo_dir), "log", "--all", "--format=%h %cI", "-1", "--", t],
            capture_output=True, text=True).stdout.strip()
        sha = last.split(" ")[0] if last else None
        ancestor = None
        if sha:
            rc = subprocess.run(["git", "-C", str(repo_dir), "merge-base", "--is-ancestor", sha, "HEAD"],
                                capture_output=True, text=True).returncode
            ancestor = rc == 0
        out.append({
            "path": t,
            "last_commit": last or None,
            "deliverable_in_baseline_history": ancestor,
            "classification": ("removed_from_baseline" if ancestor
                               else "not_in_baseline_history"),
        })
    return out


def probe_repo(repo_dir: pathlib.Path, targets: list[str], timeout: float) -> dict:
    found, missing = resolve_targets(repo_dir, targets)
    head = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "--abbrev-ref", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    head_date = subprocess.run(["git", "-C", str(repo_dir), "log", "-1", "--format=%cI"],
                               capture_output=True, text=True).stdout.strip()
    if not found:
        return {"command": None, "returncode": None, "counts": {}, "resolvable": False,
                "stdout_tail": "no P15 probe target resolvable at this head",
                "resolved_targets": [], "unresolved_targets": missing,
                "unresolved_detail": missing_target_detail(repo_dir, missing),
                "repo_head": head, "repo_branch": branch, "repo_head_date": head_date}
    cmd = [sys.executable, "-m", "pytest", *found, "-q", "-p", "no:cacheprovider"]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(repo_dir / "src"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    try:
        proc = subprocess.run(cmd, cwd=str(repo_dir), capture_output=True, text=True,  # nosec
                              timeout=timeout, env=env)
        out = (proc.stdout or "") + (proc.stderr or "")
        return {"command": " ".join(cmd), "returncode": proc.returncode, "resolvable": True,
                "counts": parse_pytest_counts(out), "stdout_tail": out[-4000:],
                "resolved_targets": found, "unresolved_targets": missing,
                "unresolved_detail": missing_target_detail(repo_dir, missing),
                "repo_head": head, "repo_branch": branch, "repo_head_date": head_date}
    except subprocess.TimeoutExpired:
        return {"command": " ".join(cmd), "returncode": None, "counts": {},
                "stdout_tail": f"TIMEOUT after {timeout:g}s",
                "resolved_targets": found, "unresolved_targets": missing,
                "repo_head": head, "repo_branch": branch}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkouts", required=True, help="directory holding <Repo>/main checkouts")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--only", default="", help="comma separated repo names to probe")
    ap.add_argument("--merge", action="store_true",
                    help="merge into an existing --out file instead of overwriting")
    args = ap.parse_args()

    root = pathlib.Path(args.checkouts)
    baseline = json.loads(pathlib.Path(args.baseline).read_text())
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    results, skipped, heads = {}, {}, {}
    out_path = pathlib.Path(args.out)
    if args.merge and out_path.is_file():
        prev = json.loads(out_path.read_text())
        results.update(prev.get("results") or {})
        skipped.update(prev.get("skipped") or {})
        heads.update(prev.get("repo_heads") or {})
    for node in baseline.get("nodes") or []:
        ext, repo = node.get("ext_id"), node.get("repo")
        targets = (node.get("probe") or {}).get("test_targets") or []
        if only and repo not in only:
            continue
        repo_dir = root / str(repo)
        if not repo_dir.is_dir():
            skipped[ext] = f"checkout {repo_dir} absent"
            continue
        results[ext] = probe_repo(repo_dir, targets, args.timeout)
        heads[repo] = {"head": results[ext].get("repo_head"),
                       "branch": results[ext].get("repo_branch")}
        print(f"{ext}: {results[ext]['counts']} unresolved={results[ext]['unresolved_targets']}",
              file=sys.stderr)

    payload = {"schema": PROBE_RESULTS_SCHEMA, "results": results, "skipped": skipped,
               "repo_heads": heads,
               "note": ("Probes re-executed on the current head of each solver repository; "
                        "unresolved_targets lists P15 test files absent at that head.")}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    print(json.dumps({k: v["counts"] for k, v in results.items()}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())