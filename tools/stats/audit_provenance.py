#!/usr/bin/env python3
"""Audit provenance chain for statistics artifacts (numbers.json + sha256 + git).

Rationale
---------
An audit that can be edited by hand is not an audit.  Every numeric artifact
written through this module carries a verifiable chain:

    numbers.json          canonical (sorted-key, fixed-separator) JSON
    numbers.json.sha256   sha256 of the exact bytes above
    manifest.json         {commit, dirty, python/numpy versions, source hashes,
                           numbers sha256, reproduce command, figures}

``verify_provenance`` recomputes the digest from disk and compares it with the
manifest *and* with the sidecar file, so any post-hoc edit of a number is
detected (the numbers cannot be hand-copied without also rewriting three
mutually-consistent files, and the source hashes pin the generator).

Zero domain semantics: this module only knows about files, hashes and git.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

NUMBERS_FILE = "numbers.json"
MANIFEST_FILE = "manifest.json"
SIDECAR_FILE = "numbers.json.sha256"


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------

def canonical_json(obj: Any) -> bytes:
    """Deterministic byte serialisation (sorted keys, no NaN, fixed float fmt)."""
    return json.dumps(obj, sort_keys=True, indent=2,
                      separators=(",", ": "), allow_nan=False).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(args: list[str], cwd: Path) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                             text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def git_commit(repo_root: str | Path) -> dict:
    """Current commit + dirty flag (empty strings if git is unavailable)."""
    root = Path(repo_root)
    commit = _git(["rev-parse", "HEAD"], root)
    dirty = _git(["status", "--porcelain"], root)
    return {"commit": commit, "dirty": bool(dirty), "dirty_paths": len(dirty.splitlines()) if dirty else 0}


# ---------------------------------------------------------------------------
# write / verify
# ---------------------------------------------------------------------------

def write_provenance(
    out_dir: str | Path,
    numbers: dict,
    source_files: Iterable[str | Path] = (),
    figures: Iterable[str | Path] = (),
    repo_root: str | Path | None = None,
    reproduce_cmd: str = "",
    extra: dict | None = None,
) -> dict:
    """Write numbers.json (+ sha256 sidecar) and manifest.json.  Returns manifest."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    payload = canonical_json(numbers)
    digest = sha256_bytes(payload)
    (out / NUMBERS_FILE).write_bytes(payload)
    (out / SIDECAR_FILE).write_text(f"{digest}  {NUMBERS_FILE}\n", encoding="utf-8")

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    git = git_commit(root)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "numbers_file": NUMBERS_FILE,
        "numbers_sha256": digest,
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "git_dirty_paths": git["dirty_paths"],
        "source_sha256": {str(Path(f)): sha256_file(f) for f in source_files},
        "figure_sha256": {str(Path(f).name): sha256_file(f) for f in figures
                          if Path(f).exists()},
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "reproduce_cmd": reproduce_cmd,
        **({"extra": extra} if extra else {}),
    }
    (out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True),
                                     encoding="utf-8")
    return manifest


def verify_provenance(out_dir: str | Path) -> dict:
    """Recompute digests from disk and check the whole chain is intact."""
    out = Path(out_dir)
    num_path = out / NUMBERS_FILE
    man_path = out / MANIFEST_FILE
    side_path = out / SIDECAR_FILE
    report: dict = {
        "numbers_present": num_path.exists(),
        "manifest_present": man_path.exists(),
        "sidecar_present": side_path.exists(),
        "ok": False,
        "problems": [],
    }
    if not (num_path.exists() and man_path.exists() and side_path.exists()):
        report["problems"].append("missing artifact")
        return report

    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    recomputed = sha256_file(num_path)
    report["recomputed_sha256"] = recomputed
    report["manifest_sha256"] = manifest.get("numbers_sha256")
    report["sidecar_sha256"] = side_path.read_text(encoding="utf-8").split()[0]

    if not (recomputed == report["manifest_sha256"] == report["sidecar_sha256"]):
        report["problems"].append("sha256 mismatch (numbers.json was modified)")
    if not manifest.get("git_commit"):
        report["problems"].append("git_commit missing")
    for src, digest in (manifest.get("source_sha256") or {}).items():
        if not Path(src).exists():
            report["problems"].append(f"source missing: {src}")
            continue
        if sha256_file(src) != digest:
            report["problems"].append(f"source modified: {src}")
    report["git_commit"] = manifest.get("git_commit")
    report["ok"] = not report["problems"]
    return report


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "artifacts/coverage_audit_wave2"
    res = verify_provenance(target)
    print(json.dumps(res, indent=2))
    sys.exit(0 if res["ok"] else 1)