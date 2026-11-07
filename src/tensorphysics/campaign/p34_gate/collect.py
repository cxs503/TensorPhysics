"""Evidence collection for the P34 integration gate.

The collector is deliberately dumb: it *observes* the node worktrees (read only)
and records what is actually there -- git state, declared source modules, pytest
runs, artifact files with sha256 and format validation.  It never infers a
deliverable from a claim; an absent file yields an entry with ``present: false``.

Two properties matter for auditability:

* every recorded artifact carries ``sha256`` and, for PNG, the magic-byte check
  result, so a later reader can re-verify the byte stream;
* every pytest record carries the node's ``head_sha`` and the interpreter
  architecture, because a green run on a different commit is not evidence about
  the delivered commit.

Node worktrees are addressed as ``{root_template}/m{mission_id}`` and may be
overridden per node through ``roots``.
"""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

from . import spec as S

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
PNG_IEND = b"IEND\xaeB`\x82"

EVIDENCE_SCHEMA = "p34_node_evidence.v1"
COLLECTION_SCHEMA = "p34_collection_index.v1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, limit: int = 64 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(root: Path, *args: str, timeout: float = 30.0) -> tuple[int, str]:
    try:
        p = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True, text=True, timeout=timeout,
        )
        return p.returncode, (p.stdout or p.stderr).strip()
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - env dependent
        return 127, f"git error: {exc}"


def check_png(path: Path) -> dict:
    """PNG validity: magic bytes on both ends (header and IEND trailer)."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return {"ok": False, "note": f"unreadable: {exc}"}
    header_ok = raw[:8] == PNG_MAGIC
    tail_ok = PNG_IEND in raw[-32:]
    return {
        "ok": bool(header_ok and tail_ok and len(raw) > 128),
        "note": ("magic+IEND ok" if header_ok and tail_ok else
                 f"magic={header_ok} iend={tail_ok}"),
        "bytes": len(raw),
    }


def check_json(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return {"ok": False, "note": f"unreadable: {exc}"}
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        return {"ok": False, "note": f"invalid json: {exc}"}
    return {"ok": True, "note": f"json ok ({type(obj).__name__})", "bytes": len(text)}


def _describe_path(root: Path, rel: str) -> dict:
    path = root / rel
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file())
        digest = hashlib.sha256()
        for p in files:
            digest.update(str(p.relative_to(root)).encode())
            digest.update(p.read_bytes()[:1 << 16])
        return {"path": rel, "present": True, "kind": "dir", "n_files": len(files),
                "sha256": digest.hexdigest()}
    if path.is_file():
        return {"path": rel, "present": True, "kind": "file",
                "sha256": sha256_file(path), "bytes": path.stat().st_size}
    return {"path": rel, "present": False, "kind": "missing"}


def collect_declared(root: Path, node: S.NodeSpec) -> list[dict]:
    return [_describe_path(root, rel) for rel in (*node.modules, *node.tests)]


def collect_artifacts(root: Path, node: S.NodeSpec) -> list[dict]:
    """Resolve each declared artifact family to concrete files, validated."""
    out: list[dict] = []
    for req in node.artifacts:
        matches = sorted(root.glob(req.pattern))
        matches = [m for m in matches if m.is_file()]
        entry = {"family": req.pattern, "kind": req.kind, "note": req.note,
                 "n_matches": len(matches), "files": []}
        for m in matches[:64]:
            rec = {"path": str(m.relative_to(root)), "sha256": sha256_file(m),
                   "bytes": m.stat().st_size}
            rec.update(check_png(m) if req.kind == "png" else check_json(m))
            entry["files"].append(rec)
        entry["ok"] = bool(matches) and all(f["ok"] for f in entry["files"])
        out.append(entry)
    return out


# --------------------------------------------------------------------------
# pytest evidence
# --------------------------------------------------------------------------
@dataclass
class PytestRecord:
    command: str = ""
    exit_code: int | None = None
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    collected: int = 0
    duration_s: float = 0.0
    timed_out: bool = False
    arch: str = ""
    head_sha: str = ""
    junit: str = ""
    note: str = ""
    test_files_run: list[str] = field(default_factory=list)

    @property
    def green(self) -> bool:
        return (self.exit_code == 0 and self.failed == 0 and self.errors == 0
                and self.collected > 0 and not self.timed_out)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["green"] = self.green
        return d


def _parse_junit(path: Path) -> dict:
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0, "collected": 0}
    if not path.is_file():
        return counts
    try:
        tree = ElementTree.parse(path)
    except ElementTree.ParseError:
        return counts
    root = tree.getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    for suite in suites:
        for tc in suite.iter("testcase"):
            counts["collected"] += 1
            kids = {child.tag for child in tc}
            if "failure" in kids:
                counts["failed"] += 1
            elif "error" in kids:
                counts["errors"] += 1
            elif "skipped" in kids:
                counts["skipped"] += 1
            else:
                counts["passed"] += 1
    return counts


def run_node_tests(root: Path, node: S.NodeSpec, out_dir: Path,
                   timeout: float = 900.0) -> PytestRecord:
    """Run the node's declared pytest files inside its own worktree."""
    rec = PytestRecord(arch=platform.machine(), junit="")
    _, head = _git(root, "rev-parse", "HEAD")
    rec.head_sha = head if len(head) == 40 else ""
    present = [t for t in node.tests if (root / t).is_file()]
    rec.test_files_run = present
    if not present:
        rec.note = "no declared test file present in node worktree"
        return rec
    out_dir.mkdir(parents=True, exist_ok=True)
    # absolute: the subprocess runs with cwd=<node worktree> and evidence must
    # never be written into another repository's working tree
    junit = (out_dir / f"{node.ext_id}_junit.xml").resolve()
    cmd = [sys.executable, "-m", "pytest", *present, "-q", "-p", "no:cacheprovider",
           f"--junitxml={junit}"]
    rec.command = " ".join(cmd)
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True,
                              timeout=timeout)
        rec.exit_code = proc.returncode
        rec.note = (proc.stdout or "").strip().splitlines()[-1][:300] if proc.stdout else ""
    except subprocess.TimeoutExpired:
        rec.timed_out = True
        rec.exit_code = 124
        rec.note = f"pytest exceeded {timeout:.0f}s"
    rec.duration_s = round(time.time() - t0, 2)
    counts = _parse_junit(junit)
    for key, value in counts.items():
        setattr(rec, key, value)
    if junit.is_file():
        rec.junit = str(junit)
    return rec


# --------------------------------------------------------------------------
# node / campaign assembly
# --------------------------------------------------------------------------
def resolve_root(node: S.NodeSpec, roots: dict[str, Path] | None,
                 template: str = "/root/tp-agent/work/m{mission_id}") -> Path | None:
    if roots and node.ext_id in roots:
        return Path(roots[node.ext_id])
    candidate = Path(template.format(mission_id=node.mission_id))
    return candidate if candidate.is_dir() else None


def collect_node(node: S.NodeSpec, root: Path | None,
                 pytest_rec: PytestRecord | None = None,
                 report: dict | None = None) -> dict:
    ev: dict = {
        "schema": EVIDENCE_SCHEMA,
        "ext_id": node.ext_id,
        "mission_id": node.mission_id,
        "repo": node.repo,
        "title": node.title,
        "worktree": str(root) if root else None,
        "min_tests": node.min_tests,
        "collector": {
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": platform.python_version(),
            "arch": platform.machine(),
        },
    }
    if root is None:
        ev.update({"git": {"available": False}, "declared": [], "artifacts": [],
                   "pytest": (pytest_rec or PytestRecord()).as_dict(),
                   "report": report, "delivered": False,
                   "gaps": ["node worktree not found"]})
        return ev

    rc, branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    rc2, head = _git(root, "rev-parse", "HEAD")
    rc3, ahead = _git(root, "rev-list", "--count", "origin/main..HEAD")
    rc4, dirty = _git(root, "status", "--porcelain")
    ev["git"] = {
        "available": rc == 0,
        "branch": branch if rc == 0 else "",
        "head": head if rc2 == 0 else "",
        "commits_ahead_of_origin_main": int(ahead) if rc3 == 0 and ahead.isdigit() else None,
        "dirty_files": len([ln for ln in dirty.splitlines() if ln.strip()]) if rc4 == 0 else None,
    }
    ev["declared"] = collect_declared(root, node)
    ev["artifacts"] = collect_artifacts(root, node)
    ev["pytest"] = (pytest_rec or PytestRecord()).as_dict()
    ev["report"] = report

    gaps: list[str] = []
    for rec in ev["declared"]:
        if not rec["present"]:
            gaps.append(f"missing {rec['path']}")
    for fam in ev["artifacts"]:
        if not fam["ok"]:
            gaps.append(f"artifact family incomplete: {fam['family']}")
    if not ev["pytest"] or not ev["pytest"].get("green"):
        gaps.append("node pytest not green/ not run")
    if not report:
        gaps.append("no platform report recorded")
    ev["gaps"] = gaps
    ev["delivered"] = not gaps
    return ev


def collect_campaign(nodes: tuple[S.NodeSpec, ...] = S.NODES,
                     roots: dict[str, Path] | None = None,
                     template: str = "/root/tp-agent/work/m{mission_id}",
                     out_dir: Path | None = None,
                     run_tests: bool = False,
                     test_timeout: float = 900.0,
                     reports: dict[str, dict] | None = None) -> dict:
    reports = reports or {}
    records: list[dict] = []
    for node in nodes:
        root = resolve_root(node, roots, template)
        rec = None
        if run_tests and root is not None and out_dir is not None:
            rec = run_node_tests(root, node, out_dir / "junit", timeout=test_timeout)
        records.append(collect_node(node, root, rec, reports.get(node.ext_id)))
    index = {
        "schema": COLLECTION_SCHEMA,
        "campaign_id": S.CAMPAIGN_ID,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arch": platform.machine(),
        "root_template": template,
        "nodes": records,
        "summary": {
            "n_nodes": len(records),
            "n_worktree_found": sum(1 for r in records if r["worktree"]),
            "n_delivered": sum(1 for r in records if r.get("delivered")),
            "n_pytest_green": sum(1 for r in records if (r.get("pytest") or {}).get("green")),
        },
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "nodes").mkdir(parents=True, exist_ok=True)
        for rec in records:
            (out_dir / "nodes" / f"{rec['ext_id']}.json").write_text(
                json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "collection_index.json").write_text(
            json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return index