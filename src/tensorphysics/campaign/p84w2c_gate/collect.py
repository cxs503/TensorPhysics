"""Evidence collection for the P84-W2C gate: artifacts, pytest runs, platform state.

The collector never trusts a node's self-description: it re-reads the delivered
files from the node worktree, re-parses JSON, verifies PNG magic and IEND
trailer, records sha256 of every artefact, and *executes* the declared pytest
suite itself (junit XML → normalized JSON).  A node that only claims to have
run tests therefore produces a ``missing`` pytest record.

Everything is stdlib-only (``json``, ``hashlib``, ``subprocess``,
``xml.etree``).  Platform evidence is fetched over HTTP only when a token is
supplied; otherwise the gate degrades to disk-only evidence and says so.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
PNG_IEND = b"IEND\xaeB`\x82"


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def resolve_glob(root: Path, pattern: str) -> list[Path]:
    try:
        return sorted(p for p in root.glob(pattern) if p.is_file())
    except (OSError, ValueError):
        return []


def classify_artifact(path: Path) -> dict[str, Any]:
    """Parse-check a single delivered artefact (JSON parse / PNG magic+IEND)."""
    kind = "png" if path.suffix.lower() == ".png" else "json"
    record: dict[str, Any] = {
        "path": str(path),
        "kind": kind,
        "exists": path.is_file(),
        "parseable": False,
    }
    if not path.is_file():
        record["detail"] = "missing"
        return record
    size = path.stat().st_size
    record["bytes"] = int(size)
    record["sha256"] = sha256_file(path)
    try:
        if kind == "png":
            head = path.open("rb").read(8)
            tail = path.open("rb").read()[-8:]
            record["parseable"] = head == PNG_MAGIC and tail == PNG_IEND
            record["detail"] = "" if record["parseable"] else "bad png magic/trailer"
        else:
            json.loads(path.read_text(encoding="utf-8"))
            record["parseable"] = True
            record["detail"] = ""
    except (OSError, ValueError, UnicodeDecodeError) as exc:  # pragma: no cover - io guard
        record["parseable"] = False
        record["detail"] = f"unparseable: {type(exc).__name__}"
    return record


def scan_artifacts(root: Path, globs: Sequence[str]) -> list[dict[str, Any]]:
    seen: set[Path] = set()
    out: list[dict[str, Any]] = []
    for pattern in globs:
        for path in resolve_glob(root, pattern):
            if path in seen:
                continue
            seen.add(path)
            out.append(classify_artifact(path))
    return sorted(out, key=lambda r: r["path"])


def load_json_payloads(artifacts: Iterable[Mapping[str, Any]]) -> list[tuple[str, Any]]:
    payloads: list[tuple[str, Any]] = []
    for rec in artifacts:
        if rec.get("kind") != "json" or not rec.get("parseable"):
            continue
        try:
            payloads.append((rec["path"], json.loads(Path(rec["path"]).read_text(encoding="utf-8"))))
        except (OSError, ValueError):
            continue
    return payloads


def find_metric(payloads: Sequence[tuple[str, Any]],
                aliases: Sequence[str]) -> tuple[Any, str] | tuple[None, str]:
    """Depth-first search for the first key matching one of ``aliases``.

    Matching is case/punctuation insensitive so that ``C_D0``, ``cd0`` and
    ``cd-0`` are the same key.  Numbers are returned as ``float``; booleans are
    preserved as ``bool`` (a negative control is a boolean statement).
    """
    wanted = [_norm_key(a) for a in aliases]
    ordered = {a: i for i, a in enumerate(wanted)}

    def walk(node: Any, path: str, depth: int = 0) -> tuple[Any, str, int] | None:
        if depth > 12:
            return None
        if isinstance(node, Mapping):
            best: tuple[Any, str, int] | None = None
            for key, value in node.items():
                nk = _norm_key(key)
                if nk in ordered and isinstance(value, (int, float, bool)):
                    candidate = (value, f"{path}.{key}", ordered[nk])
                    if best is None or candidate[2] < best[2]:
                        best = candidate
            if best is not None:
                return best
            for key, value in node.items():
                found = walk(value, f"{path}.{key}", depth + 1)
                if found is not None:
                    return found
            return None
        if isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                found = walk(value, f"{path}[{index}]", depth + 1)
                if found is not None:
                    return found
            return None
        return None

    for path, payload in payloads:
        found = walk(payload, path)
        if found is not None:
            value = found[0]
            if isinstance(value, bool):
                return value, found[1]
            return float(value), found[1]
    return None, ""


# ---------------------------------------------------------------------------
# criterion-number resolution (declared aliases -> canonical block -> scoped block)
# ---------------------------------------------------------------------------
#: token families used to recognise *which* number inside a reported criterion
#: block is the machine-checkable one.  Order matters: the most specific token
#: wins, which keeps `R2_vs_inv_n` from being shadowed by `R2_vs_n`.
_ERROR_TOKENS = (
    "maxrelativeerror", "relativeerror", "relerror", "relerr", "errpct", "reldiff",
    "residualrel", "residual",
    "relativedifference", "maxrelativedifference", "finalrelativedifference",
    "relativedeviation", "deviation", "imbalance", "relativedev", "error",
)
_GOODNESS_TOKENS = (
    "r2vsinvn", "r2linear", "protectionratio", "coverage", "ratio", "r2",
)
_RANGE_TOKENS = (
    "etaovercarnot", "efficiencyratio", "meantemperature", "pinch", "ratio",
)
_BOOL_TOKENS = ("pass", "passed", "allok", "allwithintol", "monotone", "flip")

TOKENS_BY_COMPARATOR = {
    "rel_dev_le": _ERROR_TOKENS,
    "le": _ERROR_TOKENS,
    "ge": _GOODNESS_TOKENS,
    "in_range": _RANGE_TOKENS,
    "is_true": _BOOL_TOKENS,
}


def _numeric_leaves(node: Any, path: str = "", depth: int = 0):
    """Yield ``(normalized_key, key, value, path)`` for numeric/boolean leaves."""
    if depth > 14:
        return
    if isinstance(node, Mapping):
        for key, value in node.items():
            child = f"{path}.{key}" if path else str(key)
            if isinstance(value, bool):
                yield _norm_key(key), str(key), value, child
            elif isinstance(value, (int, float)):
                yield _norm_key(key), str(key), float(value), child
            else:
                yield from _numeric_leaves(value, child, depth + 1)
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            yield from _numeric_leaves(value, f"{path}[{index}]", depth + 1)


def _unit_normalised(key: str, value: float) -> float:
    """Percent-valued keys are folded to a fraction (units are part of the contract)."""
    lowered = _norm_key(key)
    if "pct" in lowered or "percent" in lowered:
        return value / 100.0
    return value


def _criteria_blocks(payloads: Sequence[tuple[str, Any]]):
    """Yield ``(block_path, block)`` for every ``criteria`` container found."""
    def walk(node: Any, path: str, depth: int = 0):
        if depth > 6:
            return
        if isinstance(node, Mapping):
            for key, value in node.items():
                child = f"{path}.{key}" if path else str(key)
                if _norm_key(key) == "criteria":
                    yield child, value
                yield from walk(value, child, depth + 1)
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                yield from walk(value, f"{path}[{index}]", depth + 1)
    for path, payload in payloads:
        yield from walk(payload, path)


def _tokens_hit(tokens: Sequence[str], leaves) -> list[dict[str, Any]]:
    leaves = list(leaves)
    for token in tokens:
        hits = [leaf for leaf in leaves if token in leaf[0]]
        if hits:
            return [{"value": _unit_normalised(leaf[1], leaf[2]) if not isinstance(leaf[2], bool)
                     else leaf[2], "source": leaf[3]} for leaf in hits]
    return []


def criterion_candidates(payloads: Sequence[tuple[str, Any]], crit) -> list[dict[str, Any]]:
    """Resolve every numeric candidate for ``crit`` using a declared precedence.

    1. **declared aliases** (``spec.Criterion.aliases``) -- exact, punctuation
       and case insensitive key match anywhere in the payloads;
    2. **canonical block** -- a ``criteria`` container that names the criterion
       explicitly: list form ``[{"id": "C1", "value": ...}]`` or dict form
       ``{"C1": {...}}``;
    3. **scoped block** -- any path segment that starts with the criterion id
       (``C1_dual_implementation``, ``C3_monotonicity`` ...), from which the most
       specific comparator token picks the number.

    The first phase that yields numbers wins; aggregation (worst case) happens in
    :mod:`criteria`.  Percent-valued keys are normalised to fractions.
    """
    wanted = {_norm_key(a) for a in crit.aliases}
    cid = _norm_key(crit.cid)
    tokens = TOKENS_BY_COMPARATOR.get(crit.comparator, ())

    # (1) declared aliases
    alias_hits: list[dict[str, Any]] = []
    for path, payload in payloads:
        for norm, key, value, source in _numeric_leaves(payload, path):
            if norm in wanted:
                alias_hits.append({"value": _unit_normalised(key, value)
                                   if not isinstance(value, bool) else value,
                                   "source": source})
    if alias_hits:
        return alias_hits

    # (2) canonical criteria block
    block_hits: list[dict[str, Any]] = []
    for block_path, block in _criteria_blocks(payloads):
        if isinstance(block, (list, tuple)):
            for index, entry in enumerate(block):
                if not isinstance(entry, Mapping):
                    continue
                ident = _norm_key(entry.get("id", entry.get("cid", "")))
                if ident != cid:
                    continue
                value = entry.get("value")
                if isinstance(value, bool):
                    block_hits.append({"value": value,
                                       "source": f"{block_path}[{index}].value"})
                elif isinstance(value, (int, float)):
                    block_hits.append({"value": float(value),
                                       "source": f"{block_path}[{index}].value"})
                elif isinstance(value, Mapping):
                    block_hits += _tokens_hit(tokens, _numeric_leaves(
                        value, f"{block_path}[{index}].value"))
                if crit.comparator == "is_true" and isinstance(entry.get("pass"), bool):
                    block_hits.append({"value": entry["pass"],
                                       "source": f"{block_path}[{index}].pass"})
        elif isinstance(block, Mapping):
            for key, value in block.items():
                if _norm_key(key) != cid:
                    continue
                leaf_path = f"{block_path}.{key}"
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    block_hits.append({"value": float(value), "source": leaf_path})
                else:
                    block_hits += _tokens_hit(tokens, _numeric_leaves(value, leaf_path))
    if block_hits:
        return block_hits

    # (3) criterion-id scoped block: token *priority* is resolved over the whole
    # scoped leaf set, so `R2_vs_inv_n` is not shadowed by the generic `R2` token.
    scoped_leaves = []
    for path, payload in payloads:
        for norm, key, value, source in _numeric_leaves(payload, path):
            segments = [_norm_key(seg) for seg in re.split(r"[.\[\]]", source) if seg]
            if any(seg.startswith(cid) for seg in segments):
                scoped_leaves.append((norm, key, value, source))
    return _tokens_hit(tokens, scoped_leaves)


# ---------------------------------------------------------------------------
# pytest evidence
# ---------------------------------------------------------------------------
_COUNT_RE = {
    "collected": re.compile(r"(\d+)\s+tests?\s+collected"),
    "passed": re.compile(r"(\d+)\s+passed"),
    "failed": re.compile(r"(\d+)\s+failed"),
    "errors": re.compile(r"(\d+)\s+errors?"),
    "skipped": re.compile(r"(\d+)\s+skipped"),
}


def _parse_junit(path: Path) -> dict[str, int]:
    tree = ET.parse(path)
    root = tree.getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        return {}
    def _int(name: str) -> int:
        try:
            return int(suite.get(name, "0"))
        except ValueError:
            return 0
    tests = _int("tests")
    return {
        "collected": tests,
        "failed": _int("failures"),
        "errors": _int("errors"),
        "skipped": _int("skipped"),
        "passed": max(tests - _int("failures") - _int("errors") - _int("skipped"), 0),
    }


def _junit_failures(path: Path, limit: int = 10) -> list[str]:
    """Names of failing/erroring test cases (for the gap analysis)."""
    try:
        tree = ET.parse(path)
    except (OSError, ET.ParseError):
        return []
    names: list[str] = []
    for case in tree.iter("testcase"):
        if case.find("failure") is not None or case.find("error") is not None:
            name = f"{case.get('classname', '')}::{case.get('name', '')}".strip(":")
            names.append(name)
            if len(names) >= limit:
                break
    return names


def _parse_stdout(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, pattern in _COUNT_RE.items():
        match = pattern.search(text)
        if match:
            counts[name] = int(match.group(1))
    counts.setdefault("collected", counts.get("passed", 0) + counts.get("failed", 0)
                      + counts.get("errors", 0) + counts.get("skipped", 0))
    return counts


def run_pytest(repo_root: Path,
               tests: Sequence[str],
               record_path: Path,
               timeout: int = 900,
               python: str | None = None) -> dict[str, Any]:
    """Run the declared node suite and persist a normalized JSON record.

    If none of the declared test files exist yet, the collector falls back to
    ``--collect-only`` over ``tests/`` so the gate can still tell "suite exists
    and imports" apart from "node delivered nothing".  The fallback is recorded
    as ``collect-only`` and never counts as a passing suite.
    """
    python = python or sys.executable
    present = [t for t in tests if (repo_root / t).is_file()]
    mode = "executed" if present else "collect-only"
    targets = present if present else (["tests"] if (repo_root / "tests").is_dir() else [])
    record: dict[str, Any] = {
        "declared": list(tests),
        "present": bool(present),
        "parseable": False,
        "mode": mode if targets else "missing",
        "status": "NOT_SUPPORTED",
    }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    if not targets:
        record["detail"] = "no declared test file and no tests/ directory"
        record["record_path"] = _write_json(record_path, record)
        record["parseable"] = True
        return record

    junit = record_path.with_suffix(".junit.xml")
    if junit.exists():
        junit.unlink()
    cmd = [python, "-m", "pytest", *targets, "-q", "-p", "no:cacheprovider",
           "-o", "addopts=", "--continue-on-collection-errors",
           f"--junitxml={junit}"]
    if mode == "collect-only":
        cmd.append("--collect-only")
    started = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True,
                              timeout=timeout)
        stdout, exit_code, timed_out = proc.stdout, proc.returncode, False
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - long suites
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        exit_code, timed_out = 124, True
    except OSError as exc:  # pragma: no cover - missing interpreter
        stdout, exit_code, timed_out = str(exc), 127, False
    duration = round(time.time() - started, 3)

    counts: dict[str, int] = {}
    if junit.is_file():
        try:
            counts = _parse_junit(junit)
        except ET.ParseError:  # pragma: no cover - truncated xml
            counts = {}
    if not counts:
        counts = _parse_stdout(stdout or "")

    record.update(
        collected=int(counts.get("collected", 0)),
        passed=int(counts.get("passed", 0)),
        failed=int(counts.get("failed", 0)),
        errors=int(counts.get("errors", 0)),
        skipped=int(counts.get("skipped", 0)),
        exit_code=int(exit_code),
        duration_s=duration,
    )
    if timed_out:
        record["detail"] = f"timeout after {timeout}s"
    if mode == "collect-only":
        record["status"] = "FAIL"
        record["detail"] = ("declared test file(s) not delivered: "
                            f"{', '.join(tests)}")
    elif record["failed"] or record["errors"] or exit_code != 0:
        record["status"] = "FAIL"
    elif record["collected"] == 0:
        record["status"] = "WITHHELD"
        record["detail"] = "no tests collected"
    else:
        record["status"] = "PASS"
    failures = _junit_failures(junit) if junit.is_file() else []
    if failures:
        record["failures"] = failures
    record["parseable"] = True
    record["record_path"] = _write_json(record_path, record)
    return record


def _load_record(record_path: Path, spec: Any) -> dict[str, Any]:
    """Reuse an already normalized pytest record (``--no-pytest`` runs).

    Reuse is explicit and auditable: the record keeps its own ``mode``/counts and
    is re-parsed from disk, so a stale or placeholder record cannot be laundered
    into fresh evidence.
    """
    if Path(record_path).is_file():
        try:
            record = json.loads(Path(record_path).read_text(encoding="utf-8"))
        except ValueError:
            record = None
        if isinstance(record, dict) and record.get("mode") in (
                "executed", "collect-only", "missing"):
            junit = Path(record_path).with_suffix(".junit.xml")
            if "failures" not in record and junit.is_file():
                names = _junit_failures(junit)
                if names:
                    record["failures"] = names
            record.setdefault("declared", list(spec.tests))
            record["parseable"] = True
            record["reused"] = True
            record["record_path"] = str(record_path)
            return record
    record = {
        "declared": list(spec.tests),
        "present": False,
        "parseable": True,
        "mode": "missing",
        "status": "WITHHELD",
        "detail": "no stored pytest record and suite execution disabled (--no-pytest)",
    }
    record["record_path"] = _write_json(Path(record_path), record)
    return record


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


def collect_node(repo_root: Path,
                 spec: Any,
                 evidence_dir: Path,
                 run_suite: bool = True,
                 timeout: int = 900,
                 platform_status: str = "") -> dict[str, Any]:
    """Collect one node's disk evidence (artifacts + pytest + metric lookup)."""
    root = Path(repo_root)
    artifacts = scan_artifacts(root, spec.artifact_globs) if root.is_dir() else []
    record_path = Path(evidence_dir) / f"{spec.node_key}.pytest.json"
    if not root.is_dir():
        pytest_record = {
            "declared": list(spec.tests), "present": False, "parseable": True,
            "mode": "missing", "status": "NOT_SUPPORTED",
            "detail": f"node worktree absent: {root}"}
        pytest_record["record_path"] = _write_json(record_path, pytest_record)
    elif run_suite:
        pytest_record = run_pytest(root, spec.tests, record_path, timeout=timeout)
    else:
        pytest_record = _load_record(record_path, spec)
    payloads = load_json_payloads(artifacts)
    key_numbers: dict[str, Any] = {}
    metric_sources: dict[str, str] = {}
    candidates: dict[str, list[dict[str, Any]]] = {}
    for crit in spec.criteria:
        found = criterion_candidates(payloads, crit)
        candidates[crit.cid] = found
        value, source = find_metric(payloads, crit.aliases)
        key_numbers[crit.cid] = value
        if source:
            metric_sources[crit.cid] = source
    return {
        "node_key": spec.node_key,
        "mission_id": spec.mission_id,
        "repo": spec.repo,
        "title": spec.title,
        "physics_domain": spec.physics_domain,
        "root": str(root),
        "root_present": root.is_dir(),
        "platform_status": platform_status,
        "artifacts": artifacts,
        "pytest": pytest_record,
        "key_numbers": key_numbers,
        "metric_sources": metric_sources,
        "candidates": candidates,
        "payloads": payloads,
    }


# ---------------------------------------------------------------------------
# platform evidence (optional; token-gated)
# ---------------------------------------------------------------------------
def api_get(base_url: str, path: str, token: str, timeout: float = 20.0) -> dict[str, Any]:
    url = base_url.rstrip("/") + path
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as exc:
        return {"_error": f"{type(exc).__name__}: {exc}"}


def collect_platform(base_url: str | None,
                     token: str | None,
                     plan_id: int,
                     mission_ids: Sequence[int],
                     timeout: float = 20.0) -> dict[str, Any]:
    """Fetch plan/node/mission state; degrade honestly when auth is unavailable."""
    evidence: dict[str, Any] = {"available": False, "plan_id": int(plan_id), "nodes": {},
                                "missions": {}}
    if not base_url or not token:
        evidence["reason"] = "no API token supplied (offline disk-only evidence)"
        return evidence
    plan = api_get(base_url, f"/tensor/campaigns/plans/{plan_id}/nodes", token, timeout)
    if "_error" in plan:
        evidence["reason"] = f"plan lookup failed: {plan['_error']}"
        return evidence
    evidence["available"] = True
    items = plan.get("items", []) if isinstance(plan, dict) else []
    for item in items:
        evidence["nodes"][str(item.get("id"))] = {
            "name": item.get("name"),
            "status": item.get("status"),
            "mission_id": item.get("mission_id"),
            "dependencies": item.get("dependencies"),
        }
    for mission_id in mission_ids:
        row = {"status": None, "report_count": None, "capability_delta": None}
        detail = api_get(base_url, f"/tensor/missions/{mission_id}", token, timeout)
        if "_error" not in detail:
            row["status"] = detail.get("status")
            row["claimed_by"] = detail.get("claimed_by")
        reports = api_get(base_url, f"/tensor/reports?mission_id={mission_id}", token, timeout)
        if "_error" not in reports:
            row["report_count"] = len(reports.get("items", reports if isinstance(reports, list) else []))
        delta = api_get(base_url, f"/tensor/missions/{mission_id}/capability-delta", token, timeout)
        if "_error" not in delta:
            row["capability_delta"] = delta.get("category") or delta.get("items") or delta
        evidence["missions"][str(mission_id)] = row
    return evidence


def env_token() -> tuple[str | None, str | None]:
    base = os.environ.get("TENSOR_API_BASE")
    user = os.environ.get("TENSOR_AGENT_USER")
    password = os.environ.get("TENSOR_AGENT_PASS")
    if not (base and user and password):
        return base, None
    body = json.dumps({"username": user, "password": password}).encode()
    request = urllib.request.Request(
        base.rstrip("/") + "/login", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read())
        return base, payload.get("access_token")
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError):
        return base, None