"""Criteria evaluators for the P34 campaign integration gate (C1..C5).

Design rules that keep this honest:

* a criterion never becomes ``PASS`` because a node said so; every PASS is
  recomputed from recorded numbers (test counts, file bytes, photo magic, AST);
* missing upstream evidence is ``WITHHELD`` -- never a pass, never a fail;
* an evaluation function returns *numbers plus the paths* that produced them, so
  the verdict JSON can be re-checked by a third party.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import spec as S

PASS, FAIL, WITHHELD = "PASS", "FAIL", "WITHHELD"


@dataclass
class Verdict:
    criterion_id: str
    label: str
    status: str
    detail: str
    metrics: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "criterion_id": self.criterion_id,
            "label": self.label,
            "status": self.status,
            "detail": self.detail,
            "metrics": self.metrics,
            "evidence": self.evidence,
        }


def _aggregate(statuses) -> str:
    statuses = list(statuses)
    if not statuses:
        return WITHHELD
    if FAIL in statuses:
        return FAIL
    if all(s == PASS for s in statuses):
        return PASS
    return WITHHELD


# --------------------------------------------------------------------------
# C1 -- node pytest green
# --------------------------------------------------------------------------
def evaluate_c1(nodes: list[dict]) -> Verdict:
    per_node, statuses = [], []
    for ev in nodes:
        rec = ev.get("pytest") or {}
        green = bool(rec.get("green"))
        test_files = rec.get("test_files_run") or []
        if not rec or rec.get("command") in (None, ""):
            status = WITHHELD
            reason = "no pytest run recorded for this node worktree"
        elif not test_files:
            status = WITHHELD
            reason = "declared test file not present yet"
        elif rec.get("timed_out"):
            status = FAIL
            reason = f"pytest timed out after {rec.get('duration_s')}s"
        elif green and rec.get("passed", 0) >= ev.get("min_tests", 1):
            status = PASS
            reason = f"{rec.get('passed')} passed >= min {ev.get('min_tests')}"
        elif green:
            status = FAIL
            reason = (f"only {rec.get('passed')} tests passed, "
                      f"node promised >= {ev.get('min_tests')}")
        else:
            status = FAIL
            reason = (f"exit={rec.get('exit_code')} passed={rec.get('passed')} "
                      f"failed={rec.get('failed')} errors={rec.get('errors')}")
        statuses.append(status)
        per_node.append({"ext_id": ev["ext_id"], "status": status, "reason": reason,
                         "passed": rec.get("passed", 0), "failed": rec.get("failed", 0),
                         "errors": rec.get("errors", 0), "skipped": rec.get("skipped", 0),
                         "min_tests": ev.get("min_tests"),
                         "head_sha": rec.get("head_sha", ""),
                         "junit": rec.get("junit", "")})
    n_green = sum(1 for n in per_node if n["status"] == PASS)
    status = _aggregate(statuses)
    return Verdict(
        "C1", S.CRITERIA["C1"], status,
        f"{n_green}/{len(per_node)} nodes pytest green "
        f"(withheld={sum(1 for s in statuses if s == WITHHELD)}, "
        f"fail={sum(1 for s in statuses if s == FAIL)})",
        metrics={"n_nodes": len(per_node), "n_pass": n_green,
                 "n_withheld": sum(1 for s in statuses if s == WITHHELD),
                 "n_fail": sum(1 for s in statuses if s == FAIL),
                 "total_tests_passed": sum(n["passed"] for n in per_node)},
        evidence={"nodes": per_node},
    )


# --------------------------------------------------------------------------
# C2 -- artifacts present, parseable, real PNG/JSON
# --------------------------------------------------------------------------
def evaluate_c2(nodes: list[dict]) -> Verdict:
    per_node, statuses = [], []
    for ev in nodes:
        fams = ev.get("artifacts") or []
        n_fam = len(fams)
        n_ok = sum(1 for f in fams if f.get("ok"))
        bad = [f["family"] for f in fams if not f.get("ok")]
        n_files = sum(f.get("n_matches", 0) for f in fams)
        if n_fam == 0:
            status, reason = WITHHELD, "no artifact families declared"
        elif n_files == 0:
            status, reason = WITHHELD, "no artifact file on disk for any declared family"
        elif n_ok == n_fam:
            status, reason = PASS, f"{n_ok}/{n_fam} families valid ({n_files} files)"
        else:
            status, reason = FAIL, f"{n_ok}/{n_fam} families valid; incomplete: {bad}"
        statuses.append(status)
        per_node.append({
            "ext_id": ev["ext_id"], "status": status, "reason": reason,
            "n_families": n_fam, "n_families_ok": n_ok, "n_files": n_files,
            "invalid": [{"family": f["family"], "n_matches": f.get("n_matches", 0),
                         "files": [{"path": x["path"], "ok": x.get("ok"),
                                    "note": x.get("note")} for x in f.get("files", [])]}
                        for f in fams if not f.get("ok")],
        })
    status = _aggregate(statuses)
    return Verdict(
        "C2", S.CRITERIA["C2"], status,
        f"{sum(1 for s in statuses if s == PASS)}/{len(per_node)} nodes with complete, "
        f"format-valid artifacts",
        metrics={"n_nodes": len(per_node),
                 "n_pass": sum(1 for s in statuses if s == PASS),
                 "n_fail": sum(1 for s in statuses if s == FAIL),
                 "n_withheld": sum(1 for s in statuses if s == WITHHELD),
                 "total_files": sum(n["n_files"] for n in per_node)},
        evidence={"nodes": per_node},
    )


# --------------------------------------------------------------------------
# C3 -- regression of pre-existing capabilities
# --------------------------------------------------------------------------
def evaluate_c3(regression: dict | None) -> Verdict:
    if not regression or not regression.get("suites"):
        return Verdict("C3", S.CRITERIA["C3"], WITHHELD,
                       "no regression run recorded", metrics={"n_suites": 0},
                       evidence={"suites": []})
    suites = regression["suites"]
    rows, seen = [], set()
    for s in suites:
        green = (s.get("exit_code") == 0 and s.get("failed", 0) == 0
                 and s.get("errors", 0) == 0 and s.get("collected", 0) > 0)
        fam = s.get("family", "unknown")
        seen.add(fam)
        rows.append({"family": fam, "target": s.get("target"), "repo": s.get("repo"),
                     "green": green, "passed": s.get("passed", 0),
                     "failed": s.get("failed", 0), "errors": s.get("errors", 0),
                     "skipped": s.get("skipped", 0), "arch": s.get("arch"),
                     "head_sha": s.get("head_sha"), "junit": s.get("junit"),
                     "duration_s": s.get("duration_s")})
    missing_families = [f for f in S.REQUIRED_REGRESSION_FAMILIES if f not in seen]
    red = [r for r in rows if not r["green"]]
    if red:
        status = FAIL
    elif missing_families:
        status = WITHHELD
    else:
        status = PASS
    detail = (f"{len(rows) - len(red)}/{len(rows)} suites green; "
              f"families covered {len(seen)}/{len(S.REQUIRED_REGRESSION_FAMILIES)}")
    if missing_families:
        detail += f"; missing families: {missing_families}"
    return Verdict("C3", S.CRITERIA["C3"], status, detail,
                   metrics={"n_suites": len(rows),
                            "n_green": len(rows) - len(red),
                            "n_red": len(red),
                            "families_required": list(S.REQUIRED_REGRESSION_FAMILIES),
                            "families_covered": sorted(seen),
                            "families_missing": missing_families,
                            "total_tests_passed": sum(r["passed"] for r in rows)},
                   evidence={"suites": rows})


# --------------------------------------------------------------------------
# C4 -- six-element summary report
# --------------------------------------------------------------------------
REPORT_SECTIONS = {
    "case_introduction": ("案例介绍",),
    "methodology": ("怎么做的", "方法"),
    "results": ("计算结果", "结果"),
    "figures": ("结果图", "图"),
    "notes": ("注意点", "局限"),
    "error_analysis": ("误差判定", "误差分析"),
}
_FIG_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) pairs on level-2 ATX headings.

    Only ``## `` starts a new section: deeper headings (``### 3.1``) and tables
    are part of their section body, otherwise a section whose content lives in
    subsections would look empty.
    """
    parts: list[tuple[str, str]] = []
    head, buf = "", []
    for line in text.splitlines():
        if line.startswith("## ") or line.startswith("# "):
            parts.append((head, "\n".join(buf)))
            head, buf = line.lstrip("# ").strip(), []
        else:
            buf.append(line)
    parts.append((head, "\n".join(buf)))
    return [(h, b) for h, b in parts if h or b.strip()]


def evaluate_c4(report_path: Path | None, root: Path | None = None) -> Verdict:
    if report_path is None or not Path(report_path).is_file():
        return Verdict("C4", S.CRITERIA["C4"], WITHHELD,
                       "campaign summary report not written yet",
                       metrics={"n_sections": 0, "missing": list(REPORT_SECTIONS)},
                       evidence={"report": str(report_path) if report_path else None})
    text = Path(report_path).read_text(encoding="utf-8")
    words = len(re.findall(r"[\u4e00-\u9fff]|[A-Za-z]+", text))
    sections = _split_sections(text)
    # Assign every heading to at most one criterion, by longest matching alias, so
    # that "结果图" cannot be counted both as figures and as results.
    assigned: dict[str, str] = {}
    for head, _body in sections:
        best_key, best_len = None, 0
        for key, aliases in REPORT_SECTIONS.items():
            for alias in aliases:
                if alias in head and len(alias) > best_len:
                    best_key, best_len = key, len(alias)
        if best_key is not None:
            assigned.setdefault(best_key, head)
    found, missing, thin, bodies = [], [], [], {}
    for key in REPORT_SECTIONS:
        head = assigned.get(key)
        if head is None:
            missing.append(key)
            continue
        body = next(b for h, b in sections if h == head)
        bodies[key] = len(re.sub(r"\s", "", body))
        if bodies[key] < 40:
            thin.append(key)
        found.append(key)
    figs = _FIG_RE.findall(text)
    root = Path(root) if root else Path(report_path).resolve().parent.parent
    fig_status = []
    for ref in figs:
        p = (root / ref).resolve() if not ref.startswith("/") else Path(ref)
        ok = p.is_file() and p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        fig_status.append({"path": ref, "exists": p.is_file(), "png_ok": bool(ok)})
    bad_figs = [f for f in fig_status if not (f["exists"] and f["png_ok"])]
    explicit_error_answer = "<3%" in text or "< 3%" in text
    status = PASS
    reasons = []
    if missing:
        status = FAIL
        reasons.append(f"missing sections {missing}")
    if thin:
        status = FAIL
        reasons.append(f"thin sections {thin}")
    if not figs:
        status = FAIL
        reasons.append("no embedded figure")
    if bad_figs:
        status = FAIL
        reasons.append(f"broken figure refs {[f['path'] for f in bad_figs]}")
    if not explicit_error_answer:
        status = FAIL
        reasons.append("no explicit '<3%' verdict in error analysis")
    if status == PASS:
        reasons.append(f"6/6 sections, {len(figs)} figures, {words} words")
    return Verdict("C4", S.CRITERIA["C4"], status, "; ".join(reasons),
                   metrics={"n_sections": len(found), "missing": missing,
                            "thin": thin, "n_figures": len(figs),
                            "broken_figures": [f["path"] for f in bad_figs],
                            "words": words, "section_chars": bodies,
                            "explicit_lt_3pct_verdict": explicit_error_answer},
                   evidence={"report": str(report_path), "figures": fig_status})


# --------------------------------------------------------------------------
# C5 -- zero domain-semantics scan
# --------------------------------------------------------------------------
_ASCII_TOKEN = re.compile(r"[A-Za-z0-9_]+$")


def placeholder_pattern(token: str) -> re.Pattern:
    """Token pattern that does not match inside identifiers.

    A naive substring scan would flag the metric keys of this very module
    (``n_placeholder_hits``), which is exactly the false-positive failure mode
    that makes keyword gates untrustworthy.  ASCII markers are
    therefore matched only on identifier boundaries; CJK markers are matched
    literally because CJK has no identifier boundaries.
    """
    if _ASCII_TOKEN.match(token):
        return re.compile(r"(?<![A-Za-z0-9_])" + re.escape(token) + r"(?![A-Za-z0-9_])")
    return re.compile(re.escape(token))


def scan_placeholders(paths) -> list[dict]:
    hits = []
    for p in paths:
        p = Path(p)
        if not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for token in S.PLACEHOLDER_TOKENS:
            m = placeholder_pattern(token).search(text)
            if m:
                hits.append({"path": str(p), "token": token,
                             "line": text.count("\n", 0, m.start()) + 1})
    return hits


def scan_imports(pkg_dir: Path) -> dict:
    """AST scan: the gate may only import stdlib + its own package."""
    pkg_dir = Path(pkg_dir)
    roots: dict[str, list[str]] = {}
    for py in sorted(pkg_dir.glob("*.py")):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError as exc:  # pragma: no cover - broken source
            roots.setdefault("__syntax_error__", []).append(f"{py.name}: {exc}")
            continue
        for node in ast.walk(tree):
            mods: list[str] = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module]
            for m in mods:
                roots.setdefault(m.split(".")[0], []).append(py.name)
    forbidden = {k: v for k, v in roots.items() if k in S.FORBIDDEN_IMPORT_ROOTS}
    return {"imports": roots, "forbidden": forbidden,
            "ok": not forbidden and "__syntax_error__" not in roots}


def scan_lexicon(files, lexicon=S.DOMAIN_LEXICON) -> dict:
    """Domain vocabulary must not appear in executable workflow logic."""
    hits = []
    for f in files:
        f = Path(f)
        if not f.is_file():
            continue
        text = f.read_text(encoding="utf-8")
        for term in lexicon:
            n = text.count(term)
            if n:
                hits.append({"path": str(f), "term": term, "count": n})
    return {"files": [str(f) for f in files], "hits": hits, "ok": not hits}


def evaluate_c5(pkg_dir: Path, deliverables: list[Path],
                data_modules: tuple[str, ...] = ("spec.py",)) -> Verdict:
    """Zero domain-semantics scan.

    ``spec.py`` is the declarative node table: it legitimately contains domain
    words (node titles) and the unfilled-marker token list itself, so it is exempt
    from the *content* scans.  Everything that executes is scanned.
    """
    pkg_dir = Path(pkg_dir)
    code = [p for p in sorted(pkg_dir.glob("*.py")) if p.name not in data_modules]
    imports = scan_imports(pkg_dir)
    placeholders = scan_placeholders([*deliverables, *code])
    lexicon = scan_lexicon(code)
    status = PASS if (imports["ok"] and not placeholders and lexicon["ok"]) else FAIL
    parts = [f"imports {'clean' if imports['ok'] else 'violating ' + str(list(imports['forbidden']))}",
             f"placeholders {len(placeholders)}",
             f"domain-terms-in-logic {len(lexicon['hits'])}"]
    return Verdict("C5", S.CRITERIA["C5"], status, "; ".join(parts),
                   metrics={"n_placeholder_hits": len(placeholders),
                            "n_lexicon_hits": len(lexicon["hits"]),
                            "n_forbidden_imports": len(imports["forbidden"]),
                            "n_scanned_deliverables": len(deliverables),
                            "n_scanned_modules": len(code)},
                   evidence={"placeholders": placeholders,
                             "forbidden_imports": imports["forbidden"],
                             "lexicon_hits": lexicon["hits"]})
