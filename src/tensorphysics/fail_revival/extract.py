"""Extract honest-FAIL records from campaign node acceptance artifacts.

An artifact is the verbatim snapshot written by
``tools/sync_campaign_nodes.py``: the node's goal text (which carries the
pre-registered machine criteria) plus the acceptance report markdown and its
sha256.  Extraction is *purely structural*:

1. the report digest is re-verified before a single record is lifted
   (fail-closed: a tampered artifact yields ``tampered`` and no record);
2. markdown tables are parsed into rows; a row is FAIL-marked when one of its
   cells carries a FAIL marker from ``layout.json``.  Two marker tiers keep
   precision honest: *unambiguous* markers (``FAIL``, ``未达成``, ``未收敛``,
   ``拒绝（诚实`` ...) fire anywhere, while *ambiguous* ones (``否`` / ``no``)
   only fire inside a recognised verdict column and never inside a column with
   pass semantics (``实测被拒``: there "no" means the rejection did **not**
   happen, i.e. a passing negative control).  A pass token (tick) always wins;
3. the row's verbatim text becomes the criterion string and its sha256 the
   criterion digest, so a later revival can be refuted byte-for-byte;
4. structured fields (metric / measured / reference / allowed / error) are
   resolved from the header keywords in ``layout.json`` when present, else stay
   ``None`` -- nothing is invented;
5. an attribution label is assigned by the rule table in ``layout.json`` (row
   first, then the declared report summary) and the matched token is kept as
   ``attribution_basis``.

Zero domain semantics: every lexical token used here is read from
``layout.json``; the module itself contains no engineering-domain term.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
from typing import Any, Mapping, Sequence

from .model import FailRecord, normalise_text, sha256_obj, sha256_text

LAYOUT_FILE = Path(__file__).with_name("layout.json")
ARTIFACT_SCHEMA = "campaign_node_artifact.v1"
_EMPHASIS = re.compile(r"[*_`]+")
_ASCII_WORD = re.compile(r"^[a-z]+$")
_CJK_SINGLE = re.compile(r"^[\u4e00-\u9fff]$")


def load_layout(path: Path | str | None = None) -> dict[str, Any]:
    p = Path(path) if path is not None else LAYOUT_FILE
    return json.loads(p.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# markdown tables
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Table:
    header: tuple[str, ...]
    start_line: int
    rows: tuple[tuple[int, tuple[str, ...]], ...]  # (1-based line number, cells)


def _split_row(line: str) -> tuple[str, ...]:
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    return tuple(c.strip() for c in body.split("|"))


def _is_separator(cells: Sequence[str]) -> bool:
    return bool(cells) and all(
        re.fullmatch(r":?-{2,}:?", c.replace(" ", "")) for c in cells if c != ""
    )


def iter_tables(markdown: str) -> list[Table]:
    lines = markdown.splitlines()
    tables: list[Table] = []
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("|"):
            block: list[tuple[int, str]] = []
            j = i
            while j < len(lines) and lines[j].strip().startswith("|"):
                block.append((j + 1, lines[j]))
                j += 1
            if len(block) >= 2 and _is_separator(_split_row(block[1][1])):
                tables.append(
                    Table(
                        header=_split_row(block[0][1]),
                        start_line=block[0][0],
                        rows=tuple((ln, _split_row(raw)) for ln, raw in block[2:]),
                    )
                )
            i = j
        else:
            i += 1
    return tables


# --------------------------------------------------------------------------- #
# cell classification
# --------------------------------------------------------------------------- #
def _flat(text: str) -> str:
    return normalise_text(_EMPHASIS.sub("", text or ""))


def marker_match(flat: str, token: str) -> bool:
    """Token match with the right granularity for its script.

    ASCII tokens need a word boundary (``no`` must not fire on ``notchfree``);
    single CJK tokens need a following delimiter (``否`` must not fire on
    ``否定``); longer CJK tokens are prefix matches.
    """
    low = flat.lower()
    t = token.lower()
    if _ASCII_WORD.match(t):
        return bool(re.match(r"^" + re.escape(t) + r"\b", low))
    if _CJK_SINGLE.match(t):
        return bool(re.match(r"^" + re.escape(t) + r"($|[（(，,。；;\s])", flat))
    return flat.startswith(token)


def resolve_columns(header: Sequence[str], layout: Mapping[str, Any]) -> dict[str, int]:
    """First matching index per semantic column role (keywords are data)."""
    resolved: dict[str, int] = {}
    flat_header = [re.sub(r"\s+", "", _flat(h)).lower() for h in header]
    for role, keywords in layout["column_keywords"].items():
        for idx, cell in enumerate(flat_header):
            for kw in keywords:
                k = re.sub(r"\s+", "", kw).lower()
                if k and k in cell:
                    resolved[role] = idx
                    break
            if role in resolved:
                break
    return resolved


def resolve_verdict_columns(header: Sequence[str], layout: Mapping[str, Any]) -> frozenset[int]:
    out = set()
    flat_header = [re.sub(r"\s+", "", _flat(h)).lower() for h in header]
    for idx, cell in enumerate(flat_header):
        for kw in layout["column_keywords"]["verdict"]:
            k = re.sub(r"\s+", "", kw).lower()
            if k and k in cell:
                out.add(idx)
                break
    return frozenset(out)


def pass_semantics(header_cell: str | None, layout: Mapping[str, Any]) -> bool:
    if not header_cell:
        return False
    low = _flat(header_cell).lower()
    return any(tok.lower() in low for tok in layout.get("pass_semantics_headers", []))


def classify_cell(text: str, layout: Mapping[str, Any], *,
                  verdict_column: bool = False, pass_column: bool = False) -> str | None:
    """Classify one table cell as ``fail`` / ``withheld`` / ``pass`` / ``None``."""
    flat = _flat(text)
    if not flat:
        return None
    classes = layout["verdict_classes"]
    for tok in layout.get("pass_precedence_tokens", []):
        if tok in flat:
            return "pass"
    for tok in classes["withheld"]:
        if marker_match(flat, tok):
            return "withheld"
    for tok in classes["fail_unambiguous"]:
        if marker_match(flat, tok):
            return "fail"
    for tok in classes["pass"]:
        if marker_match(flat, tok):
            return "pass"
    for tok in classes["fail_ambiguous"]:
        if marker_match(flat, tok):
            if not verdict_column:
                return None
            return "pass" if pass_column else "fail"
    return None


def classify_row(cells: Sequence[str], header: Sequence[str],
                 layout: Mapping[str, Any]) -> tuple[list[str | None], frozenset[int], dict[str, int]]:
    verdict_cols = resolve_verdict_columns(header, layout)
    classes = [
        classify_cell(
            c, layout,
            verdict_column=idx in verdict_cols,
            pass_column=pass_semantics(header[idx] if idx < len(header) else None, layout),
        )
        for idx, c in enumerate(cells)
    ]
    return classes, verdict_cols, resolve_columns(header, layout)


def scan_fail_marks(markdown: str, layout: Mapping[str, Any]) -> dict[int, tuple[str, ...]]:
    """Independent, deliberately dumb FAIL scanner: line number -> row cells.

    This is the denominator of the completeness criterion.  It performs no
    column resolution, so extraction cannot pass by quietly redefining what a
    FAIL mark is.
    """
    marks: dict[int, tuple[str, ...]] = {}
    for table in iter_tables(markdown):
        for line_no, cells in table.rows:
            classes, _, _ = classify_row(cells, table.header, layout)
            if "fail" in classes:
                marks[line_no] = cells
    return marks


def attribute(texts: Sequence[str], layout: Mapping[str, Any]) -> tuple[str, str]:
    hay = " ".join(_flat(t) for t in texts if t).lower()
    for label, tokens in layout["attribution_rules"].items():
        for tok in tokens:
            if tok.lower() in hay:
                return label, tok
    return layout.get("default_attribution", "unclassified"), ""


_THRESHOLD = re.compile(r"^\s*([<≤>≥]=?)\s*([\d.]+)\s*%?\s*[?？]?\s*$")


def threshold_of_header(cell: str) -> str | None:
    """Normalise a threshold column header (``<3 % ?`` -> ``<3%``)."""
    flat = _flat(cell)
    m = _THRESHOLD.match(flat)
    if not m:
        return None
    op, num = m.group(1), m.group(2)
    return "%s%s%s" % (op, num, "%" if "%" in flat else "")


def _pick(cells: Sequence[str], idx: int | None) -> str | None:
    if idx is None or idx >= len(cells):
        return None
    val = normalise_text(cells[idx])
    return val or None


def criterion_block(goal: str | None, layout: Mapping[str, Any]) -> tuple[str, str, tuple[str, ...]]:
    """Slice the pre-registered machine-criteria block out of the node goal."""
    text = goal or ""
    start, heading = -1, ""
    for cand in layout["criterion_headings"]:
        pos = text.find(cand)
        if pos >= 0 and (start < 0 or pos < start):
            start, heading = pos, cand
    if start < 0:
        return "", sha256_text(""), ()
    tail = text[start:]
    stops = [pos for pos in (tail.find(m, 1) for m in layout.get("criterion_stop_markers", [])) if pos > 0]
    block = normalise_text(tail[: min(stops)] if stops else tail)
    body = (block[len(heading):] if block.startswith(heading) else block).lstrip("：: \n\t")
    items = tuple(x for x in (normalise_text(p) for p in re.split(r"(?=\d+\))", body)) if x)
    return block, sha256_text(block), items


# --------------------------------------------------------------------------- #
# artifact extraction
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ArtifactExtraction:
    path: str
    mission_id: int
    plan_id: int | None
    report_id: int | None
    report_sha256: str
    report_sha256_recomputed: str
    criteria_sha256: str
    criteria_item_count: int
    fail_marks: int
    withheld_marks: int
    records: tuple[FailRecord, ...]
    status: str  # "ok" | "tampered" | "no_report"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "mission_id": self.mission_id,
            "plan_id": self.plan_id,
            "report_id": self.report_id,
            "report_sha256": self.report_sha256,
            "report_sha256_recomputed": self.report_sha256_recomputed,
            "criteria_sha256": self.criteria_sha256,
            "criteria_item_count": self.criteria_item_count,
            "fail_marks": self.fail_marks,
            "withheld_marks": self.withheld_marks,
            "record_count": len(self.records),
            "status": self.status,
        }


def extract_artifact(artifact: Mapping[str, Any], *, path: str = "<memory>",
                     layout: Mapping[str, Any] | None = None) -> ArtifactExtraction:
    lay = layout or load_layout()
    report = artifact.get("report") or {}
    mission_id = int(artifact.get("mission_id"))
    plan_id = artifact.get("plan_id")
    block, crit_sha, items = criterion_block(artifact.get("goal"), lay)
    markdown = report.get("markdown") or ""
    recomputed = sha256_text(markdown)
    stamped = report.get("sha256") or ""
    base = dict(
        path=path,
        mission_id=mission_id,
        plan_id=None if plan_id is None else int(plan_id),
        report_id=report.get("id"),
        report_sha256=stamped,
        report_sha256_recomputed=recomputed,
        criteria_sha256=crit_sha,
        criteria_item_count=len(items),
    )
    if not markdown:
        return ArtifactExtraction(records=(), status="no_report", fail_marks=0, withheld_marks=0, **base)
    if stamped and stamped != recomputed:
        return ArtifactExtraction(records=(), status="tampered", fail_marks=0, withheld_marks=0, **base)

    summary = report.get("summary") or ""
    records: list[FailRecord] = []
    fail_marks = 0
    withheld_marks = 0
    for table in iter_tables(markdown):
        for line_no, cells in table.rows:
            classes, verdict_cols, cols = classify_row(cells, table.header, lay)
            if "withheld" in classes:
                withheld_marks += 1
            if "fail" not in classes:
                continue
            fail_marks += 1
            fail_idx = classes.index("fail")
            source = "verdict_column" if fail_idx in verdict_cols else "marker_scan"
            crit_text = normalise_text(" | ".join(cells))
            crit_digest = sha256_text(crit_text)
            metric = _pick(cells, cols.get("metric")) or _pick(cells, 0) or crit_text
            allowed = _pick(cells, cols.get("allowed"))
            allowed_source = "column"
            if allowed is None:
                hdr = table.header[fail_idx] if fail_idx < len(table.header) else ""
                allowed = threshold_of_header(hdr)
                allowed_source = "header_threshold" if allowed else "absent"
            elif not allowed_source:
                allowed_source = "column"
            attribution, basis = attribute(list(cells), lay)
            basis_ref = "row"
            if attribution == lay.get("default_attribution") and summary:
                attribution, tok = attribute([summary], lay)
                basis_ref, basis = "report_summary", tok
            records.append(
                FailRecord(
                    fail_id="FR-" + sha256_obj([mission_id, crit_digest, line_no])[:16],
                    mission_id=mission_id,
                    plan_id=None if plan_id is None else int(plan_id),
                    report_id=report.get("id"),
                    report_sha256=recomputed,
                    metric=metric,
                    verdict="FAIL",
                    criterion_text=crit_text,
                    criterion_sha256=crit_digest,
                    measured=_pick(cells, cols.get("measured")),
                    reference=_pick(cells, cols.get("reference")),
                    allowed=allowed,
                    allowed_source=allowed_source,
                    error=_pick(cells, cols.get("error")),
                    attribution=attribution,
                    attribution_basis=(basis_ref + ":" + basis) if basis else basis_ref,
                    cells=tuple(cells),
                    table_header=tuple(table.header),
                    line_no=line_no,
                    verdict_source=source,
                )
            )
    return ArtifactExtraction(records=tuple(records), status="ok",
                              fail_marks=fail_marks, withheld_marks=withheld_marks, **base)


def load_artifacts(directory: Path | str) -> list[tuple[str, dict[str, Any]]]:
    d = Path(directory)
    out: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(d.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != ARTIFACT_SCHEMA:
            continue
        out.append((str(path), data))
    return out


@dataclass(frozen=True)
class Extraction:
    extractions: tuple[ArtifactExtraction, ...]

    @property
    def records(self) -> tuple[FailRecord, ...]:
        return tuple(
            sorted((r for ex in self.extractions for r in ex.records),
                   key=lambda r: (r.mission_id, r.line_no, r.criterion_sha256))
        )

    @property
    def tampered(self) -> tuple[ArtifactExtraction, ...]:
        return tuple(ex for ex in self.extractions if ex.status == "tampered")

    def to_dict(self) -> dict[str, Any]:
        recs = self.records
        return {
            "artifacts": [ex.to_dict() for ex in self.extractions],
            "artifact_count": len(self.extractions),
            "artifact_with_report": sum(1 for e in self.extractions if e.status != "no_report"),
            "record_count": len(recs),
            "fail_marks": sum(e.fail_marks for e in self.extractions),
            "withheld_marks": sum(e.withheld_marks for e in self.extractions),
            "tampered": [e.path for e in self.tampered],
            "by_mission": {str(k): sum(1 for r in recs if r.mission_id == k)
                           for k in sorted({r.mission_id for r in recs})},
        }


def extract_directory(directory: Path | str, layout: Mapping[str, Any] | None = None) -> Extraction:
    lay = layout or load_layout()
    exs = tuple(
        extract_artifact(data, path=path, layout=lay) for path, data in load_artifacts(directory)
    )
    return Extraction(extractions=tuple(sorted(exs, key=lambda e: e.mission_id)))


# --------------------------------------------------------------------------- #
# cross-check (C1b): re-read bytes from disk, independent of the parse above
# --------------------------------------------------------------------------- #
def cross_check(extraction: Extraction, directory: Path | str, *, sample: int = 20) -> dict[str, Any]:
    """Verify criterion strings and report digests against the raw files."""
    by_path = {str(Path(p)): data for p, data in load_artifacts(directory)}
    checked: list[dict[str, Any]] = []
    records = sorted(extraction.records, key=lambda r: (r.mission_id, r.line_no))
    for rec in records[:sample]:
        ex = next(e for e in extraction.extractions if e.mission_id == rec.mission_id)
        markdown = (by_path[ex.path].get("report") or {}).get("markdown") or ""
        lines = markdown.splitlines()
        report_ok = sha256_text(markdown) == rec.report_sha256
        raw_line = lines[rec.line_no - 1] if rec.line_no - 1 < len(lines) else ""
        crit_ok = (
            normalise_text(" | ".join(_split_row(raw_line))) == rec.criterion_text
            and sha256_text(rec.criterion_text) == rec.criterion_sha256
        )
        checked.append({
            "fail_id": rec.fail_id,
            "mission_id": rec.mission_id,
            "line_no": rec.line_no,
            "criterion_sha256": rec.criterion_sha256,
            "criterion_matches_raw": crit_ok,
            "report_sha256_matches_raw": report_ok,
        })
    ok = all(c["criterion_matches_raw"] and c["report_sha256_matches_raw"] for c in checked)
    return {
        "sample_requested": sample,
        "sample_checked": len(checked),
        "all_match": ok,
        "criterion_match_rate": (
            sum(1 for c in checked if c["criterion_matches_raw"]) / len(checked) if checked else 1.0
        ),
        "report_digest_match_rate": (
            sum(1 for c in checked if c["report_sha256_matches_raw"]) / len(checked) if checked else 1.0
        ),
        "checked": checked,
    }