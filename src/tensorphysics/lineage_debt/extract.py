"""Debt extraction: report error tables and terminal node states -> ledger items.

Two extraction surfaces, both driven by the injected vocabulary:

1. **error rows** -- every markdown table in a report is scanned for columns
   whose header matches the configured error-header patterns; each numeric cell
   is converted to percent, resolved against the configured threshold
   (per-quantity > per-domain > default), and emitted as an
   :class:`~tensorphysics.lineage_debt.model.ErrorObservation`.  Cells whose
   header repeats (one error column per condition group) inherit their
   condition label from the longest common prefix of the sibling headers in
   their own group -- a purely structural rule, no domain knowledge.

2. **terminal nodes** -- a node whose terminal status is in
   ``terminal_statuses`` (default ``blocked``/``failed``) is a payable debt in
   its own right: the gap is real even when no report was ever produced (the
   two blocked nodes of the replay plan have no report at all).

Precision guard: a verdict word alone (``FAIL`` inside a table cell) never
creates a debt -- only a numeric error above its threshold does.  Verdict words
are kept as annotations on the observation so that reviewers can see the report
already called the row an honest FAIL.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from .config import VocabConfig
from .model import DebtItem, ErrorObservation, lineage_steps, sha256_text

__all__ = [
    "Table",
    "parse_markdown_tables",
    "parse_numeric_pct",
    "extract_report_observations",
    "observations_to_debts",
    "extract_node_terminal_debts",
    "DEFAULT_TERMINAL_STATUSES",
]

DEFAULT_TERMINAL_STATUSES = ("blocked", "failed")

_NUM_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


class Table:
    """A parsed markdown pipe table with line provenance."""

    __slots__ = ("index", "header", "rows", "line_no")

    def __init__(self, index: int, header: list[str], rows: list[tuple[int, list[str]]], line_no: int):
        self.index = index
        self.header = header
        self.rows = rows          # (1-based line number, cells)
        self.line_no = line_no

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Table(index={self.index}, cols={len(self.header)}, rows={len(self.rows)})"


def _cells(line: str) -> list[str]:
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    return [c.strip() for c in body.split("|")]


def _is_separator(cells: Iterable[str]) -> bool:
    cells = list(cells)
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", c.replace(" ", "")) or c == "" for c in cells)


def parse_markdown_tables(markdown: str) -> list[Table]:
    """Split a markdown document into pipe tables (header + data rows)."""
    lines = (markdown or "").splitlines()
    tables: list[Table] = []
    i = 0
    n = len(lines)
    while i < n:
        if lines[i].strip().startswith("|") and lines[i].count("|") >= 2:
            start = i
            block: list[tuple[int, list[str]]] = []
            while i < n and lines[i].strip().startswith("|") and lines[i].count("|") >= 2:
                block.append((i + 1, _cells(lines[i])))
                i += 1
            if block:
                header = block[0][1]
                rest = [b for b in block[1:] if not _is_separator(b[1])]
                tables.append(Table(index=len(tables), header=header, rows=rest, line_no=start + 1))
            continue
        i += 1
    return tables


def parse_numeric_pct(raw: str, header: str = "", fraction_interpretation: bool = True) -> float | None:
    """Convert a table cell to a signed percentage.

    * ``"+25.2%"`` -> ``25.2`` (percent sign is authoritative);
    * ``"0.019"`` in a column without a percent sign -> ``1.9`` when
      ``fraction_interpretation`` (bare values with ``|v| <= 1`` are fractions);
    * any other bare number is taken at face value (already percent).

    The raw cell is always preserved on the observation, so the convention is
    auditable rather than hidden.
    """
    text = (raw or "").replace("*", "").replace("`", "").replace(" ", "")
    text = text.replace("−", "-").replace("＋", "+").replace(",", "")
    if not text:
        return None
    has_pct = "%" in text
    text = text.replace("%", "").replace("％", "")
    if not _NUM_RE.match(text):
        return None
    value = float(text)
    header_has_pct = "%" in (header or "")
    if has_pct or header_has_pct:
        return value
    if fraction_interpretation and abs(value) <= 1.0:
        return value * 100.0
    return value


def _common_prefix_tokens(cells: list[str]) -> str:
    """Longest common leading token sequence of the given header cells."""
    if not cells:
        return ""
    token_lists = [c.split() for c in cells if c.strip()]
    if not token_lists:
        return ""
    prefix: list[str] = []
    for parts in zip(*token_lists):
        if len(set(parts)) == 1:
            prefix.append(parts[0])
        else:
            break
    return " ".join(prefix).strip()


def _markers(cfg: VocabConfig, cells: Iterable[str]) -> tuple[str, ...]:
    hits: list[str] = []
    fails = cfg.fail_markers
    passes = tuple(str(m).lower() for m in ((cfg.table_contract.get("verdict_markers") or {}).get("pass") or ()))
    for cell in cells:
        low = (cell or "").lower()
        if not low:
            continue
        if any(m and m in low for m in fails):
            hits.append("fail")
        elif any(m and m in low for m in passes):
            hits.append("pass")
    return tuple(sorted(set(hits)))


def _split_parenthetical(label: str) -> tuple[str, str]:
    """``"q1 (case_A)"`` -> ``("q1", "case_A")``; keeps the last bracket group as a tag."""
    text = (label or "").strip()
    m = re.findall(r"[\(（]\s*([^\)）]{1,40})\s*[\)）]", text)
    if not m:
        return text, ""
    tag = m[-1].strip()
    stripped = re.sub(r"[\s]*[\(（]\s*" + re.escape(tag) + r"\s*[\)）][\s]*$", "", text).strip()
    return (stripped or text), tag


def _condition_label(condition: str, cfg: VocabConfig) -> str:
    """Upper-case a condition tag and normalise numeric spellings (``1.0`` -> ``1``)."""
    text = (condition or "").strip()
    if not text:
        return cfg.states.get("wildcard_condition", "ALL")
    if _is_numeric_label(text):
        try:
            value = float(text.replace("%", "").strip())
        except (TypeError, ValueError):
            return text.upper()
        # integral values lose their padding ("1.0" -> "1"), fractions keep digits
        # ("0.50" -> "0.5") so distinct conditions never collapse into one
        return "%g" % value
    return text.upper()


def _is_numeric_label(label: str) -> bool:
    """A purely numeric row label is a parameter value, i.e. a condition tag."""
    text = (label or "").replace("*", "").replace("`", "").strip()
    text = text.replace("%", "").replace(" ", "")
    return bool(_NUM_RE.match(text)) if text else False


def _report_domain_blob(report: Mapping[str, Any]) -> str:
    tags = report.get("tags") or []
    return " ".join([str(report.get("title") or ""), str(report.get("summary") or ""),
                     str(report.get("scenario_key") or "")] + [str(t) for t in tags])


def _group_condition(header: list[str], error_idx: int, prev_error_idx: int | None) -> str:
    """Condition label for one error column, from its own header group."""
    start = 0 if prev_error_idx is None else prev_error_idx + 1
    segment = [h for h in header[start:error_idx] if h.strip()]
    # the first column is usually the row label: drop it from the group scan,
    # but only for the leading group (later groups start after an error column)
    if start == 0 and len(segment) > 1:
        segment = segment[1:]
    return _common_prefix_tokens(segment)


def _strip_error_word(header_cell: str, cfg: VocabConfig) -> str:
    """Remove the error token from a header cell: ``"幅值误差"`` -> ``"幅值"``."""
    text = header_cell
    for pat in sorted(cfg.error_header_patterns, key=len, reverse=True):
        text = re.sub(re.escape(pat), "", text, flags=re.IGNORECASE)
    return text.strip(" _-*/|()（）:：")


def extract_report_observations(report: Mapping[str, Any], cfg: VocabConfig,
                                plan_id: int | None = None, node_id: int | None = None,
                                mission_id: int | None = None) -> list[ErrorObservation]:
    """Harvest every numeric error cell of one report (pass and fail alike)."""
    markdown = str(report.get("markdown") or "")
    report_id = report.get("id")
    rid = int(report_id) if report_id is not None else None
    created = str(report.get("created_at") or "")
    rsha = sha256_text(markdown)
    fraction = bool(cfg.table_contract.get("fraction_interpretation", True))
    report_domain = cfg.domain_by_alias(_report_domain_blob(report))
    wildcard = cfg.states.get("wildcard_condition", "ALL")
    out: list[ErrorObservation] = []

    for table in parse_markdown_tables(markdown):
        header = table.header
        qty_idx = None
        for idx, cell in enumerate(header):
            if cfg.quantity_headers and any(h == cell.strip().lower() or (h and h in cell.strip().lower())
                                            for h in cfg.quantity_headers):
                qty_idx = idx
                break
        if qty_idx is None:
            qty_idx = 0
        cond_idx = None
        for idx, cell in enumerate(header):
            if idx == qty_idx:
                continue
            if cfg.condition_headers and any(h == cell.strip().lower() or (h and h in cell.strip().lower())
                                             for h in cfg.condition_headers):
                cond_idx = idx
                break
        error_idx = [idx for idx, cell in enumerate(header) if cfg.is_error_header(cell)]
        if not error_idx:
            continue

        for line_no, cells in table.rows:
            if qty_idx >= len(cells):
                continue
            raw_qty_cell = cells[qty_idx]
            if cfg.is_empty(raw_qty_cell):
                continue
            label, paren_tag = _split_parenthetical(raw_qty_cell)
            numeric_label = _is_numeric_label(label)
            column_cond = ""
            if cond_idx is not None and cond_idx < len(cells) and not cfg.is_empty(cells[cond_idx]):
                column_cond = cells[cond_idx].strip()
            markers = _markers(cfg, cells)
            prev_error: int | None = None
            for ei in error_idx:
                group_cond = _group_condition(header, ei, prev_error)
                prev_error = ei
                if ei >= len(cells) or cfg.is_empty(cells[ei]):
                    continue
                err = parse_numeric_pct(cells[ei], header=header[ei], fraction_interpretation=fraction)
                if err is None:
                    continue

                # quantity / domain resolution, in documented priority order:
                #   (i) the row label itself, (ii) the error-column header group,
                #   (iii) the error header text, (iv) the report's own domain tags
                condition = column_cond or paren_tag or ""
                condition_source = "column" if column_cond else ("label" if paren_tag else "none")
                quantity_label: str | None
                domain = cfg.states.get("unclassified_domain", "unclassified")
                quantity = ""
                domain_source = "none"
                if numeric_label:
                    # a numeric row label is a parameter value, hence a condition
                    condition = condition or label
                    condition_source = condition_source if condition_source != "none" else "numeric_label"
                    candidates = [_strip_error_word(header[ei], cfg), group_cond]
                else:
                    candidates = [label, group_cond, _strip_error_word(header[ei], cfg)]
                for cand in candidates:
                    if not cand or cfg.is_empty(cand):
                        continue
                    d2, q2 = cfg.canonical_quantity(cand)
                    if d2 != cfg.states.get("unclassified_domain", "unclassified"):
                        domain, quantity, domain_source = d2, q2, "vocabulary"
                        break
                    if not quantity:
                        quantity = re.sub(r"\s+", " ", cand).strip()
                        domain_source = "raw_label"
                if domain_source != "vocabulary" and report_domain:
                    domain, domain_source = report_domain, "report_meta"
                if not condition:
                    condition = group_cond or wildcard
                    condition_source = condition_source if condition_source != "none" else "group_prefix"
                threshold = cfg.threshold_for(domain, quantity)
                out.append(ErrorObservation(
                    report_id=rid,
                    plan_id=plan_id,
                    node_id=node_id,
                    mission_id=mission_id,
                    raw_quantity=raw_qty_cell,
                    quantity=quantity,
                    domain=domain,
                    condition=_condition_label(condition, cfg),
                    err_pct=err,
                    threshold_pct=threshold,
                    exceeded=abs(err) > threshold,
                    table_index=table.index,
                    line_no=line_no,
                    header=" | ".join(header),
                    raw_value=cells[ei],
                    report_sha256=rsha,
                    verdict_markers=markers,
                    report_created_at=created,
                    domain_source=domain_source,
                    condition_source=condition_source,
                ))
    out.sort(key=lambda o: (int(o.report_id or 0), o.table_index, o.line_no, o.key))
    return out


def observations_to_debts(observations: list[ErrorObservation], cfg: VocabConfig,
                          plan_chain: list[int] | None = None,
                          node: Mapping[str, Any] | None = None,
                          collect_unresolved: list | None = None) -> list[DebtItem]:
    """Turn exceeded observations into debts; close them with later passing ones.

    Observations whose quantity could not be canonicalised *and* whose domain
    could not be resolved (not even from the report's own tags) are appended to
    ``collect_unresolved`` rather than counted: they are reported in the
    ledger's ``unresolved`` section so that the parser's blind spot stays
    visible instead of polluting the gate.

    ``state`` is *not* a matter of opinion: a debt opens at the first exceeded
    observation and can only be closed by a strictly later observation of the
    same key that sits inside the threshold.  Anything else stays ``debt_open``
    even if a descendant claims to have inherited it.
    """
    open_state = cfg.states.get("open", "debt_open")
    closed_state = cfg.states.get("closed", "debt_closed")
    unclassified = cfg.states.get("unclassified_domain", "unclassified")
    by_key: dict[str, list[ErrorObservation]] = {}
    for obs in observations:
        if obs.domain == unclassified and not obs.quantity:
            # cannot be classified even with the report-level fallback: surface the
            # row for review instead of pretending it is a payable debt
            if collect_unresolved is not None:
                collect_unresolved.append(obs.as_dict())
            continue
        by_key.setdefault(obs.key, []).append(obs)

    debts: list[DebtItem] = []
    for key, obs_list in by_key.items():
        obs_list.sort(key=lambda o: (o.report_created_at, int(o.report_id or 0), o.line_no))
        exceeded = [o for o in obs_list if o.exceeded]
        if not exceeded:
            continue
        first = exceeded[0]
        worst = max(exceeded, key=lambda o: (abs(o.err_pct), int(o.report_id or 0)))
        later_pass = [o for o in obs_list
                      if not o.exceeded and (o.report_created_at, int(o.report_id or 0)) >
                      (worst.report_created_at, int(worst.report_id or 0))]
        state = closed_state if later_pass else open_state
        closed_by = int(later_pass[0].report_id) if later_pass and later_pass[0].report_id is not None else None
        debt = DebtItem(
            key=key,
            domain=first.domain,
            quantity=first.quantity,
            condition=first.condition,
            source_kind="error_row",
            plan_id=first.plan_id,
            node_id=first.node_id,
            mission_id=first.mission_id,
            report_id=first.report_id,
            err_pct=worst.err_pct,
            threshold_pct=worst.threshold_pct,
            state=state,
            closed_by=closed_by,
            lineage=lineage_steps(plan_chain or [], node, first.report_id),
            evidence=[o.as_dict() for o in obs_list],
        )
        debts.append(debt)
    return debts


def extract_node_terminal_debts(nodes: Iterable[Mapping[str, Any]], cfg: VocabConfig,
                                plan_chain: list[int] | None = None,
                                terminal_statuses: Iterable[str] = DEFAULT_TERMINAL_STATUSES) -> list[DebtItem]:
    """Nodes that ended ``blocked``/``failed`` are payable debts of the lineage."""
    terminal = {str(s).lower() for s in terminal_statuses}
    debts: list[DebtItem] = []
    for node in nodes:
        status = str(node.get("status") or "").lower()
        if status not in terminal:
            continue
        meta = node.get("metadata") or {}
        blob = " ".join(str(x) for x in (node.get("name"), meta.get("goal"), meta.get("acceptance_criteria")))
        domain = cfg.domain_by_alias(blob) or cfg.states.get("unclassified_domain", "unclassified")
        reason = node.get("block_reason") or (meta.get("block_reason") if isinstance(meta, Mapping) else None)
        debt = DebtItem(
            key=f"{domain}|node_terminal|node:{node.get('id')}",
            domain=domain,
            quantity=None,
            condition=None,
            source_kind="node_terminal",
            plan_id=int(node.get("plan_id")) if node.get("plan_id") is not None else None,
            node_id=int(node.get("id")) if node.get("id") is not None else None,
            mission_id=int(node.get("mission_id")) if node.get("mission_id") is not None else None,
            report_id=None,
            err_pct=None,
            threshold_pct=None,
            state=cfg.states.get("open", "debt_open"),
            lineage=lineage_steps(plan_chain or [], node, None),
            evidence=[{
                "report_id": None,
                "line_no": 0,
                "node_status": status,
                "block_reason": str(reason) if reason is not None else None,
                "remedy": "no machine-checkable acceptance criterion replaced the terminal gap",
            }],
            node_status=status,
        )
        debts.append(debt)
    return debts