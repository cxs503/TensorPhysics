"""Read-only harvesting of historical criteria from repository artifacts.

Nothing is written into the source artifacts: the harvester only *reads* JSON reports and
markdown six-element tables and turns every discovered criterion into a
:class:`~.model.CriterionRecord` with full provenance (``source`` path, key names, raw
criterion string).

Discovery rules (deterministic, no domain vocabulary):

``criterion dicts``
    a mapping carrying a criterion-ish key (``criterion`` / ``requirement`` / ``判据``)
    together with a numeric ``value`` / ``actual``; the reference value is taken from the
    same mapping, its ``evidence`` sub-mapping or a paired key (``*_a``/``*_b``,
    ``analytic``/``computed`` ...).
``pair dicts``
    a mapping carrying a measured/reference key pair *and* a tolerance
    (``tol`` / ``tolerance`` / ``rtol`` / ``threshold`` / an inline criterion string).
``markdown tables``
    rows of ``docs/**/*.md`` tables whose header is criterion-ish; the numeric cell
    becomes ``reported_error`` (fraction) because the reference magnitude is usually not
    tabulated -- an absence the re-judge reports as ``WITHHELD``.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping

from .model import CriterionRecord, finite
from .parser import parse_criterion, parse_markdown_criteria_table
from .model import UNPARSED, ParsedCriterion, _is_identifier

__all__ = ["HarvestStats", "build_corpus", "harvest_json", "harvest_markdown", "iter_json_documents"]

#: schema prefix of this package's own artifacts (skipped by the harvester)
OWN_SCHEMA_PREFIX = "tensorphysics.metrology.criteria"

_CRITERION_KEYS = ("criterion", "requirement", "acceptance", "判据", "验收判据", "gate_rule")
_VALUE_KEYS = ("value", "actual", "measured_value", "computed_value")
_ERROR_KEYS = ("relative_error", "rel_error", "rel_err", "error", "error_pct", "deviation", "rel_l2", "rel-L2")
_TOL_KEYS = ("tol", "tolerance", "rtol", "rel_tol", "rel_error_tol", "atol", "threshold", "limit", "eps")
_NAME_KEYS = ("name", "anchor", "cid", "id", "check", "label", "metric", "description", "title")

_MEASURED_TOKENS = ("computed", "numerical", "numeric", "simulated", "sim", "model", "present", "result",
                    "fem", "lbm", "dem", "bem", "cfd", "measured_now", "obtained")
_REFERENCE_TOKENS = ("gold", "analytic", "analytical", "exact", "reference", "ref", "expected", "benchmark",
                     "experimental", "target", "hetenyi", "closed_form")
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_PAIR_RX = re.compile(r"^(?P<stem>.+?)_(?P<lhs>a|1|ref|gold|exact|analytic|true)$", re.IGNORECASE)


@dataclass
class HarvestStats:
    json_files: int = 0
    markdown_files: int = 0
    json_records: int = 0
    markdown_records: int = 0
    with_numeric_pair: int = 0
    with_reported_error_only: int = 0
    skipped_unparsed: int = 0
    criterion_only: int = 0
    unique_records: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


_NUM_STR_RX = re.compile(r"^\s*(?:<=|≤|<|>|>=|≥|~|≈|±)?\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(%?)\s*$")


def _parse_number(v: Any) -> float | None:
    """Numeric coercion that also understands ``"< 0.5%"`` / ``"0.2790%"`` strings."""
    direct = finite(v)
    if direct is not None:
        return direct
    if isinstance(v, str):
        m = _NUM_STR_RX.match(v)
        if m:
            x = finite(float(m.group(1)))
            if x is None:
                return None
            return x / 100.0 if m.group(2) == "%" else x
    return None


def _num(v: Any) -> float | None:
    return _parse_number(v)


def _iter_paths(root: str, subdirs: Iterable[str]) -> Iterator[str]:
    for sub in subdirs:
        base = os.path.join(root, sub)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames
                                 if not d.startswith(".") and d not in ("__pycache__", "node_modules", ".pytest_cache"))
            for fn in sorted(filenames):
                yield os.path.join(dirpath, fn)


def iter_json_documents(root: str, subdirs: Iterable[str] = ("benchmarks", "artifacts", "docs", "runs",
                                                             "outputs", "scenarios", "tests/data", "data")):
    for path in _iter_paths(root, subdirs):
        if not path.endswith(".json"):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue
        # never audit our own output: the metrology artifacts are schema-stamped
        if isinstance(doc, dict) and str(doc.get("schema", "")).startswith(OWN_SCHEMA_PREFIX):
            continue
        yield path, doc


def _walk(node: Any, path: str = "") -> Iterator[tuple[str, Mapping[str, Any]]]:
    for p, n, _chain in _walk_with_chain(node, path):
        yield p, n


def _walk_with_chain(node: Any, path: str = "", chain: tuple = ()) -> Iterator[tuple[str, Mapping[str, Any], tuple]]:
    """Yield ``(json_path, mapping, ancestors)`` -- ancestors enable tolerance inheritance."""
    if isinstance(node, Mapping):
        yield path, node, chain
        for k, v in node.items():
            yield from _walk_with_chain(v, f"{path}.{k}" if path else str(k), chain + (node,))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk_with_chain(v, f"{path}[{i}]", chain)


def _find_tolerance(d: Mapping[str, Any]) -> tuple[float | None, str | None, str]:
    """Return ``(tolerance, source_key, kind)`` with kind in fraction/absolute/unknown."""
    for k in _TOL_KEYS:
        if k in d:
            v = _num(d[k])
            if v is not None and v > 0:
                return v, k, ("fraction" if k in ("rtol", "rel_tol", "rel_error_tol") else "unknown")
    for k in _CRITERION_KEYS:
        text = d.get(k)
        if isinstance(text, str):
            parsed = parse_criterion(text)
            if parsed.tolerance:
                return parsed.tolerance, k, parsed.tolerance_kind
    return None, None, "unknown"


def _pair_in(d: Mapping[str, Any]) -> tuple[float | None, float | None, str | None, str | None]:
    """Find a (measured, reference) numeric pair inside one mapping."""
    numeric = {k: _num(v) for k, v in d.items() if _num(v) is not None}
    if len(numeric) < 2:
        return None, None, None, None
    # explicit naming
    for mk in numeric:
        low = mk.lower()
        if not any(t in low for t in _MEASURED_TOKENS):
            continue
        stem = low
        for suffix in _MEASURED_TOKENS:
            stem = stem.replace(suffix, "")
        for rk in numeric:
            if rk == mk:
                continue
            rlow = rk.lower()
            if any(t in rlow for t in _REFERENCE_TOKENS) and (stem.strip("_") in rlow or stem.strip("_") == ""):
                return numeric[mk], numeric[rk], mk, rk
    # _a / _b pairing (ledger A vs ledger B, exact vs computed)
    for k in numeric:
        m = _PAIR_RX.match(k)
        if not m:
            continue
        stem = m.group("stem")
        for cand in numeric:
            if cand != k and _PAIR_RX.match(cand) and _PAIR_RX.match(cand).group("stem") == stem:  # type: ignore[union-attr]
                if _PAIR_RX.match(cand).group("lhs").lower() in ("a", "1"):  # type: ignore[union-attr]
                    return numeric[k], numeric[cand], k, cand
                if m.group("lhs").lower() in ("a", "1"):
                    return numeric[cand], numeric[k], cand, k
    # generic: ref token anywhere
    rkeys = [k for k in numeric if any(t in k.lower() for t in _REFERENCE_TOKENS)]
    mkeys = [k for k in numeric if any(t in k.lower() for t in _MEASURED_TOKENS)]
    if len(rkeys) == 1 and len(mkeys) == 1 and rkeys[0] != mkeys[0]:
        return numeric[mkeys[0]], numeric[rkeys[0]], mkeys[0], rkeys[0]
    return None, None, None, None


def _array_pair_in(d: Mapping[str, Any]) -> tuple[list[float], list[float], str, str] | None:
    """Element-wise (measured[], reference[]) arrays of equal length inside one mapping."""
    keys = list(d)
    for a in keys:
        if not any(t in a.lower() for t in _REFERENCE_TOKENS):
            continue
        va = d[a]
        if not isinstance(va, list) or not va or not all(isinstance(x, (int, float)) for x in va):
            continue
        for b in keys:
            if a == b or not any(t in b.lower() for t in _MEASURED_TOKENS):
                continue
            vb = d[b]
            if isinstance(vb, list) and len(vb) == len(va) and all(isinstance(x, (int, float)) for x in vb):
                return [float(x) for x in vb], [float(x) for x in va], b, a
    return None


def _reported_error(d: Mapping[str, Any]) -> tuple[float | None, str | None]:
    for k in _ERROR_KEYS:
        if k in d:
            v = _num(d[k])
            if v is not None:
                return (v / 100.0 if v > 1.0 else v), k
    return None, None


def _name_of(d: Mapping[str, Any]) -> str:
    for k in _NAME_KEYS:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()[:80]
    return ""


def _rid(source: str, jpath: str, criterion: str) -> str:
    digest = hashlib.sha1(f"{source}|{jpath}|{criterion}".encode("utf-8")).hexdigest()[:12]
    return f"{os.path.basename(source)}:{jpath or '.'}:{digest}"


def harvest_json(path: str, doc: Any, stats: HarvestStats | None = None) -> list[CriterionRecord]:
    """Extract criterion records from one JSON document (ancestor-aware)."""
    stats = stats or HarvestStats()
    rel = path
    out: list[CriterionRecord] = []
    for jpath, d, chain in _walk_with_chain(doc):
        crit_text = None
        for k in _CRITERION_KEYS:
            v = d.get(k)
            if isinstance(v, str) and v.strip():
                crit_text = v.strip()
                break
        has_pair = False
        y = y_ref = None
        y_key = ref_key = None
        rep, rep_key = _reported_error(d)
        ev = d.get("evidence") if isinstance(d.get("evidence"), Mapping) else {}
        array_pair = None
        for scope in (d, ev):
            if not isinstance(scope, Mapping):
                continue
            yy, rr, mk, rk = _pair_in(scope)
            if yy is not None:
                y, y_ref, y_key, ref_key = yy, rr, mk, rk
                has_pair = True
                break
            array_pair = array_pair or _array_pair_in(scope)
        if crit_text is None:
            # pair records still qualify when a tolerance is declared (own or inherited)
            tol, tol_key, tol_kind = _find_tolerance(d)
            if tol is None:
                for parent in reversed(chain):
                    tol, tol_key, tol_kind = _find_tolerance(parent)
                    if tol is not None:
                        break
            if (not has_pair and array_pair is None) or tol is None:
                continue
            crit_text = (f"|Δ|/|y_ref| <= {tol:g}" if tol_kind == "fraction" else f"|Δ| <= {tol:g}")
            if array_pair is not None and not has_pair:
                y = None
                y_ref = None
        val_key = next((k for k in _VALUE_KEYS if _num(d.get(k)) is not None), None)
        raw_value = d.get(val_key) if val_key else None
        value = _num(raw_value) if val_key else None
        error_is_percent = False
        if value is not None and not has_pair:
            if isinstance(raw_value, str) and raw_value.strip().endswith("%"):
                # "value": "0.2790%" is a *relative error*, not a measurement
                rep = value if rep is None else rep
                rep_key = val_key
                error_is_percent = True
            else:
                y = value
        if rep is not None and rep > 1.0:
            rep = rep / 100.0
        old = d.get("passed", d.get("pass", d.get("status")))
        if old is None and isinstance(d.get("status"), str):
            old = d["status"]
        parsed = parse_criterion(crit_text)
        if not parsed.has_tolerance:
            tol, tol_key, _tol_kind = _find_tolerance(d)
            if tol is None:
                for parent in reversed(chain):
                    tol, tol_key, _tol_kind = _find_tolerance(parent)
                    if tol is not None:
                        break
            if tol is not None:
                low = crit_text.lower()
                relative = any(t in low for t in ("相对", "relative", "rel-", "误差", "error", "偏差", "deviation"))
                parsed = ParsedCriterion(raw=crit_text, metric="relative" if relative else "absolute",
                                         tolerance=tol,
                                         tolerance_kind="fraction" if relative else "absolute",
                                         unit=None, pattern="sibling_key:%s" % tol_key)
        if rep is not None and error_is_percent and parsed.tolerance_kind == "absolute":
            rep = None  # a % error cannot be compared against an absolute band
        if not parsed.has_tolerance and not has_pair and rep is None:
            stats.skipped_unparsed += 1
            continue
        rec = CriterionRecord(
            rid=_rid(rel, jpath, crit_text),
            source=rel,
            measurand=_name_of(d) or crit_text[:60],
            criterion=crit_text,
            parsed=parsed,
            y=y,
            y_ref=y_ref,
            reported_error=rep,
            unit=parsed.unit,
            scale=_num(d.get("scale")),
            noise=_num(d.get("noise", d.get("sigma_noise"))),
            family=f"{os.path.dirname(rel)}::{_name_of(d) or crit_text[:40]}",
            old_verdict=str(old) if old is not None else None,
            evidence={
                "json_path": jpath,
                "measured_key": y_key,
                "reference_key": ref_key,
                "error_key": rep_key,
                "value_key": val_key,
            },
        )
        out.append(rec)
        stats.json_records += 1
        if array_pair is not None:
            meas, refs, mk, rk = array_pair
            for i, (yv, rv) in enumerate(zip(meas, refs)):
                out.append(
                    rec.with_fields(
                        rid=_rid(rel, "%s[%d]" % (jpath, i), crit_text),
                        y=yv,
                        y_ref=rv,
                        evidence={**dict(rec.evidence), "element": i, "measured_key": mk, "reference_key": rk},
                    )
                )
                stats.with_numeric_pair += 1
        if has_pair:
            stats.with_numeric_pair += 1
        elif rep is not None:
            stats.with_reported_error_only += 1
    return out


def _all_markdown_rows(text: str) -> list[dict[str, Any]]:
    """Every markdown table row, with or without a criterion-ish header."""
    rows: list[dict[str, Any]] = []
    header: list[str] | None = None
    for i, line in enumerate((text or "").splitlines()):
        m = re.match(r"^\s*\|(.+)\|\s*$", line)
        if not m:
            if line.strip() and not line.strip().startswith("|"):
                header = None
            continue
        cells = [c.strip() for c in m.group(1).split("|")]
        if header is None:
            header = cells
            continue
        if all(set(c) <= set("-: |") for c in cells):
            continue
        rows.append({"header": "|".join(header), "row": "|".join(cells), "cells": cells, "line": str(i)})
    return rows


_NUM_CELL_RX = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?\s*%?$")


def _cell_number(cell: str) -> tuple[float | None, bool]:
    c = cell.strip().replace("**", "").replace("`", "")
    if not c or not _NUM_CELL_RX.match(c):
        return None, False
    pct = c.endswith("%")
    try:
        v = float(c.rstrip("%").strip())
    except ValueError:
        return None, False
    return v, pct


_INLINE_ERROR_RX = re.compile(
    r"(?:相对误差|误差|偏差|deviation|error|一致性|agreement)[^0-9=≈<≤]{0,14}[=≈]?\s*(" + _NUM + r")\s*(%|)", re.IGNORECASE)
_INLINE_THRESHOLD_RX = re.compile(
    r"(?:判据|阈值|标准|要求|criterion|threshold)[^0-9<≤]{0,14}(?:<=|≤|<)\s*(" + _NUM + r")\s*(%|)", re.IGNORECASE)
_INLINE_BARE_THRESHOLD_RX = re.compile(r"(?:<=|≤|<)\s*(" + _NUM + r")\s*(%|)")


def harvest_markdown_inline(path: str, text: str, stats: HarvestStats | None = None) -> list[CriterionRecord]:
    """Extract ``误差 = X（判据 < tau）`` statements from markdown prose.

    Many reports state the error in prose instead of tabulating it.  Errors and thresholds
    are paired inside the same paragraph (blank-line delimited), nearest threshold first, so
    the reported conclusion stays recomputable.  The reference magnitude is absent by
    construction, which the re-judge reports as ``WITHHELD`` rather than as a pass.
    """
    stats = stats or HarvestStats()
    out: list[CriterionRecord] = []
    for p_idx, para in enumerate(re.split(r"\n\s*\n", text or "")):
        if "|" in para and para.count("|") > 6:
            continue
        flat = " ".join(l.strip() for l in para.splitlines() if l.strip())
        if not flat:
            continue
        thresholds = [(m.start(), float(m.group(1)), (m.group(2) or "") == "%", m.group(0))
                      for m in _INLINE_THRESHOLD_RX.finditer(flat)]
        if not thresholds:
            continue
        for m in _INLINE_ERROR_RX.finditer(flat):
            try:
                err_val = float(m.group(1))
            except (TypeError, ValueError):
                continue
            err_pct = (m.group(2) or "") == "%"
            if err_pct and abs(err_val) > 1000.0:
                continue
            pos, thr_val, thr_pct, thr_txt = min(thresholds, key=lambda t: abs(t[0] - m.start()))
            reported = err_val / 100.0 if err_pct else err_val
            if thr_pct:
                parsed = ParsedCriterion(raw=thr_txt, metric="relative", tolerance=thr_val / 100.0,
                                         tolerance_kind="fraction", pattern="inline_percent")
            else:
                parsed = ParsedCriterion(raw=thr_txt, metric="absolute", tolerance=thr_val,
                                         tolerance_kind="absolute", pattern="inline_absolute")
            snippet = flat[max(0, min(m.start(), pos) - 60): max(m.end(), pos) + 60]
            rec = CriterionRecord(
                rid=_rid(path, f"para{p_idx}:{m.start()}", snippet[:80]),
                source=path,
                measurand=snippet[:80],
                criterion=thr_txt[:120],
                parsed=parsed,
                reported_error=reported,
                family=f"{os.path.dirname(path)}::inline::{snippet[:24]}",
                evidence={"paragraph": str(p_idx), "snippet": snippet[:240],
                          "error_token": m.group(0)[:40], "threshold_token": thr_txt},
            )
            out.append(rec)
            stats.markdown_records += 1
            stats.with_reported_error_only += 1
    return out


_CRIT_HEADER_RX = re.compile(r"判据|阈值|标准|要求|验收|criterion|threshold|requirement", re.IGNORECASE)
_VALUE_HEADER_RX = re.compile(r"实测|实际|结果|数值|观测|计算|仿真|actual|value|measured|result|computed|numerical|simulated", re.IGNORECASE)
_REF_HEADER_RX = re.compile(r"解析|参考|理论|文献|基准|实验|analytic|exact|reference|benchmark|theory|literature", re.IGNORECASE)
_ERROR_HEADER_RX = re.compile(r"误差|偏差|error|deviation|残差|residual", re.IGNORECASE)


def _parse_threshold_cell(header: str, cell: str) -> ParsedCriterion:
    """Parse a threshold cell that may carry only the number (``<1%``)."""
    parsed = parse_criterion(cell)
    if parsed.has_tolerance:
        return parsed
    parsed = parse_criterion(f"{header} {cell}")
    if parsed.has_tolerance:
        return parsed
    if _CRIT_HEADER_RX.search(header or ""):
        raw = cell.replace("*", "").replace("`", "").strip()
        if raw.startswith((">", "≥")):
            return UNPARSED  # a lower bound is a requirement, not an error criterion
        if not re.search(r"(<=|≤|<)", raw) and not _ERROR_HEADER_RX.search(header or ""):
            return UNPARSED  # a bare number carries no criterion semantics
        m = _NUM_STR_RX.match(raw.replace("≤", "<"))
        if m:
            value = float(m.group(1))
            is_pct = m.group(2) == "%"
            if is_pct and value > 200.0:
                return UNPARSED
            return ParsedCriterion(raw=cell, metric="relative" if is_pct else "absolute",
                                   tolerance=value / 100.0 if is_pct else value,
                                   tolerance_kind="fraction" if is_pct else "absolute",
                                   pattern="threshold_cell")
    return UNPARSED


def _error_column_present(header: list[str]) -> bool:
    return any(_ERROR_HEADER_RX.search(h) for h in header)


def harvest_markdown(path: str, text: str, stats: HarvestStats | None = None) -> list[CriterionRecord]:
    """Extract criteria from markdown tables.

    Column semantics come from the header, never from position: the criterion column is a
    header cell mentioning 判据/阈值/criterion, the measured column one mentioning
    实测/实际/结果/误差/value.  A row whose measured cell is empty or ``—`` still yields a
    record (``has_outcome = False``): a criterion archived without its outcome is exactly
    the evidence gap this metrology has to expose, and it can never be scored as a pass.
    """
    stats = stats or HarvestStats()
    out: list[CriterionRecord] = []
    section: tuple[int, Any] | None = None
    lines = text.splitlines()
    for row in _all_markdown_rows(text):
        if section is None or section[0] < int(row["line"]):
            line_no = int(row["line"])
            best = section
            for j in range(max(0, line_no - 80), min(line_no, len(lines))):
                line_txt = lines[j]
                m = _INLINE_THRESHOLD_RX.search(line_txt)
                if m:
                    try:
                        thr = float(m.group(1))
                    except (TypeError, ValueError):
                        continue
                    is_pct = (m.group(2) or "") == "%"
                    if is_pct and thr > 200.0:
                        continue
                    best = (j, ParsedCriterion(raw=line_txt.strip()[:160], metric="relative" if is_pct else "absolute",
                                               tolerance=thr / 100.0 if is_pct else thr,
                                               tolerance_kind="fraction" if is_pct else "absolute",
                                               pattern="section_threshold"))
            section = best
        header = [h.strip() for h in row["header"].split("|")]
        cells = row["cells"]
        if len(header) != len(cells):
            header = [""] * len(cells)
        crit_idx = None
        for i, cell in enumerate(cells):
            if _CRIT_HEADER_RX.search(header[i]) and _parse_threshold_cell(header[i], cell).has_tolerance:
                crit_idx = i
                break
        if crit_idx is None:
            for i, cell in enumerate(cells):
                if _parse_threshold_cell(header[i], cell).has_tolerance:
                    crit_idx = i
                    break
        if crit_idx is None:
            if section is None or not _error_column_present(header):
                continue
            parsed = section[1]
            crit_text = parsed.raw
            crit_idx = 0
        else:
            parsed = _parse_threshold_cell(header[crit_idx], cells[crit_idx])
            crit_text = cells[crit_idx]
        # column semantics from the header: an error column holds a relative error, a
        # (reference, measured) column pair holds the two numbers the criterion compares
        err_idx = val_idx = ref_idx = None
        for i, cell in enumerate(cells):
            if i == crit_idx or _cell_number(cell.replace("*", ""))[0] is None:
                continue
            if _ERROR_HEADER_RX.search(header[i]) and err_idx is None:
                err_idx = i
            elif _REF_HEADER_RX.search(header[i]) and ref_idx is None:
                ref_idx = i
            elif _VALUE_HEADER_RX.search(header[i]) and val_idx is None:
                val_idx = i
        value = None
        y_table = y_ref_table = None
        if ref_idx is not None and val_idx is not None and ref_idx != val_idx:
            rv, _ = _cell_number(cells[ref_idx].replace("*", ""))
            mv, _ = _cell_number(cells[val_idx].replace("*", ""))
            if rv is not None and mv is not None:
                y_table, y_ref_table = mv, rv
        if y_table is None and err_idx is not None:
            v, is_pct = _cell_number(cells[err_idx].replace("*", ""))
            if is_pct and parsed.tolerance_kind == "absolute":
                continue  # unit mismatch: a % error cannot be judged by an absolute band
            if v is not None:
                value = v / 100.0 if is_pct else v
                if abs(value - (parsed.tolerance or -1.0)) <= 1e-12 * max(1.0, abs(value)):
                    # the "measured" cell merely repeats the threshold: not an outcome
                    value = None
        elif y_table is None and val_idx is not None:
            v, is_pct = _cell_number(cells[val_idx].replace("*", ""))
            if v is not None:
                value = v / 100.0 if is_pct else v
        if any(ch in "{}" for cell in cells for ch in cell):
            continue  # serialised blobs are not tabulated outcomes
        if len(crit_text) > 160:
            continue
        verdict_cells = [c for c in cells if re.search(r"PASS|FAIL|✓|✗|通过|不通过|部分", c, re.IGNORECASE)]
        measurand = cells[0].strip()[:80]
        rec = CriterionRecord(
            rid=_rid(path, "md%s:%s" % (path, row["line"]), crit_text),
            source=path,
            measurand=measurand,
            criterion=crit_text,
            parsed=parsed,
            y=y_table,
            y_ref=y_ref_table,
            reported_error=value,
            unit=parsed.unit,
            family=None if _is_identifier(measurand) else f"{os.path.dirname(path)}::{measurand[:40]}",
            old_verdict=verdict_cells[0] if verdict_cells else None,
            evidence={"markdown_line": row["line"], "header": row["header"], "cells": cells,
                      "criterion_column": header[crit_idx],
                      "value_column": header[val_idx] if val_idx is not None else None,
                      "reference_column": header[ref_idx] if ref_idx is not None else None,
                      "criterion_source": ("column" if crit_text == cells[crit_idx] else "section"),
                      "error_column": header[err_idx] if err_idx is not None else None},
        )
        out.append(rec)
        stats.markdown_records += 1
        if value is None:
            stats.criterion_only += 1
        else:
            stats.with_reported_error_only += 1
    return out


_BULLET_CRIT_RX = re.compile(r"(判据|验收|阈值|标准|criterion|threshold|requirement)", re.IGNORECASE)
_ERROR_WORD_RX = re.compile(r"误差|偏差|error|deviation|残差|residual|一致性|agreement|相对差|relative|收敛阶|order", re.IGNORECASE)
_BULLET_CMP_RX = re.compile(r"(<=|≤|<|>=|≥|>)\s*[-+]?(?:\d+\.?\d*|\.\d+)")


def harvest_markdown_bullets(path: str, text: str, stats: HarvestStats | None = None) -> list[CriterionRecord]:
    """Extract acceptance bullets (``- (a) ... >= 90%``) as archived criteria.

    Bullets carry a threshold but usually no outcome; they are kept as ``has_outcome =
    False`` records so the corpus reflects what was *stated* historically, while the
    re-judge refuses to score them (``WITHHELD``).
    """
    stats = stats or HarvestStats()
    out: list[CriterionRecord] = []
    for idx, line in enumerate((text or "").splitlines()):
        probe = line.strip()
        if not probe or "|" in probe or len(probe) < 6:
            continue
        if not (_BULLET_CRIT_RX.search(probe) and _BULLET_CMP_RX.search(probe)):
            continue
        # a criterion bullet must talk about an error-like quantity, otherwise the number is
        # just prose ("t_eff = 0.25 m ...")
        if not _ERROR_WORD_RX.search(probe):
            continue
        if len(probe) > 220:
            continue
        parsed = parse_criterion(probe)
        if not parsed.has_tolerance:
            # acceptance prose such as "< 90%" without an error keyword
            if probe.strip().startswith((">", "≥")):
                continue
            m = _INLINE_BARE_THRESHOLD_RX.search(probe)
            if not m:
                continue
            try:
                thr = float(m.group(1))
            except (TypeError, ValueError):
                continue
            is_pct = (m.group(2) or "") == "%"
            parsed = ParsedCriterion(raw=probe[:160], metric="relative" if is_pct else "absolute",
                                     tolerance=thr / 100.0 if is_pct else thr,
                                     tolerance_kind="fraction" if is_pct else "absolute",
                                     pattern="bullet_threshold")
        rep = None
        rm = _INLINE_ERROR_RX.search(probe)
        if rm:
            try:
                rep = float(rm.group(1))
                if (rm.group(2) or "") == "%":
                    rep /= 100.0
            except (TypeError, ValueError):
                rep = None
        rec = CriterionRecord(
            rid=_rid(path, f"bullet{idx}", probe[:80]),
            source=path,
            measurand=probe[:80],
            criterion=probe[:160],
            parsed=parsed,
            reported_error=rep,
            family=f"{os.path.dirname(path)}::bullet::{probe[:24]}",
            evidence={"markdown_line": str(idx), "snippet": probe[:240]},
        )
        out.append(rec)
        stats.markdown_records += 1
        if rep is None:
            stats.criterion_only += 1
        else:
            stats.with_reported_error_only += 1
    return out


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def build_corpus(root: str, *, subdirs: Iterable[str] | None = None,
                 docs_subdirs: Iterable[str] = (".",)) -> tuple[list[CriterionRecord], HarvestStats]:
    """Harvest the whole repository (read-only) into a de-duplicated criterion corpus."""
    stats = HarvestStats()
    subs = tuple(subdirs) if subdirs else ("benchmarks", "artifacts", "docs", "runs", "outputs", "scenarios",
                                           "tests/data", "data")
    records: list[CriterionRecord] = []
    for path, doc in iter_json_documents(root, subs):
        stats.json_files += 1
        records.extend(harvest_json(path, doc, stats))
    for path in _iter_paths(root, docs_subdirs):
        if not path.endswith(".md"):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except (OSError, UnicodeDecodeError):
            continue
        stats.markdown_files += 1
        records.extend(harvest_markdown(path, text, stats))
        records.extend(harvest_markdown_inline(path, text, stats))
        records.extend(harvest_markdown_bullets(path, text, stats))
    for fn in sorted(os.listdir(root)):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(root, fn)
        try:
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue
        stats.json_files += 1
        records.extend(harvest_json(path, doc, stats))
    # De-duplicate at the (source, normalised criterion) level, keeping the instance that
    # actually carries the outcome: a statement repeated in prose and in a table is one
    # criterion, not two.
    best: dict[tuple[str, str], CriterionRecord] = {}
    for rec in records:
        key = (rec.source, _norm(rec.criterion))
        has_outcome = rec.reported_error is not None or rec.has_pair()
        prev = best.get(key)
        if prev is None:
            best[key] = rec
            continue
        prev_outcome = prev.reported_error is not None or prev.has_pair()
        if has_outcome and not prev_outcome:
            best[key] = rec
    seen_numbers: dict[tuple, CriterionRecord] = {}
    for rec in best.values():
        key = (rec.source, rec.y, rec.y_ref, rec.parsed.tolerance, rec.reported_error)
        seen_numbers.setdefault(key, rec)
    unique = list(seen_numbers.values())
    stats.unique_records = len(unique)
    return unique, stats