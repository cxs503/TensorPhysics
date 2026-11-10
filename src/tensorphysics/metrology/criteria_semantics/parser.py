"""Criterion parser: from a report's ⑥ error-verdict table (or a dict) to a structure.

The historical artefact is a Markdown table such as::

    ## 6. 误差判定（逐项 vs 基准，显式回答"是否<3%"）

    | 关键量 | 基准（来源） | 误差 | 是否 <3% | 说明 |
    |---|---|---|---|---|
    | 长波平台恒等 | 平台单源 STAWAVE-1（解析） | 0.0e+00 | **是** | 定义性恒等 |
    | STAWAVE-2 长波域 | 解析平台 + 平台核 | 重叠域最差 5.99% | **否**（>3%，但 <10% 判据带） |
    | STAWAVE-3 绝对精度 | 无离线 EFD 数据集 | — | **不适用** | **WITHHELD** |

Every row is a *criterion*.  The parser lifts the four-tuple

    (被测量 y, 基准 y_ref, 容差 tau, 口径 metric + unit + condition)

out of that text, keeping the raw cells in ``provenance`` so any number in a downstream
report can be traced to the exact cell it came from (zero hand-copying).

**Reconstruction policy** (documented, not hidden — every row records which branch was used
in ``provenance.reconstruction``):

``explicit_pair``
    the error cell contains ``A vs B`` (``0.0175 vs 0.04-0.06``): ``y``, ``y_ref`` are read
    verbatim from the cell — no reconstruction at all.  A reference interval is resolved to
    its **nearest bound** (interval distance), and flagged as T3 (no priority rule stated).
``y_ref_known``
    reference cell holds one number: ``y = y_ref * (1 ± err)`` (``+`` branch, documented
    worst-case reading for a printed |err|).
``reference_interval``
    reference is an interval: ``y_ref := lower bound`` (strictest for positive magnitudes),
    flagged T3.
``error_space``
    reference is *named* but numeric-free (``解析平台``, ``球 Hankel 级数``): the printed
    relative error is the quantity replayed, ``y_ref := 1`` (dimensionless unit reference).
    This is the honest reading of a string criterion: it judges the reported number.
``no_reference``
    reference explicitly absent (``无离线 EFD 数据集``, ``—``, ``不适用``): ``y_ref = None``
    → verdict WITHHELD (a criterion without a benchmark is never a pass).
``absolute_no_tau``
    the reported error is absolute (``7.4e-10``) while only a relative tolerance exists:
    ``tau_available=False`` → WITHHELD.  Comparing an absolute residual with a relative
    tolerance is a metric mismatch, and that mismatch is itself the T1 finding.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .model import Criterion, Measurement, Metric

_NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_NUM_RE = re.compile(_NUM)
_PCT_RE = re.compile(rf"({_NUM})\s*%")
_TAU_HEADER_RE = re.compile(rf"(?:<|≤|<=|不超过|within|小于)\s*({_NUM})\s*%")
_RANGE_RE = re.compile(rf"({_NUM})\s*(?:-|–|~|～|to|至)\s*({_NUM})")
#: only "A vs B" is an explicit measured/reference pair; "A / B" is two measurements of the
#: same quantity (a multi-metric warning, not a reference pair)
_VS_RE = re.compile(rf"({_NUM})\s*(?:vs\.?|VS\.?|对比)\s*({_NUM}(?:\s*(?:-|–|~|～)\s*{_NUM})?)")
_NO_REF_RE = re.compile(r"(?:—|–|无基准|无离线|无解析|无实验|不适用|n/?a|withheld|none|未评估)", re.I)

_QUANTITY_KEYS = ("关键量", "被测量", "判据", "量", "项目", "quantity", "item")
_REF_KEYS = ("基准", "来源", "参考", "对标", "requirement", "要求", "reference", "baseline")
_ERR_KEYS = ("误差", "偏差", "实际", "error", "deviation", "actual")
_VERDICT_KEYS = ("是否", "判定", "结论", "verdict", "pass")


def _to_float(tok: str) -> float | None:
    try:
        return float(tok)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def parse_error_cell(text: str, *, percent_hint: bool = False) -> dict[str, Any]:
    """Extract the numeric error from a table error cell.

    Handles ``0.0172%``, ``≤6.0e-05``, ``0.015% / 0.501%`` (two values → multi),
    ``精确 0`` (exact zero), ``—`` (no value).
    """
    raw = (text or "").strip()
    clean = raw.replace("**", "").replace("*", "")
    pcts = [float(v) for v in _PCT_RE.findall(clean)]
    if pcts:
        vals = [p / 100.0 for p in pcts]
        return {"kind": "relative", "value": max(abs(v) for v in vals), "values": vals,
                "unit": "1", "raw": raw, "n_values": len(vals)}
    m = _NUM_RE.search(clean)
    if m:
        val = _to_float(m.group(0))
        if val is not None:
            if percent_hint:
                return {"kind": "relative", "value": abs(val) / 100.0,
                        "values": [abs(val) / 100.0], "unit": "1", "raw": raw, "n_values": 1}
            kind = "ratio" if abs(val) < 0.1 else "absolute"
            return {"kind": kind, "value": abs(val), "values": [abs(val)],
                    "unit": "1" if kind == "ratio" else "", "raw": raw, "n_values": 1}
    return {"kind": "none", "value": None, "values": [], "unit": "", "raw": raw, "n_values": 0}


def parse_verdict_cell(text: str) -> dict[str, Any]:
    """Normalise a human verdict cell to ``pass|fail|partial|withheld|unknown`` + tolerated taus."""
    raw = (text or "").strip()
    clean = raw.replace("**", "").replace("*", "")
    taus = [float(v) / 100.0 for v in _TAU_HEADER_RE.findall(clean)]
    band = re.search(rf"({_NUM})\s*%\s*判据带", clean)
    if band:
        taus.append(float(band.group(1)) / 100.0)
    if re.search(r"不适用|withheld|无法|未评估|n/?a|—", clean, re.I):
        verdict = "withheld"
    elif re.search(r"部分|⚠", clean):
        verdict = "partial"
    elif re.search(r"否|fail|✗|✘|不满足|未达|超标|不符", clean, re.I):
        verdict = "fail"
    elif re.search(r"是|yes|pass|✔|✓|✅|符合|满足|通过|达标", clean, re.I):
        verdict = "pass"
    else:
        verdict = "unknown"
    return {"verdict": verdict, "taus": sorted(set(taus)), "raw": raw}


def parse_reference_cell(text: str) -> dict[str, Any]:
    """Extract the reference magnitude (single value or interval) from a reference cell."""
    raw = (text or "").strip()
    clean = raw.replace("**", "").replace("*", "").replace("×", "x")
    if not clean.strip():
        return {"value": None, "range": None, "absent": True, "raw": raw}   # empty cell
    if re.fullmatch(r"\s*(?:—|–|-)\s*", clean):
        return {"value": None, "range": None, "absent": True, "raw": raw}
    rng = _RANGE_RE.search(clean)
    if rng:
        lo, hi = _to_float(rng.group(1)), _to_float(rng.group(2))
        if lo is not None and hi is not None and lo != hi:
            return {"value": None, "range": (min(lo, hi), max(lo, hi)), "absent": False,
                    "raw": raw}
    if _NO_REF_RE.search(clean):
        return {"value": None, "range": None, "absent": True, "raw": raw}
    # a cell that only states a tolerance ("≤10%", "<3%") is a requirement, not a reference:
    # treating it as the reference value would divide by the tolerance itself
    if re.fullmatch(rf"\s*(?:<|≤|<=|不超过|≥|>=|>)\s*{_NUM}\s*%?\s*", clean):
        return {"value": None, "range": None, "absent": True, "tolerance_only": True, "raw": raw}
    nums = [float(v) for v in _NUM_RE.findall(clean)]
    if nums:
        return {"value": max(nums, key=abs), "range": None, "absent": False, "raw": raw}
    return {"value": None, "range": None, "absent": False, "raw": raw}


def parse_requirement_cell(text: str) -> dict[str, Any]:
    """Tolerances stated in a per-row requirement cell (``<2%``, ``=1±1%``, ``<10%``)."""
    clean = (text or "").replace("**", "").replace("*", "")
    taus = [float(v) / 100.0 for v in _TAU_HEADER_RE.findall(clean)]
    pm = re.search(rf"±\s*({_NUM})\s*%", clean)
    if pm:
        taus.append(float(pm.group(1)) / 100.0)
    # dimensionless tolerance written without '%' (e.g. "<1e-10", "<1e-6")
    for tok in re.findall(rf"(?:<|≤|<=)\s*({_NUM})", clean):
        val = _to_float(tok)
        if val is not None and 0 < val < 0.1:
            taus.append(val)
    return {"taus": sorted(set(taus)), "raw": text}


def _split_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _is_separator(cells: Iterable[str]) -> bool:
    return all(re.fullmatch(r"\s*:?-{2,}:?\s*", c or "") for c in cells)


def parse_markdown_tables(md: str) -> list[dict[str, Any]]:
    """Yield Markdown tables with their nearest preceding heading.

    Each entry: ``{"heading": str, "header": [...], "rows": [[...], ...]}``.
    """
    tables: list[dict[str, Any]] = []
    heading = ""
    lines = (md or "").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("#"):
            heading = line.lstrip("# ").strip()
            i += 1
            continue
        if line.strip().startswith("|") and i + 1 < len(lines) and \
                lines[i + 1].strip().startswith("|") and _is_separator(_split_row(lines[i + 1])):
            header = _split_row(line)
            rows: list[list[str]] = []
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append(_split_row(lines[j]))
                j += 1
            tables.append({"heading": heading, "header": header, "rows": rows})
            i = j
            continue
        i += 1
    return tables


def _col_index(header: list[str], keys: tuple[str, ...]) -> int | None:
    for idx, name in enumerate(header):
        low = name.lower()
        if any(k.lower() in low for k in keys):
            return idx
    return None


@dataclass
class CriterionRow:
    """A parsed criterion row plus its provenance (raw cells kept verbatim)."""

    criterion: Criterion
    recorded_verdict: str          # pass | fail | partial | withheld | unknown
    recorded_taus: tuple[float, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.criterion.name

    def as_dict(self) -> dict[str, Any]:
        m = self.criterion.measurement
        return {
            "name": self.criterion.name,
            "y": m.y,
            "y_ref": m.y_ref,
            "y_ref_range": list(m.ref_range) if m.ref_range else None,
            "scale_y": m.scale_y,
            "sigma_noise": m.sigma_noise,
            "unit": m.unit,
            "tau": self.criterion.tau,
            "tau_available": self.criterion.tau_available,
            "reported_error_is_zero": self.criterion.reported_error_is_zero,
            "reference_absent": self.criterion.measurement.reference_absent,
            "competing_taus": list(self.criterion.competing_taus),
            "metric": self.criterion.metric.value,
            "condition": self.criterion.condition,
            "recorded_verdict": self.recorded_verdict,
            "recorded_taus": list(self.recorded_taus),
            "provenance": self.provenance,
        }


def criterion_from_dict(d: dict[str, Any]) -> CriterionRow:
    """Build a criterion from a structured record (controls and frozen corpus rows)."""
    rr = d.get("y_ref_range") or d.get("ref_range")
    m = Measurement(
        y=d.get("y") if isinstance(d.get("y"), (int, float)) else None,
        y_ref=d.get("y_ref") if isinstance(d.get("y_ref"), (int, float)) else None,
        scale_y=d.get("scale_y"),
        sigma_noise=d.get("sigma_noise"),
        unit=d.get("unit", "1") or "1",
        ref_range=(tuple(rr) if rr and len(rr) == 2 else None),  # type: ignore[arg-type]
        quantity=d.get("name") or d.get("quantity") or "",
        reference_absent=bool(d.get("reference_absent", False)),
    )
    crit = Criterion(
        measurement=m,
        tau=float(d.get("tau", 0.0)),
        metric=Metric(d.get("metric", "relative")),
        condition=d.get("condition", "") or "",
        name=d.get("name") or d.get("quantity") or "",
        competing_taus=tuple(float(t) for t in (d.get("competing_taus") or ())),
        tau_available=bool(d.get("tau_available", True)),
        reported_error_is_zero=bool(d.get("reported_error_is_zero", False)),
        sign_sensitive=bool(d.get("sign_sensitive", False)),
        source={k: v for k, v in (d.get("provenance") or d.get("source") or {}).items()},
    )
    return CriterionRow(criterion=crit,
                        recorded_verdict=d.get("recorded_verdict", "unknown"),
                        recorded_taus=tuple(float(t) for t in (d.get("recorded_taus") or ())),
                        provenance=dict(d.get("provenance") or d.get("source") or {}))


def parse_row(header: list[str], cells: list[str], *, heading: str = "",
              source: dict[str, Any] | None = None) -> CriterionRow | None:
    """Parse one table row into a :class:`CriterionRow` (``None`` if not a criterion row)."""
    e_i = _col_index(header, _ERR_KEYS)
    v_i = _col_index(header, _VERDICT_KEYS)
    if v_i is None:
        # some tables name the verdict column after the tolerance itself ("<3%")
        for idx, name in enumerate(header):
            if re.search(rf"(?:<|≤|<)\s*{_NUM}\s*%", name or ""):
                v_i = idx
                break
    if e_i is None or e_i >= len(cells) or v_i is None:
        return None                      # not an error-verdict table
    q_i = _col_index(header, _QUANTITY_KEYS)
    r_i = _col_index(header, _REF_KEYS)
    q = cells[q_i] if q_i is not None and q_i < len(cells) else ""
    ref_cell = cells[r_i] if r_i is not None and r_i < len(cells) else ""
    err_cell = cells[e_i]
    ver_cell = cells[v_i] if v_i < len(cells) else ""
    # when 基准 column is absent the requirement column doubles as reference/tolerance
    req_cell = ref_cell if r_i is not None else (cells[v_i - 1] if v_i else "")

    err_header = header[e_i] if e_i < len(header) else ""
    err = parse_error_cell(err_cell, percent_hint=("%" in err_header or "百分" in err_header))
    ver = parse_verdict_cell(ver_cell)
    ref = parse_reference_cell(ref_cell)
    req = parse_requirement_cell(req_cell)

    tau_available = True
    header_taus = [float(v) / 100.0 for v in _TAU_HEADER_RE.findall(" ".join(header))]
    taus = sorted(set(header_taus + req["taus"] + list(ver["taus"])))
    if taus:
        tau = min(taus)                  # registered = strictest (documented merge rule)
        competing = tuple(t for t in taus if t != tau)
    else:
        tau, competing = 0.0, ()
        tau_available = False             # no tolerance stated -> cannot be judged

    ref_absent = bool(ref.get("absent"))
    vs = _VS_RE.search(err_cell.replace("**", ""))
    if not (q or "").strip() and err["kind"] == "none":
        return None                       # empty row (no quantity, no number)
    if vs is None and err["kind"] == "absolute" and err["value"] and not re.search(
            r"[<≤]|[eE][-+]?\d|误差|偏差|error", err_cell.replace("**", "")):
        # a bare count ("33 pass, 0 fail") is not an error measurement: not a criterion row
        return None
    placeholder = False
    y = y_ref = scale_y = None
    unit, metric = "1", Metric.RELATIVE
    recon = ""
    ref_range = ref["range"]

    if vs is not None and err["n_values"] >= 1:
        y = _to_float(vs.group(1))
        tail = vs.group(2)
        rng = _RANGE_RE.fullmatch(tail.strip())
        if rng:
            lo, hi = _to_float(rng.group(1)), _to_float(rng.group(2))
            if lo is not None and hi is not None:
                ref_range = (min(lo, hi), max(lo, hi))
                y_ref = min(ref_range, key=lambda b: abs(b - y)) if y is not None else lo
        else:
            y_ref = _to_float(tail)
        recon = "explicit_pair (y, y_ref read verbatim from the error cell)"
        ref_absent = False      # the reference came from the error cell, not from the column
        metric = Metric.RELATIVE
        scale_y = max(abs(v) for v in (y, y_ref) if isinstance(v, (int, float))) \
            if any(isinstance(v, (int, float)) for v in (y, y_ref)) else None
    elif err["kind"] == "relative":
        metric = Metric.RELATIVE
        if ref["value"] is not None and ref["value"] != 0:
            y_ref = float(ref["value"])
            y = y_ref * (1.0 + err["value"])
            scale_y = abs(y_ref)
            recon = "y_ref_known: y = y_ref*(1+err) [+branch]"
        elif ref["range"] is not None:
            y_ref = ref["range"][0]
            y = y_ref * (1.0 + err["value"])
            scale_y = abs(y_ref)
            recon = "reference_interval: y_ref := lower bound (strictest), y = y_ref*(1+err)"
        elif ref.get("absent"):
            y_ref = None
            y = None
            recon = "no_reference: y_ref := None -> WITHHELD"
        else:
            y_ref = 1.0
            y = 1.0 + err["value"]
            scale_y = 1.0
            recon = "error_space: reported relative error replayed against unit reference"
    elif err["kind"] == "ratio" and ref.get("value") == 0.0:
        # the reference is exactly zero: the residual is measured against *nothing*
        metric = Metric.RELATIVE
        unit = "1"
        y_ref = 0.0
        y = err["value"]
        scale_y = None
        recon = "zero_reference: reference cell holds exactly 0, residual kept as-is"
    elif err["kind"] == "ratio":
        # dimensionless residual replayed against a unit reference: unit-space consistent
        metric = Metric.RELATIVE
        unit = "1"
        y_ref = 1.0
        y = 1.0 + err["value"]
        scale_y = 1.0
        placeholder = True
        recon = ("dimensionless_residual: bare number <0.1 read as a dimensionless "
                 "relative error (y_ref := unit placeholder)")
        if not taus:
            tau_available = False
    elif err["kind"] == "absolute":
        metric = Metric.ABSOLUTE
        unit = ""
        # the reference is not a magnitude in the error's space: keep it absent (None) so no
        # spurious "reference==0" witness is manufactured
        y_ref = 0.0 if (ref["value"] == 0) else None
        y = err["value"]
        if err["value"] == 0.0:
            recon = "absolute_zero: exact match, metric-invariant PASS"
        else:
            # a relative tolerance cannot judge an absolute residual
            tau_available = False
            recon = ("absolute_no_tau: tolerance is relative, residual is absolute -> "
                     "metric mismatch, WITHHELD")
    else:
        if ref.get("absent") or err["kind"] == "none":
            y_ref, y = None, None
            recon = "no_value: nothing numeric to replay -> WITHHELD"
        else:  # pragma: no cover - defensive
            y_ref, y = None, None
            recon = "unparsed_cells"

    m = Measurement(y=y if isinstance(y, (int, float)) else None,
                    y_ref=y_ref if isinstance(y_ref, (int, float)) else None,
                    scale_y=scale_y, unit=unit, ref_range=ref_range,
                    quantity=q or heading, reference_is_placeholder=placeholder,
                    reference_absent=ref_absent)
    row_name = (q or "").strip()
    if not row_name:
        excerpt = re.sub(r"\s+", " ", (err_cell or "").replace("**", ""))[:24]
        row_name = f"{heading}::{excerpt}" if heading else excerpt
    crit = Criterion(measurement=m, tau=tau, metric=metric, condition=heading,
                     name=row_name, competing_taus=competing,
                     tau_available=tau_available,
                     reported_error_is_zero=(err.get("value") == 0.0),
                     source=source or {})
    prov = {
        "heading": heading,
        "quantity_cell": q,
        "reference_cell": ref_cell,
        "requirement_cell": req_cell,
        "error_cell": err_cell,
        "verdict_cell": ver_cell,
        "reconstruction": recon,
        **(source or {}),
    }
    return CriterionRow(criterion=crit, recorded_verdict=ver["verdict"],
                        recorded_taus=tuple(taus), provenance=prov)


def parse_report_criteria(md: str, *, source: dict[str, Any] | None = None,
                          headings: tuple[str, ...] = ("误差", "判据", "验收", "error")) -> list[CriterionRow]:
    """Parse every criterion row out of a diligence report's Markdown tables.

    Only tables that carry *both* an error column and a verdict column are accepted, and
    (when ``headings`` is given) only inside a section whose heading mentions acceptance
    (⑥ 误差判定).  This is what keeps result tables out of the criterion corpus.
    """
    out: list[CriterionRow] = []
    for table in parse_markdown_tables(md):
        head = table["heading"] or ""
        if headings and not any(h.lower() in head.lower() for h in headings):
            continue
        for cells in table["rows"]:
            row = parse_row(table["header"], cells, heading=head, source=source)
            if row is not None:
                out.append(row)
    return out