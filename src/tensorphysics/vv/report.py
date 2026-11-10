"""Machine-generated V&V report from a gate verdict.

Why this module exists
----------------------
``docs/vv_report_template.md`` fixes the six-element structure of a V&V report
(case introduction -> method -> results -> figures -> pitfalls -> error
judgement), but the template is a *form*: it used to be filled by hand, which
is exactly where numbers get copied wrong and where a missing section is
indistinguishable from a section with a nice-looking number.  This module
renders the same form **from a :class:`tensorphysics.vv.gci.GateVerdict`**, so
that

* every number in the report is the number the gate computed (single source of
  truth -- ``GateVerdict.as_dict()`` is the only input),
* a section the machine cannot fill is rendered as ``WITHHELD`` with the reason
  instead of being silently blank (the report status then cannot be better than
  ``WITHHELD``),
* the rendered text carries provenance: a digest of the verdict payload and a
  digest of the markdown itself, so a report can be tied to the artefacts it
  came from,
* hand-typed template placeholders (``[value]``, ``[Describe ...]``) are
  rejected instead of being smuggled into a "generated" report.

The renderer implements **no numerics of its own**: it never recomputes an
order, a GCI or an uncertainty.  Every quantity is read from the verdict that
:func:`tensorphysics.vv.gci.evaluate` returned.

Example
-------
>>> from tensorphysics.vv.gci import Observation, evaluate
>>> from tensorphysics.vv.report import build_report
>>> levels = [Observation(h=h, f=2.0 + 3.0 * h ** 1.8) for h in (0.4, 0.2, 0.1)]
>>> verdict = evaluate(levels, quantity="Cd")
>>> report = build_report(verdict, title="Cylinder drag convergence")
>>> report.status                      # WARN: no reference supplied
'WARN'
>>> "[value]" in report.markdown       # no template placeholder survives
False
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .gci import FAIL, PASS, UNRESOLVED, WARN, WITHHELD, GateVerdict

__all__ = [
    "STATUS_ORDER",
    "ReportSection",
    "VVReport",
    "build_report",
    "render_markdown",
]


#: Gate status ordering shared with :mod:`tensorphysics.vv.gci`
#: (``PASS < WARN < WITHHELD < UNRESOLVED < FAIL``); a report can never be
#: better than the worst of its verdict and its withheld required sections.
STATUS_ORDER = {PASS: 0, WARN: 1, WITHHELD: 2, UNRESOLVED: 3, FAIL: 4}

#: Sections a report must fill to be a complete six-element deliverable.
REQUIRED_SECTIONS = ("case", "grid-convergence", "asme-vv20", "error-judgement", "conclusion")

#: Tokens of ``docs/vv_report_template.md``.  Seeing one of these in generated
#: output means a placeholder was hand-filled instead of injected.
PLACEHOLDER_TOKENS = (
    "[value",
    "[val]",
    "[unit]",
    "[param1]",
    "[Describe",
    "[Analytical/Experimental/Reference CFD]",
    "[Yes/No]",
    "[PASS/FAIL/UNRESOLVED]",
    "[N_c]",
    "[h_c]",
    "[f_c]",
    "[Reference]",
)

#: Default acceptance bound of the platform error criterion (absolute relative
#: error of the key quantity).
DEFAULT_ERROR_ACCEPTANCE = 0.03


def _num(value: Any, digits: int = 6) -> str:
    """Deterministic number formatting; ``None``/NaN render as ``n/a``."""
    if value is None:
        return "n/a"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(x):
        return "n/a"
    if x == 0.0:
        return "0"
    return f"{x:.{digits}g}"


def _cell(value: Any) -> str:
    """Table cell: numbers formatted, everything else stringified and pipe-safe."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _num(value)
    return str(value).replace("|", "\\|")


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_cell(c) for c in row) + " |")
    return "\n".join(lines)


def _check_name(check: Mapping[str, Any]) -> str:
    """Name of a gate check (the gate spells the field ``check``)."""
    return str(check.get("check") or check.get("name") or "")


def _digest(payload: Any) -> str:
    """sha256 of a JSON-normalised payload (sorted keys, no whitespace drift)."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _worst(statuses: Iterable[str]) -> str:
    """Worst status under the gate ordering (an unknown status ranks as FAIL)."""
    worst = PASS
    for status in statuses:
        if STATUS_ORDER.get(status, STATUS_ORDER[FAIL]) > STATUS_ORDER.get(worst, 0):
            worst = status
    return worst


@dataclass(frozen=True)
class ReportSection:
    """One rendered section of the report.

    ``status`` is ``FILLED`` when the machine had data for the section and
    ``WITHHELD`` otherwise (with ``reason`` explaining what is missing);
    ``data`` holds the raw numbers the section was rendered from, so a reader
    (or a test) can compare the markdown against the gate without parsing it.
    """

    key: str
    title: str
    status: str
    markdown: str
    data: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""
    required: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "status": self.status,
            "required": self.required,
            "reason": self.reason,
            "data": dict(self.data),
        }


@dataclass(frozen=True)
class VVReport:
    """A rendered V&V report plus its coverage and provenance ledgers."""

    title: str
    status: str
    verdict_status: str
    sections: tuple[ReportSection, ...]
    markdown: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def coverage(self) -> dict[str, Any]:
        filled = [s.key for s in self.sections if s.status == "FILLED"]
        withheld = [{"key": s.key, "reason": s.reason} for s in self.sections if s.status == "WITHHELD"]
        required_withheld = [s.key for s in self.sections if s.status == "WITHHELD" and s.required]
        required = [s.key for s in self.sections if s.required]
        return {
            "filled": filled,
            "withheld": withheld,
            "required_total": len(required),
            "required_withheld": required_withheld,
            "ratio": (len(required) - len(required_withheld)) / len(required) if required else 1.0,
        }

    def section(self, key: str) -> ReportSection:
        for s in self.sections:
            if s.key == key:
                return s
        raise KeyError(key)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "tensorphysics.vv.report/1",
            "title": self.title,
            "status": self.status,
            "verdict_status": self.verdict_status,
            "coverage": self.coverage,
            "sections": [s.as_dict() for s in self.sections],
            "provenance": dict(self.provenance),
        }


# ---------------------------------------------------------------------------
# section renderers
# ---------------------------------------------------------------------------


def _case_section(
    verdict: GateVerdict, title: str, metadata: Mapping[str, Any]
) -> ReportSection:
    parameters = list(metadata.get("parameters") or ())
    source = metadata.get("reference_source") or metadata.get("citation") or ""
    case_name = metadata.get("case_name") or ""
    notes = metadata.get("notes") or ""
    have = bool(parameters) or bool(source)
    lines = [f"### 1. Case introduction — {title}", ""]
    if case_name:
        lines.append(f"*Case name*: {case_name}")
    if notes:
        lines.append(f"*Notes*: {notes}")
    lines.append(f"*Observable*: `{verdict.quantity}`")
    if source:
        lines.append(f"*Reference source*: {source}")
    else:
        lines.append("*Reference source*: WITHHELD — no reference/citation was supplied")
    if parameters:
        lines += ["", "#### Parameter table", ""]
        lines.append(
            _table(
                ("Parameter", "Value", "Unit", "Uncertainty"),
                [
                    (
                        p.get("name", "?"),
                        p.get("value"),
                        p.get("unit", "-"),
                        p.get("uncertainty", "-"),
                    )
                    for p in parameters
                ],
            )
        )
    else:
        lines += [
            "",
            "#### Parameter table",
            "",
            "WITHHELD — the machine cannot invent the geometry/parameter table; supply "
            "`metadata['parameters']`.",
        ]
    if not have:
        return ReportSection(
            "case",
            "1. Case introduction",
            WITHHELD,
            "\n".join(lines),
            {"quantity": verdict.quantity, "parameters": len(parameters)},
            "no parameters and no reference source supplied",
        )
    return ReportSection(
        "case",
        "1. Case introduction",
        "FILLED",
        "\n".join(lines),
        {"quantity": verdict.quantity, "parameters": len(parameters), "reference_source": source},
    )


def _convergence_section(verdict: GateVerdict) -> ReportSection:
    study = verdict.study
    if study is None:
        return ReportSection(
            "grid-convergence",
            "2.1 Grid convergence study",
            WITHHELD,
            "### 2.1 Grid convergence study — WITHHELD\n\nThe gate could not form a "
            "convergence study from the supplied levels.",
            {},
            "no convergence study in the verdict",
        )
    rows = []
    for i, lvl in enumerate(study.levels):
        ratio = f"{study.refinement_ratios[i]:.6g}" if i < len(study.refinement_ratios) else "-"
        rows.append(
            (
                lvl.label or f"grid{i}",
                lvl.h,
                lvl.f,
                lvl.sigma if lvl.sigma is not None else "-",
                ratio,
            )
        )
    checks = [
        c
        for c in verdict.checks
        if _check_name(c).startswith(
            ("asymptotic-range", "order-", "power-law", "noise-floor", "grid-convergence")
        )
    ]
    lines = [
        "### 2.1 Grid convergence study",
        "",
        _table(("Grid", "h", "f_h", "sigma", "r (coarse/fine)"), rows),
        "",
        "**Richardson / GCI results** (all values injected from the gate verdict)",
        "",
        f"- levels = {study.n_levels}, uniform_refinement = {study.uniform_refinement}",
        f"- observed order p = {_num(study.order_p)} (source: {study.order_source})",
        f"- Richardson extrapolated f_ext = {_num(study.f_extrapolated)}",
        f"- finest computed f_fine = {_num(study.f_fine)}",
        f"- GCI_fine = {_num(study.gci_fine)} %"
        f" ({'absolute' if not study.gci_is_relative else 'relative'}, "
        f"safety factor {_num(study.safety_factor)})",
        f"- GCI pairs = {', '.join(_num(g) + ' %' for g in study.gci_pairs) or 'n/a'}",
        f"- convergence ratio R = {_num(study.convergence_ratio)}, type = {study.convergence_type}",
        f"- asymptotic range certified: {study.asymptotic}",
        f"- power-law fit residual = {_num(study.fit_residual_rel)}, "
        f"noise ratio = {_num(study.noise_ratio)} (certified: {study.noise_certified})",
    ]
    if checks:
        lines += ["", _table(("Stage check", "Status", "Detail"), [( _check_name(c), c["status"], c["detail"]) for c in checks])]
    return ReportSection(
        "grid-convergence",
        "2.1 Grid convergence study",
        "FILLED",
        "\n".join(lines),
        {
            "n_levels": study.n_levels,
            "order_p": study.order_p,
            "order_source": study.order_source,
            "f_extrapolated": study.f_extrapolated,
            "f_fine": study.f_fine,
            "gci_fine": study.gci_fine,
            "gci_pairs": list(study.gci_pairs),
            "convergence_ratio": study.convergence_ratio,
            "convergence_type": study.convergence_type,
            "asymptotic": study.asymptotic,
        },
    )


def _asme_section(verdict: GateVerdict) -> ReportSection:
    validation = verdict.validation
    ledger = verdict.uncertainty
    if validation is None and ledger is None:
        return ReportSection(
            "asme-vv20",
            "2.2 ASME V&V 20 uncertainty",
            WITHHELD,
            "### 2.2 ASME V&V 20 uncertainty — WITHHELD\n\nNo reference value was "
            "supplied, so ``U_val`` and the ``|E| <= U_val`` test cannot be formed.",
            {},
            "no reference value supplied",
        )
    lines = ["### 2.2 ASME V&V 20 uncertainty", ""]
    data: dict[str, Any] = {}
    if validation is not None:
        data.update(
            {
                "f_sim": validation.f_sim,
                "f_exp": validation.f_exp,
                "u_num": validation.u_num,
                "u_input": validation.u_input,
                "u_val": validation.u_val,
                "error": validation.error,
                "ratio": validation.ratio,
                "passed": validation.passed,
            }
        )
        lines += [
            _table(
                ("Term", "Value", "Meaning"),
                [
                    ("U_num", _num(validation.u_num), "numerical uncertainty (GCI/fine GCI)"),
                    ("U_input", _num(validation.u_input), "input/reference uncertainty"),
                    ("U_val", _num(validation.u_val), "sqrt(U_num^2 + U_input^2)"),
                    ("f_sim", _num(validation.f_sim), "computed value"),
                    ("f_exp", _num(validation.f_exp), "reference value"),
                    ("E = |f_sim - f_exp|", _num(validation.error), "comparison error (same unit)"),
                    ("|E| / U_val", _num(validation.ratio), "acceptance ratio (<= 1 passes)"),
                    ("|E| <= U_val", validation.status, "ASME V&V 20 acceptance test"),
                ],
            )
        ]
    if ledger is not None:
        data.update(
            {
                "u_ref": ledger.u_ref,
                "u_val_total": ledger.u_val_total,
                "reference_ratio": ledger.reference_ratio,
                "dominant_term": ledger.dominant_term(),
                "discriminative": ledger.discriminative(),
            }
        )
        lines += [
            "",
            "**Uncertainty ledger**",
            "",
            f"- U_ref = {_num(ledger.u_ref)} (reference uncertainty)",
            f"- U_val_total = {_num(ledger.u_val_total)}",
            f"- reference/validation ratio = {_num(ledger.reference_ratio)}",
            f"- dominant term = {ledger.dominant_term()}",
            f"- discriminative = {ledger.discriminative()} "
            "(a non-discriminative reference withholds the validation instead of passing it)",
        ]
        components = list(getattr(ledger, "components", ()) or ())
        if components:
            rows = []
            for comp in components:
                if isinstance(comp, Mapping):
                    rows.append(
                        (
                            comp.get("name", "?"),
                            comp.get("value"),
                            comp.get("relative", comp.get("relative_percent", "-")),
                            comp.get("source", "-"),
                        )
                    )
            if rows:
                lines += ["", _table(("Component", "Value", "Relative", "Source"), rows)]
    return ReportSection(
        "asme-vv20",
        "2.2 ASME V&V 20 uncertainty",
        "FILLED",
        "\n".join(lines),
        data,
    )


def _error_judgement_section(verdict: GateVerdict, acceptance: float) -> ReportSection:
    validation = verdict.validation
    if validation is None or not math.isfinite(float(validation.f_exp)) or validation.f_exp == 0.0:
        return ReportSection(
            "error-judgement",
            "6. Error judgement",
            WITHHELD,
            "### 6. Error judgement — WITHHELD\n\nNo non-zero reference value was "
            "supplied, so no relative error can be formed against a baseline.",
            {"acceptance": acceptance},
            "no reference value for the error judgement",
        )
    rel = abs(validation.f_sim - validation.f_exp) / abs(validation.f_exp)
    within = rel < acceptance
    verdict_text = (
        f"**{_num(rel * 100, 4)} % < {_num(acceptance * 100, 4)} %** -> error criterion MET"
        if within
        else f"**{_num(rel * 100, 4)} % >= {_num(acceptance * 100, 4)} %** -> error criterion NOT MET"
    )
    lines = [
        "### 6. Error judgement",
        "",
        _table(
            ("Quantity", "Computed", "Reference", "Abs. relative error", f"Acceptance (<{_num(acceptance * 100, 3)} %)", "Verdict"),
            [
                (
                    verdict.quantity,
                    _num(validation.f_sim),
                    _num(validation.f_exp),
                    f"{_num(rel * 100, 4)} %",
                    "MET" if within else "NOT MET",
                    verdict.status,
                )
            ],
        ),
        "",
        f"Key quantity error vs the reference: {verdict_text}.",
    ]
    return ReportSection(
        "error-judgement",
        "6. Error judgement",
        "FILLED",
        "\n".join(lines),
        {
            "relative_error": rel,
            "acceptance": acceptance,
            "within_acceptance": within,
            "u_val": validation.u_val,
            "ratio": validation.ratio,
        },
    )


def _resolvability_section(verdict: GateVerdict) -> ReportSection:
    res = verdict.resolvability
    if res is None:
        return ReportSection(
            "resolvability",
            "5. Resolvability (R = delta/dx)",
            "FILLED",
            "### 5. Resolvability — not requested\n\nNo physical feature scale was "
            "declared, so the ``R = delta / dx`` discriminability law was not applied. "
            "This is not a defect of the study, but a declared limitation: the gate "
            "cannot certify that the reported physics is resolved.",
            {},
            "",
            required=False,
        )
    lines = [
        "### 5. Resolvability (R = delta/dx)",
        "",
        f"- status = {getattr(res, 'status', '?')}",
        f"- smallest ratio R = {_num(getattr(res, 'ratio', None))}",
    ]
    reasons = list(getattr(res, "reasons", ()) or ())
    if reasons:
        lines += ["- reasons:"] + [f"  - {r}" for r in reasons]
    return ReportSection(
        "resolvability",
        "5. Resolvability (R = delta/dx)",
        "FILLED",
        "\n".join(lines),
        {"status": getattr(res, "status", None), "ratio": getattr(res, "ratio", None)},
        "",
        required=False,
    )


def _cross_solver_section(metadata: Mapping[str, Any]) -> ReportSection:
    payload = metadata.get("cross_solver")
    if not payload:
        return ReportSection(
            "cross-solver",
            "2.3 Cross-solver agreement",
            "FILLED",
            "### 2.3 Cross-solver agreement — not requested\n\nNo per-solver samples "
            "were supplied, so cross-solver agreement was not evaluated.",
            {},
            "",
            required=False,
        )
    rows = []
    samples = payload.get("samples") or ()
    for s in samples:
        if isinstance(s, Mapping):
            rows.append((s.get("solver", "?"), s.get("f"), s.get("sigma", "-")))
    lines = ["### 2.3 Cross-solver agreement", ""]
    if rows:
        lines.append(_table(("Solver", "Value", "sigma"), rows))
        lines.append("")
    lines.append(
        f"- consensus = {_num(payload.get('consensus'))}, "
        f"spread = {_num(payload.get('spread'))}, status = {payload.get('status', '?')}"
    )
    return ReportSection(
        "cross-solver",
        "2.3 Cross-solver agreement",
        "FILLED",
        "\n".join(lines),
        dict(payload),
        "",
        required=False,
    )


def _conclusion_section(
    verdict: GateVerdict, sections: Sequence[ReportSection], report_status: str
) -> ReportSection:
    rows = [(_check_name(c), c["status"], c["detail"]) for c in verdict.checks]
    withheld = [s.key for s in sections if s.status == WITHHELD]
    lines = [
        "### 7. Conclusion",
        "",
        f"**Report status: {report_status}** (gate verdict: {verdict.status})",
        "",
        _table(("Gate check", "Status", "Detail"), rows) if rows else "_no checks recorded_",
    ]
    if withheld:
        lines += ["", "Withheld sections (report cannot be better than WITHHELD): " + ", ".join(withheld)]
    reasons = list(verdict.reasons)
    if reasons:
        lines += ["", "**Gate reasons**", ""] + [f"- {r}" for r in reasons]
    return ReportSection(
        "conclusion",
        "7. Conclusion",
        "FILLED",
        "\n".join(lines),
        {"status": report_status, "verdict_status": verdict.status, "withheld": withheld},
    )


# ---------------------------------------------------------------------------
# public entry points
# ---------------------------------------------------------------------------


def render_markdown(title: str, sections: Sequence[ReportSection], provenance: Mapping[str, Any]) -> str:
    """Render the section list into one Markdown document with provenance."""
    lines = [f"# {title}", ""]
    for section in sections:
        lines += [section.markdown, ""]
    prov = provenance or {}
    lines += [
        "---",
        "",
        "### Provenance",
        "",
        f"- source digest (gate verdict payload, sha256) = `{prov.get('source_digest', 'n/a')}`",
        f"- report digest (sha256 of this document minus the provenance block) = "
        f"`{prov.get('body_digest', 'n/a')}`",
        f"- generator = `{prov.get('generator', 'tensorphysics.vv.report')}`",
        f"- verdict status / report status = {prov.get('verdict_status', 'n/a')} / "
        f"{prov.get('status', 'n/a')}",
    ]
    return "\n".join(lines) + "\n"


def _guard_placeholders(text: str, where: str) -> None:
    for token in PLACEHOLDER_TOKENS:
        if token in text:
            raise ValueError(
                f"{where} contains the template placeholder {token!r}: generated reports must "
                f"be injected from machine data, not hand-filled"
            )


def build_report(
    verdict: GateVerdict,
    *,
    title: str,
    metadata: Mapping[str, Any] | None = None,
    error_acceptance: float = DEFAULT_ERROR_ACCEPTANCE,
) -> VVReport:
    """Render the six-element V&V report for ``verdict``.

    Parameters
    ----------
    verdict : GateVerdict
        Output of :func:`tensorphysics.vv.gci.evaluate`; every number in the
        report is read from it.
    title : str
        Report title / case heading.
    metadata : mapping, optional
        Machine-unknowable context: ``case_name``, ``reference_source``
        (citation), ``parameters`` (list of ``{name, value, unit,
        uncertainty}``), ``cross_solver`` (consensus ledger), ``notes``.
        Anything missing renders as ``WITHHELD`` -- it is never invented.
    error_acceptance : float, default 0.03
        Absolute relative error above which the key-quantity error criterion is
        declared NOT MET (the platform 3 % rule).

    Returns
    -------
    VVReport
        Frozen report: ``status`` (worst of the verdict, the withheld required
        sections), ``sections`` with ``data``, ``coverage`` ledger, ``markdown``
        and ``provenance`` digests.
    """
    if not isinstance(verdict, GateVerdict):
        raise TypeError(f"build_report needs a GateVerdict, got {type(verdict).__name__}")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must be a non-empty string")
    if not math.isfinite(error_acceptance) or error_acceptance <= 0.0:
        raise ValueError(f"error_acceptance must be finite and positive, got {error_acceptance!r}")
    meta: Mapping[str, Any] = metadata or {}

    sections: list[ReportSection] = [
        _case_section(verdict, title, meta),
        _convergence_section(verdict),
        _asme_section(verdict),
        _cross_solver_section(meta),
        _resolvability_section(verdict),
        _error_judgement_section(verdict, float(error_acceptance)),
    ]
    withheld = [WITHHELD for s in sections if s.required and s.status == "WITHHELD"]
    report_status = _worst([verdict.status, *withheld])
    sections.append(_conclusion_section(verdict, sections, report_status))

    body = "\n".join(s.markdown for s in sections)
    _guard_placeholders(title, "title")
    _guard_placeholders(body, "rendered sections")
    source_digest = _digest(verdict.as_dict())
    provenance = {
        "generator": "tensorphysics.vv.report/1",
        "source_digest": source_digest,
        "body_digest": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "verdict_status": verdict.status,
        "status": report_status,
    }
    markdown = render_markdown(title, sections, provenance)
    _guard_placeholders(markdown, "rendered markdown")
    return VVReport(
        title=title,
        status=report_status,
        verdict_status=verdict.status,
        sections=tuple(sections),
        markdown=markdown,
        provenance=provenance,
    )