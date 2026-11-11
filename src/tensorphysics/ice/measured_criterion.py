"""Truth-source governance for measured anchors (criterion metrology + evidence gaps).

The measured-anchor asset mode (``tensorphysics.ice.measured_anchor``) answers
*"where is the truth?"*.  This module answers the two questions that decide
whether a criterion built on that truth is **executable** or merely **printed**:

1. **Resolvability** -- given the measurement uncertainty of every point, what is
   the smallest model error the dataset can actually resolve?  A 15 % criterion
   that is finer than the measurement noise is not a criterion; it is a
   coincidence generator.  :func:`criterion_resolvability` turns that into
   numbers (per-point ``u/R``, span of resolvable bias, ``k-sigma`` bands) and
   splits the verdict into its two asymmetric halves:

   * *FAIL is resolvable* when the prediction lies outside the measured interval
     even at ``k = 3 sigma`` -- an over-prediction of a factor ~3.6 is a real
     finding, not noise;
   * *PASS is resolvable* only where ``tolerance * R_meas >= u`` -- otherwise a
     model can never be *confirmed* to 15 % on that point, no matter how good it
     is.

2. **Evidence gaps** -- the library can only be as strong as the sources that
   could be digitised.  Every source that was tried and *not* obtained is
   registered with its exact failure mode (URL, UTC timestamp, HTTP status or
   transport error), so the gap is auditable and the next node does not re-walk
   the same dead ends.  :func:`validate_evidence_gaps` rejects a gap that has no
   attempt record: "we could not get it" is only evidence when it is dated.

Both mechanisms are deliberately *data-driven*: nothing here knows any physical
constant, so the rule "platform code carries no domain semantics" is respected --
the domain content lives in ``data/measured/ice_resistance/``.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

__all__ = [
    "GOVERNANCE_VERSION",
    "FACILITY_KINDS",
    "BREAKING_MODES",
    "GAP_STATUSES",
    "validate_governance",
    "governance_errors_for_registry",
    "criterion_resolvability",
    "validate_evidence_gaps",
    "load_evidence_gaps",
    "evidence_gap_summary",
    "load_adjudication_rows",
]

GOVERNANCE_VERSION = "measured_dataset/v1+governance-2026-11-10"

FACILITY_KINDS = ("ice_basin", "full_scale_trial", "numerical_benchmark", "unknown")
BREAKING_MODES = ("downward_bending", "upward_bending", "mixed", "unknown")
GAP_STATUSES = ("OPEN", "PARTIAL", "CLOSED")


# ---------------------------------------------------------------------------
# 1. governance block (the schema fields the contract names explicitly)
# ---------------------------------------------------------------------------


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_governance(doc: Mapping[str, Any]) -> List[str]:
    """Validate the governance block of one series document.

    Contract fields covered here (the ones the schema of m9425 left implicit):
    facility/bench, scale, uncertainty declaration and digitisation method --
    plus the applicability envelope that decides whether a series may enter a
    15 % verdict at all.
    """

    errors: List[str] = []
    sid = doc.get("series_id", "<unknown>")

    facility = doc.get("facility")
    if not isinstance(facility, Mapping):
        errors.append(f"{sid}.facility: missing (bench/facility must be declared)")
    else:
        kind = facility.get("kind")
        if kind not in FACILITY_KINDS:
            errors.append(f"{sid}.facility.kind: {kind!r} not in {FACILITY_KINDS}")
        if not _nonempty(facility.get("name")):
            gaps = facility.get("known_gaps")
            if not (isinstance(gaps, list) and gaps and all(_nonempty(g) for g in gaps)):
                errors.append(
                    f"{sid}.facility.name: empty and no known_gaps[] explaining why "
                    "(an unknown facility is allowed only when it is declared unknown)"
                )
        scale = facility.get("model_scale")
        if scale is not None and not (isinstance(scale, (int, float)) and math.isfinite(scale) and scale > 0):
            errors.append(f"{sid}.facility.model_scale: must be null or a positive finite number")
        if not _nonempty(facility.get("reported_frame")):
            errors.append(f"{sid}.facility.reported_frame: must state the frame the digits are reported in")

    dig = doc.get("digitisation")
    if not isinstance(dig, Mapping):
        errors.append(f"{sid}.digitisation: missing (digitisation method must be declared)")
    else:
        if not _nonempty(dig.get("method")):
            errors.append(f"{sid}.digitisation.method: must be a non-empty description")
        passes = dig.get("passes")
        if not (isinstance(passes, list) and len(passes) >= 2 and all(_nonempty(p) for p in passes)):
            errors.append(f"{sid}.digitisation.passes: need >= 2 independent named passes")
        elif len(set(passes)) != len(passes):
            errors.append(f"{sid}.digitisation.passes: pass names must be distinct")
        paths_list = dig.get("independent_paths")
        if paths_list is not None and not (isinstance(paths_list, list) and all(_nonempty(p) for p in paths_list)):
            errors.append(f"{sid}.digitisation.independent_paths: must be a list of non-empty strings")

    unc = doc.get("uncertainty")
    if not isinstance(unc, Mapping):
        errors.append(f"{sid}.uncertainty: missing (uncertainty must be declared, even when unreported)")
    else:
        if not isinstance(unc.get("reported"), bool):
            errors.append(f"{sid}.uncertainty.reported: must be an explicit boolean")
        if not _nonempty(unc.get("definition")):
            errors.append(f"{sid}.uncertainty.definition: must say what the number means")
        if unc.get("reported") is True and not _nonempty(unc.get("value_kN")):
            errors.append(f"{sid}.uncertainty.value_kN: reported=True needs a pointer to the values")
        if unc.get("reported") is False:
            missing = [p.get("point_id") for p in doc.get("points", []) if p.get("R_unc_kN") is not None]
            if missing:
                errors.append(
                    f"{sid}.uncertainty: reported=False but points {missing} carry R_unc_kN "
                    "(declaration and data disagree)"
                )

    app = doc.get("applicability")
    if not isinstance(app, Mapping):
        errors.append(f"{sid}.applicability: missing (the validity envelope must be declared)")
    else:
        mode = app.get("ice_breaking_mode")
        if mode not in BREAKING_MODES:
            errors.append(f"{sid}.applicability.ice_breaking_mode: {mode!r} not in {BREAKING_MODES}")
        for key in ("lindqvist_family_applicable", "in_15pct_verdict_set"):
            if not isinstance(app.get(key), bool):
                errors.append(f"{sid}.applicability.{key}: must be an explicit boolean")
        if not _nonempty(app.get("role")):
            errors.append(f"{sid}.applicability.role: must say what the series is allowed to prove")
        if not isinstance(app.get("limitations", []), list):
            errors.append(f"{sid}.applicability.limitations: must be a list")
    return errors


def governance_errors_for_registry(
    series_docs: Iterable[Mapping[str, Any]],
) -> Dict[str, List[str]]:
    """Apply :func:`validate_governance` to every series; return ``{series_id: errors}``."""

    return {str(doc.get("series_id", "<unknown>")): validate_governance(doc) for doc in series_docs}


# ---------------------------------------------------------------------------
# 2. criterion metrology
# ---------------------------------------------------------------------------


def criterion_resolvability(
    rows: Sequence[Mapping[str, Any]],
    *,
    tolerance: float = 0.15,
    kappas: Sequence[float] = (1.0, 2.0, 3.0),
) -> Dict[str, Any]:
    """How much model error can this measured series actually resolve?

    ``rows`` come straight out of :func:`tensorphysics.ice.measured_anchor.adjudicate_resistance`
    (``R_pred_kN`` / ``R_meas_kN`` / ``R_unc_kN``).  The output separates the two
    halves of the verdict:

    * ``fail_resolvable``  -- fraction of points whose prediction is outside the
      measured interval ``[R - k u, R + k u]`` at the largest ``k``; when this is
      1.0 the FAIL verdict cannot be explained by measurement noise;
    * ``pass_resolvable``  -- fraction of points where ``tolerance * R >= k u``,
      i.e. where a 15 % agreement would be distinguishable from the noise at all.
    """

    per_point: List[Dict[str, Any]] = []
    for r in rows:
        meas = r.get("R_meas_kN")
        pred = r.get("R_pred_kN")
        unc = r.get("R_unc_kN")
        if meas is None or pred is None:
            continue
        u = float(unc) if unc is not None else None
        rec: Dict[str, Any] = {
            "point_id": r.get("point_id"),
            "h_m": r.get("h_m"),
            "V_ms": r.get("V_ms"),
            "R_meas_kN": float(meas),
            "R_pred_kN": float(pred),
            "R_unc_kN": u,
            "rel_error": abs(float(pred) - float(meas)) / abs(float(meas)),
            "u_over_R": (u / abs(float(meas))) if u is not None else None,
        }
        if u is not None and u > 0:
            ratio = abs(float(pred) - float(meas)) / u
            rec["pred_meas_gap_in_sigma"] = ratio
            for k in kappas:
                rec[f"inside_band_{int(k)}sigma"] = bool(abs(float(pred) - float(meas)) <= k * u)
            rec["tolerance_above_noise"] = bool(tolerance * abs(float(meas)) >= u)
        per_point.append(rec)

    with_u = [p for p in per_point if p.get("R_unc_kN")]
    kmax = max(kappas)
    out: Dict[str, Any] = {
        "tolerance": tolerance,
        "n_points": len(per_point),
        "n_points_with_uncertainty": len(with_u),
        "kappas": list(kappas),
        "rows": per_point,
    }
    if not with_u:
        out.update(
            {
                "median_u_over_R": None,
                "max_u_over_R": None,
                "fail_resolvable": None,
                "pass_resolvable": None,
                "n_fail_resolvable_at_kmax": None,
                "n_pass_resolvable": None,
                "min_resolvable_bias": None,
                "criterion_state": "WITHHELD",
                "note": "no measured point declares an uncertainty: resolvability is WITHHELD",
            }
        )
        return out

    u_over_R = [p["u_over_R"] for p in with_u]
    n_fail = sum(1 for p in with_u if not p[f"inside_band_{int(kmax)}sigma"])
    n_pass = sum(1 for p in with_u if p["tolerance_above_noise"])
    out.update(
        {
            "median_u_over_R": statistics.median(u_over_R),
            "max_u_over_R": max(u_over_R),
            "n_points_uncertainty_above_tolerance": sum(1 for v in u_over_R if v > tolerance),
            "n_fail_resolvable_at_kmax": n_fail,
            "fail_resolvable": n_fail / len(with_u),
            "n_pass_resolvable": n_pass,
            "pass_resolvable": n_pass / len(with_u),
            "median_abs_gap_in_sigma": statistics.median([p["pred_meas_gap_in_sigma"] for p in with_u]),
            "min_resolvable_bias": statistics.median(
                [p["R_unc_kN"] / p["R_meas_kN"] for p in with_u]
            ),
            "criterion_state": _criterion_state(n_fail, n_pass, len(with_u)),
            "note": (
                "fail_resolvable counts points where the prediction is outside the measured "
                f"+-{kmax:g} sigma interval; pass_resolvable counts points where tolerance*R >= u, "
                "i.e. where a 15 % agreement could be confirmed at all"
            ),
        }
    )
    return out


def _criterion_state(n_fail: int, n_pass: int, n: int) -> str:
    if n == 0:
        return "WITHHELD"
    if n_fail == n:
        return "FAIL_RESOLVABLE_PASS_WITHHELD" if n_pass == 0 else "FAIL_RESOLVABLE"
    if n_pass == 0:
        return "PASS_UNRESOLVABLE"
    return "BOTH_DIRECTIONS_RESOLVABLE"


# ---------------------------------------------------------------------------
# 3. evidence-gap registry
# ---------------------------------------------------------------------------


def validate_evidence_gaps(doc: Mapping[str, Any], *, root: Optional[Path] = None) -> List[str]:
    """Structural + evidentiary validation of the gap registry.

    A gap must carry at least one dated retrieval attempt.  When ``root`` is
    given, the referenced attempt log must exist and its recorded digest must
    still match, so the log cannot be rewritten after the fact.
    """

    errors: List[str] = []
    if doc.get("kind") != "measured_evidence_gap_registry":
        errors.append(f"gaps.kind: expected 'measured_evidence_gap_registry', got {doc.get('kind')!r}")
    gaps = doc.get("gaps")
    if not isinstance(gaps, list) or not gaps:
        return errors + ["gaps.gaps: need >= 1 registered gap"]

    seen: set[str] = set()
    for i, gap in enumerate(gaps):
        where = f"gaps[{i}]"
        if not isinstance(gap, Mapping):
            errors.append(f"{where}: must be a mapping")
            continue
        gid = gap.get("gap_id")
        if not _nonempty(gid):
            errors.append(f"{where}.gap_id: missing")
        elif not str(gid).startswith("GAP-"):
            errors.append(f"{where}.gap_id: {gid!r} must start with 'GAP-'")
        elif gid in seen:
            errors.append(f"{where}.gap_id: duplicate {gid!r}")
        else:
            seen.add(str(gid))
        for key in ("target", "why_needed", "unblock", "status"):
            if not _nonempty(gap.get(key)):
                errors.append(f"{where}.{key}: must be a non-empty string")
        if _nonempty(gap.get("status")) and gap["status"] not in GAP_STATUSES:
            errors.append(f"{where}.status: {gap['status']!r} not in {GAP_STATUSES}")
        attempts = gap.get("attempts")
        if not (isinstance(attempts, list) and attempts):
            errors.append(f"{where}.attempts: need >= 1 dated retrieval attempt (undated gaps are opinions)")
        else:
            for j, a in enumerate(attempts):
                aw = f"{where}.attempts[{j}]"
                if not isinstance(a, Mapping):
                    errors.append(f"{aw}: must be a mapping")
                    continue
                if not _nonempty(a.get("utc")):
                    errors.append(f"{aw}.utc: missing timestamp")
                if not _nonempty(a.get("url")):
                    errors.append(f"{aw}.url: missing target URL")
                if a.get("http_status") is None and not _nonempty(a.get("error")):
                    errors.append(f"{aw}: a failed attempt must record http_status or a transport error")

    log = doc.get("attempt_log")
    if isinstance(log, Mapping):
        rel = log.get("path")
        if not _nonempty(rel):
            errors.append("gaps.attempt_log.path: missing")
        elif root is not None:
            from .measured_anchor import sha256_file  # local import: avoid cycle at module import

            p = Path(root) / str(rel)
            if not p.is_file():
                errors.append(f"gaps.attempt_log.path: not found: {p}")
            elif _nonempty(log.get("sha256")) and sha256_file(p) != str(log["sha256"]).lower():
                errors.append(f"gaps.attempt_log.sha256: digest mismatch for {p}")
    return errors


def load_evidence_gaps(root: Path | str, *, verify: bool = True) -> Dict[str, Any]:
    """Load and (optionally) verify ``evidence_gaps.json`` below ``root``."""

    root = Path(root)
    path = root / "evidence_gaps.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_evidence_gaps(doc, root=root if verify else None)
    if errors:
        from .measured_anchor import MeasuredAnchorError

        raise MeasuredAnchorError("; ".join(errors))
    return doc


def evidence_gap_summary(doc: Mapping[str, Any]) -> Dict[str, Any]:
    """Compact, report-ready view of the gap registry."""

    gaps = doc.get("gaps", [])
    by_status: Dict[str, int] = {}
    for g in gaps:
        by_status[g.get("status", "UNKNOWN")] = by_status.get(g.get("status", "UNKNOWN"), 0) + 1
    return {
        "n_gaps": len(gaps),
        "by_status": by_status,
        "n_attempts": sum(len(g.get("attempts", [])) for g in gaps),
        "gap_ids": [g.get("gap_id") for g in gaps],
        "open_gaps": [g.get("gap_id") for g in gaps if g.get("status") == "OPEN"],
        "closed_gaps": [g.get("gap_id") for g in gaps if g.get("status") == "CLOSED"],
    }


def load_adjudication_rows(artifact: Mapping[str, Any], config: str) -> List[Dict[str, Any]]:
    """Pull one configuration's rows out of an adjudication artifact."""

    for adj in artifact.get("adjudications", []):
        if adj.get("summary", {}).get("config") == config:
            return list(adj.get("rows", []))
    raise KeyError(f"config {config!r} not present in artifact")