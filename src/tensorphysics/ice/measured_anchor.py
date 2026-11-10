"""Measured-anchor asset mode for the ice-resistance chain (mission 9425).

Why this module exists
----------------------
:mod:`tensorphysics.ice.resistance` implements a fully closed-form ice-resistance
family, but until now the *absolute* accuracy of that family was graded
``WITHHELD``: the repository held no digitised measurement of ice resistance, so
the platform's 15 % acceptance criterion could not even be executed.  This module
turns that state into a *judgeable* one by giving the chain a single, auditable
source of truth:

* a **measured-dataset asset mode** (versioned JSON series + SHA-256 provenance),
* a **digitisation ingestion path** with a *double-entry* cross-check,
* an **adjudication engine** that compares a model against every measured point
  and emits a per-point error table plus a 15 % gate verdict (``PASS`` /
  ``FAIL`` -- never silently "improved"),
* **physical-plausibility gates** (``R(h)`` increasing at fixed speed, ``R(V)``
  increasing at fixed thickness, finite ``V -> 0`` intercept with no spurious
  speed term).

Evidence vocabulary
-------------------
``MEASURED-PRIMARY``      value transcribed from a peer-reviewed table, source
                          artifact archived with its digest.
``MEASURED-SECONDARY``    value transcribed from a downstream document that
                          reports the primary measurement (e.g. a thesis
                          quoting an ice trial); used only for screening.
``MEASURED-RANGE``        the source reports an interval/range rather than a
                          point value (e.g. a power band).

Design rules (enforced by :func:`validate_series`)
-------------------------------------------------
1. every series carries at least one archived ``source`` artifact *and* the
   SHA-256 of its bytes -- a provenance without a digest is rejected;
2. every point names the table/figure it comes from (``provenance_ref``);
3. no value is ever imputed: missing statistics stay missing (``None``);
4. the schema version is explicit and the loader refuses unknown versions.

This module owns *mechanism only*; the series themselves live as data under
``data/measured/ice_resistance/`` and are produced by
``benchmarks/ice_resistance/measured/build_assets.py`` from archived
transcriptions.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCHEMA_VERSION = "1.0"

#: Platform acceptance threshold for model-vs-measurement agreement (15 %).
GATE_TOLERANCE = 0.15

#: Published/expected point count used by the campaign node.
REQUIRED_SERIES = 2

EVIDENCE_GRADES = (
    "MEASURED-PRIMARY",
    "MEASURED-SECONDARY",
    "MEASURED-RANGE",
)

MEASURED_QUANTITIES = {
    # horizontally towed / propelled ship resistance in level ice
    "resistance",
    # full-scale ship speed recorded at a measured ice thickness and a measured
    # engine-power band: adjudicated as a power-interval consistency check
    "speed_power",
}

_HEX = set("0123456789abcdef")


class MeasuredAnchorError(ValueError):
    """Raised when a measured-dataset asset violates the schema or is corrupt."""


# ---------------------------------------------------------------------------
# digests
# ---------------------------------------------------------------------------


def sha256_bytes(payload: bytes) -> str:
    """Return the lower-case hex SHA-256 of ``payload``."""

    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path | str) -> str:
    """Return the lower-case hex SHA-256 of a file's bytes."""

    return sha256_bytes(Path(path).read_bytes())


def is_sha256_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in _HEX for c in value.lower())
    )


# ---------------------------------------------------------------------------
# schema validation
# ---------------------------------------------------------------------------

REQUIRED_SERIES_KEYS = (
    "schema_version",
    "series_id",
    "title",
    "ice_state",
    "measured_quantity",
    "ship",
    "provenance",
    "points",
)

REQUIRED_SHIP_KEYS = ("name", "L_pp_m", "B_m", "T_m", "bow_angle_deg", "flare_deg")

REQUIRED_PROVENANCE_KEYS = (
    "citation",
    "doi",
    "url",
    "grade",
    "sources",
    "tables",
    "retrieval_note",
)

REQUIRED_SOURCE_KEYS = ("artifact", "sha256", "bytes", "kind")


def _require(container: Mapping[str, Any], keys: Iterable[str], where: str, errors: List[str]) -> None:
    for key in keys:
        if key not in container:
            errors.append(f"{where}: missing required key '{key}'")


def _finite_positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0.0


def validate_series(doc: Mapping[str, Any], *, root: Optional[Path] = None) -> List[str]:
    """Structurally validate one measured series.

    Returns a list of human-readable error strings -- an empty list means the
    document satisfies the schema.  When ``root`` is given, every archived
    source artifact is re-hashed and compared with its declared digest, i.e. the
    provenance is *verified*, not merely present.
    """

    errors: List[str] = []
    if not isinstance(doc, Mapping):
        return ["series document is not a mapping"]

    _require(doc, REQUIRED_SERIES_KEYS, "series", errors)
    if errors:
        return errors

    if doc["schema_version"] != SCHEMA_VERSION:
        errors.append(f"series: unsupported schema_version {doc['schema_version']!r} (want {SCHEMA_VERSION})")

    if not isinstance(doc["series_id"], str) or not doc["series_id"].strip():
        errors.append("series: series_id must be a non-empty string")

    if doc["ice_state"] not in ("level", "brash", "ridge"):
        errors.append(f"series: unknown ice_state {doc['ice_state']!r}")

    if doc["measured_quantity"] not in MEASURED_QUANTITIES:
        errors.append(f"series: unknown measured_quantity {doc['measured_quantity']!r}")

    ship = doc["ship"]
    if not isinstance(ship, Mapping):
        errors.append("series.ship: must be a mapping")
    else:
        _require(ship, REQUIRED_SHIP_KEYS, "series.ship", errors)
        for key in ("L_pp_m", "B_m", "T_m"):
            if key in ship and not _finite_positive(ship[key]):
                errors.append(f"series.ship.{key}: must be a finite positive number")
        for key in ("bow_angle_deg", "flare_deg"):
            if key in ship and not (isinstance(ship[key], (int, float)) and math.isfinite(ship[key])):
                errors.append(f"series.ship.{key}: must be a finite number")

    prov = doc["provenance"]
    if not isinstance(prov, Mapping):
        errors.append("series.provenance: must be a mapping")
    else:
        _require(prov, REQUIRED_PROVENANCE_KEYS, "series.provenance", errors)
        grade = prov.get("grade")
        if grade not in EVIDENCE_GRADES:
            errors.append(f"series.provenance.grade: {grade!r} not in {EVIDENCE_GRADES}")
        sources = prov.get("sources")
        if not isinstance(sources, list) or not sources:
            errors.append("series.provenance.sources: need >= 1 archived source artifact")
        else:
            for i, src in enumerate(sources):
                where = f"series.provenance.sources[{i}]"
                if not isinstance(src, Mapping):
                    errors.append(f"{where}: must be a mapping")
                    continue
                _require(src, REQUIRED_SOURCE_KEYS, where, errors)
                if "sha256" in src and not is_sha256_hex(src["sha256"]):
                    errors.append(f"{where}.sha256: not a 64-char hex digest")
                if not isinstance(src.get("bytes", 0), int) or src.get("bytes", 0) <= 0:
                    errors.append(f"{where}.bytes: must be a positive integer")
                if root is not None and isinstance(src.get("artifact"), str):
                    path = Path(root) / src["artifact"]
                    if not path.is_file():
                        errors.append(f"{where}.artifact: file not found: {path}")
                    else:
                        actual = sha256_file(path)
                        if actual != str(src.get("sha256", "")).lower():
                            errors.append(
                                f"{where}.sha256: digest mismatch for {path} "
                                f"(declared {src.get('sha256')}, actual {actual})"
                            )

    points = doc["points"]
    quantity = doc.get("measured_quantity")
    if not isinstance(points, list) or not points:
        errors.append("series.points: need >= 1 measured point")
        return errors

    seen: set[str] = set()
    for i, pt in enumerate(points):
        where = f"series.points[{i}]"
        if not isinstance(pt, Mapping):
            errors.append(f"{where}: must be a mapping")
            continue
        for key in ("point_id", "h_m", "V_ms", "provenance_ref"):
            if key not in pt:
                errors.append(f"{where}: missing required key '{key}'")
        pid = pt.get("point_id")
        if isinstance(pid, str):
            if pid in seen:
                errors.append(f"{where}.point_id: duplicate {pid!r}")
            seen.add(pid)
        if "h_m" in pt and not _finite_positive(pt["h_m"]):
            errors.append(f"{where}.h_m: must be finite and > 0")
        if "V_ms" in pt and not (isinstance(pt["V_ms"], (int, float)) and math.isfinite(pt["V_ms"]) and pt["V_ms"] >= 0.0):
            errors.append(f"{where}.V_ms: must be finite and >= 0")
        if quantity == "resistance" and not _finite_positive(pt.get("R_meas_kN")):
            errors.append(f"{where}.R_meas_kN: resistance series needs a finite positive measurement")
        if quantity == "speed_power":
            band = pt.get("P_band_kW")
            if not (isinstance(band, list) and len(band) == 2 and all(_finite_positive(b) for b in band)):
                errors.append(f"{where}.P_band_kW: speed_power series needs a [P_min, P_max] band in kW")
            elif band[0] > band[1]:
                errors.append(f"{where}.P_band_kW: lower bound exceeds upper bound")
    return errors


# ---------------------------------------------------------------------------
# assets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MeasuredPoint:
    point_id: str
    h_m: float
    V_ms: float
    provenance_ref: str
    R_meas_kN: Optional[float] = None
    R_unc_kN: Optional[float] = None
    R_max_kN: Optional[float] = None
    R_min_kN: Optional[float] = None
    R_alt_kN: Optional[float] = None
    P_band_kW: Optional[Tuple[float, float]] = None
    sigma_f_kPa: Optional[float] = None
    sigma_c_kPa: Optional[float] = None
    E_MPa: Optional[float] = None
    case_group: Optional[str] = None
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "point_id": self.point_id,
            "h_m": self.h_m,
            "V_ms": self.V_ms,
            "provenance_ref": self.provenance_ref,
            "R_meas_kN": self.R_meas_kN,
            "R_unc_kN": self.R_unc_kN,
            "R_max_kN": self.R_max_kN,
            "R_min_kN": self.R_min_kN,
            "R_alt_kN": self.R_alt_kN,
            "P_band_kW": list(self.P_band_kW) if self.P_band_kW else None,
            "sigma_f_kPa": self.sigma_f_kPa,
            "sigma_c_kPa": self.sigma_c_kPa,
            "E_MPa": self.E_MPa,
            "case_group": self.case_group,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class MeasuredSeries:
    """One versioned measured dataset (the asset)."""

    series_id: str
    title: str
    ice_state: str
    measured_quantity: str
    ship: Dict[str, Any]
    provenance: Dict[str, Any]
    points: Tuple[MeasuredPoint, ...]
    schema_version: str = SCHEMA_VERSION
    path: Optional[str] = None

    # -- construction ------------------------------------------------------
    @classmethod
    def from_dict(cls, doc: Mapping[str, Any], *, path: Optional[str] = None, root: Optional[Path] = None) -> "MeasuredSeries":
        errors = validate_series(doc, root=root)
        if errors:
            raise MeasuredAnchorError("; ".join(errors))
        pts = []
        for raw in doc["points"]:
            band = raw.get("P_band_kW")
            pts.append(
                MeasuredPoint(
                    point_id=str(raw["point_id"]),
                    h_m=float(raw["h_m"]),
                    V_ms=float(raw["V_ms"]),
                    provenance_ref=str(raw["provenance_ref"]),
                    R_meas_kN=_opt_float(raw.get("R_meas_kN")),
                    R_unc_kN=_opt_float(raw.get("R_unc_kN")),
                    R_max_kN=_opt_float(raw.get("R_max_kN")),
                    R_min_kN=_opt_float(raw.get("R_min_kN")),
                    R_alt_kN=_opt_float(raw.get("R_alt_kN")),
                    P_band_kW=(float(band[0]), float(band[1])) if band else None,
                    sigma_f_kPa=_opt_float(raw.get("sigma_f_kPa")),
                    sigma_c_kPa=_opt_float(raw.get("sigma_c_kPa")),
                    E_MPa=_opt_float(raw.get("E_MPa")),
                    case_group=raw.get("case_group"),
                    notes=str(raw.get("notes", "")),
                )
            )
        return cls(
            series_id=doc["series_id"],
            title=doc["title"],
            ice_state=doc["ice_state"],
            measured_quantity=doc["measured_quantity"],
            ship=dict(doc["ship"]),
            provenance=dict(doc["provenance"]),
            points=tuple(pts),
            schema_version=doc["schema_version"],
            path=path,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "series_id": self.series_id,
            "title": self.title,
            "ice_state": self.ice_state,
            "measured_quantity": self.measured_quantity,
            "ship": self.ship,
            "provenance": self.provenance,
            "points": [p.as_dict() for p in self.points],
        }

    # -- helpers -----------------------------------------------------------
    def point(self, point_id: str) -> MeasuredPoint:
        for p in self.points:
            if p.point_id == point_id:
                return p
        raise KeyError(point_id)

    @property
    def grade(self) -> str:
        return str(self.provenance.get("grade"))

    @property
    def digests(self) -> List[str]:
        return [str(s["sha256"]).lower() for s in self.provenance["sources"]]

    def asset_record(self) -> Dict[str, Any]:
        """Compact identity of the asset for ledger / registry use."""

        return {
            "series_id": self.series_id,
            "schema_version": self.schema_version,
            "grade": self.grade,
            "measured_quantity": self.measured_quantity,
            "n_points": len(self.points),
            "sha256": self.digests,
            "citation": self.provenance.get("citation"),
            "doi": self.provenance.get("doi"),
            "path": self.path,
        }


def _opt_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    out = float(value)
    return out


# ---------------------------------------------------------------------------
# registry / loader
# ---------------------------------------------------------------------------


def load_series(path: Path | str, *, verify: bool = True) -> MeasuredSeries:
    """Load and (by default) digest-verify one series asset."""

    p = Path(path)
    doc = json.loads(p.read_text(encoding="utf-8"))
    return MeasuredSeries.from_dict(doc, path=str(p), root=p.parent if verify else None)


def load_registry(root: Path | str, *, verify: bool = True) -> Dict[str, Any]:
    """Load ``registry.json`` and every series it lists.

    Returns ``{"registry": ..., "series": {series_id: MeasuredSeries},
    "errors": [...]}``; a corrupt series never aborts the whole load -- it is
    recorded in ``errors`` so the gate can fail loudly and honestly.
    """

    root = Path(root)
    reg_path = root / "registry.json"
    registry = json.loads(reg_path.read_text(encoding="utf-8"))
    series: Dict[str, MeasuredSeries] = {}
    errors: List[str] = []
    for entry in registry.get("series", []):
        rel = entry.get("path")
        if not rel:
            errors.append(f"registry entry without path: {entry!r}")
            continue
        path = root / rel
        try:
            s = load_series(path, verify=verify)
        except (MeasuredAnchorError, FileNotFoundError, json.JSONDecodeError) as exc:
            errors.append(f"{rel}: {exc}")
            continue
        if s.series_id != entry.get("series_id"):
            errors.append(f"{rel}: series_id {s.series_id!r} != registry {entry.get('series_id')!r}")
        series[s.series_id] = s
    return {"registry": registry, "series": series, "errors": errors}


# ---------------------------------------------------------------------------
# double-entry digitisation cross-check
# ---------------------------------------------------------------------------


def sampled_point_ids(points: Sequence[MeasuredPoint], fraction: float = 0.10) -> List[str]:
    """Deterministic 10 % sample of the point list (sorted ids, every k-th)."""

    ids = sorted(p.point_id for p in points)
    n = len(ids)
    take = max(1, math.ceil(fraction * n))
    if take >= n:
        return ids
    step = n / take
    return [ids[min(n - 1, int(round(i * step)))] for i in range(take)]


_FIELD_TOL = 1e-9


def double_entry_check(
    pass_a: Mapping[str, Mapping[str, Any]],
    pass_b: Mapping[str, Mapping[str, Any]],
    point_ids: Sequence[str],
    fields: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Compare two independent digitisation passes point by point.

    Both passes are ``{point_id: {field: value}}``.  A field agrees when it is
    present in both passes and numerically identical within ``1e-9`` relative
    tolerance (or equal as text).  A field present in only one pass counts as a
    *disagreement* -- a silently dropped column is a digitisation defect.

    Returns a record with the agreement rate and every mismatch, so the report
    can cite the sample rather than assert "checked by hand".
    """

    mismatches: List[Dict[str, Any]] = []
    compared = 0
    agreed = 0
    for pid in point_ids:
        ra = pass_a.get(pid)
        rb = pass_b.get(pid)
        if ra is None or rb is None:
            mismatches.append({"point_id": pid, "field": "<row>", "pass_a": ra is not None, "pass_b": rb is not None})
            compared += 1
            continue
        keys = list(fields) if fields is not None else sorted(set(ra) | set(rb))
        for key in keys:
            if key in ("point_id",):
                continue
            compared += 1
            va, vb = ra.get(key), rb.get(key)
            if _values_agree(va, vb):
                agreed += 1
            else:
                mismatches.append({"point_id": pid, "field": key, "pass_a": va, "pass_b": vb})
    rate = agreed / compared if compared else 0.0
    return {
        "n_points_sampled": len(point_ids),
        "n_fields_compared": compared,
        "n_fields_agreed": agreed,
        "agreement_rate": rate,
        "mismatches": mismatches,
        "passed": bool(compared > 0 and rate > 0.99),
    }


def _values_agree(a: Any, b: Any) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        if a == b:
            return True
        scale = max(abs(float(a)), abs(float(b)))
        return scale > 0.0 and abs(float(a) - float(b)) / scale <= _FIELD_TOL
    return str(a) == str(b)


# ---------------------------------------------------------------------------
# adjudication
# ---------------------------------------------------------------------------


def _rel_error(pred: float, meas: float) -> float:
    return abs(pred - meas) / abs(meas) if meas else math.inf


@dataclass
class Adjudication:
    """Per-point comparison of one model configuration against one series."""

    series_id: str
    config: str
    rows: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, **row: Any) -> None:
        self.rows.append(row)

    # -- summary -----------------------------------------------------------
    @property
    def errors(self) -> List[float]:
        return [r["rel_error"] for r in self.rows if r.get("rel_error") is not None]

    @property
    def median_rel_error(self) -> Optional[float]:
        e = self.errors
        return statistics.median(e) if e else None

    @property
    def mean_rel_error(self) -> Optional[float]:
        e = self.errors
        return sum(e) / len(e) if e else None

    @property
    def bias(self) -> Optional[float]:
        b = [r["signed_rel_error"] for r in self.rows if r.get("signed_rel_error") is not None]
        return sum(b) / len(b) if b else None

    @property
    def n_within_tolerance(self) -> int:
        return sum(1 for e in self.errors if e <= GATE_TOLERANCE)

    @property
    def verdict(self) -> str:
        med = self.median_rel_error
        if med is None:
            return "NOT_ADJUDICABLE"
        return "PASS_15PCT" if med <= GATE_TOLERANCE else "FAIL_15PCT"

    def summary(self) -> Dict[str, Any]:
        med = self.median_rel_error
        return {
            "series_id": self.series_id,
            "config": self.config,
            "n_points": len(self.rows),
            "median_rel_error": med,
            "mean_rel_error": self.mean_rel_error,
            "bias_signed_rel_error": self.bias,
            "n_within_15pct": self.n_within_tolerance,
            "tolerance": GATE_TOLERANCE,
            "verdict": self.verdict,
            "median_rel_error_pct": None if med is None else 100.0 * med,
        }

    def as_dict(self) -> Dict[str, Any]:
        return {"summary": self.summary(), "rows": self.rows}


def adjudicate_resistance(
    series: MeasuredSeries,
    model: Callable[[MeasuredPoint, MeasuredSeries], Dict[str, Any]],
    *,
    config: str = "default",
) -> Adjudication:
    """Compare a resistance model against every measured point of a series.

    ``model(point, series)`` must return ``{"R_kN": float, ...}``; the returned
    dict is stored verbatim in the row so the component split (crush / break /
    submersion) travels with the error, which is what makes an over-prediction
    *attributable* instead of merely visible.
    """

    adj = Adjudication(series_id=series.series_id, config=config)
    for pt in series.points:
        out = model(pt, series)
        pred = float(out["R_kN"])
        meas = pt.R_meas_kN
        if meas is None:
            adj.add(point_id=pt.point_id, h_m=pt.h_m, V_ms=pt.V_ms, R_pred_kN=pred,
                    R_meas_kN=None, rel_error=None, signed_rel_error=None, model=out)
            continue
        adj.add(
            point_id=pt.point_id,
            h_m=pt.h_m,
            V_ms=pt.V_ms,
            case_group=pt.case_group,
            R_pred_kN=pred,
            R_meas_kN=meas,
            R_unc_kN=pt.R_unc_kN,
            rel_error=_rel_error(pred, meas),
            signed_rel_error=(pred - meas) / meas,
            model=out,
        )
    return adj


def adjudicate_speed_power(
    series: MeasuredSeries,
    model: Callable[[MeasuredPoint, MeasuredSeries], Dict[str, Any]],
    *,
    config: str = "default",
) -> Dict[str, Any]:
    """Interval screening for a ``speed_power`` series.

    The measurement is the recorded ship speed at a measured ice thickness while
    the source reports only a *band* of engine power.  A point is *consistent*
    when the model's implied delivered power ``P = R(h, V) V / eta_D`` falls
    inside that band.  This is deliberately a screening test: it cannot support a
    15 % claim, and the record says so (``supports_15pct: False``).
    """

    rows = []
    for pt in series.points:
        out = model(pt, series)
        R = float(out["R_kN"])
        eta = float(out.get("eta_D", 0.55))
        P_pred = R * pt.V_ms / eta  # kN * m/s == kW
        lo, hi = pt.P_band_kW if pt.P_band_kW else (math.nan, math.nan)
        rows.append(
            {
                "point_id": pt.point_id,
                "h_m": pt.h_m,
                "V_ms": pt.V_ms,
                "R_pred_kN": R,
                "P_pred_kW": P_pred,
                "P_band_kW": [lo, hi],
                "consistent": bool(lo <= P_pred <= hi),
                "ratio_to_band_mid": P_pred / (0.5 * (lo + hi)) if hi > 0 else None,
                "model": out,
            }
        )
    n_ok = sum(1 for r in rows if r["consistent"])
    return {
        "series_id": series.series_id,
        "config": config,
        "kind": "power_band_screening",
        "n_points": len(rows),
        "n_consistent": n_ok,
        "verdict": (
            "SCREEN_CONSISTENT"
            if n_ok == len(rows)
            else ("SCREEN_PARTIAL" if n_ok > 0 else "SCREEN_FAIL")
        ),
        "supports_15pct": False,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# physical-plausibility gates
# ---------------------------------------------------------------------------


def physics_gates(
    resistance_fn: Callable[[float, float], float],
    *,
    h_grid: Sequence[float] = (0.3, 0.6, 0.9, 1.2, 1.5),
    v_grid: Sequence[float] = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0),
    tol_zero_speed: float = 1e-9,
) -> Dict[str, Any]:
    """Checks that hold for *any* level-ice resistance law (no fitting).

    ``R(h)`` monotone increasing at fixed ``V``; ``R(V)`` monotone increasing at
    fixed ``h``; ``R(V -> 0)`` finite and equal to the speed-independent part
    (no spurious speed term at rest); monotone response to a vanishing speed
    increment.
    """

    checks: List[Dict[str, Any]] = []

    def ok(name: str, passed: bool, observed: Any, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "observed": observed, "detail": detail})

    # R(h) increasing
    for V in v_grid:
        vals = [resistance_fn(h, V) for h in h_grid]
        mono = all(b > a for a, b in zip(vals, vals[1:]))
        ok(f"R_monotone_in_h@V={V}", mono, vals, "R(h) must increase with ice thickness")

    # R(V) increasing
    for h in h_grid[:3]:
        vals = [resistance_fn(h, V) for V in v_grid]
        mono = all(b > a for a, b in zip(vals, vals[1:]))
        ok(f"R_monotone_in_V@h={h}", mono, vals, "R(V) must increase with speed")

    # V -> 0 limit
    h0 = float(h_grid[len(h_grid) // 2])
    R0 = resistance_fn(h0, 0.0)
    R_eps = resistance_fn(h0, 1e-6)
    ok(
        "R_finite_at_zero_speed",
        math.isfinite(R0) and R0 > 0.0,
        R0,
        "R(V->0) must be finite and positive (quasi-static crushing + bending)",
    )
    ok(
        "R_continuous_at_zero_speed",
        abs(R_eps - R0) <= max(1e-6 * R0, tol_zero_speed),
        {"R(0)": R0, "R(1e-6)": R_eps},
        "no jump at V=0; the speed term must vanish continuously",
    )
    ok(
        "R_zero_speed_below_finite_speed",
        R0 < resistance_fn(h0, 1.0),
        {"R(0)": R0, "R(1)": resistance_fn(h0, 1.0)},
        "resistance at rest is the minimum of the branch",
    )

    return {
        "n_checks": len(checks),
        "n_passed": sum(1 for c in checks if c["passed"]),
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
        "grids": {"h_m": list(h_grid), "V_ms": list(v_grid)},
    }


# ---------------------------------------------------------------------------
# measured-data plausibility (C4 applied to the truth source itself)
# ---------------------------------------------------------------------------


def measured_monotonicity(
    series: MeasuredSeries,
    *,
    rel_tolerance: float = 0.10,
) -> Dict[str, Any]:
    """Apply the physics gates of C4 to the *measured* series itself.

    A measured series that contradicts ``R`` increasing with speed or thickness
    would not be a trustworthy truth source.  Real ice-basin data is noisy
    (the source reports standard deviations of 15-30 % of the mean), so a local
    decrease is not automatically a physical violation: it is *tolerated* when it
    stays inside ``rel_tolerance`` **or** inside the reported scatter of the pair,
    and it is always recorded verbatim in ``notes`` -- measured data is never
    "corrected" here.

    Thickness ordering uses the *envelope* (running maximum) because near-equal
    thicknesses (0.76 / 0.77 m, 0.95 / 0.96 m) appear at different speeds and a
    raw group mean would compare different dynamic conditions.
    """

    checks: List[Dict[str, Any]] = []
    notes: List[str] = []
    by_h: Dict[float, List[MeasuredPoint]] = {}
    for pt in series.points:
        if pt.R_meas_kN is None:
            continue
        by_h.setdefault(round(pt.h_m, 4), []).append(pt)

    # -- R(V) monotone at fixed thickness ----------------------------------
    for h, pts in sorted(by_h.items()):
        pts = sorted(pts, key=lambda p: p.V_ms)
        if len(pts) < 2:
            continue
        vals = [float(p.R_meas_kN) for p in pts]
        speeds = [p.V_ms for p in pts]
        strict = all(b >= a for a, b in zip(vals, vals[1:]))
        tolerated = True
        for a, b, pa, pb in zip(vals, vals[1:], pts, pts[1:]):
            if b >= a:
                continue
            drop = (a - b) / a
            scatter = 0.0
            for p_ in (pa, pb):
                if p_.R_unc_kN:
                    scatter = max(scatter, float(p_.R_unc_kN) / max(float(p_.R_meas_kN or 1.0), 1e-9))
            if drop > max(rel_tolerance, scatter):
                tolerated = False
            else:
                notes.append(
                    f"h={h} m: measured R decreases {100*drop:.1f}% from V={pa.V_ms} to {pb.V_ms} m/s "
                    f"(inside declared tolerance/scatter; recorded, not corrected)"
                )
        checks.append(
            {
                "check": f"measured_R_monotone_in_V@h={h}",
                "passed": bool(strict or tolerated),
                "strictly_monotone": bool(strict),
                "speeds": speeds,
                "values_kN": vals,
                "reason_if_failed": None if strict or tolerated else "decrease exceeds tolerance and reported scatter",
            }
        )

    # -- R(h) monotone in the envelope sense --------------------------------
    hs = sorted(by_h)
    if len(hs) >= 2:
        envelope = []
        running = float("-inf")
        for h in hs:
            running = max(running, max(float(p.R_meas_kN) for p in by_h[h]))
            envelope.append(running)
        checks.append(
            {
                "check": "measured_R_envelope_monotone_in_h",
                "passed": all(b >= a for a, b in zip(envelope, envelope[1:])),
                "h_m": hs,
                "envelope_max_kN": envelope,
                "detail": "running maximum of measured resistance never decreases with thickness",
            }
        )

    # -- finiteness ---------------------------------------------------------
    checks.append(
        {
            "check": "measured_values_finite_positive",
            "passed": all(math.isfinite(p.R_meas_kN) and p.R_meas_kN > 0 for p in series.points if p.R_meas_kN is not None),
            "detail": "every measured resistance is finite and positive",
        }
    )

    return {
        "series_id": series.series_id,
        "n_checks": len(checks),
        "n_passed": sum(1 for c in checks if c["passed"]),
        "passed": all(c["passed"] for c in checks) if checks else True,
        "notes": notes,
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# gate ledger
# ---------------------------------------------------------------------------


def gate_ledger(
    *,
    schema_report: Mapping[str, Any],
    double_entry_reports: Sequence[Mapping[str, Any]],
    adjudications: Sequence[Adjudication],
    physics: Mapping[str, Any],
    legacy_gate: Mapping[str, Any],
    min_series: int = REQUIRED_SERIES,
) -> Dict[str, Any]:
    """Assemble the campaign-node criteria C1..C5 into one auditable record."""

    c1 = bool(schema_report.get("all_valid")) and bool(schema_report.get("all_digests_verified"))
    c1 = c1 and int(schema_report.get("n_series", 0)) >= int(min_series)

    c2 = bool(double_entry_reports) and all(bool(r.get("passed")) for r in double_entry_reports)

    resistance_adjs = [a for a in adjudications if a.verdict != "NOT_ADJUDICABLE"]
    best = min(resistance_adjs, key=lambda a: a.median_rel_error) if resistance_adjs else None
    c3 = bool(best is not None and best.verdict == "PASS_15PCT")

    c4 = bool(physics.get("passed")) and bool(schema_report.get("measured_plausibility_passed", True))

    c5 = bool(legacy_gate.get("all_passed"))

    return {
        "C1_schema_and_digest": {
            "passed": c1,
            "n_series": schema_report.get("n_series"),
            "required_series": min_series,
            "n_points": schema_report.get("n_points"),
            "all_valid": schema_report.get("all_valid"),
            "all_digests_verified": schema_report.get("all_digests_verified"),
            "errors": schema_report.get("errors", []),
        },
        "C2_double_entry": {
            "passed": c2,
            "sample_fraction": 0.10,
            "min_agreement": 0.99,
            "series": [
                {
                    "series_id": r.get("series_id"),
                    "n_points_sampled": r.get("n_points_sampled"),
                    "agreement_rate": r.get("agreement_rate"),
                    "passed": r.get("passed"),
                }
                for r in double_entry_reports
            ],
        },
        "C3_15pct_adjudication": {
            "passed": c3,
            "tolerance": GATE_TOLERANCE,
            "best_series_id": None if best is None else best.series_id,
            "best_median_rel_error": None if best is None else best.median_rel_error,
            "best_verdict": None if best is None else best.verdict,
            "honest_fail": (not c3),
            "per_series": [a.summary() for a in adjudications],
        },
        "C4_physics_gates": {
            "passed": c4,
            "physics": physics,
        },
        "C5_legacy_full_spectrum": {
            "passed": c5,
            "n_anchors": legacy_gate.get("n_anchors"),
            "n_passed": legacy_gate.get("n_passed"),
            "negative_controls_passed": legacy_gate.get("negative_controls_passed"),
        },
        "all_passed": bool(c1 and c2 and c3 and c4 and c5),
        "evidence_state": "MEASURED_ANCHOR_AVAILABLE" if c1 else "NO_MEASURED_ANCHOR",
        "gate_state": "JUDGED" if (c1 and c2) else "WITHHELD",
    }