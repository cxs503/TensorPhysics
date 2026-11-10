"""Build the ice-resistance measured-anchor assets (mission 9425).

Reads the archived transcriptions under ``data/measured/ice_resistance/series/*/source``
and emits, for every series:

* ``pass_a.json`` / ``pass_b.json`` -- the two independent digitisation passes in
  canonical units (double-entry input for criterion C2);
* ``series.json`` -- the schema-validated measured-dataset asset, whose
  ``provenance.sources[*].sha256`` are computed from the archived bytes (criterion
  C1: no provenance without a digest);
* ``../../registry.json`` -- the asset registry with the digest of every series.

Nothing is typed twice: the series document is generated from the passes, so the
numbers in the report, the JSON and the assets cannot drift apart.

Run:  python benchmarks/ice_resistance/measured/build_assets.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
for _p in (str(REPO / "src"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tensorphysics.ice.measured_anchor import (  # noqa: E402
    SCHEMA_VERSION,
    sha256_bytes,
    sha256_file,
    validate_series,
)

DATA_ROOT = REPO / "data" / "measured" / "ice_resistance"

SI_FIELDS = (
    "h_m",
    "V_ms",
    "sigma_f_kPa",
    "sigma_c_kPa",
    "E_MPa",
    "R_meas_kN",
    "R_unc_kN",
    "R_max_kN",
    "R_min_kN",
    "R_alt_kN",
    "lindqvist_kN",
    "riska_kN",
    "jeong_kN",
    "keinonen_kN",
)

S2_FIELDS = ("h_m", "V_ms", "V_final_ms", "P_band_min_kW", "P_band_max_kW")


def _num(text: str) -> float:
    return float(text)


def _opt(text: str) -> float | None:
    text = text.strip()
    if text in ("", "-", "NA", "n/a"):
        return None
    return float(text)


# ---------------------------------------------------------------------------
# MT Uikku (Hu & Zhou 2016): pass A = column-major CSV, pass B = row-major text
# ---------------------------------------------------------------------------


def mtuikku_pass_a(src: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with src.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["test_id"]] = {
                "h_m": _num(row["ice_thickness_m"]),
                "V_ms": _num(row["ice_drift_speed_m_s"]),
                "sigma_f_kPa": _num(row["flexural_strength_kPa"]),
                "sigma_c_kPa": _num(row["compressive_strength_kPa"]),
                "E_MPa": _num(row["ice_elastic_modulus_MPa"]),
                "R_meas_kN": _num(row["measured_mean_resistance_table3_kN"]),
                "R_unc_kN": _opt(row["measured_std_kN"]),
                "R_max_kN": _opt(row["measured_max_kN"]),
                "R_min_kN": _opt(row["measured_min_kN"]),
                "R_alt_kN": _opt(row["experimental_reference_table8_kN"]),
                "lindqvist_kN": _opt(row["lindqvist_prediction_kN"]),
                "riska_kN": _opt(row["riska_prediction_kN"]),
                "jeong_kN": _opt(row["jeong_prediction_kN"]),
                "keinonen_kN": _opt(row["keinonen_prediction_kN"]),
            }
    return out


_TOKEN = {
    "h": ("h_m", 1.0),
    "sf": ("sigma_f_kPa", 1e3),  # MPa -> kPa
    "sc": ("sigma_c_kPa", 1e3),  # MPa -> kPa
    "E": ("E_MPa", 1e3),  # GPa -> MPa
    "V": ("V_ms", 1.0),
    "Rmean": ("R_meas_kN", 1.0),
    "Rstd": ("R_unc_kN", 1.0),
    "Rmax": ("R_max_kN", 1.0),
    "Rmin": ("R_min_kN", 1.0),
    "Rref_t8": ("R_alt_kN", 1.0),
    "Lind": ("lindqvist_kN", 1.0),
    "Riska": ("riska_kN", 1.0),
    "Jeong": ("jeong_kN", 1.0),
    "Keinonen": ("keinonen_kN", 1.0),
}


def mtuikku_pass_b(src: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for raw in src.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        pid, rest = tokens[0], tokens[1:]
        rec: Dict[str, Any] = {}
        for tok in rest:
            key, _, val = tok.partition("=")
            if key not in _TOKEN:
                raise KeyError(f"unknown token {key!r} in pass B line: {line}")
            field, scale = _TOKEN[key]
            v = _opt(val)
            rec[field] = None if v is None else v * scale
        out[pid] = rec
    return out


# ---------------------------------------------------------------------------
# Agulhas II full-scale trial: pass A = columnar CSV, pass B = row-major text
# ---------------------------------------------------------------------------


def agulhas_pass_a(src: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with src.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pid = row["case_group"].upper()
            out[pid] = {
                "h_m": _num(row["h_m"]),
                "V_ms": _num(row["V_recorded_mean_ms"]),
                "V_final_ms": _num(row["V_recorded_final_ms"]),
                "P_band_min_kW": _num(row["P_band_min_kW"]),
                "P_band_max_kW": _num(row["P_band_max_kW"]),
            }
    return out


def agulhas_pass_b(src: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for raw in src.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        pid = tokens[0].upper()
        rec: Dict[str, Any] = {}
        for tok in tokens[1:]:
            key, _, val = tok.partition("=")
            field = {
                "h": "h_m",
                "Vmean": "V_ms",
                "Vfinal": "V_final_ms",
                "Pmin": "P_band_min_kW",
                "Pmax": "P_band_max_kW",
            }[key]
            rec[field] = _num(val)
        out[pid] = rec
    return out


# ---------------------------------------------------------------------------
# series document assembly
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def build_mtuikku() -> Dict[str, Any]:
    sdir = DATA_ROOT / "series" / "mtuikku_hz2016"
    src_dir = sdir / "source"
    csv_src = src_dir / "hz2016_level_ice_table3.csv"
    txt_src = src_dir / "pass_b_rowmajor_mixed_units.txt"

    pass_a = mtuikku_pass_a(csv_src)
    pass_b = mtuikku_pass_b(txt_src)
    _write_json(sdir / "pass_a.json", pass_a)
    _write_json(sdir / "pass_b.json", pass_b)

    points: List[Dict[str, Any]] = []
    for pid in sorted(pass_a, key=lambda p: int(p)):
        a = pass_a[pid]
        points.append(
            {
                "point_id": pid,
                "h_m": a["h_m"],
                "V_ms": a["V_ms"],
                "provenance_ref": "Hu & Zhou (2016) Table 3 (mean force) / Table 8 (reference value)",
                "R_meas_kN": a["R_meas_kN"],
                "R_unc_kN": a["R_unc_kN"],
                "R_max_kN": a["R_max_kN"],
                "R_min_kN": a["R_min_kN"],
                "R_alt_kN": a["R_alt_kN"],
                "sigma_f_kPa": a["sigma_f_kPa"],
                "sigma_c_kPa": a["sigma_c_kPa"],
                "E_MPa": a["E_MPa"],
                "case_group": {"I": "a", "II": "b", "III": "c", "IV": "d"}[_group_of(pid)],
                "literature_baselines_kN": {
                    "lindqvist": a["lindqvist_kN"],
                    "riska": a["riska_kN"],
                    "jeong": a["jeong_kN"],
                    "keinonen": a["keinonen_kN"],
                    "_note": "published predictions reproduced from the source Table 8; context only, never truth",
                },
                "notes": "" if a["R_unc_kN"] is not None else "force statistics not reported in the retrieved table (left empty, not imputed)",
            }
        )

    sources = []
    for p, kind, role in (
        (csv_src, "transcription-column-major", "pass A digitisation of Tables 3/8 (SI-derived units)"),
        (txt_src, "transcription-row-major", "pass B digitisation (row-major, MPa/GPa unit path)"),
    ):
        sources.append(
            {
                "artifact": f"source/{p.name}",
                "sha256": sha256_file(p),
                "bytes": p.stat().st_size,
                "kind": kind,
                "role": role,
            }
        )

    doc = {
        "schema_version": SCHEMA_VERSION,
        "series_id": "MTUIKKU-HZ2016",
        "title": "MT Uikku level-ice model tests (ice basin, scale 1:31.56), resistance at full scale",
        "ice_state": "level",
        "measured_quantity": "resistance",
        "ship": {
            "name": "MT Uikku (double-acting icebreaking tanker, Neste/Nemarc)",
            "L_pp_m": 150.0,
            "B_m": 21.3,
            "T_m": 9.5,
            "bow_angle_deg": 21.0,
            "flare_deg": 30.0,
            "displacement_t": None,
            "source": "Hu & Zhou (2016) ship particulars (also tabulated by Koto & Afrizal 2017, Table 2)",
            "convention_notes": {
                "bow_angle_deg": "waterline entrance angle as reported (21 deg); used as the plan-view stem half-angle argument phi/2 in R_break",
                "flare_deg": "stem angle 30 deg as reported; used as the plating angle to the horizontal in R_crush (assumption A1)",
            },
        },
        "provenance": {
            "citation": "Hu, Z. & Zhou, L. (2016) Further study on level ice resistance and channel resistance for an icebreaking vessel, Int. J. Naval Architecture and Ocean Engineering 8(2) 169-176",
            "doi": "10.1016/j.ijnaoe.2016.01.004",
            "url": "https://doi.org/10.1016/j.ijnaoe.2016.01.004",
            "grade": "MEASURED-PRIMARY",
            "tables": ["Table 3 (measured force statistics)", "Table 8 (experimental reference + published predictions)"],
            "retrieval_note": (
                "Publisher PDF (Elsevier, CC BY-NC-ND) returned HTTP 403 from this worktree's network on "
                "2026-11-10; the digits were ingested from the archived in-ecosystem transcription of the same "
                "tables (TensorDEM mt_uikku_level_ice.csv) whose README documents the Table-3/Table-8 discrepancy. "
                "Pass A and pass B are two independent read orders/unit paths over that transcript."
            ),
            "sources": sources,
            "scale": {"model_scale": 31.56, "reported_frame": "full_scale"},
            "limitations": [
                "digitised summary forces, not the raw force time series",
                "Table 3 and Table 8 differ by ~1-2% on several cases; the mean-force column is used as truth and the Table 8 reference is carried as an alternative",
            ],
        },
        "points": points,
    }
    errs = validate_series(doc, root=sdir)
    if errs:
        raise SystemExit("MTUIKKU series failed validation: " + "; ".join(errs))
    _write_json(sdir / "series.json", doc)
    return doc


def _group_of(pid: str) -> str:
    return {"103": "I", "104": "I", "205": "II", "206": "II",
            "301": "III", "302": "III", "303": "III",
            "401": "IV", "402": "IV", "403": "IV"}[pid]


def build_agulhas() -> Dict[str, Any]:
    sdir = DATA_ROOT / "series" / "agulhas2_fs2012"
    src_dir = sdir / "source"
    csv_src = src_dir / "pass_a_table4_columnar.csv"
    txt_src = src_dir / "pass_b_table4_rowmajor.txt"

    pass_a = agulhas_pass_a(csv_src)
    pass_b = agulhas_pass_b(txt_src)
    _write_json(sdir / "pass_a.json", pass_a)
    _write_json(sdir / "pass_b.json", pass_b)

    points = []
    for pid in sorted(pass_a):
        a = pass_a[pid]
        points.append(
            {
                "point_id": pid,
                "h_m": a["h_m"],
                "V_ms": a["V_ms"],
                "provenance_ref": "Aalto thesis (handle 123456789/46932) Table 4, cases a-d (recorded mean speed)",
                "V_final_ms": a["V_final_ms"],
                "P_band_kW": [a["P_band_min_kW"], a["P_band_max_kW"]],
                "case_group": pid.lower(),
                "notes": "engine power reported as a range over the four level-ice periods, not per case",
            }
        )

    sources = []
    for p, kind, role in (
        (csv_src, "transcription-column-major", "pass A digitisation of Table 4"),
        (txt_src, "transcription-row-major", "pass B digitisation (row-major)"),
    ):
        sources.append(
            {
                "artifact": f"source/{p.name}",
                "sha256": sha256_file(p),
                "bytes": p.stat().st_size,
                "kind": kind,
                "role": role,
            }
        )

    doc = {
        "schema_version": SCHEMA_VERSION,
        "series_id": "AGULHAS2-FS2012",
        "title": "PSRV S.A. Agulhas II full-scale level-ice trial (Baltic Sea, 21-22 March 2012): recorded speed vs measured thickness",
        "ice_state": "level",
        "measured_quantity": "speed_power",
        "ship": {
            "name": "S.A. Agulhas II (polar supply & research vessel, PC 5)",
            "L_pp_m": 134.2,
            "B_m": 21.7,
            "T_m": 7.65,
            "bow_angle_deg": 30.0,
            "flare_deg": 25.0,
            "displacement_t": 13687.0,
            "source": "Aalto thesis Table 3 (length/breadth/draught/displacement); bow and flare angles are ASSUMED default values (not reported) -- this series is therefore screening-only",
            "convention_notes": {
                "bow_angle_deg": "ASSUMED 30 deg (PC5-typical); not measured",
                "flare_deg": "ASSUMED 25 deg (model default); not measured",
            },
        },
        "provenance": {
            "citation": "Aalto University doctoral thesis, 'Evaluation of selected state-of-the-art methods for ship performance in level ice' (Table 3, Table 4, Sec. 2.4); measurements from the S.A. Agulhas II ice trial reported by Suominen et al. (2013) and Bekker et al. (2014)",
            "doi": None,
            "url": "https://aaltodoc.aalto.fi/handle/123456789/46932",
            "grade": "MEASURED-SECONDARY",
            "tables": ["Table 3 (ship parameters)", "Table 4 (recorded vs simulated speed)", "Sec. 2.4 (power range)"],
            "retrieval_note": (
                "Full-text extract of the thesis retrieved from AaltoDspace on 2026-11-10 (the repository's "
                "text bitstream, not the publisher PDF). Per-case engine power is not tabulated; the source "
                "reports only the 2000-5400 kW range across the four level-ice periods."
            ),
            "sources": sources,
            "scale": {"model_scale": None, "reported_frame": "full_scale"},
            "limitations": [
                "no measured thrust/resistance: adjudication is a power-band consistency screen only",
                "hull bow angle and flare not reported -> assumed; used only for screening",
                "ice rubble contamination in case (d) (source notes speed drops from rubble)",
            ],
        },
        "points": points,
    }
    errs = validate_series(doc, root=sdir)
    if errs:
        raise SystemExit("AGULHAS series failed validation: " + "; ".join(errs))
    _write_json(sdir / "series.json", doc)
    return doc


def build_registry(docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    entries = []
    for doc in docs:
        path = DATA_ROOT / "series" / doc["series_id"].lower().replace("-", "_")
        entries.append(
            {
                "series_id": doc["series_id"],
                "path": f"series/{path.name}/series.json",
                "sha256": sha256_file(path / "series.json"),
                "grade": doc["provenance"]["grade"],
                "measured_quantity": doc["measured_quantity"],
                "n_points": len(doc["points"]),
                "doi": doc["provenance"]["doi"],
            }
        )
    registry = {
        "asset_mode": "measured_dataset/v1",
        "schema_version": SCHEMA_VERSION,
        "owner": "TensorPhysics :: tensorphysics.ice.measured_anchor",
        "series": entries,
        "gate": {"tolerance": 0.15, "double_entry_sample_fraction": 0.10, "min_series": 2},
    }
    _write_json(DATA_ROOT / "registry.json", registry)
    return registry


def main() -> int:
    docs = [build_mtuikku(), build_agulhas()]
    reg = build_registry(docs)
    print(json.dumps({"series": [d["series_id"] for d in docs], "n_points": sum(len(d["points"]) for d in docs),
                      "registry_digests": [e["sha256"][:16] for e in reg["series"]]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())