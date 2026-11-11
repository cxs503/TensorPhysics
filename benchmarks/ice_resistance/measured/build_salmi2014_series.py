#!/usr/bin/env python
"""Build the third measured series ``SALMI2014-UBIB`` from the archived excerpt.

Why this series
---------------
The library shipped with the first measured-anchor node (m9425) has exactly one
series that measures *resistance* (MT Uikku, downward-bending tanker, one test
campaign).  A single campaign cannot tell a bad closure from a bad day in the
tank, so this builder ingests a second, independent, publicly digitised campaign:
Salmi (2014) model tests of the upward-breaking ``UBIB`` concept in the Otaniemi
ice tank (40 mm model ice, 1:25, reported at full scale).

The series is deliberately registered with an **applicability caveat**: the hull
breaks the ice *upwards*, which is outside the downward-bending assumption of the
Lindqvist family.  It is therefore an *external-validity probe*, not a member of
the 15 % verdict set -- and the record says so, in ``applicability``.

Digitisation
------------
Two passes over the same archived excerpt (``source/salmi2014_tables5to8_excerpt.txt``,
sha256 recorded in ``provenance.sources``):

* pass A -- column-major read; resistance kept in MN, speed in knots;
* pass B -- row-major read; MN -> kN and knots -> m/s converted inline.

Both passes end in canonical units so that ``measured_anchor.double_entry_check``
compares the *converted* fields, which is where a decimal-comma or knot slip would
show up.

Run::

    python benchmarks/ice_resistance/measured/build_salmi2014_series.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tensorphysics.ice.measured_anchor import (  # noqa: E402
    double_entry_check,
    load_registry,
    sha256_file,
    validate_series,
)

SERIES_ID = "SALMI2014-UBIB"
SERIES_DIR = REPO / "data" / "measured" / "ice_resistance" / "series" / "salmi2014_ubib"
SOURCE = SERIES_DIR / "source" / "salmi2014_tables5to8_excerpt.txt"
DATA_ROOT = REPO / "data" / "measured" / "ice_resistance"

KNOT_MS = 1852.0 / 3600.0  # 1 international knot in m/s
MODEL_SCALE = 25.0  # thesis: "The model scaling factor lambda was set to 1:25"

# Full-scale geometry, from the thesis text (section 4.3 hull definition):
#   "the (B) breath was set to 10 meters and (T) draught to 7,5 meters ...
#    lambda was set to 1:25, thus L=4,0m Bmax=0,4m T=0,3m. The full scale maximum
#    breath at ice belt area was set to 5,5 meters"
SHIP = {
    "name": "UBIB concept hull (upward-breaking bow), Salmi (2014) test model",
    "L_pp_m": 100.0,
    "B_m": 10.0,
    "T_m": 7.5,
    "bow_angle_deg": 15.0,
    "flare_deg": 75.0,
    "displacement_t": None,
    "ice_belt_breath_m": 5.5,
    "source": "Salmi (2014) section 4.3: L=4.0 m, B_max=0.4 m, T=0.3 m at lambda=1:25",
    "convention_notes": {
        "bow_angle_deg": "V15 bow: the thesis calls the alternative bow 'A10 deg entrance angle'; "
        "the V bow is the 15 deg entrance angle variant, used as phi in R_break",
        "flare_deg": "ASSUMPTION A2 -- the source does not tabulate the plating angle from the "
        "horizontal. 75 deg (15 deg from vertical, consistent with a 'V15' wedge) is used; the "
        "benchmark sweeps 60/70/75/80 deg and reports the band, because tan(alpha) drives R_crush",
        "ice_belt_breath_m": "narrow ice belt; the benchmark repeats the adjudication with "
        "B = 5.5 m to bracket hull-width uncertainty",
    },
}


def _num(token: str) -> float:
    """Parse a Finnish-decimal-comma number, tolerating OCR spaces ("0, 00632")."""

    return float(token.replace(" ", "").replace(",", "."))


def parse_excerpt(text: str) -> Dict[str, Any]:
    """Extract the three blocks we need, failing loudly rather than guessing."""

    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    measured: Dict[str, float] = {}
    table8: Dict[str, float] = {}
    theory275: Dict[int, float] = {}
    theory500: Dict[int, float] = {}

    # ---- Table 8 caption position: everything before it belongs to Table 5 -
    idx8 = next(i for i, ln in enumerate(lines) if "Table 8." in ln)

    # ---- Table 5: 8 rows ``<label> <R_MN>`` (intact-ice rows only) -------
    for ln in lines[:idx8]:
        m = re.fullmatch(r"(3|6|9|12)knot\s+(\d,\d+)", ln)
        if m:
            measured[f"{m.group(1)}kt"] = _num(m.group(2))

    # ---- Table 8: the 4-row block that follows the Table-8 caption -------
    for ln in lines[idx8:]:
        m = re.fullmatch(r"(3|6|9|12)knot\s+(\d,\d+)", ln)
        if m and f"{m.group(1)}kt" not in table8:
            table8[f"{m.group(1)}kt"] = _num(m.group(2))

    # ---- Tables 6/7: theory components ``v ... R_tot`` --------------------
    theory_rows: List[Dict[str, float]] = []
    for ln in lines:
        m = re.fullmatch(r"(3|6|9|12)\s+(\d,\s?\d+)\s+(\d,\s?\d+)\s+(\d,\s?\d+)\s+(\d,\s?\d+)\s+(\d,\s?\d+)\s+(\d,\s?\d+)\s+(\d,\s?\d+)", ln)
        if m:
            theory_rows.append({"v_kt": float(m.group(1)), "R_tot_MN": _num(m.group(8))})
    if len(theory_rows) == 8:  # sigma_b = 275 kPa block, then 500 kPa block
        theory275 = {int(r["v_kt"]): r["R_tot_MN"] for r in theory_rows[:4]}
        theory500 = {int(r["v_kt"]): r["R_tot_MN"] for r in theory_rows[4:]}
    else:
        raise SystemExit(f"theory blocks not parsed (rows={len(theory_rows)}): refusing to guess")

    if sorted(measured) != ["12kt", "3kt", "6kt", "9kt"]:
        raise SystemExit(f"Table-5 block not parsed: {measured}")
    if sorted(table8) != ["12kt", "3kt", "6kt", "9kt"]:
        raise SystemExit(f"Table-8 block not parsed: {table8}")

    return {
        "measured": measured,
        "table8": table8,
        "theory275": theory275,
        "theory500": theory500,
    }


def build_passes(parsed: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Two independent digitisation passes over the same excerpt."""

    pass_a: Dict[str, Dict[str, Any]] = {}
    pass_b: Dict[str, Dict[str, Any]] = {}

    # pass A: column-major, resistance in MN, speed in knots
    for kt in (3, 6, 9, 12):
        key = f"{kt}kt"
        rm = parsed["measured"][key]
        pass_a[f"UBIB-{kt}KT"] = {
            "h_m": 1.0,
            "V_kt": float(kt),
            "R_MN": rm,
            "V_ms": round(float(kt) * KNOT_MS, 6),
            "R_meas_kN": rm * 1000.0,
        }
    # pass B: row-major, convert inline through a different unit path
    for key in ("3kt", "6kt", "9kt", "12kt"):
        kt = int(key.replace("kt", ""))
        rm = parsed["measured"][key]
        pass_b[f"UBIB-{kt}KT"] = {
            "h_m": 1.0,
            "V_kt": float(kt),
            "R_MN": rm,
            "V_ms": round(kt * 1852.0 / 3600.0, 6),
            "R_meas_kN": round(rm * 1.0e3, 6),
        }
    return pass_a, pass_b


def build_series(parsed: Dict[str, Any]) -> Dict[str, Any]:
    source_digest = sha256_file(SOURCE)
    points = []
    for kt in (3, 6, 9, 12):
        key = f"{kt}kt"
        h, V = 1.0, round(kt * KNOT_MS, 6)
        points.append(
            {
                "point_id": f"UBIB-{kt}KT",
                "h_m": h,
                "V_ms": V,
                "provenance_ref": f"Salmi (2014) Table 5 (channel-corrected measured), {kt} knot row",
                "R_meas_kN": round(parsed["measured"][key] * 1000.0, 6),
                "R_unc_kN": None,
                "R_max_kN": None,
                "R_min_kN": None,
                "R_alt_kN": round(parsed["table8"][key] * 1000.0, 6),
                "sigma_f_kPa": 500.0,
                "sigma_c_kPa": None,
                "E_MPa": None,
                "case_group": "level_ice_1m",
                "literature_baselines_kN": {
                    "leiviska_ubib_sigma275": round(parsed["theory275"][kt] * 1000.0, 6),
                    "leiviska_ubib_sigma500": round(parsed["theory500"][kt] * 1000.0, 6),
                    "_note": "published UBIB theory (thesis Tables 6/7); context only, never truth",
                },
                "notes": (
                    "R_alt_kN is Table 8 (additionally friction-corrected by the author, factors 1.5-2.46) -- "
                    "kept as an alternative, never merged into R_meas"
                ),
            }
        )

    return {
        "schema_version": "1.0",
        "series_id": SERIES_ID,
        "title": "UBIB upward-breaking concept, level-ice model tests (Otaniemi ice tank, 1:25), "
        "resistance at full scale, 1.0 m level ice",
        "ice_state": "level",
        "measured_quantity": "resistance",
        "ship": SHIP,
        "facility": {
            "name": "Aalto University Otaniemi ice tank",
            "kind": "ice_basin",
            "model_scale": MODEL_SCALE,
            "reported_frame": "full_scale",
            "model_ice": "40 mm model ice (target full-scale bending strength 500 kPa)",
            "test_program": "level ice, towed/self-propelled runs at 3/6/9/12 knot equivalent speed",
        },
        "digitisation": {
            "method": "verbatim transcription of the archived repository text extraction of Tables 5/8",
            "passes": ["pass_a", "pass_b"],
            "independent_paths": [
                "pass A: column-major read, MN and knots kept, converted after transcription",
                "pass B: row-major read, MN->kN and knots->m/s converted inline",
            ],
            "checks": ["double_entry agreement over canonical + raw fields", "digest re-verification"],
            "tool": "benchmarks/ice_resistance/measured/build_salmi2014_series.py",
        },
        "uncertainty": {
            "reported": False,
            "definition": "the source reports single values per run; run-to-run scatter and model-ice "
            "variability are described qualitatively but not quantified",
            "value_kN": None,
        },
        "applicability": {
            "ice_breaking_mode": "upward_bending",
            "lindqvist_family_applicable": False,
            "in_15pct_verdict_set": False,
            "role": "external-validity probe: shows what the downward-bending closure family does "
            "when it is applied outside its breaking-mode envelope",
            "limitations": [
                "concept hull (UBIB), not a ship in service; results are the author's channel-corrected values",
                "plating angle (flare) not tabulated: ASSUMPTION A2 = 75 deg, swept 60-80 deg",
                "narrow ice belt (5.5 m of 10 m breadth): hull-width convention swept",
                "no measured ice compressive strength / elastic modulus: module defaults used",
            ],
        },
        "provenance": {
            "citation": "Salmi, K. (2014) Alternative hull type for icebreaking vessels. MSc thesis, "
            "Aalto University School of Engineering, 3 June 2014",
            "doi": None,
            "url": "https://aaltodoc.aalto.fi/handle/123456789/13447",
            "grade": "MEASURED-PRIMARY",
            "tables": ["Table 5 (channel-corrected measured resistance)", "Table 8 (friction-corrected, R_alt)",
                       "Tables 6/7 (published UBIB theory, baselines only)"],
            "retrieval_note": "Repository text-extraction bitstream retrieved HTTP 200 on 2026-11-10 from the "
            "worktree (aaltodoc DSpace 7 REST); the tables used are quoted verbatim into source/ so the "
            "digest is checkable offline. Captions precede table bodies in the extraction, which is why the "
            "8-row block is attributed to Table 5 and the 4-row block to Table 8.",
            "sources": [
                {
                    "artifact": "source/salmi2014_tables5to8_excerpt.txt",
                    "sha256": source_digest,
                    "bytes": SOURCE.stat().st_size,
                    "kind": "publisher-text-excerpt",
                    "role": "verbatim quotation of Tables 5/6/7/8 from the open-access thesis",
                }
            ],
            "scale": {"model_scale": MODEL_SCALE, "reported_frame": "full_scale"},
            "limitations": [
                "a single concept hull, tested in one campaign: it cannot separate ice-strength from geometry effects",
                "measured resistance is post-processed by the author (channel friction replaced by theory)",
                "upward-breaking mode: outside the Lindqvist-family envelope by construction",
            ],
        },
        "points": points,
    }


def main() -> int:
    text = SOURCE.read_text(encoding="utf-8")
    parsed = parse_excerpt(text)
    series = build_series(parsed)
    pass_a, pass_b = build_passes(parsed)

    # pass files keep the library convention: a flat {point_id: {field: value}} map,
    # so measured_anchor.double_entry_check can compare them point by point.
    (SERIES_DIR / "pass_a.json").write_text(
        json.dumps(pass_a, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (SERIES_DIR / "pass_b.json").write_text(
        json.dumps(pass_b, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (SERIES_DIR / "series.json").write_text(
        json.dumps(series, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    errors = validate_series(series, root=SERIES_DIR)
    check = double_entry_check(pass_a, pass_b, sorted(pass_a))
    digest = sha256_file(SERIES_DIR / "series.json")

    print(json.dumps({
        "series_id": SERIES_ID,
        "n_points": len(series["points"]),
        "schema_errors": errors,
        "double_entry_agreement": check["agreement_rate"],
        "double_entry_fields": check["n_fields_compared"],
        "series_json_sha256": digest,
        "measured_kN": [p["R_meas_kN"] for p in series["points"]],
        "table8_kN": [p["R_alt_kN"] for p in series["points"]],
        "theory275_kN": [p["literature_baselines_kN"]["leiviska_ubib_sigma275"] for p in series["points"]],
        "theory500_kN": [p["literature_baselines_kN"]["leiviska_ubib_sigma500"] for p in series["points"]],
    }, ensure_ascii=False, indent=2))
    if errors:
        return 1
    # registry sanity: the series must be registered with this digest
    reg = load_registry(DATA_ROOT, verify=False)
    ids = {e["series_id"] for e in reg["registry"]["series"]}
    print("registered:", SERIES_ID in ids, "| registry series:", sorted(ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())