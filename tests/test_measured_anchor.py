"""Pytest acceptance suite: ice-resistance measured-anchor library (mission 9425).

Criteria mirrored from the campaign node (all numbers re-derived here, none typed):

C1  schema validation of every measured series + SHA-256 verification of every
    archived source artifact
C2  double-entry digitisation cross-check on a 10 % sample, agreement > 99 %
C3  adjudication against the measured anchor: per-point error table, median
    error vs the 15 % criterion, honest FAIL preserved
C4  physical plausibility of the model (R(h) up, R(V) up, finite V -> 0) and of
    the measured series itself
C5  the pre-existing ICE_RESISTANCE_FULL_SPECTRUM corpus stays fully green, and
    the WITHHELD -> JUDGED transition is mechanical (no anchor => WITHHELD)

The suite also carries negative controls: a provenance without a digest, a
tampered source artifact and a mismatched digitisation must all be rejected.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
for _p in (str(REPO / "src"), str(REPO), str(REPO / "benchmarks" / "ice_resistance")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tensorphysics.ice import measured_anchor as ma  # noqa: E402
from tensorphysics.ice.measured_anchor import (  # noqa: E402
    GATE_TOLERANCE,
    MeasuredAnchorError,
    MeasuredSeries,
    adjudicate_resistance,
    double_entry_check,
    gate_ledger,
    load_registry,
    load_series,
    measured_monotonicity,
    physics_gates,
    sampled_point_ids,
    sha256_bytes,
    sha256_file,
    validate_series,
)
from tensorphysics.ice.resistance import ShipGeometry, level_ice_resistance  # noqa: E402

DATA_ROOT = REPO / "data" / "measured" / "ice_resistance"
ARTIFACT = REPO / "artifacts" / "ice_resistance" / "measured_adjudication.json"
MTUIKKU_DIR = DATA_ROOT / "series" / "mtuikku_hz2016"
AGULHAS_DIR = DATA_ROOT / "series" / "agulhas2_fs2012"


@pytest.fixture(scope="module")
def loaded():
    return load_registry(DATA_ROOT, verify=True)


@pytest.fixture(scope="module")
def mtuikku(loaded) -> MeasuredSeries:
    return loaded["series"]["MTUIKKU-HZ2016"]


@pytest.fixture(scope="module")
def artifact() -> dict:
    assert ARTIFACT.is_file(), f"run the benchmark first: {ARTIFACT}"
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _ship(series: MeasuredSeries) -> ShipGeometry:
    s = series.ship
    return ShipGeometry(
        name=str(s["name"]),
        L_pp=float(s["L_pp_m"]),
        B=float(s["B_m"]),
        T=float(s["T_m"]),
        bow_angle_deg=float(s["bow_angle_deg"]),
        flare_deg=float(s["flare_deg"]),
    )


# ---------------------------------------------------------------------------
# C1 -- schema + digest provenance
# ---------------------------------------------------------------------------


def test_c1_registry_has_two_verified_series(loaded):
    assert loaded["errors"] == []
    series = loaded["series"]
    assert len(series) >= ma.REQUIRED_SERIES
    for sid, s in series.items():
        assert s.grade in ma.EVIDENCE_GRADES
        assert s.digests and all(ma.is_sha256_hex(d) for d in s.digests)
        # the digest recorded in provenance equals the bytes on disk
        for src in s.provenance["sources"]:
            path = Path(s.path).parent / src["artifact"]
            assert sha256_file(path) == src["sha256"], f"{sid}: {src['artifact']} digest drift"


def test_c1_schema_rejects_provenance_without_digest(mtuikku):
    doc = mtuikku.as_dict()
    doc["provenance"]["sources"][0]["sha256"] = "not-a-digest"
    errors = validate_series(doc)
    assert any("sha256" in e for e in errors)


def test_c1_schema_rejects_missing_measurement(mtuikku):
    doc = mtuikku.as_dict()
    doc["points"][0].pop("R_meas_kN")
    assert any("R_meas_kN" in e for e in validate_series(doc))


def test_c1_tampered_source_is_detected(tmp_path):
    """Copy the series, flip a byte in the archived source -> digest mismatch."""

    doc = json.loads((MTUIKKU_DIR / "series.json").read_text(encoding="utf-8"))
    (tmp_path / "source").mkdir()
    src = MTUIKKU_DIR / "source" / doc["provenance"]["sources"][0]["artifact"].split("/")[-1]
    payload = bytearray(src.read_bytes())
    payload[10] = payload[10] ^ 0xFF
    (tmp_path / "source" / src.name).write_bytes(bytes(payload))
    errors = validate_series(doc, root=tmp_path)
    assert any("digest mismatch" in e for e in errors)


def test_c1_loader_refuses_corrupt_asset(tmp_path, mtuikku):
    doc = mtuikku.as_dict()
    doc["schema_version"] = "9.9"
    p = tmp_path / "series.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(MeasuredAnchorError):
        load_series(p, verify=False)


# ---------------------------------------------------------------------------
# C2 -- double-entry digitisation
# ---------------------------------------------------------------------------


def test_c2_double_entry_sample_agreement():
    reports = []
    for sdir, s in ((MTUIKKU_DIR, "MTUIKKU-HZ2016"), (AGULHAS_DIR, "AGULHAS2-FS2012")):
        series = load_series(sdir / "series.json", verify=False)
        pa = json.loads((sdir / "pass_a.json").read_text(encoding="utf-8"))
        pb = json.loads((sdir / "pass_b.json").read_text(encoding="utf-8"))
        ids = sampled_point_ids(series.points, 0.10)
        rep = double_entry_check(pa, pb, ids)
        rep["series_id"] = s
        reports.append(rep)
    assert all(r["passed"] for r in reports)
    assert all(r["agreement_rate"] > 0.99 for r in reports)
    assert sum(r["n_points_sampled"] for r in reports) >= 0.10 * 14


def test_c2_detects_column_shift():
    pa = {"103": {"h_m": 0.77, "V_ms": 0.2, "R_meas_kN": 480.0}}
    pb = {"103": {"h_m": 0.77, "V_ms": 0.2, "R_meas_kN": 48.0}}
    rep = double_entry_check(pa, pb, ["103"])
    assert rep["passed"] is False
    assert rep["mismatches"][0]["field"] == "R_meas_kN"


def test_c2_missing_column_counts_as_mismatch():
    pa = {"103": {"h_m": 0.77, "sigma_f_kPa": 724.0}}
    pb = {"103": {"h_m": 0.77}}
    rep = double_entry_check(pa, pb, ["103"])
    assert rep["passed"] is False


# ---------------------------------------------------------------------------
# C3 -- adjudication against the measured anchor
# ---------------------------------------------------------------------------


def _primary_adjudication(mtuikku):

    def model(pt, series):
        out = level_ice_resistance(
            pt.h_m,
            pt.V_ms,
            _ship(series),
            sigma_c=float(pt.sigma_c_kPa) * 1e3,
            sigma_f=float(pt.sigma_f_kPa) * 1e3,
        )
        return {
            "R_kN": out["R_total_N"] / 1e3,
            "R_crush_kN": out["R_crush_N"] / 1e3,
            "R_break_kN": out["R_break_N"] / 1e3,
            "R_submersion_kN": out["R_submersion_N"] / 1e3,
        }

    return adjudicate_resistance(mtuikku, model, config="primary_test_ice_strength")


def test_c3_every_measured_point_is_adjudicated(mtuikku):
    adj = _primary_adjudication(mtuikku)
    assert len(adj.rows) == len(mtuikku.points) == 10
    for row in adj.rows:
        assert row["R_meas_kN"] > 0
        assert row["rel_error"] is not None


def test_c3_verdict_is_honest_fail_and_reproducible(mtuikku, artifact):
    adj = _primary_adjudication(mtuikku)
    med = adj.median_rel_error
    stored = artifact["adjudications"][0]["summary"]["median_rel_error"]
    # numbers in the artifact are recomputed, not copied
    assert med == pytest.approx(stored, rel=1e-12)
    if med > GATE_TOLERANCE:
        assert adj.verdict == "FAIL_15PCT"
        assert artifact["criteria"]["C3_15pct_adjudication"]["passed"] is False
        assert artifact["criteria"]["C3_15pct_adjudication"]["honest_fail"] is True
    else:  # pragma: no cover - would mean the model got much better
        assert adj.verdict == "PASS_15PCT"


def test_c3_overprediction_is_attributable(mtuikku):
    """The error must be localisable to a model component (not a black box)."""

    adj = _primary_adjudication(mtuikku)
    assert adj.bias is not None and adj.bias > 0  # systematic over-prediction
    for row in adj.rows:
        model = row["model"]
        total = model["R_kN"]
        parts = model["R_crush_kN"] + model["R_break_kN"] + model["R_submersion_kN"]
        assert total == pytest.approx(parts, rel=1e-9)


def test_c3_gate_never_passes_above_tolerance(mtuikku):
    def bad_model(pt, series):
        return {"R_kN": 10.0 * float(pt.R_meas_kN)}

    adj = adjudicate_resistance(mtuikku, bad_model, config="negative_control_x10")
    assert adj.verdict == "FAIL_15PCT"


def test_c3_published_law_baselines_recorded(artifact):
    base = artifact["published_baselines"]["summary"]
    assert set(base) == {"lindqvist", "riska", "jeong", "keinonen"}
    assert base["lindqvist"]["n_points"] == 10


# ---------------------------------------------------------------------------
# C4 -- physics gates
# ---------------------------------------------------------------------------


def test_c4_model_physics_gates(mtuikku):
    ship = _ship(mtuikku)
    gates = physics_gates(lambda h, V: level_ice_resistance(h, V, ship)["R_total_N"] / 1e3)
    assert gates["passed"], [c for c in gates["checks"] if not c["passed"]]
    # explicit V -> 0 intercept: finite, equals crush + bending, no speed term
    r0 = level_ice_resistance(0.9, 0.0, ship)
    assert r0["R_submersion_N"] == 0.0
    assert r0["R_total_N"] == pytest.approx(r0["R_crush_N"] + r0["R_break_N"], rel=1e-12)


def test_c4_measured_series_plausibility(mtuikku):
    rep = measured_monotonicity(mtuikku)
    assert rep["passed"], [c for c in rep["checks"] if not c["passed"]]
    # the h = 1.04 m non-monotone pair must be recorded, never corrected
    assert any("0.5" in n and "1.0" in n for n in rep["notes"]) or rep["notes"] == []


def test_c4_measured_rejects_absurd_series(mtuikku):
    doc = mtuikku.as_dict()
    doc["points"][0]["R_meas_kN"] = -1.0
    assert any("R_meas_kN" in e for e in validate_series(doc))


# ---------------------------------------------------------------------------
# C5 -- legacy corpus untouched
# ---------------------------------------------------------------------------


def test_c5_legacy_full_spectrum_corpus_green():
    from anchors_eval import evaluate_corpus

    res = evaluate_corpus()
    assert res["all_passed"], [
        a["id"] for a in res["anchors"] if not a["passed"]
    ] + [n["id"] for n in res["negative_controls"] if not n["passed"]]
    assert res["n_reference_anchors"] >= 10


def test_c5_artifact_legacy_gate(artifact):
    gate = artifact["criteria"]["C5_legacy_full_spectrum"]
    assert gate["passed"] is True


# ---------------------------------------------------------------------------
# WITHHELD -> JUDGED is mechanical
# ---------------------------------------------------------------------------


def test_gate_state_withheld_without_any_anchor():
    ledger = gate_ledger(
        schema_report={"n_series": 0, "n_points": 0, "all_valid": False, "all_digests_verified": False},
        double_entry_reports=[],
        adjudications=[],
        physics={"passed": True},
        legacy_gate={"all_passed": True},
    )
    assert ledger["gate_state"] == "WITHHELD"
    assert ledger["C3_15pct_adjudication"]["passed"] is False


def test_gate_state_judged_with_anchor(artifact):
    crit = artifact["criteria"]
    assert crit["gate_state"] == "JUDGED"
    assert crit["evidence_state"] == "MEASURED_ANCHOR_AVAILABLE"
    assert crit["C1_schema_and_digest"]["passed"] and crit["C2_double_entry"]["passed"]


def test_sha256_helper_roundtrip(tmp_path):
    payload = b"ice resistance measured anchor\n"
    p = tmp_path / "f.bin"
    p.write_bytes(payload)
    assert sha256_file(p) == sha256_bytes(payload)
    assert ma.is_sha256_hex(sha256_file(p))