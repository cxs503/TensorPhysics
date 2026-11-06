"""Acceptance tests for the sha256-anchored ice-navigation reference corpus.

What is asserted here
=====================
(a) coverage      -- at least 12 reference assets covering the five source families;
(b) schema        -- every asset carries ``source_citation``, ``values``,
                     ``tolerance`` and ``applicable_regime``;
(c) anchoring     -- each asset file reproduces its sha256 from ``MANIFEST.json``
                     and each numeric payload reproduces its ``value_sha256``;
(d) tolerances    -- present, well formed (mode/value/band consistency) and usable
                     by the ``check``/``certify`` consumer API;
(e) negative cases-- an entry without (or with placeholder) citation is *rejected*,
                     a tampered file or a relabelled number is *detected*;
(f) reproducibility-- rebuilding the corpus from ``build_corpus.py`` gives
                     byte-identical files (identical anchors);
(g) cross-check   -- the corpus numbers agree with this repository's own kernels
                     (hullgirder global pressure, ISO 19906 local pressure,
                     ice_anchor Lindqvist transcription).
"""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

from tensorphysics.ice_nav_refs import (
    EVIDENCE_TIERS,
    REQUIRED_FIELDS,
    ROLES,
    CorpusError,
    CorpusIntegrityError,
    MissingCitationError,
    MissingFieldError,
    Tolerance,
    ToleranceError,
    asset_from_dict,
    canonical_json,
    describe_corpus,
    load_corpus,
    sha256_digest,
    value_digest,
    verify_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = REPO_ROOT / "benchmarks" / "ice_nav_refs"
BUILD_SCRIPT = CORPUS_ROOT / "build_corpus.py"

MIN_ASSETS = 12
EXPECTED_FAMILIES = {
    "lindqvist_1989",
    "keinonen_brash",
    "riska_kujala_pressure_area",
    "iacs_ur_i_pc6",
    "polar_code_i_a",
}


@pytest.fixture(scope="module")
def corpus():
    return load_corpus(CORPUS_ROOT)


@pytest.fixture(scope="module")
def builder():
    spec = importlib.util.spec_from_file_location("ice_nav_refs_build", BUILD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- (a) coverage
def test_coverage_minimum_assets(corpus):
    assert len(corpus) >= MIN_ASSETS, f"only {len(corpus)} reference assets"
    assert corpus.manifest["asset_count"] == len(corpus)


def test_coverage_required_families(corpus):
    assert set(corpus.families) == EXPECTED_FAMILIES
    summary = describe_corpus(corpus)
    assert summary["asset_count"] >= MIN_ASSETS
    assert summary["family_count"] == len(EXPECTED_FAMILIES)
    # every family must carry at least two independent data sets
    assert all(count >= 2 for count in summary["assets_by_family"].values()), summary["assets_by_family"]


def test_asset_ids_are_unique_and_content_addressed(corpus):
    ids = corpus.asset_ids
    assert len(ids) == len(set(ids))
    digests = [a.value_sha256 for a in corpus]
    assert len(set(digests)) == len(digests)
    for digest in digests:
        assert len(digest) == 64
        int(digest, 16)


# ------------------------------------------------------------------ (b) schema
def test_every_asset_has_the_four_mandatory_fields(corpus):
    for asset in corpus:
        payload = asset.to_dict()
        for field_name in REQUIRED_FIELDS:
            assert field_name in payload, f"{asset.asset_id} lacks {field_name}"
            assert payload[field_name], f"{asset.asset_id} has an empty {field_name}"
        assert isinstance(payload["values"], dict) and payload["values"]
        assert isinstance(payload["applicable_regime"], dict) and payload["applicable_regime"]
        assert payload["source_citation"].strip()
        assert len(payload["source_citation"]) >= 12
        assert any(ch.isdigit() for ch in payload["source_citation"])


def test_every_asset_declares_evidence_tier_and_role(corpus):
    for asset in corpus:
        assert asset.evidence_tier in EVIDENCE_TIERS
        assert asset.role in ROLES
        assert isinstance(asset.primary_source_verified, bool)
        # audit discipline: an unverified value must say *why*
        if not asset.primary_source_verified:
            assert len(asset.verification_note) >= 40, asset.asset_id


def test_applicable_regime_shape(corpus):
    for asset in corpus:
        regime = asset.applicable_regime
        assert isinstance(regime["ice_regime"], str) and regime["ice_regime"]
        span = regime["ice_thickness_m"]
        assert len(span) == 2 and float(span[0]) <= float(span[1]), asset.asset_id
        assert float(span[0]) >= 0.0


def test_units_cover_every_numeric_value(corpus):
    for asset in corpus:
        for key, value in asset.values.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                assert key in asset.units, f"{asset.asset_id}: no unit declared for {key}"


# ------------------------------------------------------------- (c) sha256 anchor
def test_manifest_sha256_matches_files_on_disk():
    report = verify_manifest(CORPUS_ROOT)
    assert report["ok"], report["errors"]
    assert len(report["files"]) == len(EXPECTED_FAMILIES)
    assert report["aggregate_sha256"] == json.loads(
        (CORPUS_ROOT / "MANIFEST.json").read_text(encoding="utf-8")
    )["aggregate_sha256"]


def test_each_asset_file_is_pinned_in_the_manifest():
    manifest = json.loads((CORPUS_ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    listed_ids = {asset_id for entry in manifest["files"].values() for asset_id in entry["asset_ids"]}
    on_disk = {
        payload["asset_id"]
        for path in sorted((CORPUS_ROOT / "assets").glob("*.json"))
        for payload in json.loads(path.read_text(encoding="utf-8"))["assets"]
    }
    assert listed_ids == on_disk
    assert len(listed_ids) >= MIN_ASSETS


def test_value_sha256_reproduces_from_recorded_numbers(corpus):
    for asset in corpus:
        expected = value_digest(asset.asset_id, asset.values, asset.tolerance)
        assert asset.value_sha256 == expected, asset.asset_id
        # prose edits must not change the numeric anchor
        assert value_digest(asset.asset_id, asset.values, asset.tolerance) == asset.value_sha256


# ---------------------------------------------------------------- (d) tolerance
def test_tolerance_present_and_well_formed(corpus):
    for asset in corpus:
        tol = asset.tolerance
        assert isinstance(tol, Tolerance)
        assert tol.mode in {"relative", "absolute", "band", "exact"}
        assert tol.value is not None
        assert tol.describe()
        if tol.mode == "band":
            assert tol.lower is not None and tol.upper is not None
            assert tol.lower < tol.upper
            assert tol.lower <= tol.upper
        if tol.mode == "relative":
            assert 0.0 < tol.value <= 0.5, f"{asset.asset_id}: implausible relative tolerance"
        if tol.mode == "exact":
            assert tol.value == 0.0


def test_tolerance_is_enforced_by_the_consumer_api(corpus):
    for asset in corpus:
        nominal = asset.check({k: v for k, v in asset.values.items()})
        assert nominal["pass"], (asset.asset_id, nominal["values"])
        assert nominal["worst_rel_error"] == pytest.approx(0.0, abs=1e-12)


def test_out_of_tolerance_value_is_rejected(corpus):
    asset = corpus.get("RA_LOCAL_PRESSURE_TABLE")
    bad = {"p_A1_MPa": asset.values["p_A1_MPa"] * 1.5}
    verdict = asset.check(bad, key="p_A1_MPa")
    assert not verdict["pass"]
    assert verdict["worst_rel_error"] == pytest.approx(0.5, rel=1e-9)
    with pytest.raises(CorpusError):
        asset.check({"not_a_key": 1.0})


def test_certify_batch_reports_failures(corpus):
    payload = {a.asset_id: dict(a.values) for a in corpus}
    good = corpus.certify(payload)
    assert good["pass"] and good["asset_count"] == len(corpus)
    payload["LIN_C_CRUSH_PHI30"]["C_c"] *= 1.01
    bad = corpus.certify(payload)
    assert not bad["pass"]
    assert bad["failed"] == ["LIN_C_CRUSH_PHI30"]


# ------------------------------------------------------------ (e) negative cases
def _valid_payload(corpus):
    return copy.deepcopy(corpus.get("LIN_C_CRUSH_PHI30").to_dict())


def test_positive_control_valid_payload_is_accepted(corpus):
    asset = asset_from_dict(_valid_payload(corpus))
    assert asset.asset_id == "LIN_C_CRUSH_PHI30"


def test_entry_without_citation_is_rejected(corpus):
    payload = _valid_payload(corpus)
    del payload["source_citation"]
    with pytest.raises(MissingFieldError):
        asset_from_dict(payload)


@pytest.mark.parametrize(
    "placeholder",
    ["", "   ", "n/a", "TBD", "todo", "unknown", "-", None, "literature", "K. Neumann"],
)
def test_placeholder_citation_is_rejected(corpus, placeholder):
    payload = _valid_payload(corpus)
    payload["source_citation"] = placeholder
    with pytest.raises(MissingCitationError):
        asset_from_dict(payload)


def test_non_string_citation_is_rejected(corpus):
    payload = _valid_payload(corpus)
    payload["source_citation"] = 1989
    with pytest.raises(MissingCitationError):
        asset_from_dict(payload)


def test_missing_values_tolerance_or_regime_is_rejected(corpus):
    for field_name in ("values", "tolerance", "applicable_regime"):
        payload = _valid_payload(corpus)
        del payload[field_name]
        with pytest.raises(MissingFieldError):
            asset_from_dict(payload)


def test_malformed_tolerance_is_rejected(corpus):
    bad_values = [
        {"mode": "relative"},                      # no value
        {"value": 0.1},                            # no mode
        {"mode": "fuzzy", "value": 0.1},           # unknown mode
        {"mode": "relative", "value": -0.1},       # negative
        {"mode": "relative", "value": 0.0},        # zero
        {"mode": "band", "value": 1.0, "lower": 2.0, "upper": 1.0},  # inverted band
        {"mode": "band", "value": 1.0},            # band without bounds
    ]
    for tol in bad_values:
        payload = _valid_payload(corpus)
        payload["tolerance"] = tol
        with pytest.raises(ToleranceError):
            asset_from_dict(payload)


def test_relabelled_number_is_detected(corpus):
    payload = _valid_payload(corpus)
    payload["values"]["C_c"] = payload["values"]["C_c"] * 1.0001  # keeps the old digest
    with pytest.raises(CorpusIntegrityError):
        asset_from_dict(payload)


def test_tampered_asset_file_is_detected(tmp_path):
    target = tmp_path / "corpus"
    shutil.copytree(CORPUS_ROOT, target)
    path = target / "assets" / "riska_kujala_pressure_area.json"
    payload = path.read_text(encoding="utf-8")
    assert '"3.0"' in payload or "3.0" in payload
    path.write_text(payload.replace("3.0", "3.1", 1), encoding="utf-8")
    with pytest.raises(CorpusIntegrityError):
        load_corpus(target)
    report = verify_manifest(target)
    assert not report["ok"]
    assert any("sha256 mismatch" in err for err in report["errors"])


def test_missing_manifest_is_rejected(tmp_path):
    (tmp_path / "assets").mkdir()
    with pytest.raises(CorpusError):
        load_corpus(tmp_path)


def test_duplicate_asset_id_across_files_is_rejected(tmp_path):
    source = json.loads((CORPUS_ROOT / "assets" / "lindqvist_1989.json").read_text(encoding="utf-8"))
    one = source["assets"][0]
    files = {}
    for name in ("a.json", "b.json"):
        doc = {"schema": source["schema"], "family": "dup", "assets": [one]}
        data = canonical_json(doc)
        (tmp_path / name).write_bytes(data)
        files[name] = {"sha256": sha256_digest(data), "bytes": len(data), "asset_ids": [one["asset_id"]]}
    aggregate = sha256_digest("\n".join(f"{n}:{files[n]['sha256']}" for n in sorted(files)).encode())
    (tmp_path / "MANIFEST.json").write_bytes(
        canonical_json(
            {
                "schema": source["schema"],
                "asset_count": 2,
                "files": files,
                "aggregate_sha256": aggregate,
            }
        )
    )
    with pytest.raises(CorpusError, match="duplicate asset_id"):
        load_corpus(tmp_path)


def test_manifest_asset_count_mismatch_is_rejected(tmp_path):
    doc = json.loads((CORPUS_ROOT / "assets" / "keinonen_brash.json").read_text(encoding="utf-8"))
    data = canonical_json(doc)
    (tmp_path / "k.json").write_bytes(data)
    files = {"k.json": {"sha256": sha256_digest(data), "bytes": len(data), "asset_ids": []}}
    aggregate = sha256_digest(f"k.json:{files['k.json']['sha256']}".encode())
    (tmp_path / "MANIFEST.json").write_bytes(
        canonical_json({"schema": doc["schema"], "asset_count": 99, "files": files,
                        "aggregate_sha256": aggregate})
    )
    with pytest.raises(CorpusError, match="declares"):
        load_corpus(tmp_path)


# --------------------------------------------------------- (f) reproducibility
def test_rebuild_is_byte_identical(tmp_path, builder):
    manifest = builder.build(tmp_path)
    reference = json.loads((CORPUS_ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["asset_count"] == reference["asset_count"]
    assert manifest["aggregate_sha256"] == reference["aggregate_sha256"]
    for rel, entry in reference["files"].items():
        assert manifest["files"][rel]["sha256"] == entry["sha256"], rel
        assert (tmp_path / rel).read_bytes() == (CORPUS_ROOT / rel).read_bytes(), rel


def test_rebuild_check_mode_succeeds(builder):
    assert builder.main(["--root", str(CORPUS_ROOT), "--check"]) == 0


# ------------------------------------------------------------- (g) cross-check
def test_global_pressure_agrees_with_hullgirder_kernel(corpus):
    from tensorphysics.hullgirder.ice_load import IceCondition, global_ice_pressure

    asset = corpus.get("RA_GLOBAL_PRESSURE_TABLE")
    for key, nominal in asset.values.items():
        h = float(key[len("pG_h"): -len("_MPa")])
        computed = global_ice_pressure(IceCondition(thickness=h, concentration=1.0, speed=2.0)) / 1e6
        assert computed == pytest.approx(nominal, rel=1e-12), (h, computed, nominal)


def test_local_pressure_agrees_with_iso19906_kernel(corpus):
    from tensorphysics.ice_plasticity.iso19906 import ISO19906Pressure

    asset = corpus.get("RA_LOCAL_PRESSURE_TABLE")
    model = ISO19906Pressure(
        p0=float(asset.reference_condition["p0_MPa"]),
        ex=0.45,
    )
    for key, nominal in asset.values.items():
        area = float(key[len("p_A"): -len("_MPa")])
        computed = float(model.pressure(area))
        assert computed == pytest.approx(nominal, rel=1e-12), (area, computed, nominal)


def test_lindqvist_transcription_agrees_with_ice_anchor_module(corpus, builder):
    pytest.importorskip("ice_anchor.anchors")
    from ice_anchor.anchors import lindqvist_1989

    reference = corpus.get("LIN_RESISTANCE_COMPONENTS")
    result = lindqvist_1989(h=1.0, B=20.0, phi=30.0, V=2.0, L=150.0)
    assert result["C_c"] == pytest.approx(reference.values["C_c"], rel=1e-12)
    assert result["C_f"] == pytest.approx(reference.values["C_f_m"], rel=1e-12)
    assert result["C_s"] == pytest.approx(reference.values["C_s"], rel=1e-12)
    assert result["R_c_N"] == pytest.approx(reference.values["R_c_N"], rel=1e-12)
    assert result["R_f_N"] == pytest.approx(reference.values["R_f_N"], rel=1e-12)
    assert result["R_s_N"] == pytest.approx(reference.values["R_s_N"], rel=1e-12)
    assert result["R_total_kN"] == pytest.approx(reference.values["R_total_kN"], rel=1e-12)


def test_corpus_condition_matches_builder_constants(builder, corpus):
    assert builder.SHIP["B_m"] == 20.0
    reference = corpus.get("LIN_RESISTANCE_COMPONENTS")
    assert reference.reference_condition["h_m"] == builder.ICE["h_m"]


# --------------------------------------------------------------- value sanity
def test_pressure_assets_are_monotone_decreasing_in_area(corpus):
    for asset_id in ("RA_LOCAL_PRESSURE_TABLE", "IACS_PC6_PANEL_PRESSURE_TABLE"):
        asset = corpus.get(asset_id)
        items = sorted(
            (float(k.split("_A")[1].split("_")[0]), v) for k, v in asset.values.items()
        )
        pressures = [p for _, p in items]
        assert pressures == sorted(pressures, reverse=True), asset_id
        assert all(p > 0.0 for p in pressures)


def test_all_numeric_values_are_physical(corpus):
    for asset in corpus:
        for key, value in asset.values.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                if key.endswith("_C") and "pst" in key:
                    assert -60.0 <= value <= 0.0, (asset.asset_id, key, value)
                if key.endswith("MPa"):
                    assert 0.0 < value < 50.0, (asset.asset_id, key, value)
                if key.endswith("_N") or key.endswith("_kN"):
                    assert value > 0.0, (asset.asset_id, key, value)