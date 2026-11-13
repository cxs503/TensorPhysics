"""Tests for the ice pressure-area parameter library, convention algebra and calibration.

The tests assert *invariants* (theorem-level statements about the area-convention
algebra, analytic anchors for the pooling, identification behaviour of the
meta-regression) rather than merely executing the code, and they pin the two
in-repository contradictions that GAP-PP-2 exists to expose.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorphysics.ice_pressure_area import (  # noqa: E402
    AREA_CONVENTIONS,
    DEFAULT_DAMAGE_RATIO_SWEEP,
    REFERENCE_CONVENTION,
    ConventionError,
    LibraryError,
    LibraryIntegrityError,
    PressureAreaLaw,
    band_to_sigma,
    demand_bandwidth_study,
    dl_pool,
    estimates_from_library,
    fit_power_law,
    from_reference,
    load_library,
    meta_regression,
    ratio_factor,
    required_area_ratio,
    seal,
    to_reference,
)
from tensorphysics.ice_pressure_area.calibrate import Estimate  # noqa: E402


# ---------------------------------------------------------------------------
# library
# ---------------------------------------------------------------------------



def _reseal_raw(directory: Path) -> None:
    """Rebuild MANIFEST.json from the raw JSON *without* semantic validation.

    Needed to isolate load-time rules (band consistency, citation hygiene) from
    the build-time validation performed by :func:`seal`.
    """
    from tensorphysics.ice_pressure_area.library import sha256_digest, value_digest

    path = directory / "pressure_area_sources.json"
    raw = json.loads(path.read_text())
    entry_sha = {e["asset_id"]: value_digest(e) for e in raw["entries"]}
    manifest = {
        "schema": raw["schema"],
        "file": path.name,
        "file_sha256": sha256_digest(path.read_bytes()),
        "entry_count": len(raw["entries"]),
        "entry_sha256": entry_sha,
        "aggregate_rule": "sha256 over the sorted 'asset_id:sha256' lines",
    }
    manifest["aggregate_sha256"] = sha256_digest(
        "\n".join(f"{k}:{v}" for k, v in sorted(entry_sha.items())).encode()
    )
    (directory / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_library_loads_and_verifies_manifest():
    library = load_library()
    assert len(library) == 7
    assert set(library.asset_ids) == {
        "SANDERSON_1988_GLOBAL_PRESSURE_AREA",
        "MASTERSON_FREDKING_1993_LOCAL_PRESSURE_AREA",
        "ISO_19906_2010_A8_FIRST_YEAR",
        "IACS_UR_I2_PC6_PATCH_PRESSURE",
        "JOHNSTON_ET_AL_1998_MSI_INDENTATION",
        "TENSORPHYSICS_LOCAL_PRESSURE_IACS_EX015",
        "TENSORPHYSICS_FREDERKSON_REF_TABLE_EX015",
    }
    assert len(library.accuracy_references()) == 5
    assert len(library.anchors()) == 2


def test_every_entry_declares_a_known_area_convention_and_units():
    library = load_library()
    for entry in library:
        assert entry.area_definition in AREA_CONVENTIONS
        assert entry.units["ex"]  # units recorded per value
        assert entry.conditions["area_range_m2"]


def test_integrity_detects_byte_drift(tmp_path):
    src = REPO / "data" / "ice_pressure_area"
    dst = tmp_path / "ice_pressure_area"
    shutil.copytree(src, dst)
    path = dst / "pressure_area_sources.json"
    path.write_text(path.read_text().replace('"ex": 0.7', '"ex": 0.71'), encoding="utf-8")
    with pytest.raises(LibraryIntegrityError):
        load_library(dst)


def test_entry_with_ex_outside_its_band_is_rejected(tmp_path):
    src = REPO / "data" / "ice_pressure_area"
    dst = tmp_path / "ice_pressure_area"
    shutil.copytree(src, dst)
    raw = json.loads((dst / "pressure_area_sources.json").read_text())
    raw["entries"][0]["reported"]["ex_band"] = [0.4, 0.5]
    (dst / "pressure_area_sources.json").write_text(json.dumps(raw))
    _reseal_raw(dst)  # reseal without validation: only the semantic rule can fail
    with pytest.raises(LibraryError, match="outside its own band"):
        load_library(dst)


def test_placeholder_citation_is_rejected(tmp_path):
    src = REPO / "data" / "ice_pressure_area"
    dst = tmp_path / "ice_pressure_area"
    shutil.copytree(src, dst)
    raw = json.loads((dst / "pressure_area_sources.json").read_text())
    raw["entries"][0]["citation"] = "n/a"
    (dst / "pressure_area_sources.json").write_text(json.dumps(raw))
    _reseal_raw(dst)
    with pytest.raises(LibraryError, match="placeholder citation"):
        load_library(dst)


def test_known_family_conflicts_are_detected():
    """The two in-repo IACS transcriptions disagree with the corpus asset."""
    library = load_library()
    conflicts = library.detect_conflicts()
    pairs = {tuple(c["asset_ids"]) for c in conflicts}
    assert (
        "IACS_UR_I2_PC6_PATCH_PRESSURE",
        "TENSORPHYSICS_LOCAL_PRESSURE_IACS_EX015",
    ) in pairs
    assert (
        "IACS_UR_I2_PC6_PATCH_PRESSURE",
        "TENSORPHYSICS_FREDERKSON_REF_TABLE_EX015",
    ) in pairs
    blocking = library.blocking_conflicts()
    assert len(blocking) == 2
    for conflict in blocking:
        assert conflict["family"] == "iacs_ur_i"
        assert conflict["z"] > 4.0


# ---------------------------------------------------------------------------
# convention algebra (theorem-level)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("convention", ["line_load_area", "panel_nominal_area", REFERENCE_CONVENTION])
@pytest.mark.parametrize("ex", [0.15, 0.3, 0.45, 0.7])
def test_constant_area_ratio_leaves_ex_invariant(convention, ex):
    """Ratio-invariance: a constant area rescaling must not move the exponent."""
    assert to_reference(ex, convention, 1.0) == pytest.approx(ex, abs=1e-15)
    assert from_reference(ex, convention, 37.5) == pytest.approx(ex, abs=1e-15)


def test_damage_zone_ratio_is_below_one_and_monotone():
    for delta in DEFAULT_DAMAGE_RATIO_SWEEP[1:]:
        r = ratio_factor("damaged_zone_area", 1.0, lambda_perimeter=4.25, delta_over_sqrtA=delta)
        assert 0.0 < r < 1.0
    rs = [
        ratio_factor("damaged_zone_area", 1.0, lambda_perimeter=4.25, delta_over_sqrtA=d)
        for d in DEFAULT_DAMAGE_RATIO_SWEEP
    ]
    assert rs[0] == pytest.approx(1.0)  # zero process zone -> identity
    assert all(a > b for a, b in zip(rs, rs[1:]))  # more damage -> stronger shift


def test_transfer_round_trip():
    for ex in (0.1, 0.45, 0.9):
        for delta in (0.05, 0.2):
            mapped = to_reference(
                ex, "damaged_zone_area", 1.0, lambda_perimeter=4.25, delta_over_sqrtA=delta
            )
            back = from_reference(
                mapped, "damaged_zone_area", 1.0, lambda_perimeter=4.25, delta_over_sqrtA=delta
            )
            assert back == pytest.approx(ex, rel=1e-12)


def test_transfer_closed_form_matches_finite_difference():
    """Law (T) is the log-derivative of the damage-zone area map."""
    lam, delta, A = 4.25, 0.1, 1.0

    def lnAd(lnA: float) -> float:
        area = math.exp(lnA)
        return math.log(area + lam * delta * math.sqrt(area))

    for h, tol in ((1e-3, 1e-5), (1e-5, 1e-8)):
        numeric = (lnAd(math.log(A) + h) - lnAd(math.log(A) - h)) / (2 * h)
        analytic = ratio_factor(
            "damaged_zone_area", A, lambda_perimeter=lam, delta_over_sqrtA=delta
        )
        assert numeric == pytest.approx(analytic, abs=tol)


def test_transfer_shift_is_bounded_by_process_zone_width():
    """|ex_nominal - ex_damaged| <= (1 - ex_damaged) * t/2 with t = lambda*delta/sqrt(A)."""
    lam, delta, ex = 4.25, 0.1, 0.45
    t = lam * delta
    mapped = to_reference(ex, "damaged_zone_area", 1.0, lambda_perimeter=lam, delta_over_sqrtA=delta)
    assert abs(mapped - ex) <= (1.0 - ex) * t / 2.0 + 1e-12


def test_unknown_convention_is_rejected():
    with pytest.raises(ConventionError):
        to_reference(0.45, "banana_area", 1.0)
    with pytest.raises(ConventionError):
        ratio_factor("damaged_zone_area", 0.0)


# ---------------------------------------------------------------------------
# law consumer
# ---------------------------------------------------------------------------


def test_law_requires_a_declared_convention_and_consistent_bands():
    law = PressureAreaLaw(p0_MPa=2.75, ex=0.45, ex_band=(0.4, 0.5))
    assert law.pressure_MPa(1.0) == pytest.approx(2.75)
    assert law.demand_bandwidth(2.3593) == pytest.approx(2.3593 ** 0.45)
    with pytest.raises(ConventionError):
        PressureAreaLaw(p0_MPa=2.75, ex=0.45, convention="guess")
    with pytest.raises(ConventionError):
        PressureAreaLaw(p0_MPa=2.75, ex=0.45, ex_band=(0.5, 0.6))
    with pytest.raises(ConventionError):
        law.pressure_MPa(0.0)


def test_library_can_assemble_a_stratified_law():
    library = load_library()
    law = library.law("design_rule")
    assert law.convention == REFERENCE_CONVENTION
    assert law.ex == pytest.approx(0.45)
    assert law.ex_band[0] <= law.ex <= law.ex_band[1]
    lo, hi = law.demand_bandwidth_band(2.3593)
    assert lo < hi


# ---------------------------------------------------------------------------
# fit + pooling analytic anchors
# ---------------------------------------------------------------------------


def test_fit_power_law_recovers_a_known_exponent_exactly():
    areas = [0.5, 1.0, 2.0, 4.0, 8.0]
    pressures = [2.5 * a ** -0.45 for a in areas]
    fit = fit_power_law(areas, pressures)
    assert fit["ex"] == pytest.approx(0.45, rel=1e-10)
    assert fit["p0_MPa"] == pytest.approx(2.5, rel=1e-10)
    assert fit["r2_logspace"] == pytest.approx(1.0, abs=1e-10)
    with pytest.raises(ValueError):
        fit_power_law([1.0, 0.0], [1.0, 1.0])
    with pytest.raises(ValueError):
        fit_power_law([1.0], [1.0])


def test_fit_recovers_the_sha256_anchored_corpus_table():
    library = load_library()
    entry = library.get("IACS_UR_I2_PC6_PATCH_PRESSURE")
    fit = fit_power_law(entry.data_points["area_m2"], entry.data_points["pressure_MPa"])
    assert fit["ex"] == pytest.approx(entry.ex, rel=1e-9)
    assert fit["p0_MPa"] == pytest.approx(entry.p_ref_MPa, rel=1e-9)


def test_band_to_sigma_matches_the_normal_quantile():
    assert band_to_sigma(0.4, 0.5) == pytest.approx(0.1 / (2 * 1.6448536269514722))
    assert band_to_sigma(0.0, 1.0, confidence=0.95) == pytest.approx(0.5 / 1.959963984540054, rel=1e-9)
    with pytest.raises(ValueError):
        band_to_sigma(0.5, 0.4)


def test_dl_pool_with_equal_variances_reduces_to_the_arithmetic_mean():
    """Analytic anchor: equal variances + no between-source scatter -> mean, tau2 = 0."""
    ests = [
        Estimate(f"e{i}", ex, 0.05, (ex - 0.082, ex + 0.082), {})
        for i, ex in enumerate([0.499, 0.5, 0.501])
    ]
    pooled = dl_pool(ests)
    assert pooled["ex_fixed"] == pytest.approx(0.5)
    assert pooled["ex_pooled"] == pytest.approx(0.5)
    assert pooled["tau2"] == pytest.approx(0.0)  # Q = 0.0008 < df = 2
    assert pooled["heterogeneity"] == "homogeneous"
    assert pooled["ci90"][1] - pooled["ci90"][0] == pytest.approx(
        2 * 1.6448536269514722 * 0.05 / math.sqrt(3)
    )


def test_dl_pool_detects_heterogeneity_and_widens_the_interval():
    homogeneous = dl_pool(
        [Estimate(f"e{i}", 0.45 + 1e-4 * i, 0.02, (0.42, 0.48), {}) for i in range(4)]
    )
    scattered = dl_pool(
        [Estimate(f"e{i}", ex, 0.02, (ex - 0.03, ex + 0.03), {}) for i, ex in enumerate([0.3, 0.45, 0.6, 0.7])]
    )
    assert scattered["tau2"] > homogeneous["tau2"]
    width_h = homogeneous["ci90"][1] - homogeneous["ci90"][0]
    width_s = scattered["ci90"][1] - scattered["ci90"][0]
    assert width_s > width_h


def test_single_source_pool_degenerates_to_its_own_interval():
    pooled = dl_pool([Estimate("solo", 0.45, 0.03, (0.4, 0.5), {})])
    assert pooled["k"] == 1
    assert pooled["ex_pooled"] == pytest.approx(0.45)
    assert pooled["ci90"][0] == pytest.approx(0.45 - 1.6448536269514722 * 0.03)


# ---------------------------------------------------------------------------
# meta-regression identification
# ---------------------------------------------------------------------------


def test_meta_regression_withholds_unidentifiable_covariates():
    ests = [
        Estimate("a", 0.45, 0.03, (0.4, 0.5), {}),
        Estimate("b", 0.55, 0.04, (0.48, 0.62), {}),
    ]
    out = meta_regression(
        ests,
        {
            "temperature": [None, -12.0],  # only one source reports it
            "thickness": [1.0, 1.0],  # no contrast
        },
    )
    assert out["dropped"] == ["temperature", "thickness"]
    assert "temperature" in out["identification"]
    assert out["identification"]["temperature"]["identifiable"] is False
    assert out["rank"] == 1
    assert out["coefficients"]["intercept"] == pytest.approx(
        (0.45 / 0.03 ** 2 + 0.55 / 0.04 ** 2) / (1 / 0.03 ** 2 + 1 / 0.04 ** 2)
    )


def test_meta_regression_fits_a_real_contrast():
    ests = [Estimate("a", 0.45, 0.03, (0.4, 0.5), {}), Estimate("b", 0.65, 0.03, (0.6, 0.7), {})]
    out = meta_regression(ests, {"thickness": [1.0, 2.0]})
    assert out["rank"] == 2
    assert out["coefficients"]["thickness"] == pytest.approx(0.2, rel=1e-6)


# ---------------------------------------------------------------------------
# demand bandwidth
# ---------------------------------------------------------------------------


def test_area_ratio_inversion_is_consistent_for_a_pure_power_law():
    r = required_area_ratio((1.32, 1.74), (0.3, 0.7))
    assert r ** 0.3 == pytest.approx(1.32, rel=0.06)
    assert r ** 0.7 == pytest.approx(1.74, rel=0.06)


def test_calibrated_stratum_narrows_the_declared_band():
    library = load_library()
    design = dl_pool(estimates_from_library(library, "design_rule"))
    study = demand_bandwidth_study(
        (0.30, 0.70), (1.32, 1.74), {"design_rule": tuple(design["ci90"])}
    )
    row = study["rows"][0]
    assert row["width"] < study["declared_width"]
    assert row["narrowing_factor_vs_declared"] > 3.0
    assert row["inside_declared"] is True


def test_measured_stratum_is_not_contained_in_the_declared_band():
    """Honest negative result: measured local pressures exceed the declared band."""
    library = load_library()
    measured = dl_pool(estimates_from_library(library, "measured_local"))
    study = demand_bandwidth_study(
        (0.30, 0.70), (1.32, 1.74), {"measured_local": tuple(measured["ci90"])}
    )
    assert study["rows"][0]["inside_declared"] is False


# ---------------------------------------------------------------------------
# end-to-end acceptance run
# ---------------------------------------------------------------------------


def test_acceptance_run_passes_all_reference_checks():
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "run_gap_pp2_pressure_area.py"), "--no-figures"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin:/usr/local/bin"},
    )
    assert out.returncode == 0, out.stdout + out.stderr
    payload = json.loads((REPO / "artifacts" / "ice_pressure_area" / "calibration.json").read_text())
    assert payload["verdict"]["all_ref_vs_got_pass"] is True
    assert payload["verdict"]["n_rows"] >= 7
    for row in payload["ref_vs_got"]:
        assert row["error_pct"] <= row["tol_pct"]
    # the convention-normalized design-rule stratum must beat the declared band
    design = payload["pooling_by_stratum"]["design_rule"]
    assert design["ci90"][1] - design["ci90"][0] < 0.5 * (0.70 - 0.30)