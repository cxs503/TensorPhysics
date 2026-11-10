"""P82W2-N08 -- narrative/artifact consistency guard.

The node report (``docs/P82W2_N08_ICE_WAVE_COMBINATION_LAW.md``) and the publish
payload (``artifacts/p82w2_n08/report_payload.json``) must be *derived* from the
experiment artifact ``artifacts/p82w2_n08/combination_law.json``.  Nothing may be
typed by hand: a stale hand-typed summary once quoted ``kappa(r=1)=1.406`` while
the artifact said ``1.4136``.

This module re-derives every headline number from the JSON and asserts it appears
in the rendered document / summary, and that the JSON verdict block agrees with
the pre-declared criteria.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "p82w2_n08"
DATA_PATH = ART / "combination_law.json"
DOC = ROOT / "docs" / "P82W2_N08_ICE_WAVE_COMBINATION_LAW.md"
PAYLOAD = ART / "report_payload.json"

pytestmark = pytest.mark.skipif(
    not DATA_PATH.exists() or not DOC.exists(),
    reason="P82W2-N08 artifacts not materialised in this checkout",
)


def _load_publish_module():
    """Import scripts/p82w2_n08_publish.py as a module (it reads the artifact)."""
    path = ROOT / "scripts" / "p82w2_n08_publish.py"
    spec = importlib.util.spec_from_file_location("p82w2_n08_publish", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pub():
    return _load_publish_module()


@pytest.fixture(scope="module")
def data():
    return json.loads(DATA_PATH.read_text())


def test_document_is_artifact_derived(pub):
    """The committed doc must be byte-identical to the JSON-derived rendering."""
    assert DOC.read_text() == pub.md, (
        "docs/P82W2_N08_ICE_WAVE_COMBINATION_LAW.md is not the rendering of "
        "artifacts/p82w2_n08/combination_law.json -- re-run "
        "`python scripts/p82w2_n08_publish.py --no-post`"
    )


def test_summary_numbers_track_artifact(pub, data):
    """Every number quoted in the publish summary is recomputed from the JSON."""
    s = pub.SUMMARY
    crit = data["criteria"]
    rows = data["scan"]["rows"]

    def kappa(r, split):
        row = [x for x in rows
               if abs(x["f_ratio"] - r) < 1e-9
               and abs(x["sigma_ice_MPa"] / x["sigma_wave_MPa"] - split) < 1e-6]
        assert len(row) == 1, (r, split)
        return row[0]["ratio_mean"]

    # (a) equal-amplitude separation kappas and the strongest cell
    for r in (10.0, 30.0, 100.0):
        assert f"{kappa(r, 1.0):.3f}" in s
    assert f"{kappa(100.0, 2.0):.3f}" in s
    worst = max(x["ratio_mean"] for x in rows)
    assert f"{worst:.3f}" in s
    # the fused-band analytic anchor
    assert f"{kappa(1.0, 1.0):.4f}" in s
    assert f"{2.0 ** 0.5:.4f}" in s
    # (b) worst spectral-closure error
    b = crit["b_spectral_closures"]
    max_err = max(max(r["err_dirlik_abs_pct_mean"], r["err_tb_abs_pct_mean"]) for r in b["rows"])
    assert f"{max_err:.1f}" in s
    # (c) whipping share-law deviations
    c = crit["c_whipping_accounting"]
    assert f"{c['synthetic_deviation_pp']:.0f}" in s
    assert f"{max(c['physical_deviations_pp']):.0f}" in s
    assert c["slow_varying_below_8pct"] is True
    # (d) commutativity magnitude
    d = crit["d_rainflow_commutativity"]
    comm = max(d["periodic_record"]["segment_accumulation"],
               d["periodic_record"]["time_reversal"],
               d["stationary_composite"]["segment_accumulation"],
               d["stationary_composite"]["time_reversal"])
    assert f"{comm:.1e}" in s
    # numbers trimmed from the (500-char capped) summary must still be in the body
    body = pub.md
    assert f"{100.0 * c['slow_varying_share_peak']:.2f}" in body
    assert f"{d['stationary_raw_open_segment_permutation_max']:.2e}" in body
    assert f"{crit['e_negative_control_m5528']['rel_dev_pct']:.6f}" in body


def test_summary_fits_reports_api_cap(pub):
    """The backend rejects summaries longer than 500 characters (HTTP 422)."""
    assert len(pub.SUMMARY) <= 500, len(pub.SUMMARY)


def test_summary_carries_no_stale_hand_typed_value(pub):
    """Regression guard for the 1.406 <- 1.4136 drift observed on 2026-11-09."""
    assert "1.406" not in pub.SUMMARY


def test_payload_file_matches_live_summary(pub):
    if not PAYLOAD.exists():
        pytest.skip("payload not regenerated in this checkout")
    stored = json.loads(PAYLOAD.read_text())
    assert stored["summary"] == pub.SUMMARY
    assert stored["mission_id"] == 7763
    assert stored["scenario_key"] == "p82w2_n08"


def test_json_verdict_matches_predeclared_criteria(data):
    """The verdict block must be recomputable from the criterion measurements."""
    crit = data["criteria"]
    a = crit["a_linear_superposition"]
    b = crit["b_spectral_closures"]
    c = crit["c_whipping_accounting"]
    d = crit["d_rainflow_commutativity"]
    e = crit["e_negative_control_m5528"]

    sep = [row for row in a["cells_at_separation_ge_10"]]
    assert sep, "scan must contain separation >= 10 cells"
    a_fail = not a["all_in_tolerance"]
    recomputed = {
        "a": "FAIL" if a_fail else "PASS",
        "b": "PASS" if b["passes"] else "FAIL",
        "c": "PASS" if c["passes"] else "FAIL",
        "d": "PASS" if d["passes"] else "FAIL",
        "e": "PASS" if abs(e["rel_dev_pct"]) <= e["tolerance_pct"] else "FAIL",
    }
    assert data["verdict"] == recomputed
    # honest-FAIL must not be silently overwritten by a later annotation
    assert recomputed["a"] == "FAIL" and recomputed["c"] == "FAIL"


def test_report_figures_exist_and_are_png():
    figs = sorted(ART.glob("fig*.png"))
    assert len(figs) >= 6, [p.name for p in figs]
    for p in figs:
        head = p.read_bytes()[:8]
        assert head == b"\x89PNG\r\n\x1a\n", p.name
        assert p.stat().st_size > 20_000, p.name