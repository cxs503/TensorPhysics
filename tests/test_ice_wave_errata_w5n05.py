"""Regression tests for the W5-N05 (m994) errata: exponent basis + EnKF statistics.

Two layers are covered:

1. ``tensorphysics.ice_wave.miz.sweep_attenuation`` now carries the *basis* of its
   two published exponents (which slice is held fixed) plus a whole-grid joint
   estimate -- so a reader can reproduce or reject the numbers without guessing
   the convention;
2. ``examples/ice_wave_w5n05_errata.py`` re-derives the values of report #823
   from the stored artifact only.  These tests assert the three adjudication
   facts established by the erratum:

   * the published ``p``/``q`` are bit-for-bit the slice slopes at h = 1.5 m and
     n_c = 0.6;
   * the six numbers quoted by the independent review are reproducible from the
     8-point sub-grid *printed* in the report (so the disagreement is a data-set
     mismatch, not an arithmetic error);
   * ``8.65 %`` is correct and ``8.33 %`` is the artifact of re-deriving the
     percentage from 4-decimal rendered RMSEs.

The artifact-dependent tests skip when ``artifacts/w5n05/results.json`` is not
present (it is a run product, not a source file).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from tensorphysics.ice_wave import miz

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "w5n05" / "results.json"
ERRATA_PY = ROOT / "examples" / "ice_wave_w5n05_errata.py"
ERRATA_MD = ROOT / "docs" / "papers" / "w5n05_ice_wave" / "ERRATA.md"

needs_artifact = pytest.mark.skipif(
    not RESULTS.exists(), reason="run artifact artifacts/w5n05/results.json absent"
)


def _load_errata_module():
    if not ERRATA_PY.exists():
        pytest.skip("errata script absent")
    spec = importlib.util.spec_from_file_location("ice_wave_w5n05_errata", ERRATA_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_results():
    return json.loads(RESULTS.read_text())


# --------------------------------------------------------------------------- #
# 1. sweep_attenuation carries the exponent basis
# --------------------------------------------------------------------------- #
def test_sweep_exponents_declare_their_slice():
    conc = [0.3, 0.5, 0.7, 0.9]
    thick = [0.5, 1.0, 1.5, 2.0]
    sweep = miz.sweep_attenuation(conc, thick, period=10.0, depth=1000.0)
    grid = np.asarray(sweep["alpha_grid"])

    # the published exponents are the middle slices -- now named explicitly
    i_mid, j_mid = len(thick) // 2, len(conc) // 2
    assert sweep["concentration_exponent_slice_thickness"] == pytest.approx(thick[i_mid])
    assert sweep["thickness_exponent_slice_concentration"] == pytest.approx(conc[j_mid])

    p_expected = np.polyfit(np.log(conc), np.log(grid[i_mid, :]), 1)[0]
    q_expected = np.polyfit(np.log(thick), np.log(grid[:, j_mid]), 1)[0]
    assert sweep["alpha_vs_concentration_exponent"] == pytest.approx(p_expected, rel=1e-12)
    assert sweep["alpha_vs_thickness_exponent"] == pytest.approx(q_expected, rel=1e-12)

    # every slice is tabulated
    assert len(sweep["concentration_exponent_slices"]) == len(thick)
    assert len(sweep["thickness_exponent_slices"]) == len(conc)

    joint = sweep["joint_exponents"]
    assert joint["n_points"] == len(conc) * len(thick)
    # joint estimate differs from the local slice values -> provenance matters
    assert joint["concentration_exponent"] != pytest.approx(
        sweep["alpha_vs_concentration_exponent"], rel=1e-6
    )
    lo, hi = joint["concentration_slice_range"]
    assert lo < sweep["alpha_vs_concentration_exponent"] < hi
    assert "LOCAL" in sweep["exponent_note"]


def test_power_law_exponents_matches_independent_solve():
    conc = np.array([0.2, 0.5, 0.9])
    thick = np.array([0.5, 1.0, 2.0])
    # exact separable power law alpha = 3 n^0.4 h^1.1 -> must be recovered
    grid = 3.0 * conc[None, :] ** 0.4 * thick[:, None] ** 1.1
    fit = miz.power_law_exponents(conc, thick, grid)
    assert fit["concentration_exponent"] == pytest.approx(0.4, abs=1e-10)
    assert fit["thickness_exponent"] == pytest.approx(1.1, abs=1e-10)
    assert fit["r2_log"] == pytest.approx(1.0, abs=1e-12)
    assert fit["n_points"] == 9
    # an exact power law has zero slice dispersion
    assert fit["concentration_slice_range"][1] - fit["concentration_slice_range"][0] < 1e-10


def test_power_law_exponents_rejects_bad_inputs():
    with pytest.raises(miz.MIZError):
        miz.power_law_exponents([0.3, 0.6], [1.0], np.ones((2, 2)))


# --------------------------------------------------------------------------- #
# 2. Errata re-derivation against the stored artifact
# --------------------------------------------------------------------------- #
@needs_artifact
def test_published_exponents_are_the_declared_slices():
    errata = _load_errata_module()
    adj = errata.adjudicate(_load_results())
    conv = adj["miz_census"]["code_convention"]
    assert conv["p_slice_thickness"] == pytest.approx(1.5)
    assert conv["q_slice_concentration"] == pytest.approx(0.6)
    assert conv["p_abs_err_vs_stored"] <= 1e-12
    assert conv["q_abs_err_vs_stored"] <= 1e-12


@needs_artifact
def test_review_values_come_from_the_printed_subgrid():
    errata = _load_errata_module()
    adj = errata.adjudicate(_load_results())
    ident = adj["review_convention_identification"]
    # the reviewer quoted 4 decimals, so identification is exact to rounding
    assert ident["max_abs_diff"] <= 5e-4
    assert ident["p_joint_from_subgrid"] == pytest.approx(0.4011, abs=5e-4)
    assert ident["q_joint_from_subgrid"] == pytest.approx(1.1225, abs=5e-4)
    assert ident["p_slices_from_subgrid"][0] == pytest.approx(0.6278, abs=5e-4)
    assert ident["p_slices_from_subgrid"][-1] == pytest.approx(0.1744, abs=5e-4)
    assert ident["q_slices_from_subgrid"][0] == pytest.approx(1.5288, abs=5e-4)
    assert ident["q_slices_from_subgrid"][-1] == pytest.approx(0.8122, abs=5e-4)


@needs_artifact
def test_enkf_reduction_is_86484_percent_not_83333():
    errata = _load_errata_module()
    adj = errata.adjudicate(_load_results())
    e = adj["enkf"]
    assert e["reduction_recomputed"] == pytest.approx(e["reduction_stored"], abs=1e-12)
    assert e["reduction_recomputed"] == pytest.approx(0.0864835462836, abs=1e-12)
    # the reviewer's 8.33 % is exactly 1 - 0.0044/0.0048
    assert e["reduction_from_displayed_4dp"] == pytest.approx(1.0 - 0.0044 / 0.0048, abs=1e-12)
    assert e["reduction_from_displayed_4dp"] == pytest.approx(0.0833, abs=1e-4)


@needs_artifact
def test_errata_document_is_fully_rendered():
    errata = _load_errata_module()
    adj = errata.adjudicate(_load_results())
    figures = [Path("artifacts/w5n05/errata") / name for name in
               ("errata_fig1_exponent_conventions.png",
                "errata_fig2_grid_vs_printed.png",
                "errata_fig3_enkf_rounding.png")]
    text = errata.render_errata(adj, figures)
    assert "{{" not in text and "}}" not in text
    assert "Errata" in text
    if ERRATA_MD.exists():
        assert "{{" not in ERRATA_MD.read_text(encoding="utf-8")


@needs_artifact
def test_check_gate_passes():
    errata = _load_errata_module()
    assert errata.main(["--check"]) == 0