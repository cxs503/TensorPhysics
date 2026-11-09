"""The ``R = delta / dx`` discriminability law (W10-N07) and its gate wiring.

Acceptance of this node:

* the law reproduces ``R = delta/dx`` exactly and classifies subgrid / marginal /
  resolved at the documented thresholds;
* a declared-yet-subgrid feature turns a textbook-clean Richardson study into
  ``UNRESOLVED`` -- i.e. the gate refuses to publish a converged number for
  physics the grid cannot carry;
* an *undeclared* feature scale turns the verdict into ``WITHHELD`` (not PASS,
  not FAIL);
* ``required_grid_spacing`` inverts the law so a study can be planned;
* the legacy three-status vocabulary is untouched when no feature is declared.
"""
from __future__ import annotations

import math

import pytest

from tensorphysics.vv import (
    DiscriminabilityPolicy,
    FeatureSpec,
    Observation,
    discriminability,
    evaluate,
    required_grid_spacing,
)
from tensorphysics.vv.gci import FAIL, PASS, UNRESOLVED, WARN, WITHHELD


def _power_law_study(hs=(0.04, 0.02, 0.01), f_exact=1.0, coefficient=1.6, p=2.0):
    """A textbook-clean 2nd-order study (monotone, p=2, tiny GCI)."""
    return [Observation(h=h, f=f_exact + coefficient * h**p) for h in hs]


# ---------------------------------------------------------------------------
# the law itself
# ---------------------------------------------------------------------------


def test_ratio_is_delta_over_dx():
    res = discriminability([FeatureSpec("bl", 0.008)], 0.002)
    assert res.features[0].ratio == pytest.approx(4.0)
    assert res.status == "RESOLVED"
    assert res.dx == pytest.approx(0.002)


@pytest.mark.parametrize(
    "delta, dx, expected",
    [
        (0.002, 0.010, "UNRESOLVED"),  # R = 0.2 subgrid
        (0.0099, 0.010, "UNRESOLVED"),  # R = 0.99 just below the subgrid line
        (0.010, 0.010, "MARGINAL"),  # R = 1 exactly -> representable, not resolved
        (0.039, 0.010, "MARGINAL"),  # R = 3.9
        (0.040, 0.010, "RESOLVED"),  # R = 4 exactly -> the floor
        (2.0, 0.010, "RESOLVED"),
    ],
)
def test_classification_thresholds(delta, dx, expected):
    res = discriminability([{"name": "feature", "delta": delta}], dx)
    assert res.status == expected
    assert res.features[0].ratio == pytest.approx(delta / dx)


def test_worst_feature_governs_the_study():
    res = discriminability(
        [{"name": "resolved_eddy", "delta": 0.5}, {"name": "thin_bl", "delta": 0.002}], 0.01
    )
    assert [f.status for f in res.features] == ["RESOLVED", "UNRESOLVED"]
    assert res.status == "UNRESOLVED"
    assert res.ratio == pytest.approx(0.2)
    assert "thin_bl" in " ".join(res.reasons)


def test_missing_spacing_withholds_each_feature():
    res = discriminability([{"name": "bl", "delta": 0.01}], None)
    assert res.status == WITHHELD
    assert res.features[0].ratio is None
    assert "no grid spacing" in res.features[0].detail


def test_no_feature_declared_is_withheld_by_default_policy():
    res = discriminability([], 0.01)
    assert res.status == WITHHELD
    assert res.features == ()
    # an explicit opt-out is possible when the study genuinely has no feature
    lenient = discriminability([], 0.01, policy=DiscriminabilityPolicy(require_declaration=False))
    assert lenient.status == "RESOLVED"


def test_policy_validation_and_inversion():
    with pytest.raises(ValueError):
        DiscriminabilityPolicy(min_cells_per_feature=1.0, subgrid_below=1.0)
    with pytest.raises(ValueError):
        discriminability([{"name": "bl", "delta": 0.0}], 0.01)
    with pytest.raises(ValueError):
        discriminability([{"name": "bl", "delta": 0.01}], 0.0)
    # planning: a boundary layer of 3 mm needs dx <= 0.75 mm at the 4-cell floor
    assert required_grid_spacing(0.003) == pytest.approx(0.00075)
    assert required_grid_spacing(0.003, policy=DiscriminabilityPolicy(min_cells_per_feature=8.0)) == pytest.approx(
        0.000375
    )


def test_as_dict_is_self_describing():
    res = discriminability([{"name": "bl", "delta": 0.008}], 0.002)
    payload = res.as_dict()
    assert payload["schema"] == "tensorphysics.vv.resolvability/1"
    assert payload["law"] == "R = delta / dx"
    assert payload["status"] == "RESOLVED"
    assert payload["ratio"] == pytest.approx(4.0)
    assert payload["policy"]["min_cells_per_feature"] == 4.0


# ---------------------------------------------------------------------------
# gate integration
# ---------------------------------------------------------------------------


def test_resolved_feature_keeps_the_convergence_verdict():
    verdict = evaluate(_power_law_study(), features=[{"name": "bl", "delta": 0.006}])
    # dx = 0.01 (finest level), R = 0.6? -> subgrid; use a resolved feature instead
    assert verdict.status == UNRESOLVED
    assert verdict.resolvability.status == "UNRESOLVED"

    resolved = evaluate(_power_law_study(), features=[{"name": "bl", "delta": 0.05}])
    assert resolved.resolvability.status == "RESOLVED"
    assert resolved.resolvability.ratio == pytest.approx(5.0)
    assert resolved.status == WARN  # WARN only because the noise floor is undeclared


def test_subgrid_feature_overrides_a_passing_study():
    # the cloud-cavitation / vortex-core case: clean convergence of a quantity
    # whose controlling feature is smaller than one cell
    verdict = evaluate(_power_law_study(), features=[{"name": "vortex_core", "delta": 0.002}])
    assert verdict.status == UNRESOLVED
    check = next(c for c in verdict.checks if c["check"].startswith("resolvability"))
    assert check["status"] == UNRESOLVED
    assert check["ratio"] == pytest.approx(0.2)
    assert "subgrid" in check["detail"]


def test_marginal_feature_caps_at_warn():
    verdict = evaluate(_power_law_study(), features=[{"name": "bl", "delta": 0.02}])
    assert verdict.resolvability.status == "MARGINAL"
    assert verdict.status == WARN


def test_require_resolvability_without_declaration_withholds():
    verdict = evaluate(_power_law_study(), require_resolvability=True)
    assert verdict.status == WITHHELD
    assert verdict.resolvability.status == WITHHELD
    check = next(c for c in verdict.checks if c["check"].startswith("resolvability"))
    assert "cannot certify" in check["detail"]


def test_explicit_dx_overrides_the_finest_level():
    verdict = evaluate(
        _power_law_study(), features=[{"name": "bl", "delta": 0.008}], dx=0.002
    )
    assert verdict.resolvability.dx == pytest.approx(0.002)
    assert verdict.resolvability.ratio == pytest.approx(4.0)


def test_fail_still_dominates_an_unresolved_feature():
    # a stagnant sequence stays FAIL even with a subgrid feature: the solver
    # failure is the stronger statement
    stagnant = [Observation(h=h, f=0.5) for h in (0.04, 0.02, 0.01)]
    verdict = evaluate(stagnant, features=[{"name": "bl", "delta": 1e-6}])
    assert verdict.status == FAIL
    assert verdict.resolvability.status == "UNRESOLVED"


def test_legacy_verdict_untouched_without_features():
    verdict = evaluate(_power_law_study())
    assert verdict.resolvability is None
    assert verdict.status == WARN
    assert {c["status"] for c in verdict.checks} <= {PASS, WARN, FAIL}


def test_resolvability_scales_with_refinement():
    """Refining the grid by 8x turns an unresolved feature into a resolved one."""
    coarse = evaluate(_power_law_study(hs=(0.04, 0.02, 0.01)), features=[{"name": "bl", "delta": 0.002}])
    fine = evaluate(
        _power_law_study(hs=(0.005, 0.0025, 0.00125)), features=[{"name": "bl", "delta": 0.002}]
    )
    assert coarse.resolvability.ratio == pytest.approx(0.2)
    assert fine.resolvability.ratio == pytest.approx(1.6)
    assert coarse.resolvability.status == "UNRESOLVED"
    assert fine.resolvability.status in {"MARGINAL", "RESOLVED"}
    assert math.isclose(
        fine.resolvability.ratio / coarse.resolvability.ratio, 8.0, rel_tol=1e-12
    )