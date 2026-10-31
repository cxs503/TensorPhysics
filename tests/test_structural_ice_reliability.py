"""Verification of the ISO 19906 ice-belt reliability model and thickness design."""
from __future__ import annotations

from math import pi

import pytest

from tensorphysics.structural import (
    BUCKLING,
    FATIGUE,
    YIELD,
    PolarMidshipReliability,
    ReliabilityError,
    ReliabilityProblem,
    default_variables,
    required_thickness,
)

TOLERANCE_BETA = 0.01


@pytest.fixture(scope="module")
def analysis() -> PolarMidshipReliability:
    return PolarMidshipReliability()


@pytest.fixture(scope="module")
def forms(analysis: PolarMidshipReliability):
    return analysis.form_all()


# --------------------------------------------------------------------------- #
# Deterministic panel model
# --------------------------------------------------------------------------- #
def test_hand_calculation_of_bending_stress(analysis: PolarMidshipReliability):
    geometry = analysis.geometry
    pressure, k_mod, corrosion = 2.0, 1.1, 2.5
    t_net = geometry.t_nom - corrosion
    expected = k_mod * 0.5 * pressure * (geometry.stiffener_spacing / t_net) ** 2
    assert geometry.bending_stress(pressure, corrosion, k_mod) == pytest.approx(expected, rel=1e-12)
    # representative magnitude: a 2 MPa patch on a 350 mm span with 31.5 mm net plate
    assert 100.0 < expected < 200.0


def test_elastic_buckling_stress_matches_plate_formula(analysis: PolarMidshipReliability):
    geometry = analysis.geometry
    t_net = geometry.t_nom - 2.0
    coefficient = 4.0 * pi**2 * geometry.E / (12.0 * (1.0 - geometry.nu**2))
    expected = coefficient * (t_net / geometry.frame_spacing) ** 2
    assert geometry.elastic_buckling_stress(2.0) == pytest.approx(expected, rel=1e-12)
    # 32 mm plate over a 2.4 m bay: elastic buckling stress is of the order of 130 MPa
    assert 100.0 < expected < 200.0


def test_johnson_ostenfeld_branches(analysis: PolarMidshipReliability):
    geometry = analysis.geometry
    # elastic branch: sigma_E <= 0.5 fy -> capacity equals the elastic stress
    sigma_e = geometry.elastic_buckling_stress(2.0)
    fy_high = 4.0 * sigma_e
    assert geometry.buckling_stress(2.0, 2.0, 1.0, fy_high) == pytest.approx(sigma_e, rel=1e-12)
    # elastic-plastic branch: sigma_E > 0.5 fy -> Johnson-Ostenfeld reduction
    fy_low = 0.5 * sigma_e
    expected = fy_low * (1.0 - fy_low / (4.0 * sigma_e))
    assert geometry.buckling_stress(2.0, 2.0, 1.0, fy_low) == pytest.approx(expected, rel=1e-12)
    assert expected < fy_low


def test_fatigue_damage_scaling_laws(analysis: PolarMidshipReliability):
    geometry = analysis.geometry
    d1 = geometry.fatigue_damage(2.0, 1.0)
    d2 = geometry.fatigue_damage(2.0, 2.0)
    assert d2 / d1 == pytest.approx(8.0, rel=1e-9)  # cubic in the model factor
    d3 = geometry.fatigue_damage(2.0, 1.0)
    thick = geometry.with_thickness(geometry.t_nom * 2.0 - 2.0)
    # delta-sigma ~ t^-2 and the S-N slope is 3 -> damage ~ t^-6
    expected_ratio = ((geometry.t_nom - 2.0) / (thick.t_nom - 2.0)) ** 6
    assert thick.fatigue_damage(2.0, 1.0) / d3 == pytest.approx(expected_ratio, rel=1e-9)


def test_nominal_stresses_are_consistent(analysis: PolarMidshipReliability):
    nominal = analysis.nominal_stresses()
    geometry = analysis.geometry
    assert nominal["t_net"] == pytest.approx(geometry.t_nom - 2.0, rel=1e-12)
    assert nominal["bending_stress"] == pytest.approx(
        geometry.bending_stress(2.5, 2.0, 1.0), rel=1e-12
    )
    assert nominal["fatigue_damage"] == pytest.approx(0.18, abs=0.02)
    assert nominal["fatigue_damage"] < nominal["damage_allow"]


# --------------------------------------------------------------------------- #
# Random variable model
# --------------------------------------------------------------------------- #
def test_default_variable_set_structure(analysis: PolarMidshipReliability):
    variables = analysis.variables
    assert variables.names == ("p_ice", "fy", "dt_corr", "k_mod")
    assert variables.units == ("MPa", "MPa", "mm", "-")
    description = variables.describe()
    assert len(description) == 4
    assert description[0]["distribution"] == "gumbel_max"
    assert description[1]["distribution"] == "lognormal"
    assert description[2]["distribution"] == "normal"
    assert description[3]["distribution"] == "lognormal"
    assert description[0]["mean"] == pytest.approx(2.5, rel=1e-12)
    assert description[0]["cov"] == pytest.approx(0.28, rel=1e-6)
    assert description[1]["mean"] == pytest.approx(355.0, rel=1e-12)
    correlation = variables.correlation
    assert correlation[0][3] == pytest.approx(0.25, rel=1e-12)
    assert correlation[0][1] == 0.0
    for i in range(4):
        assert correlation[i][i] == 1.0
        for j in range(4):
            assert correlation[i][j] == pytest.approx(correlation[j][i], abs=1e-15)


def test_variable_defaults_are_overridable():
    variables = default_variables(pressure_mean=3.0, load_model_correlation=0.0)
    assert variables.marginals[0].mean() == pytest.approx(3.0, rel=1e-12)
    assert variables.correlation[0][3] == 0.0
    transform = variables.transform()
    assert transform.rho0[0][3] == 0.0


# --------------------------------------------------------------------------- #
# FORM / MC / sensitivities (criteria b and d)
# --------------------------------------------------------------------------- #
def test_batch_limit_states_match_scalar(analysis: PolarMidshipReliability):
    numpy = pytest.importorskip("numpy")
    batch = analysis.geometry.batch_limit_states()
    assert set(batch) == {YIELD, BUCKLING, FATIGUE}
    rng = numpy.random.default_rng(5)
    rows = numpy.column_stack(
        [
            rng.uniform(1.0, 6.0, 300),
            rng.uniform(280.0, 430.0, 300),
            rng.uniform(0.5, 4.0, 300),
            rng.uniform(0.8, 1.6, 300),
        ]
    )
    for name, function in analysis.geometry.limit_states.items():
        scalar = numpy.asarray([function(list(row)) for row in rows])
        vector = numpy.asarray(batch[name](rows))
        assert numpy.allclose(scalar, vector, rtol=1e-12, atol=1e-12)


def test_component_beta_matches_monte_carlo(analysis, forms):
    """Criterion (b): FORM index vs 1e6-sample Monte Carlo within 2%."""
    for name in (YIELD, BUCKLING, FATIGUE):
        mc = analysis.problems[name].monte_carlo_pf(samples=1_000_000, seed=2026)
        relative = abs(forms[name].beta - mc.beta) / mc.beta
        assert mc.samples == 1_000_000
        assert mc.censor_ok
        assert relative < 0.02, f"{name}: FORM {forms[name].beta:.4f} vs MC {mc.beta:.4f}"
        assert forms[name].converged


def test_design_points_lie_in_physical_ranges(analysis, forms):
    for name, form in forms.items():
        pressure, fy, corrosion, k_mod = form.design_point_x
        assert 1.0 < pressure < 12.0, name
        assert 250.0 < fy < 480.0, name
        assert 0.0 <= corrosion < 6.0, name
        assert 0.7 < k_mod < 3.0, name
        assert abs(form.g_design) < 1e-5 * max(1.0, abs(fy)), name


def test_sensitivity_table_matches_finite_difference(analysis):
    """Criterion (d): design-point sensitivity vs central differences (h = 1e-4)."""
    for limit_state in (YIELD, BUCKLING, FATIGUE):
        table = analysis.sensitivities(limit_state)
        assert len(table) == 8
        for key, entry in table.items():
            if abs(entry["finite_difference"]) < 1e-9:
                assert abs(entry["analytic"]) < 1e-8, key
                continue
            assert entry["relative_error"] < 0.01, f"{limit_state}/{key}: {entry}"
    # physically required signs: thicker plate raises beta, higher corrosion lowers it
    table = analysis.sensitivities(YIELD)
    assert table["dt_corr.mu"]["analytic"] < 0.0
    assert table["p_ice.loc"]["analytic"] < 0.0
    assert table["fy.mu_ln"]["analytic"] > 0.0
    # the elastic buckling capacity does not depend on the yield strength
    assert analysis.sensitivities(BUCKLING)["fy.mu_ln"]["analytic"] == pytest.approx(0.0, abs=1e-8)


def test_correlation_between_load_model_and_pressure_lowers_beta(analysis, forms):
    uncorrelated = PolarMidshipReliability(
        geometry=analysis.geometry, variables=default_variables(load_model_correlation=0.0)
    )
    beta_uncorrelated = uncorrelated.form_all()[YIELD].beta
    beta_correlated = forms[YIELD].beta
    # a positive correlation between load bias and load intensity shifts the
    # joint design point into the tail and reduces beta
    assert beta_correlated < beta_uncorrelated


# --------------------------------------------------------------------------- #
# Thickness design (target beta = 4.0)
# --------------------------------------------------------------------------- #
def test_beta_increases_with_plate_thickness(analysis):
    betas = [analysis.beta_of_thickness(t, YIELD) for t in (30.0, 34.0, 38.0, 42.0)]
    assert betas == sorted(betas)
    assert betas[-1] > betas[0]


def test_required_thickness_reaches_target_beta(analysis):
    for limit_state in (YIELD, BUCKLING, FATIGUE):
        outcome = analysis.required_thickness_for(limit_state, target_beta=4.0)
        assert abs(outcome["beta_achieved"] - 4.0) <= TOLERANCE_BETA
        assert outcome["t_required"] > analysis.geometry.t_nom
        assert outcome["iterations"] > 1
    system = analysis.required_thickness_for(None, target_beta=4.0)
    assert abs(system["beta_achieved"] - 4.0) <= TOLERANCE_BETA
    assert system["measure"] == "system(ditlevsen)"
    # the series system can never be safer than its weakest component
    assert system["t_required"] >= analysis.required_thickness_for(YIELD, 4.0)["t_required"] - 1e-6


def test_required_thickness_rejects_unbracketed_target(analysis):
    with pytest.raises(ReliabilityError):
        required_thickness(
            lambda t: analysis.beta_of_thickness(t, YIELD), target_beta=25.0, lower=40.0, upper=45.0
        )


def test_system_bounds_bracket_simulated_union(analysis, forms):
    pytest.importorskip("numpy")
    import numpy as np

    from tensorphysics.structural._batch import batch_u_to_x

    bounds = analysis.system_bounds(forms)
    rng = np.random.default_rng(2027)
    x = batch_u_to_x(analysis.transform, rng.standard_normal((400_000, 4)))
    union = np.zeros(len(x), dtype=bool)
    for function in analysis.geometry.batch_limit_states().values():
        union |= function(x) < 0.0
    pf_sim = float(union.mean())
    assert bounds.pf_lower * 0.9 <= pf_sim <= bounds.pf_upper * 1.1
    assert bounds.pf_lower <= bounds.pf_upper
    assert bounds.beta_lower <= bounds.beta_upper
    # buckling and fatigue are almost co-directional in u-space
    assert bounds.bimodal_rho[1][2] > 0.9


def test_thin_plate_median_point_violation_is_reported(analysis):
    geometry = analysis.geometry.with_thickness(18.0)
    problem = ReliabilityProblem(
        analysis.transform, geometry.limit_states[BUCKLING], name=BUCKLING
    )
    with pytest.raises(ReliabilityError):
        problem.form()
    assert analysis.beta_of_thickness(18.0, BUCKLING) == float("-inf")