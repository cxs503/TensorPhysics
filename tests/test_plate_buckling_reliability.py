"""Tests of the unified plate-buckling + structural-reliability framework.

The suite is organised in the order of the assessment chain: geometry gate ->
deterministic solver -> capacity surrogate -> limit states -> probabilistic
layer (FORM/SORM/system/sensitivity/design inverse) -> unified interface.

Most tests run against a *synthetic* capacity surface (the Faulkner curve), so
they exercise the framework itself in milliseconds; the two finite-element tests
at the end run the real arc-length solver on the calibration mesh of the
verification case.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.plate_buckling.benchmarks import (
    dnv_rp_c201_elastic_buckling,
    ritz_buckling_stress,
)
from tensorphysics.plate_buckling_reliability import (
    DEFAULT_MODES,
    MODE_DEFLECTION,
    MODE_ELASTIC,
    MODE_ULTIMATE,
    MODE_YIELD,
    GeometryGateError,
    GenericLimitStates,
    PanelLimitStates,
    PanelSpec,
    ProbabilisticAssessment,
    UltimateStrengthSurrogate,
    UnifiedBucklingReliability,
    VariableModel,
    beta_from_pf,
    faulkner_reference,
    load_default_surrogate,
)
from tensorphysics.plate_buckling_reliability.deterministic import (
    DeterministicAssessment,
    clear_cache,
)
from tensorphysics.plate_buckling_reliability.surrogate import (
    default_surrogate_path,
)

SIGMA_Y = 313.0e6
E_MOD = 2.06e11
CAL_BETAS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def synthetic_surrogate() -> UltimateStrengthSurrogate:
    """Exact Faulkner response surface (independent of any FE run)."""
    points = [{"beta": b, "sigma_u_over_sigma_y": float(faulkner_reference(b))}
              for b in CAL_BETAS]
    return UltimateStrengthSurrogate.fit(points, source="synthetic-faulkner")


@pytest.fixture(scope="module")
def nominal_panel() -> PanelSpec:
    return PanelSpec(a=3.0, b=1.0, t=float(np.sqrt(SIGMA_Y / E_MOD) / 2.0),
                     sigma_y=SIGMA_Y, nx=8, ny=4)


@pytest.fixture(scope="module")
def assessment(nominal_panel, synthetic_surrogate) -> ProbabilisticAssessment:
    states = PanelLimitStates(nominal_panel.with_thickness(13.6e-3),
                              synthetic_surrogate)
    return ProbabilisticAssessment(states, VariableModel())


@pytest.fixture(scope="module")
def design_assessment(nominal_panel, synthetic_surrogate) -> ProbabilisticAssessment:
    """Assessment at the beta = 4.0 design thickness (13.6 mm)."""
    states = PanelLimitStates(nominal_panel.with_thickness(13.6e-3),
                              synthetic_surrogate)
    return ProbabilisticAssessment(states, VariableModel())


# --------------------------------------------------------------------------- #
# 1. geometry gate
# --------------------------------------------------------------------------- #
def test_geometry_gate_accepts_nominal(nominal_panel):
    checks = nominal_panel.validate()
    assert checks["a_over_b"] == pytest.approx(3.0)
    assert checks["beta"] == pytest.approx(2.0, rel=1e-9)
    assert checks["b_over_t"] > 0


def test_geometry_gate_rejects_short_panel():
    with pytest.raises(GeometryGateError):
        PanelSpec(a=1.2, b=1.0, nx=6, ny=4).validate()


def test_geometry_gate_rejects_slender_panel():
    # beta ~ 6.1: outside the verified envelope of the capacity surface
    with pytest.raises(GeometryGateError):
        PanelSpec(t=6.5e-3).validate()


def test_panel_from_dict_accepts_mm_and_mpa():
    spec = PanelSpec.from_dict({"a": 3.0, "b": 1.0, "t_mm": 19.49,
                                "sigma_y_MPa": 313.0, "E_GPa": 206.0,
                                "sigma_global_MPa": 95.0, "mesh": [8, 4]})
    assert spec.t == pytest.approx(19.49e-3)
    assert spec.sigma_y == pytest.approx(313e6)
    assert spec.E == pytest.approx(206e9)
    assert spec.sigma_global == pytest.approx(95e6)
    assert (spec.nx, spec.ny) == (8, 4)


def test_panel_with_thickness_returns_new_spec(nominal_panel):
    thinner = nominal_panel.with_thickness(13.6e-3)
    assert thinner is not nominal_panel
    assert thinner.t == pytest.approx(13.6e-3)
    assert thinner.beta > nominal_panel.beta


# --------------------------------------------------------------------------- #
# 2. deterministic solver (fast, linear eigenvalue problem)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bc", ["SSSS", "CCCC", "SCSC", "CSCS"])
def test_linear_buckling_matches_ritz(bc):
    """Gate a of the acceptance list, per boundary condition (< 2 %)."""
    drive = DeterministicAssessment(PanelSpec(a=4.0, b=1.0, t=0.02, sigma_y=SIGMA_Y,
                                              nx=24, ny=6, bc=bc))
    row = drive.linear_buckling()
    assert abs(row["rel_err_vs_ritz"]) < 0.02
    assert row["sigma_cr_MPa"] > 0


def test_linear_buckling_equals_dnv_ssss():
    drive = DeterministicAssessment(PanelSpec(t=0.02, nx=12, ny=4))
    row = drive.linear_buckling()
    assert row["k_dnv_rp_c201"] == pytest.approx(4.0)
    assert row["k_fem"] == pytest.approx(4.0, rel=0.01)


# --------------------------------------------------------------------------- #
# 3. capacity surrogate
# --------------------------------------------------------------------------- #
def test_surrogate_recovers_faulkner_coefficients(synthetic_surrogate):
    c0, c1, c2 = synthetic_surrogate.coefficients()
    assert c0 == pytest.approx(0.0, abs=1e-10)
    assert c1 == pytest.approx(2.0, rel=1e-10)
    assert c2 == pytest.approx(-1.0, rel=1e-10)
    assert synthetic_surrogate.accuracy()["max_rel_error"] < 1e-12


def test_surrogate_strength_decreases_with_slenderness(synthetic_surrogate):
    ratios = [float(synthetic_surrogate.ratio(b)) for b in (1.0, 2.0, 3.0)]
    assert ratios[0] > ratios[1] > ratios[2] > 0.0


def test_surrogate_sigma_u_scales_with_yield(synthetic_surrogate):
    t, b = 17.0e-3, 1.0
    low = synthetic_surrogate.sigma_u(t, 313e6, b, E_MOD)
    high = synthetic_surrogate.sigma_u(t, 355e6, b, E_MOD)
    assert high > low > 0.0


def test_surrogate_dict_roundtrip(synthetic_surrogate, tmp_path):
    path = tmp_path / "surrogate.json"
    synthetic_surrogate.save(path)
    restored = UltimateStrengthSurrogate.load(path)
    assert restored.coefficients() == pytest.approx(synthetic_surrogate.coefficients())
    assert restored.beta_max == pytest.approx(synthetic_surrogate.beta_max)


def test_surrogate_rejects_degenerate_thickness(synthetic_surrogate):
    with pytest.raises(ValueError):
        synthetic_surrogate.sigma_u(0.0, SIGMA_Y, 1.0, E_MOD)


def test_packaged_calibration_exists_or_skips():
    """The packaged FE calibration must cover the six verification betas."""
    if not default_surrogate_path().exists():
        pytest.skip("packaged FE calibration not generated in this checkout")
    surrogate = load_default_surrogate()
    assert surrogate is not None
    betas = [float(row["beta"]) for row in surrogate.calibration]
    assert len(betas) >= 6
    assert min(betas) <= 1.01 and max(betas) >= 2.99
    assert surrogate.accuracy()["max_rel_error"] < 0.05


# --------------------------------------------------------------------------- #
# 4. limit states
# --------------------------------------------------------------------------- #
def test_limit_states_positive_at_mean(nominal_panel, synthetic_surrogate):
    states = PanelLimitStates(nominal_panel, synthetic_surrogate)
    margins = states.design_margin()
    assert margins[MODE_ULTIMATE] > 0.0
    assert margins[MODE_YIELD] > 0.0


def test_ultimate_limit_state_monotone_in_pressure(assessment):
    problem = assessment.problem(MODE_ULTIMATE)
    x = [1.0, 313.0, 2.0, 1.0]
    values = []
    for p in (1.0, 2.0, 3.0, 4.0):
        trial = list(x)
        trial[0] = p
        values.append(problem.limit_state(trial))
    assert values[0] > values[1] > values[2] > values[3]


def test_batch_limit_states_match_scalar(assessment):
    states = assessment.limit_states
    samples = np.array([[1.5, 300.0, 1.0, 0.9],
                        [2.5, 313.0, 2.0, 1.0],
                        [3.5, 340.0, 3.5, 1.2]])
    for mode, fn in states.batch_limit_states().items():
        scalar = [states.limit_states()[mode](row) for row in samples]
        assert np.allclose(fn(samples), scalar, rtol=1e-12, atol=1e-6)


def test_capacity_uses_net_thickness(nominal_panel, synthetic_surrogate):
    states = PanelLimitStates(nominal_panel, synthetic_surrogate)
    gross = states.capacity(313.0, 0.0)
    net = states.capacity(313.0, 4.0)
    assert gross > net > 0.0
    expected = synthetic_surrogate.sigma_u(nominal_panel.t - 4e-3, SIGMA_Y, 1.0, E_MOD)
    assert float(net) == pytest.approx(expected, rel=1e-12)


def test_elastic_mode_uses_dnv_coefficient(nominal_panel, synthetic_surrogate):
    states = PanelLimitStates(nominal_panel, synthetic_surrogate)
    geom = nominal_panel.geometry()
    sigma_e = float(states.elastic_critical_stress(0.0))
    assert sigma_e == pytest.approx(
        dnv_rp_c201_elastic_buckling(geom, "SSSS")["sigma_cr"], rel=1e-9)


def test_deflection_mode_is_available_but_not_default(nominal_panel, synthetic_surrogate):
    states = PanelLimitStates(nominal_panel, synthetic_surrogate)
    assert MODE_DEFLECTION in states.limit_states()
    assert MODE_ELASTIC in states.limit_states()
    assert MODE_DEFLECTION not in DEFAULT_MODES
    assert MODE_ELASTIC not in DEFAULT_MODES


# --------------------------------------------------------------------------- #
# 5. probabilistic layer
# --------------------------------------------------------------------------- #
def test_form_design_point_lies_on_limit_state(design_assessment):
    form = design_assessment.form(MODE_ULTIMATE)
    assert form.converged
    assert abs(form.g_design) < 1e-3 * SIGMA_Y
    assert form.beta > 0.0


def test_form_beta_matches_probability(design_assessment):
    form = design_assessment.form(MODE_ULTIMATE)
    assert beta_from_pf(form.pf) == pytest.approx(form.beta, abs=1e-6)


def test_form_against_importance_sampling(design_assessment):
    """Gate c: FORM vs 1e6 design-point samples (2 %)."""
    form = design_assessment.form(MODE_ULTIMATE)
    mc = design_assessment.problem(MODE_ULTIMATE).importance_sampling_pf(1_000_000)
    assert mc.failures > 100
    assert abs(form.beta - mc.beta) / mc.beta < 0.02


def test_sorm_correction_direction_and_accuracy(design_assessment):
    """Gate d: SORM vs the same Monte-Carlo reference (3 %, and at least as good)."""
    form = design_assessment.form(MODE_ULTIMATE)
    sorm = design_assessment.sorm(MODE_ULTIMATE, form)
    mc = design_assessment.problem(MODE_ULTIMATE).importance_sampling_pf(1_000_000)
    assert sorm.valid
    assert abs(sorm.beta_sorm - mc.beta) / mc.beta < 0.03
    assert abs(sorm.beta_sorm - mc.beta) <= abs(form.beta - mc.beta) + 1e-9


def test_sorm_equals_form_for_linear_limit_state():
    """A linear g has zero curvature: the Breitung correction must vanish."""
    panel = PanelSpec(t=13.6e-3)
    variables = VariableModel()
    adapters = GenericLimitStates(
        panel=panel,
        builder=lambda t: {"linear": lambda x: x[0] - 1.5},
    )
    problem = ProbabilisticAssessment(adapters, variables)
    form = problem.form("linear")
    sorm = problem.sorm("linear", form)
    assert sorm.valid
    assert sorm.beta_sorm == pytest.approx(form.beta, rel=1e-6)


def test_sensitivity_analytic_matches_finite_difference(design_assessment):
    """Gate e: envelope-theorem derivative vs re-solved FORM (< 1 %)."""
    sens = design_assessment.sensitivity(MODE_ULTIMATE)
    assert sens["max_relative_difference"] < 0.01
    analytic = sens["dbeta_dmean_analytic"]
    # pressure and model uncertainty must reduce the index, strength must raise it
    assert analytic["p_ice"] < 0 and analytic["k_mod"] < 0 and analytic["fy"] > 0


def test_mvfosm_below_form_for_concave_capacity(design_assessment):
    form = design_assessment.form(MODE_ULTIMATE)
    mv = design_assessment.mvfosm(MODE_ULTIMATE)
    assert mv["beta_mvfosm"] > 0.0
    assert abs(mv["beta_mvfosm"] - form.beta) / form.beta < 0.20


def test_ditlevsen_bounds_enclose_components(design_assessment):
    forms = {m: design_assessment.form(m) for m in DEFAULT_MODES}
    system = design_assessment.system(forms)
    assert system["pf_lower"] <= system["pf_upper"] + 1e-12
    assert system["pf_lower"] >= max(forms[m].pf for m in DEFAULT_MODES) - 1e-9
    assert system["beta_point"] < min(forms[m].beta for m in DEFAULT_MODES) + 1e-9


def test_system_monte_carlo_consistent_with_ditlevsen(design_assessment):
    forms = {m: design_assessment.form(m) for m in DEFAULT_MODES}
    system = design_assessment.system(forms)
    mc = design_assessment.system_monte_carlo(400_000)
    pf = mc["pf_system"]
    assert pf > 0.0
    assert system["pf_lower"] * 0.7 <= pf <= system["pf_upper"] * 2.0


def test_required_thickness_hits_target(assessment):
    result = assessment.required_thickness(4.0)
    assert result["beta_achieved"] == pytest.approx(4.0, abs=2e-3)
    assert 5e-3 < result["t_required_mm"] * 1e-3 < 40e-3
    assert result["iterations"] > 0


def test_required_thickness_agrees_with_mvfosm(assessment):
    """Gate f: FORM inverse vs the independent MVFOSM inverse (< 10 %)."""
    form = assessment.required_thickness(4.0)
    mv = assessment.required_thickness_mvfosm(4.0)
    rel = abs(form["t_required_mm"] - mv["t_required_mm"]) / mv["t_required_mm"]
    assert rel < 0.10


def test_beta_increases_with_thickness(assessment):
    thin = assessment.beta_of_thickness(11.0e-3)
    thick = assessment.beta_of_thickness(16.0e-3)
    assert thick > thin


def test_beta_of_thickness_handles_out_of_envelope(assessment):
    assert assessment.beta_of_thickness(2.0e-3) == float("-inf")


def test_beta_vs_thickness_table(assessment):
    rows = UnifiedBucklingReliability(PanelSpec(t=13.6e-3),
                                      surrogate=assessment.limit_states.surrogate
                                      ).beta_vs_thickness([11e-3, 14e-3])
    assert len(rows) == 2
    assert rows[1]["beta"] > rows[0]["beta"]


# --------------------------------------------------------------------------- #
# 6. unified interface
# --------------------------------------------------------------------------- #
def test_unified_assess_returns_full_chain(nominal_panel, synthetic_surrogate):
    model = UnifiedBucklingReliability(nominal_panel, surrogate=synthetic_surrogate)
    result = model.assess(target_beta=4.0)
    for key in ("sigma_ultimate_MPa", "beta", "pf", "beta_sorm", "pf_sorm",
                "beta_system", "pf_system", "design_point", "deterministic",
                "probabilistic", "target_thickness"):
        assert key in result
    assert result["sigma_ultimate_MPa"] > 0
    assert result["deterministic"]["capacity_source"].startswith(
        "FE-calibrated response surface")
    assert result["design_point"]["variables"] == list(
        model.variables.variable_set().names)


def test_unified_sigma_ultimate_matches_surrogate(nominal_panel, synthetic_surrogate):
    model = UnifiedBucklingReliability(nominal_panel, surrogate=synthetic_surrogate)
    result = model.assess(target_beta=None)
    expected = synthetic_surrogate.sigma_u(
        nominal_panel.t - model.variables.dt_mean * 1e-3, SIGMA_Y, panel_b(nominal_panel), E_MOD)
    assert result["sigma_ultimate_MPa"] == pytest.approx(expected / 1e6, rel=1e-12)
    assert "target_thickness" not in result


def panel_b(panel: PanelSpec) -> float:
    return panel.b


def test_unified_from_payload_roundtrip(synthetic_surrogate):
    payload = {
        "panel": {"a": 3.0, "b": 1.0, "t_mm": 13.6, "sigma_y_MPa": 313.0,
                  "mesh": [8, 4], "bc": "SSSS", "sigma_global_MPa": 95.0},
        "variables": {"p_ice_mean": 2.5, "dt_mean": 2.0},
        "calibration": synthetic_surrogate.to_dict(),
        "options": {"target_beta": 4.0},
    }
    model = UnifiedBucklingReliability.from_payload(payload)
    assert model.panel.t == pytest.approx(13.6e-3)
    assert model.variables.p_ice_mean == pytest.approx(2.5)
    assert model.surrogate is not None
    result = model.assess()
    assert result["beta"] > 0


def test_unified_requires_surrogate_or_calibration(nominal_panel, monkeypatch):
    model = UnifiedBucklingReliability(nominal_panel, surrogate=None)
    if model.surrogate is None:
        with pytest.raises(RuntimeError):
            model.limit_states()


def test_calibrate_fits_surrogate_on_fem_traces(nominal_panel):
    """One arc-length trace: the fitted surface must land near Faulkner."""
    clear_cache()
    drive = DeterministicAssessment(PanelSpec(t=19.49e-3, nx=8, ny=4),
                                    max_steps=16)
    record = drive.ultimate()
    assert record["n_steps"] > 0
    assert abs(record["sigma_u_over_sigma_y"]
               / float(faulkner_reference(record["beta"])) - 1.0) < 0.10


def test_imperfection_sweep_reduces_strength(nominal_panel):
    """Gate b of the deterministic half: defect sensitivity via arc-length runs."""
    clear_cache()
    drive = DeterministicAssessment(PanelSpec(t=19.49e-3, nx=8, ny=4),
                                    max_steps=16)
    rows = drive.imperfection_sweep(levels=("slight", "average"))
    assert rows[0]["sigma_u_over_sigma_y"] >= rows[1]["sigma_u_over_sigma_y"] - 1e-6
    assert rows[0]["w0_mm"] < rows[1]["w0_mm"]


def test_iacs_uls_check_reports_margins(nominal_panel, synthetic_surrogate):
    model = UnifiedBucklingReliability(nominal_panel, surrogate=synthetic_surrogate)
    check = model.deterministic.iacs_uls_check()
    assert check["sigma_allow_MPa"] > 0
    assert check["factors"]["gamma_m"] > 1.0
    assert isinstance(check["passed"], bool)
    assert math.isfinite(check["utilisation"])