"""Verification of FORM / SORM / simulation / system bounds against closed-form values."""
from __future__ import annotations

from math import exp, sqrt

import pytest

from tensorphysics.structural import (
    ExponentialDistribution,
    LognormalDistribution,
    NatafTransform,
    NormalDistribution,
    ReliabilityError,
    ReliabilityProblem,
    ditlevsen_bounds,
    monte_carlo_pf,
    norm_cdf,
    norm_pdf,
)
from tensorphysics.structural._linalg import jacobi_eigen


def _simpson(f, a: float, b: float, n: int = 4000) -> float:
    if n % 2:
        n += 1
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        total += (4.0 if i % 2 else 2.0) * f(a + i * h)
    return total * h / 3.0


def _beta_from_pf(pf: float) -> float:
    from tensorphysics.structural import norm_ppf

    return -norm_ppf(pf)


# --------------------------------------------------------------------------- #
# Criterion (a): closed-form limit states
# --------------------------------------------------------------------------- #
def test_form_linear_limit_state_matches_analytic_beta():
    """g = R - S with both normal: beta = (mu_R - mu_S)/sqrt(sigma_R^2 + sigma_S^2)."""
    transform = NatafTransform(
        [NormalDistribution(400.0, 40.0), NormalDistribution(200.0, 30.0)]
    )
    problem = ReliabilityProblem(transform, lambda x: x[0] - x[1], name="linear")
    result = problem.form()
    reference = 200.0 / sqrt(40.0**2 + 30.0**2)
    assert reference == pytest.approx(4.0, rel=1e-12)
    assert result.converged
    assert result.beta == pytest.approx(reference, rel=1e-9)
    assert result.pf == pytest.approx(norm_cdf(-reference), rel=1e-12)


def test_form_exponential_limit_state_matches_analytic_beta():
    """g = exp(2) - exp(X) with X ~ N(1, 0.5): failure iff X > 2, so beta = 2 exactly.

    The limit state is strongly nonlinear in X, yet the exact reliability index is
    known in closed form because the failure boundary is a single point.
    """
    from math import exp

    transform = NatafTransform([NormalDistribution(1.0, 0.5)])
    problem = ReliabilityProblem(transform, lambda x: exp(2.0) - exp(x[0]), name="exp")
    result = problem.form()
    assert result.converged
    assert result.beta == pytest.approx(2.0, rel=5e-3)
    assert result.design_point_x[0] == pytest.approx(2.0, rel=1e-4)


def test_form_exponential_marginal_matches_quadrature():
    """g = R - E, R ~ N(3, 0.3), E ~ Exp(1): Pf = int exp(-e) Phi((e-3)/0.3) de."""
    transform = NatafTransform([NormalDistribution(3.0, 0.3), ExponentialDistribution(1.0)])
    problem = ReliabilityProblem(transform, lambda x: x[0] - x[1], name="exp-load")
    result = problem.form()
    # exponential pdf (rate 1) times the conditional failure probability of R
    reference_pf = _simpson(
        lambda e: exp(-e) * norm_cdf((e - 3.0) / 0.3), 0.0, 30.0, 12000
    )
    reference_beta = _beta_from_pf(reference_pf)
    assert result.converged
    assert abs(result.beta - reference_beta) / reference_beta < 5e-3


# --------------------------------------------------------------------------- #
# Criterion (b): FORM vs Monte Carlo
# --------------------------------------------------------------------------- #
def test_form_matches_monte_carlo_within_two_percent():
    transform = NatafTransform(
        [NormalDistribution(300.0, 40.0), NormalDistribution(200.0, 30.0)]
    )
    problem = ReliabilityProblem(transform, lambda x: x[0] - x[1], name="mc-linear")
    form = problem.form()
    mc = problem.monte_carlo_pf(samples=1_000_000, seed=2026)
    assert mc.samples == 1_000_000
    assert abs(form.beta - mc.beta) / mc.beta < 0.02
    assert form.beta == pytest.approx(2.0, rel=1e-6)


def test_monte_carlo_is_reproducible_and_seed_sensitive():
    transform = NatafTransform([NormalDistribution(0.0, 1.0)])
    problem = ReliabilityProblem(transform, lambda x: 2.0 - x[0], name="repro")
    a = problem.monte_carlo_pf(samples=50_000, seed=7)
    b = problem.monte_carlo_pf(samples=50_000, seed=7)
    c = problem.monte_carlo_pf(samples=50_000, seed=8)
    assert a.pf == b.pf
    assert a.pf != c.pf
    assert abs(a.pf - c.pf) / a.pf < 0.2


def test_monte_carlo_stdlib_and_numpy_paths_agree():
    pytest.importorskip("numpy")
    transform = NatafTransform([NormalDistribution(0.0, 1.0), NormalDistribution(0.0, 1.0)])
    problem = ReliabilityProblem(transform, lambda x: 2.0 - x[0] - 0.5 * x[1], name="paths")
    fast = problem.monte_carlo_pf(samples=200_000, seed=3, backend="numpy")
    slow = problem.monte_carlo_pf(samples=200_000, seed=3, backend="stdlib")
    assert fast.pf == pytest.approx(slow.pf, rel=0.05)
    assert fast.beta == pytest.approx(slow.beta, rel=0.02)


# --------------------------------------------------------------------------- #
# Criterion (c): SORM on a strongly nonlinear surface
# --------------------------------------------------------------------------- #
PARABOLIC_A = 3.0
PARABOLIC_C = 1.0


def _parabolic_reference_pf(a: float = PARABOLIC_A, c: float = PARABOLIC_C) -> float:
    """Pf for g = a - u1 + c (u2^2 - 1) by 1-D quadrature over the u2 marginal."""
    return _simpson(
        lambda u2: norm_pdf(u2) * norm_cdf(-(a + c * (u2 * u2 - 1.0))),
        -10.0,
        10.0,
        20000,
    )


def _parabolic_batch(a: float, c: float):
    numpy = pytest.importorskip("numpy")

    def batch(x):
        array = numpy.asarray(x, dtype=float)
        return a - array[:, 0] + c * (array[:, 1] ** 2 - 1.0)

    return batch


def test_sorm_curvature_reproduces_parabola_curvature():
    transform = NatafTransform([NormalDistribution(0.0, 1.0), NormalDistribution(0.0, 1.0)])
    problem = ReliabilityProblem(
        transform,
        lambda x: PARABOLIC_A - x[0] + PARABOLIC_C * (x[1] ** 2 - 1.0),
        name="parabola",
    )
    form = problem.form()
    sorm = problem.sorm(form)
    assert sorm.valid
    # d2g/du2^2 = 2c and |grad g| = 1 at the design point -> kappa = 2c
    assert sorm.principal_curvatures[-1] == pytest.approx(2.0 * PARABOLIC_C, rel=2e-3)
    assert form.beta == pytest.approx(PARABOLIC_A - PARABOLIC_C, rel=1e-6)


def test_sorm_improves_on_form_and_matches_monte_carlo():
    transform = NatafTransform([NormalDistribution(0.0, 1.0), NormalDistribution(0.0, 1.0)])
    problem = ReliabilityProblem(
        transform,
        lambda x: PARABOLIC_A - x[0] + PARABOLIC_C * (x[1] ** 2 - 1.0),
        name="parabola",
        batch_limit_state=_parabolic_batch(PARABOLIC_A, PARABOLIC_C),
    )
    form = problem.form()
    sorm = problem.sorm(form)
    mc = problem.monte_carlo_pf(samples=1_000_000, seed=2026)
    reference_beta = _beta_from_pf(_parabolic_reference_pf())
    form_error = abs(form.beta - reference_beta) / reference_beta
    sorm_error = abs(sorm.beta_sorm - reference_beta) / reference_beta
    assert abs(mc.beta - reference_beta) / reference_beta < 0.03
    assert form_error > 0.03          # the case is genuinely nonlinear
    assert sorm_error < 0.03          # criterion: SORM within 3%
    assert sorm_error < form_error    # criterion: SORM better than FORM
    assert abs(sorm.beta_sorm - mc.beta) / mc.beta < 0.03


def test_sorm_reports_invalid_when_curvature_factor_is_zero():
    """A spherical limit state g = R - |u| has curvature -1/R, so 1 + beta*kappa = 0."""
    transform = NatafTransform([NormalDistribution(0.0, 1.0), NormalDistribution(0.0, 1.0)])
    problem = ReliabilityProblem(
        transform, lambda x: 3.0 - sqrt(x[0] ** 2 + x[1] ** 2), name="sphere"
    )
    # |u| has a cusp at the origin, so a non-degenerate start point is required
    form = problem.form(start_u=[1.0, 0.0])
    assert form.beta == pytest.approx(3.0, rel=1e-5)
    sorm = problem.sorm(form)
    assert sorm.principal_curvatures[-1] == pytest.approx(-1.0 / 3.0, rel=2e-2)
    assert not sorm.valid
    assert "not applicable" in sorm.note


# --------------------------------------------------------------------------- #
# Design point / sensitivities
# --------------------------------------------------------------------------- #
def test_design_point_satisfies_kkt_conditions():
    transform = NatafTransform(
        [NormalDistribution(400.0, 40.0), NormalDistribution(200.0, 30.0)]
    )
    problem = ReliabilityProblem(
        transform, lambda x: x[0] - x[1] + 0.02 * (x[0] - 300.0) ** 2 / 100.0, name="kkt"
    )
    form = problem.form()
    assert abs(form.g_design) < 1e-6
    alpha_norm = sqrt(sum(a * a for a in form.alpha))
    assert alpha_norm == pytest.approx(1.0, rel=1e-10)
    # at the Hasofer-Lind point u* = beta * alpha, with alpha = -grad g/|grad g|
    for ui, ai in zip(form.design_point_u, form.alpha):
        assert ui == pytest.approx(form.beta * ai, abs=1e-6)
    assert form.beta == pytest.approx(sqrt(sum(u * u for u in form.design_point_u)), rel=1e-12)


def test_median_point_in_failure_domain_is_rejected():
    transform = NatafTransform([NormalDistribution(0.0, 1.0)])
    problem = ReliabilityProblem(transform, lambda x: x[0] - 1.0, name="broken")
    with pytest.raises(ReliabilityError):
        problem.form()


def test_sensitivity_matches_closed_form_for_linear_limit_state():
    transform = NatafTransform(
        [NormalDistribution(300.0, 40.0), NormalDistribution(200.0, 30.0)]
    )
    problem = ReliabilityProblem(transform, lambda x: x[0] - x[1], name="sens")
    form = problem.form()
    table = problem.design_point_sensitivity(form)
    finite = problem.sensitivity_finite_difference(form, relative_step=1e-4)
    # closed form: d beta / d mu_R = 1/sqrt(sR^2 + sS^2), d beta / d sigma_R = -dmu*sR/(s^2)^1.5
    assert table[("X1", "mu")] == pytest.approx(1.0 / 50.0, rel=1e-9)
    assert table[("X2", "mu")] == pytest.approx(-1.0 / 50.0, rel=1e-9)
    assert table[("X1", "sigma")] == pytest.approx(-100.0 * 40.0 / 125000.0, rel=1e-6)
    for key, value in table.items():
        assert abs(value - finite[key]) / max(abs(finite[key]), 1e-12) < 0.01


def test_sensitivity_of_lognormal_parameters_matches_finite_difference():
    transform = NatafTransform(
        [NormalDistribution(300.0, 40.0), LognormalDistribution(5.3, 0.1)]
    )
    problem = ReliabilityProblem(transform, lambda x: x[0] - x[1], name="sens-ln")
    form = problem.form()
    analytic = problem.design_point_sensitivity(form)
    finite = problem.sensitivity_finite_difference(form, relative_step=1e-4)
    for key in (("X2", "mu_ln"), ("X2", "sigma_ln")):
        assert abs(analytic[key] - finite[key]) / abs(finite[key]) < 0.01


# --------------------------------------------------------------------------- #
# System bounds
# --------------------------------------------------------------------------- #
def test_ditlevsen_bounds_are_ordered_and_inside_sum():
    transform = NatafTransform([NormalDistribution(0.0, 1.0), NormalDistribution(0.0, 1.0)])
    problems = [
        ReliabilityProblem(transform, lambda x: 2.5 - x[0] - 0.3 * x[1], name="g1"),
        ReliabilityProblem(transform, lambda x: 2.2 - 0.4 * x[0] - x[1], name="g2"),
        ReliabilityProblem(transform, lambda x: 3.0 - x[0], name="g3"),
    ]
    results = [p.form() for p in problems]
    bounds = ditlevsen_bounds(results)
    assert bounds.pf_lower <= bounds.pf_upper
    assert bounds.pf_lower >= max(r.pf for r in results) - 1e-15
    assert bounds.pf_upper <= sum(r.pf for r in results) + 1e-15
    assert bounds.beta_lower <= bounds.beta_upper  # lower beta bound <-> upper Pf bound
    # identical limit states collapse the bounds onto the single component value
    duplicate = ditlevsen_bounds([results[2], results[2]])
    assert duplicate.pf_lower == pytest.approx(results[2].pf, rel=1e-2)
    assert duplicate.pf_upper == pytest.approx(results[2].pf, rel=1e-2)
    assert abs(duplicate.pf_upper - duplicate.pf_lower) / results[2].pf < 1e-2


def test_system_bounds_bracket_simulated_union_probability():
    numpy = pytest.importorskip("numpy")
    from tensorphysics.structural._batch import batch_u_to_x

    transform = NatafTransform(
        [NormalDistribution(0.0, 1.0), NormalDistribution(0.0, 1.0), NormalDistribution(0.0, 1.0)]
    )
    states = [
        lambda x: 2.6 - x[0] - 0.2 * x[1],
        lambda x: 2.4 - 0.3 * x[1] - x[2],
        lambda x: 2.9 - x[0] - 0.4 * x[2],
    ]
    results = [
        ReliabilityProblem(transform, g, name=f"g{i}").form() for i, g in enumerate(states)
    ]
    bounds = ditlevsen_bounds(results)
    rng = numpy.random.default_rng(4242)
    x = batch_u_to_x(transform, rng.standard_normal((400_000, 3)))
    batched = [
        lambda a: 2.6 - a[:, 0] - 0.2 * a[:, 1],
        lambda a: 2.4 - 0.3 * a[:, 1] - a[:, 2],
        lambda a: 2.9 - a[:, 0] - 0.4 * a[:, 2],
    ]
    union = numpy.zeros(len(x), dtype=bool)
    for g in batched:
        union |= g(x) < 0.0
    pf_sim = float(union.mean())
    assert bounds.pf_lower * 0.9 <= pf_sim <= bounds.pf_upper * 1.1


def test_importance_sampling_reproduces_form_for_tail_probability():
    transform = NatafTransform([NormalDistribution(0.0, 1.0), NormalDistribution(0.0, 1.0)])
    problem = ReliabilityProblem(transform, lambda x: 4.0 - x[0] - 0.3 * x[1], name="tail")
    form = problem.form()
    assert form.beta > 3.5
    isr = problem.importance_sampling_pf(samples=200_000, seed=99, form_result=form)
    assert abs(isr.beta - form.beta) / form.beta < 0.03


def test_module_level_convenience_functions():
    transform = NatafTransform([NormalDistribution(0.0, 1.0)])
    result = monte_carlo_pf(transform, lambda x: 2.0 - x[0], samples=100_000, seed=5)
    assert result.beta == pytest.approx(2.0, abs=0.1)


def test_jacobi_eigen_decomposition_reference():
    matrix = [
        [4.0, 1.0, 0.0],
        [1.0, 3.0, 1.0],
        [0.0, 1.0, 2.0],
    ]
    values, vectors = jacobi_eigen(matrix)
    numpy = pytest.importorskip("numpy")
    reference = sorted(numpy.linalg.eigvalsh(numpy.asarray(matrix)).tolist())
    for value, ref in zip(values, reference):
        assert value == pytest.approx(ref, abs=1e-10)
    for value, vector in zip(values, vectors):
        applied = [sum(matrix[i][j] * vector[j] for j in range(3)) for i in range(3)]
        for i in range(3):
            assert applied[i] == pytest.approx(value * vector[i], abs=1e-10)