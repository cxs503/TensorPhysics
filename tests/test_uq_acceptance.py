"""Acceptance criteria of the UQ node, expressed as executable checks.

The six machine-checkable criteria of the campaign node map one-to-one onto the
tests below (the numbering follows the mission statement):

``C1``  Ishigami ``S1/S2/ST`` absolute error <= 1e-3 (PCE) and Monte-Carlo
        relative error <= 5 %;
``C2``  ``eps_LOO`` decreases monotonically as the PCE degree grows
        (Spearman <= -0.9), and as the sample count grows in the regime where
        regression noise dominates;
``C3``  same seed + same sample set => bit-identical Sobol' indices;
``C4``  NaN in the samples / too few samples / ``eps_LOO > 0.1`` => WITHHELD;
``C5``  PCE analytical indices vs Saltelli estimate within 5 %;
``C6``  end-to-end hand-off into the ASME V&V 20 validation workflow.
"""
from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.uq import (
    FAIL,
    PASS,
    WITHHELD,
    CallableAdapter,
    InputModel,
    UQOptions,
    UQPolicy,
    UQStudy,
    Uniform,
    attach_to_validation,
    compare_indices,
    saltelli_indices,
    sample_input_model,
    spearman_rank_correlation,
)
from tensorphysics.uq.benchmarks import (
    g_function,
    g_function_analytic_indices,
    g_function_input_model,
    ishigami_analytic_indices,
    ishigami_function,
    ishigami_input_model,
)
from tensorphysics.uq.pce import fit_pce, fit_pce_sweep


# ---------------------------------------------------------------------------
# C1: PCE indices against the exact Ishigami decomposition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [12345, 77])
def test_c1_pce_indices_match_analytic_within_1e_3(seed: int) -> None:
    model = ishigami_input_model()
    adapter = CallableAdapter(ishigami_function, model.names, name="ishigami")
    result = UQStudy(
        model,
        adapter,
        UQOptions(
            seed=seed,
            n_samples=2048,
            degrees=(2, 3, 4, 5, 6, 8, 10, 12),
            cross_check=False,
            n_draws_ci=200,
        ),
    ).run()
    exact = ishigami_analytic_indices()
    s1 = result.indices
    assert s1 is not None
    assert np.max(np.abs(s1.S1 - exact.S1)) <= 1e-3
    assert np.max(np.abs(s1.ST - exact.ST)) <= 1e-3
    # the closed second-order index of (x1, x3)
    assert s1.S2["x1-x3"] == pytest.approx(exact.S2["x1-x3"], abs=1e-3)
    # x3 must come out with zero first-order effect, which a low-order or
    # badly truncated basis cannot reproduce
    assert abs(s1.S1[2]) <= 1e-3
    # the surrogate resolves the variance
    assert result.fit.loo_error <= 1e-3
    # the gate certifies it (cross-check disabled -> WITHHELD by policy)
    assert result.verdict.status == WITHHELD
    assert "CROSS_CHECK_MISSING" in result.verdict.reasons


def test_c1_mc_saltelli_relative_error_within_5_percent() -> None:
    model = ishigami_input_model()
    exact = ishigami_analytic_indices()
    mc, _ = saltelli_indices(
        model,
        ishigami_function,
        n_base=8192,
        seed=2024,
        n_boot=200,
    )
    cmp_ = compare_indices(exact, mc, abs_floor=1e-3)
    assert cmp_.max_rel_dev_S1_defined <= 0.05
    assert cmp_.max_rel_dev_ST_defined <= 0.05
    # the analytically-zero first-order index of x3 stays small in absolute terms
    assert abs(mc.S1[2]) <= 0.02


# ---------------------------------------------------------------------------
# C2: monotone decrease of eps_LOO
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [12345, 77])
def test_c2_loo_decreases_with_degree(seed: int) -> None:
    model = ishigami_input_model()
    samples = sample_input_model(model, 2048, method="sobol", seed=seed)
    scores = model.standard_scores(samples.unit, "legendre")
    sweep = fit_pce_sweep(scores, ishigami_function(samples.physical), names=model.names,
                          degrees=(2, 3, 4, 5, 6, 8, 10, 12))
    degrees = [r.degree for r in sweep.records]
    loos = [r.loo_error for r in sweep.records]
    rho = spearman_rank_correlation(degrees, loos)
    assert rho <= -0.9, (degrees, loos, rho)
    # and the best model is the most accurate one
    assert sweep.best.degree >= 8


def test_c2_loo_versus_sample_count_is_characterised_not_assumed() -> None:
    """The sample-count half of criterion C2 is *falsified* and pinned here.

    Measured behaviour of the greedy + LOO construction (see
    ``docs/UQ_PCE_SOBOL_GATE.md`` section "where the criterion does not hold"):
    as the sample count grows at a fixed degree, ``eps_LOO`` does **not**
    decrease monotonically.  It converges to the truncation/noise floor from
    *below*, because the minimum-over-path selection becomes less optimistic
    with more samples.  This test records the measured shape so that a future
    change of the estimator cannot silently "fix" the numbers: the sample budget
    is therefore enforced a priori by the policy floor ``n >= 2 * basis_size``,
    never by trusting ``eps_LOO`` alone.

    The model is a degree-2 polynomial (exactly representable) plus Gaussian
    "solver noise" of 0.05, i.e. the pure regression-noise regime.
    """
    model = InputModel(["a", "b"], [Uniform(-1, 1), Uniform(-1, 1)])

    def model_with_noise(X: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(7)
        return (
            1.0
            + X[:, 0]
            + 0.5 * X[:, 0] * X[:, 1]
            + 2.0 * X[:, 1] ** 2
            + rng.normal(0.0, 0.05, size=X.shape[0])
        )

    sample_counts = [128, 256, 512, 1024, 2048]
    loos = []
    for n in sample_counts:
        samples = sample_input_model(model, n, method="sobol", seed=31)
        scores = model.standard_scores(samples.unit, "legendre")
        fit = fit_pce(scores, model_with_noise(samples.physical), names=model.names, degree=2)
        loos.append(fit.loo_error)
    # eps_LOO approaches noise^2 / Var(y) ~= 0.0025 / 0.717 = 3.5e-3 from below
    assert all(2e-3 <= v <= 4e-3 for v in loos), loos
    assert loos[0] < loos[-1]  # optimistic at small n, converged at large n
    rho = spearman_rank_correlation(sample_counts, loos)
    assert rho > 0.5, (sample_counts, loos, rho)


# ---------------------------------------------------------------------------
# C3: reproducibility
# ---------------------------------------------------------------------------


def test_c3_same_seed_gives_bit_identical_indices() -> None:
    model = ishigami_input_model()
    adapter = CallableAdapter(ishigami_function, model.names)
    options = UQOptions(
        seed=424242,
        n_samples=1024,
        degrees=(4, 6, 8),
        cross_check=True,
        cross_check_base=512,
        cross_check_boot=100,
        n_draws_ci=200,
    )
    first = UQStudy(model, adapter, options).run()
    second = UQStudy(model, adapter, options).run()
    assert first.samples.digest() == second.samples.digest()
    assert np.array_equal(first.indices.S1, second.indices.S1)
    assert np.array_equal(first.indices.ST, second.indices.ST)
    assert first.indices.S1_low.tolist() == second.indices.S1_low.tolist()
    assert first.cross_check.meta["design_digest_sha256"] == second.cross_check.meta["design_digest_sha256"]
    assert np.array_equal(first.cross_check.ST, second.cross_check.ST)

    other = UQStudy(model, adapter, UQOptions(**{**options.__dict__, "seed": 424243})).run()
    assert other.samples.digest() != first.samples.digest()
    # a different seed must still land on the same physics within 1 %
    assert np.max(np.abs(other.indices.S1 - first.indices.S1)) <= 0.01


def test_c3_sample_digest_is_independent_of_downstream_work() -> None:
    model = ishigami_input_model()
    a = sample_input_model(model, 512, seed=9)
    b = sample_input_model(model, 512, seed=9)
    assert a.digest() == b.digest()
    c = sample_input_model(model, 513, seed=9)
    assert c.digest() != a.digest()


# ---------------------------------------------------------------------------
# C4: WITHHELD states
# ---------------------------------------------------------------------------


def test_c4_nan_in_model_output_yields_withheld() -> None:
    model = ishigami_input_model()

    def broken(X: np.ndarray) -> np.ndarray:
        y = ishigami_function(X)
        y[3] = np.nan  # one dropped sample, as a solver would leave behind
        return y

    result = UQStudy(model, CallableAdapter(broken, model.names), UQOptions(n_samples=512, degrees=(4,))).run()
    assert result.verdict.status == WITHHELD
    assert "NONFINITE_MODEL_OUTPUT" in result.verdict.reasons
    assert result.verdict.diagnostics["indices"]["estimator"] == "none"


def test_c4_too_few_samples_yields_withheld() -> None:
    model = ishigami_input_model()
    adapter = CallableAdapter(ishigami_function, model.names)
    result = UQStudy(
        model,
        adapter,
        UQOptions(n_samples=48, degrees=(6,), cross_check=False, policy=UQPolicy(min_samples=64)),
    ).run()
    assert result.verdict.status == WITHHELD
    assert "INSUFFICIENT_SAMPLES" in result.verdict.reasons


def test_c4_insufficient_oversampling_yields_withheld() -> None:
    """n above the absolute floor but below 2 x basis_size must still withhold."""
    model = ishigami_input_model()
    adapter = CallableAdapter(ishigami_function, model.names)
    result = UQStudy(
        model,
        adapter,
        UQOptions(n_samples=64, degrees=(8,), cross_check=False, min_oversampling=1.5),
    ).run()
    assert result.verdict.status == WITHHELD
    assert "INSUFFICIENT_SAMPLES" in result.verdict.reasons


def test_c4_high_loo_error_yields_withheld() -> None:
    model = ishigami_input_model()
    adapter = CallableAdapter(ishigami_function, model.names)
    result = UQStudy(
        model,
        adapter,
        UQOptions(n_samples=1024, degrees=(1, 2), cross_check=False),
    ).run()
    assert result.fit is not None and result.fit.loo_error > 0.1
    assert result.verdict.status == WITHHELD
    assert "LOO_ERROR_ABOVE_THRESHOLD" in result.verdict.reasons


def test_c4_adapter_failures_yield_withheld() -> None:
    model = ishigami_input_model()

    def fragile(X: np.ndarray) -> np.ndarray:
        if np.any(X[:, 0] > 3.0):
            raise RuntimeError("solver diverged outside its validity domain")
        return ishigami_function(X)

    result = UQStudy(model, CallableAdapter(fragile, model.names), UQOptions(n_samples=256, degrees=(4,), cross_check=False)).run()
    assert result.verdict.status == WITHHELD
    assert "NONFINITE_MODEL_OUTPUT" in result.verdict.reasons
    assert "ADAPTER_FAILURES" in result.verdict.reasons


# ---------------------------------------------------------------------------
# C5: PCE vs Saltelli agreement
# ---------------------------------------------------------------------------


def test_c5_pce_and_saltelli_agree_within_5_percent() -> None:
    model = ishigami_input_model()
    adapter = CallableAdapter(ishigami_function, model.names)
    result = UQStudy(
        model,
        adapter,
        UQOptions(
            n_samples=2048,
            degrees=(4, 5, 6, 8, 10, 12),
            cross_check=True,
            cross_check_base=4096,
            cross_check_boot=300,
            n_draws_ci=300,
        ),
    ).run()
    assert result.indices is not None and result.cross_check is not None
    cmp_ = compare_indices(result.indices, result.cross_check, abs_floor=1e-3)
    assert cmp_.max_rel_dev_S1_defined <= 0.05
    assert cmp_.max_rel_dev_ST_defined <= 0.05
    assert cmp_.n_defined >= 3
    assert result.verdict.status == PASS, result.verdict.messages


def test_c5_small_cross_check_budget_is_inconclusive_not_passing() -> None:
    """A noisy cross-check must not be reported as agreement *or* as failure."""
    model = ishigami_input_model()
    adapter = CallableAdapter(ishigami_function, model.names)
    result = UQStudy(
        model,
        adapter,
        UQOptions(n_samples=2048, degrees=(4, 6, 8), cross_check=True, cross_check_base=256, cross_check_boot=200),
    ).run()
    assert result.verdict.status == WITHHELD
    assert "CROSS_CHECK_INCONCLUSIVE" in result.verdict.reasons


# ---------------------------------------------------------------------------
# C6: hand-off into the existing V&V workflow
# ---------------------------------------------------------------------------


def test_c6_validation_bundle_uses_certified_uq_evidence() -> None:
    model = ishigami_input_model()
    adapter = CallableAdapter(ishigami_function, model.names)
    result = UQStudy(
        model,
        adapter,
        UQOptions(n_samples=2048, degrees=(6, 8, 10), cross_check_base=2048, cross_check_boot=200),
    ).run()
    assert result.verdict.status == PASS, result.verdict.messages

    nominal = ishigami_function(np.zeros((1, 3)))[0]
    bundle = attach_to_validation(
        result,
        f_sim=float(nominal),
        f_exp=float(nominal + 0.5),
        u_num=0.2,
        label="ishigami nominal vs offset reference",
    )
    assert bundle["status"] in {"PASS", "FAIL"}
    assert bundle["u_input"] == pytest.approx(float(np.sqrt(result.indices.variance)), rel=1e-12)
    assert bundle["u_val"] == pytest.approx(float(np.hypot(0.2, bundle["u_input"])), rel=1e-12)


def test_c6_withheld_uq_blocks_the_validation_gate() -> None:
    model = ishigami_input_model()

    def broken(X: np.ndarray) -> np.ndarray:
        y = ishigami_function(X)
        y[0] = np.nan
        return y

    result = UQStudy(model, CallableAdapter(broken, model.names), UQOptions(n_samples=512, degrees=(4,))).run()
    assert result.verdict.status == WITHHELD
    bundle = attach_to_validation(result, f_sim=0.0, f_exp=1.0, u_num=0.1)
    assert bundle["status"] == WITHHELD
    assert "U_input is unknown" in bundle["message"]


# ---------------------------------------------------------------------------
# bonus: the g-function (8 inputs, effective dimension ~3) shows why the sparse
# support matters -- a full tensor basis at degree 4 would have 495 terms.
# ---------------------------------------------------------------------------


def test_g_function_8d_sparse_pce_recovers_analytic_indices() -> None:
    a = (0.0, 1.0, 4.5, 9.0, 99.0, 99.0, 99.0, 99.0)
    model = g_function_input_model(a)
    adapter = CallableAdapter(lambda X: g_function(X, a), model.names, name="g-function")
    result = UQStudy(
        model,
        adapter,
        UQOptions(n_samples=4096, degrees=(2, 3, 4, 6), cross_check=False, n_draws_ci=100),
    ).run()
    exact = g_function_analytic_indices(a)
    assert result.indices is not None
    # the |4x-2| kink limits the convergence to algebraic order, so PCE at degree
    # 6 cannot reach the 1e-3 of the smooth Ishigami case; 1e-2 is the measured
    # envelope (5.8e-3 for S1 at n=4096)
    assert np.max(np.abs(result.indices.S1 - exact.S1)) <= 1e-2
    assert np.max(np.abs(result.indices.ST - exact.ST)) <= 1e-2
    # only 8 inputs, effective dimension ~3 -> the sparse support must stay small
    assert result.fit.n_terms <= 60
    assert result.fit.loo_error <= 5e-2