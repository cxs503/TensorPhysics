"""Tests of the refinement study (IN145-N08 criterion C2) and its mechanism.

The central claim is nuanced and therefore has to be pinned from both sides:

* at a **fixed truncation** the estimation error must fall monotonically with
  the sample count, for both ``eps_LOO`` and the independent holdout error;
* at a **fixed degree** the truncation error must fall monotonically as the
  degree grows;
* under **adaptive sparse selection** ``eps_LOO`` is *not* monotone in the
  sample count while the holdout error is -- the literal "eps_LOO decreases with
  N" claim is false there, and the module must say so instead of hiding it.
"""
from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.uq import ishigami, ishigami_spec
from tensorphysics.uq.refinement import (
    adaptive_selection_diagnostic,
    holdout_relative_rmse,
    monotone_decrease,
    refinement_study,
    spearman_rank,
)


def test_spearman_rank_matches_known_values() -> None:
    assert spearman_rank([1, 2, 3, 4], [10, 9, 8, 7]) == pytest.approx(-1.0)
    assert spearman_rank([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert np.isnan(spearman_rank([1, 2, 3, 4], [5, 5, 5, 5]))
    # ties receive the average rank: the rank correlation is still defined
    assert spearman_rank([1, 2, 3, 4], [4, 4, 2, 1]) < -0.8
    assert spearman_rank([1, 2, 3, 4], [1, 1, 2, 3]) > 0.8


def test_monotone_decrease_needs_four_points() -> None:
    assert monotone_decrease([4.0, 3.0, 2.0, 1.0])
    assert not monotone_decrease([4.0, 3.0, 2.0])  # only three points
    assert not monotone_decrease([1.0, 2.0, 3.0, 4.0])
    assert not monotone_decrease([1.0, float("nan"), 0.5, 0.1])


def test_refinement_axes_are_monotone_where_the_claim_is_well_posed() -> None:
    study = refinement_study(
        ishigami_spec(),
        ishigami,
        benchmark="ishigami",
        degrees=(3, 4, 6, 8, 10),
        degree_n_samples=512,
        truncation_degree=3,
        adaptive_n_samples=(128, 256, 512),
        holdout_n=4000,
    )
    estimation = study.axes["samples_at_fixed_truncation"]
    assert len(estimation.points) >= 4
    assert estimation.loo_monotone, [p.eps_loo for p in estimation.points]
    assert estimation.holdout_monotone, [p.holdout_rel_rmse for p in estimation.points]
    assert estimation.spearman_loo <= -0.9
    assert study.estimation_error_monotone

    truncation = study.axes["degree_at_fixed_n"]
    assert len(truncation.points) >= 3
    assert truncation.loo_monotone
    assert truncation.holdout_monotone
    assert study.truncation_error_monotone


def test_adaptive_selection_falsifies_the_literal_claim() -> None:
    study = refinement_study(
        ishigami_spec(),
        ishigami,
        degrees=(3, 4, 6, 8),
        degree_n_samples=512,
        truncation_degree=3,
        adaptive_degree=8,
        adaptive_n_samples=(128, 256, 512, 1024),
        holdout_n=8000,
    )
    adaptive = study.axes["samples_with_adaptive_selection"]
    # the surrogate genuinely improves ...
    assert adaptive.holdout_monotone
    assert adaptive.spearman_holdout <= -0.9
    # ... while eps_LOO does not fall, because the selection optimism decays
    assert not adaptive.loo_monotone
    assert adaptive.spearman_loo > -0.9
    assert study.adaptive_loo_nonmonotone
    diagnostic = adaptive_selection_diagnostic(study)
    assert diagnostic["spearman_loo"] > diagnostic["spearman_holdout"]
    assert "NOT monotone" in diagnostic["statement"]
    # the optimism is largest at the smallest budget
    assert adaptive.points[0].loo_optimism > adaptive.points[-1].loo_optimism


def test_holdout_error_is_an_independent_measurement() -> None:
    from tensorphysics.uq import fit_pce, sample

    spec = ishigami_spec()
    design = sample(spec, 512, seed=20240)
    points, coords = spec.from_latent(design.latent)
    surrogate = fit_pce(coords, ishigami(points), degree=8, families=spec.families)
    holdout = holdout_relative_rmse(surrogate, spec, ishigami, n_holdout=4000)
    assert 0.0 < holdout < 0.1
    # a fresh seed changes the estimate only by Monte-Carlo noise
    other = holdout_relative_rmse(surrogate, spec, ishigami, n_holdout=4000, seed=1234)
    assert abs(holdout - other) < 0.05 * max(holdout, other)


def test_refinement_study_is_serialisable_and_validates_inputs() -> None:
    study = refinement_study(
        ishigami_spec(),
        ishigami,
        degrees=(3, 4, 6),
        degree_n_samples=512,
        truncation_degree=3,
        adaptive_n_samples=(128, 256),
        holdout_n=2000,
    )
    payload = study.as_dict()
    assert set(payload["axes"]) == {
        "samples_at_fixed_truncation",
        "degree_at_fixed_n",
        "samples_with_adaptive_selection",
    }
    assert payload["threshold"] == pytest.approx(study.threshold)
    with pytest.raises(TypeError):
        refinement_study("not a spec", ishigami)  # type: ignore[arg-type]


def test_non_finite_model_output_is_refused() -> None:
    def broken(points: np.ndarray) -> np.ndarray:
        values = ishigami(points)
        values[0] = np.nan
        return values

    with pytest.raises(ValueError):
        refinement_study(
            ishigami_spec(),
            broken,
            degrees=(3, 4),
            degree_n_samples=512,
            truncation_degree=3,
            adaptive_n_samples=(128, 256),
            holdout_n=2000,
        )