"""Tests for the EnKF background interface (W5-N04 hook, W5-N05)."""

from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.ice_wave.enkf import EnKFError, ensemble_kalman_analysis


def _ensemble(n_ens: int = 40, n_state: int = 6, scale: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    truth = np.linspace(0.2, 0.35, n_state)
    background = truth + rng.normal(scale=scale, size=(n_ens, n_state))
    return truth, background


def test_exact_observations_are_reproduced():
    truth, background = _ensemble()
    result = ensemble_kalman_analysis(
        background, truth, observation_error_std=1e-6, seed=1
    )
    assert result["rms_analysis_error"] < 1e-3
    assert result["rms_analysis_error"] < result["rms_background_error"]


def test_analysis_reduces_error_and_spread():
    truth, background = _ensemble()
    result = ensemble_kalman_analysis(
        background, truth[[0, 3, 5]], operator=np.eye(6)[[0, 3, 5]],
        observation_error_std=0.01, seed=2,
    )
    assert result["rms_analysis_error"] < 0.5 * result["rms_background_error"]
    assert result["spread_after"] < result["spread_before"]
    assert result["analysis"].shape == background.shape


def test_gain_is_bounded_and_consistent():
    truth, background = _ensemble()
    H = np.eye(6)[[1, 4]]
    result = ensemble_kalman_analysis(
        background, truth[[1, 4]], operator=H, observation_error_std=0.05, seed=3
    )
    gain = result["gain"]
    assert gain.shape == (6, 2)
    # EnKF gain entries cannot exceed 1/sqrt(spread) scaling; a simple bound is
    # that the analysis mean must stay within the convex hull of the prior and
    # the observation.
    prior_mean = background.mean(axis=0)
    obs = truth[[1, 4]]
    analysed = result["analysis"].mean(axis=0)
    for idx, col in enumerate((1, 4)):
        lo = min(prior_mean[col], obs[idx]) - 1e-9
        hi = max(prior_mean[col], obs[idx]) + 1e-9
        assert lo <= analysed[col] <= hi


def test_inflation_increases_spread():
    truth, background = _ensemble()
    plain = ensemble_kalman_analysis(
        background, truth[[0]], operator=np.eye(6)[[0]], observation_error_std=0.05, seed=4
    )
    inflated = ensemble_kalman_analysis(
        background, truth[[0]], operator=np.eye(6)[[0]], observation_error_std=0.05,
        seed=4, inflation=4.0,
    )
    assert inflated["spread_after"] > plain["spread_after"]


@pytest.mark.parametrize(
    "call",
    [
        lambda: ensemble_kalman_analysis(np.zeros((2, 3)), np.zeros(3)),
        lambda: ensemble_kalman_analysis(np.zeros((5, 3)), np.zeros(4)),
        lambda: ensemble_kalman_analysis(np.zeros((5, 3)), np.zeros(2)),
        lambda: ensemble_kalman_analysis(np.zeros((5, 3)), np.zeros(3), inflation=-1.0),
        lambda: ensemble_kalman_analysis(np.zeros((5, 3)), np.zeros(3),
                                         observation_error_std=-0.5),
        lambda: ensemble_kalman_analysis(np.full((5, 3), np.nan), np.zeros(3)),
    ],
)
def test_invalid_inputs_raise(call):
    with pytest.raises(EnKFError):
        call()