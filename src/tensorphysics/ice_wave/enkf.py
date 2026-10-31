"""Ensemble Kalman analysis for ice-field background / observation fusion.

Interface for W5-N04: a homogenised brash-ice field (from
:mod:`tensorphysics.ice_wave.homogenise`) is used as the ensemble background,
in-situ / remote observations update it, and the analysis is handed back as the
background of the next assimilation cycle.

Formulation (stochastic EnKF, Burgers et al. 1998; Evensen 2003):

    1. perturbations are resampled from the ensemble anomalies,
    2. ``K = P^f H^T (H P^f H^T + R)^{-1}``,
    3. ``x^a_i = x^f_i + K (y + eps_i - H x^f_i)``,

with covariance estimated from the ensemble anomalies without the ``N - 1``
deflation used by some implementations (they differ by a common factor which
cancels in ``K``). The state is stored as ``(n_ensemble, n_state)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class EnKFError(ValueError):
    """Invalid EnKF input."""


def _as_2d(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2:
        raise EnKFError(f"{name} must be 2-D, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise EnKFError(f"{name} contains non-finite values")
    return array


def ensemble_kalman_analysis(
    background: Any,
    observations: Any,
    *,
    operator: Any = None,
    observation_error_std: Any = 1.0,
    seed: int = 0,
    inflation: float = 1.0,
) -> dict[str, Any]:
    """Perform a single stochastic EnKF analysis step.

    Parameters
    ----------
    background:
        ``(n_ensemble, n_state)`` ensemble of background states.
    observations:
        ``(n_obs,)`` observation vector.
    operator:
        ``(n_obs, n_state)`` linear observation operator ``H``; identity when the
        observation count matches the state size, otherwise required.
    observation_error_std:
        scalar or ``(n_obs,)`` observation error standard deviation.
    inflation:
        prior multiplicative variance inflation factor applied before the update.

    Returns
    -------
    dict with keys ``analysis``, ``innovation``, ``gain``, ``spread_before``,
    ``spread_after``, ``rms_background_error`` and ``rms_analysis_error``
    (the last two against ``observations`` projected through ``H``).
    """
    bg = _as_2d("background", background)
    n_ens, n_state = bg.shape
    if n_ens < 3:
        raise EnKFError(f"background needs at least 3 ensemble members, got {n_ens}")

    obs = np.asarray(observations, dtype=float).ravel()
    if obs.size == 0:
        raise EnKFError("observations must not be empty")
    if not np.all(np.isfinite(obs)):
        raise EnKFError("observations contain non-finite values")
    n_obs = obs.size

    if operator is None:
        if n_obs != n_state:
            raise EnKFError(
                "operator is required when the observation count differs from the state size"
            )
        H = np.eye(n_state)
    else:
        H = np.asarray(operator, dtype=float)
        if H.ndim == 1:
            H = H.reshape(1, -1)
        if H.shape != (n_obs, n_state):
            raise EnKFError(f"operator shape {H.shape} incompatible with ({n_obs}, {n_state})")

    R_std = np.broadcast_to(np.asarray(observation_error_std, dtype=float).ravel(), (n_obs,))
    if not np.all(np.isfinite(R_std)) or np.any(R_std < 0.0):
        raise EnKFError("observation_error_std must be finite and non-negative")

    inflation = float(inflation)
    if not np.isfinite(inflation) or inflation <= 0.0:
        raise EnKFError(f"inflation must be positive and finite, got {inflation!r}")

    mean = bg.mean(axis=0)
    anomalies = (bg - mean) * np.sqrt(inflation)
    # Unbiased-free factor (N-1) is used consistently for P^f, HPH^T and K; it
    # cancels in the gain, so only the relative scaling matters.
    denom = max(n_ens - 1, 1)
    P = anomalies.T @ anomalies / denom
    HP = H @ P
    S = HP @ H.T + np.diag(R_std**2)
    K = P @ H.T @ np.linalg.pinv(S)

    # Perturbed observations (stochastic EnKF).
    rng = np.random.default_rng(seed)
    perturbed = obs[None, :] + rng.standard_normal((n_ens, n_obs)) * R_std[None, :]
    innovations = perturbed - (bg @ H.T)
    analysis = bg + innovations @ K.T

    spread_before = float(np.mean(np.std(bg, axis=0)))
    spread_after = float(np.mean(np.std(analysis, axis=0)))
    rms_bg = float(np.sqrt(np.mean((bg.mean(axis=0) @ H.T - obs) ** 2)))
    rms_an = float(np.sqrt(np.mean((analysis.mean(axis=0) @ H.T - obs) ** 2)))
    return {
        "analysis": analysis,
        "innovation": innovations.mean(axis=0),
        "gain": K,
        "spread_before": spread_before,
        "spread_after": spread_after,
        "rms_background_error": rms_bg,
        "rms_analysis_error": rms_an,
    }