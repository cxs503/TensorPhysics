"""Spatially correlated random fields along the ship length.

The ice pressure acting on the hull girder is modelled as a stationary
random field::

    p(x, t) = mu_p(x) + sigma_p * Z(x, t)

where ``Z`` is a zero-mean, unit-variance, spatially correlated Gaussian
field.  Two covariance models are provided:

* ``exponential``   C(dx) = exp(-|dx| / lambda)
* ``squared_exponential`` (Gaussian)  C(dx) = exp(-(dx / lambda)^2)

Sampling notes
--------------
For the exponential model on a *uniform* grid the field is exactly a
first-order autoregressive (Ornstein-Uhlenbeck) process, so realisations can
be generated with an O(n) recursion and no spectral truncation error::

    Z_0 ~ N(0, 1),  Z_k = rho Z_{k-1} + sqrt(1 - rho^2) eps_k,  rho = exp(-dx/lambda)

This matters for the Monte-Carlo work: 1e4 - 5e4 realisations cost almost
nothing, and the ``lambda -> 0`` limit is exactly white noise, which is what
the "independent segment" acceptance check needs.

For the Gaussian covariance model the factorisation is obtained from a
truncated eigendecomposition of the covariance matrix (Karhunen-Loeve).

All functions are pure NumPy; no solver dependency.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

__all__ = [
    "exponential_covariance",
    "squared_exponential_covariance",
    "sample_correlated_field",
    "ar1_field",
    "karhunen_loève_field",
    "autocorrelation_length",
    "fit_exponential_length",
    "space_time_field",
    "check_uniform_grid",
]


def check_uniform_grid(x: np.ndarray, tol: float = 1e-9) -> float:
    """Return the grid spacing, raising if ``x`` is not uniform and increasing."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 3:
        raise ValueError("grid must be a 1-D array with at least 3 nodes")
    dx = np.diff(x)
    if np.any(dx <= 0.0):
        raise ValueError("grid must be strictly increasing")
    if np.ptp(dx) > tol * max(1.0, abs(float(np.mean(dx)))):
        raise ValueError("correlated field sampler requires a uniform grid")
    return float(np.mean(dx))


def exponential_covariance(x: np.ndarray, lam: float, sigma: float = 1.0) -> np.ndarray:
    """Covariance matrix of the exponential (Markov) model."""
    x = np.asarray(x, dtype=float)
    if lam <= 0.0:
        raise ValueError("correlation length must be positive")
    d = np.abs(x[:, None] - x[None, :])
    return (sigma ** 2) * np.exp(-d / lam)


def squared_exponential_covariance(x: np.ndarray, lam: float, sigma: float = 1.0) -> np.ndarray:
    """Covariance matrix of the Gaussian (squared-exponential) model."""
    x = np.asarray(x, dtype=float)
    if lam <= 0.0:
        raise ValueError("correlation length must be positive")
    d = np.abs(x[:, None] - x[None, :])
    return (sigma ** 2) * np.exp(-(d / lam) ** 2)


def ar1_field(
    x: np.ndarray,
    lam: float,
    sigma: float = 1.0,
    n_real: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Exact realisations of the exponential-covariance field on a uniform grid.

    Returns an array of shape ``(n_real, x.size)``.
    """
    dx = check_uniform_grid(x)
    if lam <= 0.0:
        raise ValueError("correlation length must be positive")
    rng = np.random.default_rng() if rng is None else rng
    rho = float(np.exp(-dx / lam))
    # exactly zero correlations for lam << dx (white-noise limit)
    eps = rng.standard_normal((int(n_real), x.size))
    z = np.empty_like(eps)
    z[:, 0] = eps[:, 0]
    scale = np.sqrt(max(0.0, 1.0 - rho * rho))
    for k in range(1, x.size):
        z[:, k] = rho * z[:, k - 1] + scale * eps[:, k]
    return sigma * z


def karhunen_loève_field(
    x: np.ndarray,
    lam: float,
    sigma: float = 1.0,
    n_real: int = 1,
    rng: np.random.Generator | None = None,
    model: str = "squared_exponential",
    n_modes: int | None = None,
    energy_tol: float = 1e-10,
) -> np.ndarray:
    """Truncated Karhunen-Loeve sampling (any covariance model, any grid)."""
    rng = np.random.default_rng() if rng is None else rng
    cov = (
        squared_exponential_covariance(x, lam, sigma)
        if model == "squared_exponential"
        else exponential_covariance(x, lam, sigma)
    )
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    evals = np.clip(evals, 0.0, None)
    total = float(evals.sum())
    if n_modes is None:
        keep = np.cumsum(evals) <= (1.0 - energy_tol) * total
        n_modes = int(max(1, np.count_nonzero(keep) + 1))
    n_modes = min(int(n_modes), evals.size)
    xi = rng.standard_normal((n_modes, int(n_real)))
    return (evecs[:, :n_modes] * np.sqrt(evals[:n_modes])[None, :] @ xi).T


def sample_correlated_field(
    x: np.ndarray,
    lam: float,
    sigma: float = 1.0,
    n_real: int = 1,
    seed: int | None = None,
    model: str = "exponential",
) -> np.ndarray:
    """Dispatch to the appropriate sampler; returns ``(n_real, x.size)``."""
    rng = np.random.default_rng(seed)
    if model == "exponential":
        return ar1_field(x, lam, sigma=sigma, n_real=n_real, rng=rng)
    if model == "squared_exponential":
        return karhunen_loève_field(x, lam, sigma=sigma, n_real=n_real, rng=rng, model=model)
    raise ValueError(f"unknown covariance model: {model!r}")


def autocorrelation_length(
    x: np.ndarray, profiles: Iterable[np.ndarray], max_lag_fraction: float = 0.25
) -> float:
    """Integral correlation scale ``int_0^inf C(dx)/C(0) d(dx)`` from samples.

    The window is truncated at ``max_lag_fraction`` of the record, so the record
    must be long compared with the correlation length (bias grows as
    ``lambda`` approaches the record length).
    """
    x = np.asarray(x, dtype=float)
    dx = check_uniform_grid(x)
    prof = np.atleast_2d(np.asarray(profiles, dtype=float))
    if prof.ndim != 2 or prof.shape[1] != x.size:
        raise ValueError("profiles must have shape (n_samples, x.size)")
    centred = prof - prof.mean(axis=1, keepdims=True)
    var = float(np.mean(centred ** 2))
    if var <= 0.0:
        raise ValueError("zero variance profiles: correlation length undefined")
    n_lag = int(max(1, min(x.size // 2, max_lag_fraction * x.size)))
    integral = 0.0
    for lag in range(n_lag):
        a = centred[:, : x.size - lag]
        b = centred[:, lag:]
        rho = float(np.mean(a * b)) / var
        if rho <= 0.0:
            break
        integral += rho * dx
    return float(integral)


def fit_exponential_length(
    x: np.ndarray, profiles: Iterable[np.ndarray], max_lag_fraction: float = 0.25
) -> float:
    """Least-squares fit of ``rho(dx) = exp(-dx / lambda)`` to sample autocorrelation.

    Fitted by grid search on the lags whose sample correlation is still above
    0.3 (a log-based estimator is strongly biased by sampling noise at large
    lag).  Same window requirement as :func:`autocorrelation_length`: the record
    must be long compared with ``lambda``.
    """
    x = np.asarray(x, dtype=float)
    dx = check_uniform_grid(x)
    prof = np.atleast_2d(np.asarray(profiles, dtype=float))
    if prof.shape[1] != x.size:
        raise ValueError("profiles must have shape (n_samples, x.size)")
    centred = prof - prof.mean(axis=1, keepdims=True)
    var = float(np.mean(centred ** 2))
    if var <= 0.0:
        raise ValueError("zero variance profiles: correlation length undefined")
    n_lag = int(max(1, min(x.size // 2, max_lag_fraction * x.size)))
    lags = np.arange(n_lag) * dx
    rho = np.array(
        [float(np.mean(centred[:, : x.size - k] * centred[:, k:])) / var for k in range(n_lag)]
    )
    # grid-search least squares over lags whose sample correlation is still
    # above 0.3 (the log-based estimator is badly biased by sampling noise once
    # rho becomes small)
    usable = rho >= 0.3
    if np.count_nonzero(usable) < 2:
        usable = rho > 0.0
    if np.count_nonzero(usable) < 2:
        raise ValueError("autocorrelation decays too fast to fit a length scale")
    lag_u, rho_u = lags[usable], rho[usable]
    grid = np.exp(np.linspace(np.log(dx * 1e-2), np.log(50.0 * lag_u.max()), 4000))
    sse = np.array([np.sum((rho_u - np.exp(-lag_u / lam)) ** 2) for lam in grid])
    return float(grid[int(np.argmin(sse))])


def space_time_field(
    x: np.ndarray,
    lam_x: float,
    lam_t: float,
    sigma: float = 1.0,
    n_steps: int = 200,
    dt: float = 0.25,
    seed: int | None = None,
) -> np.ndarray:
    """Separable space-time correlated field ``Z(x, t)``.

    Covariance ``C(dx, dt) = sigma^2 exp(-|dx|/lam_x) exp(-|dt|/lam_t)``,
    generated as a first-order Markov process in time whose innovations are the
    exactly sampled exponential spatial field::

        Z(:, t+dt) = rho_t Z(:, t) + sqrt(1 - rho_t^2) W(:, t),  rho_t = exp(-dt/lam_t)

    Returns an array of shape ``(n_steps, x.size)``.  The realised lag-one
    temporal correlation is close to but not exactly ``rho_t`` because the
    spatial filter acts on the innovation (documented approximation); it is used
    for space-time pressure contour plots only, with ``lam_t`` typically set to
    the floe transit time ``lam_x / v``.
    """
    if lam_t <= 0.0:
        raise ValueError("temporal correlation length must be positive")
    rng = np.random.default_rng(seed)
    rho_t = float(np.exp(-dt / lam_t))
    innov_scale = float(np.sqrt(max(0.0, 1.0 - rho_t * rho_t)))
    out = np.empty((int(n_steps), x.size), dtype=float)
    out[0] = ar1_field(x, lam_x, sigma=sigma, n_real=1, rng=rng)[0]
    for k in range(1, int(n_steps)):
        innovation = ar1_field(x, lam_x, sigma=sigma, n_real=1, rng=rng)[0]
        out[k] = rho_t * out[k - 1] + innov_scale * innovation
    return out
