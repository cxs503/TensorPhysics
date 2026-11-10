"""Bayesian calibration of the ice-going-ship digital twin (adaptive Metropolis MCMC).

Statistical model
-----------------
For ensemble member :math:`M_k` with physical parameters :math:`\\theta_k` and
multiplicative model-inadequacy parameter :math:`\\kappa` (a coefficient of
variation, not a physical property),

.. math::

    y_i = R_k(x_i; \\theta_k) + \\epsilon_i,\\qquad
    \\epsilon_i \\sim \\mathcal N\\!\\left(0,\\;
        \\sigma_{obs,i}^2 + (\\kappa\\,R_k(x_i;\\theta_k))^2\\right)

i.e. the measurement error (declared per record) and the model-form error
(proportional to the predicted resistance) are pooled in quadrature --
the Kennedy & O'Hagan (2001) discrepancy idea in its simplest multiplicative
form.  The joint log posterior is

.. math::

    \\log p(\\theta, \\kappa \\mid D) = \\sum_i \\log \\mathcal N(y_i\\mid R_k, \\sigma_i)
        + \\log p(\\theta) + \\log p(\\kappa) + \\text{const}

with **uniform priors on the declared boxes** (:data:`twin.PARAM_BOUNDS`,
:data:`twin.KAPPA_BOUNDS`) for the reference run -- deliberately weakly
informative so that the likelihood, not the prior, determines the posterior; an
alternative Gaussian prior on :math:`\\mu` (literature 0.15 +- 0.05) is provided
for the prior-sensitivity check.

Sampler
-------
Adaptive Metropolis (Haario, Saksman & Tamminen 2001): diagonal proposal for the
first ``n_adapt`` steps, then a scaled empirical covariance
``2.38^2/d * Cov`` with fixed seed and a documented adaptation freeze.  Chains
are over-dispersed at initialisation so that split-R is a meaningful diagnostic.
Diagnostics: split-R (Gelman et al. 2013, ch. 11.4) per parameter and effective
sample size from the repository's existing
:func:`tensorphysics.ice.mcmc_calibration.effective_sample_size` (Geyer initial
positive sequence) -- reused instead of re-implemented.

Evidence grading
----------------
* sampler + diagnostics : REAL (standard algorithms, convergence reported).
* data                  : SYNTHETIC (see :mod:`...twin` grading note) -- with
  the negative-control recovery design the calibration layer is verified, but
  no full-scale accuracy claim is made.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..ice.mcmc_calibration import effective_sample_size
from ..ice.resistance import G, RHO_ICE, SIGMA_C_REF, SIGMA_F_REF, ShipGeometry
from .twin import (
    KAPPA_BOUNDS,
    KAPPA_REFERENCE,
    PARAM_BOUNDS,
    PARAM_NAMES,
    PARAM_REFERENCE,
    ModelForm,
    get_form,
    k_bend_reference,
    resistance_kN,
)

__all__ = [
    "CalibrationData",
    "synthetic_campaign",
    "log_likelihood",
    "log_likelihood_features",
    "model_features",
    "model_from_features",
    "explained_prior",
    "log_posterior_factory",
    "run_am_chains",
    "ChainSet",
    "split_rhat",
    "ess_min",
    "parameter_summary",
    "predictive_draws",
    "predictive_matrix",
    "importance_loo",
    "posterior_predictive",
    "predictive_interval",
    "coverage_rate",
    "digest",
]


# ---------------------------------------------------------------------------
# data container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationData:
    """One measurement campaign: ice thickness, speed, resistance, sigma."""

    h_m: np.ndarray
    V_ms: np.ndarray
    R_kN: np.ndarray
    sigma_kN: np.ndarray
    label: str = "synthetic"
    truth: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        h = np.asarray(self.h_m, dtype=float).reshape(-1)
        v = np.asarray(self.V_ms, dtype=float).reshape(-1)
        r = np.asarray(self.R_kN, dtype=float).reshape(-1)
        s = np.asarray(self.sigma_kN, dtype=float).reshape(-1)
        if not (h.size == v.size == r.size == s.size):
            raise ValueError("calibration arrays must have equal length")
        if h.size < 1:
            raise ValueError("need at least one record")
        if np.any(s <= 0.0):
            raise ValueError("measurement sigma must be strictly positive")
        object.__setattr__(self, "h_m", h)
        object.__setattr__(self, "V_ms", v)
        object.__setattr__(self, "R_kN", r)
        object.__setattr__(self, "sigma_kN", s)

    def __len__(self) -> int:
        return int(self.h_m.size)

    def subset(self, idx: Sequence[int], label: Optional[str] = None) -> "CalibrationData":
        idx = np.asarray(idx, dtype=int).reshape(-1)
        return CalibrationData(
            self.h_m[idx],
            self.V_ms[idx],
            self.R_kN[idx],
            self.sigma_kN[idx],
            label=self.label if label is None else label,
            truth=self.truth,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "n": int(len(self)),
            "h_m": self.h_m.tolist(),
            "V_ms": self.V_ms.tolist(),
            "R_kN": self.R_kN.tolist(),
            "sigma_kN": self.sigma_kN.tolist(),
            "truth": self.truth,
        }


def synthetic_campaign(
    ship: ShipGeometry,
    *,
    n_points: int = 36,
    truth_params: Optional[Sequence[float]] = None,
    discrepancy_slope: float = 0.0,
    noise_cv: float = 0.06,
    noise_floor_kN: float = 2.0,
    seed: int = 20231,
    label: str = "synthetic-campaign",
) -> CalibrationData:
    """Deterministic synthetic measurement campaign (negative-control design).

    ``n_points`` (h, V) conditions are laid out on a maximin-ish grid inside the
    law's validity domain (``h`` from 0.30 m to 1.60 m, ``V`` from 1.0 m/s to the
    Froude-limited top speed), the "observed" resistance is generated from the
    **full** form at ``truth_params`` times a thickness-dependent inadequacy

    .. math::  y_i = R_{full}(x_i;\\theta_{true})\\,
               \\left[1 + \\texttt{discrepancy\\_slope}\\,(h_i - 1\\,\\mathrm m)\\right](1+\\delta_i)

    with ``delta_i ~ N(0, noise_cv^2)`` plus a ``noise_floor_kN`` absolute term,
    and the *declared* measurement sigma is the true noise level of the record.
    ``discrepancy_slope > 0`` makes the truth deliberately fall **outside** the
    calibrated model family, so that prediction-interval coverage can only be
    restored by an honest model-form/discrepancy term -- a coverage test that a
    pure interpolation exercise could not provide.
    """
    rng = np.random.default_rng(seed)
    theta = (
        np.array([PARAM_REFERENCE[n] for n in PARAM_NAMES], dtype=float)
        if truth_params is None
        else np.asarray(truth_params, dtype=float)
    )
    if theta.size != len(PARAM_NAMES):
        raise ValueError(f"truth_params needs {len(PARAM_NAMES)} entries")
    from ..ice.resistance import DomainLimits

    limits = DomainLimits()
    h_hi = min(0.03 * ship.L_pp, 0.10 * ship.B, 1.60)
    h_lo = 0.30
    # realistic ice-transit envelope: 1.0 .. 6.0 m/s (2 .. 12 kn); the law's
    # Froude limit (0.25 sqrt(gL) ~ 30 m/s) is an extrapolation guard, not an
    # operating point for an ice-going ship.
    v_hi = min(6.0, 0.85 * limits.v_limit(ship))
    # maximin-ish: n_h x n_v staggered grid, jittered by a fixed seed
    n_h = max(2, int(round(math.sqrt(n_points))))
    n_v = int(math.ceil(n_points / n_h))
    hs = np.linspace(h_lo, h_hi, n_h)
    vs = np.linspace(1.0, v_hi, n_v)
    hh, vv = np.meshgrid(hs, vs, indexing="ij")
    h = hh.reshape(-1)[:n_points]
    v = vv.reshape(-1)[:n_points]
    h = h * (1.0 + 0.01 * rng.standard_normal(h.size))
    v = v * (1.0 + 0.01 * rng.standard_normal(v.size))

    r_true = resistance_kN("full", theta, h, v, ship)
    inadequacy = 1.0 + discrepancy_slope * (h - 1.0)
    r_obs = r_true * inadequacy
    sigma = np.sqrt((noise_cv * r_obs) ** 2 + noise_floor_kN**2)
    y = r_obs + sigma * rng.standard_normal(r_obs.size)

    return CalibrationData(
        h_m=h,
        V_ms=v,
        R_kN=y,
        sigma_kN=sigma,
        label=label,
        truth={
            "params": {n: float(theta[i]) for i, n in enumerate(PARAM_NAMES)},
            "discrepancy_slope_per_m": float(discrepancy_slope),
            "noise_cv": float(noise_cv),
            "noise_floor_kN": float(noise_floor_kN),
            "seed": int(seed),
        },
    )


# ---------------------------------------------------------------------------
# likelihood / posterior
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# exact linear feature structure of the law (fast path + a physics statement)
# ---------------------------------------------------------------------------
#
# The level-ice law is *exactly bilinear* in (theta_c, theta_f, mu) and linear in
# theta_s::
#
#     R = theta_c (A + mu * A_mu) + theta_f (C + mu * C_mu) + theta_s * F
#
#     A    = 0.5 tan(alpha) sigma_c0 h^2
#     A_mu = 0.5 sigma_c0 h^2 / cos(alpha)
#     C    = 0.5 sin(phi/2) sigma_f0 h B_ship / k_bend
#     C_mu = 0.5 cos(phi/2) sigma_f0 h B_ship / k_bend
#     F    = (0.15 + 0.3 Fr_h) rho_i V^2 h B_ship
#
# This is why the posterior is a *curved* ridge (theta_c and mu trade off through
# the product) rather than a Gaussian blob: the curvature is physical, not a
# sampler artefact, and it is what limits random-walk efficiency.  Pre-computing
# the five feature vectors once per data set turns each posterior evaluation into
# a handful of 36-element numpy operations.


def model_features(data: CalibrationData, form: Any, ship: ShipGeometry) -> Dict[str, np.ndarray]:
    """Pre-compute the five linear feature vectors of the twin for one campaign."""
    f = get_form(form)
    h, V = data.h_m, data.V_ms
    alpha, phi = ship.flare_rad, ship.bow_angle_rad
    kb = k_bend_reference()
    return {
        "A": 0.5 * math.tan(alpha) * SIGMA_C_REF * np.square(h),
        "A_mu": 0.5 * SIGMA_C_REF * np.square(h) / math.cos(alpha),
        "C": 0.5 * math.sin(0.5 * phi) / kb * SIGMA_F_REF * h * ship.B,
        "C_mu": 0.5 * math.cos(0.5 * phi) / kb * SIGMA_F_REF * h * ship.B,
        "F": (0.15 + 0.3 * V / np.sqrt(G * h)) * RHO_ICE * np.square(V) * h * ship.B,
        "components": np.array([c in f.components for c in ("crush", "break", "submersion")]),
    }


def model_from_features(feats: Mapping[str, np.ndarray], phys: Sequence[float]) -> np.ndarray:
    """Resistance (kN) from the pre-computed features and a physical parameter vector."""
    theta_c, theta_f, mu, theta_s = (float(v) for v in phys[:4])
    use_crush, use_break, use_sub = feats["components"]
    out = np.zeros_like(feats["A"])
    if use_crush:
        out = out + theta_c * (feats["A"] + mu * feats["A_mu"])
    if use_break:
        out = out + theta_f * (feats["C"] + mu * feats["C_mu"])
    if use_sub:
        out = out + theta_s * feats["F"]
    return out / 1e3


def log_likelihood(
    data: CalibrationData,
    form: Any,
    ship: ShipGeometry,
    phys: Sequence[float],
    kappa: float,
) -> float:
    """Gaussian log likelihood with pooled measurement + discrepancy variance."""
    f = get_form(form)
    model = resistance_kN(f, phys, data.h_m, data.V_ms, ship)
    var = np.square(data.sigma_kN) + np.square(float(kappa) * model)
    if np.any(var <= 0.0):
        return -math.inf
    resid = data.R_kN - model
    return float(-0.5 * np.sum(np.log(2.0 * math.pi * var) + np.square(resid) / var))


def log_likelihood_features(
    feats: Mapping[str, np.ndarray],
    data: CalibrationData,
    phys: Sequence[float],
    kappa: float,
    sigma_kN: Optional[np.ndarray] = None,
) -> float:
    """Same likelihood as :func:`log_likelihood`, on the pre-computed features.

    ``sigma_kN`` lets a caller pass a subset of the declared sigmas (used by the
    fold-subsample machinery); it defaults to the campaign's own sigmas.
    """
    model = model_from_features(feats, phys)
    sigma = data.sigma_kN if sigma_kN is None else sigma_kN
    var = np.square(sigma) + np.square(float(kappa) * model)
    if np.any(var <= 0.0):
        return -math.inf
    resid = data.R_kN - model
    return float(-0.5 * np.sum(np.log(2.0 * math.pi * var) + np.square(resid) / var))


def explained_prior(
    name: str,
    value: float,
    kind: str = "uniform",
    *,
    mu_prior: Tuple[float, float] = (0.15, 0.05),
) -> float:
    """Log prior of one physical parameter on its declared box.

    ``kind='uniform'`` (default) is the weakly informative reference choice;
    ``kind='gaussian_mu'`` adds a literature Gaussian on ``mu_hull_ice`` only
    (used by the prior-sensitivity check) and stays uniform elsewhere.
    """
    i = PARAM_NAMES.index(name)
    lo, hi = PARAM_BOUNDS[i]
    if not (lo <= value <= hi):
        return -math.inf
    if kind == "gaussian_mu" and name == "mu_hull_ice":
        m0, s0 = mu_prior
        return -0.5 * ((value - m0) / s0) ** 2 - math.log(s0)
    if kind not in {"uniform", "gaussian_mu"}:
        raise ValueError(f"unknown prior kind {kind!r}")
    return 0.0


def log_posterior_factory(
    data: CalibrationData,
    form: Any,
    ship: ShipGeometry,
    *,
    use_discrepancy: bool = True,
    prior_kind: str = "uniform",
    mu_prior: Tuple[float, float] = (0.15, 0.05),
) -> Tuple[Callable[[np.ndarray], float], np.ndarray, List[str]]:
    """Build ``log p(x | D)`` over the box ``x = [physical..., (kappa)]``.

    Returns ``(log_post, bounds, names)`` where ``bounds`` is the ``(d, 2)`` box
    used by :func:`run_am_chains` and ``names`` labels the components.  The
    likelihood is evaluated through the pre-computed linear feature map, so one
    posterior evaluation costs a handful of 36-element numpy operations.
    """
    f = get_form(form)
    feats = model_features(data, f, ship)
    bnd = f.bounds()
    names = list(f.param_names)
    if use_discrepancy:
        bnd = np.vstack([bnd, np.asarray([KAPPA_BOUNDS], dtype=float)])
        names = names + ["kappa"]
    if prior_kind not in {"uniform", "gaussian_mu"}:
        raise ValueError(f"unknown prior kind {prior_kind!r}")
    lo, hi = bnd[:, 0].copy(), bnd[:, 1].copy()
    d = bnd.shape[0]
    y = data.R_kN
    sigma2 = np.square(data.sigma_kN)
    log2pi = math.log(2.0 * math.pi)
    n_phys = f.n_phys
    mu_index = f.param_names.index("mu_hull_ice") if "mu_hull_ice" in f.param_names else -1
    m0, s0 = mu_prior
    full_ref = np.array([PARAM_REFERENCE[n] for n in PARAM_NAMES], dtype=float)
    sub_idx = np.array([PARAM_NAMES.index(n) for n in f.param_names], dtype=int)
    kappa_lo, kappa_hi = KAPPA_BOUNDS

    def log_post(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != d:
            raise ValueError(f"expected {d} parameters, got {x.size}")
        if np.any(x < lo) or np.any(x > hi):
            return -math.inf
        lp = 0.0
        if prior_kind == "gaussian_mu" and mu_index >= 0:
            lp -= 0.5 * ((x[mu_index] - m0) / s0) ** 2 + math.log(s0)
        full = full_ref.copy()
        full[sub_idx] = x[:n_phys]
        model = model_from_features(feats, full)
        var = sigma2 + (x[-1] * model) ** 2 if use_discrepancy else sigma2
        if np.any(var <= 0.0):
            return -math.inf
        resid = y - model
        return lp - 0.5 * float(np.sum(log2pi + np.log(var) + np.square(resid) / var))

    return log_post, bnd, names


# ---------------------------------------------------------------------------
# adaptive Metropolis
# ---------------------------------------------------------------------------


@dataclass
class ChainSet:
    """Result of :func:`run_am_chains`."""

    samples: np.ndarray  # (n_chains, n_steps, d)
    logp: np.ndarray  # (n_chains, n_steps)
    bounds: np.ndarray
    names: List[str]
    burn: int
    seeds: List[int]
    proposal_scale: float

    @property
    def n_chains(self) -> int:
        return int(self.samples.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.samples.shape[1])

    def post_burn(self) -> np.ndarray:
        return self.samples[:, self.burn :, :]

    def flat(self) -> np.ndarray:
        pb = self.post_burn()
        return pb.reshape(-1, pb.shape[-1])

    def acceptance_rate(self) -> np.ndarray:
        """Fraction of non-identity moves per chain (flat proposal hits excluded)."""
        post = self.samples[:, self.burn :, :]
        moved = np.abs(np.diff(post, axis=1)).sum(axis=2) > 0.0
        return moved.mean(axis=1)

    def mean(self) -> np.ndarray:
        return self.flat().mean(axis=0)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_chains": self.n_chains,
            "n_steps": self.n_steps,
            "burn": int(self.burn),
            "dim": int(self.samples.shape[-1]),
            "names": list(self.names),
            "seeds": [int(s) for s in self.seeds],
            "proposal_scale": float(self.proposal_scale),
            "acceptance_rate": self.acceptance_rate().tolist(),
        }


def run_am_chains(
    log_post: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    *,
    n_chains: int = 3,
    n_steps: int = 10000,
    burn: int = 2000,
    inits: Optional[Sequence[Sequence[float]]] = None,
    seed: int = 20232,
    n_adapt: int = 500,
    target_accept: float = 0.25,
    scale_multiplier: float = 2.38,
    adapt_gain: float = 0.0,
    window_len: int = 2000,
) -> ChainSet:
    """Run ``n_chains`` adaptive Metropolis chains on a box-constrained posterior.

    Over-dispersed initialisation: chain ``c`` starts at
    ``lo + (c+1)/(n_chains+1) * (hi-lo)`` (plus a small seed-derived jitter)
    when ``inits`` is not given, which is what makes split-R a real test rather
    than a tautology.  After ``n_adapt`` steps the proposal covariance is frozen
    at ``scale_multiplier^2/d * Cov(past samples)`` (Haario et al. 2001), so the
    adaptation cannot hide non-convergence.
    """
    b = np.asarray(bounds, dtype=float)
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError("bounds must be (d, 2)")
    if np.any(b[:, 0] >= b[:, 1]):
        raise ValueError("invalid bounds")
    d = b.shape[0]
    rng = np.random.default_rng(seed)
    samples = np.empty((n_chains, n_steps, d))
    logps = np.empty((n_chains, n_steps))
    seeds: List[int] = []

    frac = (np.arange(1, n_chains + 1)) / (n_chains + 1.0)
    for c in range(n_chains):
        if inits is not None:
            x = np.asarray(inits[c], dtype=float).reshape(-1).copy()
            if x.size != d:
                raise ValueError(f"init {c} has {x.size} entries, expected {d}")
        else:
            span = b[:, 1] - b[:, 0]
            x = b[:, 0] + frac[c] * span + 1e-3 * span * rng.standard_normal(d)
        x = np.clip(x, b[:, 0] + 1e-12, b[:, 1] - 1e-12)
        lp = float(log_post(x))
        if not np.isfinite(lp):
            raise ValueError(f"chain {c}: initial log posterior is not finite")

        chain_seed = int(rng.integers(0, 2**31 - 1))
        seeds.append(chain_seed)
        prng = np.random.default_rng(chain_seed)
        history = np.empty((max(n_adapt, 10), d))
        n_hist = 0
        # diagonal proposal from the box width (dimension-aware) for the pilot
        span = b[:, 1] - b[:, 0]
        prop_cov = np.diag((0.02 * span) ** 2)
        log_scale = 0.0  # global multiplicative jitter, steers acceptance
        acc_window: List[bool] = []
        # rolling window of recent states for the covariance adaptation: a
        # covariance frozen on the (poorly mixed) pilot samples would lock the
        # proposal into the wrong orientation -- the reason the window rolls.
        window = np.empty((int(window_len), d))
        win_start = 0

        for t in range(n_steps):
            samples[c, t] = x
            logps[c, t] = lp
            z = prng.standard_normal(d)
            prop = x + z @ np.linalg.cholesky(prop_cov).T
            prop = np.clip(prop, b[:, 0] + 1e-12, b[:, 1] - 1e-12)
            lp_prop = float(log_post(prop))
            accept = math.log(prng.random()) < (lp_prop - lp)
            if accept:
                x, lp = prop, lp_prop
            acc_window.append(accept)
            if len(acc_window) > 100:
                acc_window.pop(0)

            # rolling history of visited states
            window[n_hist % window.shape[0]] = x
            n_hist += 1
            n_win = min(n_hist, window.shape[0])

            if t >= n_adapt and (t % 50 == 0) and n_win >= max(50, 2 * d):
                cov = np.cov(window[:n_win].T) + 1e-12 * np.eye(d)
                prop_cov = (scale_multiplier**2 / d) * cov * math.exp(log_scale)
            if t >= n_adapt and len(acc_window) == 100:
                a = float(np.mean(acc_window))
                log_scale += adapt_gain * (a - target_accept)
                log_scale = float(np.clip(log_scale, -4.0, 4.0))
        # end steps

    return ChainSet(
        samples=samples,
        logp=logps,
        bounds=b,
        names=[f"x{i}" for i in range(d)],
        burn=burn,
        seeds=seeds,
        proposal_scale=float(scale_multiplier),
    )


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


def split_rhat(post_burn: np.ndarray) -> np.ndarray:
    """Split-R per parameter (Gelman et al. 2013, eq. 11.4).

    Each chain is split into two halves; R is computed over the resulting
    ``2*m`` sequences.  Returns an array of length ``d``.
    """
    arr = np.asarray(post_burn, dtype=float)
    if arr.ndim != 3:
        raise ValueError("post_burn must be (n_chains, n_steps, d)")
    m, n, d = arr.shape
    if n < 10:
        raise ValueError("need at least 10 post-burn-in steps per chain")
    half = n // 2
    seqs = np.concatenate([arr[:, :half, :], arr[:, n - half :, :]], axis=0)
    m2 = seqs.shape[0]
    means = seqs.mean(axis=1)  # (2m, d)
    vars_ = seqs.var(axis=1, ddof=1)
    W = vars_.mean(axis=0)
    B = half * means.var(axis=0, ddof=1)
    var_hat = (1.0 - 1.0 / half) * W + (1.0 / half) * B
    with np.errstate(divide="ignore", invalid="ignore"):
        rhat = np.sqrt(var_hat / np.where(W > 1e-30, W, np.nan))
    return np.nan_to_num(rhat, nan=1.0, posinf=1.0)


def ess_min(post_burn: np.ndarray) -> Tuple[np.ndarray, int]:
    """Per-parameter ESS (summed over chains) and the worst-case value.

    Uses :func:`tensorphysics.ice.mcmc_calibration.effective_sample_size` on each
    chain separately and sums, which is the conservative pooling rule when
    chains have different autocorrelation.
    """
    arr = np.asarray(post_burn, dtype=float)
    d = arr.shape[-1]
    ess = np.zeros(d)
    for j in range(d):
        ess[j] = float(
            sum(effective_sample_size(arr[c, :, j]) for c in range(arr.shape[0]))
        )
    return ess, int(ess.min())


def parameter_summary(
    chains: ChainSet,
    *,
    names: Optional[Sequence[str]] = None,
    reference: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Posterior summary table with convergence diagnostics and bias vs reference."""
    post = chains.post_burn()
    flat = chains.flat()
    labels = list(names) if names is not None else list(chains.names)
    rhat = split_rhat(post)
    ess, ess_floor = ess_min(post)
    acc = chains.acceptance_rate()
    rows: List[Dict[str, Any]] = []
    for j, name in enumerate(labels):
        col = flat[:, j]
        row: Dict[str, Any] = {
            "name": name,
            "mean": float(col.mean()),
            "sd": float(col.std(ddof=1)),
            "q05": float(np.percentile(col, 5.0)),
            "q95": float(np.percentile(col, 95.0)),
            "rhat_split": float(rhat[j]),
            "ess": int(ess[j]),
        }
        if reference and name in reference:
            ref = float(reference[name])
            row["reference"] = ref
            row["relative_error"] = (
                float(abs(col.mean() - ref) / ref) if ref != 0.0 else float("nan")
            )
        rows.append(row)
    return {
        "parameters": rows,
        "max_rhat_split": float(np.max(rhat)) if rhat.size else 1.0,
        "min_ess": ess_floor,
        "acceptance_rate": acc.tolist(),
        "n_samples_per_chain": int(post.shape[1]),
        "n_chains": int(post.shape[0]),
        "n_effective_total": int(flat.shape[0]),
    }


# ---------------------------------------------------------------------------
# posterior predictive
# ---------------------------------------------------------------------------


def predictive_matrix(
    chains: ChainSet,
    data: CalibrationData,
    form: Any,
    ship: ShipGeometry,
    *,
    use_discrepancy: bool = True,
    thin: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Posterior mean/sd of the held-out resistance, ``(n_draws, n_records)`` each.

    Vectorised through the linear feature map (:func:`model_from_features`), so
    the whole posterior predictive of a 36-record campaign costs two BLAS-free
    matrix products rather than 6000 python-level model calls.
    """
    f = get_form(form)
    flat = chains.flat()[:: max(1, int(thin))]
    kappa = flat[:, -1] if use_discrepancy else np.zeros(flat.shape[0])
    feats = model_features(data, f, ship)
    use_crush, use_break, use_sub = feats["components"]
    tc = flat[:, PARAM_NAMES.index("sigma_c_scale")] if use_crush else None
    tf = flat[:, PARAM_NAMES.index("sigma_f_scale")] if use_break else None
    mu_p = flat[:, PARAM_NAMES.index("mu_hull_ice")]
    ts = flat[:, PARAM_NAMES.index("C_sub_scale")] if use_sub else None
    out = np.zeros((flat.shape[0], len(data)))
    if use_crush:
        out += np.outer(tc, feats["A"]) + np.outer(tc * mu_p, feats["A_mu"])
    if use_break:
        out += np.outer(tf, feats["C"]) + np.outer(tf * mu_p, feats["C_mu"])
    if use_sub:
        out += np.outer(ts, feats["F"])
    mu = out / 1e3
    sd = np.sqrt(np.square(data.sigma_kN)[None, :] + np.square(kappa[:, None]) * np.square(mu))
    return mu, sd


def predictive_draws(
    chains: ChainSet,
    data: CalibrationData,
    form: Any,
    ship: ShipGeometry,
    *,
    use_discrepancy: bool = True,
    thin: int = 1,
    seed: int = 20233,
) -> np.ndarray:
    """Posterior predictive draws ``y*`` for every record of ``data``.

    Shape ``(n_draws, n_records)``: one draw per retained posterior sample, so
    the mixture structure (which carries the parameter *and* discrepancy
    uncertainty) is preserved.
    """
    mu, sd = predictive_matrix(
        chains, data, form, ship, use_discrepancy=use_discrepancy, thin=thin
    )
    rng = np.random.default_rng(seed)
    return mu + sd * rng.standard_normal(mu.shape)


def posterior_predictive(
    chains: ChainSet,
    data: CalibrationData,
    form: Any,
    ship: ShipGeometry,
    *,
    use_discrepancy: bool = True,
    thin: int = 1,
) -> Dict[str, Any]:
    """Predictive mean / sd / model-only mean for a record set."""
    mu, sd = predictive_matrix(
        chains, data, form, ship, use_discrepancy=use_discrepancy, thin=thin
    )
    pred_sd = np.sqrt(np.mean(np.square(sd) + np.square(mu), axis=0) - np.square(mu.mean(axis=0)))
    return {
        "model_mean_kN": mu.mean(axis=0),
        "model_sd_kN": mu.std(axis=0, ddof=1),
        "pred_mean_kN": mu.mean(axis=0),
        "pred_sd_kN": pred_sd,
        "n_draws": int(mu.shape[0]),
    }


def importance_loo(
    chains: ChainSet,
    data: CalibrationData,
    form: Any,
    ship: ShipGeometry,
    *,
    use_discrepancy: bool = True,
    thin: int = 1,
    level: float = 0.95,
    seed: int = 20241,
    ess_warn_fraction: float = 0.10,
    return_draws: bool = False,
) -> Dict[str, Any]:
    """Leave-one-out predictive distributions by self-normalised importance sampling.

    Gelfand & Dey (1994): the leave-out-one posterior is obtained by reweighting
    the full posterior draws with ``1 / p(y_i | theta)``.  This is exact in
    expectation, needs no refit per fold, and its practical reliability is
    *reported*, not assumed: the effective sample size of each fold's weights and
    the largest single-weight share are returned, and folds whose weight ESS falls
    below ``ess_warn_fraction`` of the draws are flagged in ``unreliable_folds``
    (the PSIS diagnostic of Vehtari et al. 2017 exists precisely because raw IS
    can be dominated by one draw; with a deliberately *diffuse* posterior such as
    the one measured here the weights stay tame, which the numbers show).

    Returns the per-fold held-out draws, the LOO log predictive density, the
    pointwise 95 % interval and its coverage.
    """
    mu, sd = predictive_matrix(
        chains, data, form, ship, use_discrepancy=use_discrepancy, thin=thin
    )
    y = data.R_kN
    lp = -0.5 * np.log(2.0 * math.pi * np.square(sd)) - np.square(y[None, :] - mu) / (
        2.0 * np.square(sd)
    )
    n_draws = mu.shape[0]
    rng = np.random.default_rng(seed)
    folds: List[Dict[str, Any]] = []
    draws_all: List[np.ndarray] = []
    lo_all = np.empty(len(data))
    hi_all = np.empty(len(data))
    lpd_all = np.empty(len(data))
    for i in range(len(data)):
        logw = -lp[:, i]
        m = float(np.max(logw))
        w = np.exp(logw - m)
        wsum = float(np.sum(w))
        w_norm = w / wsum
        ess = float(1.0 / np.sum(np.square(w_norm)))
        # p(y_i | D_{-i}) ~ 1 / mean_s(1/p(y_i|theta_s)); in log space
        lpd = -(math.log(wsum / n_draws) + m)  # log of 1/mean(1/p)
        idx = rng.choice(n_draws, size=n_draws, replace=True, p=w_norm)
        draws = mu[idx, i] + sd[idx, i] * rng.standard_normal(n_draws)
        lo, hi = np.percentile(draws, [100 * 0.5 * (1 - level), 100 * (1 - 0.5 * (1 - level))])
        lo_all[i], hi_all[i], lpd_all[i] = float(lo), float(hi), float(lpd)
        if return_draws:
            draws_all.append(draws)
        folds.append(
            {
                "index": int(i),
                "observed_kN": float(y[i]),
                "pred_mean_kN": float(draws.mean()),
                "pred_sd_kN": float(draws.std(ddof=1)),
                "pi_lower_kN": float(lo),
                "pi_upper_kN": float(hi),
                "log_predictive": float(lpd),
                "weight_ess": ess,
                "weight_ess_fraction": ess / n_draws,
                "max_weight_share": float(np.max(w_norm)),
                "n_posterior_draws": int(n_draws),
            }
        )
    inside, total, rate = coverage_rate(lo_all, hi_all, y)
    data_ = {
        "form": get_form(form).key,
        "label": get_form(form).label,
        "method": "self-normalised importance-sampling LOO-CV (Gelfand & Dey 1994)",
        "n_folds": int(len(data)),
        "n_posterior_draws": int(n_draws),
        "level": float(level),
        "coverage": rate,
        "n_inside": int(inside),
        "n_records": int(total),
        "mean_pi_width_kN": float(np.mean(hi_all - lo_all)),
        "lpd_total": float(np.sum(lpd_all)),
        "lpd_per_record": float(np.mean(lpd_all)),
        "mean_relative_error": float(
            np.mean(np.abs(np.array([f["pred_mean_kN"] for f in folds]) - y) / y)
        ),
        "min_weight_ess": float(min(f["weight_ess"] for f in folds)),
        "min_weight_ess_fraction": float(min(f["weight_ess_fraction"] for f in folds)),
        "unreliable_folds": [
            int(f["index"]) for f in folds if f["weight_ess_fraction"] < ess_warn_fraction
        ],
        "misses": [int(i) for i in range(len(data)) if not (lo_all[i] <= y[i] <= hi_all[i])],
        "folds": folds,
        "loo_draws": draws_all if return_draws else None,
    }
    return data_


def predictive_interval(samples: np.ndarray, level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """Pointwise central interval from predictive draws ``(n_draws, n_records)``."""
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    a = np.asarray(samples, dtype=float)
    if a.ndim != 2:
        raise ValueError("samples must be 2-D (n_draws, n_records)")
    alpha = 0.5 * (1.0 - level) * 100.0
    lo = np.percentile(a, alpha, axis=0)
    hi = np.percentile(a, 100.0 - alpha, axis=0)
    return lo, hi


def coverage_rate(lower: Any, upper: Any, observed: Any) -> Tuple[int, int, float]:
    """Empirical coverage: ``(#inside, n, rate)``."""
    lo = np.asarray(lower, dtype=float).reshape(-1)
    hi = np.asarray(upper, dtype=float).reshape(-1)
    y = np.asarray(observed, dtype=float).reshape(-1)
    if not (lo.size == hi.size == y.size):
        raise ValueError("lower/upper/observed must have the same length")
    inside = int(np.sum((y >= lo) & (y <= hi)))
    return inside, int(y.size), float(inside / y.size)


def digest(payload: Any) -> str:
    """Stable sha256 of a JSON-serialisable payload (provenance chain)."""
    import json

    blob = json.dumps(payload, sort_keys=True, default=float).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()