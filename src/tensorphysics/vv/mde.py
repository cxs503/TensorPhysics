"""MDE_90 — minimum detectable error of an acceptance criterion.

A criterion is a **detector**: it says FAIL when the evidence is wrong.  That
it passes on clean evidence proves nothing about its sensitivity — a gate whose
threshold is 20 % and a gate whose threshold is 3 % look identical until an
error is injected.  This module measures the sensitivity instead of assuming
it: errors of known size are injected, the criterion is asked to fire, and the
detection probability is estimated over repeated draws.

Nothing here knows what the quantity *is* (no physics, no unit semantics): a
detector is any callable ``float -> bool`` that receives the **relative error
realisation** ``y`` and returns ``True`` when it rejects.  The caller is
responsible for wrapping the shipped gate into that shape.

Statistical model
-----------------
For a bias ``delta`` and an evidence noise ``sigma`` the realisation seen by the
detector is

    y = delta + sigma/sqrt(n_obs) * Z,      Z ~ N(0, 1)

i.e. the systematic part is the injected bias and the random part is the
repeatability of the evidence (finite-sample mean of ``n_obs`` observations).
``sigma = 0`` reproduces a deterministic criterion, whose detection curve is a
step and whose MDE_90 is *exactly* the criterion's own decision boundary (this
module extracts it by bisection, not by eyeballing the ladder).

Definitions
-----------
* ``MDE_90`` = smallest bias whose detection probability reaches 0.90.
* A curve that never reaches 0.90 inside the scanned range is **censored** and
  reported as such (``None`` + ``censored=">hi"``) — never silently clamped.
* Detection probabilities carry Wilson score intervals, monotone (PAVA)
  smoothing and a logistic (IRLS) fit whose ``MDE_50``/``MDE_90`` are read off
  the fitted curve.

Closed-form references for four standard detector statistics (``point``,
``mean``, ``bound``, ``proportion``) are provided so that every Monte-Carlo
number in the ledger can be audited against theory.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

__all__ = [
    "Detector",
    "DetectionCurve",
    "LogisticFit",
    "Z90",
    "Z95",
    "norm_cdf",
    "norm_ppf",
    "wilson_interval",
    "pava",
    "detection_curve",
    "mde90_interpolate",
    "mde90_bisect",
    "fit_logistic",
    "power_point",
    "power_mean",
    "power_bound",
    "power_proportion",
    "binomial_sf",
    "proportion_k_required",
    "proportion_samples_required",
    "mde90_analytic",
    "certificate",
    "classify_mde",
    "sha256_of_file",
]

#: One-sided 90 % normal quantile (detection target).
Z90 = 1.2815515655446004
#: Two-sided 95 % normal quantile (interval half-width).
Z95 = 1.959963984540054

Detector = Callable[[float], bool]


# ---------------------------------------------------------------------------
# distribution helpers (stdlib only)
# ---------------------------------------------------------------------------
def norm_cdf(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def norm_ppf(p: float) -> float:
    """Standard normal quantile (inverse CDF)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p!r}")
    return statistics.NormalDist().inv_cdf(p)


def _wilson_upper_zero(n: int, z: float) -> float:
    """Upper Wilson bound for an observed proportion of exactly zero."""
    z2 = z * z
    return (z2 / (2.0 * n) + z * math.sqrt(z2 / (4.0 * n * n))) / (1.0 + z2 / n)


def wilson_interval(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (never leaves [0, 1])."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError(f"k={k} outside [0, n={n}]")
    p = k / n
    if p == 0.0:
        return 0.0, min(1.0, _wilson_upper_zero(n, z))
    if p == 1.0:
        return max(0.0, 1.0 - _wilson_upper_zero(n, z)), 1.0
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(max(p * (1.0 - p) / n + z2 / (4.0 * n * n), 0.0))
    lo, hi = centre - half, centre + half
    return max(0.0, min(1.0, lo)), max(0.0, min(1.0, hi))


def _log_binom_pmf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 0.0 if k > 0 else 1.0
    if p >= 1.0:
        return 0.0 if k < n else 1.0
    log_pmf = (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
               + k * math.log(p) + (n - k) * math.log1p(-p))
    return math.exp(log_pmf)


def binomial_sf(k: int, n: int, p: float) -> float:
    """``P(X >= k)`` for ``X ~ Binomial(n, p)`` (exact sum, no normal approx)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return sum(_log_binom_pmf(i, n, p) for i in range(k, n + 1))


def binomial_cdf(k: int, n: int, p: float) -> float:
    """``P(X <= k)`` for ``X ~ Binomial(n, p)``."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(_log_binom_pmf(i, n, p) for i in range(0, k + 1))


# ---------------------------------------------------------------------------
# monotone smoothing and interpolation
# ---------------------------------------------------------------------------
def pava(values: Sequence[float],
         weights: Sequence[float] | None = None) -> list[float]:
    """Pool-adjacent-violators: least-squares non-decreasing fit (weighted).

    Detection probability must not decrease with the injected bias; Monte-Carlo
    noise can make it, and a negative slope in a published curve is an artefact.
    PAVA removes it with the minimum squared change and keeps the weighted mean.
    """
    y = [float(v) for v in values]
    w = [1.0] * len(y) if weights is None else [float(x) for x in weights]
    if len(w) != len(y):
        raise ValueError("weights length mismatch")
    if not y:
        return []
    stack: list[list[float]] = []          # [mean, weight, count]
    for value, weight in zip(y, w):
        if weight <= 0.0:
            raise ValueError("weights must be positive")
        stack.append([value, weight, 1.0])
        while len(stack) > 1 and stack[-2][0] > stack[-1][0]:
            m1, w1, c1 = stack.pop()
            m0, w0, c0 = stack.pop()
            wsum = w0 + w1
            stack.append([(m0 * w0 + m1 * w1) / wsum, wsum, c0 + c1])
    out: list[float] = []
    for mean, _w, count in stack:
        out.extend([mean] * int(round(count)))
    return out


# ---------------------------------------------------------------------------
# Monte-Carlo detection curve
# ---------------------------------------------------------------------------
@dataclass
class DetectionCurve:
    """Detection probability vs injected bias, with Wilson intervals."""

    deltas: list[float]
    trials: int
    hits: list[int]
    p_hat: list[float]
    p_lo: list[float]
    p_hi: list[float]
    p_smooth: list[float]
    sigma: float = 0.0
    n_obs: int = 1
    seed: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "sigma": self.sigma, "n_obs": self.n_obs, "trials": self.trials,
            "seed": self.seed,
            "rows": [{"delta": d, "hits": h, "p": p, "ci_lo": lo, "ci_hi": hi,
                      "p_smooth": ps}
                     for d, h, p, lo, hi, ps in zip(self.deltas, self.hits,
                                                    self.p_hat, self.p_lo,
                                                    self.p_hi, self.p_smooth)],
        }

    @property
    def max_p(self) -> float:
        return max(self.p_hat) if self.p_hat else 0.0


def detection_curve(detector: Detector,
                    deltas: Iterable[float],
                    *,
                    sigma: float = 0.0,
                    n_obs: int = 1,
                    n_trials: int = 100,
                    seed: int | None = None,
                    rng: random.Random | None = None) -> DetectionCurve:
    """Inject ``delta`` (relative error) ``n_trials`` times and count rejections.

    ``y = delta + sigma/sqrt(n_obs) * Z`` per trial; ``detector(y)`` must return
    ``True`` when the criterion rejects the evidence.
    """
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")
    if n_obs <= 0:
        raise ValueError("n_obs must be positive")
    ladder = [float(d) for d in deltas]
    if not ladder:
        raise ValueError("deltas must be non-empty")
    if rng is None:
        rng = random.Random(seed)
    sd = sigma / math.sqrt(n_obs)
    hits: list[int] = []
    for delta in ladder:
        count = 0
        for _ in range(n_trials):
            y = delta + (sd * rng.gauss(0.0, 1.0) if sd > 0.0 else 0.0)
            if detector(y):
                count += 1
        hits.append(count)
    p_hat = [h / n_trials for h in hits]
    bounds = [wilson_interval(h, n_trials) for h in hits]
    return DetectionCurve(deltas=ladder, trials=n_trials, hits=hits,
                          p_hat=p_hat,
                          p_lo=[b[0] for b in bounds],
                          p_hi=[b[1] for b in bounds],
                          p_smooth=pava(p_hat, [n_trials] * len(p_hat)),
                          sigma=sigma, n_obs=n_obs, seed=seed)


def mde90_interpolate(curve: DetectionCurve, target: float = 0.90,
                      space: str = "probit") -> float | None:
    """Smallest bias whose smoothed detection probability reaches ``target``.

    Interpolation is done in **probit space** by default: for the Gaussian noise
    model the detection curve is a normal CDF in the bias, so ``z = Phi^-1(p)``
    is linear in ``delta`` and a two-point interpolation between ladder rungs is
    nearly exact (linear-in-``p`` interpolation over-estimates MDE_90 by roughly
    half a ladder gap when the curve is steep).  ``space="linear"`` reproduces
    the naive estimator for comparison.  ``None`` = censored curve.
    """
    if space not in ("probit", "linear"):
        raise ValueError(f"unknown space {space!r}")
    order = sorted(range(len(curve.deltas)), key=lambda i: curve.deltas[i])
    deltas = [curve.deltas[i] for i in order]
    p = [curve.p_smooth[i] for i in order]
    transform = (lambda v: v) if space == "linear" else (lambda v: norm_ppf(min(max(v, 1e-6), 1 - 1e-6)))
    t_target = transform(target)
    for i, pi in enumerate(p):
        if pi >= target:
            if i == 0:
                return deltas[0]
            t0, t1 = transform(p[i - 1]), transform(pi)
            d0, d1 = deltas[i - 1], deltas[i]
            if t1 == t0:
                return d1
            return d0 + (t_target - t0) * (d1 - d0) / (t1 - t0)
    return None


def mde90_bisect(detector: Detector,
                 *,
                 lo: float = 1e-9,
                 hi: float = 1.0,
                 sigma: float = 0.0,
                 n_obs: int = 1,
                 seed: int = 0,
                 rtol: float = 1e-9,
                 max_iter: int = 200) -> float | None:
    """Exact decision boundary of a deterministic detector (``sigma == 0``).

    Deterministic criteria have a step detection curve; their MDE_90 *is* the
    boundary, and the boundary is worth extracting exactly instead of reading
    it off a coarse ladder.  Returns ``None`` if ``detector(hi)`` is still
    ``False`` (no detection anywhere inside the bracket).
    """
    if not detector(hi):
        return None
    if detector(lo):
        return lo
    a, b = lo, hi
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        if detector(mid):
            b = mid
        else:
            a = mid
        if b - a <= rtol * max(b, 1e-12) or b - a < 1e-15:
            break
    return b


# ---------------------------------------------------------------------------
# logistic fit (binomial IRLS)
# ---------------------------------------------------------------------------
@dataclass
class LogisticFit:
    """Logistic fit of ``P(detect) = 1/(1+exp(-(b0 + b1*delta)))``."""

    b0: float
    b1: float
    se_b0: float
    se_b1: float
    deviance: float
    converged: bool
    separation: bool = False
    mde50: float | None = None
    mde90: float | None = None
    iterations: int = 0

    def predict(self, delta: float) -> float:
        return 1.0 / (1.0 + math.exp(-(self.b0 + self.b1 * delta)))

    def as_dict(self) -> dict[str, Any]:
        return {"slope_per_unit_bias": self.b1, "intercept": self.b0,
                "se_slope": self.se_b1, "se_intercept": self.se_b0,
                "deviance": self.deviance, "converged": self.converged,
                "separation": self.separation,
                "mde50": self.mde50, "mde90": self.mde90,
                "iterations": self.iterations}


def fit_logistic(deltas: Sequence[float], hits: Sequence[int],
                 trials: int, *, max_iter: int = 200,
                 tol: float = 1e-8) -> LogisticFit:
    """IRLS logistic fit of the detection curve (weights = binomial counts).

    The predictor is centred and scaled internally (``x' = (x - mean) / scale``)
    because bias ladders are O(0.01-0.2) in magnitude and an unscaled Hessian is
    numerically singular there; the fitted coefficients are transformed back so
    ``b1`` is per unit of bias.  Near-separated ladders (a deterministic step)
    have no finite MLE: the fit reports ``separation=True`` and leaves
    ``mde50``/``mde90`` as ``None`` rather than emitting a huge slope that looks
    like a measurement.
    """
    n = len(deltas)
    if n != len(hits) or n < 3:
        raise ValueError("need >= 3 matched delta/hit points")
    x = [float(d) for d in deltas]
    y = [h / trials for h in hits]
    if all(v == 0.0 for v in y) or all(v == 1.0 for v in y):
        return LogisticFit(b0=float("nan"), b1=float("nan"),
                           se_b0=float("nan"), se_b1=float("nan"),
                           deviance=float("nan"), converged=False,
                           separation=True)
    xm = sum(x) / n
    xs = statistics.pstdev(x)
    if xs <= 0.0:
        raise ValueError("deltas must not all be identical")
    z = [(xi - xm) / xs for xi in x]

    def deviance_of(c0: float, c1: float) -> float:
        total = 0.0
        for zi, yi in zip(z, y):
            eta = max(min(c0 + c1 * zi, 35.0), -35.0)
            p = 1.0 / (1.0 + math.exp(-eta))
            pc = min(max(p, 1e-12), 1 - 1e-12)
            yc = min(max(yi, 1e-12), 1 - 1e-12)
            total += 2.0 * trials * (yc * math.log(yc / pc)
                                     + (1 - yc) * math.log((1 - yc) / (1 - pc)))
        return total

    c0, c1 = 0.0, 1.0
    dev = deviance_of(c0, c1)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        s00 = s01 = s11 = 0.0
        g0 = g1 = 0.0
        for zi, yi in zip(z, y):
            eta = max(min(c0 + c1 * zi, 35.0), -35.0)
            p = 1.0 / (1.0 + math.exp(-eta))
            w = max(p * (1.0 - p) * trials, 1e-12)
            s00 += w
            s01 += w * zi
            s11 += w * zi * zi
            g0 += (yi - p) * trials
            g1 += (yi - p) * trials * zi
        det = s00 * s11 - s01 * s01
        if det <= 1e-12:
            break
        d0 = (s11 * g0 - s01 * g1) / det
        d1 = (s00 * g1 - s01 * g0) / det
        step = 1.0
        improved = False
        for _ in range(40):
            cand = deviance_of(c0 + step * d0, c1 + step * d1)
            if math.isfinite(cand) and cand <= dev:
                c0, c1 = c0 + step * d0, c1 + step * d1
                improved = True
                break
            step *= 0.5
        if not improved:
            converged = True          # at the optimum within machine precision
            break
        new_dev = deviance_of(c0, c1)
        if abs(dev - new_dev) <= tol * max(1.0, abs(dev)):
            dev = new_dev
            converged = True
            break
        dev = new_dev
    s00 = s01 = s11 = 0.0
    for zi in z:
        eta = max(min(c0 + c1 * zi, 35.0), -35.0)
        p = 1.0 / (1.0 + math.exp(-eta))
        w = max(p * (1.0 - p) * trials, 1e-12)
        s00 += w
        s01 += w * zi
        s11 += w * zi * zi
    det = s00 * s11 - s01 * s01
    ok = det > 1e-12 and math.isfinite(c0) and math.isfinite(c1)
    se_c0 = math.sqrt(s11 / det) if ok else float("nan")
    se_c1 = math.sqrt(s00 / det) if ok else float("nan")
    # back to per-unit-bias coordinates: eta = c0 + c1*(x-xm)/xs
    b1 = c1 / xs
    b0 = c0 - b1 * xm
    se_b1 = se_c1 / xs
    se_b0 = math.sqrt(max(se_c0 ** 2 + (xm / xs) ** 2 * se_c1 ** 2, 0.0)) if ok else float("nan")
    # near-step curves have no usable MLE: either the slope blows up or every
    # fitted probability is saturated (max Bernoulli variance ~ 0)
    max_var = 0.0
    for zi in z:
        eta = max(min(c0 + c1 * zi, 35.0), -35.0)
        p_i = 1.0 / (1.0 + math.exp(-eta))
        max_var = max(max_var, p_i * (1.0 - p_i))
    separation = bool(abs(c1) > 50.0 or max_var < 1e-4)
    mde50 = mde90 = None
    if ok and c1 > 0.0 and not separation:
        mde50 = (0.0 - c0) * xs / c1 + xm
        mde90 = (math.log(9.0) - c0) * xs / c1 + xm
        if mde50 < 0.0 or mde90 < 0.0:
            # a negative minimum detectable error is not a property of the
            # detector: the ladder did not contain the transition at all
            separation, mde50, mde90 = True, None, None
    return LogisticFit(b0=b0, b1=b1, se_b0=se_b0, se_b1=se_b1,
                       deviance=dev, converged=converged,
                       separation=separation, mde50=mde50, mde90=mde90,
                       iterations=it)


# ---------------------------------------------------------------------------
# closed-form references for standard detector statistics
# ---------------------------------------------------------------------------
def _step(delta: float, threshold: float, direction: str = "abs") -> bool:
    if direction == "abs":
        return abs(delta) > threshold
    if direction == "high":
        return delta > threshold
    if direction == "low":
        return delta < -threshold
    raise ValueError(f"unknown direction {direction!r}")


def power_point(delta: float, threshold: float, sigma: float,
                direction: str = "abs") -> float:
    """``P(reject)`` for ``|delta + sigma Z| > threshold`` (one observation)."""
    if sigma <= 0.0:
        return 1.0 if _step(delta, threshold, direction) else 0.0
    if direction == "abs":
        return (norm_cdf((delta - threshold) / sigma)
                + norm_cdf((-delta - threshold) / sigma))
    if direction == "high":
        return norm_cdf((delta - threshold) / sigma)
    if direction == "low":
        return norm_cdf((-delta - threshold) / sigma)
    raise ValueError(f"unknown direction {direction!r}")


def power_mean(delta: float, threshold: float, sigma: float, n_obs: int,
               direction: str = "abs") -> float:
    """``P(reject)`` for the mean of ``n_obs`` observations."""
    if n_obs <= 0:
        raise ValueError("n_obs must be positive")
    return power_point(delta, threshold, sigma / math.sqrt(n_obs), direction)


def power_bound(delta: float, threshold: float, sigma: float,
                k: float = 1.0, direction: str = "abs") -> float:
    """``P(reject)`` when the decision boundary is inflated by ``k*sigma``.

    This is the ASME V&V 20 shape ``|E| <= U_val``: the reference uncertainty is
    subtracted from the decision threshold, so a criterion that folds its own
    uncertainty budget into the threshold is systematically *less* sensitive.
    """
    return power_point(delta, threshold - k * sigma, sigma, direction)


def proportion_k_required(n: int, p_min: float, alpha: float = 0.05) -> int | None:
    """Smallest pass count *certifying* ``p >= p_min`` at confidence ``1-alpha``.

    ``k_req = min{k : P(Bin(n, p_min) >= k) <= alpha}`` (Clopper-Pearson lower
    bound >= p_min).  ``None`` means the criterion is **vacuous**: no outcome of
    ``n`` trials certifies ``p_min`` at that confidence.  This is the
    *certification* reading (a gate that must not accept a bad ledger).
    """
    for k in range(n + 1):
        if binomial_sf(k, n, p_min) <= alpha:
            return k
    return None


def proportion_samples_required(p_min: float, alpha: float = 0.05) -> int:
    """Smallest ``n`` able to certify ``p >= p_min`` at ``1-alpha`` (all-pass)."""
    if not 0.0 < p_min < 1.0:
        raise ValueError("p_min must be in (0, 1)")
    return int(math.ceil(math.log(alpha) / math.log(p_min)))


def proportion_k_crit(n: int, p_min: float, alpha: float = 0.05) -> int:
    """Rejection threshold of the alpha-level *detection* test at ``p = p_min``.

    ``k_crit = max{k : P(Bin(n, p_min) <= k) <= alpha}``; the criterion rejects
    when ``kills <= k_crit``.  Returns ``-1`` when the sample is too small to
    form a level-``alpha`` test at all (no rejection region exists), which is
    the power-free regime this module exists to expose.
    """
    k_crit = -1
    for k in range(n + 1):
        if binomial_cdf(k, n, p_min) <= alpha:
            k_crit = k
        else:
            break
    return k_crit


def power_proportion(deficit: float, n: int, p_min: float,
                     alpha: float = 0.05) -> float:
    """Power of the alpha-level detection test against a true rate deficit.

    ``deficit = 0`` is the boundary (power = ``alpha`` by construction), and the
    power rises as the true acceptance rate falls below ``p_min``.  A criterion
    whose sample is too small to form the test (``k_crit == -1``) has **zero
    power at every deficit** — it can never detect a degradation, however large.
    """
    k_crit = proportion_k_crit(n, p_min, alpha)
    if k_crit < 0:
        return 0.0
    p_true = max(0.0, min(1.0, p_min - deficit))
    return binomial_cdf(k_crit, n, p_true)


def mde90_analytic(statistic: str, threshold: float, *, sigma: float = 0.0,
                   n_obs: int = 1, direction: str = "abs", k: float = 1.0,
                   p_min: float = 0.9, alpha: float = 0.05,
                   lo: float = 0.0, hi: float | None = None,
                   tol: float = 1e-9, max_iter: int = 200) -> float | None:
    """Closed-form/semi-analytic MDE_90 for the four reference statistics."""
    def power(d: float) -> float:
        if statistic == "point":
            return power_point(d, threshold, sigma, direction)
        if statistic == "mean":
            return power_mean(d, threshold, sigma, n_obs, direction)
        if statistic == "bound":
            return power_bound(d, threshold, sigma, k, direction)
        if statistic == "proportion":
            return power_proportion(d, n_obs, p_min, alpha)
        raise ValueError(f"unknown statistic {statistic!r}")

    if statistic == "point" and sigma <= 0.0:
        return threshold if _step(threshold * 1.0000001, threshold, direction) else None
    if hi is None:
        if statistic == "proportion":
            hi = p_min
        else:
            hi = (threshold * 2.0 + 10.0 * sigma
                  + (5.0 * sigma / math.sqrt(max(n_obs, 1))))
        hi = max(hi, 1e-9)
    for _ in range(80):
        if power(hi) >= 0.9:
            break
        hi *= 2.0
        if hi > 1e6:
            return None
    if power(hi) < 0.9:
        return None
    a, b = lo, hi
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        if power(mid) >= 0.9:
            b = mid
        else:
            a = mid
        if b - a <= tol * max(abs(b), 1e-12):
            break
    return b


# ---------------------------------------------------------------------------
# certificates
# ---------------------------------------------------------------------------
def classify_mde(mde90: float | None, *, reference: float = 0.05,
                 weak: float = 0.10, blind: float = 0.20) -> str:
    """Certification class of a criterion, given its MDE_90."""
    if mde90 is None:
        return "CENSORED"
    if mde90 <= reference:
        return "STRONG"
    if mde90 <= weak:
        return "MARGINAL"
    if mde90 <= blind:
        return "WEAK"
    return "BLIND"


def certificate(criterion_id: str,
                detector: Detector,
                *,
                deltas: Sequence[float],
                statistic: str,
                threshold: float | None = None,
                sigma: float = 0.0,
                sigma_source: str = "declared 0 (deterministic evidence)",
                n_obs: int = 1,
                n_trials: int = 100,
                seed: int = 0,
                reference: float = 0.05,
                target: float = 0.90,
                theory: float | None = None,
                deterministic: bool = True,
                dense_scan: bool = True) -> dict[str, Any]:
    """Run the injection campaign for one criterion and issue its certificate.

    Estimators, all reported side by side so none of them can hide:

    ``mde90_mc_ladder``  interpolated from the N-trial ladder (the shipped
                         estimate; quantised by the ladder spacing),
    ``mde90_exact``      decision boundary by bisection (deterministic
                         detectors only -- the ladder cannot resolve a step),
    ``mde90_fit``        read off the logistic fit (smooth detectors only),
    ``mde90_theory``     closed-form/reference value supplied by the caller.

    ``mde90`` is the authoritative pick: exact boundary > MC interpolation >
    fit.  ``None`` means censored (never reached 90 % inside the ladder).
    """
    curve = detection_curve(detector, deltas, sigma=sigma, n_obs=n_obs,
                            n_trials=n_trials, seed=seed)
    fit = fit_logistic(curve.deltas, curve.hits, curve.trials)
    mde_mc = mde90_interpolate(curve, target)
    # baseline sanity: a detector that already fires on clean evidence (delta=0)
    # certifies nothing -- its MDE_90 is 0 by convention and its class must say so
    false_alarm = bool(detector(0.0)) if deterministic else False
    mde_exact = None
    if deterministic and sigma == 0.0 and not false_alarm:
        hi = max(max(curve.deltas) * 2.0, 1e-6)
        mde_exact = mde90_bisect(detector, lo=1e-15, hi=hi)
        while mde_exact is None and hi < 1e6:
            hi *= 4.0
            mde_exact = mde90_bisect(detector, lo=1e-15, hi=hi)
    if false_alarm:
        mde_best: float | None = 0.0
    elif not deterministic:
        # a stochastic detector has no decision boundary to bisect: the
        # closed-form power (when supplied) is authoritative, the Monte-Carlo
        # curve is the evidence
        mde_best = theory if theory is not None else mde_mc
    else:
        mde_best = (mde_exact if mde_exact is not None
                    else (mde_mc if mde_mc is not None else fit.mde90))
    catches = None if mde_best is None else bool(mde_best <= reference)
    valid = not false_alarm and mde_best is not None
    p_at_reference = None
    for i, d in enumerate(curve.deltas):
        if abs(d - reference) <= 1e-15:
            p_at_reference = {"delta": d, "hits": curve.hits[i],
                              "p": curve.p_hat[i],
                              "ci": [curve.p_lo[i], curve.p_hi[i]]}
            break
    dense = None
    if dense_scan:
        dense = _dense_scan(detector, sigma=sigma, n_obs=n_obs,
                            hi=max(max(curve.deltas), (mde_best or 0.0) * 4, 1e-3))
    return {
        "criterion": criterion_id,
        "statistic": statistic,
        "threshold": threshold,
        "sigma": sigma,
        "sigma_source": sigma_source,
        "n_obs": n_obs,
        "n_trials": n_trials,
        "seed": seed,
        "target_power": target,
        "reference_bias": reference,
        "mde90_mc_ladder": mde_mc,
        "mc_ladder_floor": bool(mde_mc is not None and curve.p_smooth
                                and curve.p_smooth[0] >= target),
        "mde90_exact": mde_exact,
        "mde90_fit": fit.mde90,
        "mde90_theory": theory,
        "mde90": mde_best,
        "deterministic_detector": deterministic,
        "mde90_source": ("deterministic-boundary" if mde_exact is not None
                         else ("closed-form" if (not deterministic and theory is not None)
                               else ("monte-carlo-interpolated" if mde_mc is not None
                                     else "logistic-fit"))),
        "class": ("FALSE_ALARM" if false_alarm
                  else classify_mde(mde_best, reference=reference)),
        "valid_detector": valid,
        "false_alarm_at_zero_bias": false_alarm,
        "catches_reference_bias": catches,
        "detection_at_reference": p_at_reference,
        "power_curve": curve.as_dict(),
        "logistic": fit.as_dict(),
        "censored": mde_best is None,
        "dense_detection_boundary": dense,
    }


def _dense_scan(detector: Detector, *, sigma: float, n_obs: int,
                hi: float, points: int = 121) -> dict[str, Any]:
    """Deterministic (noise-free core) scan: locates the boundary exactly.

    For ``sigma == 0`` this is the detector's step location; for ``sigma > 0`` it
    is the location where the *expected* error reaches the threshold, which
    keeps the reported curve and the reported boundary consistent.
    """
    if hi <= 0:
        return {"deltas": [], "detects": []}
    deltas = [hi * (i / (points - 1)) for i in range(points)]
    flags = [1 if detector(d) else 0 for d in deltas]
    boundary = None
    for i, flag in enumerate(flags):
        if flag:
            boundary = deltas[i]
            break
    return {"hi": hi, "points": points, "boundary": boundary,
            "deltas": deltas, "detects": flags}


def sha256_of_file(path: str) -> str:
    """Content hash used by the ledger's provenance block."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()