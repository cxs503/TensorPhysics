"""Marginal distributions and the Nataf transformation for reliability analysis.

Standard library only: the structural reliability core must remain usable in a
minimal TensorPhysics install (see ``pyproject.toml``: no mandatory
dependencies). Everything below is deterministic, seeded or analytic.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, expm1, isfinite, log, log1p, pi, sqrt
from typing import Callable, Sequence

from ._linalg import cholesky, forward_solve

__all__ = [
    "Distribution",
    "NormalDistribution",
    "LognormalDistribution",
    "GumbelMaxDistribution",
    "ExponentialDistribution",
    "normal_from_moments",
    "gumbel_max_from_moments",
    "lognormal_from_moments",
    "norm_pdf",
    "norm_cdf",
    "norm_ppf",
    "clip_probability",
    "bivariate_normal_cdf",
    "NatafTransform",
]

_SQRT2 = sqrt(2.0)
_INV_SQRT_2PI = 1.0 / sqrt(2.0 * pi)
_EULER_GAMMA = 0.5772156649015329

# 20-point Gauss-Hermite quadrature for the standard normal weight
# (nodes = sqrt(2) * hermgauss(20) nodes, weights = hermgauss weights / sqrt(pi)).
# Regenerated and cross-checked in tests/test_structural_distributions.py.
_GH_NODES = (
    -7.619048541679759, -6.510590157013655, -5.5787388058932015,
    -4.734581334046055, -3.9439673506573163, -3.189014816553389,
    -2.458663611172368, -1.745247320814127, -1.0429453488027511,
    -0.346964157081356, 0.346964157081356, 1.0429453488027511,
    1.745247320814127, 2.458663611172368, 3.189014816553389,
    3.9439673506573163, 4.734581334046055, 5.5787388058932015,
    6.510590157013655, 7.619048541679759,
)
_GH_WEIGHTS = (
    1.2578006724379234e-13, 2.4820623623151755e-10, 6.127490259982928e-08,
    4.402121090230851e-06, 0.00012882627996192928, 0.0018301031310804933,
    0.013997837447101022, 0.0615063720639769, 0.16173933398399998,
    0.2607930634495549, 0.2607930634495549, 0.16173933398399998,
    0.0615063720639769, 0.013997837447101022, 0.0018301031310804933,
    0.00012882627996192928, 4.402121090230851e-06, 6.127490259982928e-08,
    2.4820623623151755e-10, 1.2578006724379234e-13,
)


# --------------------------------------------------------------------------- #
# Standard normal helpers
# --------------------------------------------------------------------------- #
def norm_pdf(z: float) -> float:
    return _INV_SQRT_2PI * exp(-0.5 * z * z)


def norm_cdf(z: float) -> float:
    """Standard normal CDF with full double-precision tail behaviour."""
    from math import erfc

    if z <= 0.0:
        return 0.5 * erfc(-z / _SQRT2)
    return 1.0 - 0.5 * erfc(z / _SQRT2)


# Acklam's rational approximation for the inverse normal CDF (|rel err| < 1.2e-9),
# refined by one Halley step to reach ~1e-15.
_A = (
    -3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
    1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00,
)
_B = (
    -5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
    6.680131188771972e01, -1.328068155288572e01,
)
_C = (
    -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
    -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00,
)
_D = (
    7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
    3.754408661907416e00,
)
_P_LOW = 0.02425
_P_CLIP_LOW = 1e-300
_P_CLIP_HIGH = 1.0 - 1e-16


def clip_probability(p: float) -> float:
    """Clamp a CDF value into the open interval (0, 1) before quantile calls."""
    return min(max(p, _P_CLIP_LOW), _P_CLIP_HIGH)


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (quantile function)."""
    if not isfinite(p) or p <= 0.0 or p >= 1.0:
        raise ValueError(f"norm_ppf requires 0 < p < 1, got {p!r}")
    p = clip_probability(p)
    if p < _P_LOW:
        q = sqrt(-2.0 * log(p))
        x = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        )
    elif p > 1.0 - _P_LOW:
        q = sqrt(-2.0 * log(1.0 - p))
        x = -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        )
    else:
        q = p - 0.5
        r = q * q
        x = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / (
            ((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0
        )
    # Halley refinement (quadratic convergence on the CDF residual).
    for _ in range(2):
        e = norm_cdf(x) - p
        u = e * sqrt(2.0 * pi) * exp(0.5 * x * x)
        x -= u / (1.0 + 0.5 * x * u)
    return x


def _simpson(f: Callable[[float], float], a: float, b: float, n: int = 2000) -> float:
    if n % 2:
        n += 1
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        total += (4.0 if i % 2 else 2.0) * f(a + i * h)
    return total * h / 3.0


def bivariate_normal_cdf(h: float, k: float, rho: float) -> float:
    """``P(Z1 <= h, Z2 <= k)`` for a standard bivariate normal with correlation rho.

    Evaluated as ``int_{-inf}^{h} phi(t) Phi((k - rho t)/sqrt(1-rho^2)) dt`` with
    composite Simpson quadrature on a truncated domain; the truncation at -9.5
    contributes < 1e-19 to the integral.
    """
    if not (-1.0 < rho < 1.0):
        raise ValueError("bivariate correlation must satisfy |rho| < 1")
    if h > 10.0 and k > 10.0:
        return 1.0
    scale = sqrt(1.0 - rho * rho)

    def integrand(t: float) -> float:
        return norm_pdf(t) * norm_cdf((k - rho * t) / scale)

    lower = min(-9.5, h - 11.0)
    if h <= lower:
        return _simpson(integrand, -12.0, h, 2000) if h > -12.0 else 0.0
    return _simpson(integrand, lower, h, 2000)


# --------------------------------------------------------------------------- #
# Marginal distributions
# --------------------------------------------------------------------------- #
class Distribution:
    """Interface implemented by every marginal distribution.

    ``param_names`` names the *physical* parameters used for sensitivity
    analysis. ``cdf_param_derivative`` returns dF(x; theta)/dtheta for a single
    parameter, which lets the reliability core differentiate the Nataf
    transformation analytically instead of re-solving the reliability problem.
    """

    name: str = "distribution"
    param_names: tuple[str, ...] = ()

    def mean(self) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def std(self) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def pdf(self, x: float) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def cdf(self, x: float) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def ppf(self, p: float) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def cdf_param_derivative(self, x: float, param: str) -> float:  # pragma: no cover
        raise NotImplementedError

    def ppf_z(self, z: float) -> float:
        """Inverse CDF evaluated at ``Phi(z)`` without forming the probability.

        Distributions with a closed-form expression in ``z`` override this to
        stay exact in the far tail, where ``Phi(z)`` saturates at 1 in double
        precision (the design-point search then stalls).
        """
        return self.ppf(clip_probability(norm_cdf(z)))

    def z_of_x(self, x: float) -> float:
        """Standard-normal coordinate ``Phi^{-1}(F(x))`` of a physical value."""
        return norm_ppf(clip_probability(self.cdf(x)))

    def equivalent_normal(self, x: float) -> tuple[float, float]:
        """Rackwitz-Fiessler equivalent normal (mean, std) at ``x``."""
        p = clip_probability(self.cdf(x))
        z = norm_ppf(p)
        sigma = norm_pdf(z) / max(self.pdf(x), 1e-300)
        return x - sigma * z, sigma

    def _check_x(self, x: float) -> None:
        if not isfinite(x):
            raise ValueError(f"{self.name}: x must be finite, got {x!r}")


@dataclass(frozen=True)
class NormalDistribution(Distribution):
    mu: float
    sigma: float
    name: str = "normal"
    param_names: tuple[str, ...] = ("mu", "sigma")

    def __post_init__(self) -> None:
        if not self.sigma > 0.0:
            raise ValueError("sigma must be positive")

    def mean(self) -> float:
        return self.mu

    def std(self) -> float:
        return self.sigma

    def pdf(self, x: float) -> float:
        self._check_x(x)
        return norm_pdf((x - self.mu) / self.sigma) / self.sigma

    def cdf(self, x: float) -> float:
        self._check_x(x)
        return norm_cdf((x - self.mu) / self.sigma)

    def ppf(self, p: float) -> float:
        return self.mu + self.sigma * norm_ppf(p)

    def cdf_param_derivative(self, x: float, param: str) -> float:
        self._check_x(x)
        z = (x - self.mu) / self.sigma
        f = norm_pdf(z) / self.sigma
        if param == "mu":
            return -f
        if param == "sigma":
            return -z * f
        raise ValueError(f"unknown parameter {param!r} for {self.name}")

    def ppf_z(self, z: float) -> float:
        return self.mu + self.sigma * z

    def z_of_x(self, x: float) -> float:
        return (x - self.mu) / self.sigma

    def equivalent_normal(self, x: float) -> tuple[float, float]:
        return self.mu, self.sigma


@dataclass(frozen=True)
class LognormalDistribution(Distribution):
    """Lognormal marginal, parametrised by the underlying normal (mu_ln, sigma_ln)."""

    mu_ln: float
    sigma_ln: float
    name: str = "lognormal"
    param_names: tuple[str, ...] = ("mu_ln", "sigma_ln")

    def __post_init__(self) -> None:
        if not self.sigma_ln > 0.0:
            raise ValueError("sigma_ln must be positive")

    def mean(self) -> float:
        return exp(self.mu_ln + 0.5 * self.sigma_ln**2)

    def std(self) -> float:
        m = self.mean()
        return m * sqrt(exp(self.sigma_ln**2) - 1.0)

    def _z(self, x: float) -> float:
        if x <= 0.0:
            raise ValueError("lognormal support is x > 0")
        return (log(x) - self.mu_ln) / self.sigma_ln

    def pdf(self, x: float) -> float:
        self._check_x(x)
        z = self._z(x)
        return norm_pdf(z) / (x * self.sigma_ln)

    def cdf(self, x: float) -> float:
        self._check_x(x)
        if x <= 0.0:
            return 0.0
        return norm_cdf(self._z(x))

    def ppf(self, p: float) -> float:
        return exp(self.mu_ln + self.sigma_ln * norm_ppf(p))

    def ppf_z(self, z: float) -> float:
        return exp(self.mu_ln + self.sigma_ln * z)

    def z_of_x(self, x: float) -> float:
        return self._z(x)

    def cdf_param_derivative(self, x: float, param: str) -> float:
        self._check_x(x)
        if x <= 0.0:
            return 0.0
        z = self._z(x)
        f = norm_pdf(z) / (x * self.sigma_ln)
        if param == "mu_ln":
            return -x * f
        if param == "sigma_ln":
            return -z * x * f
        raise ValueError(f"unknown parameter {param!r} for {self.name}")


@dataclass(frozen=True)
class GumbelMaxDistribution(Distribution):
    """Gumbel (maximum / type I) extreme-value marginal."""

    loc: float
    scale: float
    name: str = "gumbel_max"
    param_names: tuple[str, ...] = ("loc", "scale")

    def __post_init__(self) -> None:
        if not self.scale > 0.0:
            raise ValueError("scale must be positive")

    def mean(self) -> float:
        return self.loc + _EULER_GAMMA * self.scale

    def std(self) -> float:
        return pi * self.scale / sqrt(6.0)

    def _y(self, x: float) -> float:
        return (x - self.loc) / self.scale

    def pdf(self, x: float) -> float:
        self._check_x(x)
        y = self._y(x)
        return exp(-(y + exp(-y))) / self.scale

    def cdf(self, x: float) -> float:
        self._check_x(x)
        y = self._y(x)
        return exp(-exp(-y))

    def ppf(self, p: float) -> float:
        if not 0.0 < p < 1.0:
            raise ValueError("Gumbel ppf requires 0 < p < 1")
        return self.loc - self.scale * log(-log(p))

    def ppf_z(self, z: float) -> float:
        """Exact quantile at ``Phi(z)`` using a numerically stable tail form.

        For ``z > 0`` the upper-tail probability ``y = Phi(-z)`` is formed
        directly (``erfc`` tail) and ``-log(Phi(z)) = -log1p(-y)`` is evaluated
        without cancellation, so the transform stays invertible far beyond
        ``|u| = 8``.
        """
        if z <= 0.0:
            u = -log(clip_probability(norm_cdf(z)))
        else:
            tail = clip_probability(norm_cdf(-z))
            u = -log1p(-tail) if tail < 0.5 else -log(clip_probability(norm_cdf(z)))
        return self.loc - self.scale * log(u)

    def cdf_param_derivative(self, x: float, param: str) -> float:
        self._check_x(x)
        y = self._y(x)
        f = exp(-(y + exp(-y))) / self.scale
        if param == "loc":
            return -f
        if param == "scale":
            return -y * f
        raise ValueError(f"unknown parameter {param!r} for {self.name}")


@dataclass(frozen=True)
class ExponentialDistribution(Distribution):
    """Exponential marginal (rate ``lam``), used for load-side verification cases."""

    lam: float
    name: str = "exponential"
    param_names: tuple[str, ...] = ("lam",)

    def __post_init__(self) -> None:
        if not self.lam > 0.0:
            raise ValueError("exponential rate must be positive")

    def mean(self) -> float:
        return 1.0 / self.lam

    def std(self) -> float:
        return 1.0 / self.lam

    def pdf(self, x: float) -> float:
        self._check_x(x)
        if x < 0.0:
            return 0.0
        return self.lam * exp(-self.lam * x)

    def cdf(self, x: float) -> float:
        self._check_x(x)
        if x <= 0.0:
            return 0.0
        return -expm1(-self.lam * x)

    def ppf(self, p: float) -> float:
        if not 0.0 < p < 1.0:
            raise ValueError("exponential ppf requires 0 < p < 1")
        return -log1p(-p) / self.lam

    def ppf_z(self, z: float) -> float:
        """Quantile at ``Phi(z) = 1 - exp(-lam x)``, i.e. ``x = -log(Phi(-z))/lam``.

        ``Phi(-z)`` is formed from the accurate erfc tail, so no cancellation
        occurs for large positive ``z``.
        """
        return -log(clip_probability(norm_cdf(-z))) / self.lam

    def z_of_x(self, x: float) -> float:
        if x <= 0.0:
            return norm_ppf(clip_probability(0.0))
        survival = exp(-self.lam * x)
        return -norm_ppf(clip_probability(survival))

    def cdf_param_derivative(self, x: float, param: str) -> float:
        self._check_x(x)
        if param != "lam":
            raise ValueError(f"unknown parameter {param!r} for {self.name}")
        if x < 0.0:
            return 0.0
        return x * exp(-self.lam * x)


def normal_from_moments(mean: float, cov: float = 0.0, sigma: float | None = None) -> NormalDistribution:
    s = sigma if sigma is not None else mean * cov
    return NormalDistribution(mu=mean, sigma=s)


def gumbel_max_from_moments(mean: float, cov: float = 0.0, sigma: float | None = None) -> GumbelMaxDistribution:
    s = sigma if sigma is not None else mean * cov
    if s <= 0.0:
        raise ValueError("Gumbel standard deviation must be positive")
    scale = s * sqrt(6.0) / pi
    return GumbelMaxDistribution(loc=mean - _EULER_GAMMA * scale, scale=scale)


def lognormal_from_moments(mean: float, cov: float = 0.0, sigma: float | None = None) -> LognormalDistribution:
    s = sigma if sigma is not None else mean * cov
    if mean <= 0.0 or s <= 0.0:
        raise ValueError("lognormal requires mean > 0 and sigma > 0")
    var_ratio = (s / mean) ** 2
    sigma_ln = sqrt(log(1.0 + var_ratio))
    return LognormalDistribution(mu_ln=log(mean) - 0.5 * sigma_ln**2, sigma_ln=sigma_ln)


# --------------------------------------------------------------------------- #
# Nataf transformation
# --------------------------------------------------------------------------- #
class NatafTransform:
    """Nataf (Gaussian-copula) transformation between physical and u-space.

    ``u`` denotes independent standard normal coordinates (the space in which
    the reliability index is a Euclidean distance). The physical correlation
    matrix is mapped to the Gaussian-copula correlation ``rho0`` by solving, for
    each pair,

        rho_ij = E[ zeta_i zeta_j ] = int int  zeta_i(z_i) zeta_j(z_j) phi2(z_i, z_j; rho0) dz_i dz_j

    with ``zeta_k = (F_k^{-1}(Phi(z_k)) - mu_k)/sigma_k`` and 20-point
    Gauss-Hermite quadrature (exact for normal marginals up to round-off).
    """

    def __init__(
        self,
        marginals: Sequence[Distribution],
        correlation: Sequence[Sequence[float]] | None = None,
        rho0: Sequence[Sequence[float]] | None = None,
        tolerance: float = 1e-8,
        max_iterations: int = 200,
    ) -> None:
        if not marginals:
            raise ValueError("at least one marginal distribution is required")
        self.marginals = list(marginals)
        n = len(self.marginals)
        self.dimension = n
        if correlation is None:
            correlation = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        self.correlation = [[float(correlation[i][j]) for j in range(n)] for i in range(n)]
        for i in range(n):
            if abs(self.correlation[i][i] - 1.0) > 1e-12:
                raise ValueError("correlation matrix diagonal must be 1")
            for j in range(i + 1, n):
                if abs(self.correlation[i][j] - self.correlation[j][i]) > 1e-12:
                    raise ValueError("correlation matrix must be symmetric")
            for j in range(n):
                if abs(self.correlation[i][j]) >= 1.0 and i != j:
                    raise ValueError("off-diagonal correlations must satisfy |rho| < 1")
        if rho0 is None:
            rho0 = self._solve_rho0(tolerance=tolerance, max_iterations=max_iterations)
        self.rho0 = [[float(rho0[i][j]) for j in range(n)] for i in range(n)]
        self._lower = cholesky(self.rho0)

    # -- Nataf correlation mapping -------------------------------------------
    def _pair_integral(self, i: int, j: int, rho0: float) -> float:
        mi, mj = self.marginals[i], self.marginals[j]
        mui, si = mi.mean(), mi.std()
        muj, sj = mj.mean(), mj.std()
        scale = sqrt(max(1.0 - rho0 * rho0, 1e-300))
        total = 0.0
        for a, wa in zip(_GH_NODES, _GH_WEIGHTS):
            x_i = (mi.ppf_z(a) - mui) / si
            inner = 0.0
            for b, wb in zip(_GH_NODES, _GH_WEIGHTS):
                z_j = rho0 * a + scale * b
                x_j = (mj.ppf_z(z_j) - muj) / sj
                inner += wb * x_i * x_j
            total += wa * inner
        return total

    def _solve_rho0(self, tolerance: float, max_iterations: int) -> list[list[float]]:
        n = self.dimension
        rho0 = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                target = self.correlation[i][j]
                if target == 0.0:
                    continue
                identical = (
                    type(self.marginals[i]) is type(self.marginals[j])
                    and abs(self.marginals[i].std() - self.marginals[j].std()) < 1e-15
                )
                if isinstance(self.marginals[i], NormalDistribution) and isinstance(
                    self.marginals[j], NormalDistribution
                ):
                    rho0[i][j] = rho0[j][i] = target
                    continue
                lo, hi = -0.999, 0.999
                value = target
                for _ in range(max_iterations):
                    value = 0.5 * (lo + hi)
                    computed = self._pair_integral(i, j, value)
                    if abs(computed - target) < tolerance:
                        break
                    if computed < target:
                        lo = value
                    else:
                        hi = value
                del identical
                rho0[i][j] = rho0[j][i] = value
        return rho0

    # -- forward / backward mapping ------------------------------------------
    def u_to_z(self, u: Sequence[float]) -> list[float]:
        n = self.dimension
        lower = self._lower
        z = [0.0] * n
        for i in range(n):
            s = 0.0
            for k in range(i):
                s += lower[i][k] * u[k]
            z[i] = s + lower[i][i] * u[i]
        return z

    def u_to_x(self, u: Sequence[float]) -> list[float]:
        z = self.u_to_z(u)
        return [m.ppf_z(zi) for m, zi in zip(self.marginals, z)]

    def x_to_z(self, x: Sequence[float]) -> list[float]:
        return [m.z_of_x(xi) for m, xi in zip(self.marginals, x)]

    def x_to_u(self, x: Sequence[float]) -> list[float]:
        return forward_solve(self._lower, self.x_to_z(list(x)))

    def jacobian_zx(self, x: Sequence[float]) -> list[float]:
        """Diagonal d z_i / d x_i = phi(z_i) / f_i(x_i)."""
        z = self.x_to_z(list(x))
        return [norm_pdf(zi) / max(m.pdf(xi), 1e-300) for m, zi, xi in zip(self.marginals, z, x)]

    def gradient_x_to_u(self, grad_x: Sequence[float], x: Sequence[float]) -> list[float]:
        """Chain rule: grad_u g = L^T diag(dz/dx) grad_x g."""
        diag = self.jacobian_zx(x)
        scaled = [d * g for d, g in zip(diag, grad_x)]
        n = self.dimension
        out = [0.0] * n
        for i in range(n):
            s = 0.0
            for k in range(i, n):
                s += self._lower[k][i] * scaled[k]
            out[i] = s
        return out

    def d_theta_vector(
        self, x: Sequence[float], index: int, param: str, delta: float = 1e-6
    ) -> list[float]:
        """d u / d theta for a parameter of marginal ``index``, evaluated at fixed ``x``.

        The parametric derivative is analytic: ``d z_i / d theta = (dF_i/d theta)/phi(z_i)``
        and ``d u = L^{-1} d z``. Used for the design-point sensitivity
        ``d beta / d theta = alpha^T d u / d theta``.
        """
        n = self.dimension
        marginal = self.marginals[index]
        z = self.x_to_z(list(x))
        d_z = [0.0] * n
        df = marginal.cdf_param_derivative(x[index], param)
        d_z[index] = df / max(norm_pdf(z[index]), 1e-300)
        del delta
        return forward_solve(self._lower, d_z)

    def sample_physical(self, n: int, seed: int | None = None) -> list[list[float]]:
        """Draw ``n`` correlated samples in physical space (seeded, stdlib random)."""
        import random

        rng = random.Random(seed)
        samples = []
        for _ in range(n):
            u = [rng.gauss(0.0, 1.0) for _ in range(self.dimension)]
            samples.append(self.u_to_x(u))
        return samples