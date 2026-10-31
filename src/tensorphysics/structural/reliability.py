"""Component and system structural reliability: FORM / SORM / simulation.

Implements the classical first- and second-order reliability methods for a
vector of random variables described by the :class:`~tensorphysics.structural.
distributions.NatafTransform`:

* Hasofer-Lind / Rackwitz-Fiessler iteration for the design point and ``beta``,
* Breitung's curvature-corrected SORM estimate,
* crude Monte Carlo and design-point importance sampling references,
* Ditlevsen (bi-modal) bounds for series systems of limit states,
* design-point (sensitivity-consistent) and finite-difference sensitivities
  ``d beta / d theta`` for the distribution parameters.

The limit state is written as a Python callable ``g(x) -> float`` with the
convention ``g > 0`` = safe, ``g < 0`` = failure.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import exp, isfinite, sqrt
from typing import Callable, Iterable, Sequence

from ._linalg import (
    identity,
    inverse_from_cholesky,
    jacobi_eigen,
    matmul,
    tangent_basis,
)
from .distributions import (
    NatafTransform,
    bivariate_normal_cdf,
    norm_cdf,
    norm_ppf,
)

__all__ = [
    "ReliabilityProblem",
    "FormResult",
    "SormResult",
    "MonteCarloResult",
    "SystemReliabilityResult",
    "form_reliability",
    "sorm_correction",
    "monte_carlo_pf",
    "ditlevsen_bounds",
]


class ReliabilityError(RuntimeError):
    """A reliability analysis violated its numerical contract."""


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FormResult:
    """Outcome of a Hasofer-Lind / Rackwitz-Fiessler FORM analysis."""

    limit_state: str
    beta: float
    alpha: list[float]
    design_point_u: list[float]
    design_point_x: list[float]
    g_design: float
    pf: float
    converged: bool
    iterations: int
    gradient_norm: float
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "limit_state": self.limit_state,
            "beta": self.beta,
            "pf": self.pf,
            "alpha": list(self.alpha),
            "design_point_u": list(self.design_point_u),
            "design_point_x": list(self.design_point_x),
            "g_design": self.g_design,
            "converged": self.converged,
            "iterations": self.iterations,
            "gradient_norm": self.gradient_norm,
        }


@dataclass(frozen=True)
class SormResult:
    """Breitung second-order reliability estimate at the FORM design point."""

    limit_state: str
    beta_form: float
    beta_sorm: float
    pf_form: float
    pf_sorm: float
    principal_curvatures: list[float]
    valid: bool
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "limit_state": self.limit_state,
            "beta_form": self.beta_form,
            "beta_sorm": self.beta_sorm,
            "pf_form": self.pf_form,
            "pf_sorm": self.pf_sorm,
            "principal_curvatures": list(self.principal_curvatures),
            "valid": self.valid,
            "note": self.note,
        }


@dataclass(frozen=True)
class MonteCarloResult:
    limit_state: str
    pf: float
    beta: float
    samples: int
    failures: int
    censor_ok: bool
    method: str = "crude"
    std_error: float = 0.0

    def to_dict(self) -> dict:
        return {
            "limit_state": self.limit_state,
            "pf": self.pf,
            "beta": self.beta,
            "samples": self.samples,
            "failures": self.failures,
            "method": self.method,
            "std_error": self.std_error,
        }


@dataclass(frozen=True)
class SystemReliabilityResult:
    """Ditlevsen bi-modal bounds for a series (weakest-link) system."""

    limit_states: list[str]
    betas: list[float]
    pf_component: list[float]
    bimodal_rho: list[list[float]]
    bimodal_pf: list[list[float]]
    pf_lower: float
    pf_upper: float
    beta_lower: float
    beta_upper: float
    beta_point: float
    pf_point: float

    def to_dict(self) -> dict:
        return {
            "limit_states": list(self.limit_states),
            "betas": list(self.betas),
            "pf_component": list(self.pf_component),
            "bimodal_rho": [list(row) for row in self.bimodal_rho],
            "bimodal_pf": [list(row) for row in self.bimodal_pf],
            "pf_lower": self.pf_lower,
            "pf_upper": self.pf_upper,
            "beta_lower": self.beta_lower,
            "beta_upper": self.beta_upper,
            "beta_point": self.beta_point,
            "pf_point": self.pf_point,
        }


def _beta_from_pf(pf: float) -> float:
    if pf <= 0.0:
        return float("inf")
    if pf >= 1.0:
        return float("-inf")
    return -norm_ppf(min(max(pf, 1e-300), 1.0 - 1e-16))


# --------------------------------------------------------------------------- #
# Reliability problem
# --------------------------------------------------------------------------- #
class ReliabilityProblem:
    """A single limit state ``g(x)`` together with its random-variable model."""

    def __init__(
        self,
        transform: NatafTransform,
        limit_state: Callable[[Sequence[float]], float],
        name: str = "g",
        variable_names: Sequence[str] | None = None,
        batch_limit_state: Callable[[object], object] | None = None,
    ) -> None:
        self.transform = transform
        self.limit_state = limit_state
        self.batch_limit_state = batch_limit_state
        self.name = name
        n = transform.dimension
        self.variable_names = (
            list(variable_names)
            if variable_names is not None
            else [f"X{i + 1}" for i in range(n)]
        )
        if len(self.variable_names) != n:
            raise ValueError("variable_names must match the transform dimension")
        self._inv_lower = inverse_from_cholesky(transform._lower)

    # -- evaluation ----------------------------------------------------------
    def g_u(self, u: Sequence[float]) -> float:
        value = float(self.limit_state(self.transform.u_to_x(u)))
        if not isfinite(value):
            raise ReliabilityError(f"limit state {self.name!r} returned a non-finite value")
        return value

    def gradient_u(self, u: Sequence[float], step: float = 1e-6) -> list[float]:
        """Central finite-difference gradient of ``g`` in independent-u space."""
        n = len(u)
        grad = [0.0] * n
        base = list(u)
        for i in range(n):
            h = step * max(1.0, abs(base[i]))
            plus = list(base)
            minus = list(base)
            plus[i] += h
            minus[i] -= h
            grad[i] = (self.g_u(plus) - self.g_u(minus)) / (2.0 * h)
        return grad

    def hessian_u(self, u: Sequence[float], step: float = 1e-3) -> list[list[float]]:
        """Central finite-difference Hessian of ``g`` in u-space."""
        n = len(u)
        base = list(u)
        steps = [step * max(1.0, abs(base[i])) for i in range(n)]
        hess = [[0.0] * n for _ in range(n)]
        for i in range(n):
            hi = steps[i]
            up, down = list(base), list(base)
            up[i] += 2 * hi
            down[i] -= 2 * hi
            hess[i][i] = (self.g_u(up) - 2.0 * self.g_u(base) + self.g_u(down)) / (4.0 * hi * hi)
            for j in range(i + 1, n):
                hj = steps[j]
                values = []
                for si in (1.0, -1.0):
                    for sj in (1.0, -1.0):
                        point = list(base)
                        point[i] += si * hi
                        point[j] += sj * hj
                        values.append(self.g_u(point))
                value = (values[0] - values[1] - values[2] + values[3]) / (4.0 * hi * hj)
                hess[i][j] = hess[j][i] = value
        return hess

    # -- FORM ---------------------------------------------------------------
    def _radial_root(
        self,
        alpha: Sequence[float],
        initial: float = 1.0,
        tolerance: float = 1e-11,
        max_expansions: int = 30,
        max_iterations: int = 200,
    ) -> float:
        """Smallest ``lambda > 0`` with ``g(lambda * alpha) = 0`` (radial search).

        The Hasofer-Lind problem is solved by iterating
        ``u <- beta(alpha) * alpha`` with ``alpha = -grad g / |grad g|``. Each
        outer iteration needs a root of the one-dimensional function
        ``h(lambda) = g(lambda * alpha)``, obtained by geometric bracketing
        followed by Illinois (regula falsi) iteration. A fixed point of this map
        satisfies ``g = 0`` and ``alpha || -u``, i.e. the KKT conditions of
        ``min |u|`` subject to ``g(u) = 0``.
        """
        def h(lam: float) -> float:
            return self.g_u([lam * a for a in alpha])

        lo = 1e-9
        h_lo = h(lo)
        if h_lo <= 0.0:
            # failure domain reaches the origin along this ray
            return lo
        hi = max(initial, 1e-3)
        h_hi = h(hi)
        expansions = 0
        while h_hi > 0.0:
                expansions += 1
                if expansions > max_expansions:
                    raise ReliabilityError(
                        "limit state does not cross the radial search direction; "
                        "the design point may not exist along this ray"
                    )
                lo, h_lo = hi, h_hi
                hi *= 2.0
                h_hi = h(hi)
        # Illinois regula falsi on [lo, hi] with h(lo) > 0 > h(hi)
        side = 0
        best = hi
        for _ in range(max_iterations):
            if h_hi == 0.0:
                return hi
            if h_lo == 0.0:
                return lo
            lam = (lo * h_hi - hi * h_lo) / (h_hi - h_lo)
            if not (lo < lam < hi):
                lam = 0.5 * (lo + hi)
            h_lam = h(lam)
            best = lam
            if h_lam == 0.0 or abs(h_hi - h_lo) < 1e-300 or (hi - lo) < tolerance * max(1.0, hi):
                break
            if h_lam > 0.0:
                lo, h_lo = lam, h_lam
                if side == -1:
                    h_hi *= 0.5
                side = -1
            else:
                hi, h_hi = lam, h_lam
                if side == 1:
                    h_lo *= 0.5
                side = 1
        return best

    def form(
        self,
        start_u: Sequence[float] | None = None,
        max_iterations: int = 200,
        g_tolerance: float = 1e-7,
        u_tolerance: float = 1e-7,
        gradient_step: float = 1e-6,
        record_history: bool = True,
    ) -> FormResult:
        """Hasofer-Lind Rackwitz-Fiessler design point in the Nataf u-space.

        The iteration is the radial form of HL-RF: given the current gradient
        direction it locates the limit-state surface along that ray and moves to
        the intersection. It is far more robust than the classical secant update
        for strongly curved surfaces (e.g. the cubic Miner damage limit state),
        and it converges to the same Hasofer-Lind point (verified against Monte
        Carlo in the test suite). A damped HL-RF polish is applied once the
        radial iteration has converged, restoring quadratic convergence.
        """
        n = self.transform.dimension
        u = list(start_u) if start_u is not None else [0.0] * n
        if len(u) != n:
            raise ValueError("start_u dimension mismatch")
        history: list[dict] = []
        converged = False
        iterations = 0
        g_value = self.g_u(u)
        if g_value <= 0.0:
            raise ReliabilityError(
                f"limit state {self.name!r} is violated at the median point; "
                "the component is far outside the design range"
            )
        grad = self.gradient_u(u, gradient_step)
        grad_norm = sqrt(sum(v * v for v in grad))
        beta = sqrt(sum(v * v for v in u))
        for iteration in range(1, max_iterations + 1):
            iterations = iteration
            if grad_norm < 1e-14:
                raise ReliabilityError(
                    f"gradient of limit state {self.name!r} vanished; design point undefined"
                )
            alpha = [-v / grad_norm for v in grad]
            beta_new = self._radial_root(alpha, initial=max(beta, 1e-3))
            u = [beta_new * a for a in alpha]
            g_value = self.g_u(u)
            delta = abs(beta_new - beta)
            beta = beta_new
            if record_history:
                history.append(
                    {
                        "iteration": iteration,
                        "beta": beta,
                        "g": g_value,
                        "step": delta,
                        "alpha": list(alpha),
                    }
                )
            if abs(g_value) <= g_tolerance and delta <= u_tolerance * (1.0 + beta):
                converged = True
                break
            grad = self.gradient_u(u, gradient_step)
            grad_norm = sqrt(sum(v * v for v in grad))
        if converged and grad_norm > 0.0:
            u, g_value, grad_norm, polished = self._hlrf_polish(
                u, gradient_step, g_tolerance, u_tolerance
            )
            if polished and record_history:
                history.append(
                    {
                        "iteration": iterations + 1,
                        "beta": sqrt(sum(v * v for v in u)),
                        "g": g_value,
                        "step": 0.0,
                        "alpha": [-v / grad_norm for v in grad],
                        "polish": True,
                    }
                )
            beta = sqrt(sum(v * v for v in u))
        # recompute the gradient at the final point so alpha, grad_norm and the
        # polished design point are mutually consistent
        grad = self.gradient_u(u, gradient_step)
        grad_norm = sqrt(sum(v * v for v in grad))
        x_star = self.transform.u_to_x(u)
        alpha = (
            [-v / grad_norm for v in grad]
            if grad_norm > 0.0
            else [v / beta for v in u]
        )
        return FormResult(
            limit_state=self.name,
            beta=beta,
            alpha=alpha,
            design_point_u=u,
            design_point_x=x_star,
            g_design=g_value,
            pf=norm_cdf(-beta),
            converged=converged,
            iterations=iterations,
            gradient_norm=grad_norm,
            history=history,
        )

    def _hlrf_polish(
        self,
        u: Sequence[float],
        gradient_step: float,
        g_tolerance: float,
        u_tolerance: float,
        max_iterations: int = 20,
    ) -> tuple[list[float], float, float, bool]:
        """Damped HL-RF steps started from a converged radial point."""
        u = list(u)
        g_value = self.g_u(u)
        grad = self.gradient_u(u, gradient_step)
        grad_norm = sqrt(sum(v * v for v in grad))
        improved = False
        for _ in range(max_iterations):
            if grad_norm < 1e-14:
                break
            alpha = [-v / grad_norm for v in grad]
            projection = sum(a * b for a, b in zip(alpha, u))
            target = [(projection + g_value / grad_norm) * a for a in alpha]
            scale = max(grad_norm, 1e-12)
            current = sum(v * v for v in u) + (g_value / scale) ** 2
            lam = 1.0
            accepted = None
            for _ in range(12):
                candidate = [ui + lam * (ti - ui) for ui, ti in zip(u, target)]
                g_candidate = self.g_u(candidate)
                merit = sum(v * v for v in candidate) + (g_candidate / scale) ** 2
                if merit < current:
                    accepted = (candidate, g_candidate)
                    break
                lam *= 0.5
            if accepted is None:
                break
            u, g_value = accepted
            improved = True
            grad = self.gradient_u(u, gradient_step)
            grad_norm = sqrt(sum(v * v for v in grad))
            if abs(g_value) <= g_tolerance:
                break
        del u_tolerance
        return u, g_value, grad_norm, improved

    # -- SORM ---------------------------------------------------------------
    def sorm(
        self,
        form_result: FormResult | None = None,
        hessian_step: float = 1e-3,
        gradient_step: float = 1e-6,
    ) -> SormResult:
        """Breitung curvature correction about the FORM design point.

        The principal curvatures are the eigenvalues of ``P H P / |grad g|`` on
        the tangent space at the design point, where ``P = I - n n^T`` projects
        onto the tangent plane. With this sign convention a positive curvature
        means the limit-state surface bulges away from the origin (into the
        failure domain), so ``(1 + beta * kappa)^-1/2 < 1`` reduces the failure
        probability relative to the linearised FORM result; negative curvature
        increases it.
        """
        base = form_result if form_result is not None else self.form(gradient_step=gradient_step)
        u_star = list(base.design_point_u)
        grad = self.gradient_u(u_star, gradient_step)
        grad_norm = sqrt(sum(v * v for v in grad))
        if grad_norm <= 0.0:
            raise ReliabilityError("zero gradient at the design point; SORM undefined")
        hess = self.hessian_u(u_star, hessian_step)
        identity_matrix = identity(len(u_star))
        nn = [[a * b for b in grad] for a in grad]
        project = [
            [identity_matrix[i][j] - nn[i][j] / (grad_norm * grad_norm) for j in range(len(u_star))]
            for i in range(len(u_star))
        ]
        projected = matmul(matmul(project, hess), project)
        scaled = [[value / grad_norm for value in row] for row in projected]
        basis = tangent_basis(grad)
        m = len(basis)
        reduced = [[0.0] * m for _ in range(m)]
        for i in range(m):
            bi = basis[i]
            for j in range(m):
                bj = basis[j]
                total = 0.0
                for p in range(len(u_star)):
                    if bi[p] == 0.0:
                        continue
                    row = scaled[p]
                    total += bi[p] * sum(row[q] * bj[q] for q in range(len(u_star)))
                reduced[i][j] = total
        eigenvalues, _ = jacobi_eigen(reduced)
        curvatures = list(eigenvalues)
        factors = [1.0 + base.beta * kappa for kappa in curvatures]
        valid = all(factor > 1e-6 for factor in factors)
        if valid:
            pf_sorm = base.pf
            for factor in factors:
                pf_sorm /= sqrt(factor)
            beta_sorm = _beta_from_pf(pf_sorm)
            note = "Breitung curvature correction"
        else:
            pf_sorm = float("nan")
            beta_sorm = float("nan")
            note = "curvature factor 1+beta*kappa <= 0: Breitung formula not applicable"
        return SormResult(
            limit_state=self.name,
            beta_form=base.beta,
            beta_sorm=beta_sorm,
            pf_form=base.pf,
            pf_sorm=pf_sorm,
            principal_curvatures=curvatures,
            valid=valid,
            note=note,
        )

    # -- simulation ---------------------------------------------------------
    def monte_carlo_pf(
        self,
        samples: int = 1_000_000,
        seed: int = 2026,
        backend: str = "auto",
        return_values: bool = False,
    ) -> MonteCarloResult | tuple[MonteCarloResult, list[float]]:
        """Crude Monte Carlo estimate of ``Pf = P(g < 0)`` in u-space."""
        if samples <= 0:
            raise ValueError("samples must be positive")
        use_numpy = backend == "numpy"
        if backend == "auto":
            try:  # pragma: no cover - depends on environment
                import numpy  # noqa: F401

                use_numpy = True
            except Exception:  # pragma: no cover
                use_numpy = False
        if use_numpy:
            values = self._monte_carlo_values_numpy(samples, seed)
        else:
            values = self._monte_carlo_values_stdlib(samples, seed)
        failures = sum(1 for v in values if v < 0.0)
        pf = failures / float(samples)
        std_error = sqrt(max(pf * (1.0 - pf) / samples, 0.0))
        result = MonteCarloResult(
            limit_state=self.name,
            pf=pf,
            beta=_beta_from_pf(pf) if failures else float("inf"),
            samples=samples,
            failures=failures,
            censor_ok=failures >= 10,
            method="crude",
            std_error=std_error,
        )
        if return_values:
            return result, list(values)
        return result

    def _monte_carlo_values_numpy(self, samples: int, seed: int):  # pragma: no cover - numpy path
        import numpy as np

        from ._batch import batch_u_to_x

        rng = np.random.default_rng(seed)
        n = self.transform.dimension
        w = rng.standard_normal((samples, n))
        x = batch_u_to_x(self.transform, w)
        if self.batch_limit_state is not None:
            return np.asarray(self.batch_limit_state(x), dtype=float)
        return np.asarray([float(self.limit_state(row)) for row in x.tolist()], dtype=float)

    def _monte_carlo_values_stdlib(self, samples: int, seed: int) -> list[float]:
        import random

        rng = random.Random(seed)
        n = self.transform.dimension
        out: list[float] = []
        for _ in range(samples):
            u = [rng.gauss(0.0, 1.0) for _ in range(n)]
            out.append(float(self.limit_state(self.transform.u_to_x(u))))
        return out

    def importance_sampling_pf(
        self,
        samples: int = 200_000,
        seed: int = 2026,
        form_result: FormResult | None = None,
        backend: str = "auto",
    ) -> MonteCarloResult:
        """Design-point importance sampling (unit-variance shift)."""
        base = form_result if form_result is not None else self.form()
        u_star = list(base.design_point_u)
        n = len(u_star)
        use_numpy = backend == "numpy"
        if backend == "auto":
            try:  # pragma: no cover
                import numpy  # noqa: F401

                use_numpy = True
            except Exception:  # pragma: no cover
                use_numpy = False
        contributions: list[float] = []
        if use_numpy:  # pragma: no cover - numpy path
            import numpy as np

            from ._batch import batch_u_to_x

            rng = np.random.default_rng(seed + 1)
            w = rng.standard_normal((samples, n))
            u = w + np.asarray(u_star)
            x = batch_u_to_x(self.transform, u)
            if self.batch_limit_state is not None:
                g_values = np.asarray(self.batch_limit_state(x), dtype=float)
            else:
                g_values = np.asarray(
                    [float(self.limit_state(row)) for row in x.tolist()], dtype=float
                )
            exponent = -(w @ np.asarray(u_star)) - 0.5 * sum(v * v for v in u_star)
            weights = np.exp(exponent)
            mask = g_values < 0.0
            contributions = weights[mask]
            pf = float(contributions.sum() / float(samples))
            variance = float(((contributions - pf) ** 2).sum() / float(samples) / samples)
            failures = int(mask.sum())
        else:
            import random

            rng = random.Random(seed + 1)
            count = 0
            total = 0.0
            total_sq = 0.0
            for _ in range(samples):
                w = [rng.gauss(0.0, 1.0) for _ in range(n)]
                u = [wi + si for wi, si in zip(w, u_star)]
                ux = self.transform.u_to_x(u)
                if float(self.limit_state(ux)) < 0.0:
                    weight = exp(-sum(a * b for a, b in zip(w, u_star)) - 0.5 * sum(v * v for v in u_star))
                    count += 1
                    total += weight
                    total_sq += weight * weight
            pf = total / float(samples)
            mean_sq = total_sq / float(samples)
            variance = max(mean_sq - pf * pf, 0.0) / samples
            failures = count
        return MonteCarloResult(
            limit_state=self.name,
            pf=pf,
            beta=_beta_from_pf(pf),
            samples=samples,
            failures=failures,
            censor_ok=failures >= 10,
            method="importance_sampling",
            std_error=sqrt(max(variance, 0.0)),
        )

    # -- sensitivities ------------------------------------------------------
    def design_point_sensitivity(
        self, form_result: FormResult | None = None
    ) -> dict[tuple[str, str], float]:
        """``d beta / d theta = alpha^T d u / d theta`` at a fixed design point.

        This is the classical design-point sensitivity (Madsen et al. 1986): by
        the envelope theorem the movement of the design point contributes
        nothing at first order, so only the parametric derivative of the
        probability transformation is needed. Exact for linear limit states
        with normal marginals.
        """
        base = form_result if form_result is not None else self.form()
        out: dict[tuple[str, str], float] = {}
        for i, marginal in enumerate(self.transform.marginals):
            for param in marginal.param_names:
                du = self.transform.d_theta_vector(base.design_point_x, i, param)
                out[(self.variable_names[i], param)] = sum(
                    a * b for a, b in zip(base.alpha, du)
                )
        return out

    def sensitivity_finite_difference(
        self,
        form_result: FormResult | None = None,
        relative_step: float = 1e-4,
        which: Iterable[tuple[str, str]] | None = None,
    ) -> dict[tuple[str, str], float]:
        """Central-difference ``d beta / d theta`` obtained by re-solving FORM."""
        base = form_result if form_result is not None else self.form()
        del base
        out: dict[tuple[str, str], float] = {}
        for i, marginal in enumerate(self.transform.marginals):
            for param in marginal.param_names:
                key = (self.variable_names[i], param)
                if which is not None and key not in set(which):
                    continue
                nominal = getattr(marginal, param)
                h = relative_step * max(abs(nominal), 1e-6)
                betas = []
                for sign in (1.0, -1.0):
                    perturbed = replace(marginal, **{param: nominal + sign * h})
                    marginals = list(self.transform.marginals)
                    marginals[i] = perturbed
                    transform = NatafTransform(
                        marginals, correlation=self.transform.correlation
                    )
                    problem = ReliabilityProblem(
                        transform, self.limit_state, name=self.name, variable_names=self.variable_names
                    )
                    betas.append(problem.form().beta)
                out[key] = (betas[0] - betas[1]) / (2.0 * h)
        return out


# --------------------------------------------------------------------------- #
# Module level conveniences
# --------------------------------------------------------------------------- #
def form_reliability(
    transform: NatafTransform,
    limit_state: Callable[[Sequence[float]], float],
    name: str = "g",
) -> FormResult:
    return ReliabilityProblem(transform, limit_state, name=name).form()


def sorm_correction(
    transform: NatafTransform,
    limit_state: Callable[[Sequence[float]], float],
    name: str = "g",
) -> SormResult:
    return ReliabilityProblem(transform, limit_state, name=name).sorm()


def monte_carlo_pf(
    transform: NatafTransform,
    limit_state: Callable[[Sequence[float]], float],
    samples: int = 1_000_000,
    seed: int = 2026,
    name: str = "g",
) -> MonteCarloResult:
    return ReliabilityProblem(transform, limit_state, name=name).monte_carlo_pf(
        samples=samples, seed=seed
    )


def ditlevsen_bounds(results: Sequence[FormResult]) -> SystemReliabilityResult:
    """Ditlevsen bi-modal bounds for the union of component failure events.

    ``rho_ij = alpha_i . alpha_j`` (design-point direction cosines) is used as
    the equivalent linear correlation, and ``P_ij = Phi2(-beta_i, -beta_j, rho_ij)``.
    """
    if not results:
        raise ValueError("at least one FORM result is required")
    m = len(results)
    betas = [r.beta for r in results]
    pf = [r.pf for r in results]
    rho = [[1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]
    pair = [[0.0] * m for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            dot = sum(a * b for a, b in zip(results[i].alpha, results[j].alpha))
            dot = min(max(dot, -0.999999), 0.999999)
            rho[i][j] = rho[j][i] = dot
            value = bivariate_normal_cdf(-betas[i], -betas[j], dot)
            pair[i][j] = pair[j][i] = value
    lower = pf[0]
    for i in range(1, m):
        term = pf[i] - sum(pair[i][j] for j in range(i))
        lower += max(term, 0.0)
    upper = sum(pf)
    for i in range(1, m):
        upper -= max(pair[i][j] for j in range(i))
    upper = min(upper, 1.0)
    point = 0.5 * (lower + upper)
    return SystemReliabilityResult(
        limit_states=[r.limit_state for r in results],
        betas=betas,
        pf_component=pf,
        bimodal_rho=rho,
        bimodal_pf=pair,
        pf_lower=lower,
        pf_upper=upper,
        beta_lower=_beta_from_pf(upper),
        beta_upper=_beta_from_pf(lower),
        beta_point=_beta_from_pf(point),
        pf_point=point,
    )