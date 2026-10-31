"""Probabilistic half of the unified chain: FORM / SORM / system / design inverse.

All iteration, curvature and simulation machinery lives in the W3-N04 core
(:mod:`tensorphysics.structural.reliability`); this module only *binds* the
deterministic ultimate strength (:mod:`.limit_states`) to it and adds the
design-oriented quantities that the unified framework promises:

* Hasofer-Lind Rackwitz-Fiessler FORM with the Nataf transformation of the
  correlated marginals (same solver as W3-N04, so the two capabilities cannot
  drift apart);
* Breitung SORM curvature correction at the FORM design point;
* Ditlevsen bi-modal bounds over the three limit states of the panel;
* Monte Carlo (1e6) cross-checks of the FORM/SORM indices and of the system
  probability;
* first-order parameter sensitivities ``d beta / d mu_X`` (design-point /
  envelope-theorem derivative) validated against finite differences of the
  re-solved FORM index;
* mean-value first-order (MVFOSM) index as the independent closed-form
  reference for the target-reliability thickness inversion;
* the inverse design: bisection on the as-built thickness such that
  ``beta(t) = beta_target`` (``beta_target = 4.0``, ISO 19906 annual target).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..structural.distributions import (
    NormalDistribution,
    gumbel_max_from_moments,
    lognormal_from_moments,
)
from ..structural.ice_structure import VariableSet
from ..structural.reliability import (
    FormResult,
    MonteCarloResult,
    ReliabilityError,
    ReliabilityProblem,
    SormResult,
    ditlevsen_bounds,
)
from .limit_states import (
    DEFAULT_MODES,
    MODE_DEFLECTION,
    MODE_ELASTIC,
    MODE_ULTIMATE,
    MODE_YIELD,
    PanelLimitStates,
)
from .panel import PanelSpec

__all__ = ["VariableModel", "ProbabilisticAssessment", "GenericLimitStates",
           "beta_from_pf"]

_EULER_GAMMA = 0.5772156649015329


def beta_from_pf(pf: float) -> float:
    """Standard-normal reliability index of a failure probability."""
    from math import erf, inf, log, sqrt

    if pf <= 0.0:
        return inf
    if pf >= 1.0:
        return -inf
    # invert Phi(-beta) = pf by bisection on beta (no scipy dependency)
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        tail = 0.5 * (1.0 - erf(mid / sqrt(2.0)))
        if tail > pf:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class VariableModel:
    """Marginals of ``X = (p_ice [MPa], f_y [MPa], dt_corr [mm], k_mod [-])``.

    Defaults follow the W3-N04 ice-belt model (ISO 19906 style): annual maximum
    local ice pressure Gumbel with COV 0.28, yield strength lognormal with
    COV 0.08, corrosion margin normal, model uncertainty lognormal with COV
    0.10, and a light load-model correlation of 0.25 (same pressure mechanism
    drives both the ice load and its membrane amplification).
    """

    p_ice_mean: float = 2.5
    p_ice_cov: float = 0.28
    fy_mean: float = 313.0
    fy_cov: float = 0.08
    dt_mean: float = 2.0
    dt_sd: float = 0.5
    k_mean: float = 1.0
    k_cov: float = 0.10
    load_model_correlation: float = 0.25

    # ------------------------------------------------------------------ model
    def variable_set(self) -> VariableSet:
        marginals = (
            gumbel_max_from_moments(self.p_ice_mean, self.p_ice_cov),
            lognormal_from_moments(self.fy_mean, self.fy_cov),
            NormalDistribution(mu=self.dt_mean, sigma=self.dt_sd),
            lognormal_from_moments(self.k_mean, self.k_cov),
        )
        rho = self.load_model_correlation
        correlation = (
            (1.0, 0.0, 0.0, rho),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (rho, 0.0, 0.0, 1.0),
        )
        return VariableSet(marginals=marginals,
                           names=("p_ice", "fy", "dt_corr", "k_mod"),
                           correlation=correlation,
                           units=("MPa", "MPa", "mm", "-"))

    #: variable name -> dataclass field holding its mean
    MEAN_FIELDS = {"p_ice": "p_ice_mean", "fy": "fy_mean",
                   "dt_corr": "dt_mean", "k_mod": "k_mean"}

    def with_mean(self, name: str, value: float) -> "VariableModel":
        """Same model with one marginal *mean* replaced (sensitivity / study)."""
        if name not in self.MEAN_FIELDS:
            raise ValueError(f"unknown variable {name!r}")
        return replace(self, **{self.MEAN_FIELDS[name]: float(value)})

    def means(self) -> Dict[str, float]:
        return {"p_ice": self.p_ice_mean, "fy": self.fy_mean,
                "dt_corr": self.dt_mean, "k_mod": self.k_mean}

    def as_dict(self) -> Dict[str, object]:
        out: Dict[str, object] = {}
        for entry in self.variable_set().describe():
            out[entry["name"]] = {
                "distribution": entry["distribution"],
                "mean": entry["mean"],
                "std": entry["std"],
                "cov": entry["cov"],
                "units": entry["units"],
                "parameters": entry["parameters"],
            }
        out["load_model_correlation"] = self.load_model_correlation
        return out


@dataclass
class GenericLimitStates:
    """Adapter that lets the unified probabilistic layer drive *any* model.

    ``builder(t)`` returns ``{name: g(x)}`` for an as-built thickness ``t``; the
    optional ``batch_builder(t)`` returns the NumPy-vectorised versions (used by
    the Monte-Carlo cross-checks).  This is what makes the W4-N03 layer an
    interface rather than a copy of one model: the W3-N04 ice-belt limit states
    (or any other code-specific formulation) can be assessed with the same
    FORM/SORM/system/sensitivity/design-inverse machinery, which is exactly how
    the cross-validation against the baseline node is done.
    """

    panel: PanelSpec
    builder: Callable[[float], Dict[str, Callable[[Sequence[float]], float]]]
    batch_builder: Optional[Callable[[float], Dict[str, Callable]]] = None

    def limit_states(self) -> Dict[str, Callable[[Sequence[float]], float]]:
        return self.builder(self.panel.t)

    def batch_limit_states(self) -> Dict[str, Callable]:
        if self.batch_builder is None:
            return {}
        return self.batch_builder(self.panel.t)

    def with_thickness(self, t: float) -> "GenericLimitStates":
        return GenericLimitStates(self.panel.with_thickness(t), self.builder,
                                  self.batch_builder)


@dataclass
class ProbabilisticAssessment:
    """FORM/SORM/system/design-inverse analysis of a limit-state set.

    ``limit_states`` is normally a :class:`PanelLimitStates` (deterministic
    capacity inside the limit state) but any object exposing ``panel``,
    ``limit_states()`` and ``batch_limit_states()`` works -- see
    :class:`GenericLimitStates`.
    """

    limit_states: PanelLimitStates
    variables: VariableModel
    seed: int = 2026

    def __post_init__(self) -> None:
        self.variable_set = self.variables.variable_set()
        self.transform = self.variable_set.transform()
        self.batch = self.limit_states.batch_limit_states()
        self.problems: Dict[str, ReliabilityProblem] = {}
        for name, fn in self.limit_states.limit_states().items():
            self.problems[name] = ReliabilityProblem(
                self.transform, fn, name=name,
                variable_names=list(self.variable_set.names),
                batch_limit_state=self.batch.get(name))

    # ---------------------------------------------------------------- helpers
    @property
    def panel(self) -> PanelSpec:
        return self.limit_states.panel

    def problem(self, mode: str = MODE_ULTIMATE) -> ReliabilityProblem:
        if mode not in self.problems:
            raise KeyError(f"unknown limit-state mode {mode!r}")
        return self.problems[mode]

    # ------------------------------------------------------------- components
    def form(self, mode: str = MODE_ULTIMATE, **kwargs) -> FormResult:
        return self.problem(mode).form(**kwargs)

    def sorm(self, mode: str = MODE_ULTIMATE,
             form_result: Optional[FormResult] = None) -> SormResult:
        base = form_result if form_result is not None else self.form(mode)
        return self.problem(mode).sorm(base)

    def monte_carlo(self, mode: str = MODE_ULTIMATE, samples: int = 1_000_000,
                    seed: Optional[int] = None) -> MonteCarloResult:
        return self.problem(mode).monte_carlo_pf(
            samples=samples, seed=self.seed if seed is None else seed)

    # ------------------------------------------------------------------ system
    def system(self, forms: Optional[Dict[str, FormResult]] = None) -> Dict[str, object]:
        """Ditlevsen bi-modal bounds over the series system of limit states."""
        base = forms if forms is not None else {m: self.form(m) for m in self.problems}
        result = ditlevsen_bounds([base[m] for m in base])
        payload = result.to_dict()
        payload["modes"] = list(base)
        return payload

    def system_monte_carlo(self, samples: int = 1_000_000,
                           seed: Optional[int] = None,
                           modes: Sequence[str] = DEFAULT_MODES) -> Dict[str, object]:
        """Crude Monte Carlo of ``P(any mode fails)`` (system cross-check)."""
        from ..structural._batch import batch_u_to_x

        rng = np.random.default_rng(self.seed if seed is None else seed)
        u = rng.standard_normal((samples, len(self.variable_set.names)))
        x = np.asarray(batch_u_to_x(self.transform, u), dtype=float)
        flags = np.zeros(samples, dtype=bool)
        per_mode: Dict[str, float] = {}
        for name in modes:
            fn = self.batch[name]
            g = np.asarray(fn(x), dtype=float)
            failed = g < 0.0
            per_mode[name] = float(failed.mean())
            flags |= failed
        pf = float(flags.mean())
        return {
            "samples": int(samples),
            "pf_system": pf,
            "beta_system": beta_from_pf(pf),
            "pf_component": per_mode,
            "n_failures": int(flags.sum()),
        }

    # ------------------------------------------------------------ sensitivities
    def sensitivity(self, mode: str = MODE_ULTIMATE,
                    relative_step: float = 1e-4,
                    form_result: Optional[FormResult] = None) -> Dict[str, object]:
        """``d beta / d mu_X``: design-point derivative vs finite difference.

        Analytic side: the envelope theorem gives ``d beta/d theta = alpha . du/dtheta``
        from the Nataf transformation derivatives (:mod:`tensorphysics.structural`),
        chained to the mean through the distribution parameterisation.  Reference
        side: re-solve FORM with the marginal mean shifted by ``+- h``.
        """
        base = form_result if form_result is not None else self.form(mode)
        problem = self.problem(mode)
        theta = problem.design_point_sensitivity(base)
        marginals = list(self.transform.marginals)
        # d(parameter)/d(mean) for each marginal (scale held fixed)
        dparam: List[Dict[str, float]] = []
        for marginal in marginals:
            if isinstance(marginal, NormalDistribution):
                dparam.append({"mu": 1.0})
            elif hasattr(marginal, "mu_ln"):                     # lognormal
                mean = marginal.mean()
                dparam.append({"mu_ln": 1.0 / mean, "sigma_ln": 0.0})
            else:                                                # Gumbel (max)
                # mean = loc + gamma * scale and the COV is held fixed, so
                # shifting the mean moves both parameters:
                mean = marginal.mean()
                d_scale = marginal.scale / mean          # d scale / d mean
                d_loc = 1.0 - _EULER_GAMMA * d_scale     # d loc / d mean
                dparam.append({"loc": d_loc, "scale": d_scale})
        analytic: Dict[str, float] = {}
        for i, marginal in enumerate(marginals):
            name = self.variable_set.names[i]
            value = 0.0
            for param, factor in dparam[i].items():
                value += theta.get((name, param), 0.0) * factor
            analytic[name] = float(value)

        finite: Dict[str, float] = {}
        relative: Dict[str, float] = {}
        for i, name in enumerate(self.variable_set.names):
            mean = self.variables.means()[name]
            h = relative_step * max(abs(mean), 1e-6)
            betas = []
            for sign in (1.0, -1.0):
                variables = self.variables.with_mean(name, mean + sign * h)
                trial = ProbabilisticAssessment(self.limit_states, variables,
                                                seed=self.seed)
                betas.append(trial.form(mode).beta)
            finite[name] = (betas[0] - betas[1]) / (2.0 * h)
            denom = max(abs(finite[name]), 1e-12)
            relative[name] = abs(analytic[name] - finite[name]) / denom
        return {
            "mode": mode,
            "dbeta_dmean_analytic": analytic,
            "dbeta_dmean_finite_difference": finite,
            "relative_difference": relative,
            "max_relative_difference": float(max(relative.values())),
            "alpha": list(base.alpha),
            "beta": float(base.beta),
        }

    # ------------------------------------------------------------- MVFOSM ref.
    def mvfosm(self, mode: str = MODE_ULTIMATE, step: float = 1e-6) -> Dict[str, float]:
        """Mean-value first-order index ``beta_MV = (mu_g)/sigma_g`` (closed form).

        The gradient of ``g`` is taken at the mean point in physical space and
        combined with the *actual* marginal standard deviations, i.e. no Nataf
        transformation and no iteration.  Used as the independent reference for
        the target-thickness inversion (classical MVFOSM / code-calibration
        formula, Madsen et al. 1986).
        """
        problem = self.problem(mode)
        means = [self.variables.means()[n] for n in self.variable_set.names]
        g0 = problem.limit_state(means)
        grad = []
        for i, m in enumerate(means):
            h = step * max(abs(m), 1e-6)
            plus = list(means)
            minus = list(means)
            plus[i] += h
            minus[i] -= h
            grad.append((problem.limit_state(plus) - problem.limit_state(minus)) / (2 * h))
        stdev = [marginal.std() for marginal in self.transform.marginals]
        variance = sum((gi * si) ** 2 for gi, si in zip(grad, stdev))
        sigma_g = float(np.sqrt(variance))
        beta = float(g0 / sigma_g) if sigma_g > 0 else float("inf")
        return {"beta_mvfosm": beta, "mu_g": float(g0), "sigma_g": sigma_g,
                "gradient": grad, "stdev": stdev, "g_at_mean": float(g0)}

    # ----------------------------------------------------------- design inverse
    def beta_of_thickness(self, t_nom: float, mode: str = MODE_ULTIMATE) -> float:
        """FORM index at a trial as-built thickness (``-inf`` if not solvable).

        Thicknesses that fall outside the verified geometry envelope, or that
        leave the limit state violated at the median point, return ``-inf`` so
        the design bisection stays well defined.
        """
        try:
            states = self.limit_states.with_thickness(t_nom)
        except (ReliabilityError, ValueError):
            return float("-inf")
        trial = ProbabilisticAssessment(states, self.variables, seed=self.seed)
        try:
            return float(trial.form(mode).beta)
        except (ReliabilityError, ValueError):
            return float("-inf")

    def beta_of_thickness_mvfosm(self, t_nom: float,
                                 mode: str = MODE_ULTIMATE) -> float:
        try:
            states = self.limit_states.with_thickness(t_nom)
        except (ReliabilityError, ValueError):
            return float("-inf")
        trial = ProbabilisticAssessment(states, self.variables, seed=self.seed)
        return float(trial.mvfosm(mode)["beta_mvfosm"])

    def _bracket(self, target_beta: float, mode: str,
                 lower: Optional[float], upper: Optional[float],
                 max_expansions: int = 12) -> Tuple[float, float]:
        """Find a thickness bracket where ``beta - target`` changes sign."""
        lower = 0.5 * self.panel.t if lower is None else float(lower)
        upper = 2.0 * self.panel.t if upper is None else float(upper)
        # -inf means "the limit state is already violated at the median point"
        # (or the panel left the verified envelope): a well-defined low side.
        f_low = self.beta_of_thickness(lower, mode) - target_beta
        for _ in range(max_expansions):
            if f_low <= 0.0:
                break
            lower *= 0.8
            f_low = self.beta_of_thickness(lower, mode) - target_beta
        f_high = self.beta_of_thickness(upper, mode) - target_beta
        for _ in range(max_expansions):
            if f_high >= 0.0:
                break
            upper *= 1.25
            f_high = self.beta_of_thickness(upper, mode) - target_beta
        if f_low * f_high > 0.0:
            raise ReliabilityError(
                "target beta is not bracketed: "
                f"beta({lower * 1e3:.1f} mm)={f_low + target_beta:.4g}, "
                f"beta({upper * 1e3:.1f} mm)={f_high + target_beta:.4g}")
        return lower, upper

    def required_thickness(self, target_beta: float = 4.0, mode: str = MODE_ULTIMATE,
                           lower: Optional[float] = None, upper: Optional[float] = None,
                           tolerance: float = 1e-5, max_iterations: int = 60
                           ) -> Dict[str, object]:
        """Bisection on the as-built thickness such that ``beta_form = target``."""
        a, b = self._bracket(target_beta, mode, lower, upper)
        f_low = self.beta_of_thickness(a, mode) - target_beta
        f_high = self.beta_of_thickness(b, mode) - target_beta
        bracket = [a, b]
        iterations = 0
        for iterations in range(1, max_iterations + 1):
            mid = 0.5 * (a + b)
            value = self.beta_of_thickness(mid, mode) - target_beta
            if abs(value) <= tolerance or (b - a) <= tolerance:
                break
            if value * f_low <= 0.0:
                b = mid
            else:
                a, f_low = mid, value
        t_required = 0.5 * (a + b)
        beta_achieved = self.beta_of_thickness(t_required, mode)
        return {
            "mode": mode,
            "target_beta": float(target_beta),
            "t_required_mm": float(t_required * 1e3),
            "beta_achieved": float(beta_achieved),
            "residual": float(beta_achieved - target_beta),
            "iterations": int(iterations),
            "bracket_mm": [float(bracket[0] * 1e3), float(bracket[1] * 1e3)],
            "beta_bracket": [float(f_low + target_beta), float(f_high + target_beta)],
            "method": "FORM bisection on the as-built thickness",
        }

    def required_thickness_mvfosm(self, target_beta: float = 4.0,
                                  mode: str = MODE_ULTIMATE,
                                  lower: Optional[float] = None,
                                  upper: Optional[float] = None,
                                  tolerance: float = 1e-6,
                                  max_iterations: int = 100) -> Dict[str, object]:
        """Independent closed-form-in-the-loop reference design thickness."""
        lower = 0.5 * self.panel.t if lower is None else float(lower)
        upper = 2.0 * self.panel.t if upper is None else float(upper)
        f_low = self.beta_of_thickness_mvfosm(lower, mode) - target_beta
        for _ in range(12):
            if np.isfinite(f_low) and f_low <= 0.0:
                break
            lower *= 0.8
            f_low = self.beta_of_thickness_mvfosm(lower, mode) - target_beta
        f_high = self.beta_of_thickness_mvfosm(upper, mode) - target_beta
        for _ in range(12):
            if np.isfinite(f_high) and f_high >= 0.0:
                break
            upper *= 1.25
            f_high = self.beta_of_thickness_mvfosm(upper, mode) - target_beta
        if f_low * f_high > 0.0:
            raise ReliabilityError(
                f"MVFOSM target beta not bracketed (beta({lower*1e3:.1f} mm)="
                f"{f_low + target_beta:.4g}, beta({upper*1e3:.1f} mm)="
                f"{f_high + target_beta:.4g})")
        a, b = lower, upper
        for _ in range(max_iterations):
            mid = 0.5 * (a + b)
            value = self.beta_of_thickness_mvfosm(mid, mode) - target_beta
            if abs(value) <= tolerance or (b - a) <= tolerance:
                break
            if value * f_low <= 0.0:
                b = mid
            else:
                a, f_low = mid, value
        t_required = 0.5 * (a + b)
        return {
            "mode": mode,
            "target_beta": float(target_beta),
            "t_required_mm": float(t_required * 1e3),
            "beta_achieved": float(self.beta_of_thickness_mvfosm(t_required, mode)),
            "method": "MVFOSM bisection",
        }

    # ---------------------------------------------------------------- reporting
    def report(self, modes: Sequence[str] = DEFAULT_MODES,
               samples: int = 0, sensitivity: bool = True,
               system_samples: int = 0) -> Dict[str, object]:
        """Bundle the probabilistic block used by the unified assessment.

        ``modes`` of the *system* analysis default to
        ``limit_states.DEFAULT_MODES`` (ultimate + net-section yield); the
        additional diagnostic modes can be requested explicitly.
        """
        forms: Dict[str, FormResult] = {}
        for mode in modes:
            forms[mode] = self.form(mode)
        out: Dict[str, object] = {
            "variables": self.variables.as_dict(),
            "form": {m: f.to_dict() for m, f in forms.items()},
            "sorm": {m: self.sorm(m, forms[m]).to_dict() for m in modes},
            "system_ditlevsen": self.system(forms),
        }
        if samples:
            out["monte_carlo"] = {
                mode: self.monte_carlo(mode, samples).to_dict() for mode in modes
            }
        if system_samples:
            out["system_monte_carlo"] = self.system_monte_carlo(
                system_samples, modes=tuple(modes))
        if sensitivity:
            out["sensitivity"] = self.sensitivity(MODE_ULTIMATE, form_result=forms[MODE_ULTIMATE])
        out["mvfosm"] = self.mvfosm(MODE_ULTIMATE)
        return out