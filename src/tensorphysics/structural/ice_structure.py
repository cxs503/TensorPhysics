"""Ice-loaded polar midship section: ISO 19906 style structural reliability model.

Physical model
--------------
The ice belt of a polar ship is idealised as a stiffened shell panel:

* the shell plate spans between transverse side stiffeners
  (``stiffener_spacing`` = 350 mm) and is treated as a clamped long plate in
  bending, so the peak bending stress under a local ice patch pressure ``p`` is
  ``sigma_b = alpha_b * p * (b / t_net)^2`` (alpha_b = 0.5 for a/b >= 4);
* the same bay carries in-plane compression between deep frames
  (``frame_spacing`` = 2400 mm); the elastic plate buckling stress is
  ``sigma_E = k * pi^2 E / (12 (1 - nu^2)) * (t_net / a)^2`` and the
  elastic-plastic capacity follows the Johnson-Ostenfeld correction
  (this is the panel-level buckling reduction shared with campaign node W3-N03);
* the ice belt accumulates fatigue damage under an equivalent constant-amplitude
  stress range: Miner's rule with an S-N curve of slope ``m = 3`` is used against
  an allowable damage ``D_allow = 0.5`` (design fatigue factor 2), continuing the
  damage-accumulation chain of node m764.

``t_net = t_nom - dt_corr`` where ``dt_corr`` is the corrosion/abrasion margin.
All stresses are in MPa, pressures in MPa and thicknesses in mm.

Random variable vector (4 components)
-------------------------------------
``p_ice``   annual maximum local ice pressure, Gumbel (max) extreme value model;
``fy``      yield strength of the ice-belt steel grade, lognormal;
``dt_corr`` corrosion / ice-abrasion thickness margin, normal;
``k_mod``   model uncertainty factor on predicted stress, lognormal.

``p_ice`` and ``k_mod`` are positively correlated (rho = 0.25): the load-model
bias arises from the same unmodelled contact-area effects that drive the
pressure extreme. The correlation is handled by the Nataf transformation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import pi
from typing import Callable, Sequence

from .distributions import (
    Distribution,
    NatafTransform,
    NormalDistribution,
    gumbel_max_from_moments,
    lognormal_from_moments,
)
from .reliability import (
    FormResult,
    ReliabilityError,
    ReliabilityProblem,
    SystemReliabilityResult,
    ditlevsen_bounds,
)

__all__ = [
    "PolarPanelGeometry",
    "PolarMidshipReliability",
    "VariableSet",
    "default_variables",
    "YIELD",
    "BUCKLING",
    "FATIGUE",
]

YIELD = "yield"
BUCKLING = "buckling"
FATIGUE = "fatigue"


# --------------------------------------------------------------------------- #
# Deterministic panel model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PolarPanelGeometry:
    """Geometry, material and code parameters of the ice-belt panel."""

    t_nom: float = 34.0                 # mm, nominal (as-built) plate thickness
    stiffener_spacing: float = 350.0    # mm, b, plate span in bending
    frame_spacing: float = 2400.0       # mm, a, bay dimension for buckling
    alpha_bending: float = 0.5          # clamped long-plate bending coefficient
    k_buckling: float = 4.0             # buckling coefficient, simply supported plate
    E: float = 206000.0                 # MPa, steel Young's modulus
    nu: float = 0.3                     # Poisson ratio
    sigma_global: float = 95.0          # MPa, global hull-girder compression in ice belt
    psi_membrane: float = 1.5           # ice patch -> in-plane membrane stress transfer
    sn_log_a: float = 11.764            # log10 A of S-N curve (N = A * dsigma^-m)
    sn_slope: float = 3.0               # S-N slope m (seawater, welded joint)
    cycles: float = 5.0e5               # equivalent constant-amplitude cycles, design life
    pressure_amplitude: float = 1.0     # MPa, equivalent constant stress-range pressure
    damage_allow: float = 0.5           # allowable Miner damage (DF = 2)
    backend: str = "auto"

    # -- derived quantities --------------------------------------------------
    def net_thickness(self, corrosion: float) -> float:
        t = self.t_nom - corrosion
        if t <= 1e-6:
            raise ReliabilityError("net thickness became non-positive")
        return t

    def bending_stress(self, pressure: float, corrosion: float, k_mod: float) -> float:
        t = self.net_thickness(corrosion)
        return k_mod * self.alpha_bending * pressure * (self.stiffener_spacing / t) ** 2

    def elastic_buckling_stress(self, corrosion: float) -> float:
        t = self.net_thickness(corrosion)
        coefficient = self.k_buckling * pi**2 * self.E / (12.0 * (1.0 - self.nu**2))
        return coefficient * (t / self.frame_spacing) ** 2

    def applied_membrane_stress(self, pressure: float, k_mod: float) -> float:
        return k_mod * (self.sigma_global + self.psi_membrane * pressure)

    def buckling_stress(self, pressure: float, corrosion: float, k_mod: float, fy: float) -> float:
        """Johnson-Ostenfeld elastic-plastic plate capacity."""
        sigma_e = self.elastic_buckling_stress(corrosion)
        if sigma_e <= 0.5 * fy:
            return sigma_e
        return fy * (1.0 - fy / (4.0 * sigma_e))

    def sn_capacity(self) -> float:
        return 10.0**self.sn_log_a

    def fatigue_damage(self, corrosion: float, k_mod: float) -> float:
        t = self.net_thickness(corrosion)
        dsigma = k_mod * self.alpha_bending * self.pressure_amplitude * (
            self.stiffener_spacing / t
        ) ** 2
        return self.cycles * dsigma**self.sn_slope / self.sn_capacity()

    # -- limit states --------------------------------------------------------
    def g_yield(self, x: Sequence[float]) -> float:
        pressure, fy, corrosion, k_mod = x
        return fy - self.bending_stress(pressure, corrosion, k_mod)

    def g_buckling(self, x: Sequence[float]) -> float:
        pressure, fy, corrosion, k_mod = x
        capacity = self.buckling_stress(pressure, corrosion, k_mod, fy)
        demand = self.applied_membrane_stress(pressure, k_mod)
        return capacity - demand

    def g_fatigue(self, x: Sequence[float]) -> float:
        pressure, fy, corrosion, k_mod = x
        del pressure, fy
        return self.damage_allow - self.fatigue_damage(corrosion, k_mod)

    @property
    def limit_states(self) -> dict[str, Callable[[Sequence[float]], float]]:
        return {
            YIELD: self.g_yield,
            BUCKLING: self.g_buckling,
            FATIGUE: self.g_fatigue,
        }

    def batch_limit_states(self) -> dict[str, Callable[[object], object]] | None:
        """Optional NumPy-vectorised limit states for large simulation runs.

        Returns ``None`` when NumPy is unavailable (the scalar limit states then
        remain the only evaluator). Validated against the scalar versions in
        ``tests/test_structural_ice_reliability.py``.
        """
        try:  # pragma: no cover - environment dependent
            import numpy as np
        except Exception:  # pragma: no cover
            return None

        def split(x):
            array = np.asarray(x, dtype=float)
            return array[:, 0], array[:, 1], array[:, 2], array[:, 3]

        def g_yield(x):
            pressure, fy, corrosion, k_mod = split(x)
            t = self.t_nom - corrosion
            return fy - k_mod * self.alpha_bending * pressure * (self.stiffener_spacing / t) ** 2

        def g_buckling(x):
            pressure, fy, corrosion, k_mod = split(x)
            t = self.t_nom - corrosion
            coefficient = self.k_buckling * pi**2 * self.E / (12.0 * (1.0 - self.nu**2))
            sigma_e = coefficient * (t / self.frame_spacing) ** 2
            capacity = np.where(
                sigma_e <= 0.5 * fy, sigma_e, fy * (1.0 - fy / (4.0 * sigma_e))
            )
            demand = k_mod * (self.sigma_global + self.psi_membrane * pressure)
            return capacity - demand

        def g_fatigue(x):
            _, _, corrosion, k_mod = split(x)
            t = self.t_nom - corrosion
            dsigma = k_mod * self.alpha_bending * self.pressure_amplitude * (
                self.stiffener_spacing / t
            ) ** 2
            return self.damage_allow - self.cycles * dsigma**self.sn_slope / self.sn_capacity()

        return {YIELD: g_yield, BUCKLING: g_buckling, FATIGUE: g_fatigue}

    def with_thickness(self, t_nom: float) -> "PolarPanelGeometry":
        return replace(self, t_nom=t_nom)


# --------------------------------------------------------------------------- #
# Random variable model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VariableSet:
    """Marginal distributions, names and correlation structure."""

    marginals: tuple[Distribution, ...]
    names: tuple[str, ...]
    correlation: tuple[tuple[float, ...], ...]
    units: tuple[str, ...] = ()

    def transform(self) -> NatafTransform:
        return NatafTransform(list(self.marginals), correlation=[list(r) for r in self.correlation])

    def describe(self) -> list[dict]:
        out = []
        for name, marginal, unit in zip(self.names, self.marginals, self.units):
            entry = {
                "name": name,
                "distribution": marginal.name,
                "mean": marginal.mean(),
                "std": marginal.std(),
                "cov": marginal.std() / marginal.mean() if marginal.mean() else float("nan"),
                "parameters": {
                    key: getattr(marginal, key) for key in marginal.param_names
                },
                "units": unit,
            }
            out.append(entry)
        return out


def default_variables(
    pressure_mean: float = 2.5,
    pressure_cov: float = 0.28,
    fy_mean: float = 355.0,
    fy_cov: float = 0.08,
    corrosion_mean: float = 2.0,
    corrosion_sd: float = 0.5,
    model_mean: float = 1.0,
    model_cov: float = 0.10,
    load_model_correlation: float = 0.25,
) -> VariableSet:
    """ISO 19906 style random variable vector for the ice-belt panel.

    ``pressure_mean`` [MPa] is the annual maximum local ice pressure; the Gumbel
    parameters are obtained from the first two moments. ``fy_mean`` [MPa] is the
    nominal yield strength of the grade. ``corrosion_mean`` [mm] is the total
    corrosion/abrasion margin deducted from the as-built thickness.
    """
    marginals = (
        gumbel_max_from_moments(pressure_mean, pressure_cov),
        lognormal_from_moments(fy_mean, fy_cov),
        NormalDistribution(mu=corrosion_mean, sigma=corrosion_sd),
        lognormal_from_moments(model_mean, model_cov),
    )
    correlation = (
        (1.0, 0.0, 0.0, load_model_correlation),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (load_model_correlation, 0.0, 0.0, 1.0),
    )
    return VariableSet(
        marginals=marginals,
        names=("p_ice", "fy", "dt_corr", "k_mod"),
        correlation=correlation,
        units=("MPa", "MPa", "mm", "-"),
    )


# --------------------------------------------------------------------------- #
# Reliability driver
# --------------------------------------------------------------------------- #
class PolarMidshipReliability:
    """FORM / SORM / simulation / system analysis of one polar midship panel."""

    def __init__(
        self,
        geometry: PolarPanelGeometry | None = None,
        variables: VariableSet | None = None,
    ) -> None:
        self.geometry = geometry or PolarPanelGeometry()
        self.variables = variables or default_variables()
        self.transform = self.variables.transform()
        self.problems = {
            name: self._problem(self.geometry, name)
            for name in self.geometry.limit_states
        }

    def _problem(self, geometry: PolarPanelGeometry, name: str) -> ReliabilityProblem:
        batch = geometry.batch_limit_states()
        return ReliabilityProblem(
            self.transform,
            geometry.limit_states[name],
            name=name,
            variable_names=self.variables.names,
            batch_limit_state=None if batch is None else batch[name],
        )

    # -- component analyses --------------------------------------------------
    def form_all(self) -> dict[str, FormResult]:
        return {name: problem.form() for name, problem in self.problems.items()}

    def sorm_all(self, forms: dict[str, FormResult] | None = None) -> dict:
        base = forms if forms is not None else self.form_all()
        return {
            name: self.problems[name].sorm(base[name]) for name in self.problems
        }

    def monte_carlo_all(
        self, samples: int = 1_000_000, seed: int = 2026
    ) -> dict:
        return {
            name: self.problems[name].monte_carlo_pf(samples=samples, seed=seed)
            for name in self.problems
        }

    def system_bounds(self, forms: dict[str, FormResult] | None = None) -> SystemReliabilityResult:
        base = forms if forms is not None else self.form_all()
        return ditlevsen_bounds([base[name] for name in (YIELD, BUCKLING, FATIGUE)])

    # -- deterministic stress checks ----------------------------------------
    def nominal_stresses(self) -> dict[str, float]:
        pressure = self.variables.marginals[0].mean()
        fy = self.variables.marginals[1].mean()
        corrosion = self.variables.marginals[2].mean()
        k_mod = self.variables.marginals[3].mean()
        geometry = self.geometry
        return {
            "t_net": geometry.net_thickness(corrosion),
            "bending_stress": geometry.bending_stress(pressure, corrosion, k_mod),
            "applied_membrane_stress": geometry.applied_membrane_stress(pressure, k_mod),
            "elastic_buckling_stress": geometry.elastic_buckling_stress(corrosion),
            "buckling_capacity": geometry.buckling_stress(pressure, corrosion, k_mod, fy),
            "fatigue_stress_range": geometry.fatigue_damage(corrosion, k_mod)
            * geometry.sn_capacity()
            / geometry.cycles,
            "fatigue_damage": geometry.fatigue_damage(corrosion, k_mod),
            "damage_allow": geometry.damage_allow,
            "yield_strength": fy,
        }

    # -- design point sensitivities -----------------------------------------
    def sensitivities(self, limit_state: str = YIELD) -> dict[str, dict]:
        problem = self.problems[limit_state]
        form = problem.form()
        analytic = problem.design_point_sensitivity(form)
        finite = problem.sensitivity_finite_difference(form)
        table = {}
        for key in analytic:
            a = analytic[key]
            f = finite[key]
            if abs(f) > 1e-12:
                rel = abs(a - f) / abs(f)
            else:
                rel = 0.0 if abs(a) < 1e-8 else float("inf")
            table[f"{key[0]}.{key[1]}"] = {
                "analytic": a,
                "finite_difference": f,
                "relative_error": rel,
            }
        return table

    def required_thickness_for(
        self,
        limit_state: str | None = None,
        target_beta: float = 4.0,
        lower: float = 15.0,
        upper: float = 90.0,
        tolerance: float = 5e-3,
    ) -> dict:
        """As-built plate thickness needed for a target reliability index.

        ``limit_state=None`` targets the Ditlevsen series-system index, otherwise
        the named component limit state. The returned dict is JSON-ready and
        records the achieved index and the bracketing values.
        """
        if limit_state is None:
            measure = self.system_beta_of_thickness
            label = "system(ditlevsen)"
        else:
            if limit_state not in self.geometry.limit_states:
                raise ReliabilityError(f"unknown limit state {limit_state!r}")
            measure = lambda t: self.beta_of_thickness(t, limit_state)  # noqa: E731
            label = limit_state
        outcome = required_thickness(
            measure, target_beta=target_beta, lower=lower, upper=upper, tolerance=tolerance
        )
        outcome["measure"] = label
        return outcome

    def stress_sensitivity(self, limit_state: str = YIELD) -> float:
        """d beta / d (mean ice pressure), analytic route."""
        table = self.sensitivities(limit_state)
        return table["p_ice.loc"]["analytic"]

    # -- thickness design ----------------------------------------------------
    def beta_of_thickness(self, t_nom: float, limit_state: str) -> float:
        """Reliability index of one limit state at a trial thickness.

        Returns ``-inf`` when the limit state is already violated at the median
        point (plate too thin for the model to define a design point), which
        keeps thickness bisection well defined.
        """
        geometry = self.geometry.with_thickness(t_nom)
        problem = self._problem(geometry, limit_state)
        try:
            return problem.form().beta
        except ReliabilityError:
            return float("-inf")

    def system_beta_of_thickness(self, t_nom: float) -> float:
        """Series-system reliability index at a trial thickness (Ditlevsen mean bound)."""
        geometry = self.geometry.with_thickness(t_nom)
        forms = {}
        for name in geometry.limit_states:
            problem = self._problem(geometry, name)
            try:
                forms[name] = problem.form()
            except ReliabilityError:
                return float("-inf")
        return self.system_bounds(forms).beta_point


def required_thickness(
    beta_of_thickness: Callable[[float], float],
    target_beta: float = 4.0,
    lower: float = 15.0,
    upper: float = 90.0,
    tolerance: float = 5e-3,
    max_iterations: int = 60,
) -> dict:
    """Bisection on the as-built plate thickness until ``beta(t) = target_beta``.

    Returns the thickness, the achieved index and the residual, plus the
    bounding points so the bracketing premise can be audited.
    """
    f_low = beta_of_thickness(lower) - target_beta
    f_high = beta_of_thickness(upper) - target_beta
    if f_low * f_high > 0.0:
        raise ReliabilityError(
            f"target beta = {target_beta} is not bracketed on t in [{lower}, {upper}] "
            f"(beta(lower)-target = {f_low:.6g}, beta(upper)-target = {f_high:.6g})"
        )
    a, b = lower, upper
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        mid = 0.5 * (a + b)
        value = beta_of_thickness(mid) - target_beta
        if abs(value) <= tolerance or (b - a) <= tolerance:
            break
        if value * f_low <= 0.0:
            b = mid
        else:
            a, f_low = mid, value
    t_required = 0.5 * (a + b)
    return {
        "target_beta": target_beta,
        "t_required": t_required,
        "beta_achieved": beta_of_thickness(t_required),
        "residual": beta_of_thickness(t_required) - target_beta,
        "iterations": iterations,
        "bracket": [lower, upper],
        "beta_lower": beta_of_thickness(lower),
        "beta_upper": beta_of_thickness(upper),
        "tolerance": tolerance,
    }