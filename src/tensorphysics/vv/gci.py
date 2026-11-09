"""Richardson extrapolation, observed order, Roache GCI and the convergence gate.

This module is the solver-neutral half of the V&V machine gate: it consumes a
study of one observable on at least three systematically refined grids and
decides, without human judgement, whether the study is in the asymptotic range
and what numerical uncertainty may be attached to the finest result.

Conventions
-----------
``levels`` are ordered **coarse to fine**; ``h`` is a positive grid length
(cell size, panel size, time step) that must strictly decrease from level to
level.  The refinement ratio of a pair is ``rho = h_coarse / h_fine > 1``.

For a uniform study with ratio ``rho`` the observed order of accuracy is

    p = ln( (f_coarse - f_medium) / (f_medium - f_fine) ) / ln(rho)

which is the three-grid form quoted in ASME V&V 20 / Roache.  For non-uniform
or longer studies the order is obtained from a one-dimensional least-squares
fit of ``f(h) = f_exact + C h**p`` over all levels (the closed form is then a
special case).  The Richardson estimate of the continuum value is

    f_exact = f_fine + (f_fine - f_medium) / (rho**p - 1)

and the grid convergence index of the finest pair (Roache 1994, factor of
safety ``Fs`` = 1.25 for a three-grid study and 3.0 when only two grids are
available) is

    GCI_fine = Fs * |(f_fine - f_coarse) / f_fine| / (rho**p - 1).

The gate is deliberately conservative: a study that is not in the asymptotic
range (oscillatory, diverging, stagnant, or with an order outside
``(p_min, p_max]``) is reported as FAIL, never as an unqualified PASS.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Sequence

__all__ = [
    "GatePolicy",
    "GateVerdict",
    "GridConvergenceStudy",
    "Observation",
    "PASS",
    "WARN",
    "FAIL",
    "WITHHELD",
    "UNRESOLVED",
    "evaluate",
    "gci",
    "grid_convergence_study",
    "observed_order",
    "refinement_ratio",
    "richardson_extrapolated",
    "sweep_refinement_ratio",
    "main",
]

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
#: Evidence statements of the resolvability law (``R = delta / dx``), see
#: :mod:`tensorphysics.vv.resolvability`.  They are *not* solver failures: they
#: say that this discretisation cannot decide the question, so the number must
#: not be published as converged (``WITHHELD``) or is converging to the wrong
#: physics because the feature is subgrid (``UNRESOLVED``).
WITHHELD = "WITHHELD"
UNRESOLVED = "UNRESOLVED"
_STATUS_ORDER = {PASS: 0, WARN: 1, WITHHELD: 2, UNRESOLVED: 3, FAIL: 4}


# ---------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Observation:
    """One grid level of a single observable.

    ``sigma`` is an optional 1-sigma uncertainty of ``f`` (round-off, sampling
    noise, measurement scatter).  When supplied the gate can state whether the
    discretisation differences dominate the noise floor at all.
    """

    h: float
    f: float
    label: str = ""
    sigma: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.h, (int, float)) or not math.isfinite(self.h) or self.h <= 0.0:
            raise ValueError(f"h must be finite and positive, got {self.h!r}")
        if not isinstance(self.f, (int, float)) or not math.isfinite(self.f):
            raise ValueError(f"f must be finite, got {self.f!r}")
        if self.sigma is not None:
            if not math.isfinite(self.sigma) or self.sigma < 0.0:
                raise ValueError(f"sigma must be finite and non-negative, got {self.sigma!r}")

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"h": self.h, "f": self.f}
        if self.label:
            out["label"] = self.label
        if self.sigma is not None:
            out["sigma"] = self.sigma
        return out

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "Observation":
        diff = {k: payload.get(k) for k in ("h", "f", "label", "sigma") if payload.get(k) is not None}
        if "h" not in diff:
            for alias in ("mesh_size", "dx", "dt", "spacing", "cell_size"):
                if payload.get(alias) is not None:
                    diff["h"] = payload[alias]
                    break
        if "f" not in diff:
            for alias in ("value", "metric", "observable", "error", "l2_error"):
                if payload.get(alias) is not None:
                    diff["f"] = payload[alias]
                    break
        if "h" not in diff or "f" not in diff:
            raise ValueError(f"observation needs h and f, got keys {sorted(payload)}")
        return cls(**diff)


@dataclass(frozen=True)
class GatePolicy:
    """Thresholds of the machine gate.  Defaults are the documented ASME/ITTC set."""

    safety_factor_three: float = 1.25
    safety_factor_two: float = 3.0
    p_min: float = 0.05
    p_max: float = 6.0
    assumed_order: float | None = None
    expected_order: float | None = None
    order_tolerance: float = 0.2
    residual_warn: float = 2.0e-3
    residual_fail: float = 2.0e-2
    order_consistency_warn: float = 0.2
    order_consistency_fail: float = 0.5
    noise_warn_ratio: float = 0.05
    noise_fail_ratio: float = 0.25
    require_monotone: bool = True
    strict: bool = False
    pass_requires_noise_certificate: bool = True
    gci_absolute: bool = False
    gci_increase_tolerance: float = 0.05

    def merged(self, **overrides: Any) -> "GatePolicy":
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass(frozen=True)
class GridConvergenceStudy:
    """Numerical (grid/timestep) convergence study of one observable."""

    levels: tuple[Observation, ...]
    order_p: float | None
    order_source: str
    f_extrapolated: float | None
    gci_fine: float | None
    safety_factor: float
    gci_is_relative: bool
    convergence_ratio: float | None
    convergence_type: str
    refinement_ratios: tuple[float, ...]
    uniform_refinement: bool
    fit_residual_rel: float | None
    order_triplets: tuple[float, ...] = ()
    order_spread: float | None = None
    noise_ratio: float | None = None
    noise_certified: bool = False
    gci_pairs: tuple[float, ...] = ()
    gci_monotone: bool | None = None
    extrapolation_gain: float | None = None
    notes: tuple[str, ...] = ()

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    @property
    def f_fine(self) -> float:
        return self.levels[-1].f

    @property
    def asymptotic(self) -> bool:
        return (
            self.convergence_type == "monotone"
            and self.order_p is not None
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_levels": self.n_levels,
            "levels": [lvl.as_dict() for lvl in self.levels],
            "order_p": self.order_p,
            "order_source": self.order_source,
            "f_extrapolated": self.f_extrapolated,
            "f_fine": self.f_fine,
            "gci_fine": self.gci_fine,
            "gci_is_relative": self.gci_is_relative,
            "safety_factor": self.safety_factor,
            "convergence_ratio": self.convergence_ratio,
            "convergence_type": self.convergence_type,
            "refinement_ratios": list(self.refinement_ratios),
            "uniform_refinement": self.uniform_refinement,
            "fit_residual_rel": self.fit_residual_rel,
            "order_triplets": list(self.order_triplets),
            "order_spread": self.order_spread,
            "noise_ratio": self.noise_ratio,
            "noise_certified": self.noise_certified,
            "gci_pairs": list(self.gci_pairs),
            "gci_monotone": self.gci_monotone,
            "extrapolation_gain": self.extrapolation_gain,
            "asymptotic": self.asymptotic,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class GateVerdict:
    """Machine verdict of the convergence / validation gate."""

    status: str
    quantity: str
    study: GridConvergenceStudy | None
    checks: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    validation: Any | None = None
    policy: GatePolicy = field(default_factory=GatePolicy)
    resolvability: Any | None = None
    uncertainty: Any | None = None

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(c["detail"] for c in self.checks if c["status"] != PASS)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "tensorphysics.vv.gate/1",
            "status": self.status,
            "quantity": self.quantity,
            "checks": list(self.checks),
            "reasons": list(self.reasons),
            "study": None if self.study is None else self.study.as_dict(),
            "validation": None if self.validation is None else self.validation.as_dict(),
            "policy": self.policy.as_dict(),
            "resolvability": None if self.resolvability is None else self.resolvability.as_dict(),
            "uncertainty": None if self.uncertainty is None else self.uncertainty.as_dict(),
        }


# ---------------------------------------------------------------------------
# numerical helpers
# ---------------------------------------------------------------------------


def _sorted_levels(levels: Iterable[Observation]) -> tuple[Observation, ...]:
    ordered = tuple(sorted(levels, key=lambda lvl: -lvl.h))
    if len(ordered) < 2:
        raise ValueError("a convergence study needs at least two grid levels")
    for coarse, fine in zip(ordered, ordered[1:]):
        if not coarse.h > fine.h:
            raise ValueError("grid levels must have distinct h values")
    return ordered


def refinement_ratio(coarse: Observation, fine: Observation) -> float:
    """``rho = h_coarse / h_fine > 1`` for one consecutive pair."""
    return coarse.h / fine.h


def _power_law_fit(h: Sequence[float], f: Sequence[float], p: float) -> tuple[float, float, float]:
    """Least-squares fit ``f = a + b * h**p``; returns ``(a, b, ssr)``."""
    hp = [hi**p for hi in h]
    n = float(len(h))
    shp = math.fsum(hp)
    shp2 = math.fsum(v * v for v in hp)
    sf = math.fsum(f)
    shpf = math.fsum(v * fi for v, fi in zip(hp, f))
    den = n * shp2 - shp * shp
    if den <= 0.0 or not math.isfinite(den):
        return float("nan"), float("nan"), float("inf")
    b = (n * shpf - shp * sf) / den
    a = (sf - b * shp) / n
    ssr = math.fsum((fi - a - b * hpi) ** 2 for fi, hpi in zip(f, hp))
    return a, b, ssr


def _fit_order(h: Sequence[float], f: Sequence[float], p_min: float, p_max: float) -> tuple[float, float, float]:
    """Minimise the power-law residual over ``p in [p_min, p_max]``.

    A log-spaced scan brackets the global minimum, a golden-section search then
    refines the bracket.  Returns ``(p, a_intercept, ssr)``.
    """
    lo = max(p_min, 1e-3)
    hi = max(p_max, lo * 1.05)
    n_scan = 160
    grid = [lo * (hi / lo) ** (k / (n_scan - 1)) for k in range(n_scan)]
    values = [_power_law_fit(h, f, p) for p in grid]
    ssr = [v[2] for v in values]
    best = min(range(n_scan), key=lambda k: ssr[k])
    a_lo = grid[max(best - 1, 0)]
    a_hi = grid[min(best + 1, n_scan - 1)]
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = a_hi - phi * (a_hi - a_lo)
    x2 = a_lo + phi * (a_hi - a_lo)
    f1 = _power_law_fit(h, f, x1)[2]
    f2 = _power_law_fit(h, f, x2)[2]
    for _ in range(80):
        if f1 < f2:
            a_hi, x2, f2 = x2, x1, f1
            x1 = a_hi - phi * (a_hi - a_lo)
            f1 = _power_law_fit(h, f, x1)[2]
        else:
            a_lo, x1, f1 = x1, x2, f2
            x2 = a_lo + phi * (a_hi - a_lo)
            f2 = _power_law_fit(h, f, x2)[2]
    p_best = 0.5 * (a_lo + a_hi)
    a_best, _, ssr_best = _power_law_fit(h, f, p_best)
    if ssr[best] < ssr_best:
        return grid[best], values[best][0], values[best][2]
    return p_best, a_best, ssr_best


def observed_order(
    levels: Sequence[Observation],
    *,
    p_min: float = 0.05,
    p_max: float = 6.0,
) -> tuple[float | None, str, float | None]:
    """Observed order of accuracy of a study.

    Returns ``(p, source, residual_rel)`` where ``source`` is
    ``"closed-form"`` for a uniform three-grid study, ``"least-squares"``
    otherwise, and ``None`` when no order can be extracted (stagnant data or
    differences that do not share a sign).
    """
    ordered = _sorted_levels(levels)
    f = [lvl.f for lvl in ordered]
    diffs = [a - b for a, b in zip(f, f[1:])]
    if any(abs(d) == 0.0 for d in diffs):
        return None, "undefined", None
    ratios = [refinement_ratio(c, fn) for c, fn in zip(ordered, ordered[1:])]
    if len(ordered) == 3 and abs(ratios[0] - ratios[1]) <= 1e-12 * max(ratios):
        eps_coarse, eps_fine = diffs[0], diffs[1]
        if eps_coarse * eps_fine <= 0.0:
            return None, "undefined", None
        p = math.log(eps_coarse / eps_fine) / math.log(ratios[0])
        return p, "closed-form", 0.0
    h = [lvl.h for lvl in ordered]
    p, _a, ssr = _fit_order(h, f, p_min, p_max)
    scale = math.fsum(v * v for v in f)
    residual_rel = math.sqrt(ssr / scale) if scale > 0.0 and math.isfinite(ssr) else None
    return p, "least-squares", residual_rel


def richardson_extrapolated(levels: Sequence[Observation], p: float) -> float:
    """Continuum estimate from the two finest levels and the observed order."""
    ordered = _sorted_levels(levels)
    fine, medium = ordered[-1], ordered[-2]
    rho = refinement_ratio(medium, fine)
    k = rho**p - 1.0
    if k == 0.0:
        raise ValueError("degenerate Richardson factor: rho**p == 1")
    return fine.f + (fine.f - medium.f) / k


def gci(
    coarse: Observation,
    fine: Observation,
    p: float,
    safety_factor: float = 1.25,
    force_absolute: bool = False,
) -> tuple[float, bool]:
    """Roache grid convergence index of one pair.

    Returns ``(value, is_relative)``.  The relative form is used whenever
    ``|f_fine|`` is not negligible, otherwise the absolute form is returned so
    that a vanishing observable (e.g. an error norm) stays well defined.
    ``force_absolute=True`` always returns the absolute band; that is the
    correct form for an error norm, because a relative band around a vanishing
    quantity is unbounded by construction.
    """
    rho = refinement_ratio(coarse, fine)
    denom = rho**p - 1.0
    if denom <= 0.0:
        raise ValueError("degenerate GCI denominator: rho**p <= 1 (order must be positive)")
    delta = abs(fine.f - coarse.f)
    if force_absolute:
        return safety_factor * delta / denom, False
    scale = abs(fine.f) if abs(fine.f) > 1.0e-300 else abs(coarse.f)
    if scale > 1.0e-300:
        return safety_factor * delta / scale / denom, True
    return safety_factor * delta / denom, False


def sweep_refinement_ratio(
    model: Callable[[float], float],
    *,
    h_coarse: float,
    p: float,
    ratios: Sequence[float] = (2.0, 3.0, 4.0),
    safety_factor: float = 3.0,
) -> tuple[tuple[float, float], ...]:
    """Two-grid GCI of a fixed coarse grid while the fine grid is refined.

    With ``h_coarse`` fixed and ``rho`` growing, the fine grid becomes finer and
    the GCI attached to the fine result must decrease monotonically for a model
    that is genuinely ``O(h**p)``.  This is the machine-checkable form of "the
    GCI decreases with refinement"; note that for a *fixed fine grid* the
    relative GCI ``Fs*|df|/f/(rho**p-1)`` is invariant in ``rho`` (both the
    difference and the denominator scale as ``rho**p - 1``), so the sweep has to
    be anchored on the coarse grid to be informative.
    """
    out = []
    for rho in ratios:
        coarse = Observation(h=h_coarse, f=model(h_coarse))
        fine = Observation(h=h_coarse / rho, f=model(h_coarse / rho))
        value, _rel = gci(coarse, fine, p, safety_factor=safety_factor)
        out.append((rho, value))
    return tuple(out)


# ---------------------------------------------------------------------------
# study construction
# ---------------------------------------------------------------------------


def grid_convergence_study(
    levels: Sequence[Observation],
    *,
    policy: GatePolicy | None = None,
    assumed_order: float | None = None,
) -> GridConvergenceStudy:
    """Assemble the full numerical study (order, extrapolation, GCI, asymptotics)."""
    policy = policy or GatePolicy()
    ordered = _sorted_levels(levels)
    notes: list[str] = []
    f = [lvl.f for lvl in ordered]
    diffs = [a - b for a, b in zip(f, f[1:])]
    ratios = tuple(refinement_ratio(c, fn) for c, fn in zip(ordered, ordered[1:]))
    uniform = all(abs(r - ratios[0]) <= 1e-9 * ratios[0] for r in ratios)

    order = assumed_order if assumed_order is not None else policy.assumed_order
    order_source = "assumed"
    residual: float | None = None
    if len(ordered) >= 3:
        p, source, residual = observed_order(ordered, p_min=policy.p_min, p_max=policy.p_max)
        if p is not None:
            order, order_source = p, source
        else:
            order, order_source = None, "undefined"
    elif order is None:
        notes.append(
            "two-grid study: the order of accuracy is not observable, an assumed "
            "order is required and the factor of safety is raised to 3.0"
        )

    if order is not None and not (policy.p_min < order <= policy.p_max) and order_source != "assumed":
        notes.append(
            f"observed order {order:.4f} outside the admissible band "
            f"({policy.p_min}, {policy.p_max}]"
        )

    # convergence ratio on the two finest levels
    convergence_ratio: float | None = None
    convergence_type = "undefined"
    if len(ordered) >= 3:
        eps_coarse, eps_fine = diffs[-2], diffs[-1]
        if eps_coarse == 0.0:
            convergence_type = "stagnant"
        else:
            convergence_ratio = eps_fine / eps_coarse
            if convergence_ratio < 0.0:
                convergence_type = "oscillatory"
            elif abs(convergence_ratio - 1.0) <= 1e-9:
                convergence_type = "stagnant"
            elif convergence_ratio > 1.0:
                convergence_type = "diverging"
            else:
                convergence_type = "monotone"
    elif len(ordered) == 2:
        convergence_type = "not-applicable"

    f_exact = None
    if order is not None and len(ordered) >= 2:
        try:
            f_exact = richardson_extrapolated(ordered, order)
        except ValueError:
            f_exact = None
    if f_exact is not None and len(ordered) >= 3:
        f_exact_lsq = _power_law_fit([l.h for l in ordered], f, order)[0]
        if math.isfinite(f_exact_lsq) and abs(f_exact_lsq - f_exact) > 1e-9 * max(1.0, abs(f_exact)):
            notes.append(
                f"finest-pair Richardson value {f_exact:.12g} differs from the "
                f"all-level least-squares intercept {f_exact_lsq:.12g} (non-uniform study)"
            )

    safety_factor = policy.safety_factor_two if len(ordered) == 2 else policy.safety_factor_three
    gci_pairs: list[float] = []
    gci_fine: float | None = None
    gci_is_relative = True
    if order is not None and order > 0.0:
        for coarse, fine in zip(ordered, ordered[1:]):
            try:
                value, is_rel = gci(coarse, fine, order, safety_factor=safety_factor,
                                    force_absolute=policy.gci_absolute)
            except ValueError:
                continue
            gci_pairs.append(value)
            gci_fine, gci_is_relative = value, is_rel

    gci_monotone: bool | None = None
    if len(gci_pairs) >= 2:
        tol = policy.gci_increase_tolerance
        gci_monotone = all(
            nxt <= cur * (1.0 + tol) + 1e-300 for cur, nxt in zip(gci_pairs, gci_pairs[1:])
        )

    # per-triplet orders -> consistency of the observed order
    triplet_orders: list[float] = []
    if len(ordered) >= 4:
        for i in range(len(ordered) - 2):
            p_i, src_i, _ = observed_order(ordered[i : i + 3], p_min=policy.p_min, p_max=policy.p_max)
            if p_i is not None and src_i != "undefined":
                triplet_orders.append(p_i)
    spread = (max(triplet_orders) - min(triplet_orders)) if len(triplet_orders) >= 2 else None

    # noise floor
    noise_ratio: float | None = None
    sigmas = [lvl.sigma for lvl in ordered if lvl.sigma]
    if sigmas and len(diffs) >= 1 and diffs[-1] != 0.0:
        sigma_typ = math.sqrt(math.fsum(s * s for s in sigmas) / len(sigmas))
        noise_ratio = sigma_typ / abs(diffs[-1])
    noise_certified = len(ordered) >= 4 or noise_ratio is not None
    if not noise_certified:
        notes.append(
            "three-grid certificate: a power-law fit through three levels has no "
            "residual, so an undeclared noise floor cannot be detected from the "
            "study itself; declare per-level sigma or supply >=4 grids"
        )

    extrapolation_gain = None
    if f_exact is not None and diffs[-1] != 0.0:
        extrapolation_gain = abs(f_exact - ordered[-1].f) / abs(diffs[-1])

    return GridConvergenceStudy(
        levels=ordered,
        order_p=order,
        order_source=order_source,
        f_extrapolated=f_exact,
        gci_fine=gci_fine,
        safety_factor=safety_factor,
        gci_is_relative=gci_is_relative,
        convergence_ratio=convergence_ratio,
        convergence_type=convergence_type,
        refinement_ratios=ratios,
        uniform_refinement=uniform,
        fit_residual_rel=residual,
        order_triplets=tuple(triplet_orders),
        order_spread=spread,
        noise_ratio=noise_ratio,
        noise_certified=noise_certified,
        gci_pairs=tuple(gci_pairs),
        gci_monotone=gci_monotone,
        extrapolation_gain=extrapolation_gain,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------


def _check(name: str, status: str, detail: str, **metrics: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {"check": name, "status": status, "detail": detail}
    entry.update(metrics)
    return entry


def _worst(statuses: Sequence[str]) -> str:
    return max(statuses, key=lambda s: _STATUS_ORDER[s]) if statuses else PASS


def evaluate(
    levels: Sequence[Observation],
    *,
    quantity: str = "observable",
    policy: GatePolicy | None = None,
    f_exp: float | None = None,
    u_input: float | None = None,
    u_num: float | None = None,
    f_sim: float | None = None,
    features: Sequence[Any] | None = None,
    dx: float | None = None,
    resolvability_policy: Any | None = None,
    require_resolvability: bool = False,
    u_ref: float | None = None,
    discrimination_factor: float = 1.0,
) -> GateVerdict:
    """Run the full gate.

    Stages, in order:

    1. numerical convergence (:func:`grid_convergence_study`): order, GCI,
       asymptotic-range checks;
    2. resolution of the physical features declared in ``features`` under the
       ``R = delta / dx`` law (:mod:`tensorphysics.vv.resolvability`), with
       ``dx`` defaulting to the finest grid spacing of the study.  A subgrid
       feature downgrades the verdict to ``UNRESOLVED``; a missing declaration
       (``require_resolvability=True`` without ``features``) downgrades it to
       ``WITHHELD``;
    3. ASME V&V 20 validation against ``f_exp``.  The plain ``|E| <= U_val``
       test is kept as the historical ``validation/V&V20`` check; when the
       reference uncertainty ``u_ref`` is supplied the full uncertainty ledger
       (:mod:`tensorphysics.vv.uncertainty`) is added and a non-discriminative
       reference turns the validation into ``WITHHELD`` instead of a PASS.
    """
    from .validation import validate

    policy = policy or GatePolicy()
    checks: list[dict[str, Any]] = []
    try:
        study = grid_convergence_study(levels, policy=policy)
    except ValueError as exc:
        checks.append(_check("input", FAIL, f"invalid convergence study: {exc}"))
        return GateVerdict(status=FAIL, quantity=quantity, study=None, checks=tuple(checks), policy=policy)

    checks.append(
        _check(
            "input",
            PASS,
            f"{study.n_levels} monotonically refined levels, h in "
            f"[{study.levels[-1].h:g}, {study.levels[0].h:g}], uniform_refinement={study.uniform_refinement}",
        )
    )

    # 1. asymptotic range: convergence type
    if study.n_levels >= 3:
        if study.convergence_type == "monotone":
            checks.append(
                _check(
                    "asymptotic-range/convergence-type",
                    PASS,
                    f"monotone contraction, convergence ratio R={study.convergence_ratio:.4f}",
                    convergence_ratio=study.convergence_ratio,
                )
            )
        else:
            checks.append(
                _check(
                    "asymptotic-range/convergence-type",
                    FAIL,
                    f"non-monotone or non-contracting sequence: {study.convergence_type} "
                    f"(R={study.convergence_ratio})",
                    convergence_ratio=study.convergence_ratio,
                )
            )
    else:
        checks.append(
            _check(
                "asymptotic-range/convergence-type",
                WARN,
                "two grids: the convergence ratio needs three levels; asymptotics not certified",
            )
        )

    # 2. admissible order band
    if study.order_p is None:
        if study.n_levels == 2:
            checks.append(
                _check(
                    "order-band",
                    WARN,
                    "two grids: the order of accuracy is not observable, supply "
                    "assumed_order to obtain a GCI band of the finest result",
                )
            )
        else:
            checks.append(_check("order-band", FAIL, "an observed order of accuracy could not be extracted"))
    elif study.order_source == "assumed":
        checks.append(
            _check(
                "order-band",
                WARN,
                f"order assumed p={study.order_p:.4f} (not observable from the supplied levels)",
            )
        )
    elif policy.p_min < study.order_p <= policy.p_max:
        checks.append(
            _check(
                "order-band",
                PASS,
                f"observed order p={study.order_p:.6f} ({study.order_source}) inside "
                f"({policy.p_min}, {policy.p_max}]",
                order_p=study.order_p,
            )
        )
    else:
        checks.append(
            _check(
                "order-band",
                FAIL,
                f"observed order p={study.order_p:.6f} outside ({policy.p_min}, {policy.p_max}]: "
                "the study is not in the asymptotic range",
                order_p=study.order_p,
            )
        )

    # 3. consistency with a theoretical order
    if policy.expected_order is not None and study.order_p is not None:
        dev = abs(study.order_p - policy.expected_order)
        if dev <= policy.order_tolerance:
            checks.append(
                _check(
                    "order-band/expected",
                    PASS,
                    f"|p - p_theory| = {dev:.4f} <= {policy.order_tolerance} (p_theory={policy.expected_order})",
                    order_deviation=dev,
                )
            )
        elif dev <= 3.0 * policy.order_tolerance:
            checks.append(
                _check(
                    "order-band/expected",
                    WARN,
                    f"|p - p_theory| = {dev:.4f} exceeds the tolerance {policy.order_tolerance}",
                    order_deviation=dev,
                )
            )
        else:
            checks.append(
                _check(
                    "order-band/expected",
                    FAIL,
                    f"|p - p_theory| = {dev:.4f} is far outside the tolerance {policy.order_tolerance}: "
                    "the observed order contradicts the discretisation",
                    order_deviation=dev,
                )
            )

    # 4. power-law residual (needs >=4 levels)
    if study.fit_residual_rel is not None:
        res = study.fit_residual_rel
        if res <= policy.residual_warn:
            checks.append(
                _check("power-law-residual", PASS, f"relative fit residual {res:.3e}", residual=res)
            )
        elif res <= policy.residual_fail:
            checks.append(
                _check(
                    "power-law-residual",
                    WARN,
                    f"relative fit residual {res:.3e} in the noise band "
                    f"({policy.residual_warn:.0e}, {policy.residual_fail:.0e}]",
                    residual=res,
                )
            )
        else:
            checks.append(
                _check(
                    "power-law-residual",
                    FAIL,
                    f"relative fit residual {res:.3e} > {policy.residual_fail:.0e}: the data do not "
                    "follow a single power law",
                    residual=res,
                )
            )

    # 5. order consistency across triplets
    if study.order_spread is not None:
        spread = study.order_spread
        if spread <= policy.order_consistency_warn:
            checks.append(
                _check("order-consistency", PASS, f"triplet order spread {spread:.4f}", spread=spread)
            )
        elif spread <= policy.order_consistency_fail:
            checks.append(
                _check(
                    "order-consistency",
                    WARN,
                    f"triplet order spread {spread:.4f} in ({policy.order_consistency_warn}, "
                    f"{policy.order_consistency_fail}]",
                    spread=spread,
                )
            )
        else:
            checks.append(
                _check(
                    "order-consistency",
                    FAIL,
                    f"triplet order spread {spread:.4f} > {policy.order_consistency_fail}: the "
                    "apparent order drifts with refinement",
                    spread=spread,
                )
            )

    # 6. noise floor
    if study.noise_ratio is not None:
        nr = study.noise_ratio
        if nr <= policy.noise_warn_ratio:
            checks.append(_check("noise-floor", PASS, f"declared sigma / finest difference = {nr:.4f}", noise_ratio=nr))
        elif nr <= policy.noise_fail_ratio:
            checks.append(
                _check(
                    "noise-floor",
                    WARN,
                    f"declared sigma is {nr * 100:.1f}% of the finest grid difference: the "
                    "extrapolation is noise-sensitive",
                    noise_ratio=nr,
                )
            )
        else:
            checks.append(
                _check(
                    "noise-floor",
                    FAIL,
                    f"declared sigma is {nr * 100:.1f}% of the finest grid difference: the study is "
                    "noise dominated and cannot certify an order",
                    noise_ratio=nr,
                )
            )

    # 7. GCI monotonicity with refinement
    if study.gci_monotone is not None:
        if study.gci_monotone:
            checks.append(
                _check(
                    "gci-monotonicity",
                    PASS,
                    "GCI decreases toward the fine end: " + ", ".join(f"{v:.3e}" for v in study.gci_pairs),
                )
            )
        else:
            checks.append(
                _check(
                    "gci-monotonicity",
                    FAIL,
                    "GCI does not decrease with refinement: " + ", ".join(f"{v:.3e}" for v in study.gci_pairs),
                )
            )

    # 8. strict mode: a three-grid certificate cannot exclude an undeclared noise floor
    if policy.strict and not study.noise_certified:
        checks.append(
            _check(
                "certificate-strength",
                WARN,
                "strict mode: a three-grid PASS cannot certify an undeclared noise floor; "
                "supply >=4 grids or per-level sigma",
            )
        )

    # 9. noise certificate (default conservative policy)
    if policy.pass_requires_noise_certificate and not study.noise_certified:
        checks.append(
            _check(
                "noise-certificate",
                WARN,
                "the noise floor of this study is not certified (three levels without "
                "per-level sigma): the numbers may be a PASS, but an undeclared noise "
                "floor cannot be excluded, so the pass is capped at WARN",
                n_levels=study.n_levels,
            )
        )

    # 10. ASME V&V 20 validation (optional)
    validation = None
    if f_exp is not None:
        if study.gci_fine is None:
            checks.append(
                _check("validation/V&V20", FAIL, "no numerical uncertainty available for validation")
            )
        else:
            u_num_eff = u_num
            if u_num_eff is None:
                u_num_eff = study.gci_fine * abs(study.f_fine) if study.gci_is_relative else study.gci_fine
            validation = validate(
                study.f_fine if f_sim is None else f_sim,
                f_exp,
                u_num=u_num_eff,
                u_input=0.0 if u_input is None else u_input,
            )
            detail = (
                f"E={validation.error:.6g}, U_val={validation.u_val:.6g}, "
                f"|E|/U_val={validation.ratio:.4f}"
            )
            status = PASS if validation.passed else FAIL
            # With a declared reference uncertainty the ledger owns the decision
            # (see stage 12): the plain check is then informational only and must
            # not turn a WITHHELD comparison into a FAIL.
            if u_ref is None:
                checks.append(
                    _check(
                        "validation/V&V20",
                        status,
                        detail if validation.passed else detail
                        + ": the model is not validated against the reference",
                        error=validation.error,
                        u_val=validation.u_val,
                        ratio=validation.ratio,
                    )
                )

    # 11. resolvability law R = delta / dx (W10-N07)
    resolvability = None
    if features is not None or require_resolvability:
        from .resolvability import (
            MARGINAL,
            RESOLVED,
            UNRESOLVED as UNRESOLVED_FEATURE,
            WITHHELD as WITHHELD_FEATURE,
            discriminability,
        )

        feats = list(features or ())
        dx_eff = dx if dx is not None else (study.levels[-1].h if study is not None else None)
        policy_res = resolvability_policy
        if policy_res is not None:
            policy_res = policy_res.merged()
        resolvability = discriminability(feats, dx_eff, policy=policy_res)
        check_status = {
            RESOLVED: PASS,
            MARGINAL: WARN,
            UNRESOLVED_FEATURE: UNRESOLVED,
            WITHHELD_FEATURE: WITHHELD,
        }[resolvability.status]
        checks.append(
            _check(
                "resolvability/R=delta/dx",
                check_status,
                "; ".join(resolvability.reasons)
                if resolvability.reasons
                else f"all features resolved: min R={resolvability.ratio:.4f} on dx={dx_eff:g}",
                ratio=resolvability.ratio,
                dx=dx_eff,
                features=len(resolvability.features),
            )
        )

    # 12. ASME V&V 20 uncertainty ledger (always reported, decision-changing
    #     only when the reference uncertainty U_ref is declared)
    uncertainty = None
    if f_exp is not None and study is not None and study.gci_fine is not None:
        from .uncertainty import compare as _compare
        from .uncertainty import ledger_from_study

        u_input_eff = 0.0 if u_input is None else u_input
        u_ref_eff = 0.0 if u_ref is None else u_ref
        uncertainty = ledger_from_study(study, u_input=u_input_eff, u_ref=u_ref_eff)
        if u_ref is not None:
            f_sim_eff = study.f_fine if f_sim is None else f_sim
            ledger_verdict = _compare(
                f_sim_eff, f_exp, uncertainty, discrimination_factor=discrimination_factor
            )
            checks.append(
                _check(
                    "validation/V&V20-ledger",
                    {PASS: PASS, "FAIL": FAIL, "WITHHELD": WITHHELD}[ledger_verdict.status],
                    ledger_verdict.detail,
                    error=ledger_verdict.error,
                    u_val=uncertainty.u_val,
                    u_ref=uncertainty.u_ref,
                    reference_ratio=uncertainty.reference_ratio,
                )
            )

    statuses = [c["status"] for c in checks]
    status = _worst(statuses)
    return GateVerdict(
        status=status,
        quantity=quantity,
        study=study,
        checks=tuple(checks),
        validation=validation,
        policy=policy,
        resolvability=resolvability,
        uncertainty=uncertainty,
    )


def _cli() -> int:  # pragma: no cover - thin wrapper
    from .cli import main

    return main()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())