"""Three-state UQ verdict: PASS / FAIL / WITHHELD with machine-readable reasons.

The verdict is the *acceptance gate* the rest of the platform can chain on: it
turns the UQ output into a single number -- the width of the 95% output
interval ``U`` -- compares it with an engineering threshold (default 15% of the
mean), and refuses to answer (``WITHHELD``) whenever the evidence behind that
number is not trustworthy.  Withholding is deliberately *not* a failure: it
carries the reason codes that say what to fix.

Definition of ``U``
-------------------
``U`` is the **95% prediction interval width** of the quantity of interest,

    U = q_{0.975}[QoI] - q_{0.025}[QoI],

obtained by propagating the input uncertainty through the certified surrogate
(``n_mc`` fresh draws, independent of the training set).  ``U`` is compared
with ``threshold * |mean|``; the dimensionless quotients
``relative_width = U / |mean|`` and ``ratio = relative_width / threshold`` are
reported so that a marginal pass is visible.  The interval of the *mean*
(``1.96 sd / sqrt(n_mc)``) is also reported but is **not** the gate: it shrinks
with the number of Monte-Carlo draws and would let a wide physical error band
pass by drawing more points.

Reason codes
------------
====================  =================================================
code                  meaning
====================  =================================================
``UQ-W01-NONFINITE``  solver returned NaN/Inf for >= 1 sample
``UQ-W02-TOOFEW``     too few valid samples (per input or per active term)
``UQ-W03-LOO``        surrogate LOO error above ``eps_loo_max``
``UQ-W04-CROSSCHECK`` chaos vs Monte-Carlo indices disagree
``UQ-W05-DEGENERATE`` QoI variance is zero / not estimable
``UQ-W06-HASH``       results belong to a different sample set
``UQ-W07-NOMC``       Monte-Carlo cross-check missing (when required)
``UQ-W08-NOPCE``      no surrogate available to certify the budget
====================  =================================================

``UQ-N01-CORRELATED``, ``UQ-N02-NOISY-MC`` and ``UQ-N03-WIDE-INDEX-CI`` are
non-blocking notes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from .pce import PCESurrogate, SobolIndices
from .sample_spec import SampleSpec
from .sampler import draw_unit
from .sobol_mc import SobolMCResult

__all__ = [
    "DEFAULT_EPS_LOO_MAX",
    "DEFAULT_MIN_SAMPLES",
    "DEFAULT_THRESHOLD",
    "PASS",
    "FAIL",
    "WITHHELD",
    "CrossCheck",
    "CrossCheckEntry",
    "OutputBudget",
    "Reason",
    "UQVerdict",
    "cross_check_indices",
    "decide",
    "min_samples_floor",
    "output_budget",
]

PASS = "PASS"
FAIL = "FAIL"
WITHHELD = "WITHHELD"

#: Default engineering threshold on the relative 95% interval width.
DEFAULT_THRESHOLD = 0.15
#: Default surrogate-quality floor on the normalised LOO error.
DEFAULT_EPS_LOO_MAX = 0.10
#: Default lower bound on the number of *valid* solver runs.
DEFAULT_MIN_SAMPLES = 32
#: Samples per input demanded by :func:`min_samples_floor`.
SAMPLES_PER_DIM = 10
#: Samples per *active chaos term*.  PRESS (the LOO estimate) is known to be
#: optimistic when the active set is nearly as large as the sample set: measured
#: on the Ishigami anchor at degree 6, ``N = 64`` with 30 active terms gave
#: ``eps_LOO = 0.072`` while the true error against 20 000 fresh points was
#: ``0.241``; at ``N >= 512`` the two agreed to within 5%.  Four samples per
#: active term is the empirical floor below which the LOO evidence is refused.
SAMPLES_PER_TERM = 4.0


def min_samples_floor(dim: int) -> int:
    """Minimum valid sample count for ``dim`` inputs (``max(32, 10*dim)``)."""
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    return max(DEFAULT_MIN_SAMPLES, SAMPLES_PER_DIM * dim)


@dataclass(frozen=True)
class Reason:
    """One machine-readable diagnosis attached to a verdict."""

    code: str
    message: str
    blocking: bool = True

    @property
    def severity(self) -> str:
        return "blocking" if self.blocking else "note"

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
        }


@dataclass(frozen=True)
class OutputBudget:
    """95% prediction interval of the quantity of interest."""

    mean: float
    sd: float
    lower95: float
    upper95: float
    width: float
    relative_width: float
    n_mc: int
    level: float = 0.95
    definition: str = "prediction_interval"

    @property
    def ci_mean_halfwidth(self) -> float:
        """Half-width of the confidence interval *of the mean* (informational)."""
        if self.n_mc <= 0:
            return math.nan
        return 1.959963984540054 * self.sd / math.sqrt(self.n_mc)

    def as_dict(self) -> dict[str, Any]:
        return {
            "definition": self.definition,
            "mean": self.mean,
            "sd": self.sd,
            "lower95": self.lower95,
            "upper95": self.upper95,
            "width": self.width,
            "relative_width": self.relative_width,
            "n_mc": int(self.n_mc),
            "level": self.level,
            "ci_mean_halfwidth": self.ci_mean_halfwidth,
        }


@dataclass(frozen=True)
class CrossCheckEntry:
    """One index compared between the chaos and Monte-Carlo routes."""

    kind: str
    label: str
    pce: float
    mc: float
    deviation: float
    relative_deviation: float | None
    tolerance: float | None = None
    within_tolerance: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "label": self.label,
            "pce": self.pce,
            "mc": self.mc,
            "deviation": self.deviation,
            "relative_deviation": self.relative_deviation,
            "tolerance": self.tolerance,
            "within_tolerance": self.within_tolerance,
        }


@dataclass(frozen=True)
class CrossCheck:
    """Agreement between the analytic (PCE) and sampled (Saltelli) indices."""

    entries: tuple[CrossCheckEntry, ...]
    max_relative_deviation: float
    agreement: bool
    rel_tol: float
    floor: float
    n_compared: int
    max_second_relative_deviation: float = 0.0
    n_second_compared: int = 0
    rel_tol_second: float = 0.25
    n_noise_limited: int = 0
    material_floor: float = 0.05
    max_material_relative_deviation: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "agreement": self.agreement,
            "rel_tol": self.rel_tol,
            "floor": self.floor,
            "n_compared": self.n_compared,
            "max_relative_deviation": self.max_relative_deviation,
            "max_second_relative_deviation": self.max_second_relative_deviation,
            "n_second_compared": self.n_second_compared,
            "n_noise_limited": self.n_noise_limited,
            "max_material_relative_deviation": self.max_material_relative_deviation,
            "material_floor": self.material_floor,
            "rel_tol_second": self.rel_tol_second,
            "entries": [e.as_dict() for e in self.entries],
        }


@dataclass(frozen=True)
class UQVerdict:
    """Outcome of the UQ acceptance gate."""

    status: str
    reasons: tuple[Reason, ...]
    budget: OutputBudget | None
    threshold: float
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == PASS

    @property
    def withheld(self) -> bool:
        return self.status == WITHHELD

    @property
    def reason_codes(self) -> tuple[str, ...]:
        return tuple(r.code for r in self.reasons)

    @property
    def blocking_codes(self) -> tuple[str, ...]:
        return tuple(r.code for r in self.reasons if r.blocking)

    @property
    def ratio(self) -> float | None:
        if self.budget is None or self.threshold <= 0.0:
            return None
        return self.budget.relative_width / self.threshold

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "status": self.status,
            "threshold": self.threshold,
            "ratio": self.ratio,
            "reason_codes": list(self.reason_codes),
            "blocking_codes": list(self.blocking_codes),
            "reasons": [r.as_dict() for r in self.reasons],
        }
        if self.budget is not None:
            out["budget"] = self.budget.as_dict()
        out["details"] = dict(self.details)
        return out


def output_budget(
    surrogate: PCESurrogate,
    spec: SampleSpec,
    *,
    n_mc: int = 20000,
    seed: int = 9091,
    level: float = 0.95,
) -> OutputBudget:
    """Propagate the input uncertainty through ``surrogate`` to a 95% band."""
    if not (0.5 < level < 1.0):
        raise ValueError(f"level must be in (0.5, 1), got {level}")
    unit = draw_unit(int(n_mc), spec.dim, rule="sobol", seed=seed)
    latent = spec.latent_from_unit(unit)
    _, coords = spec.from_latent(latent)
    values = np.asarray(surrogate.evaluate(coords), dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("surrogate produced non-finite values during budget propagation")
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    alpha = 0.5 * (1.0 - level) * 100.0
    lo = float(np.percentile(values, alpha))
    hi = float(np.percentile(values, 100.0 - alpha))
    width = hi - lo
    rel = width / abs(mean) if mean != 0.0 else math.inf
    return OutputBudget(
        mean=mean,
        sd=sd,
        lower95=lo,
        upper95=hi,
        width=width,
        relative_width=float(rel),
        n_mc=int(values.size),
        level=level,
    )


def cross_check_indices(
    pce_indices: SobolIndices,
    mc: SobolMCResult,
    *,
    names: Sequence[str] | None = None,
    rel_tol: float = 0.05,
    floor: float = 0.02,
    rel_tol_second: float = 0.25,
    material_floor: float = 0.05,
) -> CrossCheck:
    """Compare PCE (analytic) and Saltelli (sampled) Sobol' indices.

    Only indices whose chaos value is at least ``floor`` in magnitude are
    scored with a *relative* deviation -- an index that is numerically zero has
    no meaningful relative error, and MC noise always leaves a few percent
    absolute scatter there.

    First- and total-order indices must agree within ``rel_tol`` (default 5%),
    *or* within three Monte-Carlo standard errors when the Saltelli run carries
    bootstrap confidence intervals: a 4% deviation on an index of 0.03 is not a
    methodological disagreement, it is the sampling noise of the Monte-Carlo
    route, and scoring it as a 5% relative error would either force an
    unnecessarily huge ``n_base`` or withhold a perfectly good chaos fit.  The
    number of indices that only pass thanks to the noise allowance is reported
    as ``n_noise_limited`` so the reader can see when the MC route is the
    limiting factor.
    Closed second-order indices are scored separately against
    ``rel_tol_second`` (default 25%): the pick-freeze second-order estimator
    contains a four-term product and therefore carries several times the
    sampling variance of the first-order estimator at the same ``n_base``
    (measured on the Ishigami anchor: ~10% at ``n_base = 2**15`` versus ~3% for
    first/total order).  The measured deviations are always reported, so the
    looser tolerance is visible rather than hidden.
    """
    dim = int(pce_indices.first.size)
    if mc.dim != dim:
        raise ValueError(f"dimension mismatch: pce {dim}, mc {mc.dim}")
    names = tuple(names) if names is not None else tuple(f"x{i}" for i in range(dim))
    entries: list[CrossCheckEntry] = []
    worst = 0.0
    worst_second = 0.0
    n_compared = 0
    n_second = 0
    n_noise = 0
    worst_material = 0.0

    def _se(ci: tuple[np.ndarray, np.ndarray] | None, i: int) -> float | None:
        if ci is None:
            return None
        lo, hi = ci
        width = float(hi[i] - lo[i])
        if not math.isfinite(width) or width <= 0.0:
            return None
        return width / (2.0 * 1.959963984540054)

    def _add(
        kind: str, label: str, p: float, m: float, se: float | None = None
    ) -> None:
        nonlocal worst, n_compared, worst_second, n_second, n_noise, worst_material
        dev = abs(p - m)
        rel = dev / abs(p) if p != 0.0 else None
        if kind == "second":
            if abs(p) >= floor:
                tol = rel_tol_second * abs(p)
                ok = dev <= tol
                n_second += 1
                worst_second = max(worst_second, float(rel if rel is not None else math.inf))
            else:
                # the chaos says "no interaction here": the sampled route must
                # agree that the interaction is below the reporting floor.
                tol = floor
                ok = abs(m) < floor
        else:
            noise_abs = 3.0 * se if se is not None else 0.0
            if abs(p) >= floor:
                rel_tol_abs = rel_tol * abs(p)
                tol = max(rel_tol_abs, noise_abs)
                ok = dev <= tol
                if noise_abs > rel_tol_abs:
                    n_noise += 1
                n_compared += 1
                value = float(rel if rel is not None else math.inf)
                worst = max(worst, value)
                if abs(p) >= material_floor:
                    worst_material = max(worst_material, value)
            else:
                # the chaos says "negligible main effect": the sampled route must
                # agree that the index is below the reporting floor
                tol = max(floor, noise_abs)
                ok = abs(m) < tol
        entries.append(
            CrossCheckEntry(
                kind=kind,
                label=label,
                pce=float(p),
                mc=float(m),
                deviation=float(dev),
                relative_deviation=rel,
                tolerance=float(tol) if tol is not None else None,
                within_tolerance=bool(ok),
            )
        )

    for i in range(dim):
        _add(
            "first",
            names[i],
            float(pce_indices.first[i]),
            float(mc.first[i]),
            _se(mc.first_ci, i),
        )
    for i in range(dim):
        _add(
            "total",
            names[i],
            float(pce_indices.total[i]),
            float(mc.total[i]),
            _se(mc.total_ci, i),
        )
    for (i, j), value in pce_indices.second.items():
        if (i, j) in mc.second:
            _add("second", f"{names[i]}*{names[j]}", float(value), float(mc.second[(i, j)]))
    agreement = (
        bool(entries)
        and all(e.within_tolerance for e in entries)
        and (n_second == 0 or worst_second <= rel_tol_second)
    )
    return CrossCheck(
        entries=tuple(entries),
        max_relative_deviation=float(worst),
        agreement=bool(agreement),
        rel_tol=float(rel_tol),
        floor=float(floor),
        n_compared=int(n_compared),
        max_second_relative_deviation=float(worst_second),
        n_second_compared=int(n_second),
        rel_tol_second=float(rel_tol_second),
        n_noise_limited=int(n_noise),
        material_floor=float(material_floor),
        max_material_relative_deviation=float(worst_material),
    )


def decide(
    *,
    budget: OutputBudget | None,
    threshold: float = DEFAULT_THRESHOLD,
    surrogate: PCESurrogate | None = None,
    mc: SobolMCResult | None = None,
    cross_check: CrossCheck | None = None,
    samples=None,
    results=None,
    eps_loo_max: float = DEFAULT_EPS_LOO_MAX,
    min_samples: int | None = None,
    min_samples_per_term: float = SAMPLES_PER_TERM,
    require_cross_check: bool = True,
    budget_certified_directly: bool = False,
    extra_notes: Iterable[Reason] = (),
    details: dict[str, Any] | None = None,
) -> UQVerdict:
    """Decide PASS / FAIL / WITHHELD from the UQ evidence.

    Every input is optional except ``budget``: what is missing is *reported*
    (a missing cross-check withholds unless ``require_cross_check=False``), so
    a partial run can never masquerade as a pass.

    ``budget_certified_directly`` declares that ``budget`` was measured on
    solver output (a direct quasi-Monte-Carlo prediction interval) rather than
    propagated through a chaos surrogate.  Only then is the missing surrogate
    *not* a blocking reason (``UQ-W08-NOPCE``); the route is still recorded on
    the verdict through the caller's non-blocking note.  The default ``False``
    keeps the surrogate-required semantics unchanged for every existing caller.
    """
    if threshold <= 0.0 or not math.isfinite(threshold):
        raise ValueError(f"threshold must be positive and finite, got {threshold!r}")
    reasons: list[Reason] = []
    detected_dim = None
    if samples is not None:
        detected_dim = int(samples.dim) if hasattr(samples, "dim") else None

    # ---- provenance ----------------------------------------------------------
    if samples is not None and results is not None:
        expected = getattr(samples, "hash", None)
        received = getattr(results, "sample_hash", None)
        if expected is not None and received is not None and expected != received:
            reasons.append(
                Reason(
                    "UQ-W06-HASH",
                    f"results are tagged {str(received)[:12]}..., expected {str(expected)[:12]}...",
                )
            )
    if results is not None:
        n_invalid = int(getattr(results, "n", 0)) - int(getattr(results, "n_valid", 0))
        if n_invalid:
            idx = list(getattr(results, "invalid_indices", ()))[:10]
            msgs = list(getattr(results, "error_messages", ()))[:3]
            reasons.append(
                Reason(
                    "UQ-W01-NONFINITE",
                    f"{n_invalid} of {getattr(results, 'n', '?')} solver samples are "
                    f"non-finite (first indices {idx}); messages: {msgs}",
                )
            )

    # ---- sample-count and surrogate quality ---------------------------------
    n_valid = None
    if results is not None:
        n_valid = int(getattr(results, "n_valid", 0))
    elif surrogate is not None:
        n_valid = int(surrogate.n_samples)
    dim = detected_dim or (surrogate.dim if surrogate is not None else 1)
    floor_n = min_samples if min_samples is not None else min_samples_floor(dim)
    if n_valid is not None and n_valid < floor_n:
        reasons.append(
            Reason(
                "UQ-W02-TOOFEW",
                f"{n_valid} valid samples < lower bound {floor_n} "
                f"(max({DEFAULT_MIN_SAMPLES}, {SAMPLES_PER_DIM}*dim))",
            )
        )
    if (
        surrogate is not None
        and n_valid is not None
        and surrogate.n_terms
        and n_valid < min_samples_per_term * surrogate.n_terms
    ):
        reasons.append(
            Reason(
                "UQ-W02-TOOFEW",
                f"{n_valid} valid samples for {surrogate.n_terms} active chaos terms "
                f"(< {min_samples_per_term:g} per term): the LOO estimate is known to be "
                "optimistic in this regime",
            )
        )
    if surrogate is None:
        if budget_certified_directly:
            reasons.append(
                Reason(
                    "UQ-N05-MC-FAILSAFE",
                    "no polynomial-chaos surrogate: the band is a direct solver "
                    "prediction interval and the indices come from the quasi-Monte-Carlo "
                    "route (degraded mode)",
                    blocking=False,
                )
            )
        else:
            reasons.append(
                Reason(
                    "UQ-W08-NOPCE",
                    "no polynomial-chaos surrogate available to certify the budget",
                )
            )
    elif surrogate.eps_loo > eps_loo_max:
        reasons.append(
            Reason(
                "UQ-W03-LOO",
                f"surrogate LOO error {surrogate.eps_loo:.3g} > {eps_loo_max:.3g} "
                f"(normalised by std(y); absolute {surrogate.eps_loo_abs:.3g})",
            )
        )

    # ---- cross-check ---------------------------------------------------------
    if cross_check is None:
        if require_cross_check:
            reasons.append(
                Reason(
                    "UQ-W07-NOMC",
                    "no Monte-Carlo/Saltelli cross-check was performed "
                    "(independent second opinion required)",
                )
            )
    elif not cross_check.agreement:
        reasons.append(
            Reason(
                "UQ-W04-CROSSCHECK",
                f"PCE and Saltelli indices disagree: max relative deviation "
                f"{cross_check.max_relative_deviation:.3g} > {cross_check.rel_tol:.3g} "
                f"over {cross_check.n_compared} first/total indices "
                f"(second order {cross_check.max_second_relative_deviation:.3g} vs "
                f"{cross_check.rel_tol_second:.3g} over {cross_check.n_second_compared})",
            )
        )

    # ---- budget --------------------------------------------------------------
    if budget is None:
        reasons.append(Reason("UQ-W05-DEGENERATE", "no output budget was computed"))
    else:
        if not (math.isfinite(budget.width) and math.isfinite(budget.relative_width)):
            reasons.append(
                Reason("UQ-W05-DEGENERATE", "output interval width is not finite")
            )
        elif budget.sd <= 1e-12 * max(1.0, abs(budget.mean)):
            reasons.append(
                Reason(
                    "UQ-W05-DEGENERATE",
                    f"output standard deviation {budget.sd:.3g} is numerically zero; "
                    "the error band cannot be certified",
                )
            )

    # ---- notes ---------------------------------------------------------------
    if samples is not None and getattr(samples.spec, "correlated", False):
        reasons.append(
            Reason(
                "UQ-N01-CORRELATED",
                "correlated inputs: Sobol indices are latent-coordinate (Gaussian "
                "copula) indices, not marginal ones",
                blocking=False,
            )
        )
    if mc is not None:
        spread = float(np.max(mc.estimator_spread)) if mc.dim else 0.0
        if spread > 0.05:
            reasons.append(
                Reason(
                    "UQ-N02-NOISY-MC",
                    f"Jansen and Saltelli first-order estimators differ by {spread:.3g}; "
                    "Monte-Carlo noise at this n_base is high",
                    blocking=False,
                )
            )
        if mc.first_ci is not None:
            lo, hi = mc.first_ci
            widest = float(np.max(hi - lo)) if mc.dim else 0.0
            if widest > 0.10:
                reasons.append(
                    Reason(
                        "UQ-N03-WIDE-INDEX-CI",
                        f"widest 95% index interval is {widest:.3g}; indices are "
                        "resolved only coarsely",
                        blocking=False,
                    )
                )
    reasons.extend(extra_notes)

    blocking = [r for r in reasons if r.blocking]
    if blocking:
        status = WITHHELD
    elif budget is not None and budget.relative_width > threshold:
        status = FAIL
    else:
        status = PASS

    detail_out = dict(details or {})
    detail_out.setdefault("n_valid", n_valid)
    detail_out.setdefault("min_samples", floor_n)
    detail_out.setdefault("min_samples_per_term", float(min_samples_per_term))
    detail_out.setdefault("eps_loo_max", float(eps_loo_max))
    detail_out.setdefault(
        "eps_loo", None if surrogate is None else float(surrogate.eps_loo)
    )
    detail_out.setdefault(
        "cross_check",
        None if cross_check is None else cross_check.as_dict(),
    )
    return UQVerdict(
        status=status,
        reasons=tuple(reasons),
        budget=budget,
        threshold=float(threshold),
        details=detail_out,
    )