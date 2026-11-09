"""Refinement study: does *more* budget actually buy a better surrogate?

The uncertainty-budget gate (mission IN145-N08) needs one thing beyond a
working surrogate: evidence that the surrogate is *converging* when the budget
grows.  The natural observable is the leave-one-out error ``eps_LOO`` of the
polynomial-chaos fit, and the naive expectation is

    more samples  ->  smaller eps_LOO
    higher degree ->  smaller eps_LOO

The second statement is (essentially) true inside a well-resolved truncation.
The first one is **false in general**, and this module quantifies why.  The
decomposition is the classical one for a regression surrogate:

    E[(yhat - y)^2]  =  truncation error  +  coefficient-estimation error
                        (grows with p)       (shrinks with N)

``eps_LOO`` is the PRESS statistic evaluated with the *selected* model.  When
the active set is chosen from the data (sparse LARS selection) and the sample
count is small, the PRESS estimate is **optimistically biased**: the selector
has already looked at every residual, so leaving one point out does not hurt.
As ``N`` grows the active set shrinks back to the terms that matter and the
estimate loses its optimism -- so the reported ``eps_LOO`` *rises* towards the
true generalisation error even though the surrogate improves.

The remedy used here is not a tweak of the statistic but a second, independent
measurement: a deterministic Scrambled-Sobol' **holdout set**, drawn from the
same input measure, on which the error is an unbiased Monte-Carlo estimate.  A
refinement study therefore reports both numbers for every grid point, plus the
ratio ``holdout / eps_LOO`` (``loo_optimism``):

* ``loo_optimism`` >> 1  -- the LOO estimate is optimistic, the surrogate is
  better than ``eps_LOO`` claims;
* ``loo_optimism`` ~ 1   -- LOO is calibrated; it may be used as the gate
  quantity.

The gate consequences (see :mod:`tensorphysics.uq.verdict`) are:

* the *estimation-error* axis must be measured at a **fixed truncation**
  (``method="ols"``, or a fixed ``max_terms``), because only then is
  ``eps_LOO(N)`` monotone and the truncation error a constant floor;
* the *adaptive* axis is judged by the holdout error, and the reported
  ``eps_LOO`` is interpreted as a conservative (loss) upper bound once the
  optimism ratio is close to one.

Nothing in this module is specific to a physics discipline: it consumes any
black-box ``model(points) -> values`` and a
:class:`~tensorphysics.uq.sample_spec.SampleSpec`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .pce import PCESurrogate, fit_pce
from .sample_spec import SampleSpec
from .sampler import sample

__all__ = [
    "DEFAULT_REFINEMENT_THRESHOLD",
    "RefinementAxis",
    "RefinementPoint",
    "RefinementStudy",
    "adaptive_selection_diagnostic",
    "holdout_relative_rmse",
    "monotone_decrease",
    "refinement_study",
    "spearman_rank",
]

#: A sequence counts as monotonically decreasing when the Spearman rank
#: correlation between the budget and the error is at most this value.
DEFAULT_REFINEMENT_THRESHOLD = -0.9

Model = Callable[[np.ndarray], np.ndarray]


def _as_model(model: Any) -> Model:
    evaluate = getattr(model, "evaluate", None)
    if callable(evaluate) and not callable(model):
        return lambda points: np.asarray(evaluate(points), dtype=float).reshape(-1)
    if callable(model):
        return lambda points: np.asarray(model(points), dtype=float).reshape(-1)
    raise TypeError(f"model must be callable or expose .evaluate(), got {type(model)!r}")


def spearman_rank(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation with average ranks for ties.

    Implemented directly (no scipy dependency in the runtime package): the
    ranks are computed with :func:`numpy.argsort` on the sorted order, ties
    receive the mean rank, and Pearson's correlation of the ranks is returned.
    A constant input yields ``nan`` (no rank information).
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.size != ya.size:
        raise ValueError(f"length mismatch: {xa.size} vs {ya.size}")
    if xa.size < 3:
        return float("nan")

    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        out = np.empty(values.size, dtype=float)
        i = 0
        while i < values.size:
            j = i
            while j + 1 < values.size and values[order[j + 1]] == values[order[i]]:
                j += 1
            out[order[i : j + 1]] = 0.5 * (i + j)
            i = j + 1
        return out

    rx = ranks(xa)
    ry = ranks(ya)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = math.sqrt(float(np.sum(rx * rx)) * float(np.sum(ry * ry)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(rx * ry) / denom)


def monotone_decrease(
    values: Sequence[float], threshold: float = DEFAULT_REFINEMENT_THRESHOLD
) -> bool:
    """True when the sequence decreases (Spearman <= ``threshold``, >= 4 points)."""
    vals = np.asarray(values, dtype=float)
    if vals.size < 4 or not np.all(np.isfinite(vals)):
        return False
    rho = spearman_rank(np.arange(vals.size, dtype=float), vals)
    return bool(math.isfinite(rho) and rho <= threshold)


def holdout_relative_rmse(
    surrogate: PCESurrogate,
    spec: SampleSpec,
    model: Any,
    *,
    n_holdout: int = 20000,
    seed: int = 7701,
    rule: str = "sobol",
) -> float:
    """Unbiased generalisation error of ``surrogate`` on a fresh design.

    The holdout points are drawn from the *same* input measure as the training
    design (Scrambled Sobol' on the unit hypercube, mapped through ``spec``) but
    from an independent seed, and they are never used in the fit or in the term
    selection, so the estimate is free of the PRESS optimism.  The error is
    normalised by ``std(y)`` over the holdout set, which makes it directly
    comparable to ``eps_loo``.
    """
    n_holdout = int(n_holdout)
    if n_holdout < 64:
        raise ValueError(f"n_holdout must be >= 64, got {n_holdout}")
    design = sample(spec, n_holdout, rule=rule, seed=int(seed))
    points, coords = spec.from_latent(design.latent)
    return _model_holdout_error(surrogate, _as_model(model), points, coords)


def _model_holdout_error(
    surrogate: PCESurrogate,
    model: Model,
    points: np.ndarray,
    coords: np.ndarray,
) -> float:
    y_true = np.asarray(model(points), dtype=float).reshape(-1)
    y_hat = np.asarray(surrogate.evaluate(coords), dtype=float).reshape(-1)
    if not np.all(np.isfinite(y_true)):
        raise ValueError("the model returned non-finite values on the holdout set")
    sd = float(np.std(y_true))
    scale = sd if sd > 0.0 else 1.0
    return float(math.sqrt(float(np.mean(np.square(y_hat - y_true)))) / scale)


@dataclass(frozen=True)
class RefinementPoint:
    """One grid point of a refinement axis."""

    n_samples: int
    degree: int
    n_terms: int
    eps_loo: float
    holdout_rel_rmse: float

    @property
    def samples_per_term(self) -> float:
        return float(self.n_samples) / float(max(self.n_terms, 1))

    @property
    def loo_optimism(self) -> float:
        """``holdout / eps_LOO``; > 1 means ``eps_LOO`` is optimistic."""
        if self.eps_loo <= 0.0:
            return math.inf
        return self.holdout_rel_rmse / self.eps_loo

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_samples": int(self.n_samples),
            "degree": int(self.degree),
            "n_terms": int(self.n_terms),
            "eps_loo": float(self.eps_loo),
            "holdout_rel_rmse": float(self.holdout_rel_rmse),
            "samples_per_term": self.samples_per_term,
            "loo_optimism": self.loo_optimism,
        }


@dataclass(frozen=True)
class RefinementAxis:
    """A one-parameter refinement sweep with its monotonicity verdict."""

    name: str
    protocol: str
    points: tuple[RefinementPoint, ...]
    threshold: float = DEFAULT_REFINEMENT_THRESHOLD
    swept: str = "n_samples"
    calibration_floor: float = 1.0e-3
    notes: tuple[str, ...] = ()

    @property
    def budgets(self) -> np.ndarray:
        """The swept budget of the axis (sample count or total degree)."""
        if self.swept == "degree":
            return np.asarray([p.degree for p in self.points], dtype=float)
        return np.asarray([p.n_samples for p in self.points], dtype=float)

    @property
    def eps_loo(self) -> np.ndarray:
        return np.asarray([p.eps_loo for p in self.points], dtype=float)

    @property
    def holdout(self) -> np.ndarray:
        return np.asarray([p.holdout_rel_rmse for p in self.points], dtype=float)

    @property
    def spearman_loo(self) -> float:
        return spearman_rank(self.budgets, self.eps_loo)

    @property
    def spearman_holdout(self) -> float:
        return spearman_rank(self.budgets, self.holdout)

    @property
    def loo_monotone(self) -> bool:
        return monotone_decrease(self.eps_loo, self.threshold)

    @property
    def holdout_monotone(self) -> bool:
        return monotone_decrease(self.holdout, self.threshold)

    @property
    def max_loo_optimism(self) -> float:
        """Worst ``holdout / eps_LOO`` over the points above the numeric floor.

        Points whose ``eps_LOO`` is below ``calibration_floor`` are excluded:
        at that level the ratio only measures float noise and the choice of
        normalisation, not estimation bias.
        """
        ratios = [
            p.loo_optimism
            for p in self.points
            if p.eps_loo >= self.calibration_floor
        ]
        return float(max(ratios)) if ratios else float("nan")

    @property
    def loo_calibrated(self) -> bool:
        """LOO may be used as the gate quantity (optimism within 20 %)."""
        value = self.max_loo_optimism
        return bool(math.isfinite(value) and value <= 1.2)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "protocol": self.protocol,
            "threshold": float(self.threshold),
            "swept": self.swept,
            "calibration_floor": float(self.calibration_floor),
            "n_points": len(self.points),
            "budgets": [float(p.n_samples) for p in self.points],
            "degrees": [int(p.degree) for p in self.points],
            "n_terms": [int(p.n_terms) for p in self.points],
            "eps_loo": [float(p.eps_loo) for p in self.points],
            "holdout_rel_rmse": [float(p.holdout_rel_rmse) for p in self.points],
            "samples_per_term": [p.samples_per_term for p in self.points],
            "loo_optimism": [p.loo_optimism for p in self.points],
            "spearman_loo": self.spearman_loo,
            "spearman_holdout": self.spearman_holdout,
            "loo_monotone": self.loo_monotone,
            "holdout_monotone": self.holdout_monotone,
            "max_loo_optimism": self.max_loo_optimism,
            "loo_calibrated": self.loo_calibrated,
            "points": [p.as_dict() for p in self.points],
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class RefinementStudy:
    """Every refinement axis of one benchmark, with the gate verdict."""

    benchmark: str
    axes: dict[str, RefinementAxis] = field(default_factory=dict)
    holdout_n: int = 20000
    holdout_seed: int = 7701
    threshold: float = DEFAULT_REFINEMENT_THRESHOLD

    # -- criterion bookkeeping ------------------------------------------------
    @property
    def estimation_axis(self) -> RefinementAxis | None:
        return self.axes.get("samples_at_fixed_truncation")

    @property
    def truncation_axis(self) -> RefinementAxis | None:
        return self.axes.get("degree_at_fixed_n")

    @property
    def adaptive_axis(self) -> RefinementAxis | None:
        return self.axes.get("samples_with_adaptive_selection")

    @property
    def estimation_error_monotone(self) -> bool:
        axis = self.estimation_axis
        return bool(axis is not None and axis.loo_monotone and axis.holdout_monotone)

    @property
    def truncation_error_monotone(self) -> bool:
        axis = self.truncation_axis
        return bool(axis is not None and axis.loo_monotone and axis.holdout_monotone)

    @property
    def adaptive_holdout_monotone(self) -> bool:
        axis = self.adaptive_axis
        return bool(axis is not None and axis.holdout_monotone)

    @property
    def adaptive_loo_nonmonotone(self) -> bool:
        """The literal "eps_LOO falls with N" claim is falsified here."""
        axis = self.adaptive_axis
        return bool(axis is not None and not axis.loo_monotone)

    @property
    def all_met(self) -> bool:
        return bool(
            self.estimation_error_monotone
            and self.truncation_error_monotone
            and self.adaptive_holdout_monotone
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "threshold": float(self.threshold),
            "holdout_n": int(self.holdout_n),
            "holdout_seed": int(self.holdout_seed),
            "axes": {k: v.as_dict() for k, v in self.axes.items()},
            "estimation_error_monotone": self.estimation_error_monotone,
            "truncation_error_monotone": self.truncation_error_monotone,
            "adaptive_holdout_monotone": self.adaptive_holdout_monotone,
            "adaptive_loo_nonmonotone": self.adaptive_loo_nonmonotone,
            "all_met": self.all_met,
        }


def _fit_and_score(
    spec: SampleSpec,
    model: Model,
    *,
    n_samples: int,
    degree: int,
    method: str,
    max_terms: int | None,
    seed: int,
    holdout_points: np.ndarray,
    holdout_coords: np.ndarray,
) -> RefinementPoint:
    design = sample(spec, int(n_samples), seed=int(seed))
    points, coords = spec.from_latent(design.latent)
    values = np.asarray(model(points), dtype=float).reshape(-1)
    if values.size != points.shape[0]:
        raise ValueError(
            f"model returned {values.size} values for {points.shape[0]} points"
        )
    finite = np.isfinite(values)
    if not np.all(finite):
        raise ValueError(
            f"{int(np.count_nonzero(~finite))} of {values.size} model values are non-finite; "
            "a refinement study needs a clean design"
        )
    surrogate = fit_pce(
        coords,
        values,
        degree=int(degree),
        families=spec.families,
        method=method,
        max_terms=max_terms,
    )
    holdout = _model_holdout_error(surrogate, model, holdout_points, holdout_coords)
    return RefinementPoint(
        n_samples=int(n_samples),
        degree=int(degree),
        n_terms=int(surrogate.n_terms),
        eps_loo=float(surrogate.eps_loo),
        holdout_rel_rmse=float(holdout),
    )


def _n_terms_total(dim: int, degree: int) -> int:
    return int(math.comb(int(degree) + int(dim), int(dim)))


def refinement_study(
    spec: SampleSpec,
    model: Any,
    *,
    benchmark: str = "",
    degrees: Sequence[int] = (2, 3, 4, 5, 6, 8, 10, 12),
    degree_n_samples: int = 1024,
    degree_seed: int = 20240,
    truncation_degree: int = 4,
    truncation_n_samples: Sequence[int] | None = None,
    adaptive_degree: int = 8,
    adaptive_n_samples: Sequence[int] = (128, 256, 512, 1024, 2048),
    holdout_n: int = 20000,
    holdout_seed: int = 7701,
    threshold: float = DEFAULT_REFINEMENT_THRESHOLD,
    sample_seed: int = 20240,
) -> RefinementStudy:
    """Measure ``eps_LOO`` and an independent holdout error on three axes.

    The axes are chosen so that each answers a well-posed question:

    ``samples_at_fixed_truncation``
        The truncation is frozen and every term is fitted, so the only error
        left to shrink is coefficient estimation.  This is where the statement
        "more samples -> smaller error" is a theorem, and both estimators must
        be monotone.
    ``degree_at_fixed_n``
        The sample budget is frozen and the total degree grows, so the
        truncation error shrinks and eventually the estimation error takes over.
        Both estimators must be monotone up to the sweet spot; the grid stops
        where ``samples_per_term`` would fall below the gate floor.
    ``samples_with_adaptive_selection``
        Sparse LARS selection at a fixed degree: the *surrogate* improves (the
        holdout error is monotone) but ``eps_LOO`` is not monotone because the
        selector's optimism decays with the sample count.  This axis is what
        makes the literal "eps_LOO decreases" claim falsifiable, and it is
        reported as such (``adaptive_loo_nonmonotone``).
    """
    if not isinstance(spec, SampleSpec):
        raise TypeError(f"spec must be a SampleSpec, got {type(spec)!r}")
    fn = _as_model(model)
    holdout_n = int(holdout_n)
    design_h = sample(spec, holdout_n, seed=int(holdout_seed))
    holdout_points, holdout_coords = spec.from_latent(design_h.latent)

    axes: dict[str, RefinementAxis] = {}

    # ---- 1. estimation error at fixed truncation ----------------------------
    if truncation_n_samples is None:
        p_terms = _n_terms_total(spec.dim, truncation_degree)
        truncation_n_samples = tuple(
            int(mult * p_terms) for mult in (2, 4, 8, 16, 32)
        )
    points = tuple(
        _fit_and_score(
            spec,
            fn,
            n_samples=int(n),
            degree=int(truncation_degree),
            method="ols",
            max_terms=None,
            seed=sample_seed,
            holdout_points=holdout_points,
            holdout_coords=holdout_coords,
        )
        for n in truncation_n_samples
    )
    axes["samples_at_fixed_truncation"] = RefinementAxis(
        name="samples_at_fixed_truncation",
        protocol=(
            f"method=ols, total degree {truncation_degree} fixed "
            f"(P={_n_terms_total(spec.dim, truncation_degree)} terms), "
            "N in {2,4,8,16,32}xP: only the estimation error can shrink"
        ),
        points=points,
        threshold=threshold,
    )

    # ---- 2. truncation error at fixed sample budget -------------------------
    kept: list[RefinementPoint] = []
    for degree in degrees:
        p_terms = _n_terms_total(spec.dim, int(degree))
        if degree_n_samples < 4 * min(p_terms, max(20, p_terms // 2)):
            # samples-per-active-term would fall under the gate floor: the LOO
            # estimate is not usable there (see verdict.SAMPLES_PER_TERM)
            continue
        kept.append(
            _fit_and_score(
                spec,
                fn,
                n_samples=int(degree_n_samples),
                degree=int(degree),
                method="lars",
                max_terms=None,
                seed=sample_seed,
                holdout_points=holdout_points,
                holdout_coords=holdout_coords,
            )
        )
    axes["degree_at_fixed_n"] = RefinementAxis(
        name="degree_at_fixed_n",
        protocol=(
            f"method=lars (sparse), N={degree_n_samples} fixed, total degree "
            "swept; grid points that would break samples-per-term >= 4 are dropped"
        ),
        points=tuple(kept),
        threshold=threshold,
        swept="degree",
    )

    # ---- 3. adaptive selection at fixed degree (falsification axis) ---------
    adaptive = tuple(
        _fit_and_score(
            spec,
            fn,
            n_samples=int(n),
            degree=int(adaptive_degree),
            method="lars",
            max_terms=None,
            seed=sample_seed,
            holdout_points=holdout_points,
            holdout_coords=holdout_coords,
        )
        for n in adaptive_n_samples
    )
    axes["samples_with_adaptive_selection"] = RefinementAxis(
        name="samples_with_adaptive_selection",
        protocol=(
            f"method=lars (sparse selection re-run at every N), total degree "
            f"{adaptive_degree} fixed: the active set changes with N, so eps_LOO "
            "carries model-selection bias"
        ),
        points=adaptive,
        threshold=threshold,
        notes=(
            "eps_LOO is compared against an independent holdout error; a "
            "non-monotone eps_LOO with monotone holdout error is the signature "
            "of decaying selection optimism, not of a worse surrogate",
        ),
    )

    return RefinementStudy(
        benchmark=benchmark,
        axes=axes,
        holdout_n=holdout_n,
        holdout_seed=int(holdout_seed),
        threshold=float(threshold),
    )


def adaptive_selection_diagnostic(study: RefinementStudy) -> dict[str, Any]:
    """Compact statement of the ``eps_LOO`` monotonicity falsification.

    Returns the two Spearman coefficients, the optimism ratios and a plain
    sentence that a report can quote, so the finding is not buried in a table.
    """
    axis = study.adaptive_axis
    if axis is None:
        raise ValueError("study has no adaptive-selection axis")
    loo_seq = [float(v) for v in axis.eps_loo]
    hold_seq = [float(v) for v in axis.holdout]
    verdict = (
        "eps_LOO is NOT monotone in the sample count while the holdout error is "
        "monotone: the literal refinement claim only holds at fixed truncation"
        if (not axis.loo_monotone and axis.holdout_monotone)
        else "both metrics are monotone (no selection-optimism effect observed)"
    )
    return {
        "protocol": axis.protocol,
        "budgets": [float(p.n_samples) for p in axis.points],
        "eps_loo": loo_seq,
        "holdout_rel_rmse": hold_seq,
        "spearman_loo": axis.spearman_loo,
        "spearman_holdout": axis.spearman_holdout,
        "loo_optimism": [p.loo_optimism for p in axis.points],
        "max_loo_optimism": axis.max_loo_optimism,
        "loo_calibrated_at_largest_n": bool(axis.points[-1].loo_optimism <= 1.2),
        "statement": verdict,
    }