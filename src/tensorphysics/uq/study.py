"""End-to-end non-intrusive UQ study, and its hand-off to the V&V validation gate.

The chain implemented here is the contract this node delivers:

    input distributions  ->  low-discrepancy sampling  ->  solver adapter
                         ->  sparse PCE + eps_LOO        ->  S1/ST/S2 + 95% CI
                         ->  independent Saltelli cross-check
                         ->  PASS / FAIL / WITHHELD  ->  (V&V 20 gate)

The last arrow is why the gate exists.  :mod:`tensorphysics.vv.validation` accepts
a model prediction when ``|E| <= U_val`` with
``U_val = sqrt(U_num^2 + U_input^2)``.  ``U_input`` -- the standard uncertainty of
the simulated quantity attributable to the uncertain inputs -- is exactly what a
UQ study produces, and it is meaningless if the surrogate behind it is not
certified.  :func:`attach_to_validation` therefore refuses to emit a validation
bundle unless the UQ verdict is ``PASS``; a ``WITHHELD`` UQ verdict makes the
engineering error gate ``WITHHELD`` too, which is the whole point of putting UQ
*before* the error gate instead of next to it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from .adapter import SolverAdapter, as_adapter
from .distributions import InputModel
from .gate import (
    WITHHELD,
    UQDiagnostics,
    UQPolicy,
    UQVerdict,
    evaluate_gate,
)
from .indices import SobolIndices, compare_indices
from .mc import saltelli_indices
from .pce import PCEFit, PCESweep, fit_pce, fit_pce_sweep, spearman_rank_correlation
from .sampling import SampleSet, next_power_of_two, sample_input_model

__all__ = [
    "UQOptions",
    "UQResult",
    "UQStudy",
    "attach_to_validation",
    "propagated_input_uncertainty",
    "DEFAULT_DEGREES",
]

DEFAULT_DEGREES = (2, 3, 4, 5, 6, 8, 10, 12)


@dataclass(frozen=True)
class UQOptions:
    """Everything that controls one UQ study (frozen: a study is reproducible)."""

    method: str = "sobol"
    seed: int = 0
    n_samples: int | None = None
    max_samples: int = 8192
    degrees: Sequence[int] = DEFAULT_DEGREES
    q: float = 1.0
    interaction: int | None = None
    min_oversampling: float = 2.0
    max_terms: int | None = None
    complexity_penalty: float = 0.0
    ci_level: float = 0.95
    n_draws_ci: int = 2000
    second_order: bool = True
    cross_check: bool = True
    cross_check_base: int = 1024
    cross_check_boot: int = 400
    policy: UQPolicy = UQPolicy()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "seed": int(self.seed),
            "n_samples": None if self.n_samples is None else int(self.n_samples),
            "max_samples": int(self.max_samples),
            "degrees": [int(d) for d in self.degrees],
            "q": float(self.q),
            "interaction": None if self.interaction is None else int(self.interaction),
            "min_oversampling": float(self.min_oversampling),
            "max_terms": None if self.max_terms is None else int(self.max_terms),
            "ci_level": float(self.ci_level),
            "n_draws_ci": int(self.n_draws_ci),
            "cross_check": bool(self.cross_check),
            "cross_check_base": int(self.cross_check_base),
            "cross_check_boot": int(self.cross_check_boot),
            "policy": self.policy.as_dict(),
        }


@dataclass
class UQResult:
    """The auditable output of one UQ study."""

    verdict: UQVerdict
    model: InputModel
    adapter: SolverAdapter
    options: UQOptions
    samples: SampleSet | None = None
    sweep: PCESweep | None = None
    indices: SobolIndices | None = None
    cross_check: SobolIndices | None = None
    loo_curve: List[Dict[str, Any]] = field(default_factory=list)
    loo_rank_correlation: float | None = None
    output_mean: float | None = None
    output_std: float | None = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    # -- headline numbers -------------------------------------------------
    @property
    def passed(self) -> bool:
        return self.verdict.passed

    @property
    def fit(self) -> PCEFit | None:
        return self.sweep.best if self.sweep is not None else None

    def most_influential(self, n: int = 3) -> List[tuple[str, float]]:
        if self.indices is None:
            return []
        return self.indices.rank_by_total()[:n]

    # -- serialisation ----------------------------------------------------
    def as_dict(self) -> Dict[str, Any]:
        fit = self.fit
        out: Dict[str, Any] = {
            "verdict": self.verdict.as_dict(),
            "input_model": self.model.as_dict(),
            "adapter": self.adapter.as_dict(),
            "options": self.options.as_dict(),
            "provenance": dict(self.provenance),
            "output_mean": None if self.output_mean is None else float(self.output_mean),
            "output_std": None if self.output_std is None else float(self.output_std),
            "loo_curve": list(self.loo_curve),
            "loo_rank_correlation": None if self.loo_rank_correlation is None else float(self.loo_rank_correlation),
        }
        if self.samples is not None:
            out["samples"] = self.samples.as_dict()
        if fit is not None:
            out["pce"] = fit.as_dict()
        if self.sweep is not None:
            out["pce_sweep"] = self.sweep.as_dict()
        if self.indices is not None:
            out["sobol"] = self.indices.as_dict()
        if self.cross_check is not None:
            out["sobol_cross_check"] = self.cross_check.as_dict()
        return out

    def summary_table(self) -> str:
        lines = [f"UQ gate: {self.verdict.status}"]
        if self.verdict.messages:
            lines.extend(f"  - {m}" for m in self.verdict.messages)
        if self.indices is not None:
            lines.append(self.indices.table())
        if self.cross_check is not None:
            lines.append("cross-check (Saltelli/Jansen):")
            lines.append(self.cross_check.table())
        if self.loo_curve:
            curve = ", ".join(f"p={r['degree']}:{r['loo_error']:.2e}" for r in self.loo_curve)
            lines.append(f"eps_LOO curve: {curve}")
        return "\n".join(lines)


def _auto_sample_count(options: UQOptions, dim: int, policy: UQPolicy) -> int:
    """Power-of-two sample count covering the largest admissible degree."""
    from .basis import MultiIndexSet

    target = policy.min_samples
    for degree in sorted(int(d) for d in options.degrees):
        size = MultiIndexSet.build(dim, degree, q=options.q, interaction=options.interaction).size
        need = int(np.ceil(options.min_oversampling * size))
        target = max(target, need)
    return next_power_of_two(min(target, options.max_samples))


class UQStudy:
    """Run the full UQ chain for one input model and one solver adapter."""

    def __init__(self, model: InputModel, adapter, options: UQOptions | None = None) -> None:
        self.model = model
        self.adapter = as_adapter(adapter, list(model.names))
        self.options = options or UQOptions()

    # -- helpers ----------------------------------------------------------
    def _empty_result(self, verdict: UQVerdict, **kwargs) -> UQResult:
        return UQResult(
            verdict=verdict,
            model=self.model,
            adapter=self.adapter,
            options=self.options,
            provenance=self._provenance(),
            **kwargs,
        )

    def _provenance(self) -> Dict[str, Any]:
        import platform

        import scipy

        return {
            "package": "tensorphysics.uq",
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "python": platform.python_version(),
            "family": self.options.method,
            "input_model_names": list(self.model.names),
            "n_parameters": int(self.model.dim),
        }

    def _fit(self, scores: np.ndarray, y: np.ndarray) -> PCESweep:
        opts = self.options
        return fit_pce_sweep(
            scores,
            y,
            names=list(self.model.names),
            family="legendre",
            degrees=opts.degrees,
            q=opts.q,
            interaction=opts.interaction,
            max_terms=opts.max_terms,
            min_oversampling=opts.min_oversampling,
            complexity_penalty=opts.complexity_penalty,
        )

    # -- main -------------------------------------------------------------
    def run(self) -> UQResult:
        opts = self.options
        policy = opts.policy
        dim = self.model.dim
        n = int(opts.n_samples) if opts.n_samples is not None else _auto_sample_count(opts, dim, policy)

        samples = sample_input_model(self.model, n, method=opts.method, seed=opts.seed)
        y = np.asarray(self.adapter.evaluate(samples.physical), dtype=float).ravel()
        y = y.reshape(n) if y.size == n else y
        n_nonfinite_y = int(np.sum(~np.isfinite(y)))
        n_nonfinite_x = int(np.sum(~np.isfinite(samples.physical)))

        output_mean = float(np.nanmean(y)) if n_nonfinite_y < n else float("nan")
        output_std = float(np.nanstd(y, ddof=1)) if n_nonfinite_y < n - 1 else float("nan")
        base_extra = {
            "sample_count": n,
            "n_nonfinite_output": n_nonfinite_y,
            "n_nonfinite_input": n_nonfinite_x,
            "sampling_method": opts.method,
            "seed": int(opts.seed),
        }

        if n_nonfinite_y > 0 or n_nonfinite_x > 0 or not self.model.pce_compatible:
            diag = UQDiagnostics(
                n_samples=n,
                dim=dim,
                basis_size=1,
                n_terms=0,
                degree=0,
                loo_error=float("nan"),
                output_variance=output_std**2 if np.isfinite(output_std) else float("nan"),
                indices=SobolIndices(names=list(self.model.names), S1=np.zeros(dim), ST=np.zeros(dim), variance=float("nan"), estimator="none"),
                nonfinite_output_count=n_nonfinite_y,
                nonfinite_input_count=n_nonfinite_x,
                adapter_failures=self.adapter.stats.n_failures,
                correlated_inputs=not self.model.pce_compatible,
                extra=base_extra,
            )
            verdict = evaluate_gate(diag, policy)
            return self._empty_result(verdict, samples=samples, output_mean=output_mean, output_std=output_std)

        family = "legendre"
        scores = self.model.standard_scores(samples.unit, family)
        try:
            sweep = self._fit(scores, y)
        except ValueError as exc:
            # the sweep could not fit a single admissible degree inside the
            # sample budget: that is a WITHHELD study, never an exception that a
            # caller could mistake for "no result yet"
            from .basis import MultiIndexSet

            smallest = min(int(d) for d in opts.degrees)
            size = MultiIndexSet.build(dim, smallest, q=opts.q, interaction=opts.interaction).size
            diag = UQDiagnostics(
                n_samples=n,
                dim=dim,
                basis_size=size,
                n_terms=0,
                degree=smallest,
                loo_error=float("nan"),
                output_variance=float(np.var(y, ddof=1)),
                indices=SobolIndices(
                    names=list(self.model.names), S1=np.zeros(dim), ST=np.zeros(dim), variance=float("nan"), estimator="none"
                ),
                nonfinite_output_count=n_nonfinite_y,
                nonfinite_input_count=n_nonfinite_x,
                adapter_failures=self.adapter.stats.n_failures,
                correlated_inputs=not self.model.pce_compatible,
                extra={**base_extra, "fit_error": str(exc)},
            )
            verdict = evaluate_gate(diag, policy)
            return self._empty_result(
                verdict, samples=samples, output_mean=output_mean, output_std=output_std
            )
        best = sweep.best
        indices = best.sobol(
            ci_level=opts.ci_level,
            n_draws=opts.n_draws_ci,
            seed=opts.seed,
            second_order=opts.second_order,
        )

        cross_check = None
        comparison = None
        if opts.cross_check:
            cross_check, design = saltelli_indices(
                self.model,
                lambda X: self.adapter.evaluate(X),
                n_base=int(opts.cross_check_base),
                method=opts.method,
                seed=int(opts.seed) + 7919,
                n_boot=int(opts.cross_check_boot),
                ci_level=opts.ci_level,
            )
            comparison = compare_indices(indices, cross_check, abs_floor=policy.cross_check_abs_floor)
            base_extra["saltelli_design_digest_sha256"] = design["digest"]
            base_extra["saltelli_evaluations"] = int(design["evaluations"])

        loo_curve = [
            {
                "degree": int(f.degree),
                "basis_size": int(f.basis_size),
                "n_terms": int(f.n_terms),
                "loo_error": float(f.loo_error),
                "r2": float(f.r2),
            }
            for f in sweep.records
        ]
        loo_rho = None
        if len(loo_curve) >= 3:
            loo_rho = spearman_rank_correlation(
                [r["degree"] for r in loo_curve], [r["loo_error"] for r in loo_curve]
            )

        diag = UQDiagnostics(
            n_samples=n,
            dim=dim,
            basis_size=best.basis_size,
            n_terms=best.n_terms,
            degree=best.degree,
            loo_error=float(best.loo_error),
            output_variance=float(np.var(y, ddof=1)),
            indices=indices,
            cross_check=cross_check,
            comparison=comparison,
            nonfinite_output_count=n_nonfinite_y,
            nonfinite_input_count=n_nonfinite_x,
            adapter_failures=self.adapter.stats.n_failures,
            correlated_inputs=not self.model.pce_compatible,
            extra={
                **base_extra,
                "sample_digest_sha256": samples.digest(),
                "physical_digest_sha256": samples.physical_digest(),
                "loo_rank_correlation": loo_rho,
            },
        )
        verdict = evaluate_gate(diag, policy)
        return self._empty_result(
            verdict,
            samples=samples,
            sweep=sweep,
            indices=indices,
            cross_check=cross_check,
            loo_curve=loo_curve,
            loo_rank_correlation=loo_rho,
            output_mean=output_mean,
            output_std=output_std,
        )


# ---------------------------------------------------------------------------
# hand-off to the ASME V&V 20 validation gate
# ---------------------------------------------------------------------------


def propagated_input_uncertainty(result: UQResult) -> float:
    """Standard uncertainty of the simulated quantity from the uncertain inputs.

    This is the standard deviation of the certified PCE surrogate, i.e. the part
    of the output scatter that the declared input distributions explain.  It is
    the ``U_input`` term of ASME V&V 20 -- **only** when the UQ verdict is
    ``PASS``; :func:`attach_to_validation` enforces that.
    """
    if result.indices is None:
        raise ValueError("UQ result carries no PCE indices; input uncertainty is undefined")
    return float(np.sqrt(max(result.indices.variance, 0.0)))


def attach_to_validation(
    result: UQResult,
    *,
    f_sim: float,
    f_exp: float,
    u_num: float,
    label: str = "",
) -> Dict[str, Any]:
    """Feed the UQ evidence into :func:`tensorphysics.vv.validation.validate`.

    The validation verdict is emitted only when the UQ gate says ``PASS``;
    otherwise the returned bundle is ``WITHHELD`` and says which UQ reason
    blocked it.  This is the "UQ before the error gate" rule expressed in code.
    """
    from tensorphysics.vv.validation import validation_uncertainty

    bundle: Dict[str, Any] = {
        "label": label,
        "uq_verdict": result.verdict.status,
        "uq_reasons": list(result.verdict.reasons),
        "f_sim": float(f_sim),
        "f_exp": float(f_exp),
        "u_num": float(u_num),
    }
    if not result.passed:
        bundle["status"] = WITHHELD
        bundle["message"] = (
            "UQ evidence not certified, so U_input is unknown and the validation "
            f"uncertainty U_val cannot be formed (UQ reasons: {result.verdict.reasons})"
        )
        return bundle
    u_input = propagated_input_uncertainty(result)
    u_val = validation_uncertainty(u_num, u_input)
    from tensorphysics.vv.validation import validate

    vr = validate(f_sim=f_sim, f_exp=f_exp, u_num=u_num, u_input=u_input, label=label)
    bundle.update(
        {
            "status": vr.status,
            "u_input": float(u_input),
            "u_val": float(u_val),
            "error": float(vr.error),
            "ratio": float(vr.ratio),
            "message": "U_val = sqrt(U_num^2 + U_input^2) with U_input from the certified PCE",
        }
    )
    return bundle