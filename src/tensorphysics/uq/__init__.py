"""Non-intrusive uncertainty quantification as a first-class acceptance gate.

What this package provides (campaign node IN143-N11)
----------------------------------------------------
* :mod:`tensorphysics.uq.distributions` -- input marginals (uniform / normal /
  lognormal / beta) and an optional Gaussian-copula correlation model;
* :mod:`tensorphysics.uq.sampling` -- deterministic Sobol' / Latin-hypercube /
  Monte-Carlo sampling with a SHA-256 digest per sample set;
* :mod:`tensorphysics.uq.basis` -- orthonormal Legendre / Hermite chaos bases and
  (hyperbolic) multi-index truncation;
* :mod:`tensorphysics.uq.regression` -- sparse OMP regression with closed-form
  leave-one-out error;
* :mod:`tensorphysics.uq.pce` -- the polynomial chaos expansion, its analytical
  first/second/total Sobol' indices and coefficient-covariance confidence
  intervals;
* :mod:`tensorphysics.uq.mc` -- independent Saltelli/Jansen Monte-Carlo indices
  with bootstrap intervals (the cross-check that a surrogate is faithful);
* :mod:`tensorphysics.uq.adapter` -- the solver-adapter contract (non-intrusive,
  evaluation-counting, exception-tolerant);
* :mod:`tensorphysics.uq.gate` -- the PASS / FAIL / WITHHELD acceptance gate;
* :mod:`tensorphysics.uq.study` -- the end-to-end study and the hand-off of
  ``U_input`` into the ASME V&V 20 validation gate.

Quick start
-----------
>>> from tensorphysics.uq import (InputModel, Uniform, UQStudy, UQOptions, CallableAdapter)
>>> import numpy as np
>>> model = InputModel(["x1", "x2"], [Uniform(-1, 1), Uniform(-1, 1)])
>>> adapter = CallableAdapter(lambda X: X[:, 0] + 0.5 * X[:, 1], model.names, name="demo")
>>> study = UQStudy(model, adapter, UQOptions(n_samples=512, degrees=(2, 3, 4, 5),
...                                           cross_check_base=256))
>>> result = study.run()                                    # doctest: +SKIP
>>> result.verdict.status in {"PASS", "FAIL", "WITHHELD"}    # doctest: +SKIP
True
"""

from .adapter import AdapterFailure, CallableAdapter, RecordedAdapter, SolverAdapter, as_adapter
from .basis import HERMITE, LEGENDRE, MultiIndexSet, orthonormal_basis
from .distributions import (
    Beta,
    GaussianCopulaError,
    InputModel,
    LogNormal,
    Marginal,
    Normal,
    Uniform,
    marginal_from_dict,
)
from .gate import FAIL, PASS, REASON_MEANING, WITHHELD, UQDiagnostics, UQPolicy, UQVerdict, evaluate_gate
from .indices import IndexComparison, SobolIndices, compare_indices
from .mc import saltelli_indices
from .pce import PCEFit, PCESweep, fit_pce, fit_pce_sweep, spearman_rank_correlation
from .sampling import SampleSet, draw_unit, sample_input_model, saltelli_design, sobol_unit
from .study import (
    DEFAULT_DEGREES,
    UQOptions,
    UQResult,
    UQStudy,
    attach_to_validation,
    propagated_input_uncertainty,
)

__all__ = [
    # distributions
    "Beta",
    "GaussianCopulaError",
    "InputModel",
    "LogNormal",
    "Marginal",
    "Normal",
    "Uniform",
    "marginal_from_dict",
    # sampling
    "SampleSet",
    "draw_unit",
    "sample_input_model",
    "saltelli_design",
    "sobol_unit",
    # basis / regression / pce
    "HERMITE",
    "LEGENDRE",
    "MultiIndexSet",
    "orthonormal_basis",
    "PCEFit",
    "PCESweep",
    "fit_pce",
    "fit_pce_sweep",
    "spearman_rank_correlation",
    # indices
    "IndexComparison",
    "SobolIndices",
    "compare_indices",
    "saltelli_indices",
    # adapters
    "AdapterFailure",
    "CallableAdapter",
    "RecordedAdapter",
    "SolverAdapter",
    "as_adapter",
    # gate / study
    "FAIL",
    "PASS",
    "WITHHELD",
    "REASON_MEANING",
    "UQDiagnostics",
    "UQPolicy",
    "UQVerdict",
    "evaluate_gate",
    "DEFAULT_DEGREES",
    "UQOptions",
    "UQResult",
    "UQStudy",
    "attach_to_validation",
    "propagated_input_uncertainty",
]