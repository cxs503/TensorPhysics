"""Uncertainty quantification as a first-class workflow contract.

This package turns "how much do the uncertain inputs move the answer?" into a
machine-adjudicated artefact that the rest of the Tensor ecosystem can gate on.
It is deliberately **non-intrusive**: it never imports, patches or copies a
solver kernel -- a solver is any black box ``points (n, d) -> values (n,)``.

Pipeline
--------
1. :mod:`~tensorphysics.uq.sample_spec` -- uncertain inputs (uniform / normal /
   lognormal marginals + optional Gaussian copula) and the map from the unit
   hypercube to physical parameters and to the polynomial basis coordinate.
2. :mod:`~tensorphysics.uq.sampler` -- deterministic Scrambled-Sobol' / LHS /
   MC designs with a sha256 **sample-set hash**.
3. :mod:`~tensorphysics.uq.adapter` -- solver fan-out with hash-pinned merging
   and NaN bookkeeping (the hook for cluster shards).
4. :mod:`~tensorphysics.uq.pce` -- non-intrusive polynomial chaos (orthonormal
   Legendre / Hermite, greedy sparse selection on the PRESS/LOO criterion) and
   the *analytic* Sobol' indices of the surrogate.
5. :mod:`~tensorphysics.uq.sobol_mc` -- the independent sampled second opinion
   (Saltelli / Jansen pick-freeze estimators, including second order).
6. :mod:`~tensorphysics.uq.verdict` -- the three-state gate: the 95% output
   interval width ``U`` against an engineering threshold (default 15%), with
   explicit reason codes for ``WITHHELD``.
7. :mod:`~tensorphysics.uq.workflow` -- ``run_uq_study``, which chains 1-6 and
   returns a JSON-serialisable :class:`~tensorphysics.uq.workflow.UQStudy`.

Quick start
-----------
>>> from tensorphysics.uq import run_uq_study, ishigami_spec, ishigami
>>> study = run_uq_study(ishigami, ishigami_spec(), n_samples=1024, degree=8,
...                      run_mc=True, n_base_mc=1 << 11, n_boot=0)
>>> study.status in {"PASS", "FAIL", "WITHHELD"}
True

Notes and limits
----------------
* The chaos route assumes the QoI is smooth in the inputs at the chosen total
  degree; ``eps_loo`` is the machine-checked escape hatch (a poor surrogate
  withholds the verdict rather than producing confident nonsense).
* For correlated inputs the indices are latent-coordinate (Gaussian-copula)
  indices -- the only orthogonal decomposition available; the verdict attaches
  the ``UQ-N01-CORRELATED`` note.  The sampled (Saltelli) route only supports
  independent inputs and raises otherwise.
* ``U`` is a *prediction* interval of the QoI, not the confidence interval of
  its mean (see :mod:`tensorphysics.uq.verdict`).
"""
from .adapter import (
    CallableAdapter,
    SampleHashMismatch,
    SampleResults,
    SolverAdapter,
    cluster_shard_command,
    fanout_evaluate,
)
from .benchmarks import (
    ISHIGAMI_QUOTED,
    HoltropTypeBenchmark,
    holtrop_type_benchmark,
    ishigami,
    ishigami_analytic,
    ishigami_spec,
    sobol_g,
    sobol_g_analytic,
    sobol_g_spec,
)
from .pce import (
    PCESurrogate,
    SobolIndices,
    basis_matrix,
    basis_values,
    bootstrap_sobol,
    fit_pce,
    hermite_orthonormal,
    legendre_orthonormal,
    multi_indices,
)
from .failsafe import (
    DEFAULT_Q2_MIN,
    MODE_MONTECARLO,
    MODE_PCE,
    FailsafeDecision,
    FailsafeStudy,
    assess as assess_failsafe,
    direct_budget,
    indices_from_mc,
    loo_q2,
    run_with_failsafe,
)
from .holton import (
    HoltonBenchmark,
    holton,
    holton_analytic,
    holton_reference_indices,
    holton_spec,
    make_holton,
)
from .solver_outputs import (
    SUPPORTED_SOLVERS,
    BenchmarkArtifact,
    BenchmarkQoI,
    BudgetComponent,
    CombinedBudget,
    UnknownBenchmarkError,
    combine_budget,
    detect_solver,
    gate_with_benchmark,
    parse_artifact,
    parse_benchmarks,
    relative_error as benchmark_relative_error,
    select as select_benchmark_qoi,
)

from .ci_coverage import CoverageEntry, CoverageReport, coverage_study, wilson_interval
from .gate import (
    SCHEMA_VERSION,
    GateDecision,
    PreEvidence,
    UQEvidenceError,
    chain_asme_validation,
    engineering_error_gate,
    pre_evidence,
    require_pass,
)
from .refinement import (
    RefinementAxis,
    RefinementPoint,
    RefinementStudy,
    adaptive_selection_diagnostic,
    refinement_study,
)
from .robustness import InjectionCase, InjectionReport, injection_suite
from .sample_spec import Marginal, SampleSpec, latent_from_unit
from .sampler import RULES, SampleSet, compute_sample_hash, draw_unit, sample
from .sobol_mc import SobolMCResult, saltelli_design, saltelli_indices
from .verdict import (
    DEFAULT_EPS_LOO_MAX,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_THRESHOLD,
    FAIL,
    PASS,
    WITHHELD,
    CrossCheck,
    CrossCheckEntry,
    OutputBudget,
    Reason,
    UQVerdict,
    cross_check_indices,
    decide,
    min_samples_floor,
    output_budget,
)
from .workflow import UQStudy, run_uq_study

__all__ = [
    "CallableAdapter",
    "CrossCheck",
    "CrossCheckEntry",
    "DEFAULT_EPS_LOO_MAX",
    "DEFAULT_MIN_SAMPLES",
    "DEFAULT_THRESHOLD",
    "FAIL",
    "HoltropTypeBenchmark",
    "ISHIGAMI_QUOTED",
    "Marginal",
    "OutputBudget",
    "PASS",
    "PCESurrogate",
    "RULES",
    "Reason",
    "SampleHashMismatch",
    "SampleResults",
    "SampleSet",
    "SampleSpec",
    "SobolIndices",
    "SobolMCResult",
    "SolverAdapter",
    "UQStudy",
    "UQVerdict",
    "WITHHELD",
    "basis_matrix",
    "basis_values",
    "bootstrap_sobol",
    "cluster_shard_command",
    "compute_sample_hash",
    "cross_check_indices",
    "decide",
    "draw_unit",
    "fanout_evaluate",
    "fit_pce",
    "hermite_orthonormal",
    "holtrop_type_benchmark",
    "ishigami",
    "ishigami_analytic",
    "ishigami_spec",
    "latent_from_unit",
    "legendre_orthonormal",
    "min_samples_floor",
    "multi_indices",
    "output_budget",
    "run_uq_study",
    "saltelli_design",
    "saltelli_indices",
    "sample",
    "sobol_g",
    "sobol_g_analytic",
    "sobol_g_spec",
]

# ---- P142W3-N10 additions: Holton anchor, fail-safe, cross-repo adapters, CLI
__all__ += [
    "BenchmarkArtifact",
    "BenchmarkQoI",
    "BudgetComponent",
    "CombinedBudget",
    "DEFAULT_Q2_MIN",
    "FailsafeDecision",
    "FailsafeStudy",
    "HoltonBenchmark",
    "MODE_MONTECARLO",
    "MODE_PCE",
    "SUPPORTED_SOLVERS",
    "UnknownBenchmarkError",
    "assess_failsafe",
    "benchmark_relative_error",
    "combine_budget",
    "detect_solver",
    "direct_budget",
    "gate_with_benchmark",
    "holton",
    "holton_analytic",
    "holton_reference_indices",
    "holton_spec",
    "indices_from_mc",
    "loo_q2",
    "make_holton",
    "parse_artifact",
    "parse_benchmarks",
    "run_with_failsafe",
    "select_benchmark_qoi",
]

# ---- IN145-N08 additions: refinement, interval calibration, robustness, gate
__all__ += [
    "CoverageEntry",
    "CoverageReport",
    "GateDecision",
    "InjectionCase",
    "InjectionReport",
    "PreEvidence",
    "RefinementAxis",
    "RefinementPoint",
    "RefinementStudy",
    "SCHEMA_VERSION",
    "UQEvidenceError",
    "adaptive_selection_diagnostic",
    "chain_asme_validation",
    "coverage_study",
    "engineering_error_gate",
    "injection_suite",
    "pre_evidence",
    "refinement_study",
    "require_pass",
    "wilson_interval",
]
