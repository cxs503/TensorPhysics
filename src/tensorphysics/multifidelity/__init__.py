"""Multi-fidelity model selection under an error budget and a compute budget.

The engine answers one question per query: *which fidelity level of this
quantity should I run, given that I need the answer to within ``epsilon`` and I
have ``C`` seconds of compute?*  and it answers it with a probability statement
rather than a bare recommendation.

Layers
------
1. :mod:`~tensorphysics.multifidelity.model` -- the registry: per-model error
   prior (log-normal, median-parameterised), cost prior, validity box on the
   query axes, fidelity tier and provenance.  Data only, no domain vocabulary.
2. :mod:`~tensorphysics.multifidelity.selection` -- the two optimisation
   problems (``min_cost`` s.t. ``P(error<=eps)>=alpha`` and ``P(cost<=C)>=beta``,
   ``min_error`` s.t. the same), the Pareto front, and the refusal reason codes
   (an ``epsilon`` below the accuracy floor of the admissible set is a FAIL, not
   a recommendation).
3. :mod:`~tensorphysics.multifidelity.calibration` -- conformal recalibration of
   the registered priors from validation residuals, pooled or stratified on a
   declared axis partition; the reported bound becomes the conformal order
   statistic, which repairs a mis-registered prior without any distributional
   assumption.
4. :mod:`~tensorphysics.multifidelity.backtest` -- truth generator + query
   sampler + rate bookkeeping: the only way a probabilistic promise can be
   falsified.
5. :mod:`~tensorphysics.multifidelity.verification` -- the end-to-end protocol
   behind ``docs/MULTIFIDELITY_MODEL_SELECTION_P9003W1_N05.md``: four registry
   variants against two truths, 2000 queries, acceptance checks and the physical
   anchor on the KCS hull.

Quick start
-----------
>>> from tensorphysics.multifidelity import Budget, ModelRegistry, select_model
>>> registry = ModelRegistry.from_json(
...     "src/tensorphysics/multifidelity/configs/drag_multifidelity.v1.json")
>>> sel = select_model(registry, {"Fn": 0.22, "Cb": 0.70, "Re": 1.0e9},
...                    Budget(epsilon=0.05, cost=1.0e4))
>>> sel.model_id, sel.uncertainty["prob_error_le_epsilon"]     # doctest: +SKIP
('ittc1978_prediction', 0.99...)

Honest limits
-------------
* the accuracy/cost priors are *inputs*: the engine cannot be better calibrated
  than its validation set, which is exactly why recalibration is part of the
  contract rather than an optional extra;
* the truth generator used for verification is a declared stochastic model of
  "how a prior can be wrong", not a physical resistance model;
* stratification is only as good as the declared partition -- the
  ``conformal_mispartitioned`` variant in the verification quantifies what
  happens when the partition is wrong.
"""
from __future__ import annotations

from .backtest import (AxisSpec, BacktestResult, QueryRecord, RegimeRule, TruthGenerator,
                       TruthSpec, generate_validation_residuals, run_backtest, sample_queries)
from .calibration import StratumSpec, StratifiedEmpiricalError, calibrate_registry, calibrate_registry_stratified
from .model import (ApplicabilityDomain, CostPrior, ModelRegistry, ModelSpec, PriorErrorView,
                    RegistryConfigError)
from .selection import (Budget, Candidate, FAIL, MIN_COST, MIN_ERROR, PASS, REASON_CODES,
                        Selection, evaluate_candidates, frontier_curve, pareto_front, select_model)
from .stats import EmpiricalDistribution, LognormalPrior, coverage_rate, wilson_interval
from .verification import (NODE_ID, VerificationResult, evaluate_acceptance, load_config,
                           physical_anchor, run_verification)

__all__ = [
    "ApplicabilityDomain",
    "AxisSpec",
    "BacktestResult",
    "Budget",
    "Candidate",
    "CostPrior",
    "EmpiricalDistribution",
    "FAIL",
    "LognormalPrior",
    "MIN_COST",
    "MIN_ERROR",
    "ModelRegistry",
    "ModelSpec",
    "NODE_ID",
    "PASS",
    "PriorErrorView",
    "QueryRecord",
    "REASON_CODES",
    "RegimeRule",
    "RegistryConfigError",
    "Selection",
    "StratifiedEmpiricalError",
    "StratumSpec",
    "TruthGenerator",
    "TruthSpec",
    "VerificationResult",
    "calibrate_registry",
    "calibrate_registry_stratified",
    "coverage_rate",
    "evaluate_acceptance",
    "evaluate_candidates",
    "frontier_curve",
    "generate_validation_residuals",
    "load_config",
    "pareto_front",
    "physical_anchor",
    "run_backtest",
    "run_verification",
    "sample_queries",
    "select_model",
    "wilson_interval",
]