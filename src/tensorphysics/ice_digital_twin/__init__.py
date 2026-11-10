"""Ice-going ship digital twin: model validation and uncertainty quantification.

Public surface
--------------
``twin``          parameterised, vectorised front end of the Lindqvist (1989)
                  level-ice law (anchored to ``ice.resistance``) plus the
                  structural ensemble :data:`twin.FORMS`.
``bayes``         Gaussian likelihood with a multiplicative model-inadequacy
                  term, adaptive-Metropolis MCMC, split-R / ESS diagnostics,
                  posterior-predictive intervals.
``sensitivity``   Saltelli/Jansen Sobol' indices of the twin response (reusing
                  ``uq.sobol_mc``) plus the additive-submodel closure check.
``model_form``    pseudo-BMA weights from LOO predictive scores, model-averaged
                  predictive intervals, between/within variance split.
``online``        resample-move sequential calibration with a per-step time
                  budget.
``validation``    leave-one-out coverage validation and the campaign driver that
                  emits the auditable JSON report.

Quick start
-----------
>>> from tensorphysics.ice_digital_twin import run_digital_twin_uq
>>> report = run_digital_twin_uq(loo_steps=800, n_calib_steps=3000, sobol_n_base=1 << 10)
>>> sorted(report["criteria"])[:3]
['a_max_relative_error', 'a_posterior_mean_vs_literature_under_20pct', 'b_max_rhat_split']

Evidence grading: forward law REAL, UQ machinery REAL, campaign data SYNTHETIC
(real full-scale validation WITHHELD) -- see the ``twin`` module docstring.
"""
from __future__ import annotations

from .bayes import (
    CalibrationData,
    ChainSet,
    coverage_rate,
    importance_loo,
    log_likelihood,
    log_likelihood_features,
    model_features,
    model_from_features,
    predictive_matrix,
    log_posterior_factory,
    parameter_summary,
    posterior_predictive,
    predictive_draws,
    predictive_interval,
    run_am_chains,
    split_rhat,
    synthetic_campaign,
)
from .model_form import bma_mixture, bma_report, pseudo_bma_weights
from .online import OnlineCalibrator, run_online_calibration
from .sensitivity import (
    additive_submodel_exact,
    additive_submodel_sobol,
    closure_metrics,
    reference_condition,
    sensitivity_report,
    twin_sobol,
)
from .twin import (
    DEFAULT_FORM,
    FORMS,
    KAPPA_BOUNDS,
    ModelForm,
    PARAM_BOUNDS,
    PARAM_NAMES,
    PARAM_REFERENCE,
    component_breakdown_kN,
    get_form,
    parameter_reference_table,
    physical_to_unit,
    reference_law_kN,
    resistance_components,
    resistance_kN,
    resistance_kN_batch,
    unit_to_physical,
)
from .validation import (
    bma_cross_validation,
    calibrate,
    fixed_parameter_baseline,
    loo_form_refit,
    run_digital_twin_uq,
    write_report,
)

__all__ = [
    "CalibrationData",
    "ChainSet",
    "ModelForm",
    "FORMS",
    "DEFAULT_FORM",
    "PARAM_NAMES",
    "PARAM_BOUNDS",
    "PARAM_REFERENCE",
    "KAPPA_BOUNDS",
    "OnlineCalibrator",
    "additive_submodel_exact",
    "additive_submodel_sobol",
    "bma_cross_validation",
    "bma_mixture",
    "bma_report",
    "calibrate",
    "closure_metrics",
    "component_breakdown_kN",
    "coverage_rate",
    "fixed_parameter_baseline",
    "get_form",
    "log_likelihood",
    "log_likelihood_features",
    "importance_loo",
    "model_features",
    "model_from_features",
    "predictive_matrix",
    "log_posterior_factory",
    "loo_form_refit",
    "parameter_reference_table",
    "parameter_summary",
    "physical_to_unit",
    "posterior_predictive",
    "predictive_draws",
    "predictive_interval",
    "pseudo_bma_weights",
    "reference_condition",
    "reference_law_kN",
    "resistance_components",
    "resistance_kN",
    "resistance_kN_batch",
    "run_am_chains",
    "run_digital_twin_uq",
    "run_online_calibration",
    "sensitivity_report",
    "split_rhat",
    "synthetic_campaign",
    "twin_sobol",
    "unit_to_physical",
    "write_report",
]