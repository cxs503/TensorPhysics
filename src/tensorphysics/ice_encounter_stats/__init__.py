"""P233W2-N05 -- statistical pre-conditions for DEM ice-resistance convergence.

The plan-233 ice-resistance studies all report an *average* resistance.  This
module supplies the missing decision layer: how long a DEM/statistical simulation
must run before that average is meaningful, and on which two statistical
pre-conditions the answer rests.

Public surface
--------------
* :mod:`~tensorphysics.ice_encounter_stats.config`     -- injected case /
  manifest configuration (no domain constant lives in code).
* :mod:`~tensorphysics.ice_encounter_stats.field`      -- Poisson and
  Neyman--Scott encounter point processes on the swept corridor.
* :mod:`~tensorphysics.ice_encounter_stats.signal`     -- Poisson shot-noise
  synthesis of the resistance record with its injected marks.
* :mod:`~tensorphysics.ice_encounter_stats.events`     -- threshold extraction of
  encounter events, inter-arrival intervals and single-event pulses.
* :mod:`~tensorphysics.ice_encounter_stats.statistics` -- KS test against the
  exponential, integral correlation time, Campbell's theorem.
* :mod:`~tensorphysics.ice_encounter_stats.stopping`   -- closed form
  ``T_req = 2 tau_c (CV/eps)^2`` and the reusable stopping rule.
* :mod:`~tensorphysics.ice_encounter_stats.prereg`     -- pre-registration
  manifest and the machine judge of its criteria.
* :mod:`~tensorphysics.ice_encounter_stats.acceptance` -- AC1..AC6 driver.
* :mod:`~tensorphysics.ice_encounter_stats.report`     -- figures + paper-style
  markdown report.

Quick start
-----------
>>> from tensorphysics.ice_encounter_stats import load_manifest, run_acceptance
>>> manifest = load_manifest()
>>> results = run_acceptance(manifest, outdir="artifacts/p233w2_n05")  # doctest: +SKIP
"""

from .acceptance import AcceptanceResult, run_acceptance
from .config import (CaseSpec, ClusterSpec, ConfigError, ForceModelSpec,
                     Manifest, PulseSpec, ScalarSpectrum, load_manifest,
                     manifest_hash)
from .events import EncounterEvents, auto_level, extract_events
from .field import (encounter_times, interval_dispersion,
                    neyman_scott_encounter_times, poisson_encounter_times)
from .prereg import build_criteria, summarize, validate
from .signal import Record, population_marks, simulate_record, window_mean
from .statistics import (CampbellPrediction, KSResult, autocorrelation,
                         bonferroni_adjust, campbell_moments, correlation_time,
                         effective_sample_size, integral_time_scale,
                         ks_exponential, relative_error,
                         standard_error_of_mean)
from .stopping import (PilotStatistics, StopDecision, StoppingRule, normal_ppf,
                       rule_from_execution, tail_factor_from_quantile)

NODE_ID = "P233W2-N05"

__all__ = [
    "NODE_ID",
    "AcceptanceResult",
    "CampbellPrediction",
    "CaseSpec",
    "ClusterSpec",
    "ConfigError",
    "EncounterEvents",
    "ForceModelSpec",
    "KSResult",
    "Manifest",
    "PilotStatistics",
    "PulseSpec",
    "Record",
    "ScalarSpectrum",
    "StopDecision",
    "StoppingRule",
    "auto_level",
    "autocorrelation",
    "bonferroni_adjust",
    "build_criteria",
    "campbell_moments",
    "correlation_time",
    "effective_sample_size",
    "encounter_times",
    "extract_events",
    "integral_time_scale",
    "interval_dispersion",
    "ks_exponential",
    "load_manifest",
    "manifest_hash",
    "neyman_scott_encounter_times",
    "poisson_encounter_times",
    "population_marks",
    "relative_error",
    "run_acceptance",
    "simulate_record",
    "standard_error_of_mean",
    "rule_from_execution",
    "tail_factor_from_quantile",
    "normal_ppf",
    "summarize",
    "validate",
    "window_mean",
]