"""Tests for the P233W2-N05 ice-encounter statistics node.

Coverage:
* injected configuration (no hidden domain constants, missing keys rejected),
* Poisson / Neyman--Scott encounter processes,
* Poisson shot-noise synthesis against Campbell's theorem,
* threshold event extraction (exact interval recovery, dead-time merging),
* KS test against the exponential (power on a clustered sample), correlation time,
* the closed-form / calibrated stopping rule,
* the pre-registration validator (missing, unregistered, tampered),
* an end-to-end (small) acceptance run and a regression on the shipped artifacts.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from tensorphysics.ice_encounter_stats import acceptance as acc
from tensorphysics.ice_encounter_stats.config import (CaseSpec, ConfigError,
                                                      ScalarSpectrum, deep_merge,
                                                      load_manifest, manifest_hash)
from tensorphysics.ice_encounter_stats.events import auto_level, extract_events
from tensorphysics.ice_encounter_stats.field import (encounter_times,
                                                     neyman_scott_encounter_times,
                                                     poisson_encounter_times)
from tensorphysics.ice_encounter_stats.prereg import validate
from tensorphysics.ice_encounter_stats.signal import population_marks, simulate_record
from tensorphysics.ice_encounter_stats.statistics import (bonferroni_adjust,
                                                          campbell_moments,
                                                          correlation_time,
                                                          effective_sample_size,
                                                          ks_exponential,
                                                          relative_error)
from tensorphysics.ice_encounter_stats.stopping import (PilotStatistics,
                                                        StoppingRule, normal_ppf,
                                                        tail_factor_from_quantile)

ARTIFACTS = pathlib.Path(__file__).resolve().parent.parent / "artifacts" / "p233w2_n05"


@pytest.fixture(scope="module")
def manifest():
    return load_manifest()


@pytest.fixture(scope="module")
def base_case(manifest):
    return manifest.base_case()


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
def test_manifest_is_injected_and_complete(manifest):
    assert manifest.node == "P233W2-N05"
    ids = [entry["id"] for entry in manifest.criteria]
    assert {"AC1", "AC2", "AC3", "AC4", "AC5", "AC6"}.issubset(set(ids))
    for entry in manifest.criteria:
        assert entry["op"] in {">=", "<=", ">", "<", "=="}
    assert manifest_hash(manifest.payload) == manifest_hash(manifest.payload)


def test_case_requires_every_physical_key():
    with pytest.raises(ConfigError):
        CaseSpec.from_dict({"name": "broken", "beam_m": 20.0})


def test_override_merges_only_the_named_keys(manifest):
    dense = manifest.case("dense")
    base = manifest.base_case()
    assert dense.encounter_rate > base.encounter_rate
    assert dense.floe_size.params == base.floe_size.params


def test_scalar_spectrum_sampling_is_in_range():
    rng = np.random.default_rng(7)
    spec = ScalarSpectrum("loguniform", {"min": 0.5, "max": 2.0})
    sample = spec.sample(rng, 5000)
    assert sample.min() >= 0.5 and sample.max() <= 2.0
    assert 0.9 < float(np.median(sample)) < 1.15


def test_deep_merge_is_recursive():
    merged = deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}})
    assert merged == {"a": {"b": 1, "c": 3}}


# --------------------------------------------------------------------------- #
# encounter processes
# --------------------------------------------------------------------------- #
def test_poisson_encounter_rate_matches_lambda():
    rng = np.random.default_rng(11)
    rate, duration = 0.7, 20000.0
    times = poisson_encounter_times(rng, rate, duration)
    assert abs(times.size / duration - rate) < 0.05 * rate
    assert np.all(np.diff(times) > 0.0)


def test_clustered_encounter_process_is_over_dispersed():
    rng = np.random.default_rng(13)
    poisson = np.diff(poisson_encounter_times(rng, 0.6, 20000.0))
    clustered = np.diff(neyman_scott_encounter_times(rng, 0.1, 6.0, 1.5, 20000.0))
    cv_poisson = poisson.std(ddof=1) / poisson.mean()
    cv_clustered = clustered.std(ddof=1) / clustered.mean()
    assert 0.9 < cv_poisson < 1.1
    assert cv_clustered > 1.2
    assert ks_exponential(clustered).p_value < 1e-6


def test_encounter_times_uses_the_cluster_driver(manifest):
    base = manifest.base_case()
    clustered = manifest.case("clustered")
    assert clustered.driver_rate < base.encounter_rate
    rng = np.random.default_rng(17)
    times = encounter_times(rng, clustered, 5000.0)
    assert abs(times.size / 5000.0 / clustered.encounter_rate - 1.0) < 0.15


# --------------------------------------------------------------------------- #
# shot-noise synthesis and Campbell's theorem
# --------------------------------------------------------------------------- #
def test_pulse_integrals_are_the_triangle_values(manifest):
    pulse = manifest.base_case().pulse
    a1, b1 = pulse.integrals(4096)
    assert a1 == pytest.approx(0.5, rel=1e-3)
    assert b1 == pytest.approx(1.0 / 3.0, rel=1e-3)


def test_shot_noise_moments_match_campbell(base_case):
    record = simulate_record(base_case, 6000.0, 9001, samples_per_pulse=40)
    areas, sq = population_marks(base_case, 200_000, seed=31)
    camp = campbell_moments(base_case.encounter_rate, areas, sq)
    assert relative_error(record.mean, camp.mean) < 0.05
    assert relative_error(record.var, camp.variance) < 0.15
    assert relative_error(record.cv, camp.cv) < 0.10


def test_simulation_is_deterministic_for_a_seed(base_case):
    a = simulate_record(base_case, 300.0, 5, 20)
    b = simulate_record(base_case, 300.0, 5, 20)
    assert np.array_equal(a.force, b.force)


# --------------------------------------------------------------------------- #
# event extraction
# --------------------------------------------------------------------------- #
def test_extract_events_recovers_known_intervals():
    dt = 0.005
    t = np.arange(0.0, 30.0, dt)
    starts = np.array([1.0, 3.5, 4.25, 9.0, 12.5, 19.0, 25.5])
    force = np.zeros_like(t)
    for s in starts:
        m = int(round(0.4 / dt))
        tau = (np.arange(m) + 0.5) / m
        pulse = np.where(tau < 0.5, tau / 0.5, (1.0 - tau) / 0.5)
        i0 = int(round(s / dt))
        force[i0:i0 + m] += 3.0 * pulse
    events = extract_events(t, force, level=0.5)
    assert events.n_events == starts.size
    # level crossing lags the pulse onset by the rise time of the threshold
    # crossing (all pulses share the shape, so the lag cancels in the intervals)
    assert np.allclose(events.start_times, starts + 0.035, atol=2 * dt)
    assert np.allclose(events.intervals, np.diff(starts), atol=2 * dt)
    assert np.allclose(events.peaks, 3.0, rtol=0.02)  # peak lies between samples


def test_dead_time_merges_split_contacts():
    dt = 0.01
    t = np.arange(0.0, 5.0, dt)
    force = np.zeros_like(t)
    for s in (1.00, 1.10, 1.20):  # one contact that dips through the level
        m = int(round(0.06 / dt))
        i0 = int(round(s / dt))
        force[i0:i0 + m] = np.maximum(force[i0:i0 + m], 2.0)
    split = extract_events(t, force, level=0.5, min_gap_s=0.0)
    merged = extract_events(t, force, level=0.5, min_gap_s=0.2)
    assert split.n_events == 3
    assert merged.n_events == 1


def test_auto_level_scales_with_the_signal():
    force = np.concatenate([np.zeros(500), np.linspace(1.0, 10.0, 500)])
    assert auto_level(force, 0.9, 0.5) == pytest.approx(0.5 * np.quantile(force, 0.9))
    assert auto_level(force, 0.9, 1.0) == pytest.approx(np.quantile(force, 0.9))
    assert auto_level(force, 0.9, 1.0) > auto_level(force, 0.9, 0.5)


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def test_ks_accepts_exponential_and_rejects_clustered():
    rng = np.random.default_rng(23)
    exponential = rng.exponential(scale=2.0, size=800)
    assert ks_exponential(exponential).p_value > 0.05
    gamma = rng.gamma(shape=0.5, scale=2.0, size=800)  # under-dispersed
    assert ks_exponential(gamma).p_value < 1e-3


def test_bonferroni_caps_at_one():
    adjusted, family = bonferroni_adjust([0.4, 0.7, 0.9])
    assert adjusted == 1.0 and family == 3
    adjusted, _ = bonferroni_adjust([0.001, 0.2])
    assert adjusted == pytest.approx(0.002)


def test_correlation_time_of_an_ar1_process():
    rng = np.random.default_rng(29)
    dt = 0.01
    phi = 0.98
    n = 200000
    x = np.zeros(n)
    noise = rng.standard_normal(n)
    for i in range(1, n):  # AR(1): rho(k) = phi^k, integral scale = dt*phi/(1-phi)
        x[i] = phi * x[i - 1] + noise[i]
    _, rho, tau_c, _ = correlation_time(np.arange(n) * dt, x, max_lag_s=20.0)
    expected = dt * phi / (1.0 - phi)
    assert tau_c == pytest.approx(expected, rel=0.15)
    assert rho[0] == pytest.approx(1.0)


def test_effective_sample_size_and_campbell_closed_forms():
    assert effective_sample_size(1000.0, 0.5) == pytest.approx(1000.0)
    camp = campbell_moments(2.0, np.array([1.0, 3.0]), np.array([2.0, 4.0]))
    assert camp.mean == pytest.approx(4.0)
    assert camp.variance == pytest.approx(6.0)
    assert camp.cv == pytest.approx(np.sqrt(6.0) / 4.0)


def test_normal_ppf_matches_known_quantiles():
    assert normal_ppf(0.95) == pytest.approx(1.6449, abs=1e-3)
    assert normal_ppf(0.975) == pytest.approx(1.9600, abs=1e-3)
    assert normal_ppf(0.5) == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# stopping rule
# --------------------------------------------------------------------------- #
def test_required_time_is_the_closed_form():
    rule = StoppingRule(eps=0.05)
    assert rule.required_time(cv=2.0, tau_c=0.5) == pytest.approx(2 * 0.5 * 4.0 / 0.0025)
    assert rule.required_effective_samples(cv=2.0) == pytest.approx(1600.0)


def test_tail_factor_scales_the_duration_quadratically():
    one_sigma = StoppingRule(eps=0.05)
    confident = StoppingRule(eps=0.05, tail_factor=2.0)
    assert confident.required_time(2.0, 0.5) == pytest.approx(
        4.0 * one_sigma.required_time(2.0, 0.5))


def test_assess_reports_convergence_against_the_band():
    rule = StoppingRule(eps=0.10)
    pilot = PilotStatistics(duration_s=100.0, mean=1.0, std=2.0, tau_c=0.5)
    decision = rule.assess(pilot)
    assert decision.required_time_s == pytest.approx(2 * 0.5 * (2.0 / 0.10) ** 2)
    assert decision.converged is False
    long_pilot = PilotStatistics(duration_s=1e6, mean=1.0, std=2.0, tau_c=0.5)
    assert rule.assess(long_pilot).converged is True


def test_tail_factor_from_quantile():
    errors = np.linspace(0.0, 0.2, 101)
    zeta = tail_factor_from_quantile(errors, eps=0.05, alpha=0.10)
    assert zeta == pytest.approx(np.quantile(errors, 0.90) / 0.05)
    with pytest.raises(ConfigError):
        tail_factor_from_quantile([0.01, 0.02], eps=0.05, alpha=0.1)


# --------------------------------------------------------------------------- #
# pre-registration validator
# --------------------------------------------------------------------------- #
def _fake_results(manifest, **overrides):
    values = {"ac1.min_adjusted_p_value": 0.5,
              "ac2.max_adjusted_p_value": 0.001,
              "ac3.max_variance_relative_error": 0.05,
              "ac4.max_relative_error": 0.10,
              "ac5.coverage": 0.95,
              "ac5.calibration_relative_error": 0.02,
              "ac6.complete_score": 1.0}
    results = {"manifest_sha256": manifest_hash(manifest.payload),
               "criterion_ids": [entry["id"] for entry in manifest.criteria]}
    for path, value in values.items():
        head, tail = path.split(".")
        results.setdefault(head, {})[tail] = value
    for path, value in overrides.items():
        head, tail = path.split(".")
        results.setdefault(head, {})[tail] = value
    return results


def test_validator_passes_on_conforming_results(manifest):
    verdict = validate(manifest, _fake_results(manifest))
    assert verdict["verdict"] == "PASS"
    assert len(verdict["checks"]) == len(manifest.criteria)


def test_validator_rejects_a_missing_metric(manifest):
    results = _fake_results(manifest)
    del results["ac3"]["max_variance_relative_error"]
    verdict = validate(manifest, results)
    assert verdict["verdict"] == "INVALID"
    assert any(p["code"] == "MISSING_METRIC" for p in verdict["problems"])


def test_validator_rejects_an_unregistered_criterion(manifest):
    results = _fake_results(manifest)
    results["criterion_ids"] = list(results["criterion_ids"]) + ["AC7-made-up"]
    verdict = validate(manifest, results)
    assert verdict["verdict"] == "INVALID"
    assert any(p["code"] == "UNREGISTERED_CRITERION" for p in verdict["problems"])


def test_validator_detects_manifest_tampering(manifest):
    results = _fake_results(manifest)
    results["manifest_sha256"] = "0" * 64
    verdict = validate(manifest, results)
    assert verdict["verdict"] == "INVALID"
    assert any(p["code"] == "MANIFEST_MISMATCH" for p in verdict["problems"])


def test_validator_fails_a_threshold_violation(manifest):
    results = _fake_results(manifest, **{"ac5.coverage": 0.5})
    verdict = validate(manifest, results)
    assert verdict["verdict"] == "FAIL"
    failed = [c for c in verdict["checks"] if not c["passed"]]
    assert [c["id"] for c in failed] == ["AC5"]


# --------------------------------------------------------------------------- #
# quick end-to-end
# --------------------------------------------------------------------------- #
def _quick_manifest(manifest):
    payload = json.loads(json.dumps(manifest.payload))
    execution = payload["execution"]
    execution["ac1"].update({"duration_s": 600.0, "seeds": [1101, 1102]})
    execution["ac2"].update({"duration_s": 600.0, "seeds": [2101]})
    execution["ac3"].update({"duration_s": 400.0, "seeds": [3001, 3002]})
    execution["ac4"].update({"reference_duration_s": 400.0, "ensembles": 20,
                             "multipliers": [0.5, 2.0]})
    execution["ac5"].update({"n_cases": 5, "pilot_duration_s": 100.0})
    execution["ac5"]["calibration"].update({"n_cases": 25, "pilot_duration_s": 100.0})
    from tensorphysics.ice_encounter_stats.config import Manifest

    return Manifest(payload=payload, source="<quick-test>")


def test_quick_run_produces_every_block(manifest, tmp_path):
    quick = _quick_manifest(manifest)
    result = acc.run_acceptance(quick, outdir=str(tmp_path), make_figures=False, write=True)
    for block in ("ac1", "ac2", "ac3", "ac4", "ac5", "ac6"):
        assert block in result.results
    assert result.results["ac1"]["min_adjusted_p_value"] >= 0.0
    assert 0.0 <= result.results["ac5"]["coverage"] <= 1.0
    assert result.verdict["verdict"] in {"PASS", "FAIL", "INVALID"}
    assert (tmp_path / "acceptance.json").exists()
    assert (tmp_path / "verdict.json").exists()
    # the quick run must never be able to claim the shipped manifest hash
    assert result.results["manifest_sha256"] != manifest_hash(manifest.payload)


@pytest.mark.skipif(not (ARTIFACTS / "acceptance.json").exists(),
                    reason="shipped artifacts are not present")
def test_shipped_acceptance_artifacts_pass():
    results = json.loads((ARTIFACTS / "acceptance.json").read_text(encoding="utf-8"))
    verdict = json.loads((ARTIFACTS / "verdict.json").read_text(encoding="utf-8"))
    assert results["node"] == "P233W2-N05"
    assert verdict["verdict"] == "PASS", verdict["problems"]
    assert all(check["passed"] for check in verdict["checks"])
    assert results["manifest_sha256"] == verdict["manifest_sha256"]
    assert results["ac1"]["min_adjusted_p_value"] > 0.05
    assert results["ac2"]["max_adjusted_p_value"] < 0.01
    assert results["ac3"]["max_variance_relative_error"] <= 0.10
    assert results["ac4"]["max_relative_error"] <= 0.15
    assert results["ac5"]["coverage"] >= 0.90