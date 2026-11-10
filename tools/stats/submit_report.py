#!/usr/bin/env python3
"""Submit the coverage audit report to the Tensor API."""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Load results
results_path = ROOT / "artifacts" / "coverage_audit" / "case_studies.json"
with open(results_path) as f:
    results = json.load(f)

# Load figures as base64
fig_dir = ROOT / "artifacts" / "coverage_audit" / "figures"
figures = []
for fig_name in ["m7272_coverage_curves.png", "m7269_coverage_curves.png", 
                  "coverage_summary.png", "stopping_rule.png"]:
    fig_path = fig_dir / fig_name
    with open(fig_path, "rb") as f:
        png_b64 = base64.b64encode(f.read()).decode()
    figures.append({
        "name": fig_name.replace(".png", ""),
        "caption": fig_name.replace(".png", "").replace("_", " ").title(),
        "png_base64": png_b64
    })

# Build markdown report
markdown = """# Statistical Coverage Audit (SBC): Empirical Coverage Calibration

## ① Case Introduction

**Physical Background**: Confidence intervals (CIs) are ubiquitous in uncertainty quantification for ship performance prediction. However, the *actual* coverage probability of a claimed 95% CI is rarely verified. This audit provides a systematic methodology to measure empirical coverage and diagnose calibration failures.

**Two Mission Cases Audited**:

1. **m7272 — Weibull shape parameter m**: Small-sample (n=8) estimation of Weibull shape parameter from extreme value data. Estimator: log-linear regression + nonparametric bootstrap (B=500). Target: 95% percentile CI.

2. **m7269 — σ divergence**: Peak stress standard deviation estimation from N_peak≈60 observations. Estimator: sample standard deviation + chi-squared CI. Target: 95% exact CI.

**Error Injection Levels**:
- Level 0: Pure measurement noise (idealised Gaussian/Weibull sampling)
- Level 1: Model structure error (contamination / heavy tails)
- Level 2: Model error + heteroscedasticity (mixture distributions)

## ② Methodology

**Audit Engine**: `tools/stats/coverage_audit.py` — domain-agnostic Monte-Carlo coverage calibration.

**Protocol**:
1. Given truth generator G(rng, n, level) → (data, θ*) and estimator E(data) → (θ̂, CI_lo, CI_hi)
2. For N=500 independent replications:
   - Generate synthetic dataset with known θ*
   - Compute CI via estimator
   - Record whether θ* ∈ [CI_lo, CI_hi]
3. Empirical coverage = (number of covers) / N
4. Wilson score interval for coverage uncertainty

**Three-State Verdict**:
- WELL_CALIBRATED: coverage ∈ [0.93, 0.97]
- CONSERVATIVE: coverage > 0.97
- DISTORTED: coverage < 0.93 OR (coverage > 0.99 AND CI width anomalous)

**Stopping Rule**: Minimum replications n_min for ±0.02 Wilson half-width resolution.

## ③ Results

### Case 1: Weibull shape parameter m (m7272)

| Error Level | Coverage | Verdict | Wilson HW | Mean CI Width |
|-------------|----------|---------|-----------|---------------|
"""

m7272 = results["m7272_weibull_m"]
for lv in m7272["levels"]:
    markdown += f"| {lv['error_level']} | {lv['coverage']:.3f} | {lv['verdict']} | {lv['wilson_hw']:.4f} | {lv['mean_ci_width']:.3f} |\n"

markdown += f"""
**Overall verdict**: {m7272['overall_verdict']}

**Interpretation**: The Weibull log-linear bootstrap CI severely under-covers at all error levels. At Level 0 (pure noise), coverage is only 79.8% — far below the nominal 95%. This indicates the bootstrap percentile method is anti-conservative for small samples (n=8). The log-linear estimator itself has high variance at n=8, and the bootstrap fails to capture this adequately.

### Case 2: σ divergence (m7269)

| Error Level | Coverage | Verdict | Wilson HW | Mean CI Width |
|-------------|----------|---------|-----------|---------------|
"""

m7269 = results["m7269_sigma"]
for lv in m7269["levels"]:
    markdown += f"| {lv['error_level']} | {lv['coverage']:.3f} | {lv['verdict']} | {lv['wilson_hw']:.4f} | {lv['mean_ci_width']:.3f} |\n"

markdown += f"""
**Overall verdict**: {m7269['overall_verdict']}

**Interpretation**: The chi-squared CI for σ is well-calibrated under ideal Gaussian assumptions (Level 0: 96.0% coverage). However, under model structure error (heavy-tailed t-distribution, Level 1) or heteroscedasticity (Level 2), coverage collapses to 72.4% and 64.6% respectively. This demonstrates the chi-squared CI's fragility to non-Gaussian data — a critical finding for peak stress analysis where heavy tails are common.

### Stopping Rule

For ±0.02 resolution on coverage estimation, the minimum replications required:
- n_min = 2401 (for all cases, based on worst-case p=0.5)

Our N=500 replications achieve Wilson HW ≈ 0.02–0.04, which is adequate for diagnostic purposes but not for high-precision calibration claims.

## ④ Results Figures

See attached figures:
1. **m7272_coverage_curves**: Convergence of empirical coverage for Weibull m across three error levels
2. **m7269_coverage_curves**: Convergence of empirical coverage for σ divergence across three error levels
3. **coverage_summary**: Bar chart comparing empirical vs nominal coverage across both cases
4. **stopping_rule**: Wilson half-width vs replications for different coverage levels

## ⑤ Limitations & Caveats

1. **Small sample size (n=8) for Weibull**: The log-linear estimator has high variance at n=8. MLE would be more efficient but is computationally expensive for bootstrap. Results are indicative but not definitive.

2. **Bootstrap B=500**: Increasing B to 1000 or 2000 would reduce bootstrap noise but was computationally prohibitive for 500 replications.

3. **Synthetic error models**: The three error levels are stylised. Real-world model errors may be more complex.

4. **Single true parameter value**: We tested m=2.0 and σ=10.0 MPa. Coverage may vary across the parameter space.

5. **N=500 replications**: Achieves ±0.03 resolution, not the ±0.02 target. For high-stakes decisions, N≥2401 is recommended.

## ⑥ Error Assessment

**Quantitative comparison to nominal 95%**:

| Case | Level | Empirical Coverage | Deviation from 95% | Within [93%, 97%]? |
|------|-------|-------------------|-------------------|-------------------|
"""

for lv in m7272["levels"]:
    dev = abs(lv["coverage"] - 0.95)
    within = "YES" if 0.93 <= lv["coverage"] <= 0.97 else "NO"
    markdown += f"| m7272 | {lv['error_level']} | {lv['coverage']:.3f} | {dev:.3f} | {within} |\n"

for lv in m7269["levels"]:
    dev = abs(lv["coverage"] - 0.95)
    within = "YES" if 0.93 <= lv["coverage"] <= 0.97 else "NO"
    markdown += f"| m7269 | {lv['error_level']} | {lv['coverage']:.3f} | {dev:.3f} | {within} |\n"

markdown += """
**Key findings**:
- m7272 Weibull bootstrap CI: **FAILS** calibration at all levels (deviations 0.04–0.15)
- m7269 chi-squared CI: **PASSES** at Level 0 (deviation 0.01) but **FAILS** at Levels 1–2 (deviations 0.15–0.23)

**Recommendation**: Both estimators require improvement before their CIs can be trusted for decision-making. For m7272, consider BCa bootstrap or profile likelihood. For m7269, consider robust variance estimators or bootstrap when non-Gaussianity is suspected.

## Audit Chain

- Tool: `tools/stats/coverage_audit.py` (domain-agnostic engine)
- Case studies: `tools/stats/case_studies.py` (domain-injected via protocols)
- Tests: `tests/test_coverage_audit.py` (16 tests, all passing)
- Results: `artifacts/coverage_audit/case_studies.json`
- Figures: `artifacts/coverage_audit/figures/*.png`
- Reproducibility: seed=42, N=500 replications, deterministic RNG
"""

# Build report JSON
report = {
    "title": "Statistical Coverage Audit (SBC): Empirical Coverage Calibration for m7272 Weibull m and m7269 σ Divergence",
    "summary": "Domain-agnostic coverage audit tool applied to two mission cases. Weibull bootstrap CI severely under-covers (79.8% at Level 0). Chi-squared CI for σ is well-calibrated under Gaussian assumptions (96.0%) but fails under model error (72.4% and 64.6%). Both estimators require improvement before their 95% CIs can be trusted.",
    "scenario_key": "p270w1-n03-statistical-coverage-audit",
    "mission_id": 9362,
    "tags": ["coverage-audit", "uncertainty-quantification", "confidence-interval", "bootstrap", "weibull", "statistical-calibration"],
    "markdown": markdown,
    "figures": figures
}

# Save report JSON
report_path = ROOT / "artifacts" / "coverage_audit" / "report.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"Report saved to {report_path}")
print(f"Report size: {len(json.dumps(report))} bytes")
print(f"Markdown length: {len(markdown)} chars")
print(f"Figures: {len(figures)}")
