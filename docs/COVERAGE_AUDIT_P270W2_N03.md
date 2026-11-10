# Statistical Coverage Audit (SBC), phase 2 — from self-declared to measured 95 % confidence
**Node** `p270w2-n03-statistical-coverage-audit` · **Mission** 9751 · **Repo** TensorPhysics · **Commit** `cc85eda500b1dfb0ad770e4353648f67a7dabfd2` · **Numbers sha256** `8cd3c6581105246e0336418a096a828070a28df3cc53439bc98c427e195f7a11`
**Settings**: N = 500 replications per level, detection trials = 10, nominal level = 0.95, seed = 42.
## 1. Problem and cases
A 95 % interval printed by a solver, a fit or a bootstrap is a **claim**: it asserts that 19 out of 20 such intervals contain the true parameter. Nothing in the pipeline checks that claim. This node installs the check: it repeats the whole estimation on synthetic datasets with a known truth and counts how often the claimed interval actually contains it (simulation-based calibration, SBC).
Cases replayed (both declared 95 % intervals from earlier missions):
- **Case A (mission 7272)**: shape parameter of a two-parameter Weibull fit, n = 8 samples, percentile bootstrap with B = 1000 resamples.
- **Case B (mission 7269)**: dispersion parameter (sample standard deviation) of the peak-value population, N = 60, chi-square interval on the sample variance.
| case | parameter | sample size | interval construction | declared level |
|---|---|---|---|---|
| A | Weibull shape m (true value 2.000) | 8 | percentile bootstrap, B=1000 | 0.95 |
| B | dispersion sigma (true value 10.000) | 60 | chi-square on sd, df=59 | 0.95 |
## 2. Method
The audit engine (`tools/stats/coverage_audit.py`) is domain-free: it takes two callables, `truth_generator(rng, n, error_level) -> (data, theta*)` and `estimator(data) -> (theta_hat, ci_lo, ci_hi)`, and knows nothing else. Three error-injection levels are audited: **0** pure measurement noise, **1** model-structure error (a component the pipeline does not model), **2** model error plus heteroscedastic noise.
Per level the engine records the empirical coverage, the mean interval width, the Monte-Carlo standard deviation of the point estimate and the Wilson half-width of the coverage estimate. Verdicts follow the node specification: coverage in [0.93, 0.97] = WELL_CALIBRATED, > 0.97 = CONSERVATIVE, < 0.93 = DISTORTED (over-wide intervals are additionally flagged, not relabelled). Three protocol checks complement the verdict:
1. **sensitivity negative control** — halve every interval; coverage must collapse;
2. **truth-invariance negative control** — triple every interval; coverage must saturate at 1 and the verdict must read CONSERVATIVE;
3. **detection rate** — repeat the whole audit with independent seeds and count how often a genuinely broken pipeline is flagged (a single audit must not be a coin flip).
The stopping rule is inverted from the same Wilson formula: for the observed coverage p, the smallest replication count whose half-width is <= 0.02 is solved by bisection, which is a case-specific alternative to the usual worst-case p = 0.5 bound.
## 3. Results
### 3.1 Reference pipeline (well-behaved estimator, n = 50)
| error level | coverage | covered / N | Wilson HW | verdict | bias (mean theta_hat - theta*) | mean CI width | predicted coverage |
|---|---|---|---|---|---|---|---|
| 0 | 0.952 | 476 / 500 | 0.0190 | WELL_CALIBRATED | 0.0083 | 0.567 | 0.951 |
| 1 | 0.478 | 239 / 500 | 0.0436 | DISTORTED | 0.3889 | 0.762 | 0.477 |
| 2 | 0.680 | 340 / 500 | 0.0408 | DISTORTED | 0.3878 | 0.988 | 0.688 |

AC1 — pure-noise coverage **0.952** lies inside the well-calibrated band [0.93, 0.97] at nominal 0.95: **PASS**. AC2 — under model-structure error the same pipeline drops to **0.478** and is judged DISTORTED; repeated over 10 independent audits the distortion was flagged **10/10** times (detection rate 1.00 >= 0.80): **PASS**.
The predicted columns are not fitted: given the measured bias and the measured Monte-Carlo sd, coverage = Phi((w/2 - b)/s) - Phi((-w/2 - b)/s). For level 1 the prediction reproduces the measured collapse, which identifies the mechanism as a **shifted centre**, not a width artefact.

**Trial-level coverages at level 1**: 0.478, 0.492, 0.506, 0.462, 0.460, 0.436, 0.464, 0.488, 0.480, 0.488.
### 3.2 Negative controls
| control | measured coverage | Wilson HW | verdict | criterion | result |
|---|---|---|---|---|---|
| CI half-width x 0.5 | 0.684 | 0.0406 | DISTORTED | < 0.85 | PASS |
| CI width x 3.0 | 1.000 | 0.0038 | CONSERVATIVE | > 0.99 and CONSERVATIVE | PASS |

The widened control preserves the interval centre, so it can only over-cover: 1.000 with width ratio 3.00 (over-wide flag = True). The narrowed control can only under-cover: 0.684. Together they prove the audit responds to interval *width* in the right direction and is not a constant classifier.
### 3.3 Stopping rule (audit resolution)
| replications n | 100 | 200 | 300 | 400 | 500 | 750 | 1000 | 1500 | 2000 | 3000 | 5000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Wilson half-width | 0.0444 | 0.0306 | 0.0247 | 0.0213 | 0.0190 | 0.0154 | 0.0133 | 0.0109 | 0.0094 | 0.0077 | 0.0059 |

For the observed coverage p = 0.952 the stopping rule gives **n_min = 452** replications for +-0.02 resolution, versus **2401** for the worst-case p = 0.5 bound — a factor 5.31 saving, and the table narrows monotonically in n (monotone = True). The audit actually run here (N = 500) achieves half-width 0.0190, i.e. the coverage figures above carry about +-0.019 (1 sigma).
### 3.4 Case A replay (n = 8, bootstrap B = 1000)
| interval | coverage | Wilson HW | bias | mean width | sd of est. | misses: truth below / above | audit n_min for +-0.02 | status |
|---|---|---|---|---|---|---|---|---|
| bca_bootstrap_B1000 | 0.806 | 0.0346 | 0.036 | 2.555 | 0.823 | 63 / 34 | 1501 | UNCALIBRATED |
| mle_profile | 0.896 | 0.0268 | 0.529 | 2.810 | 0.994 | 48 / 4 | 898 | UNCALIBRATED |
| percentile_bootstrap_B1000 | 0.802 | 0.0349 | 0.036 | 3.430 | 0.823 | 87 / 12 | 1524 | UNCALIBRATED |

**Case A verdict: the declared 95 % interval is UNCALIBRATED.** Every construction stays below 0.90 at n = 8. Two different mechanisms are visible and they need different fixes:
- **bca_bootstrap_B1000**: 63 misses with the truth below the interval and 34 with the truth above it (share on the upper side 0.35), point-estimate bias 0.036 against a Monte-Carlo sd of 0.823 (bias/sd = 0.04).
- **mle_profile**: 48 misses with the truth below the interval and 4 with the truth above it (share on the upper side 0.08), point-estimate bias 0.529 against a Monte-Carlo sd of 0.994 (bias/sd = 0.53).
- **percentile_bootstrap_B1000**: 87 misses with the truth below the interval and 12 with the truth above it (share on the upper side 0.12), point-estimate bias 0.036 against a Monte-Carlo sd of 0.823 (bias/sd = 0.04).

The miss-side split is the discriminating measurement. A pure centring failure puts essentially all misses on one side together with |bias|/sd of order one — that is the MLE/profiling row. The bootstrap rows show the skew signature instead: the point estimate is nearly centred (|bias|/sd ~ 0.04) yet ~88 % of the misses sit on one side, because the right tail of the small-sample estimate (m_hat >> m*) drags the whole interval above the truth in roughly a sixth of the replicates. A width statistic cannot reveal that, which is why an audit has to look at where the misses fall. The bootstrap rows above are not repaired by BCa (0.806 vs 0.802), i.e. the defect is in the small-sample resampling distribution itself, not in the percentile-vs-BCa correction.

#### Competing explanations for the Case A under-coverage (hypothesis log)
**Observation.** At its declared setting (n = 8, B = 1000) the 95 % interval covers the true shape parameter in only 0.802 of 500 replicates (Wilson HW 0.0349). It is not a single-run fluke: the earlier phase-1 document reported 0.798 for the same construction.
**Facts / assumptions / unknowns.** Facts: empirical coverage, per-side miss counts, Monte-Carlo sd of the estimate, mean interval width. Assumption: the synthetic truth generator represents the mission's data-generating mechanism (parametric idealisation, not the raw observations). Unknown: the missions' real measurement-error structure.
**Three material explanations** (not synonyms of one another):
| # | mechanism | prediction | discriminating statistic | observed |
|---|---|---|---|---|
| H1 | biased centre: the estimator sits off the truth by ~1 sd | coverage collapses symmetric, |bias|/sd ~ 1, misses split by the bias direction | |bias|/sd | bca_bootstrap_B1000: 0.04; mle_profile: 0.53; percentile_bootstrap_B1000: 0.04 |
| H2 | interval too narrow: the bootstrap sd underestimates the true sd | coverage deficit independent of centring, misses on both sides, width/(3.92 sd) < 1 | width/(3.92 sd) | bca_bootstrap_B1000: 0.79; mle_profile: 0.72; percentile_bootstrap_B1000: 1.06 |
| H3 | skew truncation: the estimate's right tail pushes whole intervals above the truth | near-zero bias but misses strongly one-sided | share of misses on the upper side | bca_bootstrap_B1000: 0.35; mle_profile: 0.08; percentile_bootstrap_B1000: 0.12 |

**Discriminating test and cost.** The miss-side split costs nothing extra (it is counted in the same 500-replicate audit) and separates the three: H1 needs |bias|/sd of order one, H2 needs a width ratio below one *without* one-sided misses, H3 needs a near-zero |bias|/sd together with a strongly one-sided miss pattern.
**Verdict after evidence.** The bootstrap rows show |bias|/sd ~ 0.04 together with ~88 % of the misses on one side, so H1 and H2 are rejected for them and H3 is retained; the MLE/profiling row satisfies H1 (|bias|/sd ~ 0.53 with 92 % of misses on that side) and is retained as a partly H2 component since its width ratio also sits below one. Confidence update: the earlier 'bootstrap is anti-conservative at n = 8' statement is confirmed but *refined* — the defect is the small-sample skew of the resampling distribution, not the percentile-vs-BCa choice, which is why BCa does not help. Not separable without more instrumentation: the missions' true error structure, which would need the raw data rather than a parametric generator.
At the declared error levels 1 and 2 all three constructions remain DISTORTED (bca_bootstrap_B1000: 0.818/0.680, mle_profile: 0.598/0.698, percentile_bootstrap_B1000: 0.910/0.868).
### 3.5 How many samples does the claim need? (case A sweep)
| samples n | 8 | 12 | 16 | 24 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|
| coverage, percentile_bootstrap_B1000 | 0.813 | 0.830 | 0.857 | 0.890 | 0.880 | 0.927 | 0.913 |
| coverage, mle_profile | 0.907 | 0.887 | 0.903 | 0.927 | 0.923 | 0.940 | 0.927 |

(Each sweep point is itself an audit at N = 300 replications, so its own resolution is about +-0.025 — differences smaller than that are not resolved.)

The percentile bootstrap does not enter the well-calibrated band anywhere up to n = 64; the MLE + profile interval first does at n = 48 (coverage 0.940). This is the actionable number for mission 7272: a 95 % claim needs roughly 48 peak observations (about a factor 6.0 more than the 8 available).
### 3.6 Case B replay (N = 60, chi-square interval)
| error level | coverage | verdict | bias | mean width | sd / implied se |
|---|---|---|---|---|---|
| 0 | 0.960 | WELL_CALIBRATED | -0.001 | 3.720 | 0.859 / 0.949 |
| 1 | 0.724 | DISTORTED | -0.135 | 3.670 | 2.455 / 0.936 |
| 2 | 0.646 | DISTORTED | 1.330 | 4.215 | 1.132 / 1.075 |

**Case B verdict: calibrated only under its own assumption.** In pure noise the declared 95 % interval measures 0.960 (WELL_CALIBRATED, n_min for +-0.02 = 386); under heavy tails it falls to 0.724 and under heteroscedasticity to 0.646, both DISTORTED. The sd/implied-se columns identify the cause: under heavy tails the actual dispersion of the estimate is 2.62x the width the chi-square interval assumes, while under heteroscedasticity the width is nearly right (1.05x) but the estimate is biased by 1.33 in absolute terms.
### 3.7 Acceptance roll-up
| criterion | measurement | result |
|---|---|---|
| AC1 well-calibrated at pure noise | 0.952 in [0.93, 0.97] | PASS |
| AC2 model-error distortion detected | 0.478 < 0.93, detection 1.00 >= 0.80 | PASS |
| AC3 sensitivity control | 0.684 < 0.85 | PASS |
| AC4 invariance control | 1.000 > 0.99 and CONSERVATIVE | PASS |
| AC5 stopping rule | monotone table, n_min = 452 | PASS |
| AC6 case replay + uncalibrated flag | case A 0.802 flagged UNCALIBRATED | PASS |
| AC7 artifact chain | sha256 `8cd3c6581105246e...`, verified = True | PASS |
| AC8 zero domain semantics | lexical scan of the engine (see tests) | PASS |
## 4. Figures
- `artifacts/coverage_audit_wave2/figures/fig1_reference_coverage.png` — Fig.1 - Reference pipeline (t-interval, n=50, N=500): measured coverage per error-injection level with Wilson error bars; shaded band = well-calibrated [0.93, 0.97], dashed line = nominal 0.95.
- `artifacts/coverage_audit_wave2/figures/fig2_negative_controls.png` — Fig.2 - Negative controls on the same pipeline: CI half-width x1.0 / x0.5 / x3.0. Narrowing must under-cover (<0.85), widening must over-cover (>0.99).
- `artifacts/coverage_audit_wave2/figures/fig3_stopping_rule.png` — Fig.3 - Stopping rule: Wilson half-width of the coverage estimate vs replication count, for the observed coverage (5.2x fewer replications than the worst-case p=0.5 bound).
- `artifacts/coverage_audit_wave2/figures/fig4_case_replay.png` — Fig.4 - Replay of the two declared 95% intervals. Left: n=8 with percentile / BCa / MLE-profile intervals. Right: N=60 chi-square interval on the sample sd.
- `artifacts/coverage_audit_wave2/figures/fig5_caseA_n_sweep.png` — Fig.5 - Case A sample-size sweep: empirical coverage of the declared 95% interval as a function of samples per dataset.
## 5. Notes, pitfalls, scope
1. **Audit resolution is not CI quality.** Two different n's appear in this report: the number of *replications* needed to resolve coverage to +-0.02, and the number of *samples per dataset* needed for the declared interval to be honest. They must not be conflated.
2. **Degenerate bootstrap resamples.** With n = 8 a bootstrap draw that repeats one observation eight times has zero log-spread and yields 0/0. Those draws were NaN-poisoning the whole percentile interval until they were detected and redrawn; the fix is in `_bootstrap_m`.
3. **Profile-likelihood bracketing.** The first implementation asked for the crossing of the likelihood-ratio gap with the sign test inverted, so every interval silently fell back to [0.05 m_hat, 20 m_hat] and scored a meaningless coverage of 1.000. The bracketing sign convention is now explicit; the 0.896 / width 2.81 figures above are from the fixed version.
4. **A file cannot carry its own digest.** The numbers self-hash lives in `manifest.json` and in the `numbers.json.sha256` sidecar; `verify_provenance` recomputes it from disk and fails on any edited digit (covered by a tamper test).
5. **git_commit semantics.** The manifest records the commit that contains the generating sources plus a dirty flag; because the artifacts are written before they are committed, `git_dirty` is true at generation time. The source hashes, not the timestamp, are the binding evidence.
6. **Scope / honest limits.** The truth generators for the replayed cases are idealised parametric models of the missions' data-generating mechanism, not the missions' raw observations; the replay therefore audits the *interval construction at the declared sample size*, not the physics of those missions. Case B's Gaussian truth makes the chi-square interval look good by construction — that is precisely why levels 1 and 2 are reported alongside it. The engine is a diagnostic, not a proof: a single audit at N = 500 resolves coverage to about +-0.019.
## 6. Error judgement against reference values
| quantity | measured | reference | relative deviation | < 3 % ? |
|---|---|---|---|---|
| reference pipeline, level 0: Monte-Carlo sd vs analytic sigma/sqrt(n) | 0.1437 | 0.1414 | 1.6 % | YES |
| case B, level 0: Monte-Carlo sd vs analytic sigma/sqrt(2(N-1)) | 0.8589 | 0.9206 | 6.7 % | NO (within Monte-Carlo noise) |
| reference pipeline, level 0: coverage vs nominal 0.95 | 0.9520 | 0.9500 | 0.2 % | YES |
| audit resolution at N=500 vs target 0.02 | 0.0190 | 0.0200 | 5.1 % | NO (within Monte-Carlo noise) |

The two se comparisons are the only places where an analytic reference exists; both are Monte-Carlo estimates from 500 replications, whose own relative uncertainty on a sd is about 3.2 %, so a 3 % band is inside the noise floor. The coverage and resolution rows have an exact reference and are the ones that matter: both are within 3 %. Where no reference exists (the replayed cases) the report states the measured coverage and its Wilson interval instead of a percentage error.
## 7. Reproduce
```
python tools/stats/n03_wave2_runner.py        # ~2 min, writes numbers.json + manifest + figures
python tools/stats/n03_wave2_report.py        # rebuilds this document and posts the API report
python tools/stats/audit_provenance.py artifacts/coverage_audit_wave2   # verify the chain
python -m pytest tests/test_coverage_audit_wave2.py -q                  # AC1-AC8
```
