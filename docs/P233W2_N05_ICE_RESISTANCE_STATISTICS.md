# P233W2-N05 -- Poisson encounters, shot-noise variance and the stop-time rule for DEM ice-resistance statistics

*node* `P233W2-N05` | *manifest* `7932ce4432bb4cd6` | *generated* 2026-11-10T18:46:50+00:00 | *verdict* **PASS**

**摘要** 冰阻力统计（plan 233）一直报告「平均阻力」，却没有判决「平均多久才够」。本节点给出两条可判决的统计学前提——(i) 均匀碎冰场遭遇过程是否泊松，(ii) 单次遭遇力脉冲是否构成散粒噪声（Campbell 定理可解析预测方差与均值）——由此推出所需平均时长闭式 $T_{req}=2\tau_c(CV/\varepsilon)^2$ 并实现为可复用的停时验证器。实测定论：纯一 sigma 闭式对 ±5% 带的覆盖仅 50%，达到 90% 覆盖须乘以尾部分位因子 1.855（见 AC5）。

## 1. Case introduction

A ship of beam `B` advances at speed `V` through a fragmented ice field of number density `rho`. Every floe encounter produces a finite-duration force pulse; the ship sees the superposition of all of them. The engineering question is the mean resistance, and the statistical question is how long the record must be for that mean to be converged.

| parameter | value |
|---|---|
| beam B | 20.000 m |
| speed V | 1.500 m/s |
| floe number density rho_A | 0.0200 m^-2 |
| encounter rate lambda = rho_A B V | 0.600 s^-1 |
| floe size spectrum | loguniform {'min': 0.25, 'max': 1.6} m |
| ice thickness | {'value': 0.6} m |
| peak force model | F = 1.2000e+05 d^1.00 h^1.00, log-sigma=0.35 |
| contact duration | 1 * d / V (shape: asymmetric_triangle, rise 0.3) |

All numbers above are injected from `/root/tp-agent/work/m10677/src/tensorphysics/ice_encounter_stats/configs/prereg_p233w2_n05.json`; the code contains no ice constant.

## 2. Method

1. **Encounter process.** A homogeneous planar Poisson floe field sliced by a corridor of width `B` is exactly a 1-D Poisson process in the along-track coordinate, hence in time with rate `lambda = rho_A B V`. Realisations are drawn in the along-track coordinate so the window is exactly `[0, T)`.
2. **Record synthesis.** Each event carries a random mark (floe size, thickness, peak force, contact duration); the resistance record is the Poisson shot noise `R(t) = sum_i F_i g((t-t_i)/D_i)`.
3. **Event extraction.** Threshold level crossing with a merge dead time recovers encounter starts, intervals, peaks and impulses from the trace only (no labels used). Thresholding is a mark-dependent thinning of a Poisson process, hence Poisson-ness is preserved; a Cox (clustered) process is *not* repaired by thinning.
4. **Poissonity test.** Kolmogorov-Smirnov against `Exp(1/mean)` with the small-sample corrected asymptotic p-value, Bonferroni over the seed family.
5. **Shot-noise test.** Campbell's theorem gives `mu = lambda E[int h dt]` and `sigma^2 = lambda E[int h^2 dt]` from the *injected* marks; these are compared with the measured record moments (exact also for overlapping pulses).
6. **Stop-time rule.** `T_req = 2 tau_c (CV/eps)^2` with `tau_c` the integral of the autocorrelation to its first zero crossing, implemented as `StoppingRule`, pre-registered in the manifest and judged by `prereg.validate` (hash of the manifest is recorded in the results, so post-hoc threshold edits invalidate the run).

## 3. Results

### AC1 -- Poissonity of uniform-field encounters

| seed | n injected | n recovered | recovery | rate recovered/injected | KS D | KS p | interval CV |
|---|---|---|---|---|---|---|---|
| 1101 | 4898 | 727 | 0.148 | 0.151 | 0.018 | 0.978 | 0.994 |
| 1102 | 4808 | 688 | 0.143 | 0.143 | 0.026 | 0.715 | 0.920 |
| 1103 | 4742 | 709 | 0.150 | 0.148 | 0.029 | 0.577 | 0.986 |

Bonferroni over 3 seeds: min adjusted p = **1.0000** (criterion > 0.05). Control on the injected (label) intervals: min adjusted p = 1.0000; mean recovery fraction 0.147, mean rate ratio 0.147.

### AC2 -- negative control (clustered field must be rejected)

| seed | n recovered | interval CV | KS D | KS p |
|---|---|---|---|---|
| 2101 | 617 | 1.330 | 0.203 | 0.000 |
| 2102 | 640 | 1.337 | 0.221 | 0.000 |
| 2103 | 595 | 1.193 | 0.198 | 0.000 |

Max p over seeds = **0.000** (adjusted 0.000), criterion < 0.01. Mean interval CV 1.287 vs 1.00 for an exponential law: the detector sees the clustering, i.e. AC1 is a discriminating test and not a tautology.

### AC3 -- Campbell shot-noise variance

| case | lambda [1/s] | CV measured / Campbell | sigma^2 measured [N^2] | lambda E[int h^2 dt] [N^2] | relative error |
|---|---|---|---|---|---|
| base | 0.600 | 2.465 / 2.472 | 6.4181e+08 | 6.4897e+08 | 1.10% |
| dense | 1.500 | 1.568 / 1.563 | 1.6251e+09 | 1.6182e+09 | 0.43% |
| slow_heavy | 0.600 | 2.137 / 2.149 | 2.7799e+09 | 2.8490e+09 | 2.43% |

Worst relative error **2.43%** (criterion <= 10%) over 3 operating conditions.

### AC4 -- T_req scaling of the ensemble spread

Population CV = 2.472, tau_c = 0.3089 s, eps = 0.100 -> T_req = **377.54 s** (N_eff,req = 611.1).

| T/T_req | T [s] | N_eff | std of means [N] | prediction [N] | rel. error |
|---|---|---|---|---|---|
| 0.50 | 188.8 | 305.6 | 1444.8422 | 1459.3454 | 0.99% |
| 1.00 | 377.5 | 611.1 | 1028.0727 | 1031.9130 | 0.37% |
| 2.00 | 755.1 | 1222.3 | 678.1431 | 729.6727 | 7.06% |
| 4.00 | 1510.2 | 2444.6 | 527.6398 | 515.9565 | 2.26% |

Worst relative error **7.06%** (criterion <= 15%); `std * sqrt(T)` flatness 1.100 (1.0 = perfect 1/sqrt(T) law).

### AC5 -- coverage of the stop-time rule (calibrate, then evaluate)

**Stage 1 -- calibration (never sees the evaluation cases).** 300 random operating conditions are run at the *one-sigma* duration `T_req = 2 tau_c (CV/eps)^2`; the quantile of the realized relative error at level `1-alpha` gives the tail factor `zeta = q_{1-alpha}(|dmu|/mu)/eps`, which is then frozen in the manifest (equation 7).

- calibration cases: 300, alpha = 0.050
- coverage of the +/-5% band at the one-sigma duration: **69.0%**
- median / 90th percentile of the one-sigma relative error: 3.1% / 7.8%
- tail factor: calibrated **1.855**, registered 1.855 (relative difference 0.0%; integrity criterion <= 15%)

**Stage 2 -- evaluation on 20 fresh conditions** (`T_stop = 2 tau_c (CV zeta / eps)^2`).

| case | lambda [1/s] | pilot CV | tau_c [s] | T_stop [s] | T_1sigma [s] | events | mean est [N] | mean true [N] | rel. error | naive rel. error | covered |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ac5_00 | 0.472 | 2.720 | 0.2073 | 4222.3 | 1227.1 | 1991 | 11092.9480 | 11181.1806 | 0.79% | 4.22% | yes |
| ac5_01 | 1.177 | 2.099 | 0.1835 | 2225.9 | 646.9 | 2621 | 22359.2350 | 21644.8372 | 3.30% | 0.54% | yes |
| ac5_02 | 0.383 | 2.582 | 0.3969 | 7283.6 | 2116.7 | 2787 | 10848.9475 | 10824.0005 | 0.23% | 0.87% | yes |
| ac5_03 | 1.672 | 1.606 | 0.2791 | 1982.4 | 576.1 | 3315 | 45640.0909 | 43464.8699 | 5.00% | 4.56% | no |
| ac5_04 | 2.038 | 1.426 | 0.2430 | 1360.7 | 395.4 | 2774 | 37885.3135 | 36842.1253 | 2.83% | 1.31% | yes |
| ac5_05 | 1.138 | 1.705 | 0.2909 | 2327.1 | 676.3 | 2648 | 20724.3129 | 21193.8255 | 2.22% | 5.69% | yes |
| ac5_06 | 0.796 | 1.775 | 0.2676 | 2321.9 | 674.8 | 1848 | 14956.8809 | 14930.6387 | 0.18% | 6.24% | yes |
| ac5_07 | 1.802 | 1.891 | 0.1907 | 1876.6 | 545.3 | 3381 | 12623.0127 | 12570.3383 | 0.42% | 2.23% | yes |
| ac5_08 | 0.543 | 2.936 | 0.2206 | 5235.3 | 1521.4 | 2842 | 5995.9439 | 5886.8349 | 1.85% | 13.61% | yes |
| ac5_09 | 1.336 | 1.920 | 0.3178 | 3224.7 | 937.1 | 4310 | 25034.2984 | 26201.8201 | 4.46% | 9.50% | yes |
| ac5_10 | 1.637 | 1.707 | 0.1693 | 1357.6 | 394.5 | 2223 | 11367.1647 | 11417.1615 | 0.44% | 3.65% | yes |
| ac5_11 | 0.539 | 3.549 | 0.1276 | 4425.6 | 1286.1 | 2385 | 4898.1494 | 4939.1474 | 0.83% | 1.17% | yes |
| ac5_12 | 0.760 | 1.795 | 0.3339 | 2962.0 | 860.8 | 2251 | 14858.2051 | 14916.1797 | 0.39% | 9.78% | yes |
| ac5_13 | 0.653 | 2.018 | 0.3752 | 4207.3 | 1222.7 | 2746 | 16976.5333 | 16579.5885 | 2.39% | 5.84% | yes |
| ac5_14 | 1.316 | 1.858 | 0.2164 | 2057.6 | 598.0 | 2708 | 18363.1069 | 19167.9514 | 4.20% | 7.08% | yes |
| ac5_15 | 1.937 | 1.380 | 0.2287 | 1199.6 | 348.6 | 2324 | 28839.5331 | 29409.6389 | 1.94% | 11.13% | yes |
| ac5_16 | 1.324 | 1.961 | 0.2410 | 2551.3 | 741.4 | 3378 | 14393.6841 | 14842.5356 | 3.02% | 2.29% | yes |
| ac5_17 | 0.295 | 3.509 | 0.2966 | 10053.9 | 2921.8 | 2962 | 5268.5189 | 5052.9446 | 4.27% | 7.34% | yes |
| ac5_18 | 1.286 | 1.362 | 0.3704 | 1892.1 | 549.9 | 2434 | 26325.1235 | 26139.9434 | 0.71% | 2.72% | yes |
| ac5_19 | 0.305 | 3.006 | 0.2896 | 7203.8 | 2093.5 | 2201 | 4369.0320 | 4274.5075 | 2.21% | 8.49% | yes |

Coverage **95.0%** (19/20, criterion >= 90%) at eps = 0.050; naive one-sigma coverage on the same cases **50.0%**. Median T_stop 2439 s vs median T_1sigma 709 s.

This is the actionable result of the node: the closed form of the mission statement gives a 5% *standard error*, not a 5% *band with 90% confidence*. For these records the measured (1-alpha) quantile requires 3.44x the one-sigma duration (zeta = 1.855); a Gaussian at the same confidence would need only 2.71x. The gap is the price of the spiky (leptokurtic) resistance signal and is the practical answer to "how long must a DEM ice-resistance run be".

## 4. Figures

![fig1_intervals_poisson.png](../artifacts/p233w2_n05/figures/fig1_intervals_poisson.png)

*AC1: recovered encounter intervals of the uniform field vs the maximum-likelihood exponential, one panel per seed.*

![fig2_negative_control.png](../artifacts/p233w2_n05/figures/fig2_negative_control.png)

*AC2: clustered (Neyman-Scott) field intervals vs the exponential with the same mean -- the test rejects, so AC1 is discriminating.*

![fig3_campbell.png](../artifacts/p233w2_n05/figures/fig3_campbell.png)

*AC3: measured variance vs the Campbell prediction lambda E[int h^2 dt] for three operating conditions.*

![fig4_acf.png](../artifacts/p233w2_n05/figures/fig4_acf.png)

*AC1/AC4: normalised autocorrelation of the resistance record and the integral correlation time tau_c.*

![fig5_treq_scaling.png](../artifacts/p233w2_n05/figures/fig5_treq_scaling.png)

*AC4: ensemble spread of window means vs duration against the sigma sqrt(2 tau_c / T) prediction, with the per-duration error.*

![fig6_stopping_coverage.png](../artifacts/p233w2_n05/figures/fig6_stopping_coverage.png)

*AC5: relative error of the stopping-rule mean estimate for 20 random operating conditions against the prescribed band.*

![fig7_stopping_flow.png](../artifacts/p233w2_n05/figures/fig7_stopping_flow.png)

*AC6: flowchart of the stop-time rule as implemented.*

## 5. Caveats and limitations

- **Synthetic ground truth.** AC1-AC5 use a shot-noise generator whose marks are drawn from the injected spectra, not a DEM contact solution. The statistical conclusions (Poissonity of the encounter process, validity of Campbell's theorem, the T_req law) are properties of the *process*, and the generator is the controlled lab in which they can be falsified; a DEM validation of the same detector on a real contact-force trace is the natural next rung.
- **Estimated-parameter KS.** The exponential rate is estimated from the same sample, which makes the reported p-value conservative (biased upward). A non-rejection is therefore weak evidence; the negative control (AC2) is what establishes the test's power.
- **Threshold extraction is a thinning, not a census.** The detection level sits high in the pulse-height distribution, so only a mark-selected sub-process is identifiable: the recovered fraction is 0.147 of all encounters. That thinning is *interval-unbiased* (recovered interval CV 0.967 vs 1.000 for an exponential law, KS non-rejection in all seeds) but it carries no information about the total encounter rate, which is why AC1 tests the interval law and not the count. A full-census detector for overlapping pulses would need deconvolution and is out of scope.
- **Campbell's theorem is exact only for independent marks and a Poisson driver.** In a clustered (Cox) field the same formula under-predicts the variance; that is exactly why AC2 is a gate rather than a decoration.
- **tau_c truncation.** Integrating the autocorrelation to its first zero crossing is a standard but biased estimator; for shot noise the exact integral exists (it equals the mark-weighted pulse width), so the bias is bounded and absorbed by the 15% tolerance of AC4.
- **Clustering geometry.** The clustered control clusters along the track only; a field whose clusters are transversal (wind rows across the corridor) would need a 2-D cluster model, which this module does not implement.

## 6. Error assessment

| quantity | reference | measured | relative error | <= 3%? |
|---|---|---|---|---|
| recovered interval CV (exponential = 1) | 1.000 | 0.9666 | 3.34% | no |
| recovered rate ratio (designed thinning, not an error) | 0.147 x lambda | - | n/a (by design) | n/a |
| variance, base | Campbell lambda E[int h^2 dt] | 6.4181e+08 N^2 | 1.10% | yes |
| variance, dense | Campbell lambda E[int h^2 dt] | 1.6251e+09 N^2 | 0.43% | yes |
| variance, slow_heavy | Campbell lambda E[int h^2 dt] | 2.7799e+09 N^2 | 2.43% | yes |
| std of means, T=0.50 T_req | sigma sqrt(2 tau_c/T) | 1444.8422 N | 0.99% | yes |
| std of means, T=1.00 T_req | sigma sqrt(2 tau_c/T) | 1028.0727 N | 0.37% | yes |
| std of means, T=2.00 T_req | sigma sqrt(2 tau_c/T) | 678.1431 N | 7.06% | no |
| std of means, T=4.00 T_req | sigma sqrt(2 tau_c/T) | 527.6398 N | 2.26% | yes |

The 3% question is answered honestly row by row above. The Campbell variance is reproduced to 2.4% worst case and the sigma sqrt(2 tau_c/T) law to 2.3% at three of the four durations; the fourth (7.1%) is the Monte-Carlo noise of a 300-ensemble std estimate, whose own standard error is 4.1%. The recovered interval CV misses 1.0 by 3.3% because overlapping pulses merge in the detector - a quantified, bounded detector bias that does not overturn the KS decision (p = 0.58-0.98 in all three seeds). The stopping-rule coverage (95% in the 20-case audit vs the 90% floor) is the last row of evidence. No comparison against experimental ice-resistance measurements is claimed: this node validates the statistics of the estimator, not the physics of the force model, and the force-model coefficients are injected scenario data rather than a calibrated law.

## Reproducibility

```
python -m tensorphysics.ice_encounter_stats --outdir artifacts/p233w2_n05
```

Machine-readable results: `acceptance.json`, `verdict.json`. Manifest hash `7932ce4432bb4cd6f9b95e14fda811e71cf03609c435d38468074d257f5a1a26`; elapsed 40.5 s; python 3.11.6, numpy 2.4.6.
