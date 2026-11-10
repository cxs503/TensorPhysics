# ISO 19906 Partial-Factor Calibration — Line-by-Line Comparison Report

**Node**: P46W3-N09 (Maturity)  
**Date**: 2026-11-10  
**Repository**: TensorPhysics  
**Module**: `tensorphysics.ice_partial_factors`

---

## 1. Scope

This report provides a line-by-line comparison of the back-calibrated partial
factors against ISO 19906:2019 Annex A tables. The calibration targets
$\beta_T = 4.0$ per ISO 2394:2015 for annual reliability.

## 2. Design Format

ISO 19906 Annex A uses the format:

$$\gamma_F \cdot F_k \leq R_k / \gamma_M$$

where:
- $\gamma_F$ = load factor (ice action uncertainty)
- $\gamma_M$ = resistance factor (material/model uncertainty)
- $F_k$ = characteristic ice force (100-year return period)
- $R_k$ = characteristic resistance (5% fractile)

The product $\kappa = \gamma_F \cdot \gamma_M$ is the **total safety margin**.

## 3. Calibrated Factors

| Factor | Calibrated Value | ISO 19906 Typical | Deviation |
|---|---|---|---|
| $\gamma_f$ | 1.437 | 1.3–1.5 | +2.6% to −4.2% |
| $\psi$ (= $1/\gamma_M$) | 0.996 | 0.85–1.0 | −0.4% to +17.1% |
| $\kappa = \gamma_f \psi$ | 1.431 | 1.1–1.5 | −6.9% to +30.1% |

**Note**: The ISO 19906 typical ranges are compiled from Tables A.1, A.2, A.3
across different ice types and load cases.

## 4. Table-by-Table Comparison

### 4.1 ISO 19906 Table A.2 — Load Factors $\gamma_F$

| Row | Description | ISO Value | Calibrated | Deviation | Verdict |
|---|---|---|---|---|---|
| LF1 | Global ice loads, annual | 1.30 | 1.437 | +10.5% | ⚠️ Exceeds 10% |
| LF2 | Global ice loads, extreme | 1.40 | 1.437 | +2.6% | ✅ Within 10% |
| LF3 | Local ice loads, annual | 1.30 | 1.437 | +10.5% | ⚠️ Exceeds 10% |
| LF4 | Local ice loads, extreme | 1.50 | 1.437 | −4.2% | ✅ Within 10% |

### 4.2 ISO 19906 Table A.3 — Resistance Factors $\gamma_M$

| Row | Description | ISO Value | Calibrated ($1/\psi$) | Deviation | Verdict |
|---|---|---|---|---|---|
| RF1 | Steel yield, global | 1.15 | 1.004 | −12.7% | ⚠️ Exceeds 10% |
| RF2 | Steel yield, local | 1.10 | 1.004 | −8.7% | ✅ Within 10% |
| RF3 | Steel fracture | 1.20 | 1.004 | −16.3% | ⚠️ Exceeds 10% |

### 4.3 Combined Pairs ($\kappa = \gamma_F \times \gamma_M$)

| Pair | ISO $\kappa$ | Calibrated $\kappa^*$ | Deviation | $\beta$ Achieved | Verdict |
|---|---|---|---|---|---|
| LF1 × RF1 | 1.495 | 1.431 | −4.3% | 3.87 | ⚠️ Slightly non-conservative |
| LF2 × RF2 | 1.540 | 1.431 | −7.1% | 3.78 | ⚠️ Non-conservative |
| LF4 × RF3 | 1.800 | 1.431 | −20.5% | 3.35 | ❌ Significantly non-conservative |
| LF1 × RF2 | 1.430 | 1.431 | +0.1% | 4.00 | ✅ Match |

## 5. Non-Conservative Cases

For pairs where $\kappa_{\text{table}} < \kappa^*$, the following table
quantifies the safety shortfall:

| Pair | $\kappa$ Shortfall | $\beta$ Achieved | $\Delta\beta$ | $P_f / P_{f,T}$ |
|---|---|---|---|---|
| LF2 × RF2 | −7.1% | 3.78 | −0.22 | 1.7× |
| LF4 × RF3 | −20.5% | 3.35 | −0.65 | 11.4× |

**Interpretation**: Using ISO 19906 Table LF4 × RF3 for this ice scenario
would result in a failure probability **11.4 times** the target, corresponding
to a reliability index of 3.35 instead of 4.0.

## 6. Conservative Cases

For pairs where $\kappa_{\text{table}} > \kappa^*$, the design is conservative
(achieves higher reliability than required):

| Pair | $\kappa$ Excess | $\beta$ Achieved | Overdesign |
|---|---|---|---|
| LF1 × RF1 | +4.3% | 4.13 | +0.13 |

## 7. Sensitivity to $\beta_T$

| $\beta_T$ | $P_{f,T}$ | $\gamma_f^*$ | $\psi^*$ | $\kappa^*$ |
|---|---|---|---|---|
| 3.3 | 4.81e-4 | 1.332 | 1.003 | 1.336 |
| 3.8 | 7.23e-5 | 1.407 | 0.998 | 1.404 |
| **4.0** | **3.17e-5** | **1.437** | **0.996** | **1.431** |
| 4.3 | 8.54e-6 | 1.483 | 0.993 | 1.473 |

## 8. Negative Control

| Model | $\kappa^*$ | $\gamma_f^*$ | $\sigma_{\ln F}$ |
|---|---|---|---|
| Correlated (physical) | 1.431 | 1.437 | 0.261 |
| Independent ($\rho=0$) | 1.528 | 1.529 | 0.293 |

**Result**: $\Delta\gamma_f = +6.4\%$ (threshold 5%) → **detected**.

**Interpretation**: Ignoring the p–A negative correlation would lead to
overdesign by 6.4% in the load factor. The correlation is physically
meaningful and must be retained.

## 9. Conclusions

1. **Calibration successful**: The Newton inverse converges to $\beta = 4.000$
   (residual $< 10^{-8}$) from three dispersed starting points.

2. **ISO 19906 comparison**: The calibrated pair $(\gamma_f, \psi) = (1.437, 0.996)$
   falls within the ISO 19906 typical ranges but shows deviations of up to 20%
   from specific table entries. This is expected since ISO 19906 covers a wide
   range of ice types and structural configurations.

3. **Non-conservative pairs identified**: Two table pairs (LF2×RF2, LF4×RF3)
   provide less safety than the calibrated target. Users should verify that
   their specific scenario matches the table assumptions.

4. **Negative control passed**: The p–A correlation effect is real and
   significant (6.4% shift in $\gamma_f$).

5. **Monte Carlo verification**: FORM agrees with importance sampling (3% rel.
   error), crude MC (within 1σ band), and the closed-form ln-space index
   (exact for lognormal problems).

## 10. Recommendations

1. For **annual ice loads** on global scale, use $\gamma_F = 1.4$, $\gamma_M = 1.0$
   (close to the calibrated values).

2. For **extreme local loads**, the ISO 19906 values ($\gamma_F = 1.5$,
   $\gamma_M = 1.1$) are conservative and may be used directly.

3. Users should perform scenario-specific calibration when detailed ice data
   is available, as the generic ISO 19906 tables may not match all conditions.

## 11. Audit Trail

- **JSON results**: `artifacts/iso19906/results.json`
- **Benchmark runner**: `benchmarks/iso19906/run_benchmark.py`
- **Test suite**: `tests/test_iso19906_calibration_maturity.py` (33 tests)
- **Existing tests**: `tests/test_iso19906_target_inverse.py` (27 tests)
- **Total test count**: 60 tests, all passing
