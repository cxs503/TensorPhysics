# C1-ICE-RESIST: Ice Resistance Full-Spectrum Empirical Chain
## Lindqvist / Keinonen-B / Riska vs MT Uikku — Final Analysis

**Date**: 2026-11-11 12:23 UTC  
**Mission**: m14881 (C1-ICE-RESIST)  
**External Anchor**: MT Uikku (Hu & Zhou 2016)

---

## 1. Case Introduction

Three classical semi-empirical ice resistance models evaluated against MT Uikku full-scale
model test data (10 points, level ice, h=0.63-1.04m, V=0.2-1.0 m/s).

**Ship**: MT Uikku (DAS tanker), L=150m, B=21.3m, φ=21°, flare=30°

---

## 2. Methodology

### Our Implementations
- **Lindqvist (1989)**: Corrected buoyancy interpretation (R_f = 0.5·ρ_i·g·h·B·wedge)
- **Keinonen (1983)**: R = C0·σ_f·h^1.5·B^0.5·(1+C1·Fr_h)·f(φ), C0=1.8, C1=0.25
- **Riska (1997)**: DAS ahead mode with 0.4 coefficients

All use **per-test measured** ice properties (σ_f, σ_c) from Hu & Zhou Table 3.

### Literature Baselines
From Hu & Zhou (2016) Table 8 — predictions reproduced in the source paper.

---

## 3. Results

### 3.1 Per-Point Comparison

| Pt | h[m] | V[m/s] | Meas[kN] | Our_Lin (%) | Lit_Lin (%) | Our_Kei (%) | Lit_Kei (%) | Our_Ris (%) | Lit_Ris (%) |
|---|------|--------|----------|-------------|-------------|-------------|---------------|-------------|-------------|
| 103 | 0.77 | 0.2 | **480** | 413 (14%) | 320 (33%) | 1975 (311%) | 510 (6%) | 330 (31%) | 610 (27%) |
| 104 | 0.76 | 0.5 | **560** | 499 (11%) | 380 (32%) | 2318 (314%) | 600 (7%) | 400 (29%) | 630 (12%) |
| 205 | 0.96 | 0.2 | **670** | 666 (1%) | 510 (24%) | 3486 (420%) | 740 (10%) | 533 (20%) | 800 (19%) |
| 206 | 0.95 | 0.5 | **720** | 661 (8%) | 560 (22%) | 3485 (384%) | 810 (12%) | 529 (27%) | 840 (17%) |
| 301 | 1.04 | 0.2 | **540** | 1038 (92%) | 426 (21%) | 2306 (327%) | 663 (23%) | 830 (54%) | 892 (65%) |
| 302 | 1.04 | 0.5 | **830** | 1042 (26%) | 538 (35%) | 2923 (252%) | 783 (6%) | 833 (0%) | 942 (14%) |
| 303 | 1.04 | 1.0 | **780** | 1010 (30%) | 572 (27%) | 2684 (244%) | 847 (9%) | 807 (4%) | 1025 (31%) |
| 401 | 0.63 | 0.2 | **150** | 621 (314%) | 251 (67%) | 1634 (989%) | 425 (183%) | 497 (231%) | 473 (215%) |
| 402 | 0.63 | 0.5 | **290** | 823 (184%) | 322 (11%) | 2142 (639%) | 526 (81%) | 658 (127%) | 501 (73%) |
| 403 | 0.63 | 1.0 | **390** | 710 (82%) | 351 (10%) | 1970 (405%) | 579 (48%) | 568 (46%) | 548 (40%) |

### 3.2 Aggregate Statistics

| Model | Mean | Median | Max | Points <25% |
|-------|------|--------|-----|-------------|
| Our Lindqvist (buoyancy) | 76.1% | 27.5% | 314.3% | 4/10 (40%) |
| Lit. Lindqvist (Hu&Zhou) | 28.3% | 25.3% | 67.3% | 5/10 (50%) |
| Our Keinonen | 428.6% | 355.5% | 989.3% | 0/10 (0%) |
| Lit. Keinonen (Hu&Zhou) | 38.6% | 11.4% | 183.3% | 7/10 (70%) |
| Our Riska | 56.9% | 29.9% | 231.4% | 3/10 (30%) |
| Lit. Riska (Hu&Zhou) | 51.4% | 29.2% | 215.3% | 4/10 (40%) |

### 3.3 Criterion Assessment (≥80% points < 25% error)

- **Our Lindqvist (buoyancy)**: 40% → ❌ FAIL
- **Lit. Lindqvist (Hu&Zhou)**: 50% → ❌ FAIL
- **Our Keinonen**: 0% → ❌ FAIL
- **Lit. Keinonen (Hu&Zhou)**: 70% → ❌ FAIL
- **Our Riska**: 30% → ❌ FAIL
- **Lit. Riska (Hu&Zhou)**: 40% → ❌ FAIL

### 3.4 Figures

![fig_uikku_comparison](artifacts/ice_resistance/fig_uikku_comparison.png)

![fig_uikku_scatter](artifacts/ice_resistance/fig_uikku_scatter.png)


---

## 4. Key Findings

### Critical Discovery: Literature Baselines Also Fail the Criterion

**Even the original published predictions from Hu & Zhou (2016) don't meet the 80% < 25% criterion:**
- Literature Lindqvist: 50% (5/10 points < 25%)
- Literature Keinonen: 70% (7/10 points < 25%)
- Literature Riska: 40% (4/10 points < 25%)

This means the criterion itself may be too strict for this dataset, or the dataset has
inherent challenges (e.g., points 401-403 with very low measured resistance are outliers).

### Our Implementations vs Literature

| Model | Our Performance | Literature Performance | Gap |
|-------|----------------|----------------------|-----|
| Lindqvist | 40% < 25% | 50% < 25% | -10% |
| Keinonen | 0% < 25% | 70% < 25% | -70% |
| Riska | 30% < 25% | 40% < 25% | -10% |

**Our implementations underperform the literature baselines.** Root causes:

1. **Ice property usage**: We use per-test measured σ_f/σ_c; literature likely used
   standard/effective values (lower than measured uniaxial strength).

2. **Keinonen overprediction**: Our Keinonen gives 3-10x overprediction because the
   formula R ∝ σ_f·h^1.5 is very sensitive to σ_f. Literature likely used σ_f ~ 500 kPa
   (standard) instead of measured 540-1029 kPa.

3. **Lindqvist discrepancy**: Our corrected Lindqvist is 1.2-2.5x higher than literature,
   suggesting literature used effective σ_c ~ 0.6-0.8x of measured values.

---

## 5. Applicability Domains

| Model | Ice Regime | h [m] | V [m/s] | L [m] | φ [deg] |
|-------|-----------|-------|---------|-------|---------|
| Lindqvist (1989) | level first-year | 0.3-1.5 | 0.5-6.0 | 50-250 | 15-60 |
| Keinonen (1983) | brash/repeated | 0.3-1.2 | 0.5-5.0 | 50-250 | 20-45 |
| Riska (1997) | level (DAS) | 0.3-1.5 | 0.5-6.0 | 50-250 | 15-60 |

---

## 6. Honest-FAIL Assessment

**Status**: ❌ FAIL — criterion ≥80% < 25% not met by any model (ours or literature)

**What was accomplished**:
- ✅ Three model families implemented (Lindqvist/Keinonen/Riska)
- ✅ MT Uikku data loaded and processed (10 points)
- ✅ Formulation error identified and corrected (buoyancy vs bending)
- ✅ Literature baselines compared (critical finding: they also fail)
- ✅ Error tables, figures, and applicability domains generated

**Root causes of underperformance**:
1. ❌ Using per-test measured ice properties (not standard/effective values)
2. ❌ Keinonen formula too sensitive to σ_f variations
3. ❌ No effective strength reduction factor applied
4. ❌ No model-scale corrections

**What remains for Round 2**:
- → Investigate standard ice property values used in literature
- → Implement effective strength reduction (σ_eff = k·σ_measured, k ~ 0.6-0.8)
- → Calibrate closure constants against Uikku data
- → Study whether 80% criterion is achievable for this dataset

---

## References

1. Hu, Z. & Zhou, L. (2016). Int. J. Naval Architecture and Ocean Engineering 8(2), 169-176.
2. Lindqvist, G. (1989). POAC'89, Luleå, Vol.2, pp.722-735.
3. Keinonen, A. (1983). POAC'83, Helsinki, pp.429-444.
4. Riska, K. (1997). POAC'97, Espoo, Vol.2, pp.303-314.

---
*Report generated by C1-ICE-RESIST analysis (m14881, final)*
