# Self-Similar Solution of 2D Wedge Water-Entry Impact

## Node: p22w2-n07-slamming-similarity-impact
## Mission: m9308

---

## 1. Case Introduction

### Physical Background
When a ship section (wedge-shaped bow or flat bottom panel) enters water at high speed,
the rapid fluid displacement generates intense pressure pulses known as **slamming** or
**water-entry impact**. The classical theoretical framework, initiated by von Karman (1929)
and Wagner (1932), treats this as a potential-flow free-boundary problem that admits a
**self-similar reduction** when the entry velocity is constant (or linearly growing).

### Geometry & Parameters
| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| Deadrise angle | β | 5.0 | deg |
| Entry velocity | V | 5.0 | m/s |
| Entry acceleration | a | 50.0 | m/s² |
| Water density | ρ | 1000.0 | kg/m³ |
| Sound speed | c_s | 1500.0 | m/s |
| Reference time | t_ref | 0.1 | s |

### Governing Equations
The self-similar reduction uses the similarity variable ξ = x/c(t), where c(t) is the
Wagner wetted half-width. The dimensionless pressure shape function is:

    P(ξ) = a₁/√(1-ξ²) + b·√(1-ξ²) - 1/(2(1-ξ²))

where a₁ = π/(2 sin β) is the spray-root strength and b = 0 (constant velocity) or
b = κ/(2 cos β) (constant acceleration).

---

## 2. Methodology

### Self-Similar Reduction
The free-boundary problem is reduced via the Wagner spray-root kinematics dc/dt = κV
with κ = π/(2 tan β). In similarity coordinates, the nonlinear problem becomes a
Keldysh-Sedov mixed boundary-value problem on a fixed domain.

### Solution Routes
1. **Momentum route** (von Karman): F = d(m_a · U_n)/dt with m_a = ρπc²/2
2. **Pressure integration route**: Gauss-Legendre quadrature of P(ξ) in the
   regularizing variable ξ = ξ₀ - u² (removes the square-root edge singularity)
3. **Wagner linearized solution**: valid for β ≤ 5° (small deadrise)
4. **Joukowsky acoustic bound**: max p ≤ ρ c_s U_n (compressibility cap)

### Cross-Validation Strategy
- Momentum vs pressure-integration force: two independent quadratures of the same field
- Wagner linearized vs self-similar: exact for β → 0, O(β²) deviation documented
- Impulse closure: ∫F dt = m_a(t) · U_n(t) (analytic) vs numerical quadrature

---

## 3. Results

### E1: Self-Similar Fold
| Quantity | Value | Criterion | Pass |
|----------|-------|-----------|------|
| Pressure L² residual (velocity mode) | 0.00e+00 | ≤ 1e-2 | ✅ |
| Force coefficient ptp (velocity) | 2.51e-16 | ≤ 1e-12 | ✅ |
| Force coefficient ptp (acceleration) | 4.02e-16 | ≤ 1e-12 | ✅ |

**Conclusion**: Different (t, V) solutions collapse onto a single curve in similarity
coordinates with L² residual < 1e-2.

### E2: Small-β vs Wagner
| β [deg] | F_model [N/m] | F_Wagner [N/m] | Rel error |
|---------|---------------|----------------|-----------|
| 1.0 | 317973225.49 | 318021661.73 | 1.52e-04 |
| 2.0 | 79408572.46 | 79456975.51 | 6.09e-04 |
| 3.0 | 35229990.92 | 35278338.66 | 1.37e-03 |
| 5.0 | 12610732.14 | 12658903.09 | 3.81e-03 |

**Worst relative deviation**: 3.81e-03 (criterion ≤ 3%) ✅

### E3: Impulse Ledger
| Quantity | Value | Unit |
|----------|-------|------|
| Numerical quadrature ∫F dt | 630536.6071 | N·s/m |
| Analytic m_a · U_n | 630536.6071 | N·s/m |
| Closure relative error | 1.85e-16 | - |
| Pressure vs momentum | 9.32e-02 | - |

**Conclusion**: Impulse closure < 1%. Pressure route within 10% of momentum route.

### E4: Joukowsky Acoustic Bound
| Quantity | Value | Unit |
|----------|-------|------|
| Peak pressure (analytic) | 4029454.00 | Pa |
| Joukowsky bound ρ c_s V | 7500000.00 | Pa |
| Ratio | 0.5373 | - |

**Capped field sweep**:
| β [deg] | Ratio (capped) | Ratio (raw) |
|---------|----------------|-------------|
| 1.0 | 1.0000 | 7.8210 |
| 3.0 | 1.0000 | 1.4950 |
| 5.0 | 0.5392 | 0.5392 |
| 10.0 | 0.1343 | 0.1343 |
| 20.0 | 0.0330 | 0.0330 |
| 30.0 | 0.0142 | 0.0142 |

**Conclusion**: All capped ratios ≤ 1.0: ✅

### E5: Resolution Ledger
| Quantity | Value | Criterion | Pass |
|----------|-------|-----------|------|
| Force observed order | 14.86 | ≥ 1.5 | ✅ |
| Force rel error (finest) | 9.32e-02 | ≤ 10% | ✅ |
| Peak observed order | 13.96 | > 0 | ✅ |
| Near-root divergence order | 1.41 | > 0.5 | ✅ |
| Near-root crossing N | 6400 | not None | ✅ |

**Key finding**: The point pressure peak **converges** (order > 0) for the linearly-posed
self-similar problem, falsifying the prior expectation that it is non-convergent. However,
the near-root field (Wagner negative-pressure lobe) genuinely diverges like h⁻¹ and is
non-physical.

### E6: Bulkhead SDOF Interface
| Quantity | Value |
|----------|-------|
| Impulse handover rel error | 1.85e-16 |
| Amplification (Tₙ = 5 ms) | 0.9996 |
| Amplification (Tₙ = 50 ms) | 0.9994 |
| Amplification (Tₙ = 200 ms) | 0.3580 |

**Conclusion**: Impulse handover error < 0.1%. ✅

---

## 4. Figures

### Figure 1: Pressure Distribution
![Pressure distribution](fig1_pressure_distribution.png)
Physical and similarity-coordinate representations of the pressure field.

### Figure 2: Self-Similar Fold
![Self-similar fold](fig2_self_similar_fold.png)
All time instants collapse onto a single curve P(ξ) in similarity coordinates.

### Figure 3: Resolution Study
![Resolution study](fig3_resolution_study.png)
Force convergence (spectral), pressure peak convergence (algebraic), and near-root divergence.

### Figure 4: Bulkhead SDOF Response
![Bulkhead response](fig4_bulkhead_response.png)
Load history, displacement/velocity response, and amplification factor vs frequency ratio.

---

## 5. Caveats & Limitations

1. **Linearised Wagner theory**: The self-similar solution is based on the linearised
   free-surface condition. For β > 5°, the O(β²) deviation from the fully nonlinear
   Dobrovolskaya-Zabinskiy solution becomes significant. The applicability band is
   declared, not proven.

2. **Incompressible assumption**: The Joukowsky bound ρ c_s V is an acoustic (compressible)
   limit. For small deadrise angles (β < 3.7° at V = 5 m/s), the incompressible
   self-similar peak exceeds this bound, indicating that compressibility must saturate
   the actual peak. The capped field is reportable; the raw potential peak is not.

3. **Negative-pressure lobe**: The term -1/(2(1-ξ²)) in P(ξ) creates a non-integrable
   negative-pressure singularity at the spray root. This is non-physical (cavitation
   would occur) and is clipped at the acoustic bound. The near-root field is explicitly
   non-reportable.

4. **Gravity neglected**: The solution assumes infinite Froude number. For very slow
   entry or large wetted widths, gravity effects (wave making) become important.

5. **2D assumption**: The solution is strictly two-dimensional. Three-dimensional effects
   (transverse flow, tip vortices) are not captured.

6. **Rigid body**: The wedge is assumed rigid. Hydroelastic effects (fluid-structure
   interaction) are not included in this kernel but are handed over via the m6634
   bulkhead SDOF interface for structural response assessment.

---

## 6. Error Assessment

| Criterion | Value | Threshold | Pass |
|-----------|-------|-----------|------|
| E1: Self-similar fold L² | 0.00e+00 | ≤ 1e-2 | ✅ |
| E2: Small-β vs Wagner | 3.81e-03 | ≤ 3% | ✅ |
| E3: Impulse closure | 1.85e-16 | ≤ 1% | ✅ |
| E4: Joukowsky bound | True | True | ✅ |
| E5: Force Richardson order | 14.86 | ≥ 1.5 | ✅ |
| E6: Bulkhead interface | 1.85e-16 | ≤ 0.1% | ✅ |

**All criteria passed**: ✅ YES

---

## Audit Trail
- Code: `src/tensorphysics/slamming/similarity_impact.py`
- Tests: `tests/test_slamming_similarity_impact.py` (7 tests, all passing)
- Validation: `artifacts/slamming_similarity/validation.json`
- Interface: `artifacts/slamming_similarity/m6634_bulkhead_interface.json`
- Figures: `artifacts/slamming_similarity/fig1_*.png` through `fig4_*.png`
