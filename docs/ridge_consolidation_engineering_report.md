# Ice Ridge Consolidation Thermodynamics: Engineering Model
## Campaign Node P46W3-N05

**Mission ID**: 9153  
**Date**: 2026-11-10  
**Repository**: TensorPhysics  
**Branch**: agent/m9153

---

## 1. Case Introduction

### Physical Background

Ice ridges are formed when ice floes collide and pile up, creating a sail above the waterline and a keel below. The keel is a porous rubble body with porosity φ ∈ [0.3, 0.5], where the pore space is filled with seawater. In Arctic winter conditions, the pore water refreezes from the cold surfaces inward, transforming the loose rubble into a bonded "consolidated layer" whose strength increases with time.

This consolidation process has critical implications for ship navigation in ice-infested waters:
- **Unconsolidated ridges** (C ≈ 0): The ship only needs to push aside loose rubble
- **Consolidated ridges** (C → 1): The ship must break through a solid ice layer, requiring significantly higher resistance

### Geometric Configuration

The simplified engineering model considers:
- **Keel depth**: h_keel ∈ [5, 20] m (typical Arctic ridges)
- **Porosity**: φ ∈ [0.3, 0.5] (measured range, Timco & Weeks 2010)
- **Air temperature**: T_air ∈ [-40, -10]°C (Arctic winter range)
- **Consolidation time**: t ∈ [0, 90] days (first winter)

### Parameter Table

| Parameter | Symbol | Value | Unit | Source |
|-----------|--------|-------|------|--------|
| Reference time constant | τ₀ | 1.0×10⁶ | s | Engineering fit (~11.6 days at -10°C) |
| Activation energy | Eₐ | 50×10³ | J/mol | Ice refreezing kinetics |
| Reference temperature | T_ref | -10 | °C | Typical Arctic condition |
| Reference ice strength | σ₀ | 5.0 | MPa | Timco & Weeks (2010) |
| Power-law exponent | m | 1.75 | - | Leppäranta & Hakala (1992) |
| Resistance coupling | k | 0.7 | - | Timco & Burden (1994) |
| Reference resistance | R₀ | 1000 | kN | Typical 8m keel |

---

## 2. Methodology

### 2.1 Consolidation Degree: Exponential Law

The consolidation degree evolves according to:

$$C(t) = 1 - \exp\left(-\frac{t}{\tau(T)}\right)$$

where the time constant τ(T) follows an Arrhenius-like relation:

$$\tau(T) = \tau_0 \cdot \exp\left(-\frac{E_a}{R}\left(\frac{1}{T_K} - \frac{1}{T_{ref,K}}\right)\right)$$

**Physical basis**: This is the solution of a lumped-capacitance model where the consolidation rate is proportional to the remaining unfrozen fraction: dC/dt = (1-C)/τ. The negative sign in the Arrhenius exponent ensures that colder temperatures give smaller τ (faster consolidation), consistent with the physics of heat extraction driving refreezing.

**Implementation**: `src/tensorphysics/ice_mechanics/ridge_consolidation_engineering.py`
- Function: `consolidation_degree_exponential(t, T, tau_0, E_a, T_ref)`
- Function: `arrhenius_time_constant(T, tau_0, E_a, T_ref)`

### 2.2 Strength-Consolidation: Power-Law Relation

The consolidated ice strength follows:

$$\sigma_c = \sigma_0 \cdot C^m$$

where m ∈ [1.5, 2.0] is the power-law exponent.

**Physical basis**: The power-law relation captures the fact that strength grows faster than linearly with consolidation degree. For m=1, strength is proportional to the load-bearing area fraction. For m>1, additional strengthening occurs due to bond formation and stress redistribution. The exponent m ∈ [1.5, 2.0] is based on experimental observations of partially consolidated ice ridges (Leppäranta & Hakala 1992).

**Implementation**:
- Function: `strength_power_law(C, sigma_0, m)`

### 2.3 Resistance-Consolidation: Linear Coupling

The ridge resistance increases with consolidation:

$$R_{ice} = R_0 \cdot (1 + k \cdot C)$$

where k ≥ 0.5 is the linear coupling coefficient.

**Physical basis**: At C=0 (unconsolidated), R_ice = R₀. At C=1 (fully consolidated), R_ice = R₀·(1+k). The constraint k ≥ 0.5 ensures that a fully consolidated ridge has at least 50% higher resistance than the unconsolidated baseline, consistent with field observations (Timco & Burden 1994).

**Implementation**:
- Function: `resistance_linear_coupling(C, R_0, k)`

### 2.4 Heat Conduction Verification

For steady-state conduction without heat generation:

$$\frac{d^2T}{dx^2} = 0$$

The analytical solution is linear: T(x) = T_left + (T_right - T_left)·x/L. The numerical residual is computed using central finite differences and normalized by the temperature scale.

**Implementation**:
- Function: `heat_conduction_residual_1d(T_left, T_right, L, k, n_points)`

### 2.5 Tool Chain

- **Language**: Python 3.11.6
- **Numerical libraries**: NumPy 2.4.6, Matplotlib 3.10.8
- **Testing framework**: pytest
- **Platform**: Linux-6.6.52 (loongarch64)

---

## 3. Computational Results

### 3.1 Acceptance Criteria Summary

All five acceptance criteria pass:

| Criterion | Description | Threshold | Achieved | Status |
|-----------|-------------|-----------|----------|--------|
| (a) | Exponential consolidation with Arrhenius τ | τ decreases with colder T | τ(-30°C)/τ(-20°C) = 0.376 | ✓ PASS |
| (b) | Power-law strength σ_c = σ₀·C^m | m ∈ [1.5, 2.0] | m = 1.75 (default) | ✓ PASS |
| (c) | Linear resistance R_ice = R₀·(1+k·C) | k ≥ 0.5 | k = 0.7 (default) | ✓ PASS |
| (d) | Heat conduction residual | ≤ 1% | 2.3×10⁻¹² | ✓ PASS |
| (e) | Negative control: T→T_melt→C→0→σ_c→0 | C=0, σ_c=0 at T_melt | C=0.0, σ_c=0.0 | ✓ PASS |

### 3.2 Consolidation Degree Evolution

**Key results** (from `runs/ridge_thermo/results.json`):

| Temperature | Time constant τ [days] | C(30 days) | C(60 days) |
|-------------|------------------------|------------|------------|
| -10°C | 11.57 | 0.925 | 0.994 |
| -20°C | 4.69 | 0.998 | 1.000 |
| -30°C | 1.77 | 1.000 | 1.000 |
| -40°C | 0.67 | 1.000 | 1.000 |

**Observations**:
- At -20°C, consolidation reaches 99.8% in 30 days
- Colder temperatures dramatically accelerate consolidation (τ decreases by factor 2.65 from -20°C to -30°C)
- The exponential law captures the rapid initial consolidation followed by asymptotic approach to C=1

### 3.3 Strength Evolution

**Power-law strength at C=0.5** (from verification suite):

| Exponent m | σ_c [MPa] | σ_c/σ₀ |
|------------|-----------|--------|
| 1.5 | 1.77 | 0.354 |
| 1.75 | 1.49 | 0.297 |
| 2.0 | 1.25 | 0.250 |

**Observations**:
- Higher exponent m → lower strength at partial consolidation (super-linear growth)
- At C=1, all curves converge to σ₀ = 5.0 MPa
- The exponent m=1.75 provides a good balance between physical realism and computational simplicity

### 3.4 Resistance Evolution

**Linear resistance coupling at C=1** (from verification suite):

| Coupling k | R_ice [kN] | R_ice/R₀ |
|------------|------------|----------|
| 0.5 | 1500 | 1.50 |
| 0.7 | 1700 | 1.70 |
| 1.0 | 2000 | 2.00 |

**Observations**:
- Fully consolidated ridge (C=1) has 50-100% higher resistance than unconsolidated baseline
- The linear model provides a simple yet physically reasonable coupling
- The constraint k ≥ 0.5 ensures minimum 50% increase, consistent with field data

### 3.5 Heat Conduction Verification

**Residual vs grid resolution** (from verification suite):

| Grid points | Normalized residual |
|-------------|---------------------|
| 10 | 3.5×10⁻¹⁰ |
| 100 | 2.3×10⁻¹² |
| 1000 | 3.5×10⁻¹⁰ |

**Observations**:
- For linear temperature profile, central differences are exact (machine precision)
- Residual is well below the 1% threshold at all resolutions
- The verification confirms the numerical implementation is correct

---

## 4. Results Figures

### Figure 1: Consolidation Degree Evolution

**Caption**: Exponential consolidation law C(t) = 1 - exp(-t/τ(T)) for temperatures ranging from -10°C to -40°C. Colder temperatures lead to faster consolidation (smaller τ). At -20°C, consolidation reaches 99.8% in 30 days.

**File**: `runs/ridge_thermo/figures/fig1_consolidation_degree.png`

### Figure 2: Arrhenius Time Constant

**Caption**: Arrhenius time constant τ(T) = τ₀·exp(-Eₐ/R·(1/T - 1/T_ref)) showing the strong temperature dependence. At the reference temperature T_ref = -10°C, τ = 11.6 days. Colder temperatures give exponentially smaller τ (faster consolidation).

**File**: `runs/ridge_thermo/figures/fig2_arrhenius_time_constant.png`

### Figure 3: Strength-Power Law

**Caption**: Power-law strength relation σ_c = σ₀·C^m for exponents m ∈ [1.5, 2.0]. Higher exponent leads to lower strength at partial consolidation (super-linear growth). All curves converge to σ₀ = 5.0 MPa at C=1.

**File**: `runs/ridge_thermo/figures/fig3_strength_power_law.png`

### Figure 4: Resistance Coupling

**Caption**: Linear resistance coupling R_ice = R₀·(1+k·C) for coupling coefficients k ∈ [0.5, 1.0]. At C=0 (unconsolidated), R_ice = R₀ = 1000 kN. At C=1 (fully consolidated), R_ice increases by 50-100%.

**File**: `runs/ridge_thermo/figures/fig4_resistance_coupling.png`

### Figure 5: Coupled Evolution

**Caption**: Coupled evolution of consolidation degree C, compressive strength σ_c, and ridge resistance R_ice over 90 days at T = -20°C. The three quantities are linked through the consolidation degree C(t).

**File**: `runs/ridge_thermo/figures/fig5_coupled_evolution.png`

### Figure 6: Heat Conduction Verification

**Caption**: 1D steady-state heat conduction verification showing the normalized residual |d²T/dx²|/(ΔT/L²) vs number of grid points. For a linear temperature profile, central differences are exact (machine precision), well below the 1% threshold.

**File**: `runs/ridge_thermo/figures/fig6_heat_conduction_verification.png`

---

## 5. Discussion and Limitations

### 5.1 Model Validity

The simplified engineering model is valid for:
- **Temperature range**: T ∈ [-40, -1.8]°C (Arctic winter conditions)
- **Porosity range**: φ ∈ [0.3, 0.5] (measured keel porosity)
- **Time scale**: t ∈ [0, 90] days (first winter consolidation)
- **Consolidation degree**: C ∈ [0, 1] (physical range)

### 5.2 Comparison with Rigorous Stefan Model

The engineering model is a **lumped-capacitance approximation** to the rigorous Stefan-based model implemented in `src/tensorphysics/ice_mechanics/ridge_consolidation.py`. The key differences are:

| Aspect | Engineering Model | Stefan Model |
|--------|-------------------|--------------|
| Consolidation law | Exponential C(t) = 1-exp(-t/τ) | Stefan similarity h_c = 2λ√(κt) |
| Temperature dependence | Arrhenius τ(T) | Explicit heat conduction |
| Computational cost | O(1) per evaluation | O(√t) per evaluation |
| Physical fidelity | Lumped approximation | Distributed parameter |
| Use case | System-level simulations | Detailed process studies |

The engineering model sacrifices some physical fidelity for computational efficiency, making it suitable for:
- Rapid parametric studies
- System-level simulations (e.g., ship routing in ice)
- Real-time decision support

### 5.3 Known Limitations

1. **Lumped capacitance**: The exponential law assumes uniform temperature within the keel, which is not physically accurate. The rigorous Stefan model accounts for the temperature gradient.

2. **Constant properties**: The model assumes constant thermophysical properties (k, ρ, c), whereas in reality these depend on temperature and salinity.

3. **No snow cover**: The model does not account for the insulating effect of snow on the ridge surface, which can significantly delay consolidation.

4. **No brine pockets**: The model does not explicitly track brine pocket formation and drainage, which affects the strength evolution.

5. **Empirical closures**: The power-law exponent m and coupling coefficient k are empirical and may need calibration for specific ridge types or geographic regions.

### 5.4 Applicability Domain

The model is **NOT** applicable for:
- Summer conditions (T > -1.8°C)
- Multi-year ridges (different consolidation mechanics)
- Ridges with significant snow cover (requires modified τ)
- Rid
