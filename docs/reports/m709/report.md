# SIR-PC2|F02: Plate–Water Added Mass Integration to FEM Mass Matrix

## 1. Case Introduction

### Physical Background
A floating ice sheet vibrating in contact with water experiences a **hydrodynamic added-mass effect**: the surrounding water must be accelerated along with the plate, effectively increasing its inertia. This is a fundamental fluid–structure interaction (FSI) phenomenon in ice mechanics.

For a plate mode with wavenumber *k*, the added mass per unit area from linear potential-flow theory is:

$$m_a(k) = \frac{\rho_w}{k} \tanh(kH)$$

where ρ_w is water density and H is water depth (H→∞ for deep water).

### Geometry & Parameters

| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| Plate length | a | 100.0 | m |
| Plate width | b | 100.0 | m |
| Ice thickness | h | 2.0 | m |
| Young's modulus | E | 5.0×10⁹ | Pa |
| Poisson's ratio | ν | 0.33 | — |
| Ice density | ρ_i | 917.0 | kg/m³ |
| Water density | ρ_w | 1025.0 | kg/m³ |
| Gravity | g | 9.81 | m/s² |
| Water depth | H | ∞ (deep) | m |

Derived quantities:
- Flexural rigidity: D = 3.74×10⁹ N·m
- Winkler modulus: k_w = 10055 Pa/m
- Fundamental wavenumber: k₁ = 0.0444 1/m
- Structural mass/area: 1834 kg/m²
- **Added mass/area: 23071 kg/m²** (ratio = 12.6× structural)

## 2. Methodology

### Approach
1. **Added mass formula**: Implemented `water_added_mass_per_area(k, ρ_w, H)` from potential-flow theory
2. **FEM mass matrix assembly**: Bilinear quadrilateral elements with consistent mass formulation
3. **Frequency computation**: Rayleigh quotient with analytical modal stiffness and FEM modal mass

### Key Functions
- `water_added_mass_per_area(k, ρ_w, H)` — exact added mass per unit area
- `fem_mass_matrix_with_added_mass(params, nx, ny)` — assembles M_total = M_struct + M_added
- `fem_natural_frequency(params, nx, ny)` — computes ω from Rayleigh quotient

### Tool Chain
- Python 3.11 + NumPy + SciPy
- pytest for verification
- Matplotlib for figures

## 3. Results

### Key Numbers

| Quantity | Value | Unit |
|----------|-------|------|
| Reference frequency (ω_ref) | 0.9945 | rad/s |
| FEM frequency (ω_fem) | 0.9986 | rad/s |
| Reference frequency (f_ref) | 0.1583 | Hz |
| FEM frequency (f_fem) | 0.1589 | Hz |
| **Frequency error** | **0.41%** | — |
| Min mass matrix eigenvalue | 7.09×10⁴ | kg |
| Added mass ratio (m_a/m_s) | 12.58 | — |

### Acceptance Criteria

| Criterion | Value | Threshold | Status |
|-----------|-------|-----------|--------|
| natural_frequency_error_vs_fluid_solid | 0.0041 | < 0.05 | ✅ PASS |
| mass_matrix_positive_definite | True | True | ✅ PASS |
| pytest_test_added_mass_integration | PASS | PASS | ✅ PASS |

## 4. Figures

### Figure 1: Added Mass vs Wavenumber
![Added mass vs wavenumber](artifacts/added_mass_vs_wavenumber.png)

Shows the added mass per unit area as a function of wavenumber for deep water and finite depth (H=200 m). The vertical line marks the fundamental wavenumber k₁ for the 100×100 m plate.

### Figure 2: Frequency Comparison
![Frequency comparison](artifacts/frequency_comparison.png)

Natural frequency vs ice thickness, with and without added mass. The added mass effect is most pronounced for thin ice (ratio > 50× for h < 0.5 m).

### Figure 3: FEM Convergence
![FEM convergence](artifacts/fem_convergence.png)

Relative frequency error vs grid resolution. Error drops below 5% at n=11 and converges to ~0.4% at n=31.

### Figure 4: Mass Matrix Eigenvalues
![Mass matrix eigenvalues](artifacts/mass_matrix_eigenvalues.png)

Eigenvalue spectrum of the mass matrix with and without added mass. All eigenvalues are positive, confirming positive definiteness.

## 5. Notes & Limitations

### Key Insights
1. **Added mass dominates**: For typical sea ice (h=2 m), the added mass is 12.6× the structural mass. This dramatically reduces the natural frequency.
2. **Thin ice effect**: For thin ice (h < 0.5 m), the added mass ratio exceeds 50×, making FSI essential for accurate frequency prediction.
3. **Convergence**: The FEM frequency converges to the analytical reference with mesh refinement, reaching <1% error at n=21.

### Limitations
1. **Single-mode approximation**: The current implementation uses the fundamental mode shape. Higher modes require separate wavenumbers.
2. **Uniform added mass**: The added mass is assumed constant across the plate (evaluated at k₁). For non-uniform modes, spatial variation should be considered.
3. **Simply supported BCs**: The analytical stiffness assumes simply supported boundaries. Other BCs (clamped, free) require different treatment.
4. **Inviscid water**: The added mass formula assumes inviscid, irrotational flow. Viscous damping is not included.

### Applicable Range
- Plate aspect ratio: 0.1 < a/b < 10
- Ice thickness: 0.1 m < h < 10 m
- Water depth: H > 10 m (deep water) or finite depth with tanh correction
- Frequency range: 0.01–1 Hz (fundamental mode)

## 6. Error Assessment

| Quantity | FEM Value | Reference | Error | < 3%? |
|----------|-----------|-----------|-------|-------|
| Natural frequency (ω) | 0.9986 rad/s | 0.9945 rad/s | 0.41% | ✅ Yes |
| Mass matrix PD | True | True | — | ✅ Yes |
| pytest | 21/21 PASS | — | — | ✅ Yes |

**All key quantities are within the 3% tolerance.** The frequency error of 0.41% is well below the 5% acceptance threshold and the 3% paper-quality threshold.

## Audit Chain

- **Upstream**: m667 (SIR-PC2|F01) — Winkler plate implementation
- **This work**: m709 (SIR-PC2|F02) — Added mass integration
- **Physical domain**: fluid_structure_interaction
- **Campaign**: Ice Navigation (m611 master plan)
- **Evidence level**: REAL (analytical formula + FEM assembly + verified against reference)
