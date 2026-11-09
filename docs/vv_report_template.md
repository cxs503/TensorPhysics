# V&V Machine Gate Report Template

## 1. Case Introduction

### 1.1 Physical Background
[Describe the physical problem being solved]

### 1.2 Geometry and Parameters
| Parameter | Value | Unit | Uncertainty |
|-----------|-------|------|-------------|
| [param1]  | [val] | [unit] | [±unc] |
| [param2]  | [val] | [unit] | [±unc] |

### 1.3 Reference Data
- **Source**: [Analytical/Experimental/Reference CFD]
- **Reference value**: [value ± uncertainty]
- **Citation**: [Reference]

## 2. Methodology

### 2.1 Grid Convergence Study (Richardson Extrapolation)

**Grid sets used:**
| Grid | Cells/Elements | Characteristic size h | Solution f_h |
|------|----------------|----------------------|--------------|
| Coarse | [N_c] | [h_c] | [f_c] |
| Medium | [N_m] | [h_m] | [f_m] |
| Fine   | [N_f] | [h_f] | [f_f] |

**Refinement ratios:**
- r_coarse = h_medium / h_coarse = [value]
- r_fine = h_fine / h_medium = [value]

**Richardson extrapolation results:**
- Observed order of convergence: p = [value]
- Extrapolated value (zero mesh spacing): f_ext = [value]
- GCI_fine = [value] %
- GCI_coarse = [value] %
- GCI ratio: [value] (should be ≈1.0 if in asymptotic range)
- Asymptotic range: [Yes/No]
- **Convergence status**: [PASS/FAIL/UNRESOLVED]

### 2.2 ASME V&V 20 Uncertainty Quantification

**Numerical uncertainty (U_num):**
- From GCI: [value] %
- From extrapolation error: [value] %
- **U_num = [value] %**

**Input parameter uncertainty (U_input):**
| Parameter | Value | Uncertainty | Relative (%) | Sensitivity | Contribution (%) |
|-----------|-------|-------------|--------------|-------------|------------------|
| [param1]  | [val] | [unc]       | [rel%]       | [sens]      | [contrib]        |
| [param2]  | [val] | [unc]       | [rel%]       | [sens]      | [contrib]        |
| **Total U_input** | | | | | **[value] %** |

**Model-form uncertainty (U_model):**
- Comparison error: δ = |f_computed - f_experimental| / |f_experimental| × 100 = [value] %
- Residual after accounting for U_num and U_input: U_model = [value] %

**Total validation uncertainty:**
```
U_val = sqrt(U_num² + U_input² + U_model²)
U_val = sqrt([U_num]² + [U_input]² + [U_model]²)
U_val = [value] %
```

### 2.3 Cross-Solver Comparison (if applicable)

**Solvers compared:**
- Solver A: [name, version, discretization]
- Solver B: [name, version, discretization]
- Solver C: [name, version, discretization]

**Comparison results:**
| Quantity | Solver A | Solver B | Solver C | Reference | Max deviation (%) | Status |
|----------|----------|----------|----------|-----------|-------------------|--------|
| [qty1]   | [val]    | [val]    | [val]    | [ref]     | [dev%]            | [PASS/FAIL] |
| [qty2]   | [val]    | [val]    | [val]    | [ref]     | [dev%]            | [PASS/FAIL] |

**Pairwise deviations:**
- A vs B: [value] %
- A vs C: [value] %
- B vs C: [value] %

## 3. Results

### 3.1 Computed Values
| Quantity | Computed | Reference | Error (%) | U_val (%) | Validated? |
|----------|----------|-----------|-----------|-----------|------------|
| [qty1]   | [val]    | [ref]     | [err%]    | [u_val%]  | [Yes/No]   |
| [qty2]   | [val]    | [ref]     | [err%]    | [u_val%]  | [Yes/No]   |

### 3.2 Validation Assessment
- **Number of quantities validated**: [n_validated] / [n_total]
- **Maximum validation uncertainty**: [max_u_val] %
- **Acceptable uncertainty threshold**: [threshold] %
- **Overall validation status**: [VALIDATED / PARTIALLY_VALIDATED / NOT_VALIDATED]

### 3.3 Figures

**Figure 1: Grid convergence plot**
[Insert plot showing f_h vs h with extrapolated value]

**Figure 2: Comparison with reference data**
[Insert plot comparing computed vs reference values with uncertainty bands]

**Figure 3: Cross-solver comparison**
[Insert plot showing solver outputs with deviation bands]

## 4. Discussion

### 4.1 Key Findings
1. [Finding 1]
2. [Finding 2]
3. [Finding 3]

### 4.2 Limitations and Caveats
- [Limitation 1]
- [Limitation 2]
- [Applicability range]

### 4.3 Recommendations
- [Recommendation 1]
- [Recommendation 2]

## 5. Error Assessment

### 5.1 Quantitative Error Summary
| Quantity | Error (%) | Threshold (%) | Pass? |
|----------|-----------|---------------|-------|
| [qty1]   | [err%]    | [thresh%]     | [✓/✗] |
| [qty2]   | [err%]    | [thresh%]     | [✓/✗] |

**Answer: Are all errors < 3%?** [Yes/No/Not applicable - explain]

### 5.2 Audit Trail
- **Grid convergence analysis**: [PASS/FAIL] - p=[value], GCI=[value]%
- **Uncertainty quantification**: U_val=[value]% < [threshold]% threshold
- **Cross-solver agreement**: Max deviation=[value]% < 5% threshold
- **Overall V&V status**: [VALIDATED/PARTIALLY_VALIDATED/NOT_VALIDATED]

## 6. Reproducibility

### 6.1 Software Versions
- TensorPhysics: [version]
- Solver A: [version]
- Solver B: [version]
- Python: [version]
- NumPy: [version]

### 6.2 Input Files
- Grid files: [paths]
- Configuration: [paths]
- Boundary conditions: [paths]

### 6.3 Execution
```bash
# Commands to reproduce results
[command 1]
[command 2]
```

### 6.4 Data Availability
- Raw simulation data: [location]
- Post-processing scripts: [location]
- This report: [location]

---

**Report generated**: [timestamp]
**Report author**: [agent/system]
**Review status**: [Draft/Reviewed/Approved]
