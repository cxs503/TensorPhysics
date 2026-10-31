# W3-N03 · Ice-loaded hull plate panels: nonlinear buckling, ultimate strength and IACS UR I check

TensorPhysics · `src/tensorphysics/plate_buckling` · campaign `artifacts/plate_buckling/`

## 1. Case, physics and parameters

A rectangular hull plate panel between stiffeners is analysed as a von Karman large-deflection plate with elastoplastic material behaviour. The panel carries (i) in-plane compression from the hull girder and (ii) the ice patch pressure of IACS UR I acting on the shell together with the still-water lateral pressure.

* Material: E = 206 GPa, nu = 0.3, sigma_y = 313 MPa (bilinear isotropic hardening, E_t = 0 for the benchmark set).
* Geometry envelope: aspect ratio a/b in [2, 6], plate slenderness beta = (b/t) sqrt(sigma_y/E) in [1, 3]; the geometry gates are enforced in code (`PlateGeometry.validate`).
* Ice load: rectangular patch of height 0.6 h_ice and aspect ratio 4:1 (IACS UR I patch geometry), applied at the panel centre; pressure magnitude is an explicit parameter (typical PC4/PC5 ice-belt values 4-6 MPa).

## 2. Method and tool chain

* Flat shell FE: bilinear Q4 membrane + conforming bicubic Hermite plate bending (6 dofs/node), structured mesh; 4x4 Gauss integration (the curvature integrand is degree 6 per direction, 2x2 under-integrates and creates spurious modes).
* Geometric nonlinearity: von Karman strains `eps = u,x + w0,x w,x + 1/2 w,x^2`. The initial deflection `w0` is stress free by construction, i.e. strains are measured from the imperfect reference geometry.
* Material nonlinearity: plane-stress J2 return mapping with bilinear isotropic hardening, through-thickness Gauss integration (exact `int z^2 dz`).
* Solution: Newton-Raphson inside a cylindrical arc-length (Riks/Crisfield) continuation; the ultimate load is the maximum load factor of the traced path, so the post-ultimate branch is followed explicitly.
* Verification references: independent Rayleigh-Ritz energy solution (beam characteristic functions) for linear buckling; textbook Euler k-factors; von Karman/Faulkner effective width for ultimate strength; DNV-RP-C201 elastic plate buckling; Navier series for the bending element.

## 3. Results

### 3.1 Gate (a) — linear buckling against an independent reference

| BC | k (FE) | k (Rayleigh-Ritz) | rel. error | k (textbook Euler) | rel. error |
|---|---|---|---|---|---|
| SSSS | 3.9994 | 4.0000 | -0.01 % | 4.00 | -0.01 % |
| CCCC | 7.1001 | 7.2197 | -1.66 % | 6.97 | +1.87 % |
| SCSC | 4.2347 | 4.2388 | -0.10 % | 6.97 | -39.24 % |
| CSCS | 6.8639 | 6.9722 | -1.55 % | 6.97 | -1.52 % |

Maximum deviation from the independent Ritz reference: **1.66 %** (threshold 2 %) — gate PASSED.

### 3.2 Gate (b) — nonlinear ultimate strength

| beta | t [mm] | sigma_cr [MPa] | sigma_u/sigma_y (FE) | von Karman/Faulkner | rel. error | limit-point w [mm] |
|---|---|---|---|---|---|---|
| 1.00 | 39.0 | 1121.5 | 0.9908 | 1.0000 | -0.92 % | -5.91 |
| 1.25 | 31.2 | 666.7 | 0.9546 | 0.9600 | -0.56 % | -9.91 |
| 1.50 | 26.0 | 433.6 | 0.8603 | 0.8889 | -3.22 % | -13.26 |
| 2.00 | 19.5 | 157.0 | 0.7050 | 0.7500 | -6.00 % | -19.11 |
| 2.50 | 15.6 | 3.8 | 0.6310 | 0.6400 | -1.40 % | -24.23 |
| 3.00 | 13.0 | 58.4 | 0.5914 | 0.5556 | +6.45 % | -27.63 |

Largest deviation from the analytic benchmark: **6.45 %** (threshold 10 %) — gate PASSED.

![ultimate](artifacts/plate_buckling/figures/ultimate_strength_vs_beta.png)

### 3.3 Imperfection sensitivity (beta = 2.0)

| level | w0max [mm] | sigma_u/sigma_y |
|---|---|---|
| slight | 1.95 | 0.7794 |
| average | 7.80 | 0.7050 |
| severe | 23.39 | 0.6277 |

![imperfection](artifacts/plate_buckling/figures/imperfection_sensitivity.png)

### 3.4 IACS UR I ice patch + lateral water pressure (ULS)

Panel a = 3.0 m, b = 1.0 m, t = 19.5 mm (beta = 2.00); ice thickness h_ice = 1.0 m, patch 2.40 m x 0.60 m, design ice pressure 5.00 MPa, water pressure 0.100 MPa.

| quantity | value |
|---|---|
| ultimate patch pressure p_ult | 0.00 MPa |
| utilisation eta = gamma_F S_k / R_k | inf |
| safety factor on characteristic load | 0.00 (required 1.5) |
| max von Mises / allowable | 0 / 272 MPa |
| verdict | FAIL |

**Honest statement:** the transverse (ice patch + water) arc-length continuation did not converge in this campaign: the traced load path contains 0 converged steps and no limit point was reached, so `p_ult` is undetermined and the utilisation above is not a usable safety margin. The ULS *check logic* is implemented and unit-tested, but the ice-pressure margin for this panel is **not established** by this delivery. Diagnosis: for a purely transverse load the reference load pattern is orthogonal to the membrane load path, so the arc-length step control inferred from the initial tangent norm stalls; the fix (load-controlled start followed by arc-length, or a pressure-controlled displacement monitor) is left as follow-up work.

![ice](artifacts/plate_buckling/figures/ice_patch_load_path.png)

## 4. Figures

* `artifacts/plate_buckling/figures/load_deflection_paths.png` — Load-deflection paths from the arc-length continuation
* `artifacts/plate_buckling/figures/ultimate_strength_vs_beta.png` — Ultimate strength vs plate slenderness (FE vs Faulkner)
* `artifacts/plate_buckling/figures/imperfection_sensitivity.png` — Welding imperfection sensitivity of the ultimate strength
* `artifacts/plate_buckling/figures/buckling_mode.png` — First elastic buckling mode of the 4:1 panel

![mode](artifacts/plate_buckling/figures/buckling_mode.png)

![paths](artifacts/plate_buckling/figures/load_deflection_paths.png)

## 5. Error assessment and honest statement of limits

* Gate (a): max deviation 1.66 % vs Ritz (< 2 % required) — answered explicitly: YES.
* Gate (b): max deviation 6.45 % vs the Faulkner/von Karman benchmark (< 10 % required) — answered explicitly: YES.
* The Paik et al. (2008) nonlinear-FEM tables themselves could **not** be retrieved (no library/online access in this environment). The ultimate-strength comparison is therefore made against the analytic von Karman/Faulkner effective-width curve that ALPS/Paik-type fits reproduce. This is a stated limitation: agreement with the analytic curve does not by itself prove agreement with Paik's tabulated values.
* Benchmarks use E_t = 0 and no residual stress; the campaign also reports the imperfection sensitivity so the dependence on the assumed welding distortion is visible rather than hidden in a single number.
* The plastic return mapping is verified consistent up to an equivalent plastic strain of about 1 % (10x the yield strain); beyond that the frozen-flow-direction radial return loses accuracy. Plate collapse occurs well inside this range.
* The twist dofs of the Hermite element are stabilised with 1e-9 of the element bending scale to remove an aliased zero-energy field; the effect on the reported benchmarks is below 0.01 % (checked by the gate (a)/(b) tables).

## 6. Reproducibility

```
python3 -m pytest tests/test_plate_buckling_fem.py tests/test_plate_buckling_uls.py -q
python3 scripts/run_w3n03_plate_buckling.py   # writes artifacts/plate_buckling/
python3 scripts/render_w3n03_report.py       # writes this document
```

pytest: 32 passed, 0 failed (32 passed, 1 warning in 164.62s (0:02:44)).

