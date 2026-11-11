# Plate/shell static verification face (campaign node p405w1-n05)

Module `tensorphysics.plate_shell` | driver `verify_fem_plate_shell.py` |
payload `artifacts/plate_shell/verify_fem_plate_shell.json`

Verdict: **PASS** (7/7 gates).

## 1. Cases, geometry and parameters

Two static plate/shell cases in thin-structure ship proportions, each with an **independent closed-form anchor** (no reference value is taken from the finite element itself).

### 1a. Uniformly loaded simply supported square plate

The classical Panel/plate-bending benchmark: all four edges simply supported, uniform pressure `q`, Kirchhoff thin-plate theory. Anchor = Navier double Fourier series.

| parameter | value |
|---|---|
| a | 1 |
| b | 1 |
| t | 0.01 |
| E | 2.1e+11 |
| nu | 0.3 |
| q | 10000 |
| a/t (thin-plate validity) | 100 |
| D = E t^3/(12(1-nu^2)) | 19230.7692 N m |

### 1b. Thin cylindrical shell under internal pressure

Open-ended cylinder, `R/t = 100`, internal pressure `p`; one end clamped against radial displacement and rotation, the other held at the membrane expansion. Anchor = Donnell axisymmetric closed form with the `exp(-beta x)` bending boundary layer.

| parameter | value |
|---|---|
| R | 1 |
| L | 0.8 |
| t | 0.01 |
| E | 2.1e+11 |
| nu | 0.3 |
| p | 1e+06 |
| R/t | 100 |
| beta = (E t/(4 D R^2))^(1/4) | 12.854070 1/m |
| 1/beta | 0.077796 m |
| sqrt(R t) | 0.100000 m |
| membrane sigma_theta = pR/t | 1.000000e+08 Pa |
| membrane w = pR^2/(Et) | 4.761905e-04 m |

## 2. How it is done

* **Anchor 1 (Navier series, `plate_shell/navier.py`).** `w = sum A_mn sin(m pi x/a) sin(n pi y/b)` with `A_mn = 16 q/(pi^6 D m n (m^2/a^2 + n^2/b^2)^2)` (odd `m, n`); `Mx, My, Mxy, Qx, Qy` are differentiated from the same coefficients, so they are mutually consistent by construction. Truncation error is *measured* (table 3.1), not assumed.
* **FE 1 (`plate_shell/plate_fe.py`).** Conforming `C^1` bicubic Hermite (Bogner-Fox-Schmit) Kirchhoff rectangle, 16 dof `(w, dw/dy, -dw/dx, d2w/dxdy)` per element, 4x4 Gauss, consistent pressure load including the nodal moment terms. SSSS in its *natural* variational form: only `w = 0` on the edges, so the normal moment is natural. Moments/shears recovered from element second/third derivatives.
* **Anchor 2 (Donnell shell, `plate_shell/shell_axisym.py`).** `D w'''' + (E t/R^2) w = p` with the full four-term homogeneous part `exp(+-beta x)(cos beta x, sin beta x)` and `w(0) = w'(0) = 0`, `w(L) = pR^2/(Et)`, `w'(L) = 0`; the finite length is solved exactly, not by a semi-infinite approximation.
* **FE 2.** Axisymmetric shell element with `u` linear (`C^0`) and `w` cubic Hermite (`C^1`), 4-point Gauss. The element carries the full membrane-bending coupling `K(eps_x^2 + 2 nu eps_x eps_theta + eps_theta^2) + D kappa_x^2`, i.e. the closed form is an independent solution of the same variational problem.
* **h-refinement.** Plate meshes [4, 8, 16, 32, 64], shell element counts [8, 16, 32, 64, 128]; convergence orders from a log-log least-squares fit of the error.
* **Tooling.** numpy/scipy (sparse direct solve), matplotlib; everything reproducible via `python verify_fem_plate_shell.py` and `python generate_p405w1_n05_report.py`. No number in this document is typed by hand.

## 3. Results

### 3.1 Analytic anchor sanity checks

* centre deflection coefficient `alpha = w_max D/(q a^4) = 0.00406235` vs the published `0.00406` (relative difference 5.79e-04).
* centre moments `Mx = My = 478.8640 N m/m` = `0.04789 q a^2` (published 0.0479).
* series truncation, relative change between successive `n_terms`:

| n_terms | d w | d Mx | d Qx |
|---|---|---|---|
| 41 | 1.36e-08 | 1.87e-05 | 8.81e-04 |
| 81 | 5.28e-10 | 2.57e-06 | 2.33e-04 |
| 161 | 1.84e-11 | 3.37e-07 | 5.99e-05 |
| 321 | 6.10e-13 | 4.32e-08 | 1.52e-05 |

### 3.2 Plate: errors against the Navier series

| mesh | h [m] | w_max err | L2(w) err | Mx err | My err | L2(Mx) err | Qx err | reactions / load [N] |
|---|---|---|---|---|---|---|---|---|
| 4x4 | 0.2500 | 3.552e-03 | 6.101e-03 | 2.669e-02 | 2.669e-02 | 7.699e-02 | 1.474e-01 | 10000.0 / 10000.0 |
| 8x8 | 0.1250 | 1.326e-04 | 2.158e-04 | 5.757e-03 | 5.757e-03 | 1.396e-02 | 3.448e-02 | 10000.0 / 10000.0 |
| 16x16 | 0.0625 | 5.406e-06 | 8.244e-06 | 1.365e-03 | 1.365e-03 | 2.913e-03 | 8.423e-03 | 10000.0 / 10000.0 |
| 32x32 | 0.0312 | 2.491e-07 | 3.802e-07 | 3.362e-04 | 3.362e-04 | 5.912e-04 | 2.042e-03 | 10000.0 / 10000.0 |
| 64x64 | 0.0156 | 1.318e-08 | 2.140e-08 | 8.347e-05 | 8.347e-05 | 1.359e-04 | 4.505e-04 | 10000.0 / 10000.0 |

Measured orders: `w_max p = 4.514`, `L2(w) p = 4.539`, `Mx p = 2.074`, `Qx p = 2.078`, `L2(Mx) p = 2.285`.

### 3.3 Cylindrical shell: errors against the Donnell closed form

| elements | beta*h | edge Mx err | sigma_theta membrane err (x>=6/beta) | pointwise sigma_theta err | w field err |
|---|---|---|---|---|---|
| 8 | 1.285 | 2.187e-01 | 1.842e-03 | 1.955e-02 | 1.955e-02 |
| 16 | 0.643 | 6.439e-02 | 1.843e-03 | 1.556e-03 | 1.556e-03 |
| 32 | 0.321 | 1.690e-02 | 1.857e-03 | 1.896e-04 | 1.896e-04 |
| 64 | 0.161 | 4.282e-03 | 1.864e-03 | 4.800e-05 | 4.800e-05 |
| 128 | 0.080 | 1.074e-03 | 1.865e-03 | 1.202e-05 | 1.202e-05 |

Measured orders: edge `Mx p = 1.925`; pointwise `sigma_theta` with the equilibrium-consistent recovery `sigma_theta = E w/R` `p = 2.635`; with the element-constant `eps_x` (constitutive) recovery `p = 0.979` (i.e. one order lower, as expected for a piecewise-constant axial strain).

### 3.4 Boundary-layer length scale

Fitting `w - w_p = R0 exp(-beta x) cos(beta x - phi)` to the FE field gives `beta_fit = 12.853609 1/m` vs `beta_theory = 12.854070 1/m` (relative difference 3.59e-05); the attenuation length is `1/beta = 0.7780 sqrt(R t)`, matching the classical `(3(1-nu^2))^(-1/4) = 0.7780`.

### 3.5 Gate table

| gate | verdict | criterion |
|---|---|---|
| G5_tests | PASS | >= 12 pytest checks, all green (tests/test_plate_shell_verification.py) |
| G1 | PASS | w_max relative error <= 1% at 4x4 and <= 0.1% at 16x16 |
| G2 | PASS | Mx, My relative error <= 2% (8x8 and all finer meshes) |
| G4_plate | PASS | h-refinement order >= 2 for the C^0 quantities (moments/shear); C^1 plate deflection >= 3 |
| G3 | PASS | sigma_theta relative error <= 1% in the membrane region (x >= 6/beta) and pointwise <= 1% at the finest mesh |
| G3b_edge_effect | PASS | edge bending moment <= 2% at 64 axial elements and measured attenuation length matches sqrt(R t) |
| order_shell | PASS | edge-moment h-refinement order >= 1.8 (closed-form anchor) |

Negative controls: `closed_form_satisfies_shell_ode` = True, `navier_series_truncation_below_gate` = True, `over_constraint_makes_model_stiffer` = True, `plate_natural_bc_used` = True, `shell_energy_balance_identity` = True.

## 4. Figures

![fig1_plate_deflection_field.png](artifacts/plate_shell/fig1_plate_deflection_field.png)

*FE vs Navier deflection field of the square plate (8x8 mesh) and the normalised discretisation-error pattern (peak error in the panel title).*

![fig2_plate_moments_shear.png](artifacts/plate_shell/fig2_plate_moments_shear.png)

*Bending moments Mx, My and shear Qx along the plate centre lines: FE vs Navier series.*

![fig3_convergence_orders.png](artifacts/plate_shell/fig3_convergence_orders.png)

*log-log h-refinement: plate (w_max, L2(w), Mx, Qx with fitted orders) and shell (edge Mx, pointwise hoop stress) with the 1% gate line.*

![fig4_shell_pressure_edge_effect.png](artifacts/plate_shell/fig4_shell_pressure_edge_effect.png)

*Cylindrical shell under internal pressure: hoop stress sigma_theta/(pR/t), edge bending moment Mx and the measured exp(-beta x) boundary layer with 1/beta = 0.778 sqrt(R t).*

## 5. Pitfalls, limitations, applicability

1. **The SSSS constraint set matters.** Fixing the edge twist dof `w,xy` (a natural but wrong reading of `w = 0` along a supported edge) restricts the trial space and makes the model systematically stiffer: the negative control shifts the 8x8 centre deflection by -2.92 %.
2. **Kirchhoff thin-plate validity**: `a/t = 100`; transverse shear is neglected, so plates with `a/t < 20` need a Mindlin/FSDT element.
3. **Shear recovery is element-constant** (third derivative of a cubic), so `Qx/Qy` converge one order lower than `w` and are compared at interior points only -- never on the edges, where the Kirchhoff edge reaction also carries `d Mxy/dy`.
4. **Donnell axisymmetric reduction**: no torsion, no axial load, axisymmetric pressure, thin shell (`R/t = 100`). The closed form is exact only inside that reduction, which is also the element's model.
5. **Membrane-region stress.** At `x = 6/beta` the *exact* solution is still 0.19 % away from `pR/t`; that column is a physics check of the membrane asymptote, not a discretisation error. The discretisation error is the pointwise column of section 3.3.
6. **Scope: code verification, not validation.** Linear statics only; no buckling, plasticity, large deflection or experiment. The baseline is the classical closed-form solution, which is the correct reference for element verification; experimental validation of the shell imperfection sensitivity is out of scope.

## 6. Error judgement against the baselines

| quantity | baseline | FE value | baseline value | rel. error | below 3%? |
|---|---|---|---|---|---|
| plate w_max (4x4) | Navier series | 2.119926e-03 | 2.112423e-03 | 0.3552 % | yes |
| plate w_max (16x16) | Navier series | 2.112435e-03 | 2.112423e-03 | 0.0005 % | yes |
| plate Mx centre (8x8) | Navier series | 4.816207e+02 | 4.788640e+02 | 0.5757 % | yes |
| plate My centre (8x8) | Navier series | 4.816207e+02 | 4.788640e+02 | 0.5757 % | yes |
| plate Qx (a/4,b/2) (64x64) | Navier series | 1.364406e+03 | 1.363792e+03 | 0.0450 % | yes |
| shell sigma_theta membrane (128 el.) | pR/t | 9.994216e+07 | 1.000000e+08 | 0.1865 % | yes |
| shell edge Mx (64 el.) | Donnell closed form | 3.013179e+03 | 3.026138e+03 | 0.4282 % | yes |
| shell edge bending stress (64 el.) | Donnell closed form | -1.807908e+08 | -1.815683e+08 | 0.4282 % | yes |
| boundary-layer beta | Donnell theory | 1.285361e+01 | 1.285407e+01 | 0.0036 % | yes |

All quantities compared at the reported mesh are one to four orders of magnitude below the 15 % acceptance limit and inside the tighter 1-2 % node gates. Exceedances on the *coarsest* meshes are stated explicitly instead of being dropped: plate centre moment at 4x4 is 2.67 % (gate 2 %) and reaches 0.58 % at 8x8; the shell edge bending moment is 21.87 % at 8 elements and only crosses the 2 % gate at 32 elements (1.69 %). This is the expected O(h^2) moment behaviour of a C^1 plate element / Hermite shell element with an element-constant strain recovery.

