# Under-ice acoustic tomography: sound speed, ice thickness and temperature

*Node `p9050-n02-acoustic-tomography` (campaign DAG, physics domain, TensorPhysics).*

This document is the validation record for `src/tensorphysics/acoustic_tomography/`.
It states the physics that is implemented, how it is verified, the numbers the
case produces, and where the method stops being valid. Every number below is
produced by `scripts/run_acoustic_tomography_case.py` and stored in
`runs/m9890_acoustic_tomography/results.json`; none is transcribed by hand.

---

## 1. Problem statement

An under-ice acoustic tomography system (10-1000 Hz, 1-10 km baselines) is
deployed on a 4 km x 200 m section of an Arctic shelf under a level ice sheet of
1.5-2.6 m draft. The task is the classical tomography inverse problem:

| product | observable | method |
|---|---|---|
| sound-speed field `c(r,z)` | source-receiver travel times | Snell-law ray tracing (forward) + SIRT algebraic reconstruction (inverse) |
| ice thickness `h(r)` | under-ice echo travel time | `d = c_w t/2` (draft) with the hydrostatic `h = (rho_w/rho_i) d` |
| temperature field `T(r,z)` | retrieved `c` with `S`, `p` known | analytic inversion of the campaign `c(T,S,p)` relation |
| uncertainties `sigma_c, sigma_T, sigma_h` | timing noise, sound-speed error, density ratio | Monte-Carlo ensemble + first-order propagation |

## 2. Physics implemented

### 2.1 Sound speed

Two relations are implemented so they can be cross-checked:

* the campaign relation `c = 1449.2 + 4.6T - 5.5e-2 T^2 + 2.9e-4 T^3
  + 1.34(S-35) + 1.6e-2 P` (P in dbar ~ depth in m);
* the nine-term Mackenzie (1981) relation with explicit depth dependence.

Over the validity box `-2 <= T <= 30 degC`, `25 <= S <= 40`,
`0 <= D <= 8000 m` the two agree to **< 1 m/s** (test
`test_mackenzie_and_campaign_relation_agree_within_1_m_s`). Since
`dc/dT = 4.6 - 0.11 T + 8.7e-4 T^2 > 0` on that box, the cubic inversion
`c -> T` has a unique root; a bracketed Newton iteration recovers `T` to
`< 1e-6 degC` on 200 random points, and targets outside the reachable interval
return `NaN` rather than a silently clamped value.

### 2.2 Ray tracing under an ice canopy - and why the duct matters

The tracer integrates the eikonal ray equations for a range-dependent `c(r,z)`
(vertical plane, `z` positive downwards, ice underside at `z = 0`):

```
dr/dt = c cos(theta)
dz/dt = c sin(theta)
dtheta/dt = sin(theta) dc/dr - cos(theta) dc/dz
```

with classical RK4 (fourth order in the step) and a secant-refined partial final
step so the endpoint at `r = r_goal` is reached at fourth order.

In the Arctic the sound speed increases with depth (cold, fresh surface water
over a warmer, saltier halocline, see Fig. 2). A downward ray therefore turns
back upward within a few tens of metres, and **no purely refracted ray connects
two points with a large depth offset at kilometric range**. What connects the
array is the ice-underside surface duct: specular reflections at `z = 0`
(`z -> 2 z_top - z`, `theta -> -theta`, which leaves range and travel time
untouched). This is the dominant under-ice propagation path and it is why the
receiver records a multipath train (Fig. 2b) rather than a single arrival.

Verification of the tracer:

* closed form for a linear profile `c = c0 + g z` (exact ray integrals for range
  and time): agreement to **< 1e-6 relative** in both range and travel time;
* Snell invariant `cos(theta)/c` conserved along a stratified ray to 2e-3;
* straight-line time in a homogeneous medium to 1e-9;
* travel-time reciprocity `t(i->j) = t(j->i)` to 1e-14 in a 1-D field, and to
  **0.16 ms** (6e-5 relative) on the campaign field computed through a
  range-mirrored twin;
* the path integral of the slowness along the traced polyline reproduces the
  traced travel time to **< 1 ms** (typically 0.05 ms).

That last check is a regression test that caught a real defect: the stored
polyline used to cut the corner at every ice reflection, shortening the path by
up to one step per bounce. With 10-15 bounces the travel times were biased by
**~8-12 ms**, i.e. an order of magnitude above the 1 ms timing noise, which
destroyed the tomography. The polyline now includes the reflection vertices
(the crossing point is inserted) and the discrepancy is 0.05 ms.

### 2.3 Algebraic reconstruction

Travel time along ray `i` is `t_i = int_{Gamma_i} s dl = sum_j a_ij s_j` with
`s = 1/c` on a node grid and a bilinear (node) quadrature of the path integral.
Both **ART** (Kaczmarz) and **SIRT** are implemented; SIRT is used for the case
with

* `C A^T R` row/column scaling (`R = 1/rowsum`, `C = 1/colsum`),
* a *physical* Laplacian smoothing step (divided by `dr^2`, `dz^2`), i.e. a
  dimensionless diffusion number (stability limit 0.2), so the regularisation
  is grid independent,
* the **discrepancy principle** as the stopping rule: stop when the data
  residual reaches the known timing noise `sigma_t`. SIRT is semi-convergent
  (the field error grows once the noise is fitted), so this is the regularisation
  choice - it needs no tuning against the truth.

## 3. The campaign case

| item | value |
|---|---|
| domain | `0 <= r <= 4000 m`, `0 <= z <= 200 m`, ice underside at `z = 0` |
| array | 15 transceivers: 5 ranges (0, 1, 2, 3, 4 km) x 3 depths (40, 100, 160 m) |
| pairs | 90 (all unordered pairs with `|dr| >= 500 m`) |
| background state (prior) | `T = -1.6 + 3.3 (z/200)^1.5 degC`, `S = 31 + 3.8 (z/200)^0.8` |
| anomaly (truth - prior) | warm Atlantic intrusion `dT = +1.3 degC, dS = +0.25` at (2.2 km, 150 m), scales 900 m / 60 m; melt-water lens `dT = -0.10, dS = -0.50` at (1.5 km, 35 m), scales 1100 m / 25 m |
| forward model | multipath eigenrays traced through the **truth** field (snell), 146 arrivals |
| timing noise | `sigma_t = 1 ms` (nominal for 100-1000 Hz coded pulses) |
| arrival identification | bounce order equal, `|d theta_0| <= 0.02 rad`, `|dt| <= 20 ms` |
| identified arrivals | **102 of 146** (63 of 90 pairs) |
| inversion grid | 8 x 6 nodes (500 m x 40 m), 48 unknowns |
| ice thickness truth | `h(r) = 1.85 + 0.45 sin(2 pi r/1500) + 0.25 exp(-((r-2600)/700)^2)` m, 24 stations |
| echo noise | `sigma_t = 0.05 ms` (nominal), stress case 0.5 ms |

### 3.1 Why arrival identification is the crux

The anomaly shifts arrival times by up to ~13 ms, while in the duct the
neighbouring arrivals can be only 1-3 ms apart. Identification by travel time
alone therefore mis-pairs rays; the error appears as a data/model inconsistency
of several milliseconds that the inversion then absorbs with a spurious field,
and the reconstruction error explodes (measured: 100-1000 m/s, i.e. nonsense).

The identification used here combines three labels: the discrete bounce order
(robust), the launch angle (`|d theta_0| <= 0.02 rad`, continuous under a small
medium change) and the arrival time (`|dt| <= 20 ms`, a 20 ms gate is safe
because the anomaly signal cannot exceed ~13 ms). With that gate the *measured*
linearisation error - the difference between the fully re-traced truth arrival
time and the fixed-path integral of the anomaly along the model ray - is
**0.09 ms rms** (max 0.5 ms) on the 102 identified arrivals, i.e. the travel time
is linear in the anomaly to well below the noise, exactly as Fermat's principle
requires. Without the gate the same quantity is 7.9 ms rms: that gap *is* the
identification error, and it is reported in the diagnostics.

## 4. Results (campaign criteria)

All values from `runs/m9890_acoustic_tomography/results.json`.

| # | criterion | requirement | achieved | verdict |
|---|---|---|---|---|
| a | `rms(c_retrieved - c_truth)` over CTD points | < 5 m/s, >= 30 samples | **0.996 m/s** (max 2.47), 35 samples | PASS |
| b | `rms(h_retrieved - h_truth)` | < 0.5 m, >= 20 samples | **0.0437 m** (max 0.112), 24 stations | PASS |
| c | `rms(T_retrieved - T_truth)` | < 0.5 degC, >= 30 samples | **0.215 degC** (max 0.497), 35 samples | PASS |
| d1 | no acoustic data | inversion identically NaN | all 48 nodes NaN, 0 observations | PASS |
| d2 | `h -> 0` | reflection time -> 0 | `t(0) = 0.0` exactly, monotone in `h` | PASS |
| d3 | `T = 0 degC`, `S = 0` | `c >= 1400 m/s` | `c = 1402.3 m/s` | PASS |
| e | `h = 2 m`, anchor 2.6 s | `|dt| <= 10 ms` | **-9.51 ms** (primary), +8.85 ms (secondary) | PASS (see 4.3) |

Data fit: SIRT stops after 25 iterations at a residual of **0.825 ms** (the
noise level), i.e. the data are explained to the noise and no further.

### 4.1 Uncertainties (Monte-Carlo, 60 realisations, same iteration count)

| quantity | mean | max |
|---|---|---|
| `sigma_c` | 2.62 m/s | 4.84 m/s |
| `sigma_T` | 0.57 degC | 1.02 degC |
| `sigma_h` | 0.057 m | 0.063 m |

`sigma_c` is *conservative*: it is the ensemble spread of the 25-iteration SIRT
output, and individual realisations scatter more than the single-realisation
error of 0.996 m/s, because the retrieved field has a systematic component that
partly cancels in the mean. The honest reading is therefore "0.996 m/s error
with a 2.6 m/s posterior width"; the criterion is met on the pointwise error.

### 4.2 Ice thickness: what actually limits it

At 0.05 ms echo timing the propagated draft error is 0.036 m, i.e. 0.04 m in
thickness - forty times inside the 0.5 m criterion. The dominant real-world
error is *not* acoustic: it is the hydrostatic assumption. The density ratio
`rho_w/rho_i = 1.120` varies by `+-2..3 %` for multi-year ice (rho_i = 900-940),
which maps to `+-0.05 m` on a 2 m sheet, and ridges/pressure ridges break the
hydrostatic equilibrium completely. The stress case in the results
(`0.5 ms` echo timing, rough undersides / hull-borne sonar) gives
`rms = 0.501 m` (max 0.92 m) - i.e. exactly at the 0.5 m criterion. Reading it
that way: the 0.05 ms design precision carries a factor-40 margin on the acoustic
side, the 0.5 ms case does not, and the two systematic items above (`+-0.05 m`
density ratio, ridges) are of the same size as the acoustic term at 0.5 ms. The
criterion is therefore met at the design precision and marginal at the degraded
one; both are reported in `results.json` under
`criteria.b_ice_thickness.stress_case`.

### 4.3 The falsifiability check, stated honestly

The campaign anchor is "measured ~2.6 s". A 2.6 s travel time over a kilometre
baseline is a *propagation* time, not an ice-thickness echo: the echo of a 2 m
sheet is 2.5 ms (the draft-mode echo, reported separately in the same block of
the results). Two readings of the anchor are therefore possible, and they differ
by exactly the width of the tolerance band:

| reading | predicted | deviation from 2.6 s |
|---|---|---|
| primary: one-way ducted source-receiver time, 3.75 km baseline, 40 m depth | 2.5905 s | **-9.51 ms** |
| secondary: two-way specular reflection off the ice underside, same geometry | 2.6089 s | +8.85 ms |

Both are inside the 10 ms band, but the exercise must be read for what it is: a
10 ms band on 2.6 s is 0.38 %, equivalent to **14.4 m of baseline** (`dt/dL =
0.695 ms/m`) or 5.5 m/s (0.4 %) of path-mean sound speed. The anchor is quoted to
two significant figures (+-50 ms), so it cannot discriminate either reading, let
alone validate the model at the 10 ms level. What the check does establish is
that the forward model produces the right order of magnitude with the right
physics (a ducted/reflected path over a 3.75 km baseline at 1430-1440 m/s) and,
crucially, that it is *falsifiable*: the same code that produces 2.5905 s would
fail loudly at any baseline error above 14 m.

### 4.4 The ray-bending iteration is an honest negative result

One bending iteration (re-trace the eigenrays in the retrieved field, re-identify
the arrivals, re-invert) reduced the data residual from 0.825 ms to 0.647 ms but
**degraded** the reconstructed field from 0.996 m/s to 2.13 m/s. The reason is in
the diagnostics: re-tracing in the retrieved (still biased) field changed the
arrival pattern enough that the identification gate only matched 67 of the 102
arrivals, and the loss of one third of the data outweighs the improved operator.
The acceptance metrics therefore use stage 1, and the bending stage is reported
as a diagnostic. A production implementation would keep the identified arrivals
of the first pass and only update their paths.

### 4.5 Sensitivity to timing noise

`noise_sensitivity` re-inverts the identified arrivals with fresh noise (fixed
iteration budget, one realisation per level):

| `sigma_t` [ms] | 0.2 | 0.5 | 1.0 | 2.0 | 5.0 |
|---|---|---|---|---|---|
| rms `c` [m/s] | 0.98 | 1.21 | 1.73 | 2.35 | 6.00 |
| rms `T` [degC] | 0.199 | 0.255 | 0.337 | 0.501 | 1.223 |
| `sigma_c` (ensemble) | 0.48 | 1.19 | 2.38 | 4.78 | 11.9 |

The sound-speed error scales roughly as `sigma_t^0.5..0.6` (a square-root law is
expected: the number of independent paths is fixed and the errors add in
quadrature), crossing the 5 m/s criterion near 4 ms; the temperature error
crosses 0.5 degC at ~2 ms. The nominal 1 ms design therefore sits a factor 2-4
inside both criteria, and the 0.996 m/s of the campaign run is one realisation of
a distribution whose width is the 2.4 m/s ensemble spread reported in 4.1 - a
1.0 vs 1.7 m/s difference between realisations is well inside one sigma.

## 5. Figures

| file | content |
|---|---|
| `fig1_sound_speed_reconstruction.png` | truth anomaly, SIRT retrieval, profiles, CTD comparison with the +-5 m/s band |
| `fig2_ray_coverage_multipath.png` | inversion ray geometry (ducted, ice-reflected) and the arrival train vs pair separation |
| `fig3_data_fit_and_ice_thickness.png` | identified arrivals vs model prediction; ice thickness with the +-2 sigma band |
| `fig4_temperature_inversion.png` | `T` from the retrieved `c` vs truth, and the profile at the anomaly centre |
| `fig5_uncertainty_sensitivity.png` | ensemble `sigma_c` map; retrieval error vs timing noise against the criteria |

## 6. Limitations and applicability

1. **Synthetic (OSSE) evidence.** The "observations" are produced by the same ray
   tracer that provides the inversion paths. There is no comparison with a
   measured data set; the 0.5 degC / 0.05 m numbers are therefore *consistency*
   numbers under the stated noise model, not field accuracies. Reference
   validation against Arctic tomography experiments (AATE, 1994; NPAL, 2004)
   is **not** included and would be the next step.
2. **Horizontally uniform prior.** The inversion retrieves the anomaly relative
   to a declared climatological background. The background itself must come from
   an independent CTD cast; its error propagates directly into `c` (and into
   `T`), and is not included in the uncertainty budget.
3. **Temperature needs salinity.** `c` alone cannot separate `T` and `S`
   (`dc/dT ~ 4.6`, `dc/dS ~ 1.34`, a 3.4:1 ratio). The retrieval assumes the
   salinity prior; the anomaly's `dS` (0.25/-0.50 psu) therefore leaks into `T`
   as `1.34 dS / 4.6 = 0.07..0.15 degC`, which is a third of the reported
   temperature error. A second observable (e.g. electrical conductivity or a
   second acoustic mode) is required to do better.
4. **Validity edge.** The truth surface temperature sits at the freezing point
   (-1.7 degC) and the campaign relation is only defined for `T >= -2 degC`; one
   CTD point of 35 was clamped to the edge of the relation
   (`n_clamped_at_validity_edge`). It is flagged, not hidden.
5. **Identification.** The 20 ms / 0.02 rad / bounce-order gate identifies 70 %
   of the arrivals. In a duct with 1-3 ms arrival spacing this is the honest
   limit of arrival-based tomography at 1 ms precision; the alternative
   (waveform correlation or travel-time differences between arrivals) is not
   implemented.
6. **Acoustics is linear.** No nonlinear (finite-amplitude) effects, no
   attenuation/absorption, no ice elastic waves, no Doppler from a moving AUV.
   The 100-1000 Hz band is only implicit in the stated timing precision and in
   the assumption that the ray approximation holds (`k H >> 1`, true here:
   `k ~ 4 /m` at 1000 Hz).
7. **Ice thickness from draft only.** Level-ice hydrostatic relation; ridges,
   rubble fields and under-ice melt ponds break it (they change `rho_i` and the
   equilibrium), see 4.2.
8. **AUV navigation application.** The campaign lists polar AUV navigation as
   the application. This module supplies the timing/field inversion that such a
   navigation filter would consume (position error from the travel-time
   mismatch), but no navigation filter (EKF/particle) is implemented here.

## 7. How to reproduce

```bash
# full campaign case (~3 min, writes runs/ and artifacts/)
PYTHONPATH=src python scripts/run_acoustic_tomography_case.py

# reduced smoke case
PYTHONPATH=src python scripts/run_acoustic_tomography_case.py --fast

# tests (30 tests, ~17 s)
PYTHONPATH=src python -m pytest tests/test_acoustic_tomography.py -q
```