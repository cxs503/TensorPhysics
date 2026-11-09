# Non-intrusive UQ as a pre-validation acceptance gate (IN143-N11)

**What this node delivers.** A workflow-level uncertainty-quantification contract for the Tensor
ecosystem: input distributions + solver adapter -> low-discrepancy sampling -> sparse polynomial
chaos (PCE) surrogate -> first- / total- / second-order Sobol' indices with 95 % confidence
intervals -> an independent Saltelli/Jansen cross-check -> a **PASS / FAIL / WITHHELD** verdict
that an engineering error gate is not allowed to bypass.

Package: `src/tensorphysics/uq/` (distributions, sampling, basis, regression, pce, mc, indices,
adapter, gate, study, benchmarks). Tests: `tests/test_uq_core.py`, `tests/test_uq_acceptance.py`,
`tests/test_uq_gate.py`. End-to-end run: `scripts/uq_pce_sobol_gate_e2e.py` ->
`artifacts/uq_pce_sobol_gate/results.json` (this document is generated from that file by
`scripts/uq_pce_sobol_gate_report.py`).


## 1. Case introduction (physics, geometry, parameters)

### 1.1 Verification case: Ishigami function

`Y = sin(x1) + a sin^2(x2) + b x3^4 sin(x1)`, `x_i ~ U(-pi, pi)`, `a = 7`, `b = 0.1`.
The variance decomposition is exact: with `V1 = 0.5 (1 + b pi^4/5)^2`, `V2 = a^2/8`,
`V13 = b^2 pi^8 (1/18 - 1/50)` and `V = V1 + V2 + V13`,
`S1 = (V1, V2, 0)/V`, `ST = (V1+V13, V2, V13)/V`, `S13 = V13/V`.

The case is chosen because it is a *hard* test for a sparse surrogate: `x3` has **zero**
first-order effect but a large total effect, `x1` enters through a non-polynomial (sine) term, and
`x3` acts only through a degree-4 interaction with `x1`.

| parameter | distribution | S1 (exact) | ST (exact) |
|---|---|---|---|
| x1 | U(-3.14159, 3.14159) | 0.313905 | 0.557589 |
| x2 | U(-3.14159, 3.14159) | 0.442411 | 0.442411 |
| x3 | U(-3.14159, 3.14159) | 0 | 0.243684 |

Exact closed second-order index: `S2(x1,x3) = 0.243684` (all of `ST(x3)` is interaction with `x1`). Note that the widely quoted table values (0.3139 / 0.4424 / 0 /
0.5576 / 0.4424 / 0.2439) are rounded; `ST(x3)` differs from the exact number by 2.2e-4, which is
why the acceptance criterion is measured against the exact decomposition.


### 1.2 Engineering case: Lindqvist (1989) level-ice resistance, PC6 reference ship

Three-component law `R_total = R_crush + R_break + R_submersion` with
`R_crush = 0.5 (tan(alpha) + mu/cos(alpha)) sigma_c h^2`,
`R_break = 0.5 (sin(phi/2) + mu cos(phi/2)) sigma_f h B / k_bend`,
`R_sub = (0.15 + 0.3 Fr_h) rho_i V^2 h B`, implemented in
`tensorphysics.ice.resistance.level_ice_resistance` (node IN10-C1).

| quantity | value |
|---|---|
| L_pp | 150.0 m |
| B | 24.0 m |
| T | 8.5 m |
| waterline half-angle phi | 30.0 deg |
| bow flare alpha | 25.0 deg |
| displacement | 21000.0 t |

| input | marginal | parameters | mean | std |
|---|---|---|---|---|
| h_m | uniform | low=0.5, high=1.3 | 0.9 | 0.2309 |
| V_ms | uniform | low=1, high=4.5 | 2.75 | 1.01 |
| sigma_c_Pa | normal | mu=5.00e+06, sigma=7.50e+05 | 5e+06 | 7.5e+05 |
| sigma_f_Pa | normal | mu=5.00e+05, sigma=7.5e+04 | 5e+05 | 7.5e+04 |
| mu | uniform | low=0.1, high=0.22 | 0.16 | 0.03464 |

The distribution choices are the uncertainties a designer actually faces: level-ice thickness over a
PC6 operating envelope (uniform), transit speed (uniform over the 1.0-4.5 m/s domain), ice
compressive/flexural strength (Gaussian about the ITTC/ISO 19906 reference values, 15 % CoV) and
hull-ice friction (uniform 0.10-0.22). The declared model validity domain (h <= min(0.03 L, 0.10 B),
Fr_L <= 0.25) is respected by every sample, so the study never leaves the law's envelope.


### 1.3 Validation hand-off case: Rayleigh-Ritz plate buckling vs DNV-RP-C201

| quantity | value |
|---|---|
| panel a x b | 3.0 m x 0.6 m |
| thickness t | 0.012 m |
| E, nu, sigma_y | 2.06e+11 Pa, 0.3, 3.15e+08 Pa |
| aspect ratio a/b | 5.000 |
| uncertain inputs | E_Pa, nu, t_m, a_m |

## 2. How it was done

### 2.1 Pipeline

1. **Input model** (`uq.distributions`): independent marginals (uniform / normal / lognormal /
   beta) and an optional Gaussian-copula correlation matrix. Correlated inputs are accepted by the
   sampler and *refused* by the PCE (the product basis requires independence) -> the gate returns
   WITHHELD instead of publishing biased indices.
2. **Sampling** (`uq.sampling`): scrambled Sobol' sequence (default), Latin hypercube or Monte
   Carlo, always on the canonical unit hypercube. Every sample set carries a SHA-256 digest over
   its float64 payload plus sampler/seed/dimension, so reproducibility is checkable and not
   asserted.
3. **Adapter** (`uq.adapter`): `f(X) -> y` with evaluation counting and per-sample exception
   capture; a solver that raises leaves a `NaN` behind (evidence) instead of crashing the study.
4. **Basis** (`uq.basis`): orthonormal Legendre (uniform scores) or Hermite (standard normal)
   tensor products, total-degree or hyperbolic (`q`-norm) truncation. Orthonormality is verified
   by Gauss quadrature to machine precision (`max |G - I| < 1e-12`), not by assumption.
5. **Sparse regression** (`uq.regression`): orthogonal matching pursuit over the truncated basis,
   with the closed-form leave-one-out error `eps_LOO = sum (r_i/(1-h_ii))^2 / sum (y-mean)^2` at
   every path step; the support size and the total degree are both chosen by minimising `eps_LOO`
   (Blatman-Sudret style model selection with OMP in place of LAR).
6. **Analytical Sobol' indices** (`uq.pce`): because the basis is orthonormal,
   `Var(Y) = sum_{alpha != 0} c_alpha^2` and the first / total / closed second-order indices are
   sums over multi-index subsets. Confidence intervals come from sampling the coefficient
   covariance `sigma^2 (A^T A)^-1` (regression noise only - truncation bias is what `eps_LOO`
   measures; the two are reported separately).
7. **Independent cross-check** (`uq.mc`): Saltelli (2010) design with Jansen (1999) estimators and
   a percentile bootstrap over the base samples.
8. **Gate** (`uq.gate`): PASS / FAIL / WITHHELD with named reason codes; **WITHHELD** for
   inadmissible evidence (NaN, sample budget below the policy floor, `eps_LOO > 0.1`, zero
   variance, correlated inputs, missing cross-check, inconclusive cross-check) and **FAIL** for
   contradictory evidence (estimators disagree beyond tolerance with disjoint intervals; variance
   algebra violated: `sum S1 > 1`, `ST_i < S1_i`, `sum ST < sum S1`, negative indices).
9. **Hand-off** (`uq.study.attach_to_validation`): the certified `U_input = sqrt(Var_PCE)` enters
   ASME V&V 20's `U_val = sqrt(U_num^2 + U_input^2)`; if the UQ verdict is not PASS, the
   validation verdict is forced to WITHHELD (no fabricated `U_val`).

### 2.2 Key settings

| setting | value |
|---|---|
| verification sample count | n = 2048 (power of two, Sobol') |
| degree sweep | 2,3,4,5,6,8,10,12 (skipped if n < 1.5 x basis size) |
| PCE support | degree 12, 76 retained terms out of 455 |
| Saltelli cross-check | n_base = 8192 -> 40960 evaluations, 300 bootstrap replicates |
| A/B independence | max |corr(A_j, B_j)| = 1.9e-18 (one 2d-dim Sobol' set, split in half) |
| CI level | 95 % |
| gate policy | eps_LOO <= 0.1, n >= 2 x basis_size (>= 64), 5 % cross-estimator tolerance |
| reproducibility | same seed + same sample digest -> bit-identical indices (asserted) |
| engineering case | n = 1024, degrees 2-5, Saltelli base 8192 |
| tool chain | Python 3.11.6, numpy 2.4.6, scipy (qmc/eig), pytest, matplotlib |

### 2.3 Commands actually run

```
PYTHONPATH=src python -m pytest tests/test_uq_core.py tests/test_uq_gate.py tests/test_uq_acceptance.py -q
PYTHONPATH=src python scripts/uq_pce_sobol_gate_e2e.py
PYTHONPATH=src python scripts/uq_pce_sobol_gate_report.py --post
```


## 3. Results


### 3.1 Ishigami: PCE vs exact vs Saltelli

| param | S1 exact | S1 PCE | 95% CI (PCE) | S1 Saltelli | 95% CI (MC) | |err| PCE |
|---|---|---|---|---|---|---|
| x1 | 0.313905 | 0.313905 | [0.313903, 0.313908] | 0.314453 | [0.284374, 0.344348] | 1.3e-07 |
| x2 | 0.442411 | 0.442412 | [0.442409, 0.442415] | 0.4424 | [0.419075, 0.465511] | 1.1e-06 |
| x3 | 0 | 0 | [0, 0] | 6.4208e-04 | [-0.0223683, 0.0264062] | 0 |

| param | ST exact | ST PCE | 95% CI (PCE) | ST Saltelli | 95% CI (MC) | |err| PCE |
|---|---|---|---|---|---|---|
| x1 | 0.557589 | 0.557588 | [0.557585, 0.557591] | 0.557573 | [0.526782, 0.587897] | 1.1e-06 |
| x2 | 0.442411 | 0.442412 | [0.442409, 0.442415] | 0.442415 | [0.428529, 0.457033] | 1.1e-06 |
| x3 | 0.243684 | 0.243683 | [0.24368, 0.243685] | 0.243602 | [0.233978, 0.251537] | 9.7e-07 |

Closed second-order index `S2(x1,x3)`: exact 0.243684, PCE 0.243683, error 9.7e-07. `S2(x1,x2)` and `S2(x2,x3)` come out at 1.8e-11 / 1.2e-10 as they must.


### 3.2 Acceptance criteria (machine-checkable)

| criterion | measured | threshold | pass |
|---|---|---|---|
| C1 PCE absolute error (S1/ST/S2) | S1 1.1e-06, ST 1.1e-06, S2 9.7e-07 | <= 1e-3 | yes |
| C1 Monte-Carlo relative error | S1 0.175 %, ST 0.0335 % | <= 5 % | yes |
| C2 eps_LOO vs degree (Spearman) | -1 (seed 12345), -1 (seed 77) | <= -0.9 | yes |
| C2 eps_LOO vs sample count (Spearman) | 1 | <= -0.9 | no (falsified - see 3.3) |
| C3 reproducibility | sample digest 9070733322e059c6..., S1/ST bit-identical | bitwise equal | yes |
| C4 WITHHELD triggers | NaN output / sample budget / eps_LOO > 0.1 / adapter errors / inconclusive MC: all WITHHELD | 5/5 probes | yes |
| C5 PCE vs Saltelli relative deviation | S1 0.175 %, ST 0.0331 % over 3 indices | <= 5 % | yes |
| C6 validation hand-off | converged -> PASS (ratio 1.2e-13), under-resolved -> FAIL (ratio 1.18) | PASS when resolved, FAIL when not | yes |

### 3.3 Where the criterion does not hold (honest negative result)

The second clause of criterion C2 ("more samples -> `eps_LOO` decreases monotonically") is **not**
satisfied by this estimator, and the measurement is reported rather than tuned away:

| n | eps_LOO (degree 2, exactly representable model + 0.05 noise) |
|---|---|
| 128 | 0.002665 |
| 256 | 0.003004 |
| 512 | 0.003078 |
| 1024 | 0.00311 |
| 2048 | 0.003368 |

Spearman(n, eps_LOO) = 1. The LOO error converges *upward* to the noise
floor `noise^2/Var(y) ~ 3.5e-3`: at small n the minimum-over-path model selection is optimistic, and
more samples progressively remove that optimism. Consequence for the design (already implemented):
the sample budget is enforced **a priori** by the policy floor `n >= 2 x basis_size (>= 64)`, and
`eps_LOO` is never trusted on its own to certify a sample-starved study. The degree direction of C2
holds exactly (Spearman = -1.0 for two independent seeds, eight degrees).


### 3.4 Engineering case: ice-load parameter ranking

| input | S1 | ST | 95% CI (S1) |
|---|---|---|---|
| h_m | 0.8857 | 0.9009 | [0.885, 0.887] |
| V_ms | 0.004093 | 0.004214 | [0.0039, 0.00428] |
| sigma_c_Pa | 0.04074 | 0.05052 | [0.0402, 0.0413] |
| sigma_f_Pa | 0.02598 | 0.02792 | [0.0255, 0.0264] |
| mu | 0.02793 | 0.03204 | [0.0275, 0.0284] |

`R_total = 2585.5 +- 1033 kN` (deterministic value at
the nominal point: 2418.3 kN), `eps_LOO = 6.5e-04`,
gate verdict **PASS**, `sum S1 = 0.9845`,
`sum ST = 1.016`. Ranking by total effect: h_m (0.901), sigma_c_Pa (0.0505), mu (0.032), sigma_f_Pa (0.0279), V_ms (0.00421).

The physics reads straight off the algebra: the crushing term is quadratic in thickness and linear
in compressive strength, so a 0.5-1.3 m thickness spread plus a 15 % strength spread carries almost
all of the variance; speed only enters the submergence term (`R_sub` is ~13 % of `R_total`), hence
`ST(V) < 0.01`. `sum(ST) - sum(S1) = 0.0312` quantifies the multiplicative interaction between thickness and strength.


### 3.5 Validation hand-off (ASME V&V 20 with certified U_input)

| simulation | f_sim | f_exp (DNV-RP-C201) | U_num/f | U_input | U_val | |E|/U_val | verdict |
|---|---|---|---|---|---|---|---|
| Ritz n_terms=6 (converged) | 297.896 MPa | 297.896 MPa | 5.06 % | 18.33 MPa | 23.73 MPa | 1.2e-13 | PASS |
| Ritz n_terms=3 (under-resolved) | 382.631 MPa | 297.896 MPa | 18.2 % | 18.33 MPa | 72.02 MPa | 1.18 | FAIL |

The UQ verdict for the buckling model is **PASS** with `U_input` taken from the
certified PCE variance (thickness 3 % CoV dominates: `S1(t) = 0.939`).
The converged series agrees with the closed-form reference to 2.9e-06 Pa
(relative 9.8e-15), while the deliberately
under-resolved 3-term series misses by 28 % and the V&V gate correctly returns **FAIL** with
`|E|/U_val = 1.18`: the same machinery that certifies a good model
refuses to certify a bad discretisation.


### 3.6 Gate probe matrix

| probe | verdict | reason codes |
|---|---|---|
| nominal certified study | PASS | all checks satisfied |
| NaN in model output | WITHHELD | NONFINITE_MODEL_OUTPUT, LOO_ERROR_UNAVAILABLE, CROSS_CHECK_MISSING |
| sample budget below floor | WITHHELD | INSUFFICIENT_SAMPLES, LOO_ERROR_UNAVAILABLE, CROSS_CHECK_MISSING |
| eps_LOO above 0.1 | WITHHELD | LOO_ERROR_ABOVE_THRESHOLD, CROSS_CHECK_MISSING |
| adapter exceptions | WITHHELD | NONFINITE_MODEL_OUTPUT, ADAPTER_FAILURES, LOO_ERROR_UNAVAILABLE, CROSS_CHECK_MISSING |
| cross-check too imprecise | WITHHELD | CROSS_CHECK_INCONCLUSIVE |

## 4. Figures

**fig_loo_convergence.png** — Left: eps_LOO falls monotonically with the PCE degree (Spearman = -1.0 for both seeds), four orders of magnitude below the WITHHELD threshold. Right: the sample-count direction does not hold for a saturable model -- eps_LOO approaches the noise floor from below, so the sample budget is enforced a priori.

![fig_loo_convergence.png](artifacts/uq_pce_sobol_gate/fig_loo_convergence.png)

**fig_indices_comparison.png** — Ishigami S1/ST from the exact decomposition, the sparse PCE and the Saltelli/Jansen estimator. x3 has zero first-order effect but a 0.24 total effect due to its interaction with x1: a truncated or low-order surrogate cannot reproduce this pattern.

![fig_indices_comparison.png](artifacts/uq_pce_sobol_gate/fig_indices_comparison.png)

**fig_engineering_indices.png** — Certified first-order and total Sobol' indices of the Lindqvist level-ice resistance with respect to five uncertain inputs; error bars are the 95% coefficient-covariance intervals. sum(ST) > 1 quantifies the input interactions (the Lindqvist law is multiplicative in h^2 and sigma_c).

![fig_engineering_indices.png](artifacts/uq_pce_sobol_gate/fig_engineering_indices.png)

**fig_gate_probes.png** — Gate probe matrix. NaN output, exhausted sample budget, eps_LOO above 0.1, adapter exceptions and an imprecise cross-check each map to a WITHHELD verdict with a named reason code; only the nominal study is certified.

![fig_gate_probes.png](artifacts/uq_pce_sobol_gate/fig_gate_probes.png)


## 5. Notes, traps and limitations

1. **Two independently seeded Sobol' sets are not independent.** Building the Saltelli design from
   two scrambled `scipy.stats.qmc.Sobol` engines with different seeds produced column correlations of
   up to **0.94** between `A[:,j]` and `B[:,j]`, which corrupted the estimators (S1(x1) came out as
   0.305 under a design that should have given 0.314, and ST(x3) as 0.366 instead of 0.244, i.e. a
   spurious FAIL). Fixed by splitting one `2d`-dimensional sequence into halves; the measured
   `ab_max_abs_column_correlation` is now reported in every results file (2e-7 here).
2. **Post-selection LOO error.** `eps_LOO` is computed for the *selected* support, so the minimum over
   the OMP path is optimistically biased, and the bias shrinks with n (section 3.3). It is a model
   *selection* diagnostic, not a proof of accuracy; the cross-check exists for that reason.
3. **PCE confidence intervals cover regression noise only** (they are ~1e-5 wide here because the
   surrogate is almost exact). Truncation bias is reported separately as `eps_LOO`; the two are never
   merged into a single 'uncertainty'.
4. **Cross-check cost/semantics.** A 5 % agreement cannot be demonstrated with a small base sample
   count: at n_base = 256 the gate returns WITHHELD (`CROSS_CHECK_INCONCLUSIVE`) rather than accepting
   or rejecting. For expensive solvers the policy can waive the cross-check
   (`require_cross_check=False`, used for the 13 ms-per-evaluation Ritz model in section 3.5); the
   waiver is recorded in the verdict.
5. **Correlated inputs and PCE.** The Gaussian copula is supported for sampling, but the product basis
   needs independent inputs; a correlated model yields WITHHELD (`CORRELATED_INPUTS_UNSUPPORTED`)
   instead of an invented product basis. The supplied matrix is treated as the *latent Gaussian*
   correlation - no Nataf correction is applied, because that would silently change the user's matrix.
6. **Kinks limit convergence.** For the Sobol' g-function (8 inputs, `|4x-2|` kink) the PCE reaches
   only ~6e-3 absolute error at degree 6, so no 1e-3 claim is made for kinked responses; the test suite
   asserts the measured 1e-2 envelope instead.
7. **Absolute engineering accuracy remains WITHHELD.** The Lindqvist case is validated only against
   its own published scaling laws and the L2 anchor corpus; there is no full-scale measured ice
   resistance in this repository, so no experimental error percentage is quoted for `R_total`.
8. **Not a solver.** Nothing here touches the solver internals: the adapter is the only interface, so
   the method applies unchanged to TensorLBM/FVM/FEM/DEM/BEM outputs.


## 6. Error judgement against references

| quantity | reference | error | < 3 % ? |
|---|---|---|---|
| Ishigami S1 (PCE) | exact closed form | 1.1e-06 | yes (<3 %) |
| Ishigami ST (PCE) | exact closed form | 1.1e-06 | yes (<3 %) |
| Ishigami S2(x1,x3) (PCE) | exact closed form | 9.7e-07 | yes (<3 %) |
| Ishigami S1 (Saltelli) | exact closed form | 0.175 % rel | yes (<3 %) |
| Ishigami ST (Saltelli) | exact closed form | 0.0335 % rel | yes (<3 %) |
| PCE vs Saltelli (Ishigami) | Saltelli estimate | 0.175 % / 0.0331 % rel (S1/ST) | yes (<3 %) |
| Ritz sigma_cr (n=6) vs DNV-RP-C201 | closed-form design formula | 9.8e-15 rel | yes (<3 %) |
| Ritz sigma_cr (n=3) vs DNV-RP-C201 | closed-form design formula | 0.284 rel | NO (28 %, correctly FAILed by the gate) |
| PCE surrogate vs solver (ice) | n = 1024 sampled evaluations | eps_LOO = 6.5e-04 (RMS/1sigma = 2.54 %) | yes (<3 %) |
| R_total (Lindqvist) vs experiment | no full-scale measurement in repo | not computable | WITHHELD (stated, not hidden) |

Every smooth-case comparison is orders of magnitude inside the 3 % engineering band; the only
comparison that exceeds it (the 3-term Ritz series) is an intentionally under-resolved control that
the V&V gate rejects, and the one that cannot be computed (Lindqvist vs full-scale measurement) is
declared WITHHELD rather than glossed over.


---

### Provenance

- run generated: 2026-11-08T12:32:25.476499+00:00, wall clock 45.59 s
- sample digest (verification case): `9070733322e059c67e722914a3dd0ec4e23351f528b96320c5440146ea1aa3cf`
- Saltelli design digest: `10fb0054ef8f992a43fe6024c6c2630925fcb94154d5da7a235549e23d5b5b90`
- environment: Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38, Python 3.11.6, numpy 2.4.6
- raw artefact: `artifacts/uq_pce_sobol_gate/results.json`
