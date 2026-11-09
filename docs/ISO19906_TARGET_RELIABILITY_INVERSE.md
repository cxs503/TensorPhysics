# ISO 19906 ice-force partial factors: the target-reliability inverse (IN101-N09)

**Node** IN101-N09 ｜ **Repo** TensorPhysics ｜ **Module**
`src/tensorphysics/ice_partial_factors/` ｜ **Report**
[`docs/reports/in101_n09/iso19906_target_inverse.md`](reports/in101_n09/iso19906_target_inverse.md) ｜
**Numbers** `artifacts/iso19906_target_inverse/in101_n09_results.json` ｜
**Tests** `tests/test_iso19906_target_inverse.py` (27 tests) ｜ **Runner**
`benchmarks/iso19906_partial_factors/run_in101_n09.py`

## What problem this solves

The predecessor node (IN13-W3-N10) calibrated a *single* factor product
`k = gamma_F gamma_M` for a plate-stress limit state. A design format, however, is
a *pair* `(gamma_f, psi)`:

```
gamma_f * F_k  <=  R_k / psi        <=>       R_k >= kappa F_k,  kappa = gamma_f psi
```

Only `kappa` is convention-free; the split into a load and a resistance factor is
a convention. IN101-N09 therefore solves the inverse as a **two-unknown /
two-equation system**

```
F1(gamma_f, psi) = beta(kappa = gamma_f psi) - beta_T = 0     (code calibration)
F2(gamma_f, psi) = gamma_f - F*(kappa)/F_k            = 0     (JCSS/ISO 2394 split)
```

in log coordinates `y = (ln gamma_f, ln psi)` by damped Newton, and verifies the
result five ways (closed-form index, FORM, importance sampling, crude Monte
Carlo, Rao-Blackwellised crude Monte Carlo).

## Why a new physical state (not a rerun of IN13)

IN13's limit state is driven by the *plate bending stress*, which is a function of
the local pressure only — the contact area drops out and with it every statement
about the pressure-area correlation. The mature node uses the **ice force**
`F = k_F p(A) A` with the ISO power law `p(A) = p_0 (A/A_0)^{-ex}`, because that
is (i) the quantity ISO 19906 actually tabulates partial factors for and (ii) the
only place where the negative pressure-area correlation is physically active:
`rho(ln p, ln A) = -ex sigma_lnA / sigma_lnp`, so

```
Var_corr (ln F) = sigma_kF^2 + sigma_eps^2 + (1-ex)^2 sigma_lnA^2
Var_indep(ln F) = sigma_kF^2 + sigma_eps^2 + (1+ex^2) sigma_lnA^2
```

with the difference `2 ex sigma_lnA^2` being exactly the quantity the negative
control of the mission has to detect.

## Implementation notes (the two traps worth recording)

1. **Nataf takes a physical correlation; the ISO law fixes the correlation of the
   logarithms.** Passing `rho_ln` straight into `NatafTransform` produces a model
   whose log-correlation is *not* the ISO one: the first implementation then had
   FORM and importance sampling agreeing to 0.07 % while both disagreed with the
   closed-form index by 3.0 % — i.e. the *reference* was wrong, not the solver.
   `PAreaForceModel.pressure_area_pearson()` inverts the exact lognormal relation
   `rho_phys = (e^{rho_ln s1 s2}-1)/sqrt((e^{s1^2}-1)(e^{s2^2}-1))`, and the
   transform's solved `rho0` reproduces `rho_ln` to 1e-8.
2. **On this problem FORM is exact.** The failure domain `{R < F}` is a half-space
   in `z = (ln x - mu)/sigma`, and `u = L^{-1} z` is linear, so the Hasofer-Lind
   distance equals the closed-form index `beta_ln = (mu_lnR - mu_lnF)/sqrt(sigma_lnR^2 + sigma_lnF^2)`
   (agreement 6e-9). Consequence for verification strategy: "FORM vs Monte Carlo
   within 3 %" can only test *implementation consistency* here, not FORM's
   curvature error; the report says so explicitly instead of implying more.

## Calibrated result (default state, beta_T = 4.0)

| quantity | value |
|---|---|
| `gamma_f*` | 1.4371 |
| `psi*` | 0.9960 |
| `kappa*` | 1.4313 |
| achieved beta | 4.000000 |
| Jacobian condition number | 4.79 |
| three-start spread in `gamma_f*` | 7.9e-11 |
| Jacobian | `[[3.49, 3.49], [0.239, -1.130]]` |

The Jacobian is *not* rank-deficient even though the limit state only sees
`kappa`: the split equation depends on `gamma_f` alone, which breaks the
degeneracy — the smallest singular value is `O(1)` and the condition number stays
below 5.

## Related documents

* [`docs/ICE_NAV_REFS_CORPUS.md`](ICE_NAV_REFS_CORPUS.md) — ice-action reference corpus
* [`docs/CRITERIA_MUTATION_LEDGER.md`](CRITERIA_MUTATION_LEDGER.md) — falsifiability ledger for acceptance gates
* `docs/reports/in13_w3n10/iso19906_partial_factors.md` — predecessor node (one-parameter inverse, plate limit state)