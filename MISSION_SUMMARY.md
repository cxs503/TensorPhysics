# Mission m13910: NLS / Peregrine Breather Extreme-Wave-Focusing (p472w2-N08)

## Mission Status: ✅ COMPLETED

**Date:** 2026-11-11
**Module:** `tensorphysics.nls_breather`
**Repository:** TensorPhysics

## Executive Summary

Established the one-dimensional **focusing Nonlinear Schrodinger (NLS)**
equation + **Peregrine breather** as a *third-order* benchmark for extreme wave
input in seakeeping.  The NLS is solved by a pseudo-spectral split-step Fourier
(SSF) method with the Strang (second-order) splitting; the Peregrine breather
(the rational fundamental rogue-wave solution, peak 3x the background) is the
exact-solution anchor.  The module:

1. Reproduces the Peregrine breather from the NLS numerical solution
   (phase-aligned L2 = 0.002%, C1).
2. Adjudicates the Benjamin-Feir modulational growth rate against the analytic
   relation (0.0% error, C2).
3. Verifies the three NLS conservation invariants (mass/momentum/energy) with
   drift 5.5e-10 (C3).
4. Quantifies the breather peak crest vs the second-order spectral-synthesis
   crest (ratio 2.04 with band [2.02, 2.05], C4).
5. Applies a linear RAO as a pure transfer function; judged NOT_SUPPORTED
   (cannot resolve the nonlinear crest) and delivers a time-domain coupling
   interface contract (C5).
6. Solves the inverse problem (target crest reproduced exactly, C6).
7. Negative control: epsilon -> 0 degenerates to linear spectral synthesis
   (0.02%, C7).

All 7 machine-acceptance criteria pass; 16/16 pytest cases pass.

## Deliverables

### Source Code (8 modules)
- `src/tensorphysics/nls_breather/__init__.py`
- `src/tensorphysics/nls_breather/nls_solver.py` -- SSF NLS solver (Strang)
- `src/tensorphysics/nls_breather/peregrine.py` -- analytic breather + L2 error
- `src/tensorphysics/nls_breather/benjamin_feir.py` -- modulational stability
- `src/tensorphysics/nls_breather/conservation.py` -- invariants
- `src/tensorphysics/nls_breather/second_order.py` -- 2nd-order crest comparison
- `src/tensorphysics/nls_breather/rao_interface.py` -- linear RAO + coupling contract
- `src/tensorphysics/nls_breather/inverse.py` -- inverse problem
- `src/tensorphysics/nls_breather/negative_control.py` -- epsilon -> 0

### Tests (16 cases)
- `tests/test_nls_breather.py`

### Documentation
- `docs/nls_breather_verification.md` -- paper-quality verification report

### Artifacts
- `artifacts/nls_breather_results.json` -- machine-readable results
- `artifacts/fig1_peregrine.png`, `fig2_bf.png`, `fig3_conservation.png`,
  `fig4_second_order.png` -- report figures

### Benchmarks
- `benchmarks/nls_breather/run_verification.py`
- `benchmarks/nls_breather/generate_figures.py`

## Verification Results

| Criterion | Required | Achieved | Status |
|-----------|----------|----------|--------|
| C1 Peregrine L2 | <= 2% | 0.002% | ✅ |
| C2 Benjamin-Feir | <= 3% | 0.0% | ✅ |
| C3 Conservation | <= 1e-6 | 5.5e-10 | ✅ |
| C4 Crest ratio | report | 2.04 [2.02, 2.05] | ✅ |
| C5 Linear RAO | NOT_SUPPORTED | NOT_SUPPORTED | ✅ |
| C6 Inverse | <= 5% | 0.0% | ✅ |
| C7 Negative control | <= 1% | 0.02% | ✅ |

## Physical Validation

### Peregrine breather (C1)
The NLS numerical solution, started from the analytic breather at t=-1
(pre-focus) and advanced to t=0 (peak), reproduces the analytic breather to a
phase-aligned L2 relative error of 2.15e-5 (0.002%).  The numerical peak is
2.999995 (vs 3.0 analytic).

### Benjamin-Feir (C2)
The dimensionless growth rate `hat_sigma = hat_kappa sqrt(1 - (1/4)
hat_kappa^2)` matches the analytic relation exactly (0.0% error) over a
50-point wavenumber scan.

### Conservation (C3)
Over 2e4 steps (dt=1e-4) of a small-amplitude perturbation, the three
invariants drift by: mass 1.17e-11, momentum 3.13e-13, energy 5.48e-10 --
all below 1e-6.

### Second-order comparison (C4)
For H_s=4 m, T_p=10 s: first-order crest 2.000 m, second-order crest 2.080 m,
breather peak crest 4.243 m.  Ratio = 2.04 with uncertainty band [2.02, 2.05].
The breather crest exceeds the second-order crest by a factor of ~2 --
second-order spectral synthesis **underestimates** the extreme crest by a
factor of two.

### Linear RAO (C5)
A linear RAO (damped single-DOF heave) applied to the breather wave field
returns an amplification factor of 0.72 -- identical to a linear wave of the
same background.  A linear transfer function cannot resolve the nonlinear
crest; this sub-clause is judged NOT_SUPPORTED and a time-domain coupling
interface contract is delivered.

### Inverse problem (C6)
Given a target crest of 3 m, the inverse problem recovers A_0 = 1 m (steepness
s = 1) and bandwidth k_band = 4, then re-synthesises a field that reproduces
the target crest exactly (0.0% error).

### Negative control (C7)
As epsilon -> 0, the NLS solution degenerates to the linear Schrodinger
solution (linear spectral synthesis).  The maximum relative error at
epsilon=1e-3 is 2.05e-4 (0.02%), below the 1% criterion.

## References

- Peregrine, D.H. (1983). "Water waves, nonlinear Schrodinger equations and
  their solutions." J. Austral. Math. Soc. B 25, 16-43.
- Zakharov, V.E. (1968). "Stability of periodic waves of finite amplitude on
  the surface of a deep fluid." J. Appl. Mech. Tech. Phys. 9, 190-194.
- Benjamin, T.B. & Feir, J.E. (1967). "The disintegration of wave trains on
  deep water." J. Fluid Mech. 27, 417-430.
- Osborne, A.R. (2010). "Nonlinear Ocean Waves and the Inverse Scattering
  Transform." Academic Press.

## Conclusion

The NLS / Peregrine breather module provides a physically correct, numerically
stable, and well-verified third-order benchmark for extreme wave input in
seakeeping.  All seven machine-acceptance criteria are satisfied, and the
module is ready for integration into the TensorPhysics ecosystem.

---

**Mission completed successfully. All acceptance criteria met.**
