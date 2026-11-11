# Cross-clock coupling of TensorLBM and TensorDEM: implementation-level frozen contract

**Mission** m14792 · **anchor** GAP-CROSS-CLOCK (round 2) · **repo** TensorPhysics ·
**module** `src/tensorphysics/cross_clock/library_binding.py` ·
**demo** `scripts/run_cross_clock_library_demo.py` ·
**record** `runs/m14792_cross_clock_library/results.json` (+ `SHA256SUMS`) ·
**tests** `tests/test_cross_clock_library_binding.py` (18 passed).

Round 1 (m14677) failed on this anchor without leaving a `fail_stage`. The
round-2 rule therefore was: **bind the semantics to the code that actually runs**,
not to an in-repo stand-in instrument. `cross_clock.contract` (m13863) already
stated the semantics abstractly with reference solvers; this round replaces the
stand-ins by the shipped libraries and freezes the interface against them.

## 1. What is frozen (three places where cross-library coupling drifts)

| aspect | frozen by | failure mode it prevents |
| --- | --- | --- |
| clock | `SolverClock` + `SyncSchedule`; window edges are `t0 + k*h` from the **integer** index `k` | `+= dt` accumulation, legs landing on different sync points |
| units | `UnitMap(dx, dt, rho0)`; momentum scale `rho0 dx^3/dt`, force `= momentum/dt` | lattice/SI mixing, momentum and force using inconsistent exponents |
| exchange | `TensorLBMLeg.advance` distributes **exactly** the impulse the partner integrated | lost/duplicated share, point-value interpolation of an integral |

Anti-drift pins (`FROZEN_PINS`): `tensorlbm` 0.3.0
(`sha256 9fe8ab2ca05a271a…`), `tensordem`
0.1.0 (`sha256 28c3bb41c68396fb…`),
plus the signature of every attribute the binding calls. `verify_pins()` returns
the drift list; the test asserts it is empty and that a tampered pin produces a
named message. A library upgrade that moves `stream`, `collide_bgk`,
`macroscopic`, `DEMKernel.step` or `DEMKernelConfig` cannot silently change what
"advance" means.

## 2. The two libraries, minimally coupled

* **fluid leg** — `tensorlbm` D2Q9 BGK. The leg owns the population array and
  advances it with the library's `collide_bgk` / `stream` / `macroscopic`. An
  external impulse enters through the D2Q9 momentum kick `f_i += 3 W_i (c_i.dp)`
  (`sum W c_a c_b = delta_ab/3`, `sum W c = 0`), so momentum in equals momentum
  booked and the mass is untouched.
* **particle leg** — `tensordem.coupling_kernel.DEMKernel` with `drag_model='stokes'`,
  no gravity, no walls: the only force is `-c (v - u_f)`, `c = 6 pi mu r`.
  `DEMKernelConfig` is frozen upstream, so the leg rebuilds the kernel at each
  window from its own `(x, v)`; `rebuild_roundtrip_error() == 0.0` is asserted so
  the rebuild cannot move the state.

Parameters (from `results.json`, injected programmatically):
dx = 1.0e-04 m, tau = 0.8, nu = 1e-06 m^2/s,
dt_lbm = 1.0e-03 s, dt_dem = 5.0e-05 s,
64x8 periodic cells, u_f0 = 1.0e-03 m/s,
particle r = 2.1e-03 m, rho_p = 1000 kg/m^3,
mu = 1.0e-03 Pa s, T = 0.2 s.

## 3. Library anchors (each leg alone, against a third-party closed form)

| anchor | measured | reference | relative error |
| --- | --- | --- | --- |
| tensorlbm Taylor-Green decay (ratio measured/analytic `2 nu k^2`) | 1.003838 | 1 | 3.84e-03 |
| tensordem Stokes relaxation `u + (v0-u)e^(-t/tau_p)` | 1.846063170e-04 m/s | 1.846041945e-04 m/s | 1.15e-05 |

The Taylor-Green anchor is what fixes `nu = (tau - 1/2)/3 dx^2/dt` in `UnitMap`:
with the `(tau - 1/3)/3` variant the same measurement is off by a factor 1.55
(0.45 relative), which no tolerance can hide.

## 4. Coupled runs over three clock ratios

Reference solution for the column "vs exact ZOH": the exact solution of the *same*
piecewise-constant coupling problem `v_(k+1) = u_k + (v_k - u_k) exp(-h/tau_p)`
with the measured fluid velocities `u_k` — i.e. the error attributable to the DEM
quadrature and the exchange, not to the hold itself.

| case | h [s] | r (lbm:dem) | windows | ledger drift | system momentum residual (measured) | stagger residual (max) | vs exact ZOH (v) | reference drift vs finest window (v) | clock round-trip |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fine | 1e-03 | 1:20 | 200 | 0.00e+00 | 1.05e-11 | 7.73e-06 | 1.85e-05 | n/a (reference) | yes |
| medium | 5e-03 | 5:100 | 40 | 7.22e-18 | 1.04e-11 | 3.86e-05 | 4.63e-05 | 1.55e-05 | yes |
| coarse | 1e-02 | 10:200 | 20 | 2.17e-17 | 1.04e-11 | 7.69e-05 | 8.12e-05 | 3.47e-05 | yes |

Reading of the table:

* **ledger drift 0 … 2e-17**: produced and applied impulses agree to round-off
  across a 1:20 and a 1:200 clock ratio.
* **measured system momentum residual ~1e-11**: an *independent* check, not the
  book itself: `dp_fluid + dp_particle` taken from the libraries' own state over
  the whole run, relative to the fluid's initial momentum.
* **stagger residual and reference drift grow with h** (1.8e-5 → 8.1e-5 and
  1.5e-5 → 3.5e-5 in v): the physics cost of holding the fluid velocity over a
  longer window, reported separately from the ledger on purpose.
* **clock round-trip**: step counts equal `n_windows * r` for both legs, both leg
  times equal the schedule time to <1e-15 s, sync coincidence error <=
  5.6e-17 s.

## 5. Limitations (stated, not hidden)

1. The reaction impulse is distributed **uniformly** over the fluid cells: this is
   a momentum-bookkeeping demo, not a spatially resolved immersed-boundary FSI
   (the distribution kernel is the next gap, not this one).
2. The fluid leg is periodic D2Q9 without walls; no-slip geometry is out of scope.
3. `tensorlbm` builds its stencil constants in float32, so the momentum
   round-off floor of a single kick is ~1e-6 relative
   (`test_lbm_leg_momentum_kick_is_exact` measures it); the *cumulative*
   bookkeeping stays at 1e-11 because the kick algebra itself is exact.
4. Single particle, Stokes drag (Re << 1); the Schiller-Naumann path of the DEM
   kernel is not exercised here.
5. `tensordem`'s frozen `DEMKernelConfig` forces a kernel rebuild per window; the
   rebuild is state-neutral (asserted) but it resets the library's internal
   dissipation ledger, which this demo does not use.

## 6. Reproduce

```
python scripts/run_cross_clock_library_demo.py        # writes runs/m14792_cross_clock_library/
python -m pytest tests/test_cross_clock_library_binding.py -q
```
