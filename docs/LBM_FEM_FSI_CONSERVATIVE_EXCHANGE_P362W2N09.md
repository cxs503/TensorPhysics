# Conservative LBM → FEM field exchange: what must be true before a VIV number is believable

**Rule (short form).** A partitioned fluid–structure exchange is only
"conservative" if you can show, with three *independent* measurements, that
(i) the interface creates no mass, (ii) the impulse the fluid delivers equals the
impulse the structural loads deliver, and (iii) the work delivered equals the
work performed by the mapped loads. Anything less is a claim, not a measurement.

- **Scope / applicability**: partitioned FSI where an Eulerian lattice solver
  hands a surface traction to a Lagrangian/continuum FE structure, in the
  incompressible limit and at low-to-moderate Reynolds number, with the body
  described by bounce-back-type boundary conditions. Applies to
  `tensorphysics.lbm_fem_fsi` (node `p362w2-n09`) and to any sibling coupling
  that reuses its `interface`/`coupling` modules.
- **Not applicable to**: compressible solvers, sharp-interface cut-cell methods
  (different conservation algebra), point-particle (DEM) couplings where the
  exchange is volumetric, and to strongly added-mass-unstable regimes
  ($m^* \lesssim 1$) where a partitioned explicit exchange needs under-relaxation
  or a monolithic solver before any of this is meaningful.
- **Provenance**: campaign node `p362w2-n09-cross-solver-coupling`, repo
  `TensorPhysics`, branch `agent/m14011`; runs
  `runs/p362w2-n09-cross-solver-coupling/`, artifacts
  `artifacts/p362w2-n09/*.json`, acceptance payload
  `artifacts/p362w2-n09/acceptance.json`, report
  `docs/reports/p362w2_n09_lbm_fem_fsi/report.md`. Author: build-citizen agent
  `m14011`; reviewed by the node's machine criteria (see
  `acceptance.CRITERIA`). Last verified 2026-11-11.

## Mechanism — why the three audits are not the same number

1. **`mea` (local)** — Ladd momentum exchange, one force per boundary link,
   `sum_links (f_opp + f_i) e_i`. It never reads anything outside the interface.
2. **`budget` (integral)** — the residual of the global discrete momentum
   account of the lattice: streaming loss, wall reflection, inlet, outlet, body.
   For this to close, the collision must be momentum-neutral (BGK with an exact
   equilibrium), the streaming must be accounted for where it drops populations,
   and the body region must hold no fluid (zero the solid nodes, otherwise the
   "fluid" momentum inside the body contaminates the budget).
3. **`fem` (received)** — the impulse and work that the mapped nodal loads
   actually deliver.

Watch out: an *incomplete* budget is the most likely source of a fake defect.
The first version of this budget omitted the momentum that streams *into* the
body (it must be zeroed and booked), which produced an apparent 7.5 %
Ladd-vs-budget discrepancy that looked like a property of the interpolated
bounce-back and was in fact a bookkeeping hole. Before believing any
"non-conservative operator" result, close the ledger on a case where the answer
is known (uniform flow, static body).

Agreement of (1) and (2) is a statement about the *boundary operator*;
agreement of (2) and (3) is a statement about the *map*. Measured here: (1) vs
(2) at round-off ($\sim 10^{-14}$) with half-way bounce-back, and (2) vs (3) at
round-off with the window-averaged (impulse-conserving) exchange policy.

## Recommended procedure

1. Book every boundary operation separately (streaming loss, wall, inlet,
   outlet, body) and demand the mass ledger close to round-off. Any code path
   that silently creates mass then breaks the ledger immediately.
2. Verify the fluid solver on a case with a known answer *before* coupling:
   uniform flow must be a fixed point, and a fixed cylinder must reproduce
   $C_D$ and $St$ at the target Reynolds number.
3. Calibrate the rasterised body area against $\pi R^2$ (scan the mask radius);
   an uncalibrated $R - \Delta x/2$ under-sizes the body by $\sim 12\%$ at
   $D = 12$ and biases the drag by the same amount.
4. Use an impulse-conserving exchange (window-averaged force) in production —
   the delivered impulse is then exact at any step ratio — and keep the lagged
   policy as the reference case on which refinement studies are run, because an
   exactly conserving scheme cannot demonstrate convergence of a conservation
   error.
5. Accumulate exchanged work as the exact discrete work of a load held constant
   over a step, $F \cdot \Delta u$, not as a trapezoidal power integral: the
   $O(\Delta t^3)$ defect of the latter accumulates to a spurious few-percent
   energy non-conservation in a lightly damped resonant run.
6. Evaluate the conservation ledgers over the statistical window (after the
   start-up), and report the start-up separately.

## Counterexamples / traps (measured, with the symptom that reveals them)

| trap | symptom | consequence |
|---|---|---|
| Zou/He inlet built on raw populations instead of the mirrored non-equilibrium part | uniform stream is not a fixed point; inlet $u$ comes out 30 % low | drag 30 % low, everything downstream wrong |
| Bouzidi interpolated bounce-back for a moving body | exotic but *not* momentum-breaking: once the budget includes the momentum absorbed in the body region, its Ladd-vs-budget audit also closes (7.6e-14). What it does break is mass: a secular interface leak of ~2e-4 per 10^4 steps, zero for half-way bounce-back | keep it as an accuracy option; never use it when a long-run mass budget is a criterion |
| Refilling newly wetted nodes with the *wall* velocity | interface momentum several times the physical exchange impulse | apparent energy non-conservation of order unity |
| Reflecting wall populations *and* keeping the streamed values | mass doubles in the wall rows; solution blows up in ~100 steps | solver divergence |
| Comparing accumulated momenta with a sign convention chosen per-side | audit ratio of exactly 2 or 3 | always check the per-step difference, not only the sum |
| Trapezoidal power for the energy ledger | few-percent energy residual, independent of $\Delta t$ | misdiagnoses a work definition as a physics defect |

## Regression guard

`tests/test_lbm_fem_fsi.py` locks the invariances that caught every trap above:
uniform-flow fixed point, mass-ledger closure (< 1e-10 relative), Ladd-vs-budget
agreement (< 1e-9), zero interface mass defect of the half-way operator *and*
non-zero for the interpolated one, the geometry area gate, the Q4 patch test,
the Newmark energy identity, the resultant/moment-preserving mapping, the
Strouhal estimator, and two coupled smoke tests. `acceptance.py` freezes the
node criteria with thresholds so a stored payload can be re-gated without
re-running the physics.

## Supersession note

The earlier campaign work on the same exchange operators
(`docs/EXCHANGE_CONSERVATIVE_MAPPING.md`,
`docs/FEM_LBM_CONSERVATIVE_SCALAR_MAP.md`) covers the *static* traction →
nodal-load direction on conforming panels and the FEM → LBM scalar direction.
This note does not supersede it; it adds the *time-dependent, moving-body*
part: impulse and work conservation under sub-cycling, and the lattice-side
double measurement that makes "conservative" checkable rather than asserted.