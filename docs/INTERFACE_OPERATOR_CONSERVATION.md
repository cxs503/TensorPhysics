# Cross-solver interface operators: conservation, order and symmetry contract

**Campaign node** `IN19-T2` · **module** `tensorphysics.interface_operators`
· **mission** 5207 · acceptance digest `6a0b506c8c97d1b6`

This document defines the *contract* that every cross-solver interface operator
in TensorPhysics must satisfy, implements three of them (mortar projection, RBF
scatter, LBM–DEM momentum exchange), and reports machine-checkable acceptance
numbers on five analytic fields plus two two-solver chains.

---

## 1. Physics background

"When two solvers are glued together" the exchange object is an *operator*
`T : V_src -> V_tgt` acting between two interface discretisations. Three
properties decide whether the coupled model is trustworthy:

| property | physical meaning | diagnostic |
|---|---|---|
| conservation | the flux the donor sends is the flux the acceptor receives; otherwise the coupled system creates/destroys mass, momentum or energy at the interface | `conservation_residual` |
| consistency | the transfer does not degrade the solver's own approximation order (no artificial first-order interface) | `observed_order` |
| symmetry | the operator does not depend on the interface labelling (mirror) or on the visiting order | `symmetry_flip` |

For a Dirichlet–Neumann style coupling the first two are independent: a
conservative operator can be inconsistent (donor-cell averaging) and a
consistent operator can be non-conservative (point interpolation). Both failures
produce *plausible-looking* but wrong coupled solutions, which is why the gates
below are numeric and hard-coded.

## 2. The contract

```python
from tensorphysics.interface_operators import InterfaceOperator

class MyOperator(InterfaceOperator):
    def transfer(self, src, src_values, tgt): ...      # the only mandatory method
```

`InterfaceOperator` provides, identically for every operator (single source of
truth — the acceptance harness and the tests call the same code):

* `conservation_residual(src, u, tgt, v=None) = |∫_tgt v - ∫_src u| / ∫_src |u|`
* `observed_order(field, pairs)` → `OrderEstimate(resolutions, errors, order, exact)`
  from a coarse→fine hierarchy; `order is None` + `exact=True` when the transfer
  is exact to round-off (the `p -> inf` limit, which is the mathematically honest
  bookkeeping for linear data under any consistent ≥2nd-order operator).
* `symmetry_flip(src, u, tgt)` → `SymmetryFlip(reflection, roundtrip)`,
  `reflection = ||T(Ru) - R(Tu)|| / ||T(Ru)||` about the interface mid-axis,
  `roundtrip = ||T_BA(T_AB u) - u|| / ||u||`. The headline `value` is the
  reflection defect; `roundtrip` is reported but is **not** a defect for a dual
  pair (an exchange operator and its adjoint are not inverses — see §6).
* `evaluate(field, src, tgt, pairs)` → `OperatorReport` (JSON-ready).

Interfaces are 1D segments (`SegmentMesh`) or point clouds (`PointSet`). This is
deliberate: 1D overlap integrals are analytic, so the diagnostics are exact and
the gates are reproducible to round-off. Tensor-product 2D/3D extensions reuse
the same contract; they are **not** implemented here (see §6).

## 3. The three operators

### 3.1 `MortarProjection` (area-weighted / mortar)

Target cell mean

```
v_j = (1/|T_j|) Σ_i ∫_{S_i ∩ T_j} [ u_i + s_i (x - x_i^c) ] dx ,
s_i = (u_{i+1} - u_{i-1}) / (x_{i+1}^c - x_{i-1}^c)   (central, one-sided at ends)
```

Because the overlaps partition each source cell exactly, `Σ_j v_j |T_j| = Σ_i u_i |S_i|`
for **any** data — conservation is structural, not enforced. The slope term
reproduces linear data exactly (second-order consistency), including on graded
and non-matching meshes. Matrix form `A = D + G·S` is available
(`matrix()`/`dense_matrix()`) for audits.

### 3.2 `ConservativeScatter` (RBF / MLS scatter)

Gaussian kernel (`ε = 1.8 h_src`) with local *linear* (moving-least-squares)
augmentation for consistency, followed by an explicit conservative deficit fix
`v ← v + (∫_src u − ∫_tgt v)·|T| / Σ|T|²` so that the received integral matches
the sent one to round-off. The fix is **global** (a documented limitation: local
conservation is not guaranteed, only the total).

### 3.3 `LBMDEMMomentumExchange` (ME)

* gather (`nodes -> particles`): kernel normalised **per node**, with a
  Tikhonov-regularised moment correction `λ = q(1 + γ(d - d̄))`,
  `γ = -d̄/(var + h²/16)`, so that each node hands out 100 % of its momentum and
  its distribution is centred on the node;
* spread (`particles -> nodes`): the same kernel normalised **per particle**, so
  each particle hands back 100 % of its impulse;
* the particle exchange volume `V_p = Σ_l λ_lp w_l` makes the two sides
  measure-consistent, and the reaction returned to the lattice is the negative
  of what the DEM side took, closing the momentum budget exactly (C4).

## 4. Acceptance results

Reproduce with

```bash
PYTHONPATH=src python3 -m pytest tests/test_interface_operators.py -q
PYTHONPATH=src python3 benchmarks/interface_operators/run_interface_operator_benchmark.py
```

Numbers below are copied from `artifacts/m5207/interface_operator_acceptance.json`
(digest `6a0b506c8c97d1b6`, generated by the harness itself — no hand transcription).

| gate | criterion | measured | verdict |
|---|---|---|---|
| C1 | constant field, conservation residual < 1e-14 | mortar 1.27e-16 · scatter 1.27e-16 · ME 0.0 | **PASS** |
| C2 | observed order ≥ 1.9 (remap operators) | linear field exact (max err mortar 1.4e-16, scatter 2.1e-16 ⇒ p→inf); quadratic field p = **2.12** (mortar), **2.37** (scatter) | **PASS** (ME excluded, §6) |
| C3 | total-flux residual / Σ\|flux\| < 1e-12 over all five fields | worst 2.70e-16 (ME, harmonic); mortar ≤1.40e-16, scatter ≤2.31e-16 | **PASS** |
| C4 | ME write-back closed momentum budget < 1e-10 | worst 3.61e-16 (linear field) | **PASS** |
| C5 | FVM↔LBM and LBM↔DEM post-exchange field difference < 2 % | FVM↔LBM 0.141 % · LBM↔DEM 0.957 % (worst = boundary layer) | **PASS** |
| C6 | each operator's negative control must be red | mortar\[midpoint] trips C3 (4.0e-4)/C2 (p=1.88); scatter\[no_fix] trips C3 (6.9e-3)/C2 (p=1.89); ME\[per_node_norm] trips C4 (1.95e-3) | **PASS** |

`symmetry_flip` (reflection equivariance) is ≤ 6.4e-15 for all three operators on
all five fields.

### Per-field transfer table (160 target cells, graded donor → uniform acceptor)

| field | mortar residual \| L2 | scatter residual \| L2 | ME residual \| L2 |
|---|---|---|---|
| constant | 1.27e-16 \| 8.9e-17 | 1.27e-16 \| 1.9e-16 | 0.0 \| 2.3e-16 |
| linear | 0.0 \| 1.4e-16 | 0.0 \| 1.7e-16 | 0.0 \| 2.3e-04 |
| quadratic | 1.40e-16 \| 1.5e-06 | 0.0 \| 2.1e-05 | 2.70e-16 \| 4.1e-04 |
| harmonic | 8.72e-17 \| 7.2e-06 | 8.72e-17 \| 1.2e-03 | 6.66e-17 \| 7.0e-04 |
| boundary layer | 1.16e-16 \| 4.4e-05 | 2.31e-16 \| 6.4e-03 | 0.0 \| 8.4e-03 |

### Observed orders over the 40/80/160 hierarchy

| field | mortar | scatter | ME (pointwise) |
|---|---|---|---|
| constant | exact | exact | exact |
| linear | exact (p→inf) | exact (p→inf) | 1.36 |
| quadratic | 2.12 | 2.37 | 1.27 |
| harmonic | 1.88 | 1.91 | 1.33 |
| boundary layer | 1.85 | 1.58 | 1.31 |

## 5. The two chains

* **FVM↔LBM**: FVM 44 cells on [0, 0.55] (h = 0.0125), D1Q3 lattice 55 cells on
  [0.45, 1.0] (h = 0.0100); the overlap [0.45, 0.55] holds 8 vs 10 cells, i.e.
  genuinely non-matching. Exchange both directions; conservation residual ≤ 1.4e-16,
  field difference ≤ 0.141 % (boundary layer). The cell-mean ↔ lattice-node
  (point value) mismatch is separately reported as an O(h²) diagnostic
  (`cell_average_vs_node_value`) because a D1Q3 lattice stores point values while
  FVM stores cell means.
* **LBM↔DEM**: D1Q3 window [0.2, 0.8], 60 nodes, 60 particles (lattice + 25 %
  jitter, a resolved packing). Momentum gathered onto particles, reaction spread
  back: gather residual ≤ 4.1e-16, closed momentum budget ≤ 4.1e-16, field
  difference ≤ 0.957 %.

Sensitivity (reported, **not** gated): a strongly clustered Poisson cloud
(60 particles, same window) gives field differences 0.34 % (linear), 0.53 %
(quadratic), 1.70 % (harmonic) and **4.79 % (boundary layer)** — i.e. **the 2 %
target is not met for a clustered packing on a steep field**. The mechanism is
physical: with `ε = 1.2 h` a cluster leaves gaps larger than the kernel width, so
the local interpolation error is set by the *particle spacing*, not by `h`.
Recommended practice: `ε ≥ 2 ×` the local particle spacing.

## 6. Limitations and applicability (honest scope)

1. **ME is not a remap and is first order pointwise.** A compact-kernel momentum
   exchange cannot be simultaneously exactly conservative *and* second-order
   pointwise without a global (non-local) correction; the shipped ME is
   conservative-first (C1/C3/C4 at round-off) with pointwise order **1.36**
   (linear field). C2's `p ≥ 1.9` is therefore asserted for the two remapping
   operators only; the ME entry in the payload carries `applicable: false` and a
   `scope_note`. This is a **documented non-conformance to the blanket reading
   of C2**, not a pass.
2. **Geometry is 1D.** The contract, diagnostics and gates are dimension-agnostic,
   but only 1D segments and 1D point clouds are implemented; 2D/3D mortars
   (polygon clipping) and 3D DEM are future work. The 1D results do not validate
   3D efficiency or robustness.
3. **The RBF scatter conservation fix is global** (`+deficit·|T|/Σ|T|²`), so local
   (per-cell) conservation is not guaranteed; only the total is. Codes needing
   local conservation must use the mortar operator.
4. **The refinement sequence is 40/80/160 (R = 2).** For the harmonic and
   boundary-layer fields the fitted orders (1.85–1.88 mortar, 1.58 scatter) are
   not yet asymptotic; they are reported rather than tuned away, and C2 relies on
   the linear (exact) and quadratic (2.12/2.37) evidence.
5. **Chains isolate the operator.** No LBM collision/streaming and no DEM
   integration is performed inside the chain tests: the acceptance object is the
   interface operator, not the solvers' kernels.
6. **ME with clustered DEM packings** misses the 2 % target on steep fields
   (§5) — a kernel-width/packing issue, quantified but not solved here.

## 7. Traps hit while building this (for the next agent)

* A per-node moment correction on the **gather** kernel does *not* make particle
  values exact: it centres each node's donation, not each particle's receipt
  (`x̄_p ≠ x_p`). Attempts to force both cost 1–2 orders in condition number and
  produced 1e297 blow-ups before a Tikhonov-regularised, clipped slope correction
  made it bounded.
* `Σ_j (sent−recv)·|T_j| / Σ|T|` does **not** change the integral by
  `(sent−recv)`; the correct measure-proportional split is `·|T_j| / Σ|T|²`.
  A wrong normalisation leaves a 6.8e-3 "conservation residual" that looks like
  a physics error but is arithmetic.
* Reflecting the two sides of an interface must use a **common mirror axis**;
  reflecting each side about its own extent gives a spurious `symmetry_flip = 2`.
* A random particle cloud does not refine: its clustering noise saturates, and an
  order study on it reports a fake plateau (measured p ≈ 0 instead of 2). Use a
  lattice + O(h) jitter for convergence studies and the clustered cloud only for
  sensitivity.