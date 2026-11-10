# LBM–DEM two-way coupling: momentum conservation audit and error bound
### Campaign node `p367seed-n07` · domain `ice_navigation` · repository `TensorPhysics`

Module: `src/tensorphysics/lbm_dem_momentum/` · driver: `scripts/run_lbm_dem_momentum_audit.py`
Record: `artifacts/p367seed_n07/verification.json` (all numbers below are read from it) ·
Tests: `tests/test_lbm_dem_momentum.py` (17 passed).

---

## 1. 案例介绍 (physics, geometry, parameters)

The node extends the existing **energy contract** of `tensorphysics.coupling_energy`
(`P40W3B-N09`) to the **momentum contract** of an ice/water/fragmented-ice field:
LBM (water) ↔ DEM (ice floes, two-way), three cases plus a control arm and a
grid-refinement study.

All three cases are 2-D **horizontal-plane** problems. A floating floe is in
vertical equilibrium (buoyancy balances weight), so in the plane it carries **no
net external force** — which is exactly what makes the momentum contract
testable: any drift of the system momentum must be produced by the solver.

| case | configuration | momentum input |
|---|---|---|
| `floe_drift_decay` | one circular floe (R = L/8, ρ_i/ρ_w = 0.92) released at 2U in still water, closed box | **none** (free drift, drag only) |
| `ice_field_shear` | 8 floes in a Couette layer, south wall fixed, north wall at U, contacts (2 % nominal overlap) | moving wall impulse (tallied) |
| `ship_ice_push` | prescribed circular ship body advancing at 2U into 3 floes | ship impulse (tallied) |
| `drag_convergence` | prescribed-velocity floe (tow tank, no added-mass feedback) | diagnostic for the convergence study |
| `negative_control_lbm_only` | DEM off: pure LBM, Taylor–Green vortex, same box/grid/dt/window | none (exact conservation expected) |

Fixed physics across the refinement study (`gridscan.refinement_levels`):

| quantity | value |
|---|---|
| box length `L` | 0.1 m |
| reference speed `U` | 0.01 m/s |
| kinematic viscosity `ν` | 1e-5 m²/s (Re_L = 100) |
| lattice Mach `U dt/dx` | 0.02 (Ma = 0.035; measured max 0.046) |
| audit window `T` | 0.5 s (first 25 % discarded as the impulsive-start transient) |
| grid `n` | 64 / 96 / 128 → `dx` = 1.5625 / 1.0417 / 0.78125 mm |
| `dt` | 3.125 / 2.083 / 1.563 ms → 160 / 240 / 320 steps |
| `τ` | 0.5384 / 0.5576 / 0.5768 |
| floe mass | 0.92·πR² (lattice units, ρ_w = 1); contact `k = 0.5 m`, 8 sub-steps |

## 2. 怎么做的 (method, key settings, toolchain)

**Adjoint momentum exchange.** Full-way bounce back on every fluid/solid link with
the moving-wall correction `f_ī = f_i^post − 2 w_i ρ (c_i·u_w)/cs²`. The impulse
handed to a wall by one link is `ΔP = 2 c_i f_i^post − 2 w_i ρ (c_i·u_w) c_i/cs²`,
which is *exactly* the negative of what the fluid node lost relative to free
streaming. The exchange therefore cannot create momentum — this is a property of
the code, and it is what the `1e-10` criterion measures.

**Node conversion accounting.** When a floe moves, lattice nodes switch between
fluid and solid. Destroyed nodes hand their populations' momentum to the floe,
created nodes are initialised at `(ρ0, u_floe)` and the same momentum is
subtracted from the floe. Without this the moving staircase leaks momentum of
order `ρu` per conversion (~1e-2 relative) and no 1e-10 claim would be possible.

**Two independent momentum books** are compared: the *snapshot* book
`P_sys = Σ f_i c_i + Σ m_k v_k` from the fields, and the *impulse* book from the
per-link tallies plus the prescribed forces:

```
momentum_residual = |P_sys(T) − P_sys(0) − Σ I_ext| / max(|P_sys(0)|, ρ U L)
```

**Energy book** `Δ(KE_fluid + KE_floes + PE_contact) = W_ship + W_walls − c·Φ_visc`
with `Φ_visc = Σ 2μ S:S` from the non-equilibrium stress (an *independent*
continuum estimate; using the lattice collision loss would make the budget a
tautology). `c` is an **instrument calibration** measured on the smooth
Taylor–Green reference at the same grid/τ/window (no coupling, no moving
boundary): `c = 0.989 / 0.993 / 0.995`. Two further numerical channels are
carried explicitly so they cannot hide inside the residual: the **node-conversion
energy** and the **mass-repair energy** (see §5).

**Two numerical devices** without which the audit is not measurable:
* *particle mapping* — the body is rasterised at the nearest lattice position that
  keeps the staircase mask mirror symmetric. An asymmetric mask carries a
  spurious exchange force of ~0.2 (ten times the physical drag) which **pumps**
  the fluid instead of damping the floe; snapping costs one O(dx) geometry error.
* *mass restoration* — the mass deficit of the moving bounce back is spread over
  the rest population, which carries no momentum and no lattice kinetic energy:
  mass becomes exact (drift 1.1e-16 instead of −5e-2) and momentum is untouched.

Toolchain: pure NumPy D2L9 BGK (`fluid.py`), in-house DEM (`dem.py`), ledger +
cases (`coupling.py`), frozen-criteria gate (`acceptance.py`), pytest, matplotlib.

## 3. 计算结果 (numbers from `verification.json`)

Momentum closure at the finest level (n = 128):

| case | momentum residual | criterion 1e-10 |
|---|---|---|
| `floe_drift_decay` | **4.61e-15** | PASS |
| `ice_field_shear` | **7.37e-14** | PASS |
| `ship_ice_push` | **1.09e-12** | PASS |
| `drag_convergence` (diagnostic) | 8.71e-14 | — |
| `negative_control_lbm_only` | **4.06e-14** (criterion 1e-12) | PASS |

Grid convergence of the window-averaged interface force (the quantity a user
actually reads off a coupled run):

| n | 64 | 96 | 128 | Richardson order |
|---|---|---|---|---|
| `floe_drift_decay` ⟨ΣF⟩_x | −8.448e-02 | −1.392e-01 | −1.616e-01 | **p = 2.20** |
| `drag_convergence` ⟨ΣF⟩_x | −7.947e-02 | −1.167e-01 | −1.588e-01 | p = −0.30 (see §5) |
| `ice_field_shear` ⟨ΣF⟩_x | *diverged* | −3.234e-03 | −4.279e-03 | n/a |
| `ship_ice_push` ⟨ΣF⟩_x | *diverged* | +2.291e+01 | +3.063e+01 | n/a |

Energy budget (calibrated, audit window, finest level):

| case | ΔE | work W | Φ_visc | residual | criterion 5 % |
|---|---|---|---|---|---|
| `floe_drift_decay` | −6.65e-02 | 0 | 6.251e-01 | **1.520** | FAIL |
| `ice_field_shear` | +2.56e-02 | 1.419e-01 | 9.420e-02 | **0.156** | FAIL |
| `ship_ice_push` | +1.61e+00 | 2.885e+02 | 1.529e+01 | **0.913** | FAIL |
| `negative_control_lbm_only` | −5.17e-02 | 0 | 5.197e-02 | **0.005** | PASS |

Interface power cross-check (lattice energy flux vs `F·u`), mass drift, Mach and
the conversion-energy channel:

| case | interface-power residual | mass drift | max Ma | conversion energy |
|---|---|---|---|---|
| `floe_drift_decay` | 0.027 | −1.1e-16 | 0.011 | −3.91e-01 |
| `ice_field_shear` | 0.598 | −1.1e-16 | 0.0044 | 0.00 |
| `ship_ice_push` | 1.085 | 0.0 | 0.040 | −8.19e+00 |

Frozen-criteria verdict (digest `CRITERIA_DIGEST`, `acceptance.py`):

| criterion | value | threshold | outcome |
|---|---|---|---|
| mom-conservation | 1.09e-12 | ≤ 1e-10 | **PASS** |
| energy-conservation | 1.520 | ≤ 5e-2 | **FAIL** |
| residual-separation | 1.40e+12 | ≥ 1e+6 | **PASS** |
| momentum-order | 2.200 | ≥ 1 | **PASS** |
| negative-control | 4.06e-14 | ≤ 1e-12 | **PASS** |
| interface-power (evidence) | 1.085 | ≤ 5e-2 | FAIL |
| mass-drift (evidence) | 1.11e-16 | ≤ 1e-2 | PASS |
| mach-envelope (evidence) | 0.0403 | ≤ 0.1 | PASS |

**Suite verdict: FAIL** (mandatory `energy-conservation` fails). The momentum
contract is established; the energy contract is not, and the causes are quantified
in §5. Density-ratio sweep (0.92 / 1.2 / 1.6 / 3 / 10): momentum residual
4.4e-15 / 8.3e-15 / 2.9e-15 / 2.5e-15 / 8.0e-16 — the momentum contract is
**mass-ratio independent**, as an adjoint exchange must be.

Figures: `fig_momentum_residual.png`, `fig_energy_residual.png`,
`fig_impulse_convergence.png`, `fig_mass_ratio_sweep.png`.

## 4. 结果图

* `fig_momentum_residual.png` — momentum residual vs grid, all coupled cases plus
  the control, with the 1e-10 / 1e-12 lines: every case sits at round-off and is
  grid independent (the expected signature of an adjoint exchange).
* `fig_impulse_convergence.png` — the window-averaged interface force vs `dx` on a
  log axis: the floe case converges monotonically (p = 2.2), the prescribed drag
  probe does not (p = −0.3, staircase bias).
* `fig_energy_residual.png` — energy residual vs grid: the shear case is the only
  one near the 5 % line; the floe and ship cases are at O(1).
* `fig_mass_ratio_sweep.png` — momentum residual vs floe/water density ratio: flat
  at round-off over two decades of mass ratio.

## 5. 注意点 (traps, limitations, findings)

1. **The two residuals are different statements.** The momentum residual is
   *structural* (round-off, grid independent, mass-ratio independent). The energy
   residual is a *discretisation* statement (percent level). The gate
   `residual-separation` makes that hierarchy machine-checkable; without it, a
   "1e-15 momentum" claim could be mistaken for a claim about accuracy.

2. **Why the energy budget does not close (measured, not speculative).**
   * On a *smooth* flow the estimator is excellent: Taylor–Green calibration gain
     0.989 / 0.993 / 0.995, i.e. ≤ 1.1 % bias — the windowed estimator is fine.
   * On the *coupled* cases `Σ 2μS:S` over-counts the actual kinetic-energy decay
     by 1.5×–6× (`floe_drift_decay`: ΔE = −0.0665 vs Φ = 0.625). The
     non-equilibrium stress is under-resolved in the staircase corners, so the
     continuum dissipation formula is not the lattice dissipation there. The
     residual is grid-*independent* between n = 64 and 128, i.e. it is not a
     resolution convergence path: it needs a different boundary scheme
     (interpolated bounce back with the sub-grid distance), not a finer grid.
   * The ship case additionally exchanges a large **reversible p–V work**
     (W = 288.5 vs Φ = 15.3) in the closed box: a kinetic-only budget cannot hold it.
   * Node conversion is momentum exact but **energy lossy**: the created nodes are
     initialised at `(ρ0, u_floe)` so kinetic energy is injected (measured channel
     −0.39 for the floe case, −8.2 for the ship). It is booked as its own ledger
     line; the energy criterion is failed *with* that line included.

3. **Staircase force noise and the resolution floor.** At n = 32–64 the eight-floe
   and ship cases diverge (Mach > 1000) while n = 96/128 are clean: an explicitly
   staggered coupling of *light* floes (ρ_i/ρ_w = 0.92 < added mass) is only
   stable once each body spans enough nodes for the exchange force to be smooth.
   This is an added-mass limitation of explicit coupling, not a bug of the ledger
   (the momentum residual of the same runs stays at round-off until the flow
   itself blows up).

4. **The drag probe does not converge (p = −0.30) — an honest negative result.**
   The staircase circle's covered area is biased by ≈ dx/4, so the *effective*
   radius changes with the grid and the drag converges with the geometry error, not
   with the discretisation error. The fix is an area-matched staircase or
   interpolated bounce back; the campaign cases avoid it by using the *free* floe
   (whose drag history is self-consistent) as the primary order quantity.

5. **Applicability.** Results are 2-D (horizontal plane), single-phase
   incompressible LBM (Ma ≤ 0.046), Re_L = 100, buoyancy-neutralised floes, spring
   contacts with restitution 1. They do not transfer to 3-D ice loads or to
   breaking/fracture without a new audit.

## 6. 误差判定 (quantified, with the "is it < 3 %" answer)

| quantity | reference | error | < 3 %? |
|---|---|---|---|
| system momentum closure | prescribed impulse (exact tally) | 4.6e-15 … 1.1e-12 | **yes** (≪3 %) |
| negative control (no interface) | exact conservation | 4.1e-14 | **yes** |
| mass conservation | initial lattice mass | 1.1e-16 | **yes** |
| smooth-flow dissipation estimator | kinetic-energy decay of Taylor–Green | 1.1 % | **yes** |
| grid order of the interface force | Richardson extrapolation | p = 2.20 (≥ 1 ✓) | order, not an error % |
| energy budget, shear case | external work − Φ | 15.6 % | **no** |
| energy budget, floe / ship case | external work − Φ | 152 % / 91 % | **no** |

There is **no experimental reference** for this node: it is a conservation audit,
not a validation against measurements. The only external references are the exact
tallies (momentum, mass) and the smooth-flow calibration. Per campaign discipline
the energy criterion is therefore reported as an **honest FAIL**, and it is not
overridden by the momentum PASS.