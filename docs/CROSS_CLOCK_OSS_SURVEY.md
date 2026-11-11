# OSS / method survey for GAP-CROSS-CLOCK (round 2 research)

Scope: what already exists for heterogeneous time steps across solvers, read from
the cluster's `~/oss-cache` and from the standard literature. This file records
what was **learned**, not code copied.

## 1. OSS landscape actually on disk

| source | what it does with time | what it does *not* have |
| --- | --- | --- |
| `~/oss-cache/pylbm` (pylbm) | `pylbm/simulation.py:212` defines `dt` as a *scheme parameter* (symbolic, substituted once); `simulation.py:199` `_get_algorithm` selects a splitting operator chain (`pylbm/algorithm/{base,ode,pull,transform}.py`), i.e. one fixed step = advection + local relaxation | no sub-stepping, no second clock, no coupling partner |
| `~/oss-cache/Maarten-vd-Sande_lbm` (teaching D2Q9; `oss_inventory` tier 1, "no Guo/velocity-shift") | `lbm/time_evolve.py:7` `evolve(total_timesteps, ...)` — a fixed integer step loop; body force applied inside the collision (`lbm/streamcollide.py:147-152`) as `w*rho*e.g` | no sub-cycling, no unit map, no tests |
| `oss_inventory` query `feature_slug ILIKE '%CLOCK%'` | two hits, both unrelated: `lanl/MF-LBM` wall-clock checkpoint state machine; `joconnor22/LIFE` strong-coupling IBM | nothing on cross-solver clock semantics |

Conclusion: there is **no OSS method to lift** for cross-solver clock semantics —
both LBM codes are strictly single-clock. What is worth copying is the *shape* of
the step loop (integer step count, `dt` as a scheme parameter, splitting per
step), which is what `SyncSchedule` + the two legs adopt.

## 2. Method lineage (external anchors, literature)

1. **Conservative transfer of integrals, not point values** — Farhat, Lesoinne &
   LeTallec, *CMAME* 157 (1998) 95-114: load/motion transfer on non-matching
   interfaces must preserve resultants. Transferred here as "only interval
   integrals cross the interface".
2. **Partitioned staggered (explicit) coupling and its lag defect** — Piperno,
   Farhat & Larrouturou, *CMAME* 124 (1995) 79-116: the explicit staggering error
   dominates the solution error; recorded here as `stagger_residual`.
3. **Co-simulation semantics: ZOH vs extrapolation, stability** — Gomes et al.,
   *ACM Computing Surveys* 51(3) (2018) 49: the causal zeroth-order hold is the
   robust reconstruction; linear extrapolation is conditionally unstable.
   Adopted: ZOH, documented as the only information a partitioning consumer has.
4. **Conservation in non-iterative co-simulation** — Sadjina et al. (2017),
   arXiv:1708.02810: conservation is restored by correcting the coupling
   quantity, not by shrinking the step. Adopted: transfer the integral exactly,
   report the residual separately.
5. **Sub-cycling standards** — FMI 3.0 (Modelica Association, 2022) co-simulation
   step contract; preCICE (Chourdakis et al., *PMC* 108 (2022) 107284) subcycling
   with an integer number of participant steps per window. Adopted: the integer
   clock ratio is the only admissible window definition.
6. **Lattice units** — Krueger et al., *The Lattice Boltzmann Method* (Springer,
   2017): `nu = (tau - 1/2)/3 dx^2/dt`. Adopted in `UnitMap`; the Taylor-Green
   anchor falsifies the `tau - 1/3` variant (factor 1.55).

## 3. Selection for this repo (and why)

| decision | choice | rejected alternative |
| --- | --- | --- |
| what crosses the interface | interval integral of momentum | point values / instantaneous forces (not interpolable conservatively) |
| how the consumer reconstructs | causal ZOH per window | linear extrapolation (unstable in explicit coupling) |
| window definition | `h = r_l * dt_l`, integer `r_l` per leg | one common `dt` for all legs (destroys the lattice/DEM step constraints) |
| where time error can enter | nowhere: `t = t0 + k*h` from integer `k` | `t += dt` accumulation |
| defect reporting | ledger drift (bookkeeping) kept apart from stagger/reference drift (physics) | a single "error" number that hides which one is which |

## 4. Delta over m13863 and over round 1 (m14677)

* m13863 delivered the abstract contract with in-repo reference instruments.
* m14677 (round 1 of this anchor) left no `fail_stage` and no artefacts; the
  failure mode was scoping — research and implementation in one pass, nothing
  committed early.
* This round: research first (above), then the binding to the real libraries with
  pins, 18 tests and a reproducible artefact record — every number in
  `docs/CROSS_CLOCK_LIBRARY_CONTRACT.md` is injected from
  `runs/m14792_cross_clock_library/results.json`.
