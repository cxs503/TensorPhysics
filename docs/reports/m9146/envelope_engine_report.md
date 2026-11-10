# Operating-window envelope engine: constraint model, boundary marching, contract checks

Node `m9146` · package `tensorphysics.envelope` · configuration `tensorphysics.envelope.case.v1`
· evidence `artifacts/envelope/criteria_evidence.json` · cases `scenarios/envelope_cases/`

## ① What the cases are

An *operating window* is the set of parameter values on which every constraint of
a capability holds. Three windows are shipped as pure configuration (no code
change per window), each with three load conditions:

| case | window axis (span) | constants | conditions (yes: parameters that vary) |
|---|---|---|---|
| exchange_rate_window | `exchange_rate` ∈ [10, 3000] | `gm_linear`=1.000000e-04, `gm_quadratic`=3.000000e-08, `hold_volume`=12000 | `exchange_open_sea`: available_time=12, gm_margin=0.35, pump_capacity=2400, residence_required=7.5; `exchange_coastal`: available_time=10, gm_margin=0.3, pump_capacity=1800, residence_required=5; `exchange_estuary`: available_time=8, gm_margin=0.22, pump_capacity=1600, residence_required=4 |
| stranding_tide_window | `tide_level` ∈ [0, 6] | `charted_depth`=10, `squat`=0.5, `ukc_allowance`=0.6 | `transit_full`: draught=10.3, sill_limit=3.2; `transit_medium`: draught=9.6, sill_limit=4; `transit_light`: draught=9, sill_limit=4.6 |
| sts_transfer_window | `transfer_rate` ∈ [0, 2500] | `pressure_linear`=4.000000e-04, `pressure_quadratic`=2.200000e-06 | `sts_calm`: hose_limit=2200, min_rate=300, pressure_budget=9, weather_limit=2100; `sts_moderate`: hose_limit=1900, min_rate=450, pressure_budget=7, weather_limit=1750; `sts_rough`: hose_limit=1200, min_rate=600, pressure_budget=5, weather_limit=1400 |

The constraint networks are ordinary data as well — the engine has no compiled-in
physical vocabulary:

| case | constraint | kind | margin form (data) | declared slope | declared convex |
|---|---|---|---|---|---|
| exchange_rate_window | `c_completion` | inequality | `sub` | exchange_rate:+1 | true |
| exchange_rate_window | `c_residence` | inequality | `sub` | exchange_rate:-1 | — |
| exchange_rate_window | `c_pump_capacity` | inequality | `sub` | exchange_rate:-1 | true |
| exchange_rate_window | `c_stability` | inequality | `sub` | exchange_rate:-1 | true |
| exchange_rate_window | `c_setting` | discrete | `var` | — | — |
| stranding_tide_window | `c_ukc` | inequality | `sum` | tide_level:+1 | true |
| stranding_tide_window | `c_sill` | inequality | `sub` | tide_level:-1 | true |
| sts_transfer_window | `c_min_rate` | inequality | `sub` | transfer_rate:+1 | true |
| sts_transfer_window | `c_manifold_pressure` | inequality | `sub` | transfer_rate:-1 | true |
| sts_transfer_window | `c_hose_limit` | inequality | `sub` | transfer_rate:-1 | true |
| sts_transfer_window | `c_weather_limit` | inequality | `sub` | transfer_rate:-1 | true |
| sts_transfer_window | `c_setting` | discrete | `var` | — | — |

Physical reading of the three cases (content that lives in the configuration, not
in the package): the first is the admissible tide band for a transit over a
shoal, bounded below by under-keel clearance and above by the sill ceiling; the
second is the admissible ballast-exchange rate, bounded below by the time
available to complete the exchange and above by pump capacity, the required
residence time (a reciprocal law) and the free-surface loss of metacentric
height (a quadratic law), with the attainable rate additionally constrained to a
discrete pump-setting lattice; the third is the admissible ship-to-ship transfer
rate, bounded below by the minimum controllable rate and above by the manifold
pressure budget (quadratic), the hose limit and the weather-dependent relative
motion limit, again on a discrete step lattice.

## ② How it was done

The mechanism is five stages, each in its own module, with the order fixed by
`EnvelopeEngine.run` (full description: `docs/OPERATING_WINDOW_ENVELOPE.md`):

1. **constraint model** — inequality / equality / discrete constraints are all
   reduced to one scalar margin with `feasible ⟺ margin ≥ 0`; the margin is an
   expression tree over named parameters with a closed operator table
   (`const`, `var`, `sum`, `product`, `sub`, `div`, `reciprocal`, `neg`,
   `scale`, `abs`, `square`, `sqrt`, `max`, `min`, `table`), no extrapolating
   `table`;
2. **grid scan** — uniform feasibility mask, per-cell binding constraint, cell
   area quoted only as an order-of-magnitude cross-check;
3. **boundary marching** — per-constraint sign bracketing, Brent refinement to
   machine precision (bisection fallback), span partition at the refined roots,
   midpoint feasibility per sub-interval; 2-D by line-by-line 1-D marching with
   union-find component linking and analytic end blocks for the area;
4. **contract checks** — monotonicity, convexity (finite-difference Hessian
   eigenvalues), boundary order, infeasible semantics, structure/active set;
5. **orchestration and artifacts** — one result object per case and condition,
   byte-deterministic JSON with code and artifact digests.

Toolchain: Python 3.11.6 on loongarch64, standard library
only in the package (the array backend is imported lazily and only by the
brute-force cross-check). The acceptance run executed in
13.9 s.

## ③ Results

### Detection, active sets and attainable settings

| case | condition | status | window | width | binding constraint (lower / upper) | attainable settings | evaluations | wall time [s] |
|---|---|---|---|---|---|---|---|---|
| exchange_rate_window | exchange_open_sea | feasible | [1000, 1600] | 600 | ['c_completion'] → ['c_residence'] | 3 | 2069 | 0.0100 |
| exchange_rate_window | exchange_coastal | feasible | [1200, 1800] | 600 | ['c_completion'] → ['c_pump_capacity'] | 3 | 2070 | 0.0100 |
| exchange_rate_window | exchange_estuary | feasible | [1500, 1513.13067] | 13.1306714 | ['c_completion'] → ['c_stability'] | 1 | 2051 | 0.0101 |
| stranding_tide_window | transit_full | feasible | [1.4, 3.2] | 1.8 | ['c_ukc'] → ['c_sill'] | — | 1026 | 0.0046 |
| stranding_tide_window | transit_medium | feasible | [0.7, 4] | 3.3 | ['c_ukc'] → ['c_sill'] | — | 1026 | 0.0044 |
| stranding_tide_window | transit_light | feasible | [0.1, 4.6] | 4.5 | ['c_ukc'] → ['c_sill'] | — | 1028 | 0.0047 |
| sts_transfer_window | sts_calm | feasible | [300, 1933.7325] | 1633.7325 | ['c_min_rate'] → ['c_manifold_pressure'] | 17 | 2144 | 0.0089 |
| sts_transfer_window | sts_moderate | feasible | [450, 1695.17115] | 1245.17115 | ['c_min_rate'] → ['c_manifold_pressure'] | 12 | 2120 | 0.0080 |
| sts_transfer_window | sts_rough | feasible | [600, 1200] | 600 | ['c_min_rate'] → ['c_hose_limit'] | 7 | 2099 | 0.0098 |

All 78 contract outcomes over the
nine case/condition runs passed, and all six acceptance criteria are green:

| check kind | passed / total |
|---|---|
| active_set | 3/3 |
| convexity | 27/27 |
| infeasible_semantics | 9/9 |
| monotonicity | 30/30 |
| structure | 9/9 |

Note the active-set switching: the same case binds on a different constraint per
condition (the exchange case binds on the residence law, then on pump capacity,
then on stability; the transfer case binds on the pressure budget and then on
the hose limit), which is exactly what a window mechanism is supposed to report
instead of a single number. The narrowest window is
13.13 units wide against a 250-unit setting lattice, leaving exactly one
attainable setting — the case where a continuous window alone would be
misleading.

### Criteria

| criterion | what it establishes | threshold | measured | verdict |
|---|---|---|---|---|
| C1 migration consistency | engine window vs the closed-form per-window path, 27 numbers over 3 cases × 3 conditions | rel. diff ≤ 1.000000e-12 | worst rel. diff 3.608225e-15 | PASS |
| C2 boundary order | interpolated boundary error at chord midpoints under refinement (synthetic parabolic band) | order ≥ 1.9 | min order 2 | PASS |
| C3 contract negative control | a window declared monotone although its margin folds back must be reported | violation reported, control passes | failed checks ['monotone:c_band:alpha'] | PASS |
| C4 infeasible semantics | empty window → infeasible, no boundary, no area, nearest point with residual | no extrapolated number | status infeasible, area None, residual 1 | PASS |
| C5 zero domain semantics | domain vocabulary in package sources and tests | 0 hits | 0 hits over 13 files / 4830 lines | PASS |
| C6 cost and area | one 2-D window at ≥ 5·10⁴ constraint evaluations within 60 s on one core, marched area vs brute-force grid scan | ≤ 0.001 | 65842 evaluations in 0.592 s, area rel. diff 7.383542e-05 | PASS |

### Migration consistency, number by number (C1)

| case | condition | quantity | engine | closed-form reference | rel. diff |
|---|---|---|---|---|---|
| exchange_rate_window | exchange_open_sea | window_lo | 1000 | 1000 | 0 |
| exchange_rate_window | exchange_open_sea | window_hi | 1600 | 1600 | 0 |
| exchange_rate_window | exchange_open_sea | window_width | 600 | 600 | 0 |
| exchange_rate_window | exchange_coastal | window_lo | 1200 | 1200 | 0 |
| exchange_rate_window | exchange_coastal | window_hi | 1800 | 1800 | 0 |
| exchange_rate_window | exchange_coastal | window_width | 600 | 600 | 0 |
| exchange_rate_window | exchange_estuary | window_lo | 1500 | 1500 | 0 |
| exchange_rate_window | exchange_estuary | window_hi | 1513.13067 | 1513.13067 | 0 |
| exchange_rate_window | exchange_estuary | window_width | 13.1306714 | 13.1306714 | 0 |
| stranding_tide_window | transit_full | window_lo | 1.4 | 1.4 | 3.172066e-16 |
| stranding_tide_window | transit_full | window_hi | 3.2 | 3.2 | 0 |
| stranding_tide_window | transit_full | window_width | 1.8 | 1.8 | 2.467162e-16 |
| stranding_tide_window | transit_medium | window_lo | 0.7 | 0.7 | 1.586033e-15 |
| stranding_tide_window | transit_medium | window_hi | 4 | 4 | 0 |
| stranding_tide_window | transit_medium | window_width | 3.3 | 3.3 | 4.037175e-16 |
| stranding_tide_window | transit_light | window_lo | 0.1 | 0.1 | 3.608225e-15 |
| stranding_tide_window | transit_light | window_hi | 4.6 | 4.6 | 0 |
| stranding_tide_window | transit_light | window_width | 4.5 | 4.5 | 1.973730e-16 |
| sts_transfer_window | sts_calm | window_lo | 300 | 300 | 0 |
| sts_transfer_window | sts_calm | window_hi | 1933.7325 | 1933.7325 | 2.351656e-16 |
| sts_transfer_window | sts_calm | window_width | 1633.7325 | 1633.7325 | 2.783487e-16 |
| sts_transfer_window | sts_moderate | window_lo | 450 | 450 | 0 |
| sts_transfer_window | sts_moderate | window_hi | 1695.17115 | 1695.17115 | 1.341302e-16 |
| sts_transfer_window | sts_moderate | window_width | 1245.17115 | 1245.17115 | 1.826044e-16 |
| sts_transfer_window | sts_rough | window_lo | 600 | 600 | 0 |
| sts_transfer_window | sts_rough | window_hi | 1200 | 1200 | 0 |
| sts_transfer_window | sts_rough | window_width | 600 | 600 | 0 |

### Boundary order (C2)

| boundary | resolutions | line spacings h | max chord-midpoint error | observed order |
|---|---|---|---|---|
| synthetic parabolic band | [20, 40, 80] | ['0.05263', '0.02564', '0.01266'] | ['5.540166e-04', '1.314924e-04', '3.204615e-05'] | ['2', '2'] |
| synthetic parabolic band | [40, 80, 160] | ['0.02564', '0.01266', '0.006289'] | ['1.314924e-04', '3.204615e-05', '7.911079e-06'] | ['2', '2'] |
| synthetic parabolic band | [80, 160, 320] | ['0.01266', '0.006289', '0.003135'] | ['3.204615e-05', '7.911079e-06', '1.965389e-06'] | ['2', '2'] |
| synthetic disc (vertical tangents, limitation) | [40, 80, 160] | ['0.05128', '0.02532', '0.01258'] | ['0.02375', '0.01647', '0.01154'] | ['0.5187', '0.5091'] |

### Area against a brute-force grid scan (C6)

| n per axis | queries | brute-force area | rel. diff vs closed form | wall time [s] |
|---|---|---|---|---|
| 512 | 262144 | 1.76672118 | 2.403267e-04 | 0.014 |
| 1024 | 1048576 | 1.76732035 | 9.873759e-05 | 0.025 |
| 2048 | 4194304 | 1.7671634 | 9.923559e-06 | 0.227 |
| Richardson (1024→2048) | — | 1.76700646 | two-level consistency 5.167332e-04 | — |

The marched area is 1.76687599 (tip_model), the
closed form is 1.76714587: relative difference
1.527201e-04. The raw cell count is a
first-order estimator, so it is compared at three resolutions and also linearly
extrapolated; the engine is compared against the extrapolated value.

### Cost (C6)

160 scan lines × 320 samples × 1 constraint =
65842 constraint evaluations in 0.592 s on one
core (111275 evaluations/s); the criterion of 5·10⁴
evaluations within 60 s corresponds to
0.449 s projected, i.e. two orders
of magnitude of margin. A 1-D window with four constraints costs of order 2·10³
evaluations (table ③).

### Test suite

```
$ /usr/local/bin/python -m pytest tests/test_envelope_engine.py tests/test_envelope_contracts.py -q
54 passed, 1 warning in 3.18s
```

Return code 0.

## ④ Figures

![envelope_windows.png](../../../artifacts/envelope/figures/envelope_windows.png)

*Constraint margins along the window axis for the three shipped cases (three load conditions each). Shaded bands are the detected windows, vertical lines their boundaries, ticks the attainable discrete settings on the declared lattice, dotted line the feasibility threshold.*

![envelope_marching.png](../../../artifacts/envelope/figures/envelope_marching.png)

*Left: marched boundary of the synthetic disc (two-component case reported separately in the text) against the analytic circle and the coarse feasibility mask. Middle: boundary reconstruction order for the parabolic band under refinement, with the O(h^2) reference line. Right: relative area error of the same closed window versus the number of scan lines, against the 1e-3 acceptance line.*

![envelope_contracts.png](../../../artifacts/envelope/figures/envelope_contracts.png)

*Left: the deliberate negative control -- a margin that bends back although it is declared monotone; the checker reports the violation. Right: the empty-window control -- the engine returns 'infeasible' with the nearest examined point and its residual instead of extrapolating a boundary.*

## ⑤ Notes, pitfalls and limitations

* **The order criterion needs smooth boundaries.** On the synthetic disc the
  boundary becomes vertical at the tips, where the chord error behaves like
  `h^{1/2}`; the pointwise maximum error therefore does not converge at order 2
  (measured 0.519).
  The criterion is stated on a window with smooth boundaries, and the same disc is
  used for the area criterion, where it does converge (measured order 1.2–1.5
  with a small constant). This is a property of the reconstruction, not a defect
  to be papered over.
* **Brute force has its own bias.** A cell count is first order in the cell size
  (2.403267e-04 at
  512 lines against
  9.923559e-06 at
  2048). Comparing the marched area to a raw
  count at a moderate resolution would measure the reference, not the engine.
* **Discrete feasibility is applied after the continuous window.** A lattice has
  no boundary to march; snapping is reported in its own field, so a continuous
  boundary is never presented as an attainable setting.
* **Convexity is a sufficient test** (`margin` Hessian negative semidefinite ⇒
  convex feasible set). A monotone margin with a single converged root has a
  convex feasible set without satisfying it; such constraints are simply not
  declared `convex`, and the check is silent rather than wrong.
* **Clipped and complex components degrade explicitly**: `clipped`,
  `complex_topology` and `area_method` fields say so, and a 1-D window that ends
  on the span boundary instead of on a constraint sets `span_truncated`.
* **Reproducibility is checkable, not assumed**: the 1-D detection re-runs on a
  grid twice as fine and reports `verified`; an unreproducible window is
  reported as unverified instead of being trusted.
* **Two axes only.** A 3-D window is reduced by slicing; there is no 3-D marcher.
* **Scope of validity.** These are *algorithmic* criteria: the agreement is with
  closed-form algebra on the same declared constraints, and the physical content
  of the three cases is configuration data with representative coefficients. No
  model-test or full-scale reference data is involved, so nothing here validates
  the empirical correlations themselves — only the window machinery that consumes
  them.

## ⑥ Error judgement

| key quantity | reference | measured error | < 3 % ? |
|---|---|---|---|
| window bounds and width (18 numbers) | closed-form per-window path | worst rel. diff 3.608225e-15 | yes (by 12 orders of magnitude) |
| binding constraint per boundary (18 labels) | closed-form path | 18/18 identical | yes (exact) |
| attainable discrete settings (9 lists) | closed-form path | identical to 1e-9 | yes (exact) |
| boundary position order | analytic parabolic band | order 2 vs 2.0 exact | yes (second order) |
| area of a closed window | closed form πR² | 1.527201e-04 | yes |
| area vs brute-force grid scan | extrapolated cell count | 7.383542e-05 | yes |
| cost | — | 65842 evaluations in 0.592 s | yes (≤ 60 s with margin) |

No quantity in this node is compared against experimental or full-scale data, so
no percentage is claimed against one; the "≤ 3 %" answers above refer to the
declared references (closed-form algebra and the analytic boundary), and the
honest statement is that physical validation of the correlations carried by the
three configurations is out of scope here and remains open.

## Audit chain

* code digest (SHA-256 over the package sources): `c4f90b5207b4f38e43629dc674cc17228a9bed11abc9ccb24176c2f268baa124`
* case digests: `exchange_rate_window` = `146740aa98bc00b8…`, `stranding_tide_window` = `84a70cb5fccd415e…`, `sts_transfer_window` = `354f4397206b9bd2…`
* artifact digest: `7a27091a378b3bbbf86d58aeec252e04b530834f2b8462ca8e40de37fb915784`
* environment: Python 3.11.6, loongarch64, Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38
* reproduce: `python scripts/envelope_cases_acceptance.py --out artifacts/envelope`
  then `python scripts/generate_envelope_report.py`
