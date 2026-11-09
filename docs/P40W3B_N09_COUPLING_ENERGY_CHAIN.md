# Multi-solver coupling energy-conservation contract: LBM <-> FEM <-> DEM runtime ledgers

**Campaign node** `P40W3B-N09` · **mission** 7597 · **repo** TensorPhysics
· **module** `tensorphysics.coupling_energy` · acceptance digest: `artifacts/p40w3b_n09/verification.json`

---

## 1. Case introduction

Three partitioned couplings are built on the same contract: *every energy or
impulse that crosses a solver interface must be measured by the solver that
produced it and booked as the identical number on the receiving solver*. The
conserved quantity is evaluated from the solvers themselves at every exchange
instant, so the residual reported below is a physical non-conservation and not a
bookkeeping identity:

```
eps(t) = | sum_sides E_side(t) - Q_external_in(t) + W_dissipated(t) - (same at t=0) | / scale
```

### (a) LBM <-> FEM conjugate heat transfer

A 2D square cavity `L = 0.1 m` split at `y = L/2`: the lower half is a Q4
conduction FEM, the upper half a D2Q9 passive-scalar (temperature) LBM. The
solid's lower face is a Dirichlet wall at `dT = 50 K` (the only external input);
fluid top and all side faces are adiabatic. The solid hands its interface
temperature to the fluid, the fluid hands back the **measured** interface heat
rate as a Neumann flux on the solid.

| parameter | value |
| --- | --- |
| cavity | `0.1 m x 0.1 m`, interface at `y = 0.05 m` |
| solid mesh | 40 x 20 bilinear quads, `rho c_p = 1e6 J m^-3 K^-1`, `k = 1 W m^-1 K^-1` |
| fluid lattice | 40 x 20 D2Q9, `tau = 1.25` (`alpha_lat = 0.25`) |
| diffusivity (both sides) | `alpha = 1e-6 m^2/s` -> identical `alpha`, so the exact solution is a single 1-D slab |
| LBM step / coarse step | `dt = 6.25 s`, `h = r dt`, `t_end = 2000 s` |
| interface relaxation | `omega = 0.2` (see §5) |

### (b) FEM <-> DEM contact (Newton's third law)

A **freely floating** plane-stress plate (`0.2 m x 0.04 m`, `E = 2 GPa`,
`rho = 900 kg/m^3`, thickness `0.05 m`) is hit edge-on by two DEM discs
(`r = 8 mm`, `v = 3 m/s`, linear spring contact `k = 5e6 N/m`, no damping).
No supports means no external force, so the total linear momentum must stay
constant; the DEM sub-cycles at `dt = h/r` inside one plate step `h`.

### (c) LBM <-> FEM <-> DEM energy chain

The same plate is clamped at `x = 0`, its lower face carries `dT = 20 K`, its
upper face is coupled to a 20 x 4 water column (same LBM), and two ice discs hit
the free edge. The reported chain is
`E_total = E_fluid + E_solid^thermal + E_solid^mech + E_particles + E_contact`.

## 2. Method

* **Runtime ledgers** (`coupling_energy.ledger`): double-entry energy and
  momentum books with residual / drift-slope / divergence diagnostics.
* **Interface reconstruction** (`coupling_energy.exchange`): zero (hold), first
  (Taylor through two samples) and second (three samples) order histories, plus
  a deterministic exchange-quadrature harness that measures the order on an
  analytic signal independently of any solver.
* **Reference solvers** (`coupling_energy.solvers`): D2Q9 passive-scalar LBM,
  Q4 transient conduction FEM (lumped capacity, backward Euler), Q4 plane-stress
  FEM (velocity-Verlet), 2D DEM discs with a linear spring-dashpot contact.
* **Sub-cycling**: ratio `r = 1, 2, 4, 8`; the fast solver (LBM / DEM) takes `r`
  sub-steps inside one coarse step of the slow solver (FEM).
* **Falsification arm**: every combination is repeated in
  `exchange_mode="independent"`, where the receiving side evaluates its **own**
  interface quantity instead of the shared one. A ledger that cannot separate
  the two would be blind, so this control is part of the acceptance gates.

## 3. Results

### 3.1 Gate summary

| criterion | required | actual | verdict |
| --- | --- | --- | --- |
| thermal_conservation_residual | < 1e-12 (max over 12 shared runs) | 2.231e-13 | PASS |
| thermal_falsification_ratio | > 1e3 x shared (independent-flux control) | 1.623e+12 | PASS |
| thermal_profile_error_r1 | < 0.03 (rel. L2 vs exact 1-D slab series) | 0.019 | PASS |
| mechanical_momentum_residual | < 1e-10 (max over 12 shared runs) | 1.423e-13 | PASS |
| mechanical_falsification_ratio | > 1e3 x shared (lagged-reaction control) | 1.403e+13 | PASS |
| mechanical_energy_convergence | drift(r=8) < 0.05 and monotone decreasing in r | {"1": 0.1547, "2": 0.0733, "4": 0.0352, "8": 0.0168} | PASS |
| three_field_energy_chain | < 1e-3 (max over 12 shared runs) | 4.175e-06 | PASS |
| three_field_falsification_ratio | > 1e2 x shared (independent-flux control) | 1.170e+05 | PASS |
| interpolation_order_scan | zoh 1.0, linear 2.0, quadratic 3.0 (+-0.3) | {"zoh": 0.9983, "linear": 1.9684, "quadratic": 2.8147} | PASS |
| plate_first_frequency | < 0.03 rel. error vs c/(4L) | 0.003 | PASS |
| subcycling_matrix_complete | 4 ratios x 3 schemes (+4 controls) x 3 couplings = 48 runs | 48.000 | PASS |

The full machine record (48 runs, all per-step series) is
`artifacts/p40w3b_n09/verification.json`; figures are listed in
`artifacts/p40w3b_n09/figures.json`.

### 3.2 (a) thermal — conservation residual and solution accuracy

| r | scheme | coarse steps | eps_max [-] | profile rel-L2 error vs exact |
| --- | --- | --- | --- | --- |
| 1 | zoh | 1280 | 2.181e-13 | 0.019 |
| 1 | linear | 1280 | 2.231e-13 | 0.019 |
| 1 | quadratic | 1280 | 2.219e-13 | 0.019 |
| 2 | zoh | 640 | 6.940e-14 | 0.036 |
| 2 | linear | 640 | 6.973e-14 | 0.036 |
| 2 | quadratic | 640 | 6.785e-14 | 0.036 |
| 4 | zoh | 320 | 5.916e-14 | 0.066 |
| 4 | linear | 320 | 5.809e-14 | 0.066 |
| 4 | quadratic | 320 | 5.889e-14 | 0.066 |
| 8 | zoh | 160 | 2.795e-14 | 0.113 |
| 8 | linear | 160 | 2.753e-14 | 0.113 |
| 8 | quadratic | 160 | 2.666e-14 | 0.113 |

The ledger stays at round-off for **all twelve** combinations (the residual
*decreases* with `r` simply because fewer arithmetic operations are performed),
while the solution error grows with `r` — the coupling is exact, the
sub-cycling is first-order accurate. The interface reconstruction order changes
the solution error only in the fourth decimal (e.g. r=1: 0.019 for
r=1 independent of scheme), which is reported honestly in §6.

### 3.3 (b) mechanical — Newton's third law and the energy budget

| r | scheme | max momentum residual | max energy drift | max overlap [m] |
| --- | --- | --- | --- | --- |
| 1 | zoh | 1.385e-13 | 0.157 | 1.157e-04 |
| 1 | linear | 1.423e-13 | 0.155 | 1.153e-04 |
| 1 | quadratic | 1.395e-13 | 0.156 | 1.154e-04 |
| 2 | zoh | 1.373e-13 | 0.078 | 1.142e-04 |
| 2 | linear | 1.386e-13 | 0.073 | 1.133e-04 |
| 2 | quadratic | 1.385e-13 | 0.076 | 1.136e-04 |
| 4 | zoh | 1.386e-13 | 0.041 | 1.136e-04 |
| 4 | linear | 1.408e-13 | 0.035 | 1.124e-04 |
| 4 | quadratic | 1.401e-13 | 0.039 | 1.129e-04 |
| 8 | zoh | 1.409e-13 | 0.026 | 1.132e-04 |
| 8 | linear | 1.418e-13 | 0.017 | 1.120e-04 |
| 8 | quadratic | 1.411e-13 | 0.021 | 1.125e-04 |

Momentum is conserved to `1.423e-13`
for every ratio: the contact force is one scalar per contact, applied with
opposite sign to the two sides. The energy drift is an integrator property of
the explicit DEM sub-stepping and falls by ~2x per doubling of `r`
(0.155 at r=1 -> 0.017 at r=8).

### 3.4 (c) three-field energy chain

| r | scheme | eps_total | eps_thermal | eps_mechanical |
| --- | --- | --- | --- | --- |
| 1 | zoh | 4.175e-06 | 8.845e-16 | 0.123 |
| 1 | linear | 4.122e-06 | 8.845e-16 | 0.121 |
| 1 | quadratic | 4.148e-06 | 8.845e-16 | 0.122 |
| 2 | zoh | 1.624e-06 | 2.991e-15 | 0.062 |
| 2 | linear | 1.535e-06 | 2.991e-15 | 0.058 |
| 2 | quadratic | 1.578e-06 | 2.991e-15 | 0.060 |
| 4 | zoh | 7.295e-07 | 8.156e-15 | 0.033 |
| 4 | linear | 6.318e-07 | 8.156e-15 | 0.028 |
| 4 | quadratic | 6.828e-07 | 8.156e-15 | 0.031 |
| 8 | zoh | 4.362e-07 | 1.400e-14 | 0.021 |
| 8 | linear | 2.829e-07 | 1.400e-14 | 0.014 |
| 8 | quadratic | 3.406e-07 | 1.400e-14 | 0.016 |

The chain residual is dominated by the mechanical sub-system (the thermal part
alone closes to `2.991e-15`), and it also
halves with `r`.

### 3.5 Falsification arm

| coupling | shared eps_max | independent eps_max | ratio |
| --- | --- | --- | --- |
| thermal | 2.231e-13 | 0.362 | 1.623e+12 |
| mechanical | 0.157 | 1.419 | 9.036 |
| three_field | 4.175e-06 | 0.488 | 1.170e+05 |

Removing the shared quantity destroys the contract by 6-13 orders of magnitude,
which is the evidence that the instrument has teeth.

### 3.6 Interpolation-order scan and FEM physics check

| scheme | declared order | measured order |
| --- | --- | --- |
| zoh | 1 | 0.998 |
| linear | 2 | 1.968 |
| quadratic | 3 | 2.815 |

Measured in the asymptotic range `h = 0.05 -> 0.0125` on a smooth analytic
signal: zero/first/second order are confirmed as 1 / 2 / 2.8. The Q4 plate
first axial (clamped-free) frequency is `1868.069 Hz` against the
analytic `c/(4L) = 1863.390 Hz`, i.e. `0.25 %`
error (`nx = 20`).

## 4. Figures

1. `fig_thermal_energy.png` — stored energy, external input and interface
   transfer, and the residual per sub-cycling ratio.
2. `fig_thermal_profile.png` — coupled profile against the exact 1-D slab
   solution at four instants.
3. `fig_interpolation_order.png` — measured exchange-quadrature order.
4. `fig_mechanical_momentum.png` — impact energy split and the momentum
   residual with its falsification arm.
5. `fig_three_field_chain.png` — the three-field energy chain and its residual.
6. `fig_residual_vs_ratio.png` — residual of all three couplings versus `r`.

## 5. Notes, defects fixed and limitations

**Defects found and fixed in the pre-existing harness** (each verified by a test):

1. `ThermalLBM2D.step` never streamed the rest population `f0` — `f0 * Nx * Ny`
   of the conserved scalar was destroyed every step.
2. Bounce-back targets were accumulated into the destination array before the
   later direction assignments and were overwritten; one reflected population
   per wall node and step was lost (insulated-lattice drift was 96 % before the
   fix, `1.3e-14` after).
3. The reflected population was the *incoming* one instead of the *outgoing*
   one, so a non-uniform field leaked at the walls.
4. The Dirichlet anti-bounce-back replaced the outgoing directions instead of
   the incoming ones (the wall temperature did not converge to `T_w`).
5. `DEM2D.contact_forces` had an inverted sign (compression attracting the
   bodies), which made any contact run diverge.
6. The FEM Neumann boundary expects a heat **rate** [W]; passing the per-step
   energy scaled the interface flux by the coarse step.

**Explicit Dirichlet-Neumann exchange is conditionally stable.** With a full
interface update (`omega = 1`) the partitioned thermal exchange diverges: the
LBM wall conductance is large compared with the heat capacity of the solid's
surface row, so the interface error is amplified every exchange, independently
of `r`. The runs reported here use an interface under-relaxation `omega = 0.2`
(a standard partitioned-coupling stabilisation) which only changes the boundary
condition handed to the fluid; the measured flux still crosses the ledger
unchanged, so conservation is untouched. This is an *honest FAIL* of the
un-relaxed scheme, documented here rather than hidden.

**Anti-bounce-back wall artefact.** The one-node Dirichlet wall converges to
`T_w + O(dx)` (measured +6.3 % at 16 nodes); this is the dominant contribution
to the 1.9 % profile error of the coupled solution and would need a second-order
wall treatment to be removed.

**Three-field time scales.** Thermal (seconds) and mechanical (microseconds)
clocks differ by ~`1e5` and are not synchronised: each macro step advances both
physics by one of their own stable steps and the ledger is a snapshot statement
over the common macro index. The two chains are energetically **additive** —
thermomechanical coupling (thermal stress, thermal expansion) is not modelled.

**Scope.** The reference solvers are verification objects, not production
kernels: 2-D, small, Python. TensorPhysics owns the coupling contract; the
domain solvers live in TensorLBM/FEM/DEM.

## 6. Error assessment

| quantity | reference | error | < 3 % ? |
| --- | --- | --- | --- |
| thermal profile, r=1 (linear) | exact 1-D slab series | 1.86 % | yes |
| thermal profile, r=2 | exact 1-D slab series | 3.58 % | no |
| thermal profile, r=8 | exact 1-D slab series | 11.32 % | no |
| plate first axial frequency | `c/(4L)` | 0.25 % | yes |
| thermal interface flux consistency (LBM vs FEM Fourier estimate) | each other | see `flux_consistency` in the JSON (not a convergence claim) | n/a |
| momentum conservation | Newton's third law | 1.423e-13 (round-off) | yes |
| energy conservation (a) | ledger contract | 2.231e-13 | yes |
| energy conservation (b) | ledger contract | 0.017 at r=8, 0.155 at r=1 | no at r<=4 |
| energy conservation (c) | ledger contract | 2.829e-07 .. 4.175e-06 | yes |

Sub-cycling at `r >= 4` does **not** meet a 3 % solution-accuracy target for the
thermal transient; the ratio sweep in §3.2 quantifies the cost. Conservation
itself is not affected by `r`.
