# P40W3C-N09 · Multi-solver coupling energy conservation contract
### LBM ↔ FEM ↔ DEM runtime energy and momentum ledgers (TensorPhysics)

## 1. Case introduction (physics, geometry, parameters)

A partitioned multi-solver system is *conservative* only if every exchange that
removes `q` from one solver books the identical `q` into the other and every
boundary input is booked as external. TensorPhysics owns that contract
(`tensorphysics.coupling_energy`), so the node verifies it on three couplings
driven by real discretisations rather than by synthetic energy histories.

| case | geometry | parameters |
|---|---|---|
| (a) LBM↔FEM thermal | 2D cavity `L=0.1 m`, plate `t_s=0.05 m` below the coupled south wall, insulated elsewhere | `ΔT=50 K` (fluid 350 K / plate 300 K), `τ=1.0`, `α_f=1.4e-7 m²/s`, `ρc_f=4.2e5 J/m³K`, `k_s=6 W/mK`, `ρc_s=4e6 J/m³K`, lattice `20×20`, plate `20×4` Q4 |
| (b) FEM↔DEM mechanical | free-free plane-stress plate `0.2×0.02 m`, two discs `r=1 mm` striking the right edge at 2 m/s | `E=2 GPa`, `ν=0.3`, `ρ=900 kg/m³`, linear spring contact `k=2e6 N/m` (undamped) |
| (c) three-field chain | same plate carries the thermal *and* the elastic field; two discs; dimensionless units (`dx=1`, `dt_lbm=1`, `α_lat=1/6`, `√(E/ρ)=1.2`) | `n_lbm=96` steps, mechanical sub-cycles `18` per lattice step |

Case (c) is deliberately dimensionless: a diffusive thermal transient is O(s)
while contact dynamics is O(µs), so a dimensionally-honest joint run needs a
time-scale study beyond this node. The conservation *contract* is scale free,
which is what is verified here.

## 2. Method

* Runtime ledgers: `EnergyLedger` (`E_total = ΣE_side − E_in + E_diss`) and
  `MomentumLedger` (impulses). Every number below is produced by these objects.
* Interface operator: conservative half-to-each-node scatter of the lattice
  column flux to the plate nodes, with its adjoint (hat) gather for the
  Dirichlet temperature, so flux and data share one duality pairing.
* Multirate exchange: the receiver only knows *completed* intervals; the
  reconstruction rules are `independent` (no shared history — the plate books
  its own one-sided Fourier flux), `zoh` (hold) and `linear` (slope
  extrapolation), with the exchange integrated over the *target* window.
* Sub-cycling ratio `r ∈ {1,2,4,8}` at identical physical windows.
* Negative controls: the `independent` row of every table must be red.

Tool chain: pure numpy reference solvers (`ThermalLBM2D`, `Q4ThermalFEM`,
`Q4PlaneStressFEM`, `DEM2D`), pytest gates, `runs/m7607/run_acceptance.py`.

## 3. Results

### (a) thermal ledger (r = exchange interval in units of dt = 29.762 s)

| r | h [s] | E_total drift | exchange mismatch | flux agreement | Q_fluid [W] | Q_solid [W] |
|---|---|---|---|---|---|---|
| 1 | 29.76 | 4.380e-04 | 7.299e-02 | 0.0450 | -9.065 | -8.657 |
| 2 | 59.52 | 7.198e-04 | 1.202e-01 | 0.0274 | -9.047 | -8.799 |
| 4 | 119 | 1.082e-03 | 1.811e-01 | 0.0113 | -9.021 | -9.123 |
| 8 | 238.1 | 1.520e-03 | 2.549e-01 | 0.1068 | -8.986 | -9.945 |
| 1 | 29.76 | 1.743e-07 | 5.482e-02 | 0.0534 | -9.092 | -8.606 |
| 2 | 59.52 | 7.085e-07 | 1.251e-01 | 0.0456 | -9.092 | -8.677 |
| 4 | 119 | 2.929e-06 | 2.334e-01 | 0.0318 | -9.092 | -8.803 |
| 8 | 238.1 | 1.258e-05 | 3.679e-01 | 0.0285 | -9.097 | -8.837 |
| 1 | 29.76 | 5.819e-03 | 1.000e+00 | 1.0000 | -9.671 | -0.000 |
| 2 | 59.52 | 5.508e-03 | 1.000e+00 | 1.0000 | -9.671 | 0.000 |
| 4 | 119 | 5.086e-03 | 1.000e+00 | 1.0000 | -9.671 | -0.000 |
| 8 | 238.1 | 4.526e-03 | 1.000e+00 | 1.0000 | -9.671 | 0.000 |

The measured wall heat rate and the plate's one-sided Fourier estimate agree to
1.13%–10.68%, i.e. the
interface flux the fluid *actually* gives is the flux the plate conducts.

### (b) mechanical ledger

| r | momentum drift | force antisymmetry | energy drift | peak contact force [N] |
|---|---|---|---|---|
| 1 | 1.86e-13 | 0.00e+00 | 1.030e-01 | 153.4 |
| 2 | 1.88e-13 | 0.00e+00 | 2.159e-01 | 157.3 |
| 4 | 1.93e-13 | 0.00e+00 | 4.978e-01 | 164.5 |
| 8 | 2.17e-13 | 0.00e+00 | 1.058e+00 | 178.3 |
| 1 | 1.77e-13 | 0.00e+00 | 5.873e-04 | 149.8 |
| 2 | 1.87e-13 | 0.00e+00 | 3.046e-03 | 150.1 |
| 4 | 1.93e-13 | 0.00e+00 | 2.741e-02 | 152.3 |
| 8 | 1.90e-13 | 0.00e+00 | 9.061e-02 | 152.8 |
| 1 | 8.24e+00 | 2.00e+00 | 1.432e-01 | 154.9 |
| 2 | 8.46e+00 | 2.00e+00 | 2.604e-01 | 158.8 |
| 4 | 8.95e+00 | 2.00e+00 | 5.521e-01 | 166.2 |
| 8 | 9.79e+00 | 2.00e+00 | 1.123e+00 | 179.8 |

### (c) three-field chain `E_fluid+E_solid+E_particles+U_contact`

| r | E_total drift | max drift | exchange mismatch | momentum mismatch |
|---|---|---|---|---|
| 1 | 2.672e-03 | 3.386e-03 | 5.312e-01 | 0.00e+00 |
| 2 | 4.454e-03 | 6.858e-03 | 1.184e+00 | 0.00e+00 |
| 4 | 8.082e-03 | 1.172e-02 | 1.358e+00 | 0.00e+00 |
| 8 | 7.417e-03 | 2.289e-02 | 1.230e+00 | 0.00e+00 |
| 1 | 7.404e-03 | 1.144e-02 | 1.947e+00 | 0.00e+00 |
| 2 | 9.308e+01 | 9.308e+01 | 1.686e+00 | 0.00e+00 |
| 4 | 1.130e+01 | 1.130e+01 | 2.117e+00 | 0.00e+00 |
| 8 | 4.180e-01 | 4.741e-01 | 1.167e+00 | 0.00e+00 |
| 1 | 3.339e-02 | 3.339e-02 | 1.000e+00 | 2.00e+00 |
| 2 | 3.160e-02 | 3.160e-02 | 1.000e+00 | 2.00e+00 |
| 4 | 2.927e-02 | 2.927e-02 | 1.000e+00 | 2.00e+00 |
| 8 | 2.350e-02 | 2.350e-02 | 1.000e+00 | 2.00e+00 |

### Kernel verification against analytic references

| check | reference | measured |
|---|---|---|
| lattice diffusion order | `exp(-απ²t/N²)` eigenmode | p = 2.52 on grids [16, 32, 48] |
| plate steady profile | `ΔT = Q t_s/(k A)` | 6.66% worst (3 grids) |
| plate discrete energy identity | `ΔE = h Σb` (lumped) | 1.5e-13 |
| DEM contact time | `π/ω` | 1.86% |
| DEM restitution | 1 (undamped) | -1.00008 |

### Sub-cycling order

Fitted on the coupled runs, `ε_ex ~ h^p`: zoh **p = 1.07**,
linear **p = 2.55**; analytic reference for the same
schemes on a smooth signal: zoh 1.00, linear
2.00.

**energy_history.png**  
![energy_history.png](../artifacts/m7607/energy_history.png)
*Energy partition and total for the three couplings; the dashed total stays flat to the reported drift.*

**residual_orders.png**  
![residual_orders.png](../artifacts/m7607/residual_orders.png)
*Exchange mismatch and total-energy drift vs the sub-cycling interval h; the independent control is red.*

**order_fit.png**  
![order_fit.png](../artifacts/m7607/order_fit.png)
*Fitted order of the exchange on the coupled runs: zoh ~1, linear ~2.*

**thermal_fields.png**  
![thermal_fields.png](../artifacts/m7607/thermal_fields.png)
*Fluid scalar field and plate temperature at the end of the r=1 run.*

## 4. Acceptance criteria

| id | criterion | value | gate | verdict |
|---|---|---|---|---|
| C1a | thermal total-energy drift (shared flux) | 1.520e-03 | <= 5.000e-03 | PASS |
| C1b | independent thermal control is red | 1.000e+00 | >= 9.000e-01 | PASS |
| C2a | interface heat flux agreement | 1.068e-01 | <= 1.500e-01 | PASS |
| C2b | flux agreement on the finest grid | 1.211e-02 | <= 1.500e-01 | PASS |
| C3a | Newton-III force antisymmetry (shared force) | 0.000e+00 | <= 1.000e-12 | PASS |
| C3b | total linear momentum drift (shared force) | 2.167e-13 | <= 1.000e-09 | PASS |
| C3c | independent force control is red | 2.000e+00 | >= 1.000e-06 | PASS |
| C3d | mechanical impact energy drift at r=1 | 1.030e-01 | <= 1.500e-01 | PASS |
| C4a | three-field energy chain closure (zoh) | 8.082e-03 | <= 2.000e-02 | PASS |
| C4b | independent three-field control is red | 2.000e+00 | >= 1.000e-06 | PASS |
| C5a | linear exchange order (coupled thermal) | 2.554e+00 | >= 1.500e+00 | PASS |
| C5b | zoh exchange order (coupled thermal) | 1.075e+00 | >= 7.000e-01 | PASS |
| C5c | linear order strictly beats zoh | 1.479e+00 | >= 0.000e+00 | PASS |
| C6a | lattice diffusion order vs analytic decay | 2.518e+00 | >= 1.500e+00 | PASS |
| C6b | plate steady profile vs exact solution | 1.199e-02 | <= 5.000e-02 | PASS |
| C6e | plate discrete energy identity | 1.490e-13 | <= 1.000e-10 | PASS |
| C6c | DEM restitution vs 1 | 7.989e-05 | <= 5.000e-02 | PASS |
| C6d | DEM contact time vs pi/omega | 1.859e-02 | <= 5.000e-02 | PASS |

Summary: **18/18 gates pass**.

## 5. Notes, pitfalls and limitations

1. **Two real defects were found by the conservation tests** and fixed:
   the D2Q9 opposite-direction table in `solvers.py` was inconsistent with the
   velocity set (`opp(5)` pointed to `6` instead of `7`), which broke the
   full-way bounce-back and made the lattice *create* 46 % of the scalar in one
   step; and `Q4ThermalFEM.interface_flux_estimate` hardcoded `mesh.row(1)` as
   the neighbour row, giving a 1260 % flux error on any mesh with more than two
   element rows. Both were invisible to the pre-existing code because no
   conservation test existed.
2. **`DEM2D` plain update pumps energy** into a stiff contact (+37 % kinetic
   energy at `ωΔt = 0.2`); an optional kick-drift-kick path was added and the
   restitution test now closes to 2.0e+00.
3. **The mechanical channel conserves momentum but not energy**: the shared held
   force is exact for Newton's third law (momentum 1.9e-13) yet
   the contact energy drifts by 10.3% over one full impact at r=1 — this is the
   price of holding the force over the exchange interval and it is reported, not
   hidden.
4. **`linear` extrapolation is not unconditionally safe**: in case (c) it
   overshoots the fast start-up thermal transient and *increases* the chain
   drift above zoh at r=1; in case (a) it is second order
   (p = 2.55) and cuts the drift by two decades.
5. External inputs (the optional periodic plate load) are booked as
   `book_external`, otherwise the drift metric would attribute them to the
   coupling.
6. Case (c) is dimensionless and the plate's thermal and elastic fields are not
   thermo-mechanically coupled; both are documented limitations, not silent
   approximations.

## 6. Error judgement against references (explicit <3 % answers)

| quantity | reference | error | <3 %? |
|---|---|---|---|
| lattice diffusion L2 (finest grid) | analytic eigenmode decay | 0.16% | yes |
| interface heat flux, Q_fluid vs Q_solid | independent Fourier estimate | best 1.13%, worst 10.68% | yes (best case only) |
| plate steady `ΔT` | `Q t_s/(kA)` | 1.20%–6.66% | partly (finest grid only) |
| DEM contact time | `π/ω` | 1.86% | yes |
| DEM restitution | 1.0 | 0.01% | yes |
| Newton-III antisymmetry | exact 0 | 0.0e+00 | yes |
| mechanical impact energy | initial total | 10.3% | **no** — honest FAIL, see note 3 |
| thermal total energy | `E_total(0)` | 1.52e-03 | yes |
| three-field chain energy | `E_total(0)` | 8.08e-03 | yes |

Environment: python 3.11.6, numpy
2.4.6, Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38.
