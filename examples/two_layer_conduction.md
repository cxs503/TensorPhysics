# Steady two-layer conduction

Run `python examples/two_layer_conduction.py` from TensorPhysics in a Python
environment containing TensorFEM, TensorFVM, TensorSPH and TensorDEM. The
command emits JSON with solver source revisions and hashes, independent series
resistance predictions, computed temperatures and fluxes, absolute errors,
tolerances and pass flags. It exits unsuccessfully if any case misses its gate.

The unit-area slab has two equal 0.5 m layers, a 20 K wall-temperature
difference and zero internal heating. Both heat-flow directions are tested.
For layer conductivities 10 and 1 W/(m K), perfect contact gives an exact
heat-flux magnitude of 36.363636 W/m². FEM, FVM, DEM and layered SPH also exercise a
finite interface resistance of 0.3 m² K/W, with distinct left and right
interface temperatures. The DEM comparison uses prescribed link resistances.

For SPH, both reported wall fluxes come from the particle-to-ghost power
sum, normalized by the represented wall area and the lattice second moment.
The default endpoint harmonic mean produces maximum absolute wall flux
errors of 0.15603, 0.07785 and 0.03888 W/m² at 16, 32 and 64 cells along x
for the forward conductivity-contrast case. With the optional layered
pair resistance, the maximum over both heat-flow directions and all three
meshes was below 1.9e-9 W/m². The option assumes a one-dimensional,
grid-aligned material profile. With a contact resistance of 0.3 m² K/W and
the same 10:1 conductivity contrast, the analytic flux magnitude is
23.529412 W/m² and the interface jump magnitude is 7.058824 K. In the 12
finite-contact SPH runs, one-sided extrapolation from the solved particle
temperatures gave jump errors below 4.1e-11 K. Wall flux errors were below
2.0e-9 W/m².

The CPU float64 verification comprised 60 cases with TensorFEM
`064df7a`, TensorFVM `782f5b99`, TensorDEM `cc343bb` and TensorSPH
`eabc543`.

## Transient FEM, FVM, DEM and SPH with unequal heat capacities

Run `OMP_NUM_THREADS=1 python examples/two_layer_transient.py` for a separate
fixed-wall unit slab with a finite thermal contact, conductivities (1,4) and
volumetric heat capacities (1,256/81). Temperatures are perturbations about
a common reference. Each half starts from a sine profile chosen to satisfy
continuous flux and the contact jump condition and both decay at the same
analytic rate. At t=0.015 s the jump is 0.75365664 K and the change in stored
heat per unit area is +0.02490182 J/m². The verification compares the field
at each solver's own locations, the two interface traces and the integrated
change in stored heat. FEM uses consistent mass, FVM uses cell heat capacities
and face contact conductances, and DEM uses fixed links and lumped nodal
heat capacities. All three use Crank–Nicolson. SPH
uses the calibrated particle operator and explicit midpoint. Each uses three
spatial resolutions (16, 32, 64) with time-step refinement. The script checks
all three errors decrease by a factor above three at each refinement and
checks absolute tolerances on the finest grid.


At 64 cells, FVM maximum temperature, interface-jump and stored-heat change
errors were 0.0001754 K, 0.001723 K and 9.953e-7 J/m², respectively.
The independent TensorFVM two-cell relaxation test refines time alone and
verifies the Crank–Nicolson temporal order using an exact exponential.

The DEM chain uses unit cross section, separate nodes on the two sides of
the contact, and prescribed link resistances. Each bulk segment assigns half
its heat capacity to each end. On 64 segments the maximum temperature,
interface-jump and stored-heat errors were 0.0001507 K, 0.0001676 K and
5.890e−6 J/m². The contact network comparison verifies heat transport for
these specified links; it does not infer real grain contact areas.

The transient report also reconstructs the left and right Fourier fluxes
using independent three-point, one-sided temperature derivatives. It reports
both relative errors against 2.38815776 W/m² at t=0.015 s. These diagnostics
are separate from the temperature, jump and storage acceptance checks.
With 64 segments the larger absolute flux errors were 0.003730 (FEM),
0.011194 (FVM), 0.005703 (DEM), and 0.027944 W/m² (SPH).
SPH's left flux error rose from 0.001060 at 32 cells to 0.010843 W/m² at
64 cells even as the temperature converged. The extra 128-cell level
(`python examples/two_layer_transient.py --extended`) reduced left and right
SPH errors to 0.008461 and 0.015902 W/m². The measured fluxes do not
establish uniform second-order convergence of interface derivatives.

The report additionally checks contact heat transfer against the exact Fourier
flux. FEM and DEM use the jump across distinct interface nodes divided by R.
FVM uses the cell temperature difference divided by the series resistance of
both half cells and the contact. SPH counts each particle pair crossing the
internal plane once. At 128 cells the relative errors are 0.00355% (FEM),
0.00789% (FVM), 0.00556% (DEM) and 0.01254% (SPH). The contact-transfer gate
requires the relative error to decrease by a factor above three at each
refinement and to be below 0.1% at 64 cells and finer.

For the SPH left gradient at 32 cells, applying the one-sided derivative to
the analytic temperatures produces a signed flux error of +0.041355 W/m².
Replacing those samples with the computed temperatures contributes
−0.042415 W/m². Their near cancellation explains the small total error at
that resolution. At 64 cells the contributions are +0.011412 and −0.022256
W/m². The native pair-transfer error instead decreases monotonically across
all four meshes.

## Convective cooling and finite contact

The steady command also runs eight FEM/FVM cases with ambient temperatures
310 and 290 K (and the reversed pair), film coefficients 12 and 3 W/(m² K),
and perfect or finite contact. With R=0.3 m² K/W and conductivities 10 and
1 W/(m K), the forward exact flux is 15.789474 W/m² and jump is 4.736842 K.
The FEM comparison uses adjacent element gradients for flux and boundary
nodes for temperature. FVM uses its boundary ledger and extrapolates the
surface temperatures from adjacent cell values.

Run `OMP_NUM_THREADS=1 python examples/two_layer_film_decay.py` for the
transient comparison. On a unit slab with a contact at x=0.5 and zero ambient
temperature perturbation, the modes sin(ax+φ) and −A sin(b(1−x)+ψ) have
a=3π/2, b=4π/3, φ=0.25 and ψ=0.4. Continuity of conductive flux sets A;
the contact resistance is the jump divided by flux. The film coefficients
a cot(φ) and 4b cot(ψ) satisfy the Robin conditions. Conductivities (1,4)
and volumetric heat capacities (1,4(b/a)²) give common decay exp(−a²t).
The comparison uses Crank–Nicolson and halves both the mesh spacing and the
time step at each refinement. At t=0.015 s and 64 cells, maximum wall
temperature errors were 2.685e-5 K (FEM) and 2.830e-5 K (FVM). Stored heat
changes are compared with the analytic integral over the two layers.

The steady film comparison includes a DEM chain with a film resistance
1/(hA) on each external link and fixed ambient reservoirs. Surface
temperatures and material-side heat fluxes come from the physical chain.
This gives twelve film cases and 72 steady cases overall.

The transient film comparison also uses lumped half-segment capacities on
the wall and contact nodes, with zero capacity at the prescribed ambient
reservoirs. It reports four resolutions per solver. At 64 segments the DEM
stored-heat change error is 1.12033e-5 J/m² against the existing absolute
limit of 1e-5. Of the signed error, +9.91130e-6 is obtained by integrating
the exact mode with the lumped capacities, and +1.29198e-6 comes from the
computed nodal temperatures. Halving the time step alone gives 1.12012e-5.
At 128 segments the error falls to 2.80181e-6 J/m². The discrete balance
between time-integrated inward film powers and nodal heat change closes
within 3e-15 J/m² at every tested resolution. All other gates pass.
The original point-sampled transient film command returns exit status 1 for the
64-segment discrepancy and still emits all results, including the failed row.


The additional `DEM_cell` case puts one capacity in each physical segment,
uses the analytic cell-average initial mode and compares with analytic
cell averages at the final time. Both ambient reservoirs have zero capacity.
Each internal link contains two half-segment resistances and any interface
resistance. This prescribed one-dimensional network has the same spatial
resistances as a cell-centred finite-volume model. The four-level film
campaign now has 16 rows, of which 15 pass. The original 64-segment nodal
DEM failure is retained. On the same mesh its separate, per-material Simpson
integral of the numerical nodal temperature change differs from the
continuum value by 1.3043e-6 J/m². Its physical nodal storage and film-power
balance are still evaluated with its own lumped capacities.

For the cell-centred network, absolute stored-heat errors are 5.1410e-5,
1.2948e-5, 3.2426e-6 and 8.1099e-7 J/m² at 16, 32, 64 and 128 segments.
This network satisfies the original limits at 64 and 128 cells. Its
time-integrated film-power discrepancy is below 3e-15 J/m² on all four
meshes. Reproduce both mappings with the same film-decay command above.

The SPH film comparison uses the slab stencil with its wall temperatures
solved from the external film laws at each stage. The steady campaign
includes both flux directions, with or without layer contact resistance:
76/76 total cases passed. For the transient film mode, inward film powers
at midpoint times are integrated independently of the particle storage.
At 64 cells the wall temperature, jump and stored-heat errors were
6.303e-5 K, 1.488e-3 K and 3.474e-6 J/m². At 128 cells they were
1.574e-5 K, 3.653e-4 K and 8.703e-7 J/m². The worst SPH heat balance
residual on the four meshes was below 9e-16 J/m².
The point-sampled film-decay report includes 20 rows. The original nodal DEM
case at 64 segments remains its only failing row (`--sampled-dem`).

The default nodal DEM initialization now projects the continuum initial
condition onto the two lumped-capacity layers independently. Adding the
constant (analytical layer heat minus sampled layer heat) divided by the
layer capacity minimizes the capacity-weighted squared perturbation under
the heat constraint. Only the initial state is projected. All final
temperatures, wall values, interface jumps and heat changes use the original
continuum references. At 64 segments the absolute heat-change error is
2.416e-6 J/m² and the maximum point-temperature error is 3.622e-4 K
(original point sampling gives 1.612e-4 K). The 128-segment heat-change
error is 6.085e-7 J/m². All 20 rows pass the default campaign. Reproduce
with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python examples/two_layer_film_decay.py`.

## Transversely varying wall flux in a cooling square

Run `OMP_NUM_THREADS=1 python examples/spatial_film_decay.py` for a separate
homogeneous unit square (unit conductivity and volumetric capacity, unit
thickness). Both x walls exchange heat with zero ambient perturbation at
h=2 tan(1). The exact field is cos(2(x−1/2)) multiplied by
exp(−4t)+0.5 cos(2πy) exp(−(4+4π²)t). The transverse cosine satisfies both
the periodic SPH y boundary and the insulated FVM y boundary.
The nonzero mean gives a separate analytic integral of stored heat change.
Maximum local wall flux errors at t=0.02 s were 0.005097, 0.001286 and
0.0003223 W/m² for FVM and 0.02246, 0.01184 and 0.006056 W/m² for SPH
at 16, 32 and 64 cells per side. The SPH flux uses the wall-row ghost-pair
powers; FVM uses its half-cell surface temperature and the film law.

The spatial film study also runs Q4 FEM with consistent capacity, linear
edge convection and Crank–Nicolson integration. The CPU solve uses sparse
assembly and SciPy sparse LU. For FEM the temperature comparison is at
nodes; wall comparisons are at edge midpoints. Integrating the two linear
wall traces along each edge and averaging in time gives the inward heat.
The stored heat is evaluated independently with the consistent capacity
matrix. With `--extended` the four levels span 16 to 128 elements per side,
up to 16,641 FEM nodes. The largest FEM discrete heat balance residual was
2.55e-14 J.

The report also reconstructs inward Fourier flux using three columns on
each side (one-sided second-order derivatives). On the 128 grid, maximum
film-law flux errors were 1.664e-4 (FEM), 8.061e-5 (FVM) and 3.059e-3
(SPH) W/m². Gradient flux errors at the same locations were 2.583e-4,
4.742e-4 and 5.740e-3 W/m², respectively. All decreased across both
standard and extended meshes. Reproduce with
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python examples/spatial_film_decay.py --extended`.

### Grid-aligned anisotropic cooling

Run `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python examples/spatial_film_decay.py --anisotropic --extended`
for FEM and FVM with conductivity components 2 and 0.5 W/(m K) along x
and y and unit volumetric heat capacity. The two analytical modes decay
at rates 8 and 8 + 2π² per second. The x-wall film coefficient is
4 tan(1) W/(m² K). Each refinement must achieve temperature order ≥1.9.
On the 128 grid the temperature errors were 2.674e-5 K (FEM) and
2.677e-5 K (FVM). Independently reconstructed Fourier flux errors were
5.158e-4 and 9.337e-4 W/m², below the 1e-3 W/m² acceptance limit.

### Signed heat supplied by fixed-temperature walls

Run `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python examples/asymmetric_slab_heat.py`.
This separate homogeneous unit slab uses zero wall temperatures, unit material
properties and the initial perturbation sin(πx) + 0.8 sin(2πx). Each mode
decays by exp(−k²t). Its time-integrated Fourier flux toward +x is
A expm1(−k²t) cos(kx)/k. Inward heat uses this sign at the left wall and
the opposite sign at the right. The right-wall cumulative heat input changes
from +0.00679407 J/m² at 0.03 s to −0.05187142 J/m² at 0.08 s.

The temperature comparison uses the same analytic cell averages for each
solver. DG uses the quadratic modal constant coefficient, FEM integrates
the linear shape functions and FVM uses cell values. The initial condition
is projected into each space. FEM uses the consistent-mass L2 projection;
pointwise nodal initialization gave a 64-cell temperature error of 5.710e-4 K
at 0.03 s, versus 2.646e-4 K with projection. At each wall the heat is
obtained from integrated SIPG flux, integrated FEM constrained-node reactions
or FVM half-cell conduction, respectively. The two times and four grid
levels give 24 passing comparisons. The worst absolute discrete heat balance
residual was 5.30e-13 J/m² or less. JSON output records each solver revision
and the exact source-file hashes.

SPH also runs in the same campaign, bringing the total to 32 comparisons.
A transverse periodic row with unit thickness represents the uniform slab.
The explicit midpoint integrator advances cell-average initial temperatures,
integrates ghost-pair powers at its midpoint and divides by the row's wall
area. Rate and power use the same second-moment calibration. On 64 particles
the maximum wall heat errors at 0.03 and 0.08 s were 8.444e-5 and
6.053e-5 J/m². Halving the time step changed the final temperatures by
at most 8.891e-8 K across the two runs.
