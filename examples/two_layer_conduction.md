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
