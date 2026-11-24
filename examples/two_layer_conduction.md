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
