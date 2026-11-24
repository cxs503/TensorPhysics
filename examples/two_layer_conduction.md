# Steady two-layer conduction

Run `python examples/two_layer_conduction.py` from TensorPhysics in a Python
environment containing TensorFEM, TensorFVM, TensorSPH and TensorDEM. The
command emits JSON with solver source revisions and hashes, independent series
resistance predictions, computed temperatures and fluxes, absolute errors,
tolerances and pass flags. It exits unsuccessfully if any case misses its gate.

The unit-area slab has two equal 0.5 m layers, a 20 K wall-temperature
difference and zero internal heating. Both heat-flow directions are tested.
For layer conductivities 10 and 1 W/(m K), perfect contact gives an exact
heat-flux magnitude of 36.363636 W/m². FEM, FVM and DEM also exercise a
finite interface resistance of 0.3 m² K/W, with distinct left and right
interface temperatures. The DEM comparison uses prescribed link resistances.

For SPH, both reported wall fluxes come from the particle-to-ghost power
sum, normalized by the represented wall area and the lattice second moment.
The default endpoint harmonic mean produces maximum absolute wall flux
errors of 0.15603, 0.07785 and 0.03888 W/m² at 16, 32 and 64 cells along x
for the forward conductivity-contrast case. With the optional layered
pair resistance, the maximum over both heat-flow directions and all three
meshes was below 1.9e-9 W/m². The option assumes a one-dimensional,
grid-aligned material profile. SPH uses perfect contact in this comparison.

The CPU float64 verification comprised 48 cases with TensorFEM
`064df7a`, TensorFVM `782f5b99`, TensorDEM `cc343bb` and TensorSPH
`c11a912`.
