"""Compare SPH and SIPG wall responses to an affine harmonic heater with off-grid ends."""
import cmath
import json
import math

import numpy as np
from scipy.integrate import quad_vec
import torch

from conduction_provenance import provenance
from tensordg import heat_1d as dg
from tensorsph import heat_conduction_2d as sph


def benchmark():
    west, east = 1/6, 2/3
    left, right = .8-.4j, -.2+.7j
    omega, capacity, k = 3.4, 2.1, 1.3
    z = cmath.sqrt(1j*omega*capacity/k)

    def source(x):
        return left + (right-left)*(x-west)/(east-west)

    # Outward Fourier flux response to a unit point source at x on a unit slab.
    def integrand(x):
        kernels = np.array([cmath.sinh(z*(1-x)), cmath.sinh(z*x)])/cmath.sinh(z)
        return kernels*source(x)

    reference = torch.from_numpy(quad_vec(integrand, west, east, epsabs=1e-14)[0])
    generation = (east-west)*(left+right)/2
    runs = []
    for n in (16, 32, 64, 128):
        dx = 1/n
        edges = torch.arange(n+1, dtype=torch.float64)*dx
        lo, hi = edges[:-1].clamp(west, east), edges[1:].clamp(west, east)
        column_source = (hi-lo)/dx * source((hi+lo)/2)
        temperature, sph_faces = sph.solve_harmonic_slab_2d(
            column_source.repeat(8, 1), dx, (0., 0.), omega, conductivity=k,
            volumetric_heat_capacity=capacity, return_x_flux=True)
        modal_source = dg.project_linear_strip_source(left, right, 1., west, east, ncells=n, degree=2)
        _, _, dg_faces, cell_power = dg.solve_harmonic_heat_layer_1d(
            omega, (0., 0.), conductivity=k, capacity=capacity, ncells=n, degree=2,
            source_modes=modal_source, return_cell_power=True)
        sph_out = torch.stack((-sph_faces[0], sph_faces[-1]))
        dg_out = torch.stack((-dg_faces[0], dg_faces[-1]))
        errors = {name: difference.abs().max().item() for name, difference in (
            ("sph", sph_out-reference), ("dg", dg_out-reference), ("cross", sph_out-dg_out))}
        orders = {name: math.log2(runs[-1]["wall_errors_W_m2"][name]/value)
                  for name, value in errors.items()} if runs else {}
        balance = {
            "sph": abs(sph_out.sum().item()+1j*omega*capacity*temperature.mean().item()-generation),
            "dg": abs(dg_out.sum().item()+cell_power["storage"].sum().item()-generation),
        }
        runs.append(dict(columns=n, wall_errors_W_m2=errors, orders=orders,
                         global_balance_W_m2=balance))
    passed = (all(max(run["global_balance_W_m2"].values()) < 1e-10 for run in runs)
              and all(min(run["orders"].values()) > 1.5 for run in runs[1:])
              and runs[-1]["wall_errors_W_m2"]["sph"] < 1e-4
              and runs[-1]["wall_errors_W_m2"]["dg"] < 1e-8)
    return dict(passed=passed, runs=runs, sources=[provenance(path) for path in
                (__file__, sph.__file__, dg.__file__)],
                reference="Unit slab with zero wall phasors. Integrate the affine complex source against each "
                          "Dirichlet wall Green function. Errors are the larger absolute phasor error of the two walls.",
                reference_outward_W_m2=[[v.real, v.imag] for v in reference.tolist()],
                reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src:../TensorDG "
                             "python examples/cross_sph_dg_strip_heater.py")


if __name__ == "__main__":
    result = benchmark()
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result["passed"] else 1)
