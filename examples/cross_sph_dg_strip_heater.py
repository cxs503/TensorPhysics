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
    # These two internal faces coincide on every mesh. The volume includes
    # the east heater edge, hence only part of the source contributes.
    a, b = .25, .75
    local_generation = (east-a)*source((a+east)/2)

    def interior_reference(y):
        # G(x,y) = sinh(z min(x,y))*sinh(z(1-max(x,y))) / (k*z*sinh(z)).
        # Differentiate in x for flux, and integrate in x for stored energy.
        def flux_kernel(x):
            if x < y:
                return -cmath.cosh(z*x)*cmath.sinh(z*(1-y))/cmath.sinh(z)
            return cmath.sinh(z*y)*cmath.cosh(z*(1-x))/cmath.sinh(z)

        pivot = min(max(y, a), b)
        integral = (cmath.sinh(z*(1-y))*(cmath.cosh(z*pivot)-cmath.cosh(z*a))
                    + cmath.sinh(z*y)*(cmath.cosh(z*(1-pivot))-cmath.cosh(z*(1-b))))
        energy_kernel = capacity*integral/(k*z*z*cmath.sinh(z))
        return source(y)*np.array([flux_kernel(a), flux_kernel(b), energy_kernel])

    exact_west, exact_east, exact_energy = quad_vec(interior_reference, west, east, points=[a])[0]
    reference_balance = abs(exact_west-exact_east+local_generation-1j*omega*exact_energy)
    # Fixed probes include both source discontinuities and points on either side.
    probes = np.array([.09, west, .31, .53, east, .83])
    def temperature_kernel(y):
        return (source(y)*np.sinh(z*np.minimum(probes,y))*np.sinh(z*(1-np.maximum(probes,y)))
                / (k*z*cmath.sinh(z)))

    exact_temperature = quad_vec(temperature_kernel, west, east,
                                 points=probes[(probes > west) & (probes < east)], epsabs=1e-14)[0]
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
        _, dg_modes, dg_faces, cell_power = dg.solve_harmonic_heat_layer_1d(
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
        lo_idx, hi_idx = n//4, 3*n//4
        storage = {"sph": capacity*dx*temperature[:, lo_idx:hi_idx].mean(dim=0).sum().item(),
                   "dg": dg.integrate_heat_stack_energy(dg_modes[None], [1.], capacity, a, b).item()}
        local = {}
        for name, faces in (("sph", sph_faces), ("dg", dg_faces)):
            q_west, q_east = faces[lo_idx].item(), faces[hi_idx].item()
            local[name] = dict(flux_error_W_m2=max(abs(q_west-exact_west), abs(q_east-exact_east)),
                               energy_error_J_m2=abs(storage[name]-exact_energy),
                               balance_W_m2=abs(q_west-q_east+local_generation-1j*omega*storage[name]))
            if runs:
                for key in ("flux_error_W_m2", "energy_error_J_m2"):
                    local[name][key+"_order"] = math.log2(runs[-1]["control_volume"][name][key]/local[name][key])
        # SPH returns centre values; interpolate linearly between adjacent
        # centres. Evaluate the DG polynomial at each probe's cell coordinate.
        centres = (np.arange(n)+.5)*dx
        sph_temperature = np.interp(probes, centres, temperature.mean(dim=0).numpy())
        cells = np.floor(probes*n).astype(int)
        samples, _ = dg.sample_heat_stack_fields(dg_modes[None], [1.], k, probes*n-cells)
        dg_temperature = samples[0, cells, np.arange(len(probes))].numpy()
        probe_errors = {name: np.abs(diff).tolist() for name, diff in (
            ("sph", sph_temperature-exact_temperature), ("dg", dg_temperature-exact_temperature),
            ("cross", sph_temperature-dg_temperature))}
        maximum = {name: max(values) for name, values in probe_errors.items()}
        probe_orders = {name: math.log2(runs[-1]["max_probe_error_K"][name]/value)
                        for name, value in maximum.items()} if runs else {}
        runs.append(dict(columns=n, wall_errors_W_m2=errors, orders=orders,
                         global_balance_W_m2=balance, control_volume=local, probe_errors_K=probe_errors,
                         max_probe_error_K=maximum, probe_orders=probe_orders))
    passed = (all(max(run["global_balance_W_m2"].values()) < 1e-10 for run in runs)
              and all(min(run["orders"].values()) > 1.5 for run in runs[1:])
              and runs[-1]["wall_errors_W_m2"]["sph"] < 1e-4
              and runs[-1]["wall_errors_W_m2"]["dg"] < 1e-8)
    passed &= reference_balance < 1e-12
    passed &= all(min(row["probe_orders"].values()) > 1 for row in runs[1:])
    for row in runs:
        for data in row["control_volume"].values():
            passed &= data["balance_W_m2"] < 1e-10
            if "flux_error_W_m2_order" in data:
                passed &= min(data["flux_error_W_m2_order"], data["energy_error_J_m2_order"]) > 1.5
    return dict(passed=bool(passed), runs=runs, sources=[provenance(path) for path in
                (__file__, sph.__file__, dg.__file__)],
                reference="Unit slab with zero wall phasors. Integrate the affine complex source against each "
                          "Dirichlet wall Green function. Errors are the larger absolute phasor error of the two walls.",
                probe_positions_m=probes.tolist(),
                reference_probe_temperature_K=[[v.real, v.imag] for v in exact_temperature],
                control_volume_m=[a,b],
                reference_energy_J_m2=[exact_energy.real, exact_energy.imag],
                reference_balance_W_m2=reference_balance,
                reference_outward_W_m2=[[v.real, v.imag] for v in reference.tolist()],
                reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src:../TensorDG "
                             "python examples/cross_sph_dg_strip_heater.py")


if __name__ == "__main__":
    result = benchmark()
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result["passed"] else 1)
