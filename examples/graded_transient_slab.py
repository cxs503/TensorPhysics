"""Transient diffusion in a graded slab with an exact transformed-coordinate mode."""
import argparse
import json

import numpy as np
from tensorfvm.cht import rod
from tensorsph import heat_conduction_2d as sph

from conduction_provenance import provenance


def measure(n, solver):
    # Unit slab, C=1+2x and k=1/C in SI units. In the coordinate
    # y=x+x², heat per unit area is T dy and rightward flux is -dT/dy.
    # The transformed PDE is T_t=T_yy on [0,2].
    faces = np.linspace(0., 1., n+1)
    y = faces+faces**2
    width = np.diff(y)
    center = (y[1:]+y[:-1])/2
    capacity = n*width
    wave = np.pi/2
    times = np.linspace(0., 1., 2*n+1)
    decay = np.exp(-wave**2*times)
    # Capacity-weighted cell means, including the exact initial heat content.
    mode = .3*np.sin(wave*center)*np.sinc(wave*width/(2*np.pi))
    exact_temperature = 1.-center/2+decay[:, None]*mode
    face_metrics = {}
    if solver == "FEM":
        import torch
        from tensorfem import thermal

        nodes = torch.from_numpy(faces)
        elements = torch.stack((torch.arange(n), torch.arange(1, n+1)), dim=1)
        c = torch.from_numpy(capacity)
        model = thermal.ThermalModel(nodes[:, None], elements, 1/c, 1., c, torch.tensor([0, n]),
                                     nodes.new_tensor([1., 0.]))
        exact_temperature = 1.-y/2+decay[:, None]*(.3*np.sin(wave*y))
        initial = torch.from_numpy(exact_temperature[0].copy())
        initial[-1] = 0.
        result = thermal.solve_transient_thermal(model, torch.from_numpy(times), initial,
                                                 method="sdirk2", sparse=True)
        temperature = result.temperature.numpy()
        local = thermal.thermal_element_nodal_heat_steps(model, result).numpy()
        cuts = np.column_stack((local[:, 0, 0], -local[:, :, 1]))
        face_metrics["max_shared_face_mismatch_J_m2"] = float(abs(local[:, :-1, 1]+local[:, 1:, 0]).max())
    elif solver == "FVM":
        temperature, _, _, _, heat = rod.solve_transient_stepped_rod(
            np.full(n, 1/n), 1., 1/capacity, capacity, (1., 0.), times, exact_temperature[0],
            method="sdirk2", return_step_heat=True)
        cuts = np.column_stack((heat["left"][:, 0], heat["right"]))
    else:
        temperature, cuts = sph.solve_transient_layered_slab(
            exact_temperature[0], times, 1/n, (1., 0.), conductivity=1/capacity,
            volumetric_heat_capacity=capacity, method="exponential", h_factor=.9, return_x_heat=True)
        temperature, cuts = temperature.numpy(), cuts.numpy()
    exact_heat = times[1:, None]/2 + .3*np.expm1(-wave**2*times[1:, None])*np.cos(wave*y)/wave
    error = abs(np.cumsum(cuts, axis=0)-exact_heat)
    cell_temperature = (temperature[:, 1:]+temperature[:, :-1])/2 if solver == "FEM" else temperature
    balance = width*np.diff(cell_temperature, axis=0)+np.diff(cuts, axis=1)
    return dict(**face_metrics, solver=solver, cells=n, max_temperature_error_K=float(abs(temperature-exact_temperature).max()),
                max_cumulative_heat_error_J_m2=float(error.max()),
                west_east_cumulative_heat_errors_J_m2=error[:, [0, -1]].max(axis=0).tolist(),
                max_cell_balance_error_J_m2=float(abs(balance).max()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-fem", action="store_true", help="also verify LINE2 nodal temperatures and face heat")
    with_fem = parser.parse_args().with_fem
    sources = [__file__, rod.__file__, sph.__file__]
    paths = "../TensorFVM/src:../TensorSPH/src"
    solvers = ["FVM", "SPH"]
    if with_fem:
        from tensorfem import thermal

        solvers.append("FEM")
        sources.append(thermal.__file__)
        paths += ":../TensorFEM/src"
    runs = []
    for solver in solvers:
        previous = None
        for n in (16, 32, 64, 128):
            row = measure(n, solver)
            errors = np.array([row["max_temperature_error_K"], row["max_cumulative_heat_error_J_m2"],
                               *row["west_east_cumulative_heat_errors_J_m2"]])
            orders = [] if previous is None else np.log2(previous/errors).tolist()
            row.update(orders=orders, passed=bool(row["max_cell_balance_error_J_m2"] < 1e-11
                       and row.get("max_shared_face_mismatch_J_m2", 0.) < 1e-11
                       and (not orders or min(orders) > 0.) and (n < 128 or min(orders) > 1.85)))
            runs.append(row)
            previous = errors
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(path) for path in sources],
                          reproduction=f"OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH={paths} python examples/graded_transient_slab.py"
                                       + (" --with-fem" if with_fem else "")),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
