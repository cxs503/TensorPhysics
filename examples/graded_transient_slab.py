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
    if solver == "DG":
        import torch
        from tensordg import poisson_face_powers_2d, solve_transient_diffusion_2d

        def initial_field(x, transverse):
            coordinate = x+x.square()
            return 1-coordinate/2+.3*torch.sin(wave*coordinate)

        c = np.tile(capacity[:, None], (1, 2))
        options = dict(bc=dict(left="dirichlet", right="dirichlet", bottom="neumann", top="neumann"),
                       dirichlet_values=dict(left=lambda x,y: torch.ones_like(x+y),
                                             right=lambda x,y: torch.zeros_like(x+y)), conductivity=1/c)
        _, _, states, means = solve_transient_diffusion_2d(
            initial_field, times, ncells=(n, 2), degree=1, capacity=c, method="sdirk2",
            return_interval_means=True, **options)
        center_x = (faces[1:]+faces[:-1])/2
        center_y = center_x+center_x**2
        exact_temperature = 1.-center_y/2+decay[:, None]*(.3*np.sin(wave*center_y))
        # The degree-one tensor basis has value equal to its constant mode at the center.
        centers = states[..., 0, 0].numpy()
        temperature = centers
        exact_temperature = exact_temperature[:, :, None]
        cuts = []
        max_local = 0.
        for step, (dt, avg) in enumerate(zip(np.diff(times), means)):
            qx, qy = poisson_face_powers_2d(avg, **options)
            cuts.append(dt*qx.sum(dim=1).numpy())
            stored = (centers[step+1]-centers[step])*c/(2*n)
            local = stored+dt*(qx.diff(dim=0)+qy.diff(dim=1)).numpy()
            max_local = max(max_local, float(abs(local).max()))
        cuts = np.asarray(cuts)
        face_metrics["max_2d_cell_balance_error_J_m2"] = max_local
    elif solver == "FEM":
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
    if solver == "DG":
        cell_temperature = temperature.mean(axis=2)
    balance = width*np.diff(cell_temperature, axis=0)+np.diff(cuts, axis=1)
    return dict(**face_metrics, solver=solver, cells=n, max_temperature_error_K=float(abs(temperature-exact_temperature).max()),
                max_cumulative_heat_error_J_m2=float(error.max()),
                west_east_cumulative_heat_errors_J_m2=error[:, [0, -1]].max(axis=0).tolist(),
                max_cell_balance_error_J_m2=float(abs(balance).max()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-fem", action="store_true", help="also verify LINE2 nodal temperatures and face heat")
    parser.add_argument("--with-dg", action="store_true", help="also verify SIPG center temperatures and face heat")
    args = parser.parse_args()
    sources = [__file__, rod.__file__, sph.__file__]
    paths = "../TensorFVM/src:../TensorSPH/src"
    solvers = ["FVM", "SPH"]
    if args.with_fem:
        from tensorfem import thermal

        solvers.append("FEM")
        sources.append(thermal.__file__)
        paths += ":../TensorFEM/src"
    if args.with_dg:
        from tensordg import transient_diffusion_2d, poisson_2d_bc

        solvers.append("DG")
        sources.extend([transient_diffusion_2d.__file__, poisson_2d_bc.__file__])
        paths += ":../TensorDG"
    runs = []
    for solver in solvers:
        previous = None
        for n in (16, 32, 64, 128):
            row = measure(n, solver)
            errors = np.array([row["max_temperature_error_K"], row["max_cumulative_heat_error_J_m2"],
                               *row["west_east_cumulative_heat_errors_J_m2"]])
            orders = [] if previous is None else np.log2(previous/errors).tolist()
            row.update(orders=orders, passed=bool(row["max_cell_balance_error_J_m2"] < 1e-11
                       and row.get("max_2d_cell_balance_error_J_m2", 0.) < 1e-11
                       and row.get("max_shared_face_mismatch_J_m2", 0.) < 1e-11
                       and (not orders or min(orders) > 0.) and (n < 128 or min(orders) > 1.85)))
            runs.append(row)
            previous = errors
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(path) for path in sources],
                          reproduction=f"OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH={paths} python examples/graded_transient_slab.py"
                                       + (" --with-fem" if args.with_fem else "") + (" --with-dg" if args.with_dg else "")),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
