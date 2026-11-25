"""Asymmetric Robin cooling on a graded slab with a mapped cosine eigenmode."""
import argparse
import json

import numpy as np
from tensorfvm.cht import rod
from tensorsph import heat_conduction_2d as sph

from conduction_provenance import provenance


def measure(n, solver):
    # For C=1+2x and k=1/C, y=x+x² transforms C*T_t=(k*T_x)_x
    # into T_t=T_yy and preserves rightward heat flux q=-T_y.
    x = np.linspace(0., 1., n+1)
    y = x+x*x
    width = np.diff(y)
    capacity = n*width
    wave, phase = .8, .55
    angle = wave*y-phase
    films = wave*np.tan([-angle[0], angle[-1]])
    times = np.linspace(0., 2., 2*n+1)
    decay = np.exp(-wave**2*times)
    mean_mode = np.cos((angle[1:]+angle[:-1])/2)*np.sinc(wave*width/(2*np.pi))
    exact = decay[:, None]*mean_mode
    if solver == "FVM":
        state, faces, _, _, heat = rod.solve_transient_stepped_rod(
            np.full(n, 1/n), 1., 1/capacity, capacity, (0., 0.), times, exact[0],
            end_types=("robin", "robin"), end_film_coefficient=films,
            method="sdirk2", return_step_heat=True)
        wall = faces[:, [0, -1]]
        cuts = np.column_stack((heat["left"][:, 0], heat["right"]))
    elif solver == "FEM":
        import torch
        from tensorfem import thermal

        nodes = torch.from_numpy(x)
        conn = torch.stack((torch.arange(n), torch.arange(1, n+1)), dim=1)
        c = torch.from_numpy(capacity)
        model = thermal.ThermalModel(
            nodes[:, None], conn, 1/c, 1., c, conn.new_empty(0), nodes[:0],
            convection_edges=conn.new_tensor([[0, 0], [n, n]]),
            convection_coefficient=torch.from_numpy(films), ambient_temperature=nodes.new_zeros(2))
        result = thermal.solve_transient_thermal(model, torch.from_numpy(times), torch.from_numpy(np.cos(angle)),
                                                 sparse=True, method="sdirk2")
        state = result.temperature[:, conn].mean(dim=2).numpy()
        wall = result.temperature[:, [0, -1]].numpy()
        local = thermal.thermal_element_nodal_heat_steps(model, result).numpy()
        cuts = np.column_stack((local[:, 0, 0], -local[:, :, 1]))
    elif solver == "DG":
        import torch
        from tensordg import poisson_face_powers_2d, solve_transient_diffusion_2d

        c = np.tile(capacity[:, None], (1, 2))
        options = dict(bc=dict(left="robin", right="robin", bottom="neumann", top="neumann"),
                       robin_films=dict(left=(films[0], 0.), right=(films[1], 0.)), conductivity=1/c)
        _, _, modes, averages = solve_transient_diffusion_2d(
            lambda x, y: torch.cos(wave*(x+x.square())-phase), times, ncells=(n, 2), degree=1,
            capacity=c, method="sdirk2", return_interval_means=True, **options)
        # Average across the strip and retain the constant and linear x coefficients.
        axial = modes[..., 0].mean(dim=2).numpy()
        state = axial[:, :, 0]
        wall = np.column_stack((state[:, 0]-axial[:, 0, 1], state[:, -1]+axial[:, -1, 1]))
        cuts = np.stack([poisson_face_powers_2d(avg, **options)[0].sum(dim=1).numpy()
                         for avg in averages])*np.diff(times)[:, None]
    else:
        state, cuts, wall = sph.solve_transient_layered_slab(
            exact[0], times, 1/n, (None, None), conductivity=1/capacity,
            volumetric_heat_capacity=capacity, x_wall_films=tuple((h, 0.) for h in films),
            method="exponential", h_factor=.9, return_x_heat=True, return_wall_temperatures=True)
        state, cuts, wall = state.numpy(), cuts.numpy(), wall.numpy()
    exact_wall = decay[:, None]*np.cos(angle[[0, -1]])
    exact_heat = -np.expm1(-wave**2*times[1:, None])*np.sin(angle)/wave
    cumulative = cuts.cumsum(axis=0)
    balance = width*np.diff(state, axis=0)+np.diff(cuts, axis=1)
    error = abs(cumulative-exact_heat)
    return dict(solver=solver, cells=n, films_W_m2_K=films.tolist(),
                max_cell_mean_error_K=float(abs(state-exact).max()),
                max_wall_temperature_errors_K=abs(wall-exact_wall).max(axis=0).tolist(),
                max_section_heat_error_J_m2=float(error.max()),
                max_wall_heat_errors_J_m2=error[:, [0, -1]].max(axis=0).tolist(),
                final_outward_wall_heat_J_m2=(cumulative[-1, [0, -1]]*[-1, 1]).tolist(),
                final_exact_outward_wall_heat_J_m2=(exact_heat[-1, [0, -1]]*[-1, 1]).tolist(),
                max_cell_balance_error_J_m2=float(abs(balance).max()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="include the LINE2 and SIPG solutions")
    all_solvers = parser.parse_args().all
    modules = [__file__, rod.__file__, sph.__file__]
    paths = "../TensorFVM/src:../TensorSPH/src"
    solvers = ["FVM", "SPH"]
    if all_solvers:
        from tensorfem import thermal
        from tensordg import transient_diffusion_2d, poisson_2d_bc

        solvers.extend(["FEM", "DG"])
        modules.extend([thermal.__file__, transient_diffusion_2d.__file__, poisson_2d_bc.__file__])
        paths += ":../TensorFEM/src:../TensorDG"
    runs = []
    for solver in solvers:
        previous = None
        for n in (16, 32, 64, 128):
            row = measure(n, solver)
            errors = np.array([row["max_cell_mean_error_K"], *row["max_wall_temperature_errors_K"],
                               row["max_section_heat_error_J_m2"], *row["max_wall_heat_errors_J_m2"]])
            orders = [] if previous is None else np.log2(previous/errors).tolist()
            row.update(orders=orders, passed=bool(row["max_cell_balance_error_J_m2"] < 1e-11
                       and (not orders or min(orders) > 0.) and (n < 128 or min(orders) > 1.85)))
            runs.append(row)
            previous = errors
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(path) for path in modules],
                          reproduction=f"OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH={paths} python examples/graded_film_decay.py"
                                       + (" --all" if all_solvers else "")),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
