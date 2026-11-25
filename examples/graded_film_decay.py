"""Asymmetric Robin cooling on a graded slab with a mapped cosine eigenmode."""
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
    runs = []
    for solver in ("FVM", "SPH"):
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
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(path) for path in
                          (__file__, rod.__file__, sph.__file__)],
                          reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFVM/src:../TensorSPH/src python examples/graded_film_decay.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
