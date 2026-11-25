"""Transient diffusion in a graded slab with an exact transformed-coordinate mode."""
import argparse
import json

import numpy as np
from tensorfvm.cht import rod
from tensorsph import heat_conduction_2d as sph

from conduction_provenance import provenance


def measure(n, solver, insulated=False, insulated_end=None):
    # Unit slab, C=1+2x and k=1/C in SI units. In the coordinate
    # y=x+x², heat per unit area is T dy and rightward flux is -dT/dy.
    # The transformed PDE is T_t=T_yy on [0,2].
    faces = np.linspace(0., 1., n+1)
    y = faces+faces**2
    width = np.diff(y)
    center = (y[1:]+y[:-1])/2
    capacity = n*width
    mixed = insulated_end is not None
    walls = [1., 0.] if not mixed else [.5, .5]
    for side, name in enumerate(("west", "east")):
        if insulated or insulated_end == name:
            walls[side] = None
    cosine = insulated or insulated_end == "west"
    wave = np.pi/4 if mixed else np.pi/2
    times = np.linspace(0., 1., 2*n+1)
    decay = np.exp(-wave**2*times)
    # Capacity-weighted cell means, including the exact initial heat content.
    def steady(coordinate):
        return .5 if insulated or mixed else 1.-coordinate/2

    def amplitude(coordinate, backend=np):
        oscillation = backend.cos if cosine else backend.sin
        return .3*oscillation(wave*coordinate)

    mode = amplitude(center)*np.sinc(wave*width/(2*np.pi))
    exact_temperature = steady(center)+decay[:, None]*mode
    face_metrics = {}
    if solver == "DG":
        import torch
        from tensordg import poisson_face_powers_2d, solve_transient_diffusion_2d

        def initial_field(x, transverse):
            coordinate = x+x.square()
            return steady(coordinate)+amplitude(coordinate, torch)

        c = np.tile(capacity[:, None], (1, 2))
        bc = dict.fromkeys(("left", "right", "bottom", "top"), "neumann")
        values = {}
        for name, wall in zip(("left", "right"), walls):
            if wall is not None:
                bc[name] = "dirichlet"
                values[name] = lambda x, y, value=wall: torch.full_like(x+y, value)
        options = dict(bc=bc, dirichlet_values=values, conductivity=1/c)
        _, _, states, means = solve_transient_diffusion_2d(
            initial_field, times, ncells=(n, 2), degree=1, capacity=c, method="sdirk2",
            return_interval_means=True, **options)
        center_x = (faces[1:]+faces[:-1])/2
        center_y = center_x+center_x**2
        exact_temperature = steady(center_y)+decay[:, None]*amplitude(center_y)
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
        fixed = torch.tensor([i for i, wall in zip((0, n), walls) if wall is not None], dtype=torch.long)
        prescribed = nodes.new_tensor([wall for wall in walls if wall is not None])
        model = thermal.ThermalModel(nodes[:, None], elements, 1/c, 1., c, fixed, prescribed)
        exact_temperature = steady(y)+decay[:, None]*amplitude(y)
        initial = torch.from_numpy(exact_temperature[0].copy())
        initial[fixed] = prescribed
        result = thermal.solve_transient_thermal(model, torch.from_numpy(times), initial,
                                                 method="sdirk2", sparse=True)
        temperature = result.temperature.numpy()
        local = thermal.thermal_element_nodal_heat_steps(model, result).numpy()
        cuts = np.column_stack((local[:, 0, 0], -local[:, :, 1]))
        face_metrics["max_shared_face_mismatch_J_m2"] = float(abs(local[:, :-1, 1]+local[:, 1:, 0]).max())
    elif solver == "FVM":
        temperature, _, _, _, heat = rod.solve_transient_stepped_rod(
            np.full(n, 1/n), 1., 1/capacity, capacity, tuple(0. if w is None else w for w in walls), times, exact_temperature[0],
            end_types=tuple("power" if w is None else "temperature" for w in walls),
            method="sdirk2", return_step_heat=True)
        cuts = np.column_stack((heat["left"][:, 0], heat["right"]))
    else:
        temperature, cuts = sph.solve_transient_layered_slab(
            exact_temperature[0], times, 1/n, tuple(walls), conductivity=1/capacity,
            volumetric_heat_capacity=capacity, method="exponential", h_factor=.9, return_x_heat=True)
        temperature, cuts = temperature.numpy(), cuts.numpy()
    heat_factor = .3*np.expm1(-wave**2*times[1:, None])/wave
    exact_heat = -heat_factor*np.sin(wave*y) if cosine else heat_factor*np.cos(wave*y)
    if not insulated and not mixed:
        exact_heat += times[1:, None]/2
    for face, wall in zip((0, -1), walls):
        if wall is None:
            exact_heat[:, face] = 0.
    error = abs(np.cumsum(cuts, axis=0)-exact_heat)
    cell_temperature = (temperature[:, 1:]+temperature[:, :-1])/2 if solver == "FEM" else temperature
    if solver == "DG":
        cell_temperature = temperature.mean(axis=2)
    if insulated:
        stored_change = (cell_temperature-cell_temperature[0]) @ width
        face_metrics["max_total_heat_drift_J_m2"] = float(abs(stored_change).max())
    balance = width*np.diff(cell_temperature, axis=0)+np.diff(cuts, axis=1)
    return dict(**face_metrics, solver=solver, cells=n, max_temperature_error_K=float(abs(temperature-exact_temperature).max()),
                max_cumulative_heat_error_J_m2=float(error.max()),
                west_east_cumulative_heat_errors_J_m2=error[:, [0, -1]].max(axis=0).tolist(),
                max_cell_balance_error_J_m2=float(abs(balance).max()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-fem", action="store_true", help="also verify LINE2 nodal temperatures and face heat")
    parser.add_argument("--with-dg", action="store_true", help="also verify SIPG center temperatures and face heat")
    boundaries = parser.add_mutually_exclusive_group()
    boundaries.add_argument("--insulated-end", choices=("west", "east"), help="insulate one end and hold the other at 0.5 K")
    boundaries.add_argument("--insulated", action="store_true", help="relax a cosine mode between insulated ends")
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
            row = measure(n, solver, args.insulated, args.insulated_end)
            errors = np.array([row["max_temperature_error_K"], row["max_cumulative_heat_error_J_m2"],
                               *row["west_east_cumulative_heat_errors_J_m2"]])
            if args.insulated:
                # The exact end heat is identically zero; test it absolutely.
                errors = errors[:2]
            insulated_error = 0.
            if args.insulated_end:
                insulated_index = 0 if args.insulated_end == "west" else 1
                insulated_error = errors[2+insulated_index]
                errors = np.delete(errors, 2+insulated_index)
                fixed_face = -1 if insulated_index == 0 else 0
                row["fixed_end_cumulative_heat_error_J_m2"] = row["west_east_cumulative_heat_errors_J_m2"][fixed_face]
            orders = [] if previous is None else np.log2(previous/errors).tolist()
            row.update(orders=orders, passed=bool(row["max_cell_balance_error_J_m2"] < 1e-11
                       and insulated_error < 1e-10
                       and row.get("max_total_heat_drift_J_m2", 0.) < 1e-10
                       and (not args.insulated or max(row["west_east_cumulative_heat_errors_J_m2"]) < 1e-10)
                       and row.get("max_2d_cell_balance_error_J_m2", 0.) < 1e-11
                       and row.get("max_shared_face_mismatch_J_m2", 0.) < 1e-11
                       and (not orders or min(orders) > 0.) and (n < 128 or min(orders) > 1.85)))
            runs.append(row)
            previous = errors
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(path) for path in sources],
                          reproduction=f"OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH={paths} python examples/graded_transient_slab.py"
                                       + (" --with-fem" if args.with_fem else "") + (" --with-dg" if args.with_dg else "")
                                       + (" --insulated" if args.insulated else "")
                                       + (f" --insulated-end {args.insulated_end}" if args.insulated_end else "")),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
