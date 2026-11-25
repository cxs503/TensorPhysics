"""Relax a slab with two inward heat fluxes and a distributed linear sink."""
import argparse
import json

import numpy as np

from conduction_provenance import provenance


def measure(n, method, solver="sph"):
    k, capacity, loss = 1.5, 2., 4.
    west, east = 1.2, .8
    faces = np.arange(n+1)/n
    x = faces if solver == "fem" else (np.arange(n)+.5)/n
    mu = np.sqrt(loss/k)
    steady = (west*np.cosh(mu*(1-x))+east*np.cosh(mu*x))/(k*mu*np.sinh(mu))
    steady_flux = (west*np.sinh(mu*(1-faces))-east*np.sinh(mu*faces))/np.sinh(mu)
    # Both perturbations have homogeneous Neumann data. The spatially
    # uniform perturbation alone determines the approach of the volume mean.
    perturbation = .15*np.cos(np.pi*x)
    rate = (loss+k*np.pi**2)/capacity
    times = np.linspace(0., 3., 8*n+1)
    dt = np.diff(times)
    initial = steady-.4+perturbation
    bulk_metrics = {}
    if solver == "dg":
        import torch
        from tensordg import poisson_face_powers_2d, solve_transient_diffusion_2d

        def initial_field(x, y):
            profile = (west*torch.cosh(mu*(1-x))+east*torch.cosh(mu*x))/(k*mu*np.sinh(mu))
            return profile-.4+.15*torch.cos(torch.pi*x)

        options = dict(bc=dict.fromkeys(("left", "right", "top", "bottom"), "neumann"),
                       neumann_flux={"left": west, "right": east}, conductivity=k)
        _, _, states, averages = solve_transient_diffusion_2d(
            initial_field, times, ncells=(n, 2), degree=1, capacity=capacity, reaction=loss,
            method=method, return_interval_means=True, **options)
        # For degree one the constant mode is both the cell mean and center value.
        history = states[..., 0, 0].numpy().transpose(0, 2, 1)
        cuts = np.stack([poisson_face_powers_2d(avg, **options)[0].sum(1).numpy()
                         for avg in averages])*dt[:, None]
        column_mean = history.mean(axis=1)
        sink = loss*dt[:, None]*averages[..., 0, 0].mean(dim=2).numpy()/n
        residual = capacity*np.diff(column_mean, axis=0)/n + sink + np.diff(cuts, axis=1)
        bulk_metrics["max_column_balance_error_J_m2"] = float(abs(residual).max())
        bulk_metrics["max_wall_heat_error_J_m2"] = float(abs(cuts[:, [0, -1]]-dt[:, None]*[west, -east]).max())
    elif solver == "fem":
        import torch
        from tensorfem import thermal

        nodes = torch.from_numpy(x)
        elements = torch.stack((torch.arange(n), torch.arange(1, n+1)), dim=1)
        loads = torch.zeros_like(nodes)
        loads[0], loads[-1] = west, east
        model = thermal.ThermalModel(nodes[:, None], elements, k, 1., capacity,
                                     elements[:0, 0], nodes[:0], nodal_heat=loads,
                                     heat_loss_coefficient=loss)
        result = thermal.solve_transient_thermal(model, torch.from_numpy(times), torch.from_numpy(initial),
                                                 theta=.5, sparse=True, method=method)
        history = result.temperature.numpy()[:, None, :]
        heat = thermal.thermal_element_nodal_heat_steps(model, result).numpy()
        cuts = np.column_stack((heat[:, 0, 0], -heat[:, :, 1]))
        bulk_metrics["max_wall_heat_error_J_m2"] = float(abs(cuts[:, [0, -1]]-dt[:, None]*[west, -east]).max())
        bulk_metrics["max_internal_face_mismatch_J_m2"] = float(abs(heat[:, :-1, 1]+heat[:, 1:, 0]).max())
    elif solver == "fvm":
        from tensorfvm.cht import rod

        cells, _, _, _, heat = rod.solve_transient_stepped_rod(
            np.full(n, 1/n), 1., k, capacity, (west, east), times, initial,
            end_types=("power", "power"), heat_loss_coefficient=loss,
            method=method, return_step_heat=True)
        history = cells[:, None, :]
        cuts = np.column_stack((heat["left"][:, 0], heat["right"]))
        residual_cells = capacity*np.diff(cells, axis=0)/n + np.diff(cuts, axis=1) - heat["bulk_exchange"]
        bulk_metrics["max_cell_balance_error_J_m2"] = float(abs(residual_cells).max())
    else:
        from tensorsph import heat_conduction_2d as sph

        history, cuts = sph.solve_transient_slab_2d(
            np.tile(initial, (4, 1)), times, 1/n, (None, None), x_wall_inward_flux=(west, east),
            heat_loss_coefficient=loss, conductivity=k, volumetric_heat_capacity=capacity,
            method=method, return_x_heat=True)
        history, cuts = history.numpy(), cuts.numpy()
    if solver == "fem":
        # Integrate the LINE2 interpolant exactly on each element.
        mean = ((history[:, 0, :-1]+history[:, 0, 1:])/2).mean(axis=1)
    else:
        mean = history.mean(axis=(1, 2))
    equilibrium = (west+east)/loss
    exact_mean = equilibrium-.4*np.exp(-loss*times/capacity)
    exact_field = steady-.4*np.exp(-loss*times[:, None]/capacity)+np.exp(-rate*times[:, None])*perturbation
    decay_average = np.exp(-rate*times[:-1])*(-np.expm1(-rate*dt))/(rate*dt)
    exact_flux = steady_flux + decay_average[:, None]*(.15*k*np.pi*np.sin(np.pi*faces))
    flux = cuts/dt[:, None]
    if method == "theta":
        stage_mean = (mean[:-1]+mean[1:])/2
    else:
        gamma = 1.-1./np.sqrt(2.)
        first = (capacity*mean[:-1]+gamma*dt*(west+east))/(capacity+gamma*dt*loss)
        stage_mean = (1.-gamma)*first+gamma*mean[1:]
    residual = capacity*np.diff(mean) + loss*dt*stage_mean - (cuts[:, 0]-cuts[:, -1])
    return dict(**bulk_metrics, columns=n, steps=len(dt), method=method, final_mean_K=float(mean[-1]),
                exact_final_mean_K=float(exact_mean[-1]), equilibrium_mean_K=equilibrium,
                max_temperature_error_K=float(abs(history-exact_field[:, None, :]).max()),
                max_mean_error_K=float(abs(mean-exact_mean).max()),
                max_mean_rise_error_K=float(abs((mean-mean[0])-(exact_mean-exact_mean[0])).max()),
                max_interior_flux_error_W_m2=float(abs(flux[:, 1:-1]-exact_flux[:, 1:-1]).max()),
                max_energy_residual_J_m2=float(abs(residual).max()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=("sph", "fvm", "fem", "dg"), default="sph")
    solver = parser.parse_args().solver
    if solver == "dg":
        from tensordg import transient_diffusion_2d as implementation
        from tensordg import poisson_2d_bc
        repo = "TensorDG"
    elif solver == "fem":
        from tensorfem import thermal as implementation
        repo = "TensorFEM"
    elif solver == "fvm":
        from tensorfvm.cht import rod as implementation
        repo = "TensorFVM"
    else:
        from tensorsph import heat_conduction_2d as implementation
        repo = "TensorSPH"
    source_files = [__file__, implementation.__file__]
    if solver == "dg":
        source_files.append(poisson_2d_bc.__file__)
    module_path = f"../{repo}" if solver == "dg" else f"../{repo}/src"
    runs = []
    for method in ("theta", "sdirk2"):
        previous = None
        for n in (8, 16, 32, 64):
            row = measure(n, method, solver)
            errors = np.array([row[key] for key in ("max_temperature_error_K", "max_mean_error_K",
                                                   "max_mean_rise_error_K", "max_interior_flux_error_W_m2")])
            orders = [] if previous is None else np.log2(previous/errors).tolist()
            row.update(orders=orders, passed=bool(row["max_energy_residual_J_m2"] < 1e-11
                       and row.get("max_column_balance_error_J_m2", 0.) < 1e-11
                       and row.get("max_cell_balance_error_J_m2", 0.) < 1e-11
                       and row.get("max_wall_heat_error_J_m2", 0.) < 1e-11
                       and row.get("max_internal_face_mismatch_J_m2", 0.) < 1e-11
                       and (not orders or min(orders) > 0.) and (n < 64 or min(orders) > 1.85)))
            runs.append(row)
            previous = errors
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(solver=solver, passed=passed, runs=runs, sources=[provenance(path) for path in source_files],
                          reproduction=f"OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH={module_path} python examples/sph_flux_wall_bulk_loss.py --solver {solver}"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
