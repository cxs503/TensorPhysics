"""Relax a slab with two inward heat fluxes and a distributed linear sink."""
import json

import numpy as np
from tensorsph import heat_conduction_2d as sph

from conduction_provenance import provenance


def measure(n, method):
    k, capacity, loss = 1.5, 2., 4.
    west, east = 1.2, .8
    x = (np.arange(n)+.5)/n
    faces = np.arange(n+1)/n
    mu = np.sqrt(loss/k)
    steady = (west*np.cosh(mu*(1-x))+east*np.cosh(mu*x))/(k*mu*np.sinh(mu))
    steady_flux = (west*np.sinh(mu*(1-faces))-east*np.sinh(mu*faces))/np.sinh(mu)
    # Both perturbations have homogeneous Neumann data. The spatially
    # uniform perturbation alone determines the approach of the volume mean.
    perturbation = .15*np.cos(np.pi*x)
    rate = (loss+k*np.pi**2)/capacity
    times = np.linspace(0., 3., 8*n+1)
    dt = np.diff(times)
    initial = np.tile(steady-.4+perturbation, (4, 1))
    history, cuts = sph.solve_transient_slab_2d(
        initial, times, 1/n, (None, None), x_wall_inward_flux=(west, east),
        heat_loss_coefficient=loss, conductivity=k, volumetric_heat_capacity=capacity,
        method=method, return_x_heat=True)
    mean = history.numpy().mean(axis=(1, 2))
    equilibrium = (west+east)/loss
    exact_mean = equilibrium-.4*np.exp(-loss*times/capacity)
    exact_field = steady-.4*np.exp(-loss*times[:, None]/capacity)+np.exp(-rate*times[:, None])*perturbation
    decay_average = np.exp(-rate*times[:-1])*(-np.expm1(-rate*dt))/(rate*dt)
    exact_flux = steady_flux + decay_average[:, None]*(.15*k*np.pi*np.sin(np.pi*faces))
    flux = cuts.numpy()/dt[:, None]
    if method == "theta":
        stage_mean = (mean[:-1]+mean[1:])/2
    else:
        gamma = 1.-1./np.sqrt(2.)
        first = (capacity*mean[:-1]+gamma*dt*(west+east))/(capacity+gamma*dt*loss)
        stage_mean = (1.-gamma)*first+gamma*mean[1:]
    residual = capacity*np.diff(mean) + loss*dt*stage_mean - (cuts[:, 0]-cuts[:, -1]).numpy()
    return dict(columns=n, steps=len(dt), method=method, final_mean_K=float(mean[-1]),
                exact_final_mean_K=float(exact_mean[-1]), equilibrium_mean_K=equilibrium,
                max_temperature_error_K=float(abs(history.numpy()-exact_field[:, None, :]).max()),
                max_mean_error_K=float(abs(mean-exact_mean).max()),
                max_mean_rise_error_K=float(abs((mean-mean[0])-(exact_mean-exact_mean[0])).max()),
                max_interior_flux_error_W_m2=float(abs(flux[:, 1:-1]-exact_flux[:, 1:-1]).max()),
                max_energy_residual_J_m2=float(abs(residual).max()))


if __name__ == "__main__":
    runs = []
    for method in ("theta", "sdirk2"):
        previous = None
        for n in (8, 16, 32, 64):
            row = measure(n, method)
            errors = np.array([row[key] for key in ("max_temperature_error_K", "max_mean_error_K",
                                                   "max_mean_rise_error_K", "max_interior_flux_error_W_m2")])
            orders = [] if previous is None else np.log2(previous/errors).tolist()
            row.update(orders=orders, passed=bool(row["max_energy_residual_J_m2"] < 1e-11
                       and (not orders or min(orders) > 0.) and (n < 64 or min(orders) > 1.85)))
            runs.append(row)
            previous = errors
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(__file__), provenance(sph.__file__)],
                          reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src python examples/sph_flux_wall_bulk_loss.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
