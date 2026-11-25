"""Verify transient row-resolved SPH heat under traveling temperature and flux boundary loads."""
import argparse
import json

import numpy as np
from tensorsph import heat_conduction_2d as sph

from conduction_provenance import provenance


def measure(n, flux_wall=None):
    # Superpose a uniform oscillation and a traveling transverse wave.
    # Each solves k*(T_xx + T_yy) = c*T_t on the unit periodic strip.
    k, capacity, omega = 1.5, 2., 8.
    x = (np.arange(n)+.5)/n
    fields = np.zeros((n, n), dtype=complex)
    walls = np.zeros((2, n), dtype=complex)
    inward = np.zeros_like(walls)
    for wave, west, east in ((0., .3, -.2j), (2*np.pi, 1., .35+.2j)):
        mu = np.sqrt(wave**2+1j*omega*capacity/k)
        phase_y = np.exp(1j*wave*x)
        axial = (west*np.sinh(mu*(1-x))+east*np.sinh(mu*x))/np.sinh(mu)
        fields += phase_y[:, None]*axial
        walls += np.array([west, east])[:, None]*phase_y
        flux = k*mu*np.array([west*np.cosh(mu)-east, east*np.cosh(mu)-west])/np.sinh(mu)
        inward += flux[:, None]*phase_y

    times = np.linspace(0., 2*np.pi/omega, 2*n+1)
    def wall(amplitude):
        return lambda t: (amplitude*np.exp(1j*omega*t)).real

    temperatures = [wall(side) for side in walls]
    fluxes = [0., 0.]
    if flux_wall is not None:
        side = ("west", "east").index(flux_wall)
        temperatures[side] = None
        fluxes[side] = wall(inward[side])
    history, cuts, heat = sph.solve_transient_slab_2d(
        fields.real, times, 1/n, temperatures, x_wall_inward_flux=fluxes,
        conductivity=k, volumetric_heat_capacity=capacity, method="sdirk2",
        return_wall_heat_by_row=True, return_x_heat=True, wall_gradient_fit="interior")
    dt = np.diff(times)
    phase = np.exp(1j*omega*times)
    interval_phase = np.exp(1j*omega*(times[1:]+times[:-1])/2)*np.sinc(omega*dt/(2*np.pi))
    field_error = abs(history.numpy()-(phase[:, None, None]*fields).real).max()
    flux_error = abs(heat.numpy()/dt[:, None, None]-(interval_phase[:, None, None]*inward).real)
    balance = capacity*np.diff(history.numpy().mean(axis=(1, 2))) - heat.numpy().mean(axis=2).sum(axis=1)
    boundary = cuts.numpy()[:, [0, -1]]*[1., -1.]
    return dict(columns=n, steps=len(dt), max_temperature_error_K=float(field_error),
                max_west_east_row_flux_errors_W_m2=flux_error.max(axis=(0, 2)).tolist(),
                max_energy_residual_J_m2=float(abs(balance).max()),
                max_wall_cut_difference_J_m2=float(abs(heat.numpy().mean(axis=2)-boundary).max()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flux-wall", choices=("west", "east"), help="prescribe inward heat flux on this wall")
    flux_wall = parser.parse_args().flux_wall
    runs = []
    previous = None
    for n in (8, 16, 32, 64):
        row = measure(n, flux_wall)
        errors = np.array([row["max_temperature_error_K"], *row["max_west_east_row_flux_errors_W_m2"]])
        orders = [] if previous is None else np.log2(previous/errors).tolist()
        row.update(orders=orders, passed=bool(row["max_energy_residual_J_m2"] < 1e-11
                   and row["max_wall_cut_difference_J_m2"] < 1e-11 and (not orders or min(orders) > 0.)
                   and (n < 64 or min(orders) > 1.85)))
        runs.append(row)
        previous = errors
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(runs=runs, prescribed_flux_wall=flux_wall, passed=passed, sources=[provenance(__file__), provenance(sph.__file__)],
                          reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src python examples/sph_traveling_wall_temperature.py"
                                       + (f" --flux-wall {flux_wall}" if flux_wall else "")),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
