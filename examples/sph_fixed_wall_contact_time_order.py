"""Temporal refinement of a driven layered SPH slab with prescribed end temperatures."""
import json

import numpy as np

from conduction_provenance import provenance
from tensorsph import heat_conduction_2d as sph


def measure(steps, theta):
    # Compact support spans just one neighboring x column in either direction.
    # On this uniform lattice each half-cell and the contact then carry the
    # same flux for a piecewise linear profile, isolating the time error.
    n = 12
    dx = 1/n
    x = (np.arange(n)+.5)*dx
    k, capacity = np.repeat([[1., 4.], [1., 2.]], n//2, axis=1)
    resistance = .32
    contacts = np.zeros(n-1)
    contacts[n//2-1] = resistance
    left, flux = .8-.3j, .65+.2j
    centers = left-flux*(np.minimum(x, .5)+np.maximum(x-.5, 0.)/4+(x>.5)*resistance)
    right = left-flux*(.5+resistance+.5/4)
    omega = 2*np.pi

    def phase(t):
        return np.exp(1j*omega*t)

    def source(t):
        return (1j*omega*capacity*centers*phase(t)).real

    s = np.linspace(0., 1., steps+1)
    times = s+.3*np.sin(2*np.pi*s)/(2*np.pi)
    history, heat = sph.solve_transient_layered_slab(
        centers.real, times, dx, (lambda t: (left*phase(t)).real, lambda t: (right*phase(t)).real),
        conductivity=k, volumetric_heat_capacity=capacity, heat_source=source,
        contact_resistance_x=contacts, h_factor=.9, theta=theta, return_x_heat=True)
    history, heat = history.numpy(), heat.numpy()
    dt = np.diff(times)[:, None]
    exact_t = (phase(times)[:, None]*centers).real
    # Exact analytic interval mean, evaluated about the midpoint.
    exact_q = (flux*phase((times[1:]+times[:-1])/2)).real*np.sinc(omega*np.diff(times)/(2*np.pi))
    q_error = abs(heat/dt-exact_q[:, None])
    loads = source(times[:, None])
    balance = (dx*capacity*np.diff(history, axis=0)+np.diff(heat, axis=1)
               - dx*dt*((1-theta)*loads[:-1]+theta*loads[1:]))
    return dict(steps=steps, theta=theta, temperature_error_K=float(abs(history-exact_t).max()),
                contact_flux_error_W_m2=float(q_error[:, n//2].max()),
                all_cut_flux_error_W_m2=float(q_error.max()),
                column_energy_residual_J_m2=float(abs(balance).max()))


if __name__ == "__main__":
    rows = []
    for theta, minimum_order in ((.5, 1.9), (1., .9)):
        previous = None
        for steps in (32, 64, 128, 256):
            row = measure(steps, theta)
            errors = np.array([row[k] for k in ("temperature_error_K", "contact_flux_error_W_m2",
                                                "all_cut_flux_error_W_m2")])
            orders = np.log2(previous/errors).tolist() if previous is not None else []
            row.update(temperature_contact_all_cut_orders=orders,
                       passed=bool(np.all(np.isfinite(errors)) and row["column_energy_residual_J_m2"] < 1e-12
                                   and all(order > minimum_order for order in orders)))
            rows.append(row)
            previous = errors
    print(json.dumps(dict(runs=rows, sources=[provenance(__file__), provenance(sph.__file__)]),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
