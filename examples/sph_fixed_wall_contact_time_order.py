"""Temporal refinement of a driven layered SPH slab with prescribed end temperatures."""
import json

import numpy as np

from conduction_provenance import provenance
from tensorsph import heat_conduction_2d as sph


def measure(steps, theta, method="theta"):
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
        contact_resistance_x=contacts, h_factor=.9, theta=theta, method=method, return_x_heat=True)
    history, heat = history.numpy(), heat.numpy()
    dt = np.diff(times)[:, None]
    exact_t = (phase(times)[:, None]*centers).real
    # Exact analytic interval mean, evaluated about the midpoint.
    exact_q = (flux*phase((times[1:]+times[:-1])/2)).real*np.sinc(omega*np.diff(times)/(2*np.pi))
    q_error = abs(heat/dt-exact_q[:, None])
    # Integrate the volumetric generation with the method's stage weights
    # before comparing it with stored heat and the two bounding cuts.
    weight = 1-1/np.sqrt(2) if method == "sdirk2" else theta
    first_time = times[:-1, None] + (weight*dt if method == "sdirk2" else 0.)
    mean_load = (1-weight)*source(first_time)+weight*source(times[1:, None])
    if method == "exponential_quadratic":
        # Simpson's rule exactly integrates the interpolated quadratic source.
        mean_load = (mean_load+2*source(times[:-1, None]+dt/2))/3
    worst_interval, worst_cut = np.unravel_index(q_error.argmax(), q_error.shape)
    balance = (dx*capacity*np.diff(history, axis=0)+np.diff(heat, axis=1)
               - dx*dt*mean_load)
    return dict(steps=steps, method=method, theta=theta if method == "theta" else None, temperature_error_K=float(abs(history-exact_t).max()),
                contact_flux_error_W_m2=float(q_error[:, n//2].max()),
                all_cut_flux_error_W_m2=float(q_error.max()),
                worst_cut_x_m=float(worst_cut*dx), worst_interval_start_s=float(times[worst_interval]),
                column_energy_residual_J_m2=float(abs(balance).max()))


if __name__ == "__main__":
    rows = []
    # Linear interpolation has O(dt²) load error. A midpoint quadratic has
    # O(dt³) load error; endpoint temperatures can converge faster.
    schemes = (("theta", .5, 1.9, 256), ("theta", 1., .9, 256),
               ("sdirk2", .5, 1.9, 8192), ("exponential", .5, 1.9, 8192),
               ("exponential_quadratic", .5, 2.9, 2048))
    for method, theta, minimum_order, finest in schemes:
        previous = None
        refinements = [2**level for level in range(5, finest.bit_length())]
        for steps in refinements:
            row = measure(steps, theta, method)
            errors = np.array([row[k] for k in ("temperature_error_K", "contact_flux_error_W_m2",
                                                "all_cut_flux_error_W_m2")])
            orders = np.log2(previous/errors).tolist() if previous is not None else []
            # Retain the whole refinement sequence. Require decreasing
            # errors throughout and the design order on both finest refinements.
            require_order = previous is not None and (method == "theta" or steps in refinements[-2:])
            row.update(temperature_contact_all_cut_orders=orders, order_required=require_order,
                       passed=bool(np.all(np.isfinite(errors)) and row["column_energy_residual_J_m2"] < 1e-12
                                   and all(order > 0 for order in orders)
                                   and (not require_order or all(order > minimum_order for order in orders))))
            rows.append(row)
            previous = errors
    print(json.dumps(dict(runs=rows, sources=[provenance(__file__), provenance(sph.__file__)]),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
