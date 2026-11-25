"""Time integration order on a fixed two-material contact chain with storing walls."""
import json

import numpy as np

from conduction_provenance import provenance
from tensordem.thermal import contact_chain


def measure(steps, theta):
    # Unit-area slab, phase convention Re(T exp(i omega t)). The spatial
    # temperature profile is linear within each material, so each center
    # temperature also represents the exact mean in its cell.
    widths = np.array([.1, .17, .23, .16, .09, .25])
    conductivity = np.repeat([1., 4.], 3)
    capacity = np.repeat([1., 2.], 3)
    resistance = .32
    films = np.array([1.8, 3.5])
    wall_capacity = np.array([.37, .81])
    links, stores = contact_chain.cell_centered_slab_network(
        widths, conductivity, capacity, films=films, wall_areal_heat_capacity=wall_capacity,
        contact_resistance=[0., 0., resistance, 0., 0.], resolve_contacts=True)
    x = np.cumsum(widths)-widths/2
    left, flux = .8-.3j, .65+.2j
    # Temperature drop is the imposed flux times the continuum resistance.
    def temperature(position, right_layer):
        return left-flux*(np.minimum(position, .5)
                          + np.maximum(position-.5, 0.)/4 + right_layer*resistance)

    centers = temperature(x, x > .5)
    walls = temperature(np.array([0., 1.]), np.array([False, True]))
    traces = temperature(.5, np.array([False, True]))
    omega = 2*np.pi
    ambient = walls + (flux*np.array([1., -1.]) + 1j*omega*wall_capacity*walls)/films
    exact_nodes = np.r_[ambient[0], walls[0], centers[:3], traces, centers[3:], walls[1], ambient[1]]
    # These powers follow from C dT/dt since the spatial heat flux is uniform.
    power = np.r_[0., 0., (capacity*widths*centers)[:3], 0., 0.,
                  (capacity*widths*centers)[3:], 0., 0.] * (1j*omega)

    s = np.linspace(0., 1., steps+1)
    times = s+.3*np.sin(2*np.pi*s)/(2*np.pi)
    phase = np.exp(1j*omega*times)
    exact = (phase[:, None]*exact_nodes).real
    source = (phase[:, None]*power).real
    history = contact_chain.transient_contact_temperatures(
        links, stores, times, exact[0], t_left=exact[:, 0], t_right=exact[:, -1],
        nodal_power=source, allow_massless=True, theta=theta)
    flow = -np.diff(history, axis=1)/links
    net = source[:, 1:-1]-np.diff(flow, axis=1)
    balance = (np.diff(history[:, 1:-1], axis=0)*stores[1:-1]
               - np.diff(times)[:, None]*((1-theta)*net[:-1]+theta*net[1:]))
    return dict(steps=steps, theta=theta, temperature_error_K=float(abs(history-exact).max()),
                contact_flux_error_W_m2=float(abs(flow[:, 5]-(phase*flux).real).max()),
                node_energy_residual_J=float(abs(balance).max()))


if __name__ == "__main__":
    rows = []
    for theta, minimum_order in ((.5, 1.9), (1., .9)):
        previous = None
        for steps in (32, 64, 128, 256):
            row = measure(steps, theta)
            errors = np.array([row["temperature_error_K"], row["contact_flux_error_W_m2"]])
            orders = np.log2(previous/errors).tolist() if previous is not None else []
            row.update(temperature_and_flux_orders=orders,
                       passed=bool(np.all(np.isfinite(errors)) and row["node_energy_residual_J"] < 1e-12
                                   and all(order > minimum_order for order in orders)))
            previous = errors
            rows.append(row)
    print(json.dumps(dict(runs=rows, sources=[provenance(__file__), provenance(contact_chain.__file__)]),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
