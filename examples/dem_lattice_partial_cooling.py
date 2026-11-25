"""Steady unit square with an insulated bottom and a partially cooled top surface."""
import json

import numpy as np
from tensordem.thermal import contact_chain

from conduction_provenance import provenance


def measure(n, start):
    h = 1 / n
    k, film = 1.7, 3.5
    x = np.arange(n + 1) * h
    y = (np.arange(n) + 0.5) * h
    # T = 300 - f(x)y². The fourth power makes T and its first three
    # x derivatives continuous where the boundary condition switches.
    width = 1 - start
    ramp = np.maximum((x - start) / width, 0)
    f = ramp**4
    exact = 300 - f[:, None] * y**2
    source = k * (12 * ramp[:, None]**2 / width**2 * y**2 + 2 * f[:, None])
    exact_flux = 2 * k * f
    active = x > start
    ambient = np.where(active, 300 - f - exact_flux/film, 280)
    g = np.where(active, h / (1/film + h/(2*k)), 0)
    field = contact_chain.lattice_network_temperatures(
        n + 1, n, 1/k, exact[0], exact[-1], nodal_power=h**2 * source,
        upper_exchange=(g, ambient),
    )
    flux = g * (field[:, -1] - ambient) / h
    # Composite trapezoid covers the full unit face. The free-column energy
    # budget separately omits the pinned columns and counts heat entering them.
    top_power = h * (flux[0]/2 + flux[1:-1].sum() + flux[-1]/2)
    exact_power = 2 * k * width / 5
    end_power = k * ((field[1] - field[0]).sum() + (field[-2] - field[-1]).sum())
    generated = h**2 * source[1:-1].sum()
    return dict(intervals=n, cooling_starts_at_x=start, cooled_power_W=float(top_power),
                reference_cooled_power_W=exact_power, power_error_W=float(abs(top_power - exact_power)),
                field_error_K=float(abs(field - exact).max()),
                cooled_flux_error_W_m2=float(abs(flux[active] - exact_flux[active]).max()),
                insulated_flux_W_m2=float(abs(flux[~active]).max()),
                free_column_balance_error_W=float(abs(generated - end_power - h * flux[1:-1].sum())))


if __name__ == "__main__":
    runs = []
    for start in (0.5, 0.37):
        previous = None
        for n in (16, 32, 64, 128):
            row = measure(n, start)
            metrics = ("field_error_K", "cooled_flux_error_W_m2", "power_error_W")
            orders = {} if previous is None else {
                key: float(np.log2(previous[key] / row[key])) for key in metrics
            }
            row.update(orders=orders, passed=row["insulated_flux_W_m2"] == 0
                       and row["free_column_balance_error_W"] < 1e-10
                       and (n != 128 or min(orders.values()) > 1.8))
            runs.append(row)
            previous = row
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs,
                          sources=[provenance(__file__), provenance(contact_chain.__file__)],
                          reproduction="PYTHONPATH=../TensorDEM/src python examples/dem_lattice_partial_cooling.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
