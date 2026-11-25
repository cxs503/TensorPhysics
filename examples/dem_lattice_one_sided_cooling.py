"""Verify uniform internal heating with one insulated face and one cooled face."""
import json

import numpy as np
from tensordem.thermal import contact_chain

from conduction_provenance import provenance


def measure(n, cooled):
    h = 1 / n
    k, heating, film, ambient = 1.7, 4.0, 3.5, 290.0
    y = (np.arange(n) + 0.5) * h
    # Distance from the insulated face. Exact outward heat flux at the
    # cooled face equals the integral of the source through unit thickness.
    distance = y if cooled == "upper" else 1 - y
    surface = ambient + heating / film
    exact = surface + heating * (1 - distance**2) / (2*k)
    conductance = h / (1/film + h/(2*k))
    exchange = {f"{cooled}_exchange": (conductance, ambient)}
    field = contact_chain.lattice_network_temperatures(
        n + 1, n, 1/k, exact, exact, nodal_power=heating*h**2, **exchange
    )
    side = -1 if cooled == "upper" else 0
    flux = conductance * (field[:, side] - ambient) / h
    # The balance region comprises just the free columns. Heat reaching the
    # pinned end columns must be included even though the continuum x flux is zero.
    cooled_power = h * flux[1:-1].sum()
    end_power = k * ((field[1] - field[0]).sum() + (field[-2] - field[-1]).sum())
    generated = heating * (n - 1) * h
    return dict(intervals=n, cooled_face=cooled, generated_W=generated,
                cooled_face_outward_W=float(cooled_power), pinned_ends_outward_W=float(end_power),
                balance_error_W=float(abs(generated - cooled_power - end_power)),
                field_error_K=float(abs(field - exact).max()),
                cooled_flux_error_W_m2=float(abs(flux - heating).max()),
                surface_error_K=float(abs(ambient + flux/film - surface).max()))


if __name__ == "__main__":
    runs = []
    for cooled in ("upper", "lower"):
        previous = None
        for n in (16, 32, 64, 128):
            row = measure(n, cooled)
            metrics = ("field_error_K", "cooled_flux_error_W_m2", "surface_error_K")
            orders = {} if previous is None else {
                key: float(np.log2(previous[key] / row[key])) for key in metrics
            }
            row.update(orders=orders, passed=row["balance_error_W"] < 1e-10
                       and all(order > 1.8 for order in orders.values())
                       and (n < 128 or all(row[key] < 1e-4 for key in metrics)))
            runs.append(row)
            previous = row
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs,
                          sources=[provenance(__file__), provenance(contact_chain.__file__)],
                          reproduction="PYTHONPATH=../TensorDEM/src python examples/dem_lattice_one_sided_cooling.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
