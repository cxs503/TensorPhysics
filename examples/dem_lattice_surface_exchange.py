"""Verify two-sided convection on a unit square of unit depth with independently prescribed temperature and heat flux."""
import argparse
import json

import numpy as np
from tensordem.thermal import contact_chain

from conduction_provenance import provenance


def measure(n, curved=False):
    h = 1 / n
    x = np.arange(n + 1) / n
    y = (np.arange(n) + 0.5) / n
    conductivity = 1.7
    # T = 300 + 3x - (2+x)y solves Laplace's equation and has varying normal flux.
    slope = -(2 + x)
    exact = 300 + 3*x[:, None] + slope[:, None]*y
    curvature = 1 + x if curved else np.zeros_like(x)
    exact += curvature[:, None] * y**2
    # The manufactured field is linear in x and quadratic in y, giving
    # volumetric power -k*d²T/dy². Each free particle represents area h².
    nodal_power = -2 * conductivity * curvature[:, None] * h**2
    faces = []
    for position, outward in ((0, -1), (1, 1)):
        surface = 300 + 3*x + slope*position + curvature*position**2
        flux = -outward * conductivity * (slope + 2*curvature*position)
        film = 3 + position + 2*x
        ambient = surface - flux / film
        # Series half-spacing conduction and convection over area h.
        conductance = h / (h / (2*conductivity) + 1/film)
        faces.append((conductance, ambient, film, surface, flux))
    field = contact_chain.lattice_network_temperatures(
        n + 1, n, 1/conductivity, exact[0], exact[-1],
        nodal_power=nodal_power, lower_exchange=faces[0][:2], upper_exchange=faces[1][:2],
    )
    measurements = []
    for side, (g, ambient, film, surface, flux) in zip((0, -1), faces):
        measured = g * (field[:, side] - ambient) / h
        # Trapezoidal integration along x, with half weight on each pinned end.
        power = h * (measured[0]/2 + measured[1:-1].sum() + measured[-1]/2)
        reference_power = float((flux[0] + flux[-1])/2)
        measurements.append(dict(reference_outward_power_W=reference_power, outward_power_W=float(power),
                                 power_error_W=float(abs(power - reference_power)),
                                 flux_error_W_m2=float(np.max(abs(measured - flux))),
                                 surface_temperature_error_K=float(np.max(abs(ambient + measured/film - surface)))))
    error = float(np.max(abs(field - exact)))
    return dict(intervals=n, field_error_K=error, lower=measurements[0], upper=measurements[1])


def errors(row):
    values = {"field_error_K": row["field_error_K"]}
    for side in ("lower", "upper"):
        for metric in ("surface_temperature_error_K", "power_error_W", "flux_error_W_m2"):
            values[f"{side}_{metric}"] = row[side][metric]
    return values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curved", action="store_true", help="measure second-order convergence with internal heat extraction")
    args = parser.parse_args()
    grids = (16, 32, 64, 128) if args.curved else (4, 15, 32, 63)
    runs = []
    previous = None
    for n in grids:
        row = measure(n, args.curved)
        current = errors(row)
        if args.curved:
            orders = {} if previous is None else {k: float(np.log2(previous[k] / v)) for k, v in current.items()}
            row["orders"] = orders
            row["passed"] = all(order > 1.8 for order in orders.values()) and (n < grids[-1] or max(current.values()) < 2e-4)
        else:
            row["passed"] = max(current.values()) < 1e-11
        runs.append(row)
        previous = current
    passed = all(run["passed"] for run in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(__file__), provenance(contact_chain.__file__)],
                          reproduction="PYTHONPATH=../TensorDEM/src python examples/dem_lattice_surface_exchange.py" + (" --curved" if args.curved else "")),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
