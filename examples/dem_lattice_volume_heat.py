"""Verify a unit-depth anisotropic square with sinusoidal internal heating/cooling."""
import json

import numpy as np
from tensordem.thermal import contact_chain

from conduction_provenance import provenance


def measure(n, periodic):
    h = 1 / n
    x = np.arange(n + 1)[:, None] * h
    y = (np.arange(n) + 0.5) * h
    kx, ky = 1.5, 0.6
    a = np.pi
    b = (2 if periodic else 1) * np.pi
    phase = 0.73 if periodic else 0.0
    transverse = np.cos(b * y + phase)
    mode = np.sin(a * x) * transverse
    # The continuum equation -div(K grad T) = source has T = 300 + mode.
    eigenvalue = kx * a**2 + ky * b**2
    power = h**2 * eigenvalue * mode
    temperature = contact_chain.lattice_network_temperatures(
        n + 1, n, (1 / kx, 1 / ky), 300, 300, nodal_power=power, periodic_y=periodic
    )
    discrete_eigenvalue = 4 * (kx * np.sin(a*h/2)**2 + ky * np.sin(b*h/2)**2) / h**2
    discrete = 300 + mode * eigenvalue / discrete_eigenvalue
    qx = -kx * np.diff(temperature, axis=0) / h
    exact_qx = -kx * a * np.cos(a * (x[:-1] + h/2)) * transverse
    if periodic:
        qy = -ky * (np.roll(temperature, -1, axis=1) - temperature) / h
        y_faces = y + h/2
        lateral_out = qy - np.roll(qy, 1, axis=1)
    else:
        qy = -ky * np.diff(temperature, axis=1) / h
        y_faces = y[:-1] + h/2
        lateral_out = np.diff(np.pad(qy, ((0, 0), (1, 1))), axis=1)
    exact_qy = ky * b * np.sin(a*x) * np.sin(b*y_faces + phase)
    imbalance = h * (np.diff(qx, axis=0) + lateral_out[1:-1]) - power[1:-1]
    return dict(intervals=n, periodic_y=periodic,
                temperature_error_K=float(np.max(abs(temperature - (300 + mode)))),
                x_flux_error_W_m2=float(np.max(abs(qx - exact_qx))),
                y_flux_error_W_m2=float(np.max(abs(qy - exact_qy))),
                discrete_error_K=float(np.max(abs(temperature - discrete))),
                balance_error_W=float(np.max(abs(imbalance))))


def main():
    runs = []
    for periodic in (False, True):
        previous = None
        for n in (16, 32, 64, 128):
            row = measure(n, periodic)
            errors = np.array([row[k] for k in ("temperature_error_K", "x_flux_error_W_m2", "y_flux_error_W_m2")])
            orders = [] if previous is None else np.log2(previous / errors).tolist()
            row.update(orders=orders, passed=row["discrete_error_K"] < 1e-11
                       and row["balance_error_W"] < 1e-11 and all(order > 1.9 for order in orders))
            runs.append(row)
            previous = errors
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(__file__), provenance(contact_chain.__file__)],
                          reproduction="PYTHONPATH=../TensorDEM/src python examples/dem_lattice_volume_heat.py"),
                     indent=2, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
