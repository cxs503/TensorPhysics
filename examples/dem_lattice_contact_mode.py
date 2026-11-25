"""Transversely varying steady heat flow across a two-material contact in a unit square."""
import json

import numpy as np
from tensordem.thermal import contact_chain

from conduction_provenance import provenance


def measure(n, resistance):
    # Odd n places the contact halfway along an axial link at x = 1/2.
    h = 1 / n
    x = np.arange(n + 1) / n
    y = (np.arange(n) + 0.5) / n
    k_left, k_right = 0.8, 3.2
    k = np.where(x < 0.5, k_left, k_right)
    lateral = np.cos(np.pi * y)
    # T = 300 + A(x) cos(pi*y). The slope -1/k gives q_x = cos(pi*y)
    # on both sides. Across the interface A drops by the contact resistance.
    amplitude = np.where(x < 0.5, 2 - x / k_left,
                         2 - 0.5 / k_left - resistance - (x - 0.5) / k_right)
    exact = 300 + amplitude[:, None] * lateral
    source = (np.pi**2 * k * amplitude)[:, None] * lateral
    rx = (1 / k[:-1] + 1 / k[1:]) / 2
    contact = n // 2
    rx[contact] += resistance / h
    temperature = contact_chain.lattice_network_temperatures(
        n + 1, n, (np.broadcast_to(rx[:, None], (n, n)),
                   np.broadcast_to(1 / k[:, None], (n + 1, n - 1))),
        exact[0], exact[-1], nodal_power=h**2 * source,
    )
    flux = -np.diff(temperature, axis=0) / (h * rx[:, None])
    interface_flux = flux[contact]
    left_trace = temperature[contact] - h * interface_flux / (2 * k_left)
    right_trace = temperature[contact + 1] + h * interface_flux / (2 * k_right)
    left_amplitude = 2 - 0.5 / k_left
    return dict(intervals=n, contact_resistance_m2_K_W=resistance,
                field_error_K=float(np.max(abs(temperature - exact))),
                interface_flux_error_W_m2=float(np.max(abs(interface_flux - lateral))),
                left_trace_error_K=float(np.max(abs(left_trace - (300 + left_amplitude * lateral)))),
                right_trace_error_K=float(np.max(abs(right_trace - (300 + (left_amplitude - resistance) * lateral)))))


def main():
    runs = []
    metrics = ("field_error_K", "interface_flux_error_W_m2", "left_trace_error_K", "right_trace_error_K")
    for resistance in (0.0, 0.35):
        previous = None
        for n in (15, 31, 63, 127):
            row = measure(n, resistance)
            orders = {} if previous is None else {
                key: float(np.log(previous[key] / row[key]) / np.log(n / previous["intervals"]))
                for key in metrics
            }
            passed = all(order > 1.9 for order in orders.values())
            if n == 127:
                passed = passed and all(row[key] < 2e-4 for key in metrics)
            runs.append(dict(**row, orders=orders, passed=passed))
            previous = row
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(__file__), provenance(contact_chain.__file__)],
                          reproduction="PYTHONPATH=../TensorDEM/src python examples/dem_lattice_contact_mode.py"),
                     indent=2, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
