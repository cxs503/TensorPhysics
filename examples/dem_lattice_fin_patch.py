"""An axially conducting fin loses heat through a finite film patch of unit width."""
import json

import numpy as np
from tensordem.thermal import contact_chain

from conduction_provenance import provenance


def measure(n, start, end):
    h = 1 / n
    x = np.arange(n + 1) * h
    axial_k_area, film_per_length = 1.7, 5.0
    ambient, left, right = 290., 311., 302.
    # Exact excess temperature inside the film is A*cosh(m*s) + B*sinh(m*s).
    # Outside it is linear. Continuity of temperature and axial current at
    # both ends gives the two wall equations for A and B.
    m = np.sqrt(film_per_length / axial_k_area)
    length = end - start
    c, s = np.cosh(m*length), np.sinh(m*length)
    a, b = np.linalg.solve([[1, -start*m],
                            [c + (1-end)*m*s, s + (1-end)*m*c]],
                           [left-ambient, right-ambient])
    along = np.clip(x - start, 0, length)
    excess = a*np.cosh(m*along) + b*np.sinh(m*along)
    excess += np.minimum(x-start, 0)*m*b + np.maximum(x-end, 0)*m*(a*s + b*c)
    reference_power = film_per_length * (a*s + b*(c-1)) / m

    # Intersect each particle's dual interval with the patch. The pinned end
    # particles represent half intervals when the patch reaches a wall.
    covered = np.maximum(0, np.minimum(x+h/2, end) - np.maximum(x-h/2, start))
    g = film_per_length * covered
    field = contact_chain.lattice_network_temperatures(
        n+1, 1, h/axial_k_area, left, right, upper_exchange=(g, ambient)
    )[:, 0]
    loss = g * (field-ambient)
    wall_input = axial_k_area/h * (field[0]-field[1] + field[-1]-field[-2])
    return dict(intervals=n, patch=[start, end], covered_length_error_m=float(abs(covered.sum()-length)),
                field_error_K=float(abs(field - (ambient+excess)).max()),
                reference_power_W=float(reference_power), cooling_power_W=float(loss.sum()),
                power_error_W=float(abs(loss.sum()-reference_power)),
                free_node_balance_error_W=float(abs(loss[1:-1].sum()-wall_input)))


if __name__ == "__main__":
    runs = []
    for start, end in ((0.25, 0.75), (0.23, 0.79), (0.37, 1.0)):
        for n in (16, 32, 64, 128, 256):
            row = measure(n, start, end)
            row["passed"] = (row["covered_length_error_m"] < 1e-14
                             and row["free_node_balance_error_W"] < 1e-10
                             and (n < 256 or (row["field_error_K"] < 2e-4 and row["power_error_W"] < 1e-3)))
            runs.append(row)
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(__file__), provenance(contact_chain.__file__)],
                          reproduction="PYTHONPATH=../TensorDEM/src python examples/dem_lattice_fin_patch.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
