"""Verify weak steady currents through a contact at prescribed energy-weighted mean temperature."""
import json

import numpy as np
from tensorfvm.cht import rod

from conduction_provenance import provenance


def measure(n, mean, current):
    h = 1/n
    x = (np.arange(n)+0.5)*h
    k = np.where(x < 0.5, 1.7, 3.2)
    capacity = np.where(x < 0.5, 2., 3.)
    resistance = 0.4
    index = n//2-1
    contact = np.zeros(n-1)
    contact[index] = resistance
    storage = np.zeros(n-1)
    storage[index] = 0.8
    cell, faces, left, right, traces = rod.solve_steady_stepped_rod(
        np.full(n, h), 1., k, capacity, [current, -current], end_types=("power", "power"),
        flux_order=2, contact_resistance=contact, mean_temperature=mean, return_traces=True,
        interface_heat_capacity=storage, interface_capacity_left_fraction=0.25,
        end_areal_heat_capacity=[0.3, 0.7],
    )
    # With unit area, integrate dT/dx=-Q/k from the left end and
    # subtract Q*R across the contact. Linear cell means are midpoint values.
    offset = -current*(np.minimum(x, 0.5)/1.7 + np.maximum(x-0.5, 0)/3.2
                       + resistance*(x > 0.5))
    upstream = -current*0.5/1.7
    downstream = upstream-current*resistance
    east = downstream-current*0.5/3.2
    total_capacity = h*capacity.sum()+1.8
    reference_level = mean - (h*(capacity @ offset) + 0.7*east + 0.2*upstream
                              + 0.6*downstream)/total_capacity
    observed_mean = (h*(capacity @ cell) + 0.3*faces[0] + 0.7*faces[-1]
                     + 0.2*traces[0][index] + 0.6*traces[1][index])/total_capacity
    return dict(cells=n, mean_K=mean, current_W=current,
                mean_error_K=float(abs(observed_mean-mean)),
                cell_error_K=float(abs(cell-(reference_level+offset)).max()),
                jump_error_K=float(abs(traces[0][index]-traces[1][index]-current*resistance)),
                current_error_W=float(max(abs(left-current).max(), abs(right-current).max())))


if __name__ == "__main__":
    runs = []
    for mean in (0., 305.):
        for current in (-1e-5, 1e-5):
            for n in (16, 64, 256):
                row = measure(n, mean, current)
                row["passed"] = (row["current_error_W"] < 1e-13 and
                                 max(row[key] for key in ("mean_error_K", "cell_error_K", "jump_error_K")) < 2e-12)
                runs.append(row)
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(__file__), provenance(rod.__file__)],
                          reproduction="PYTHONPATH=../TensorFVM/src python examples/fvm_mean_temperature_contact_current.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
