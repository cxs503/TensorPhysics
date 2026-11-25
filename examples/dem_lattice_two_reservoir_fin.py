"""Fin exchanging heat with a hot reservoir on the left and a cold reservoir on the right."""
import json

import numpy as np
from tensordem.thermal import contact_chain

from conduction_provenance import provenance


def reference(x, split, axial, films, reservoirs, walls):
    # On each segment the excess above ambient satisfies theta'' = m² theta.
    # Match the two Dirichlet solutions at the split by axial flux continuity.
    lengths = np.array([split, 1-split])
    m = np.sqrt(films / axial)
    z = m * lengths
    weights = m / np.tanh(z)
    interface = (weights @ reservoirs + (m / np.sinh(z)) @ (walls-reservoirs)) / weights.sum()
    temperature = np.empty_like(x)
    for side, mask in enumerate((x <= split, x > split)):
        distance = x[mask] if side == 0 else 1-x[mask]
        temperature[mask] = reservoirs[side] + (
            (walls[side]-reservoirs[side]) * np.sinh(m[side]*(lengths[side]-distance))
            + (interface-reservoirs[side]) * np.sinh(m[side]*distance)
        ) / np.sinh(z[side])
    # Analytic segment integrals of film*(T - ambient), positive out of the fin.
    powers = films * (walls + interface - 2*reservoirs) * np.tanh(z/2) / m
    return temperature, powers


def measure(n, split):
    h = 1 / n
    x = np.arange(n+1) * h
    axial = 1.7
    films = np.array([4., 7.])
    reservoirs = np.array([321., 283.])
    walls = np.array([305., 303.])
    exact, powers = reference(x, split, axial, films, reservoirs, walls)
    lo, hi = np.maximum(0, x-h/2), np.minimum(1, x+h/2)
    g_hot = films[0] * np.maximum(0, np.minimum(hi, split)-lo)
    g_cold = films[1] * np.maximum(0, hi-np.maximum(lo, split))
    temperature = contact_chain.lattice_network_temperatures(
        n+1, 1, h/axial, *walls, lower_exchange=(g_hot, reservoirs[0]),
        upper_exchange=(g_cold, reservoirs[1]),
    )[:, 0]
    hot = g_hot * (temperature - reservoirs[0])
    cold = g_cold * (temperature - reservoirs[1])
    wall_input = axial/h * ((temperature[0]-temperature[1]) + (temperature[-1]-temperature[-2]))
    return dict(intervals=n, split=split, field_error_K=float(np.max(abs(temperature-exact))),
                hot_outward_W=float(hot.sum()), cold_outward_W=float(cold.sum()),
                reference_hot_outward_W=float(powers[0]), reference_cold_outward_W=float(powers[1]),
                hot_error_W=float(abs(hot.sum()-powers[0])), cold_error_W=float(abs(cold.sum()-powers[1])),
                balance_error_W=float(abs(wall_input - (hot+cold)[1:-1].sum())))


if __name__ == "__main__":
    runs = []
    for split in (0.5, 0.37):
        for n in (16, 32, 64, 128, 256):
            row = measure(n, split)
            row["passed"] = (row["hot_outward_W"] < 0 < row["cold_outward_W"]
                             and row["balance_error_W"] < 1e-10
                             and (n < 256 or (row["field_error_K"] < 2e-4
                                              and max(row["hot_error_W"], row["cold_error_W"]) < 1e-3)))
            runs.append(row)
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(__file__), provenance(contact_chain.__file__)],
                          reproduction="PYTHONPATH=../TensorDEM/src python examples/dem_lattice_two_reservoir_fin.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
