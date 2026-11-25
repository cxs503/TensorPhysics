"""Verify signed heat at both walls when a two-mode source switches off."""
import json
import math

import numpy as np
import torch
from tensordg import heat_1d

from conduction_provenance import provenance


def measure(cells, width):
    # Time is scaled by the decay rate of the first sine mode.
    rate = math.pi**2
    output = np.arange(19) / 16
    start, stop = 1.2 / 16, (1.2 + width) / 16
    knots = np.union1d(output, [start, stop])
    amplitude = 2.0**-44

    def source(x):
        return amplitude * rate * (torch.sin(math.pi*x) + torch.sin(2*math.pi*x))

    _, states, info = heat_1d.solve_heat_1d_wall_history(
        knots/rate, np.zeros((len(knots), 2)), cells, 2, backend="exact-time",
        wall_penalty=40., initial_temperature=torch.zeros_like, integrate_face_fluxes=True,
        interval_sources=[source if start <= t < stop else None for t in knots[:-1]])
    indices = np.searchsorted(knots, output)
    sections = np.add.reduceat(np.asarray(info["interval_face_heat"]), indices[:-1], axis=0)
    # Outward heat has opposite orientation to the solver's +x flux at the left wall.
    walls = sections[:, [0, -1]] * [-1, 1]
    # Independently integrate -dT/dx at both walls by Gauss quadrature.
    points, weights = np.polynomial.legendre.leggauss(12)
    a, b = knots[:-1, None], knots[1:, None]
    t = (a+b)/2 + (b-a)*points/2
    on = np.clip(t-start, 0, stop-start)
    off = np.maximum(t-stop, 0)
    modes = []
    for k in (1, 2):
        response = -np.expm1(-k*k*on)*np.exp(-k*k*off)/(k*k)
        modes.append(math.pi*k*amplitude*(b-a).ravel()/(2*rate)*(response @ weights))
    reference = np.column_stack((modes[0]+modes[1], modes[0]-modes[1]))
    reference = np.add.reduceat(reference, indices[:-1], axis=0)
    scale = np.abs(reference).max()
    errors = np.abs(walls-reference).max(axis=0)/scale
    # The first mode integrates to 2/pi on the slab. The second integrates to zero.
    deposited = amplitude*2/math.pi*np.diff(np.clip(output-start, 0, stop-start))
    storage = states[indices, :, 0].numpy().mean(axis=1)
    balance = np.diff(storage) + walls.sum(axis=1) - deposited
    return dict(cells=cells, width_over_interval=width, left_error=float(errors[0]),
                right_error=float(errors[1]), balance_error=float(abs(balance).max()/scale),
                right_heat_over_scale=(walls[:, 1]/scale).tolist(),
                exact_right_heat_over_scale=(reference[:, 1]/scale).tolist(),
                signs_match=bool(np.array_equal(np.sign(walls), np.sign(reference))))


if __name__ == "__main__":
    runs = []
    for width in (.05, .5):
        previous = None
        for cells in (16, 32, 64):
            row = measure(cells, width)
            orders = {} if previous is None else {
                key: math.log2(previous[key]/row[key]) for key in ("left_error", "right_error")}
            row["orders"] = orders
            row["passed"] = (row["signs_match"] and row["balance_error"] < 1e-8 and
                             (cells < 64 or (min(orders.values()) > 3.5 and
                                             max(row["left_error"], row["right_error"]) < 1e-5)))
            runs.append(row)
            previous = row
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, degree=2, wall_penalty=40, interior_penalty=20,
                          normalization="maximum absolute analytic interval heat across both walls",
                          sources=[provenance(__file__), provenance(heat_1d.__file__)],
                          reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorDG python examples/dg_reversing_wall_pulse.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
