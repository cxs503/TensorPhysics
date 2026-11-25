"""Verify DG cell temperatures and integrated face heats for a pulse inside one output interval."""
import argparse
import json
import math

import numpy as np
import torch
from tensordg import heat_1d

from conduction_provenance import provenance


def measure(n, degree, width, *, wall_penalty=None, return_section_heat=False, return_profiles=False):
    amplitude, rate = 2**-44, math.pi**2
    dt = 1/(16*rate)
    output = np.arange(19)*dt
    start, stop = 1.2*dt, (1.2+width)*dt
    knots = np.union1d(output, [start, stop])

    def heating(x):
        return amplitude*rate*torch.sin(math.pi*x)

    sources = [heating if start <= t < stop else None for t in knots[:-1]]
    centers, states, info = heat_1d.solve_heat_1d_wall_history(
        knots, np.zeros((len(knots), 2)), n, degree, backend="exact-time", wall_penalty=wall_penalty,
        initial_temperature=torch.zeros_like, interval_sources=sources, integrate_face_fluxes=True,
        return_interval_temperatures=True)
    ids = np.searchsorted(knots, output)
    cell = states[ids, :, 0].numpy()
    sections = np.add.reduceat(np.asarray(info["interval_face_heat"]), ids[:-1], axis=0)
    duration = np.clip(output-start, 0, stop-start)
    factor = -np.expm1(-rate*duration)*np.exp(-rate*np.maximum(output-stop, 0))
    integral = np.diff(duration)-np.diff(factor)/rate
    peak = -math.expm1(-rate*(stop-start))
    cell_mode = np.sin(math.pi*centers.numpy())*np.sinc(0.5/n)
    exact_cell = amplitude*factor[:, None]*cell_mode
    exact_sections = -math.pi*amplitude*integral[:, None]*np.cos(math.pi*np.arange(n+1)/n)
    modes = np.asarray(info["interval_temperature_modes"])
    means = modes[:, :, 0]
    averaged = np.add.reduceat(means*np.diff(knots)[:, None], ids[:-1], axis=0)/np.diff(output)[:, None]
    exact_mean = amplitude*(integral/np.diff(output))[:, None]*cell_mode
    deposited = amplitude*rate*np.diff(duration)[:, None]*cell_mode/n
    residual = np.diff(cell, axis=0)/n - deposited + np.diff(sections, axis=1)
    scale = math.pi*amplitude*dt*peak
    face_error = abs(sections-exact_sections)/scale
    wall_error = face_error[:, [0, -1]].max()
    interior_error = face_error[:, 1:-1].max()
    metrics = dict(cells=n, degree=degree, width_over_interval=width,
                max_interval_mean_error_over_peak=float(abs(averaged-exact_mean).max()/(amplitude*peak)),
                max_cell_mean_error_over_peak=float(abs(cell-exact_cell).max()/(amplitude*peak)),
                max_section_heat_error_over_scale=float(face_error.max()),
                max_wall_heat_error_over_scale=float(wall_error),
                max_interior_face_heat_error_over_scale=float(interior_error),
                crossing_section_error_over_scale=float(face_error[1].max()),
                max_cell_balance_error_over_scale=float(abs(residual).max()/scale))
    if return_profiles:
        # Evaluate the Legendre expansion at local coordinate zero. For P2
        # the center differs from the zeroth (cell-average) coefficient.
        basis = np.polynomial.legendre.legvander([0.], degree)[0]
        center_integrals = (modes @ basis)*np.diff(knots)[:, None]
        center_mean = np.add.reduceat(center_integrals, ids[:-1], axis=0)/np.diff(output)[:, None]
        exact_center = (integral/np.diff(output))[:, None]*np.sin(math.pi*centers.numpy())
        metrics["center_interval_mean_error_over_peak"] = float(abs(center_mean/amplitude-exact_center).max()/peak)
        return metrics, center_mean/(amplitude*peak), sections/scale
    if return_section_heat:
        return metrics, sections/scale
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wall-penalty", type=float, default=20.,
                        help="Dirichlet penalty coefficient (interior coefficient is 20)")
    args = parser.parse_args()
    runs = []
    for degree in (1, 2):
        minimum_order = 1.7 if degree == 1 else 3.5
        error_limit = 0.004 if degree == 1 else 1e-6
        for width in (0.05, 0.5):
            previous = None
            for n in (8, 16, 32, 64):
                row = measure(n, degree, width, wall_penalty=args.wall_penalty)
                keys = ("max_interval_mean_error_over_peak", "max_cell_mean_error_over_peak", "max_section_heat_error_over_scale",
                        "max_wall_heat_error_over_scale", "max_interior_face_heat_error_over_scale",
                        "crossing_section_error_over_scale")
                orders = {} if previous is None else {k: float(np.log2(previous[k]/row[k])) for k in keys}
                row.update(orders=orders, passed=row["max_cell_balance_error_over_scale"] < 1e-8
                           and (n < 64 or (max(row[k] for k in keys) < error_limit and min(orders.values()) > minimum_order)))
                runs.append(row)
                previous = row
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, wall_penalty=args.wall_penalty, interior_penalty=20.,
                          heat_scale="pi * peak continuum center temperature * output interval duration", runs=runs, sources=[provenance(__file__), provenance(heat_1d.__file__)],
                          reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorDG python examples/dg_subinterval_heat_pulse.py"
                                       + f" --wall-penalty {args.wall_penalty:g}"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
