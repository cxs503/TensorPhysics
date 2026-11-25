"""Verify signed heat at both walls when a two-mode source switches off."""
import argparse
import json
import math

import numpy as np
import torch

from conduction_provenance import provenance


def measure(cells, width, solver="dg"):
    # Time is scaled by the decay rate of the first sine mode.
    rate = math.pi**2
    output = np.arange(19) / 16
    start, stop = 1.2 / 16, (1.2 + width) / 16
    knots = np.union1d(output, [start, stop])
    amplitude = 2.0**-44

    indices = np.searchsorted(knots, output)
    local_errors = {}
    if solver != "dg":
        subdivisions = cells//8
        grid = np.r_[np.concatenate([np.linspace(a, b, subdivisions, endpoint=False)
                                     for a, b in zip(knots, knots[1:])]), knots[-1]]/rate
        steps = indices*subdivisions
    if solver == "dg":
        from tensordg import heat_1d

        def source(x):
            return amplitude * rate * (torch.sin(math.pi*x) + torch.sin(2*math.pi*x))

        _, states, info = heat_1d.solve_heat_1d_wall_history(
            knots/rate, np.zeros((len(knots), 2)), cells, 2, backend="exact-time",
            wall_penalty=40., initial_temperature=torch.zeros_like, integrate_face_fluxes=True,
            interval_sources=[source if start <= t < stop else None for t in knots[:-1]])
        sections = np.add.reduceat(np.asarray(info["interval_face_heat"]), indices[:-1], axis=0)
        # Outward heat has opposite orientation to the solver's +x flux at the left wall.
        walls = sections[:, [0, -1]] * [-1, 1]
        storage = states[indices, :, 0].numpy().mean(axis=1)
        source_mean = 2/math.pi
    elif solver.startswith("fvm-"):
        from tensorfvm.cht import rod

        dx = 1/cells
        x = (np.arange(cells) + .5)*dx
        # Cell averages of the two sine modes give the exact deposited heat.
        profile = np.sin(math.pi*x)*np.sinc(dx/2) + np.sin(2*math.pi*x)*np.sinc(dx)
        field, _, _, _, ledger = rod.solve_transient_stepped_rod(
            np.full(cells, dx), 1., 1., 1., (305., 305.), grid, np.full(cells, 305.),
            temperature_reference=305., method=solver.removeprefix("fvm-"), flux_order=2,
            source=lambda t: amplitude*rate*profile if start/rate <= t < stop/rate else 0.,
            load_jump_times=(start/rate, stop/rate), return_step_heat=True)
        walls = -np.add.reduceat(ledger["end_input"], steps[:-1], axis=0)
        storage = field[steps].mean(axis=1)
        source_mean = 2/math.pi
        active = np.diff(np.clip(grid*rate-start, 0, stop-start))
        generated = dx*amplitude*active[:, None]*profile
        residual = dx*np.diff(field, axis=0) - generated + ledger["right"] - ledger["left"]
        local_errors["max_step_cell_balance_error"] = float(np.abs(residual).max())
        face_jump = ledger["right"][:, :-1] - ledger["left"][:, 1:]
        local_errors["max_step_shared_face_mismatch"] = float(np.abs(face_jump).max())
    else:
        from tensorsph import heat_conduction_2d as heat

        x = (torch.arange(cells, dtype=torch.float64) + .5)/cells
        profile = (torch.sin(math.pi*x) + torch.sin(2*math.pi*x)).expand(cells, -1)
        field, cuts = heat.solve_transient_slab_2d(
            torch.full_like(profile, 305.), grid, 1/cells, (305., 305.), temperature_reference=305.,
            method=solver, wall_jump_times=[start/rate, stop/rate], return_x_heat=True,
            heat_source=lambda t: amplitude*rate*profile if start/rate <= t < stop/rate else 0.)
        walls = np.add.reduceat(cuts.numpy()[:, [0, -1]], steps[:-1], axis=0)*[-1, 1]
        storage = field[steps].numpy().mean(axis=(1, 2))
        source_mean = float(profile.mean())
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
    scale = float(np.abs(reference).max())
    errors = np.abs(walls-reference).max(axis=0)/scale
    # Use the load integrated by each discretization for the energy ledger.
    deposited = amplitude*source_mean*np.diff(np.clip(output-start, 0, stop-start))
    balance = np.diff(storage) + walls.sum(axis=1) - deposited
    return dict(**{key: value/scale for key, value in local_errors.items()},
                cells=cells, width_over_interval=width, left_error=float(errors[0]),
                right_error=float(errors[1]), balance_error=float(abs(balance).max()/scale),
                right_heat_over_scale=(walls[:, 1]/scale).tolist(),
                exact_right_heat_over_scale=(reference[:, 1]/scale).tolist(),
                signs_match=bool(np.array_equal(np.sign(walls), np.sign(reference))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=("dg", "theta", "sdirk2", "fvm-theta", "fvm-sdirk2"), default="dg",
                        help="exact-time DG, SPH (theta/sdirk2), or finite volume (fvm-theta/fvm-sdirk2)")
    solver = parser.parse_args().solver
    if solver == "dg":
        from tensordg import heat_1d as implementation
        path = "../TensorDG"
        settings = dict(degree=2, wall_penalty=40, interior_penalty=20)
        required_order, tolerance = 3.5, 1e-5
    else:
        if solver.startswith("fvm-"):
            from tensorfvm.cht import rod as implementation
            path = "../TensorFVM/src"
        else:
            from tensorsph import heat_conduction_2d as implementation
            path = "../TensorSPH/src"
        settings = dict(base_temperature=305., subdivisions_per_smooth_interval="cells/8")
        required_order, tolerance = 1.8, .003
    runs = []
    for width in (.05, .5):
        previous = None
        for cells in (16, 32, 64):
            row = measure(cells, width, solver)
            orders = {} if previous is None else {
                key: math.log2(previous[key]/row[key]) for key in ("left_error", "right_error")}
            row["orders"] = orders
            row["passed"] = (row["signs_match"] and row["balance_error"] < 1e-8 and
                             (cells < 64 or (min(orders.values()) > required_order and
                                             max(row["left_error"], row["right_error"]) < tolerance)))
            if solver.startswith("fvm-"):
                row["passed"] &= max(row["max_step_cell_balance_error"],
                                     row["max_step_shared_face_mismatch"]) < 1e-10
            runs.append(row)
            previous = row
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, solver=solver, **settings,
                          normalization="maximum absolute analytic interval heat across both walls",
                          sources=[provenance(__file__), provenance(implementation.__file__)],
                          reproduction=f"OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH={path} python examples/dg_reversing_wall_pulse.py --solver {solver}"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
