"""Verify signed heat at both walls when a two-mode source switches off."""
import argparse
import json
import math

import numpy as np
import torch

from conduction_provenance import provenance


def exact_reversal_time(start, stop, rate):
    # After switch-off the two outward powers cancel at A1 exp(-s) = 2 A2 exp(-4s).
    duration = stop-start
    ratio = math.expm1(-4*duration)/(2*math.expm1(-duration))
    return (stop + math.log(ratio)/3)/rate


def reversal_bracket_index(power):
    outward = power >= 0
    crossings = np.flatnonzero(outward[:-1] != outward[1:])
    if not np.isfinite(power).all() or len(crossings) != 1 or outward[0]:
        raise ValueError("expected one sustained inward-to-outward reversal after switch-off")
    return crossings[0]


def reversal_time(grid, right_heat, start, stop, rate):
    """Interpolate signed step-average powers at step midpoints after switch-off."""
    mids = (grid[1:] + grid[:-1])/2
    after = grid[:-1] >= stop/rate
    mids = mids[after]
    power = right_heat[after]/np.diff(grid)[after]
    j = reversal_bracket_index(power)
    estimate = mids[j] - power[j]*(mids[j+1]-mids[j])/(power[j+1]-power[j])
    exact = exact_reversal_time(start, stop, rate)
    return dict(reversal_time=float(estimate), exact_reversal_time=exact,
                bracketing_midpoints=mids[j:j+2].tolist(),
                timing_error_over_output_interval=float(abs(estimate-exact)*16*rate))


def contact_chain_pulse(cells, knots, indices, start, stop, rate, amplitude):
    from scipy.optimize import brentq
    from tensordem.thermal import contact_chain

    widths = np.full(cells, 1/cells)
    links, capacity = contact_chain.cell_centered_slab_network(widths, 1., 1.)
    x = (np.arange(cells)+.5)/cells
    modes = np.arange(1, 3)
    cell_factors = np.sinc(modes/(2*cells))
    source_profile = np.sin(math.pi*x[:, None]*modes) @ cell_factors
    source = np.r_[0., amplitude*rate*widths*source_profile, 0.]
    states = [np.zeros(cells+2)]
    sections = []
    residuals = []
    for a, b in zip(knots[:-1], knots[1:]):
        power = source if start <= a < stop else np.zeros_like(source)
        dt = (b-a)/rate
        end, heat = contact_chain.constant_contact_interval(
            links, capacity, dt, states[-1], t_left=0., t_right=0., nodal_power=power)
        residuals.append((capacity*(end-states[-1])-dt*power)[1:-1] + np.diff(heat))
        states.append(end)
        sections.append(heat)
    states = np.array(states)
    storage = states[indices] @ capacity
    walls = np.add.reduceat(np.array(sections)[:, [0, -1]], indices[:-1], axis=0)*[-1, 1]
    balance_error = float(np.abs(residuals).max())
    first = np.searchsorted(knots, stop)
    left = first + reversal_bracket_index(states[first:, -2]/links[-1])
    bracket = knots[left:left+2]/rate

    def right_current(t):
        if t == bracket[0]:
            return states[left, -2]/links[-1]
        end, _ = contact_chain.constant_contact_interval(
            links, capacity, t-bracket[0], states[left], t_left=0., t_right=0.)
        return end[-2]/links[-1]

    estimate = brentq(right_current, *bracket, xtol=1e-14)
    exact = exact_reversal_time(start, stop, rate)
    # Half-cell wall links give the sine modes discrete rates 4 n^2 sin^2(k pi/2n).
    face_factors = 2*cells*np.sin(math.pi*modes/(2*cells))
    rates = face_factors**2
    wall_amplitudes = face_factors*cell_factors*(-np.expm1(-rates*(stop-start)/rate))/rates
    network_root = stop/rate + math.log(wall_amplitudes[1]/wall_amplitudes[0])/float(rates[1]-rates[0])
    timing = dict(reversal_time=estimate, exact_reversal_time=exact,
                  discrete_reversal_time=network_root, bracketing_state_times=bracket.tolist(),
                  discrete_timing_error_over_output_interval=abs(estimate-network_root)*16*rate,
                  timing_error_over_output_interval=abs(estimate-exact)*16*rate)
    return walls, storage, timing, balance_error


def measure(cells, width, solver="dg"):
    # Time is scaled by the decay rate of the first sine mode.
    rate = math.pi**2
    output = np.arange(19) / 16
    start, stop = 1.2 / 16, (1.2 + width) / 16
    knots = np.union1d(output, [start, stop])
    amplitude = 2.0**-44

    indices = np.searchsorted(knots, output)
    local_errors = {}
    timing = {}
    if solver not in ("dg", "dem"):
        subdivisions = cells//8
        grid = np.r_[np.concatenate([np.linspace(a, b, subdivisions, endpoint=False)
                                     for a, b in zip(knots, knots[1:])]), knots[-1]]/rate
        steps = indices*subdivisions
    if solver == "dg":
        from scipy.optimize import brentq
        from tensordg import heat_1d

        def source(x):
            return amplitude * rate * (torch.sin(math.pi*x) + torch.sin(2*math.pi*x))

        _, states, info = heat_1d.solve_heat_1d_wall_history(
            knots/rate, np.zeros((len(knots), 2)), cells, 2, backend="exact-time",
            wall_penalty=40., initial_temperature=torch.zeros_like, integrate_face_fluxes=True,
            interval_sources=[source if start <= t < stop else None for t in knots[:-1]])

        # Locate the bracket from post-pulse states, independently of the continuum root.
        first = np.searchsorted(knots, stop)
        right_power = np.array([float(heat_1d.sipg_heat_face_fluxes(state, wall_penalty=40.)[-1])
                                for state in states[first:]])
        left = first + reversal_bracket_index(right_power)
        bracket = knots[left:left+2]/rate

        def power_at(t):
            _, state, _ = heat_1d.solve_heat_1d_exact_time(
                cells, 2, t-bracket[0], wall_penalty=40., initial_temperature=states[left])
            return float(heat_1d.sipg_heat_face_fluxes(state, wall_penalty=40.)[-1])

        estimate = brentq(power_at, *bracket, xtol=1e-14)
        exact = exact_reversal_time(start, stop, rate)
        timing = dict(reversal_time=estimate, exact_reversal_time=exact,
                      bracketing_state_times=bracket.tolist(),
                      timing_error_over_output_interval=abs(estimate-exact)*16*rate)
        sections = np.add.reduceat(np.asarray(info["interval_face_heat"]), indices[:-1], axis=0)
        # Outward heat has opposite orientation to the solver's +x flux at the left wall.
        walls = sections[:, [0, -1]] * [-1, 1]
        storage = states[indices, :, 0].numpy().mean(axis=1)
        source_mean = 2/math.pi
    elif solver == "dem":
        walls, storage, timing, local_errors["max_step_cell_balance_error"] = contact_chain_pulse(
            cells, knots, indices, start, stop, rate, amplitude)
        source_mean = 2/math.pi
    elif solver.startswith("fem-"):
        from tensorfem import thermal

        x = torch.linspace(0., 1., cells+1, dtype=torch.float64)
        elements = torch.stack((torch.arange(cells), torch.arange(1, cells+1)), dim=1)
        model = thermal.ThermalModel(x[:, None], elements, 1., 1., 1.,
                                     torch.tensor([0, cells]), x.new_zeros(2))
        # Integrate each sine against its LINE2 hat, including the half hats at the walls.
        load = np.zeros(cells+1)
        for k in (1, 2):
            load[1:-1] += np.sin(k*math.pi*x.numpy()[1:-1])*np.sinc(k/(2*cells))**2/cells
            end_load = (1-np.sinc(k/cells))/(k*math.pi)
            load[0] += end_load
            load[-1] += (-1)**(k+1)*end_load
        active = np.diff(np.clip(grid*rate-start, 0, stop-start))
        nodal_heat = amplitude*active[:, None]*load
        result = thermal.solve_transient_thermal(
            model, torch.from_numpy(grid), torch.zeros_like(x), theta=.5,
            method=solver.removeprefix("fem-"), sparse=True, nodal_step_heat=torch.from_numpy(nodal_heat))
        reaction = result.nodal_reaction_heat_steps().numpy()
        right_heat = -reaction[:, -1]
        walls = -np.add.reduceat(reaction[:, [0, -1]], steps[:-1], axis=0)
        temperatures = result.temperature.numpy()[steps]
        storage = (.5*(temperatures[:, :-1] + temperatures[:, 1:])).mean(axis=1)
        source_mean = 2/math.pi
        local_errors["max_step_free_node_residual"] = float(abs(reaction[:, 1:-1]).max())
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
        right_heat = -ledger["end_input"][:, -1]
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
        right_heat = cuts.numpy()[:, -1]
        walls = np.add.reduceat(cuts.numpy()[:, [0, -1]], steps[:-1], axis=0)*[-1, 1]
        storage = field[steps].numpy().mean(axis=(1, 2))
        source_mean = float(profile.mean())
    if solver not in ("dg", "dem"):
        timing = reversal_time(grid, right_heat, start, stop, rate)
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
    return dict(**timing, **{key: value/scale for key, value in local_errors.items()},
                cells=cells, width_over_interval=width, left_error=float(errors[0]),
                right_error=float(errors[1]), balance_error=float(abs(balance).max()/scale),
                right_heat_over_scale=(walls[:, 1]/scale).tolist(),
                exact_right_heat_over_scale=(reference[:, 1]/scale).tolist(),
                signs_match=bool(np.array_equal(np.sign(walls), np.sign(reference))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=("dg", "dem", "theta", "sdirk2", "fvm-theta", "fvm-sdirk2", "fem-theta", "fem-sdirk2"), default="dg",
                        help="exact-time DG or DEM contact chain, SPH (theta/sdirk2), finite volume (fvm-*), or sparse LINE2 (fem-*)")
    solver = parser.parse_args().solver
    if solver == "dg":
        from tensordg import heat_1d as implementation
        path = "../TensorDG"
        settings = dict(degree=2, wall_penalty=40, interior_penalty=20)
        required_order, tolerance = 3.5, 1e-5
    elif solver == "dem":
        from tensordem.thermal import contact_chain as implementation
        path = "../TensorDEM/src"
        settings = dict(network="cell-centered unit slab, zero contact resistance", integration="exact interval")
        required_order, tolerance = 1.8, .003
    else:
        if solver.startswith("fvm-"):
            from tensorfvm.cht import rod as implementation
            path = "../TensorFVM/src"
        elif solver.startswith("fem-"):
            from tensorfem import thermal as implementation
            path = "../TensorFEM/src"
        else:
            from tensorsph import heat_conduction_2d as implementation
            path = "../TensorSPH/src"
        settings = dict(base_temperature=305., subdivisions_per_smooth_interval="cells/8")
        if solver.startswith("fem-"):
            settings.update(formulation="excess temperature relative to 305 K", sparse=True)
        required_order, tolerance = 1.8, .003
    runs = []
    for width in (.05, .5):
        previous = None
        for cells in (16, 32, 64):
            row = measure(cells, width, solver)
            orders = {} if previous is None else {
                key: math.log2(previous[key]/row[key]) for key in ("left_error", "right_error")}
            if solver in ("dg", "dem") and previous is not None:
                key = "timing_error_over_output_interval"
                orders[key] = math.log2(previous[key]/row[key])
            row["orders"] = orders
            row["passed"] = (row["signs_match"] and row["balance_error"] < 1e-8 and
                             (cells < 64 or (min(orders.values()) > required_order and
                                             max(row["left_error"], row["right_error"]) < tolerance)))
            if solver.startswith("fvm-"):
                row["passed"] &= max(row["max_step_cell_balance_error"],
                                     row["max_step_shared_face_mismatch"]) < 1e-10
            if solver == "dem":
                row["passed"] &= (row["max_step_cell_balance_error"] < 1e-10 and
                                  row["discrete_timing_error_over_output_interval"] < 1e-9)
            if solver.startswith("fem-"):
                row["passed"] &= row["max_step_free_node_residual"] < 1e-10
            if solver == "dg":
                row["passed"] &= row["timing_error_over_output_interval"] < 1e-5*(64/cells)**4
            else:
                row["passed"] &= row["timing_error_over_output_interval"] < .01*(64/cells)**2
            runs.append(row)
            previous = row
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, solver=solver, **settings,
                          normalization="maximum absolute analytic interval heat across both walls",
                          sources=[provenance(__file__), provenance(implementation.__file__)],
                          reproduction=f"OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH={path} python examples/dg_reversing_wall_pulse.py --solver {solver}"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
