"""Verify lumped interfacial storage against two analytic slab modes."""
import argparse
import json
import math

import numpy as np

from conduction_provenance import provenance
import dem_resistive_pulse as analytic
import dg_reversing_wall_pulse as pulse


def fvm_step_heat(n, resistance, capacity, method, h, k, sheet_shape, grid, start, stop):
    from tensorfvm.cht import rod

    dx = 1/n
    sheet = n//2
    x = (np.arange(n)+.5)*dx
    mode = np.sin(h*np.minimum(x, 1-x))*np.sinc(h*dx/(2*math.pi))
    mode += np.where(x < .5, np.sin(k*x), -np.sin(k*(1-x)))*np.sinc(k*dx/(2*math.pi))
    # A first-order FV cell with width d, k=d/R, and cv=C/d has precisely
    # the sheet capacity and two R/2 half-resistances. Its bookkeeping width
    # is excluded from the physical coordinates used for the continuum modes.
    widths = np.full(n+1, dx)
    conductivity = np.ones(n+1)
    conductivity[sheet] = dx/resistance
    cv = np.ones(n+1)
    cv[sheet] = capacity/dx
    generation = np.insert(mode, sheet, capacity*sheet_shape/dx)
    states, _, _, _, ledger = rod.solve_transient_stepped_rod(
        widths, 1., conductivity, cv, (0., 0.), grid, np.zeros(n+1), method=method,
        source=lambda t: generation if start <= t < stop else 0.,
        load_jump_times=(start, stop), flux_order=1, return_step_heat=True)
    return states[:, sheet], ledger["left"][:, sheet], ledger["right"][:, sheet], -ledger["end_input"][:, 1]


def fem_step_heat(n, resistance, capacity, method, h, k, sheet_shape, grid, start, stop):
    import torch
    from tensorfem import thermal

    # Three coincident degrees of freedom: left trace, sheet, right trace.
    middle = n//2
    sheet = middle+1
    x = np.insert(np.linspace(0., 1., n+1), middle, [.5, .5])
    nodes = torch.from_numpy(x)
    starts = np.r_[np.arange(middle), np.arange(middle+2, n+2)]
    elements = np.column_stack((starts, starts+1))
    lump = torch.zeros_like(nodes)
    lump[sheet] = capacity
    model = thermal.ThermalModel(nodes[:, None], torch.from_numpy(elements), 1., 1., 1.,
                                 torch.tensor([0, n+2]), nodes.new_zeros(2),
                                 contact_pairs=torch.tensor([[sheet-1, sheet], [sheet, sheet+1]]),
                                 contact_conductance=nodes.new_full((2,), 2/resistance),
                                 nodal_heat_capacity=lump)
    # Integrate each bulk source against its two local LINE2 shape functions.
    xi, weights = np.polynomial.legendre.leggauss(8)
    shapes = np.column_stack(((1-xi)/2, (1+xi)/2))
    points = x[elements] @ shapes.T
    folded = np.minimum(points, 1-points)
    source = np.sin(h*folded) + np.where(points < .5, 1., -1.)*np.sin(k*folded)
    load = np.zeros_like(x)
    np.add.at(load, elements, (source*weights) @ shapes/(2*n))
    load[sheet] = capacity*sheet_shape
    active = np.diff(np.clip(grid-start, 0, stop-start))
    result = thermal.solve_transient_thermal(
        model, torch.from_numpy(grid), torch.zeros_like(nodes), sparse=True, theta=.5, method=method,
        nodal_step_heat=torch.from_numpy(active[:, None]*load))
    traces = result.interval_temperature.numpy()[:, sheet-1:sheet+2]
    left_heat, right_heat = (-np.diff(traces, axis=1)*(2/resistance)*np.diff(grid)[:, None]).T
    return (result.temperature.numpy()[:, sheet], left_heat, right_heat,
            -result.nodal_reaction_heat_steps().numpy()[:, -1])


def measure(n, resistance, capacity, width, method, solver="fvm", time_refinement=None):
    h, k, sheet_shape = analytic.contact_modes(resistance, capacity)
    rates = np.square([h, k])
    duration = 1/(16*math.pi**2)
    start, stop = 1.2*duration, (1.2+width)*duration
    knots = np.union1d(duration*np.arange(19), [start, stop])
    if time_refinement is None:
        time_refinement = 4 if solver == "fem" else 1
    refinement = (n//4)*time_refinement
    grid = np.r_[np.concatenate([np.linspace(a, b, refinement, endpoint=False)
                                 for a, b in zip(knots, knots[1:])]), knots[-1]]
    integrate = fem_step_heat if solver == "fem" else fvm_step_heat
    sheet_temperature, left_steps, right_steps, wall_heat = integrate(
        n, resistance, capacity, method, h, k, sheet_shape, grid, start, stop)
    steps = refinement*np.arange(knots.size)
    left, right = [np.add.reduceat(values, steps[:-1]) for values in (left_steps, right_steps)]
    active = np.clip(knots-start, 0, stop-start)
    off = np.maximum(knots-stop, 0)
    amplitudes = -np.expm1(-active[:, None]*rates)*np.exp(-off[:, None]*rates)/rates
    integrals = (np.diff(active)[:, None]-np.diff(amplitudes, axis=0))/rates
    even_heat, odd_heat = (integrals*(-np.array([h, k])*np.cos(np.array([h, k])/2))).T
    reference = np.column_stack((odd_heat+even_heat, odd_heat-even_heat))
    errors = np.abs(np.column_stack((left, right))-reference).max(axis=0)/np.abs(reference).max(axis=0)
    stored = capacity*sheet_temperature[steps]
    expected = capacity*sheet_shape*amplitudes[:, 0]
    # Include pulse generation inside the sheet in the per-substep energy balance.
    residual = (capacity*np.diff(sheet_temperature)-capacity*sheet_shape*np.diff(np.clip(grid-start, 0, stop-start))
                -left_steps+right_steps)
    midpoint = .5*(grid[1:]+grid[:-1])
    power = wall_heat/np.diff(grid)
    first = np.searchsorted(grid, stop)
    j = first+pulse.reversal_bracket_index(power[first:])
    root = midpoint[j]-power[j]*(midpoint[j+1]-midpoint[j])/(power[j+1]-power[j])
    at_stop = np.array([h, k])*(-np.expm1(-rates*(stop-start)))/rates
    exact_root = stop+math.log(at_stop[1]/at_stop[0])/float(rates[1]-rates[0])
    return dict(cells_per_slab=n, method=method, resistance=resistance, capacity=capacity,
                width_over_interval=width, left_error=float(errors[0]), right_error=float(errors[1]),
                sheet_energy_error=float(abs(stored-expected).max()/abs(expected).max()),
                sheet_balance_error=float(abs(residual).max()/abs(expected).max()),
                measured_reversal_time=float(root), exact_reversal_time=exact_root,
                timing_error_over_interval=float(abs(root-exact_root)/duration))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=("fem", "fvm"), default="fvm")
    parser.add_argument("--time-refinement", type=int, choices=(1, 2, 4, 8),
                        help="multiplier on n/4 substeps per smooth interval (FEM defaults to 4, FVM to 1)")
    args = parser.parse_args()
    solver = args.solver
    factor = args.time_refinement or (4 if solver == "fem" else 1)
    if solver == "fem":
        from tensorfem import thermal as implementation
        path = "../TensorFEM/src"
    else:
        from tensorfvm.cht import rod as implementation
        path = "../TensorFVM/src"
    runs = []
    for method in ("theta", "sdirk2"):
        for resistance in (.2, 1.):
            for capacity in (.25, 1.):
                for width in (.05, .5):
                    previous = None
                    for n in (16, 32, 64):
                        row = measure(n, resistance, capacity, width, method, solver, factor)
                        keys = ("left_error", "right_error", "sheet_energy_error", "timing_error_over_interval")
                        orders = {} if previous is None else {k: math.log2(previous[k]/row[k]) for k in keys}
                        row.update(orders=orders, passed=row["sheet_balance_error"] < 1e-10 and
                                   (n < 64 or (min(orders.values()) > 1.7 and max(row[k] for k in keys) < .01)))
                        previous = row
                        runs.append(row)
    passed = all(r["passed"] for r in runs)
    print(json.dumps(dict(passed=passed, solver=solver, time_refinement=factor, runs=runs, sources=[provenance(f) for f in
                          (__file__, analytic.__file__, pulse.__file__, implementation.__file__)],
                          reproduction=f"OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH={path} python examples/fvm_resistive_sheet_pulse.py --solver {solver} --time-refinement {factor}"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
