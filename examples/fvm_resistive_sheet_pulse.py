"""Verify the FVM rod's lumped interfacial storage against two analytic slab modes."""
import json
import math

import numpy as np
from tensorfvm.cht import rod

from conduction_provenance import provenance
import dem_resistive_pulse as analytic
import dg_reversing_wall_pulse as pulse


def measure(n, resistance, capacity, width, method):
    h, k, sheet_shape = analytic.contact_modes(resistance, capacity)
    rates = np.square([h, k])
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
    duration = 1/(16*math.pi**2)
    start, stop = 1.2*duration, (1.2+width)*duration
    knots = np.union1d(duration*np.arange(19), [start, stop])
    refinement = n//4
    grid = np.r_[np.concatenate([np.linspace(a, b, refinement, endpoint=False)
                                 for a, b in zip(knots, knots[1:])]), knots[-1]]
    states, _, _, _, ledger = rod.solve_transient_stepped_rod(
        widths, 1., conductivity, cv, (0., 0.), grid, np.zeros(n+1), method=method,
        source=lambda t: generation if start <= t < stop else 0.,
        load_jump_times=(start, stop), flux_order=1, return_step_heat=True)
    steps = refinement*np.arange(knots.size)
    left, right = [np.add.reduceat(ledger[side][:, sheet], steps[:-1]) for side in ("left", "right")]
    active = np.clip(knots-start, 0, stop-start)
    off = np.maximum(knots-stop, 0)
    amplitudes = -np.expm1(-active[:, None]*rates)*np.exp(-off[:, None]*rates)/rates
    integrals = (np.diff(active)[:, None]-np.diff(amplitudes, axis=0))/rates
    even_heat, odd_heat = (integrals*(-np.array([h, k])*np.cos(np.array([h, k])/2))).T
    reference = np.column_stack((odd_heat+even_heat, odd_heat-even_heat))
    errors = np.abs(np.column_stack((left, right))-reference).max(axis=0)/np.abs(reference).max(axis=0)
    stored = capacity*states[steps, sheet]
    expected = capacity*sheet_shape*amplitudes[:, 0]
    # Include pulse generation inside the sheet in the per-substep energy balance.
    residual = (capacity*np.diff(states[:, sheet])-capacity*sheet_shape*np.diff(np.clip(grid-start, 0, stop-start))
                -ledger["left"][:, sheet]+ledger["right"][:, sheet])
    midpoint = .5*(grid[1:]+grid[:-1])
    power = -ledger["end_input"][:, 1]/np.diff(grid)
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
    runs = []
    for method in ("theta", "sdirk2"):
        for resistance in (.2, 1.):
            for capacity in (.25, 1.):
                for width in (.05, .5):
                    previous = None
                    for n in (16, 32, 64):
                        row = measure(n, resistance, capacity, width, method)
                        keys = ("left_error", "right_error", "sheet_energy_error", "timing_error_over_interval")
                        orders = {} if previous is None else {k: math.log2(previous[k]/row[k]) for k in keys}
                        row.update(orders=orders, passed=row["sheet_balance_error"] < 1e-10 and
                                   (n < 64 or (min(orders.values()) > 1.7 and max(row[k] for k in keys) < .01)))
                        previous = row
                        runs.append(row)
    passed = all(r["passed"] for r in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(f) for f in
                          (__file__, analytic.__file__, pulse.__file__, rod.__file__)],
                          reproduction="OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=../TensorFVM/src python examples/fvm_resistive_sheet_pulse.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
