"""Run python examples/native_directional_contact_heat.py from TensorPhysics.

Compare native directional contact heat from finite-volume and finite-element
rods, a fixed DEM thermal chain and an SPH slab against the two-reversal continuum solution.
Pass --with-dg to include the unit-width SIPG strip.
"""
import argparse
import json

import numpy as np
from tensordem.thermal import contact_chain
from tensorfem import thermal
from tensorfvm.cht import rod
from tensorsph import heat_conduction_2d as sph

import contact_heat_double_reversal as double
import resistive_decay_interface_comparison as case
from conduction_provenance import provenance


def fvm(n, times, modes):
    k, initial, contacts = case.cell_data(n, modes, cell_averages=True)
    _, _, _, _, ledger, parts = rod.solve_transient_stepped_rod(
        np.full(n, 1/n), 1., k, k, [0., 0.], times, initial, contact_resistance=contacts,
        method="sdirk2", flux_order=2, return_step_heat=True, return_resistive_contact_heat=True)
    cut = n//2-1
    np.testing.assert_array_equal(parts["face_indices"], [cut])
    return parts["rightward"][:, 0], parts["leftward"][:, 0], ledger["contact"][:, cut]


def dem(n, times, modes):
    links, capacity, initial = double.dem_chain(n, modes)
    _, net, parts = contact_chain.transient_contact_temperatures(
        links, capacity, times, initial, t_left=0., t_right=0., theta=.5, allow_massless=True,
        return_link_heat=True, return_directional_link_heat=True)
    cut = n//2+1
    return parts["rightward"][:, cut], parts["leftward"][:, cut], net[:, cut]


def fem(n, times, modes):
    model, initial = case.fem_initial_problem(n, modes)
    time = initial.new_tensor(times)
    # Piecewise-linear integration of contact power matches the solver's
    # native net heat for Crank-Nicolson on each individual time interval.
    result = thermal.solve_transient_thermal(model, time, initial, theta=.5)
    forward, reverse = thermal.thermal_contact_pair_directional_heat(model, time, result.temperature)
    net = thermal.thermal_contact_pair_heat_steps(model, result)
    return forward[:, 0].numpy(), reverse[:, 0].numpy(), net[:, 0].numpy()


def particles(n, times, modes):
    k, initial, contacts = case.cell_data(n, modes, cell_averages=False)
    _, net, parts = sph.solve_transient_layered_slab(
        initial, times, 1/n, (0., 0.), conductivity=k, volumetric_heat_capacity=k,
        contact_resistance_x=contacts, method="exponential", return_x_heat=True,
        return_directional_x_heat=True)
    # Section heats are per area; the unit-area slab gives the same numeric value in joules.
    cut = n//2
    return parts["rightward"][:, cut].numpy(), parts["leftward"][:, cut].numpy(), net[:, cut].numpy()


def run(with_dg=False):
    roots, end, waves, weights, modes = double.reference_modes()

    def integral(t):
        return (-np.expm1(-np.asarray(t)[..., None]*waves**2)/waves**2) @ weights

    phases = np.diff(integral(np.r_[0., roots, end]))
    reference = np.array([phases[0]+phases[2], -phases[1]])
    rows = []
    solvers = [("TensorFVM_SDIRK2", fvm), ("TensorDEM_Crank_Nicolson", dem),
               ("TensorFEM_Crank_Nicolson", fem), ("TensorSPH_exponential", particles)]
    sources = [__file__, double.__file__, case.__file__, rod.__file__, contact_chain.__file__,
               thermal.__file__, sph.__file__]
    if with_dg:
        from functools import partial
        from tensordg import poisson_2d_bc, transient_diffusion_2d
        solvers.append(("TensorDG_SDIRK2", partial(double.dg, directional=True)))
        sources.extend([poisson_2d_bc.__file__, transient_diffusion_2d.__file__])
    for solver, solve in solvers:
        previous = None
        for n in (24, 48, 96, 192):
            times = np.linspace(0., end, 2*n+1)
            # Clip the analytic integration limits to the westward phase.
            # The solver receives no event times and must apportion each mixed interval.
            reverse_ref = -np.diff(integral(np.clip(times, *roots)))
            forward_ref = np.diff(integral(times))+reverse_ref
            forward, reverse, net = solve(n, times, modes)
            mixed = (forward > 0) & (reverse > 0)
            # A discrete reversal can shift into a neighboring output interval.
            totals = np.array([forward.sum(), reverse.sum()])
            total_error = abs(totals-reference)
            interval_error = [max(abs(forward-forward_ref)), max(abs(reverse-reverse_ref))]
            balance = max(abs(forward-reverse-net))
            passed = (balance < 1e-12 and mixed.sum() == 2 and min(forward) >= 0 and min(reverse) >= 0
                      and (previous is None or np.all(previous > 3*total_error))
                      and (n < 192 or (np.all(total_error < .005*reference) and max(interval_error) < 1e-5)))
            rows.append(dict(solver=solver, resolution=n, intervals=2*n, right_left_heat_J=totals.tolist(),
                             total_errors_J=total_error.tolist(), max_interval_errors_J=interval_error,
                             net_balance_J=balance, mixed_intervals=int(mixed.sum()), passed=bool(passed)))
            previous = total_error
    return dict(reference_right_left_heat_J=reference.tolist(), analytic_reversals_s=roots.tolist(),
                end_s=end, runs=rows, sources=[provenance(p) for p in sources])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-dg", action="store_true")
    report = run(with_dg=parser.parse_args().with_dg)
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in report["runs"]) else 1)
