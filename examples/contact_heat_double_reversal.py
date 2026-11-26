"""Run python examples/contact_heat_double_reversal.py in the three-solver environment.

Three decaying contact eigenmodes drive two reversals on a shared output grid.
Positive interval heat denotes transfer from the west material to the east.
Pass --with-dg for the TensorDG SIPG strip, or --with-dem for the fixed TensorDEM
thermal chain. Both options may be used together. Add --alternating-steps for a 1:3 step ratio.
"""
import argparse
import json

import numpy as np
from scipy.optimize import brentq

import contact_heat_return_time as events
import resistive_decay_interface_comparison as case
from conduction_provenance import provenance


def detect_phases(times, heat, *, higher_order=False):
    widths = np.diff(times)
    flux = heat/widths
    cumulative = np.r_[0., np.cumsum(heat)]
    hits = np.flatnonzero(np.diff(np.signbit(flux)))
    if hits.size != 2 or flux[0] <= 0:
        raise ValueError("expected eastward, westward, then eastward transfer")
    if higher_order:
        # Interpolate cumulative heat on both sides of each turning point.
        # Scaling the local time avoids ill conditioning on fine grids.
        extrema = []
        for i in hits:
            start = max(0, min(i-1, len(times)-5))
            section = slice(start, start+5)
            fit = np.polynomial.Polynomial.fit(times[section], cumulative[section], 4)
            low, high = times[i], times[i+2]
            turn = brentq(fit.deriv(), low, high)
            extrema.append((turn, fit(turn)))
    else:
        extrema = [events.extremum(times, cumulative, widths, flux, i) for i in hits]
    turns, amounts = zip(*extrema)
    return np.asarray(turns), np.diff([0., *amounts, cumulative[-1]])


def dg(n, times, modes):
    from tensordg import poisson_2d_bc as poisson
    from tensordg import transient_diffusion_2d as diffusion
    import torch

    mid = n//2
    k = np.repeat([1., 3.], mid)[:, None]*np.ones((1, 2))
    resistance = np.zeros((n-1, 2))
    resistance[mid-1] = case.RESISTANCE
    options = dict(bc=dict(left="dirichlet", right="dirichlet", bottom="neumann", top="neumann"),
                   conductivity=k, contact_resistance={"x": resistance})

    def initial(x, y):
        return sum(a*torch.where(x < .5, torch.sin(w*x), -torch.sin(w*(1-x))/3)
                   for w, a in modes)

    _, _, _, means = diffusion.solve_transient_diffusion_2d(
        initial, times, ncells=(n, 2), degree=1, capacity=k, method="sdirk2",
        return_interval_means=True, **options)
    # Two transverse faces span the unit cross section. Use the integrator's
    # interval-average field in the same numerical face flux as the solve.
    powers = [poisson.poisson_face_powers_2d(field, **options)[0][mid].sum().item() for field in means]
    return means, np.diff(times)*powers


def dem_chain(n, modes):
    from tensordem.thermal import contact_chain

    # Unit area, cell heat stores and half-cell conduction resistances.
    # Resolve the contact as a separate link between two massless surfaces.
    k, averages, contacts = case.cell_data(n, modes, cell_averages=True)
    links, capacity = contact_chain.cell_centered_slab_network(
        np.full(n, 1/n), k, k, contact_resistance=contacts, resolve_contacts=True)
    initial = np.zeros_like(capacity)
    initial[capacity > 0] = averages
    return links, capacity, initial


def dem(n, times, modes):
    from tensordem.thermal import contact_chain

    links, capacity, initial = dem_chain(n, modes)
    states, heats = contact_chain.transient_contact_temperatures(
        links, capacity, times, initial, t_left=0., t_right=0., theta=.5,
        allow_massless=True, return_link_heat=True)
    cut = n//2+1
    stored = np.diff(states[:, 1:-1], axis=0)*capacity[1:-1]
    balance = float(np.max(abs(stored+np.diff(heats, axis=1))))
    return balance, heats[:, cut]


def reference_modes():
    exact_times = np.array([.005, .018])
    end = .053
    waves = np.array([case.WAVE]+[brentq(case.contact_condition, i*np.pi, (i+1)*np.pi)
                                  for i in (3, 5)])
    # Fix the slow-mode flux amplitude to one, then impose zero total flux
    # at each prescribed time. The solver receives only the initial field.
    decay = np.exp(-exact_times[:, None]*waves**2)
    flux_weights = np.r_[1., np.linalg.solve(decay[:, 1:], -decay[:, 0])]
    modes = list(zip(waves, flux_weights/(-waves*np.cos(waves/2))))
    return exact_times, end, waves, flux_weights, modes


def run(with_dg=False, with_dem=False, alternating_steps=False):
    exact_times, end, waves, flux_weights, modes = reference_modes()
    boundaries = np.r_[0., exact_times, end]
    exact_heat = np.diff((-np.expm1(-boundaries[:, None]*waves**2)/waves**2) @ flux_weights)
    rows = []
    solvers = [("TensorFEM", case.fem), ("TensorFVM", case.fvm), ("TensorSPH", case.particles)]
    sources = [__file__, events.__file__, case.__file__, case.thermal.__file__, case.rod.__file__, case.sph.__file__]
    if with_dg:
        from tensordg import poisson_2d_bc, transient_diffusion_2d
        sources.extend([poisson_2d_bc.__file__, transient_diffusion_2d.__file__])
        solvers.append(("TensorDG", dg))
    if with_dem:
        from tensordem.thermal import contact_chain
        sources.append(contact_chain.__file__)
        solvers.append(("TensorDEM", dem))
    for name, solve in solvers:
        previous = None
        for n in (24, 48, 96, 192):
            times = np.linspace(0., end, 2*n+1)
            if alternating_steps:
                # Keep every pair's endpoints fixed, but make the first step
                # half the uniform width and the second one and a half times it.
                times[1::2] = times[:-1:2]+.25*(times[2::2]-times[:-1:2])
            diagnostic, heat = solve(n, times, modes)
            turns, phases = detect_phases(times, heat, higher_order=alternating_steps)
            timing_error = abs(turns-exact_times)
            heat_error = abs(phases-exact_heat)
            errors = np.array([max(timing_error), max(heat_error)])
            passed = (np.all(phases*exact_heat > 0) and (previous is None or np.all(previous > 3*errors))
                      and (n < 192 or (errors[0] < 2e-5 and errors[1] < 2e-5)))
            rows.append(dict(solver=name, resolution=n, steps=2*n, reversal_times_s=turns.tolist(),
                             phase_heat_J_m2=phases.tolist(), time_errors_s=timing_error.tolist(),
                             heat_errors_J_m2=heat_error.tolist(), passed=bool(passed)))
            if alternating_steps:
                # Replay the event detector on exact interval heats to measure
                # interpolation error independently of the spatial discretization.
                cumulative = (-np.expm1(-times[:, None]*waves**2)/waves**2) @ flux_weights
                sampled_turns, sampled_heats = detect_phases(times, np.diff(cumulative), higher_order=True)
                rows[-1].update(min_step_s=np.diff(times).min(), max_step_s=np.diff(times).max(),
                                interpolation_time_errors_s=abs(sampled_turns-exact_times).tolist(),
                                interpolation_heat_errors_J_m2=abs(sampled_heats-exact_heat).tolist())
            if solve is dem:
                rows[-1]["max_node_balance_J"] = diagnostic
                rows[-1]["passed"] &= diagnostic < 1e-12
            previous = errors
    return dict(reference_reversals_s=exact_times.tolist(), reference_phase_heat_J_m2=exact_heat.tolist(),
                end_s=end, time_grid="alternating_1_to_3" if alternating_steps else "uniform",
                turn_interpolation_degree=4 if alternating_steps else 2,
                modes_wave_amplitude=modes, contact_resistance_m2K_W=case.RESISTANCE,
                runs=rows, sources=[provenance(p) for p in sources])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-dg", action="store_true")
    parser.add_argument("--with-dem", action="store_true")
    parser.add_argument("--alternating-steps", action="store_true")
    report = run(**vars(parser.parse_args()))
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in report["runs"]) else 1)
