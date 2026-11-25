"""Verify SPH, FVM, FEM or DEM interval heat at every cut under independent Dirichlet wall pulses."""
import argparse
import json

import numpy as np

from conduction_provenance import provenance


# (time, side, temperature increment). Both walls have returned to zero at 0.09 s.
EVENTS = ((.013, 0, 1.), (.028, 1, -.6), (.047, 0, -1.), (.09, 1, .6))
TIMES = np.array([0., .01, .04, .056, .09, .115, .15])


def reference(x, times, modes, cell_width=0.):
    """Continuum temperature [K] and cumulative rightward heat [J/m²], k=C=L=1."""
    wave = np.pi*np.arange(1, modes+1)
    # Average each sine over a centered cell; zero width samples its midpoint.
    mean_factor = np.sinc(wave*cell_width/(2*np.pi))
    temperature = np.zeros((len(times), len(x)))
    heat = np.zeros_like(temperature)
    for start, side, amplitude in EVENTS:
        # At the switching time the interior is continuous and the isolated
        # wall-flux singularity contributes zero instantaneous integrated heat.
        active = times > start
        distance = x if side == 0 else 1-x
        age = times[active, None]-start
        decay = np.exp(-age*wave**2)
        angle = wave[:, None]*distance
        temperature[active] += amplitude*(1-distance-(2*decay*mean_factor/wave)@np.sin(angle))
        # Sum the nondecaying cosine series analytically, including at the walls.
        integral = age + distance**2/2-distance+1/3-(2*decay/wave**2)@np.cos(angle)
        heat[active] += (1 if side == 0 else -1)*amplitude*integral
    return temperature, heat


def wall(t, side):
    return sum(change for start, end, change in EVENTS if side == end and t >= start)


def measure(n):
    from tensorsph import heat_conduction_2d as sph

    dx = 1/n
    centers = (np.arange(n)+.5)*dx
    cuts = np.arange(n+1)*dx
    history, heat, walls = sph.solve_transient_layered_slab(
        np.zeros(n), TIMES, dx, (lambda t: wall(t, 0), lambda t: wall(t, 1)),
        conductivity=np.ones(n), volumetric_heat_capacity=1., method="exponential",
        load_jump_times=[event[0] for event in EVENTS], return_x_heat=True, return_wall_temperatures=True)
    temperature, _ = reference(centers, TIMES, 64)
    _, cumulative = reference(cuts, TIMES, 64)
    error = abs(heat.numpy()-np.diff(cumulative, axis=0))
    balance = dx*np.diff(history.numpy(), axis=0)+np.diff(heat.numpy(), axis=1)
    expected_walls = [[wall(t, 0), wall(t, 1)] for t in TIMES]
    return dict(columns=n, temperature_error_K=float(abs(history.numpy()-temperature).max()),
                wall_heat_error_J_m2=float(error[:, [0, -1]].max()),
                interior_cut_heat_error_J_m2=float(error[:, 1:-1].max()),
                column_energy_residual_J_m2=float(abs(balance).max()),
                wall_trace_error_K=float(abs(walls.numpy()-expected_walls).max()))


def measure_fvm(n):
    from tensorfvm.cht import rod

    dx = 1/n
    # Refine time along with space and retain the original reporting windows.
    # Interior switches are passed to the solver as events, not output times.
    pieces = [np.linspace(a, b, int(np.ceil(2*(b-a)/dx**2))+1)[1:]
              for a, b in zip(TIMES[:-1], TIMES[1:])]
    grid = np.concatenate(([TIMES[0]], *pieces))
    indices = np.r_[0, np.cumsum([len(piece) for piece in pieces])]
    cells, faces, _, _, step = rod.solve_transient_stepped_rod(
        np.full(n, dx), 1., 1., 1., lambda t: [wall(t, 0), wall(t, 1)], grid, 0.,
        flux_order=2, method="rannacher", load_jump_times=[event[0] for event in EVENTS],
        return_step_heat=True)
    means, _ = reference((np.arange(n)+.5)*dx, TIMES, 64, cell_width=dx)
    _, cumulative = reference(np.arange(n+1)*dx, TIMES, 64)
    q = np.column_stack((step["left"][:, 0], step["right"]))
    intervals = np.add.reduceat(q, indices[:-1], axis=0)
    error = abs(intervals-np.diff(cumulative, axis=0))
    balance = dx*np.diff(cells, axis=0)-step["left"]+step["right"]
    expected_walls = [[wall(t, 0), wall(t, 1)] for t in grid]
    return dict(columns=n, time_steps=len(grid)-1,
                temperature_error_K=float(abs(cells[indices]-means).max()),
                wall_heat_error_J_m2=float(error[:, [0, -1]].max()),
                interior_cut_heat_error_J_m2=float(error[:, 1:-1].max()),
                column_energy_residual_J_m2=float(abs(balance).max()),
                interface_heat_mismatch_J_m2=float(abs(step["right"][:, :-1]-step["left"][:, 1:]).max()),
                wall_trace_error_K=float(abs(faces[:, [0, -1]]-expected_walls).max()))


def reference_temperature_integral(x):
    """Time integral of the continuum temperature at the material centers."""
    wave = np.pi*np.arange(1, 65)
    result = np.zeros((len(TIMES), len(x)))
    for start, side, amplitude in EVENTS:
        active = TIMES > start
        age = TIMES[active, None]-start
        d = x if side == 0 else 1-x
        # Twice the stationary sine series with coefficients 1/(m*pi)^3.
        polynomial = d*(1-d)*(2-d)/6
        transient = (2*np.exp(-age*wave**2)/wave**3)@np.sin(wave[:, None]*d)
        result[active] += amplitude*(age*(1-d)-polynomial+transient)
    return result


def measure_dem(n):
    from tensordem.thermal import contact_chain

    dx = 1/n
    centers = (np.arange(n)+.5)*dx
    links, stores = contact_chain.cell_centered_slab_network(np.full(n, dx), 1., 1.)
    schedule = np.union1d(TIMES, [event[0] for event in EVENTS])
    state = np.zeros(n+2)
    history = [state[1:-1]]
    indices = np.searchsorted(schedule, TIMES)
    heats, integrals = [], []
    balance = wall_error = 0.
    for a, b in zip(schedule[:-1], schedule[1:]):
        ends = [wall(a, side) for side in (0, 1)]
        new, heat, mean = contact_chain.constant_contact_interval(
            links, stores, b-a, state, t_left=ends[0], t_right=ends[1],
            return_interval_temperature=True)
        wall_error = max(wall_error, np.max(abs(new[[0, -1]]-ends)), np.max(abs(mean[[0, -1]]-ends)))
        balance = max(balance, np.max(abs(dx*(new-state)[1:-1]+np.diff(heat))))
        heats.append(heat)
        integrals.append((b-a)*mean[1:-1])
        history.append(new[1:-1])
        state = new
    temperature, _ = reference(centers, TIMES, 64)
    _, cumulative = reference(np.arange(n+1)*dx, TIMES, 64)
    heat_error = abs(np.add.reduceat(heats, indices[:-1])-np.diff(cumulative, axis=0))
    integral_error = np.add.reduceat(integrals, indices[:-1])-np.diff(reference_temperature_integral(centers), axis=0)
    return dict(columns=n, temperature_error_K=float(abs(np.asarray(history)[indices]-temperature).max()),
                interval_temperature_error_K=float(abs(integral_error/np.diff(TIMES)[:, None]).max()),
                wall_heat_error_J_m2=float(heat_error[:, [0, -1]].max()),
                interior_cut_heat_error_J_m2=float(heat_error[:, 1:-1].max()),
                column_energy_residual_J_m2=float(balance), wall_trace_error_K=float(wall_error))


def measure_fem(n):
    from dataclasses import replace
    import torch
    from tensorfem import thermal

    dx = 1/n
    x = torch.linspace(0., 1., n+1, dtype=torch.float64)
    ids = torch.arange(n)
    conn = torch.stack((ids, ids+1), dim=1)
    fixed = torch.tensor([0, n])
    model = thermal.ThermalModel(x[:, None], conn, 1., 1., 1., fixed, x.new_zeros(2))
    state = torch.zeros_like(x)
    schedule = np.union1d(TIMES, [event[0] for event in EVENTS])
    indices = np.searchsorted(schedule, TIMES)
    history, sections, reactions, impulses = [state], [], [], []
    wall_error = balance = mismatch = 0.
    for a, b in zip(schedule[:-1], schedule[1:]):
        target = x.new_tensor([wall(a, 0), wall(a, 1)])
        changed = not torch.equal(model.fixed_temperatures, target)
        local = x.new_zeros((n, 2))
        reaction = x.new_zeros(2)
        if changed:
            state, jump, local = thermal.apply_thermal_temperature_jump(
                model, state, target, sparse=True, return_element_impulse=True)
            reaction = jump[fixed].clone()
            impulses.append(dict(time_s=float(a), inward_heat_J=reaction.tolist()))
            model = replace(model, fixed_temperatures=target)
        # Jump impulses belong to the window beginning at the switch.
        # The stored endpoint histories use the left limit at reporting times.
        grid = np.linspace(a, b, int(np.ceil(2*(b-a)/dx**2))+1)
        stages = [(grid, .5)]
        if changed:
            stages = [([a, (a+grid[1])/2, grid[1]], 1.), (grid[1:], .5)]
        for times, theta in stages:
            if len(times) < 2:
                continue
            result = thermal.solve_transient_thermal(model, x.new_tensor(times), state, theta=theta, sparse=True)
            local += thermal.thermal_element_nodal_heat_steps(model, result).sum(0)
            reaction += result.nodal_reaction_heat_steps()[:, fixed].sum(0)
            wall_error = max(wall_error, float(abs(result.temperature[:, fixed]-target).max()))
            state = result.temperature[-1]
        stored = dx*(state-history[-1])[conn].mean(1)
        balance = max(balance, float(abs(stored-local.sum(1)).max()))
        mismatch = max(mismatch, float(abs(local[:-1, 1]+local[1:, 0]).max()),
                       float(abs(local[[0, -1], [0, 1]]-reaction).max()))
        sections.append(-local[:-1, 1].numpy())
        reactions.append(reaction.numpy()*[1., -1.])
        history.append(state)
    means, _ = reference((np.arange(n)+.5)*dx, TIMES, 64, cell_width=dx)
    _, cumulative = reference(np.arange(n+1)*dx, TIMES, 64)
    exact_heat = np.diff(cumulative, axis=0)
    cells = torch.stack(history)[indices][:, conn].mean(2).numpy()
    return dict(columns=n, impulses=impulses, temperature_error_K=float(abs(cells-means).max()),
                wall_heat_error_J_m2=float(abs(np.add.reduceat(reactions, indices[:-1])-exact_heat[:, [0,-1]]).max()),
                interior_cut_heat_error_J_m2=float(abs(np.add.reduceat(sections, indices[:-1])-exact_heat[:, 1:-1]).max()),
                column_energy_residual_J_m2=balance, interface_heat_mismatch_J_m2=mismatch, wall_trace_error_K=wall_error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fem", action="store_true", help="run the consistent-capacity LINE2 slab")
    group.add_argument("--dem", action="store_true", help="run the fixed thermal contact chain")
    group.add_argument("--fvm", action="store_true", help="run the finite-volume cell-mean comparison")
    args = parser.parse_args()
    rows = []
    previous = None
    if args.fem:
        from tensorfem import thermal as solver
        run = measure_fem
    elif args.dem:
        from tensordem.thermal import contact_chain as solver
        run = measure_dem
    elif args.fvm:
        from tensorfvm.cht import rod as solver
        run = measure_fvm
    else:
        from tensorsph import heat_conduction_2d as solver
        run = measure
    for n in (8, 16, 32, 64):
        row = run(n)
        errors = np.array([row[k] for k in ("temperature_error_K", "wall_heat_error_J_m2",
                                            "interior_cut_heat_error_J_m2")])
        if args.dem:
            errors = np.append(errors, row["interval_temperature_error_K"])
        orders = np.log2(previous/errors).tolist() if previous is not None else []
        row.update(orders=orders, passed=bool(np.all(np.isfinite(errors)) and
                   row["column_energy_residual_J_m2"] < 1e-12 and row["wall_trace_error_K"] < 1e-14
                   and row.get("interface_heat_mismatch_J_m2", 0.) < 1e-12
                   and all(order > 0 for order in orders) and (n < 64 or min(orders) > 1.8)))
        rows.append(row)
        previous = errors
    print(json.dumps(dict(runs=rows, sources=[provenance(__file__), provenance(solver.__file__)]),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
