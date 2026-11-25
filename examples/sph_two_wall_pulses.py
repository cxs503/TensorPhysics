"""Verify SPH, FVM, FEM, DG, LBM or DEM interval heat at every cut under independent Dirichlet wall pulses."""
import argparse
import json
from functools import partial

import numpy as np

from conduction_provenance import provenance


# (time, side, temperature increment). Both walls have returned to zero at 0.09 s.
EVENTS = ((.013, 0, 1.), (.028, 1, -.6), (.047, 0, -1.), (.09, 1, .6))
RAMP_KNOTS = np.array([t+.001/3 for t, _, _ in EVENTS])
RAMP_LEVELS = np.array([[0., 0.], [1., -.2], [.25, -.6], [0., 0.]])
TIMES = np.array([0., .01, .04, .056, .09, .115, .15])


def reference(x, times, modes, cell_width=0., events=EVENTS):
    """Continuum temperature [K] and cumulative rightward heat [J/m²], k=C=L=1."""
    wave = np.pi*np.arange(1, modes+1)
    # Average each sine over a centered cell; zero width samples its midpoint.
    mean_factor = np.sinc(wave*cell_width/(2*np.pi))
    temperature = np.zeros((len(times), len(x)))
    heat = np.zeros_like(temperature)
    for start, side, amplitude in events:
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


def measure_dg(n, ramps=False, degree=1):
    from tensordg import heat_1d

    dx = 1/n
    centers = (np.arange(n)+.5)*dx
    cuts = np.arange(n+1)*dx
    if ramps:
        schedule = np.union1d(TIMES, RAMP_KNOTS)
        endpoints = np.column_stack([np.interp(schedule, RAMP_KNOTS, side) for side in RAMP_LEVELS.T])
        means, _ = ramp_reference(centers, TIMES, RAMP_KNOTS, RAMP_LEVELS, cell_width=dx)
        _, cumulative = ramp_reference(cuts, TIMES, RAMP_KNOTS, RAMP_LEVELS)
    else:
        schedule = np.union1d(TIMES, [event[0] for event in EVENTS])
        plateaus = [[wall(t, 0), wall(t, 1)] for t in schedule[:-1]]
        endpoints = np.repeat(np.asarray(plateaus)[:, None, :], 2, axis=1)
        means, _ = reference(centers, TIMES, 64, cell_width=dx)
        _, cumulative = reference(cuts, TIMES, 64)
    indices = np.searchsorted(schedule, TIMES)
    _, history, meta = heat_1d.solve_heat_1d_wall_history(
        schedule, endpoints, n, degree=degree, backend="exact-time", initial_temperature=lambda x: 0*x,
        integrate_face_fluxes=True)
    cells = history[:, :, 0].numpy()
    heat = np.asarray(meta["interval_face_heat"])
    error = abs(np.add.reduceat(heat, indices[:-1])-np.diff(cumulative, axis=0))
    balance = dx*np.diff(cells, axis=0)+np.diff(heat, axis=1)
    return dict(columns=n, degree=degree, temperature_error_K=float(abs(cells[indices]-means).max()),
                wall_heat_error_J_m2=float(error[:, [0, -1]].max()),
                interior_cut_heat_error_J_m2=float(error[:, 1:-1].max()),
                column_energy_residual_J_m2=float(abs(balance).max()))


def ramp_reference(x, times, knots, levels, cell_width=0.):
    """Superpose exact ramp responses using slope changes at the wall knots."""
    slopes = np.diff(levels, axis=0)/np.diff(knots)[:, None]
    changes = np.diff(np.vstack(([0., 0.], slopes, [0., 0.])), axis=0)
    temperature = np.zeros((len(times), len(x)))
    heat = np.zeros_like(temperature)
    wave = np.pi*np.arange(1, 65)
    sinc = np.sinc(wave*cell_width/(2*np.pi))
    for knot, rates in zip(knots, changes):
        active = times > knot
        age = times[active, None]-knot
        decay = np.exp(-age*wave**2)
        for side, rate in enumerate(rates):
            d = x if side == 0 else 1-x
            # Integrated step response. Average its cubic spatial term over
            # the cell exactly; only the temperature is spatially averaged.
            cubic = d*(1-d)*(2-d)/6 + (d-1)*cell_width**2/24
            temperature[active] += rate*(age*(1-d)-cubic
                +(2*decay*sinc/wave**3)@np.sin(wave[:, None]*d))
            quartic = 1/45-d**2/6+d**3/6-d**4/24
            integral = (age**2/2+age*(d**2/2-d+1/3)-quartic
                        +(2*decay/wave**4)@np.cos(wave[:, None]*d))
            heat[active] += (1 if side == 0 else -1)*rate*integral
    return temperature, heat


def measure_lbm(n, off_grid=False, ramps=False):
    import torch
    from tensorlbm import scalar

    # A single periodic transverse row represents a transversely uniform slab.
    # Fix dt/dx² so that physical diffusivity is one at the same lattice tau.
    dx = 1/n
    dt = .064*dx**2
    tau = scalar.tau_from_diffusivity(dt/dx**2)
    ticks = np.rint(TIMES/dt).astype(int)
    if not np.allclose(ticks*dt, TIMES, rtol=0, atol=1e-14):
        raise ValueError("output times must lie on lattice steps")
    # Shifting by one third of the coarsest step puts all four switches
    # one third of a step past a lattice time on every refinement.
    shift = .001/3 if off_grid or ramps else 0.
    events = [(t+shift, side, change) for t, side, change in EVENTS]
    levels = np.zeros((len(events)+1, 2))
    for i, (_, side, change) in enumerate(events):
        levels[i+1] = levels[i]
        levels[i+1, side] += change
    switches = [event[0] for event in events]
    edges = np.arange(ticks[-1]+1)*dt
    centers = (np.arange(n)+.5)*dx
    cuts = np.arange(n+1)*dx
    if ramps:
        knots, values = RAMP_KNOTS, RAMP_LEVELS
        prescribed = scalar.scalar_wall_step_means(edges, knots, values, interpolation="linear")
        means, _ = ramp_reference(centers, TIMES, knots, values, cell_width=dx)
        _, cumulative = ramp_reference(cuts, TIMES, knots, values)
    else:
        prescribed = scalar.scalar_wall_step_means(edges, switches, levels)
        means, _ = reference(centers, TIMES, 64, cell_width=dx, events=events)
        _, cumulative = reference(cuts, TIMES, 64, events=events)
    zero = torch.zeros((1, n), dtype=torch.float64)
    populations = scalar.scalar_equilibrium(zero, zero, zero)
    cells = [zero[0]]
    intervals = []
    balance = 0.
    for begin, end in zip(ticks[:-1], ticks[1:]):
        integrated = zero.new_zeros(n+1)
        for step in range(begin, end):
            before = populations.sum(0)
            post = scalar.scalar_collide_bgk(populations, before, zero, zero, tau)
            populations, sides = scalar.scalar_stream_slab(post, *prescribed[step])
            heat = dx*torch.cat((sides[0], scalar.scalar_link_transfer(post, "x")[0], -sides[1]))
            residual = dx*(populations.sum(0)[0]-before[0])+heat.diff()
            balance = max(balance, float(abs(residual).max()))
            integrated += heat
        cells.append(populations.sum(0)[0])
        intervals.append(integrated)
    error = abs(torch.stack(intervals).numpy()-np.diff(cumulative, axis=0))
    return dict(columns=n, time_steps=int(ticks[-1]), switch_shift_s=shift, lattice_tau=tau, dt_s=dt,
                temperature_error_K=float(abs(torch.stack(cells).numpy()-means).max()),
                wall_heat_error_J_m2=float(error[:, [0, -1]].max()),
                interior_cut_heat_error_J_m2=float(error[:, 1:-1].max()),
                column_energy_residual_J_m2=balance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fem", action="store_true", help="run the consistent-capacity LINE2 slab")
    group.add_argument("--lbm", action="store_true", help="run the D2Q5 halfway-wall slab")
    group.add_argument("--lbm-ramps", action="store_true", help="drive independent off-grid wall ramps")
    group.add_argument("--lbm-off-grid", action="store_true", help="shift both wall pulses between lattice times")
    group.add_argument("--dg-ramps", action="store_true", help="run SIPG under independent wall ramps")
    group.add_argument("--dg", action="store_true", help="run the SIPG linear-element slab")
    group.add_argument("--dem", action="store_true", help="run the fixed thermal contact chain")
    group.add_argument("--fvm", action="store_true", help="run the finite-volume cell-mean comparison")
    parser.add_argument("--dg-degree", type=int, choices=(1, 2), help="polynomial degree for the DG slab (default 1)")
    args = parser.parse_args()
    if args.dg_degree is not None and not (args.dg or args.dg_ramps):
        parser.error("--dg-degree requires --dg or --dg-ramps")
    rows = []
    previous = None
    if args.fem:
        from tensorfem import thermal as solver
        run = measure_fem
    elif args.lbm or args.lbm_off_grid or args.lbm_ramps:
        from tensorlbm import scalar as solver
        run = partial(measure_lbm, off_grid=args.lbm_off_grid, ramps=args.lbm_ramps)
    elif args.dg or args.dg_ramps:
        from tensordg import heat_1d as solver
        run = partial(measure_dg, ramps=args.dg_ramps, degree=args.dg_degree or 1)
    elif args.dem:
        from tensordem.thermal import contact_chain as solver
        run = measure_dem
    elif args.fvm:
        from tensorfvm.cht import rod as solver
        run = measure_fvm
    else:
        from tensorsph import heat_conduction_2d as solver
        run = measure
    minimum_order = 2.8 if args.dg_degree == 2 else 1.8
    for n in (8, 16, 32, 64):
        row = run(n)
        errors = np.array([row[k] for k in ("temperature_error_K", "wall_heat_error_J_m2",
                                            "interior_cut_heat_error_J_m2")])
        if args.dem:
            errors = np.append(errors, row["interval_temperature_error_K"])
        orders = np.log2(previous/errors).tolist() if previous is not None else []
        row.update(orders=orders, passed=bool(np.all(np.isfinite(errors)) and
                   row["column_energy_residual_J_m2"] < 1e-12 and row.get("wall_trace_error_K", 0.) < 1e-14
                   and row.get("interface_heat_mismatch_J_m2", 0.) < 1e-12
                   and all(order > 0 for order in orders) and (n < 64 or min(orders) > minimum_order)))
        rows.append(row)
        previous = errors
    print(json.dumps(dict(runs=rows, sources=[provenance(__file__), provenance(solver.__file__)]),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
