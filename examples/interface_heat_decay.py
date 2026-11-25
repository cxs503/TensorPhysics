"""Free decay of two layers coupled through a thermally massive, resistive sheet."""
import json

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq
import torch

import interface_heat_storage as slab
from conduction_provenance import provenance
from tensordem.thermal import contact_chain
from tensorfem import thermal


def eigenmode(rate, resistance, sheet_capacity):
    # Both outer temperatures vanish. Start at the hot-side wall with sin(a*x).
    a = np.sqrt(rate)
    b = np.sqrt(rate*2/4)
    left = np.sin(.5*a)
    q_left = -a*np.cos(.5*a)
    sheet = left-resistance*q_left/2
    q_right = q_left+rate*sheet_capacity*sheet
    right = sheet-resistance*q_right/2
    def profile(x, left_layer):
        if left_layer:
            return np.sin(a*x)
        z = b*(x-.5)
        return right*np.cos(z)-q_right*np.sin(z)/(4*b)
    return profile, sheet, np.array([q_left, q_right]), profile(1., False)


def measure(n, steps, rate, resistance, capacity):
    mid = n//2
    profile, sheet, flux, _ = eigenmode(rate, resistance, capacity)
    # Follow two e-folding times. Integrate the reference flux over each interval.
    time = np.linspace(0., 2/rate, steps+1)
    dt = np.diff(time)
    decay = np.exp(-rate*time)
    expected_energy = -np.diff(decay)[:, None]*flux/rate
    r, c, model = slab.models(n, resistance, capacity)
    x = np.arange(n+1)/n
    center = (x[:-1]+x[1:])/2
    # Cell means of sinusoidal profiles on the two layers.
    left = profile(center[:mid], True)*np.sinc(np.sqrt(rate)/(2*n*np.pi))
    right = profile(center[mid:], False)*np.sinc(np.sqrt(rate/2)/(2*n*np.pi))
    init = np.r_[0., left, sheet, right, 0.]
    td = contact_chain.transient_contact_temperatures(r, c, time, init, t_left=0., t_right=0.)
    q = -np.diff(td, axis=1)[:, mid:mid+2]/r[mid:mid+2]
    dem_energy = .5*dt[:,None]*(q[:-1]+q[1:])
    init = np.r_[profile(x[:mid+1], True), profile(x[mid:], False), sheet]
    zeros = torch.zeros((len(time), 2), dtype=torch.float64)
    result = thermal.solve_transient_thermal(model, torch.from_numpy(time), torch.from_numpy(init),
                     .5, sparse=True, fixed_temperature_history=zeros)
    heat = thermal.thermal_element_nodal_heat_steps(model, result).numpy()
    fem_energy = np.column_stack((-heat[:, mid-1, 1], heat[:, mid, 0]))
    cases = (("DEM", td[:, mid+1], dem_energy), ("FEM", result.temperature[:, -1].numpy(), fem_energy))
    for name, temp, energy in cases:
        yield name, np.r_[abs(temp-sheet*decay).max(),
                          abs((energy-expected_energy)/dt[:, None]).max(axis=0)], float(
            abs(capacity*np.diff(temp)-energy[:, 0]+energy[:, 1]).max())


def temporal_study(resistance, capacity):
    """Fix the spatial mesh and evolve its slowest generalized eigenmode."""
    n = 24
    mid = n//2
    r, c, model = slab.models(n, resistance, capacity)
    incidence = np.diff(np.eye(len(c)), axis=0)
    network_k = incidence.T @ (incidence/r[:, None])
    matrices = thermal.assemble_thermal(model)
    cases = (("DEM", network_k, np.diag(c), [0, len(c)-1], mid+1),
             ("FEM", matrices.conductivity.numpy(), matrices.capacity.numpy(), model.fixed_nodes.numpy(), n+2))
    rows = []
    for name, k, mass, walls, sheet_id in cases:
        free = np.setdiff1d(np.arange(len(k)), walls)
        part = np.ix_(free, free)
        rates, vectors = eigh(k[part], mass[part], subset_by_index=(0,0))
        rate = rates[0]
        mode = np.zeros(len(k))
        mode[free] = vectors[:, 0]
        mode /= mode[sheet_id]
        # Use the contact law on each sheet side for the reference powers.
        if name == "DEM":
            q = -np.diff(mode)[mid:mid+2]/r[mid:mid+2]
        else:
            q = 2/resistance*np.array([mode[mid]-1, 1-mode[mid+1]])
        previous = None
        for steps in (8, 16, 32, 64, 128):
            time = np.linspace(0., 2/rate, steps+1)
            dt = np.diff(time)
            exact = np.exp(-rate*time)
            if name == "DEM":
                t = contact_chain.transient_contact_temperatures(r, c, time, mode, t_left=0., t_right=0.)
                powers = -np.diff(t, axis=1)[:,mid:mid+2]/r[mid:mid+2]
                energy = .5*dt[:, None]*(powers[:-1]+powers[1:])
            else:
                result = thermal.solve_transient_thermal(model, torch.from_numpy(time), torch.from_numpy(mode),
                                 .5, sparse=True, fixed_temperature_history=model.nodes.new_zeros((steps+1, 2)))
                t = result.temperature.numpy()
                heat = thermal.thermal_element_nodal_heat_steps(model, result).numpy()
                energy = np.column_stack((-heat[:, mid-1, 1], heat[:, mid, 0]))
            exact_energy = -np.diff(exact)[:, None]*q/rate
            errors = np.r_[abs(t[:, sheet_id]-exact).max(), abs((energy-exact_energy)/dt[:,None]).max(axis=0)]
            orders = np.log2(previous/errors) if previous is not None else None
            passed = np.isfinite(errors).all() and (orders is None or min(orders) > 1.8)
            rows.append(dict(solver=name, cells=n, steps=steps, discrete_decay_rate_per_s=rate,
                        max_sheet_error_K=errors[0], left_mean_flux_error_W_m2=errors[1],
                        right_mean_flux_error_W_m2=errors[2], orders=orders.tolist() if orders is not None else None,
                        passed=bool(passed)))
            previous = errors
    return rows


def main():
    rows = []
    resistance, capacity = .3, .8
    rate = brentq(lambda a: eigenmode(a, resistance, capacity)[-1], .1, 12., xtol=1e-14)
    previous = {}
    for n in (8, 16, 32, 64, 128):
        for name, errors, balance in measure(n, 4*n, rate, resistance, capacity):
            orders = np.log2(previous[name]/errors) if name in previous else None
            passed = np.isfinite(errors).all() and balance < 1e-12
            if orders is not None:
                passed &= min(orders) > (1.9 if n == 128 else 0.)
            rows.append(dict(solver=name, cells=n, steps=4*n, max_sheet_error_K=errors[0],
                             left_mean_flux_error_W_m2=errors[1], right_mean_flux_error_W_m2=errors[2],
                             max_sheet_balance_J_m2=balance, orders=orders.tolist() if orders is not None else None,
                             passed=bool(passed)))
            previous[name] = errors
    time_runs = temporal_study(resistance, capacity)
    print(json.dumps(dict(temporal_runs=time_runs, decay_rate_per_s=rate, resistance_m2K_W=resistance, sheet_capacity_J_m2K=capacity,
                     runs=rows, sources=[provenance(f) for f in
                     (__file__, slab.__file__, thermal.__file__, contact_chain.__file__)]), indent=2, allow_nan=False))
    return 0 if all(row["passed"] for row in rows+time_runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
