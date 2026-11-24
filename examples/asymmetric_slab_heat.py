"""Verify signed wall heat and cell-average temperature for a two-mode unit slab."""
import json
import math

import numpy as np
import torch
from scipy.optimize import brentq

from conduction_provenance import provenance
from tensordg import heat_1d as dg, sipg_sparse
from tensorfem import thermal
from tensorfvm.cht import energy
from tensorsph import heat_conduction_2d as sph


def particle_slab(initial, dx, end, steps):
    # A single periodic row represents a transversely uniform slab of width dx.
    field = torch.from_numpy(initial[None, :])
    offsets = sph.lattice_offsets(dx, 1.3*dx)
    calibration = sph.lattice_diffusion_factor_2d(1.3, dx, mode="moment")
    dt = end/steps
    powers = np.empty((steps, 2))
    for i in range(steps):
        rate = sph.stencil_rhs_2d(field, offsets, x_wall_temperatures=(0., 0.))
        midpoint = field + (.5*dt/calibration)*rate
        rate, power = sph.stencil_rhs_with_wall_power_2d(midpoint, offsets, dx, (0., 0.))
        field = field + (dt/calibration)*rate
        powers[i] = power.numpy()/(dx*calibration)
    return field[0].numpy(), powers



def downward_crossing(times, values):
    """Interpolate the unique positive-to-nonpositive crossing."""
    crossings = np.flatnonzero((values[:-1] > 0) & (values[1:] <= 0))
    if len(crossings) != 1:
        raise ValueError(f"expected one downward crossing, found {len(crossings)}")
    i = crossings[0]
    return float(times[i] + (times[i+1]-times[i])*values[i]/(values[i]-values[i+1]))


def measure(n, end=.03):
    k = np.pi * np.array([1., 2.])
    amplitudes = np.array([1., .8])
    x = np.linspace(0., 1., n+1)
    dx = 1/n
    centers = (x[1:] + x[:-1])/2
    times = np.linspace(0., end, 12*n+1)
    # Analytic averages over the physical cells, including the finite cell width.
    modes = np.sin(k[:, None]*centers) * np.sinc(k[:, None]*dx/(2*np.pi))
    initial = amplitudes @ modes
    exact = (amplitudes*np.exp(-k*k*end)) @ modes
    oriented = (amplitudes*np.expm1(-k*k*end)/k) @ np.cos(k[:, None]*x[[0,-1]])
    exact_inward = oriented*np.array([1., -1.])

    def profile(x):
        return torch.sin(math.pi*x) + .8*torch.sin(2*math.pi*x)

    _, coeff, info = dg.solve_heat_1d(n, 2, end, len(times)-1, backend="sparse",
                                      integrate_face_fluxes=True, initial_temperature=profile)
    integrated = np.asarray(info["integrated_face_fluxes"])
    # Evaluate initial DG storage from the projected field.
    _, start, _ = dg.solve_heat_1d(n, 2, 0., backend="sparse", initial_temperature=profile)
    results = [("DG", coeff[:,0].numpy(), integrated[[0,-1]]*[1.,-1.],
                float((coeff[:,0]-start[:,0]).mean()))]

    nodes = torch.from_numpy(x)
    ids = torch.arange(n+1)
    model = thermal.ThermalModel(nodes[:,None], torch.stack((ids[:-1], ids[1:]), dim=1),
                                 1., 1., 1., ids[[0,-1]], nodes.new_zeros(2))
    # L2 projection onto the continuous linear hats with zero wall values.
    # For each sine, the hat load is dx*sinc(k*dx/2)^2 times its nodal value
    # and the consistent mass eigenvalue is dx*(2+cos(k*dx))/3.
    projection = 3*np.sinc(k*dx/(2*np.pi))**2 / (2+np.cos(k*dx))
    start_nodes = (amplitudes*projection) @ np.sin(k[:,None]*x)
    sol = thermal.solve_transient_thermal(model, torch.from_numpy(times), torch.from_numpy(start_nodes),
                                          theta=.5, sparse=True)
    cells = .5*(sol.temperature[:,1:] + sol.temperature[:,:-1])
    reactions = sol.integrated_nodal_reactions()
    results.append(("FEM", cells[-1].numpy(), reactions[[0,-1]].numpy(),
                    float((cells[-1]-cells[0]).mean())))

    bc = {side: {"type": "dirichlet", "values": 0.} for side in ("west", "east")}
    temperatures, power = energy.solve_transient_conduction(
        np.full(n, dx), [1.], 1., 1., bc, times, initial, return_wall_power=True)
    history = temperatures[:,0]
    rates = np.column_stack((power["west"][:,0], power["east"][:,0]))
    increments = np.diff(times)[:,None] * (.5*(rates[:-1]+rates[1:]))
    inward = increments.sum(axis=0)
    results.append(("FVM", history[-1], inward, float((history[-1]-history[0]).mean())))

    steps = math.ceil(end/(.16*dx**2))
    particles, particle_power = particle_slab(initial, dx, end, steps)
    supplied = (end/steps)*particle_power.sum(axis=0)
    results.append(("SPH", particles, supplied, float((particles-initial).mean())))
    rows = [dict(solver=name, cells=n, time_s=end, cell_temperature_error_K=float(np.max(abs(field-exact))),
                 inward_heat_J_m2=inward.tolist(), exact_inward_heat_J_m2=exact_inward.tolist(),
                 wall_heat_error_J_m2=float(np.max(abs(inward-exact_inward))),
                 balance_residual_J_m2=float(abs(stored-inward.sum())))
            for name, field, inward, stored in results]
    if end == .08:
        # East inward power = -pi*exp(-pi²t) + 1.6*pi*exp(-4pi²t).
        exact_power_zero = math.log(1.6)/(3*math.pi**2)
        def exact_east_heat(t):
            return math.expm1(-math.pi**2*t)/math.pi - .4*math.expm1(-4*math.pi**2*t)/math.pi
        exact_heat_zero = brentq(exact_east_heat, exact_power_zero, end)
        particle_times = np.linspace(0., end, steps+1)
        for name, rate_times, east_rate, heat_times, east_heat in (
            ("FVM", times, rates[:,1], times, np.r_[0., increments[:,1].cumsum()]),
            ("SPH", .5*(particle_times[1:]+particle_times[:-1]), particle_power[:,1],
             particle_times, np.r_[0., (end/steps)*particle_power[:,1].cumsum()]),
        ):
            row = next(row for row in rows if row["solver"] == name)
            for label, t, values, reference in (
                ("power", rate_times, east_rate, exact_power_zero),
                ("heat", heat_times, east_heat, exact_heat_zero),
            ):
                crossing = downward_crossing(t, values)
                row[f"east_{label}_reversal_s"] = crossing
                row[f"exact_east_{label}_reversal_s"] = reference
                row[f"east_{label}_reversal_error_s"] = abs(crossing-reference)
    particle_row = rows[-1]
    particle_row["steps"] = steps
    if n == 64:
        smaller_dt, smaller_dt_power = particle_slab(initial, dx, end, 2*steps)
        smaller_dt_heat = (end/(2*steps))*smaller_dt_power.sum(axis=0)
        particle_row["half_step_temperature_difference_K"] = float(np.max(abs(smaller_dt-particles)))
        particle_row["half_step_heat_difference_J_m2"] = float(np.max(abs(smaller_dt_heat-supplied)))
    return rows


def campaign(end):
    previous = {}
    rows = []
    for n in (8, 16, 32, 64):
        for row in measure(n, end):
            name = row["solver"]
            wall, temperature = row["wall_heat_error_J_m2"], row["cell_temperature_error_K"]
            left, right = row["inward_heat_J_m2"]
            ref_left, ref_right = row["exact_inward_heat_J_m2"]
            passed = left*ref_left > 0 and right*ref_right > 0 and row["balance_residual_J_m2"] < 1e-11
            if name in previous:
                old_wall, old_temperature = previous[name]
                passed &= wall < old_wall/3 and temperature < old_temperature/3
            if n == 64:
                passed &= wall < 1e-4 and temperature < 5e-4
                if "east_power_reversal_s" in row:
                    passed &= row["east_power_reversal_error_s"] < 5e-5
                    passed &= row["east_heat_reversal_error_s"] < 5e-5
                if name == "SPH":
                    passed &= row["half_step_temperature_difference_K"] < 1e-6
                    passed &= row["half_step_heat_difference_J_m2"] < 1e-6
            row["passed"] = bool(passed)
            previous[name] = wall, temperature
            rows.append(row)
    return rows


def main():
    rows = campaign(.03) + campaign(.08)
    print(json.dumps({"sources": [provenance(f) for f in
                     (__file__, dg.__file__, sipg_sparse.__file__, thermal.__file__, energy.__file__, sph.__file__)],
                     "runs": rows}, indent=2))
    return 0 if all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
