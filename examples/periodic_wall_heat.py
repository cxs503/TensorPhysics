"""Harmonic boundary heating of a unit slab, with complex field and power checks."""
import argparse
import json
import math

import numpy as np
import torch

from conduction_provenance import provenance
from tensorfvm.cht import energy
from tensorfem import thermal
from tensorsph import heat_conduction_2d as sph


def measure(n,two_walls=False):
    dx, period = 1/n, .08
    omega = 2*np.pi/period
    wave = np.sqrt(1j*omega)
    faces = np.linspace(0., 1., n+1)
    # Complex cell averages and inward boundary fluxes, for T=Re(U exp(i omega t)).
    avg = -np.diff(np.cosh(wave*(1-faces)))/(dx*wave*np.sinh(wave))
    east = .45*np.exp(-.7j) if two_walls else 0.
    avg = avg+east*avg[::-1]
    flux = wave*np.array([np.cosh(wave)-east,east*np.cosh(wave)-1.])/np.sinh(wave)
    def wall_values(t):
        return math.cos(omega*t),float((east*np.exp(1j*omega*t)).real)
    steps = max(12*n, math.ceil(period/(.16*dx**2)))
    time = np.linspace(0., period, steps+1)
    dt = period/steps
    phase = np.exp(-1j*omega*time)
    bc = {"west": {"type": "dirichlet", "values": lambda t: math.cos(omega*t)},
          "east": {"type": "dirichlet", "values": lambda t:wall_values(t)[1]}}
    temp, wall = energy.solve_transient_conduction(np.full(n, dx), [1.], 1., 1., bc,
                                                   time, avg.real, return_wall_power=True)
    temp = temp[:,0]
    power = np.column_stack((wall["west"][:,0], wall["east"][:,0]))

    def harmonic(samples):
        return (phase[:-1] @ samples[:-1] + phase[1:] @ samples[1:])/steps

    exact_heat = (np.expm1(1j*omega*time[:,None])*flux/(1j*omega)).real
    def heat_error(increments):
        return float(abs(increments.cumsum(0)-exact_heat[1:]).max())

    def step_balance(temperatures,increments):
        return float(abs(np.diff(temperatures.mean(1))-increments.sum(1)).max())

    fv_heat = .5*dt*(power[:-1]+power[1:])
    runs = [("FVM", harmonic(temp), harmonic(power), (temp[-1]-temp[0]).mean()-fv_heat.sum(),heat_error(fv_heat),step_balance(temp,fv_heat))]
    nodes = torch.from_numpy(faces)
    ids = torch.arange(n+1)
    def prescribed(t):
        return torch.stack((torch.cos(omega*t),np.real(east)*torch.cos(omega*t)
                            -np.imag(east)*torch.sin(omega*t)))
    model = thermal.ThermalModel(nodes[:,None], torch.stack((ids[:-1], ids[1:]), dim=1),
                                 1., 1., 1., ids[[0,-1]], prescribed(nodes[0]))
    initial = ((np.sinh(wave*(1-faces))+east*np.sinh(wave*faces))/np.sinh(wave)).real.copy()
    mid_phase = np.exp(-1j*omega*(time[:-1]+dt/2))
    for name,method in (("FEM","theta"),("FEM_SDIRK2","sdirk2")):
        sol = thermal.solve_transient_thermal(model, torch.from_numpy(time), torch.from_numpy(initial),
                             theta=.5, sparse=True, method=method, fixed_temperature_history=prescribed)
        history = sol.temperature
        cell_mean = .5*(history[:,1:] + history[:,:-1])
        wall_heat = sol.nodal_reaction_heat_steps()[:, [0,-1]].numpy()
        runs.append((name, harmonic(cell_mean.numpy()), 2/period*(mid_phase @ wall_heat),
                     float((cell_mean[-1]-cell_mean[0]).mean())-wall_heat.sum(),heat_error(wall_heat),
                     step_balance(cell_mean.numpy(),wall_heat)))
    field = torch.from_numpy(avg.real.copy()[None,:])
    offsets = sph.lattice_offsets(dx, 1.3*dx)
    factor = sph.lattice_diffusion_factor_2d(1.3, dx, mode="moment")
    particle_temp = np.empty_like(temp)
    particle_temp[0] = field.numpy()[0]
    mid_power = np.empty((steps, 2))
    for i, t in enumerate(time[:-1]):
        rate = sph.stencil_rhs_2d(field, offsets, x_wall_temperatures=wall_values(t))
        mid = field + .5*dt/factor*rate
        rate, watts = sph.stencil_rhs_with_wall_power_2d(
            mid, offsets, dx, wall_values(t+.5*dt))
        field = field + dt/factor*rate
        particle_temp[i+1] = field.numpy()[0]
        mid_power[i] = watts.numpy()/(dx*factor)
    runs.append(("SPH", harmonic(particle_temp), 2/steps*(mid_phase @ mid_power),
                 (particle_temp[-1]-particle_temp[0]).mean()-dt*mid_power.sum(),heat_error(dt*mid_power),
                 step_balance(particle_temp,dt*mid_power)))
    return [dict(solver=name, cells=n, steps=steps,
                 complex_temperature_error_K=float(np.max(abs(u-avg))),
                 complex_wall_flux_error_W_m2=float(np.max(abs(q-flux))),
                 driven_wall_phase_error_deg=float(abs(np.angle(q[0]/flux[0], deg=True))),
                 heat_balance_J_m2=float(abs(balance)),max_cumulative_wall_heat_error_J_m2=heat,
                 max_step_balance_J_m2=residual)
                 for name,u,q,balance,heat,residual in runs]


def main(two_walls=False):
    rows = []
    previous = {}
    for n in (16,32,64,128):
        for row in measure(n,two_walls):
            name = row["solver"]
            errors = np.array([row["complex_temperature_error_K"], row["complex_wall_flux_error_W_m2"],
                               row["driven_wall_phase_error_deg"],row["max_cumulative_wall_heat_error_J_m2"]])
            passed = np.isfinite(errors).all() and max(row["heat_balance_J_m2"],row["max_step_balance_J_m2"]) < 1e-11
            if name in previous:
                passed &= (errors < previous[name]/3).all()
            row["meets_absolute_targets"] = bool((errors < [.003, .08, .5, .002]).all())
            if n == 128:
                passed &= row["meets_absolute_targets"]
            previous[name] = errors
            row["passed"] = bool(passed)
            rows.append(row)
    print(json.dumps({"sources": [provenance(f) for f in (__file__, energy.__file__, thermal.__file__, sph.__file__)],
                      "two_driven_walls": two_walls, "period_s": .08, "runs": rows}, indent=2))
    return 0 if all(r["passed"] for r in rows) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--two-walls",action="store_true",help="drive both walls with different amplitudes and phases")
    raise SystemExit(main(parser.parse_args().two_walls))
