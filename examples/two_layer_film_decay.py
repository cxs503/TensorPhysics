"""Compare FEM, FVM, DEM and SPH with a decaying contact mode cooled by two wall films."""
import argparse
import inspect
import json
import math

import numpy as np
import torch
from scipy.integrate import simpson

from conduction_provenance import provenance
from two_layer_conduction import slab_fem_model
from tensordem.thermal import contact_chain
from tensorfem import thermal
from tensorfvm.cht import energy
from tensorsph import heat_conduction_2d as sph


def measure(n, *, time_refinement=1, sampled_dem=False, implicit_sph=False):
    a, b = 1.5*math.pi, 4*math.pi/3
    phi, psi = .25, .4
    cap = 4*(b/a)**2
    amp = a*math.cos(a/2+phi) / (4*b*math.cos(b/2+psi))
    jump = math.sin(a/2+phi) + amp*math.sin(b/2+psi)
    resistance = jump / (-a*math.cos(a/2+phi))
    films = (a/math.tan(phi), 4*b/math.tan(psi))
    k = np.repeat([1., 4.], n//2)
    capacity = np.repeat([1., cap], n//2)
    time = np.linspace(0., .015, time_refinement*n+1)
    decay = math.exp(-a*a*time[-1])
    layer_heat = np.array([(math.cos(phi)-math.cos(a/2+phi))/a,
                           -cap*amp*(math.cos(psi)-math.cos(b/2+psi))/b])
    heat0 = layer_heat.sum()

    def mode(xl, xr):
        return np.r_[np.sin(a*xl+phi), -amp*np.sin(b*(1-xr)+psi)]

    model, ends = slab_fem_model(k, 0., 0., resistance, films, torch.from_numpy(capacity))
    x = np.linspace(0, 1, n+1)
    m = n//2
    initial = mode(x[:m+1], x[m:])
    fem_history = thermal.solve_transient_thermal(
        model, torch.from_numpy(time), torch.from_numpy(np.tile(initial, 2)), theta=.5).temperature
    fem_t = fem_history[-1]
    walls = fem_t[ends].mean(dim=1).numpy()
    final = fem_t.reshape(2, -1).mean(dim=0).numpy()
    delta = final-initial
    # Exact integral of the FEM piecewise-linear temperature along x.
    heat = (delta[:m].sum()+delta[1:m+1].sum()
            + cap*(delta[m+1:-1].sum()+delta[m+2:].sum()))/(2*n)
    values = [("FEM", np.tile(initial, 2), fem_t.numpy(), walls, final[m]-final[m+1], heat)]

    # Massless ambient reservoirs surround two rods with lumped nodal
    # capacities. Coincident interface nodes are linked by the contact.
    links = np.r_[1/films[0], 1/(n*k[:m]), resistance, 1/(n*k[m:]), 1/films[1]]
    nodal_capacity = np.r_[np.ones(m+1), np.full(m+1, cap)] / n
    nodal_capacity[[0, m, m+1, -1]] *= .5
    nodal_start = initial.copy()
    initial_errors = []
    for part, mass, target in zip(np.split(nodal_start, 2), np.split(nodal_capacity, 2), layer_heat):
        if not sampled_dem:
            # Minimum capacity-weighted squared change subject to the layer heat integral.
            part += (target - mass @ part) / mass.sum()
        initial_errors.append(float(mass @ part - target))
    history = contact_chain.transient_contact_temperatures(
        links, np.r_[0., nodal_capacity, 0.], time, np.r_[0., nodal_start, 0.],
        t_left=0., t_right=0.)
    dem_temperatures = history[:,1:-1]
    final = dem_temperatures[-1]
    values.append(("DEM", initial, final, final[[0,-1]], final[m]-final[m+1],
                   nodal_capacity @ (final-nodal_start)))

    # Independent endpoint powers at every step, positive into the rods.
    film_power = -history[:, 1]*films[0] - history[:, -2]*films[1]
    dem_balance = float(abs(np.diff(time) @ (.5*(film_power[1:]+film_power[:-1]))
                            - nodal_capacity @ (final-nodal_start)))

    # Split the signed continuum discrepancy into nodal-integration and
    # evolution contributions, using exact samples at the same nodes.
    exact_sample_change = nodal_capacity @ (decay*initial - nodal_start)
    integration_error = exact_sample_change - (decay - 1) * heat0
    evolution_error = nodal_capacity @ (final-nodal_start) - exact_sample_change

    # Integrate the reconstructed continuum field separately in each material;
    # averaging across the contact would erase its temperature discontinuity.
    change = final - initial
    reconstructed_heat = (simpson(change[:m+1], x=x[:m+1])
                          + cap * simpson(change[m+1:], x=x[m:]))
    exact_change = (decay-1)*heat0
    reconstructed_error = float(abs(reconstructed_heat - exact_change))

    centers = (np.arange(n)+.5)/n
    initial = mode(centers[:m], centers[m:])
    faces = np.zeros((1, n-1))
    faces[0,m-1] = resistance
    bc = {side: {"type": "robin", "values": 0., "film_coefficient": h}
          for side, h in zip(("west", "east"), films)}
    fvm_history = energy.solve_transient_conduction(np.full(n, 1/n), [1.], k, capacity, bc, time,
                                              initial, contact_resistance_x=faces)[:,0]
    final = fvm_history[-1]
    # Recover each wall trace from the half-cell drop and the film law.
    walls = final[[0,-1]] / (1 + np.asarray(films)/(2*n*k[[0,-1]]))
    contact_jump = 1.5*(final[m-1]-final[m]) - .5*(final[m-2]-final[m+1])
    values.append(("FVM", initial, final, walls, contact_jump, np.mean(capacity*(final-initial))))
    # A second prescribed contact network places one store in each material
    # segment, so no segment shares its capacity with a wall or contact node.
    # Project the two sine modes onto those volumes before marching.
    cell_initial = initial * np.repeat(np.sinc(np.array([a,b])/(2*n*np.pi)), m)
    links, network_capacity = contact_chain.cell_centered_slab_network(
        np.full(n, 1/n), k, capacity, films=films, contact_resistance=faces[0])
    mass = network_capacity[1:-1]
    history = contact_chain.transient_contact_temperatures(
        links, network_capacity, time, np.r_[0., cell_initial, 0.], t_left=0., t_right=0.)
    temp = history[-1, 1:-1]
    wall_t = temp[[0,-1]] / (np.asarray(films)*links[[0,-1]])
    q_interface = (temp[m-1] - temp[m])/links[m]
    values.append(("DEM_cell", cell_initial, temp, wall_t, resistance*q_interface,
                   mass @ (temp-cell_initial)))
    inward = -history[:, 1]/links[0] - history[:, -2]/links[-1]
    cell_balance = float(abs(np.diff(time) @ (.5*(inward[1:]+inward[:-1]))
                             - mass @ (temp-cell_initial)))
    dx = 1/n
    section_metrics = {}
    if implicit_sph:
        # Rightward power is -h_W*T_W on the west and +h_E*T_E on the east.
        flux0 = np.array([-films[0]*math.sin(phi), -a*math.cos(a/2+phi), -films[1]*amp*math.sin(psi)])
        reference = (-np.expm1(-a*a*time[1:])/a**2)[:,None]*flux0
        def check_sections(name, interval_heat, stored):
            cumulative = np.cumsum(interval_heat,axis=0)
            residual = np.diff(stored,axis=0)+np.diff(interval_heat,axis=1)
            section_metrics[name] = dict(
                max_cumulative_heat_errors_J_m2=dict(zip(("west","contact","east"),
                    np.max(abs(cumulative-reference),axis=0).tolist())),
                max_layer_balance_J_m2=float(abs(residual).max()))

        def integrate(power):
            return np.diff(time)[:,None]*(power[:-1]+power[1:])/2

        dem_power = np.column_stack((-films[0]*dem_temperatures[:,0],
                     (dem_temperatures[:,m]-dem_temperatures[:,m+1])/resistance,
                     films[1]*dem_temperatures[:,-1]))
        dem_stores = (dem_temperatures*nodal_capacity).reshape(len(time),2,m+1).sum(axis=2)
        check_sections("DEM",integrate(dem_power),dem_stores)

        cells = history[:,1:-1]
        cell_power = np.column_stack((-cells[:,0]/links[0],(cells[:,m-1]-cells[:,m])/links[m],
                                      cells[:,-1]/links[-1]))
        check_sections("DEM_cell",integrate(cell_power),(cells*mass).reshape(len(time),2,m).sum(axis=2))

        nodal = fem_history.reshape(len(time),2,-1).mean(dim=1).numpy()
        fem_power = np.column_stack((-films[0]*nodal[:,0],(nodal[:,m]-nodal[:,m+1])/resistance,
                                     films[1]*nodal[:,-1]))
        fem_stored = np.column_stack(((nodal[:,:m]+nodal[:,1:m+1]).sum(axis=1)*dx/2,
                                      cap*(nodal[:,m+1:-1]+nodal[:,m+2:]).sum(axis=1)*dx/2))
        check_sections("FEM",integrate(fem_power),fem_stored)
        fvm_power = np.column_stack((-fvm_history[:,0]/(1/films[0]+dx/(2*k[0])),
                            (fvm_history[:,m-1]-fvm_history[:,m])/(resistance+dx/(2*k[m-1])+dx/(2*k[m])),
                            fvm_history[:,-1]/(1/films[1]+dx/(2*k[-1]))))
        def cell_stores(fields):
            return (dx*capacity*fields).reshape(len(time),2,m).sum(axis=2)
        check_sections("FVM",integrate(fvm_power),cell_stores(fvm_history))

        states,interval_heat,traces = sph.solve_transient_layered_slab(initial,time,dx,(None,None),
                conductivity=k,volumetric_heat_capacity=capacity,contact_resistance_x=faces[0],
                x_wall_films=((films[0],0.),(films[1],0.)),return_x_heat=True,return_wall_temperatures=True)
        final = states[-1].numpy()
        trace_jump = 1.5*(final[m-1]-final[m])-.5*(final[m-2]-final[m+1])
        stored = np.mean(capacity*(final-initial))
        values.append(("SPH_implicit",initial,final,traces[-1].numpy(),trace_jump,stored))
        check_sections("SPH_implicit",interval_heat[:,[0,m,-1]].numpy(),cell_stores(states.numpy()))
    else:
        field = torch.from_numpy(initial[None, :])
        rhs = sph.prepare_slab_film_rhs(
            field, sph.lattice_offsets(dx, 1.3*dx), dx, films, conductivity=torch.from_numpy(k),
            specific_heat=torch.from_numpy(capacity), layered_x=True, contact_resistance_x=faces[0])
        steps = math.ceil(time[-1]*max(k/capacity)/(.16*dx**2))*time_refinement
        dt = time[-1]/steps
        integrated_film = 0.
        for _ in range(steps):
            predictor, _, _ = rhs(field, (0., 0.))
            rate, _, midwall = rhs(field + .5*dt*predictor, (0., 0.))
            integrated_film -= dt * (films[0]*float(midwall[0]) + films[1]*float(midwall[1]))
            field = field + dt*rate
        final = field[0].numpy()
        _, _, walls = rhs(field, (0., 0.))
        jump_sph = 1.5*(final[m-1]-final[m]) - .5*(final[m-2]-final[m+1])
        stored = np.mean(capacity*(final-initial))
        values.append(("SPH", initial, final, walls.numpy(), jump_sph, stored))
        sph_balance = float(abs(integrated_film-stored))
    wall_exact = decay * np.array([math.sin(phi), -amp*math.sin(psi)])
    rows = [{"solver": name, "cells": n,
             "errors": {"temperature_K": float(abs(final - decay*initial).max()),
                        "wall_K": float(abs(walls-wall_exact).max()),
                        "jump_K": float(abs(contact_jump-decay*jump)),
                        "heat_change_J_m2": float(abs(heat-(decay-1)*heat0))}}
            for name, initial, final, walls, contact_jump, heat in values]
    for row in rows:
        if row["solver"] in section_metrics:
            row.update(section_metrics[row["solver"]])
        if row["solver"] == "DEM":
            row["initial_layer_heat_errors_J_m2"] = initial_errors
            row["initialization"] = "point_samples" if sampled_dem else "conservative_layer_projection"
            row["discrete_heat_balance_residual_J_m2"] = dem_balance
            row["simpson_heat_change_error_J_m2"] = reconstructed_error
            row["signed_storage_errors_J_m2"] = {"nodal_integration": float(integration_error),
                                                  "temperature_evolution": float(evolution_error)}
        elif row["solver"] == "SPH":
            row["discrete_heat_balance_residual_J_m2"] = sph_balance
        elif row["solver"] == "DEM_cell":
            row["discrete_heat_balance_residual_J_m2"] = cell_balance
    return rows


def main(sampled_dem=False, implicit_sph=False):
    runs = []
    previous = {}
    previous_sections = {}
    for n in (16, 32, 64, 128):
        for row in measure(n, sampled_dem=sampled_dem, implicit_sph=implicit_sph):
            errors = row["errors"]
            last = previous.get(row["solver"])
            row["passed"] = all(math.isfinite(v) for v in errors.values())
            if row.get("initialization") == "conservative_layer_projection":
                row["passed"] &= max(map(abs, row["initial_layer_heat_errors_J_m2"])) < 1e-14
            if last:
                row["passed"] &= all(v < last[k]/3 for k, v in errors.items())
            if n >= 64:
                limits = {"temperature_K": .001, "wall_K": .001, "jump_K": .003, "heat_change_J_m2": 1e-5}
                row["passed"] &= all(value < limits[key] for key, value in errors.items())
            if "discrete_heat_balance_residual_J_m2" in row:
                row["passed"] &= row["discrete_heat_balance_residual_J_m2"] < 1e-12
            if "max_layer_balance_J_m2" in row:
                section_errors = row["max_cumulative_heat_errors_J_m2"]
                prior = previous_sections.get(row["solver"])
                row["passed"] &= row["max_layer_balance_J_m2"]<1e-12
                if n==128:
                    row["passed"] &= max(section_errors.values())<1e-5
                if prior:
                    row["passed"] &= all(v<prior[k]/3 for k,v in section_errors.items())
                previous_sections[row["solver"]] = section_errors
            previous[row["solver"]] = errors
            runs.append(row)
    print(json.dumps({"sources": [provenance(f) for f in
                      (__file__, inspect.getfile(slab_fem_model), thermal.__file__, energy.__file__, contact_chain.__file__, sph.__file__)],
                      **({"section_heat_reference": "Integral of rightward Fourier flux at x=0, 0.5, 1 m; maximum absolute error over all output times."} if implicit_sph else {}),
                      "time_s": .015, "runs": runs}, indent=2))
    return 0 if all(row["passed"] for row in runs) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampled-dem", action="store_true", help="reproduce the original nodal initialization")
    parser.add_argument("--implicit-sph",action="store_true",help="compare FEM, FVM, DEM and implicit SPH section heat histories")
    opts = parser.parse_args()
    raise SystemExit(main(opts.sampled_dem,opts.implicit_sph))
