"""Steady stepped rod with quadratic generation, finite contact and two prescribed ends."""
from dataclasses import replace
import json

import numpy as np
import torch

from conduction_provenance import provenance
from tensorfem import thermal
from tensorfvm.cht import rod


def measure(n, resolve_source=False, transient=False):
    limits = [-.7, .2, 1.1]
    areas, conductivities, coefficients = [.8, 1.3], [1.7, 3.1], [4., -2.2]
    conductance = 2.5
    edges, temperatures, means, sources, powers = [], [], [], [], []
    fractions = np.array([0., .17, .64, 1.])
    probe_temperature, probe_power = [], []
    start_temperature, start_power = 3., .4
    for i, (area, k, beta) in enumerate(zip(areas, conductivities, coefficients)):
        lo, hi = limits[i:i+2]
        u = np.linspace(0, 1, n+1)
        x = lo+(hi-lo)*(.7*u+.3*u**2)
        offset = start_power-area*beta*lo**3/3
        def temperature(position):
            return start_temperature-(offset*(position-lo)+area*beta*(position**4-lo**4)/12)/(k*area)
        edges.append(x)
        temperatures.append(temperature(x))
        probes = x[:-1,None]+np.diff(x)[:,None]*fractions
        probe_temperature.append(temperature(probes))
        probe_power.append(offset+area*beta*probes**3/3)
        # Three Gauss points integrate the quartic temperature exactly.
        midpoint, half = (x[1:]+x[:-1])/2, np.diff(x)/2
        shift = np.sqrt(.6)*half
        means.append((5*temperature(midpoint-shift)+8*temperature(midpoint)+5*temperature(midpoint+shift))/18)
        sources.append(beta*(midpoint**2+half**2/3))
        powers.append(offset+area*beta*x**3/3)
        start_power = powers[-1][-1]
        start_temperature = temperatures[-1][-1]-start_power/conductance
    cell_area, cell_k, beta = [np.repeat(v,n) for v in (areas,conductivities,coefficients)]
    dx = np.concatenate([np.diff(x) for x in edges])
    fixed_t = np.array([temperatures[0][0],temperatures[1][-1]])
    contacts = np.zeros(2*n-1)
    contacts[n-1] = min(areas)/conductance
    midpoints = np.concatenate([(x[:-1]+x[1:])/2 for x in edges])
    derivatives = dict(source_gradient=2*beta*midpoints, source_curvature=2*beta) if resolve_source else {}
    cell, _, left, right, (hot,cold) = rod.solve_steady_stepped_rod(
        dx,cell_area,cell_k,1.,fixed_t,source=np.concatenate(sources),
        contact_resistance=contacts,flux_order=2,return_traces=True,**derivatives)
    starts = np.r_[np.arange(n), n+1+np.arange(n)]
    fixed = torch.tensor([0,2*n+1])
    model = thermal.ThermalModel(torch.from_numpy(np.concatenate(edges)[:,None]),
                                 torch.from_numpy(np.column_stack((starts,starts+1))),
                                 torch.from_numpy(cell_k),1.,1.,fixed,torch.from_numpy(fixed_t),
                                 thickness=torch.from_numpy(cell_area),contact_pairs=torch.tensor([[n,n+1]]),
                                 contact_conductance=conductance,heat_source_quadratic=torch.from_numpy(beta))
    state = thermal.solve_steady_thermal(model,sparse=True)
    matrices = thermal.assemble_thermal(model,sparse=True)
    reactions = (matrices.conductivity@state-matrices.heat)[fixed].numpy()
    wall = np.array([powers[0][0],powers[1][-1]])
    interface = powers[0][-1]
    fem = dict(nodal_temperature_K=float(np.max(abs(state.numpy()-np.concatenate(temperatures)))),
               end_power_W=float(np.max(abs(reactions*[1,-1]-wall))),
               contact_power_W=abs(float(conductance*(state[n]-state[n+1]))-interface))
    fvm = dict(cell_average_temperature_K=float(np.max(abs(cell-np.concatenate(means)))),
               end_power_W=float(np.max(abs(np.array([left[0],right[-1]])-wall))),
               contact_power_W=abs(conductance*(hot[n-1]-cold[n-1])-interface))
    if resolve_source:
        local_power = thermal.thermal_element_nodal_power(model,state)
        sections = thermal.thermal_element_section_power(model,local_power,fractions,
            net_line_source_gradient=cell_area*(2*beta*midpoints),
            net_line_source_curvature=cell_area*2*beta)
        fem["probe_power_W"] = float(abs(sections.numpy()-np.concatenate(probe_power)).max())
        t,p = rod.reconstruct_steady_rod_field(dx,cell_area,cell_k,cell,left,right,fractions,**derivatives)
        fvm["probe_temperature_K"] = float(np.max(abs(t-np.concatenate(probe_temperature))))
        fvm["probe_power_W"] = float(np.max(abs(p-np.concatenate(probe_power))))
    if transient:
        # Add a time-linear mode with constant Fourier power rate through both
        # materials. Its nodal slope and contact jump satisfy the interface law.
        times = np.array([0.,.07,.19,.4])
        power_rate = .23
        lift = []
        start = .6
        for x,a,k in zip(edges,areas,conductivities):
            lift.append(start-power_rate*(x-x[0])/(k*a))
            start = lift[-1][-1]-power_rate/conductance
        speed = np.concatenate(lift)
        cell_speed = np.concatenate([.5*(v[1:]+v[:-1]) for v in lift])
        ends = speed[[0,-1]]
        mean_time = (times[:-1]+times[1:])/2
        ref_power = np.concatenate(probe_power)+power_rate*mean_time[:,None,None]
        ref_speed = np.vstack([v[:-1,None]+np.diff(v)[:,None]*fractions for v in lift])
        ref_field = np.concatenate(probe_temperature)+mean_time[:,None,None]*ref_speed
        transient_model = replace(model,heat_source=torch.from_numpy(speed)[model.elements])
        result = thermal.solve_transient_thermal(transient_model,torch.from_numpy(times),
            torch.from_numpy(np.concatenate(temperatures)),method="sdirk2",sparse=True,
            fixed_temperature_history=lambda t: model.fixed_temperatures+t*torch.from_numpy(ends))
        local = thermal.thermal_element_nodal_heat_steps(transient_model,result)/result.time.diff()[:,None,None]
        recovered = thermal.thermal_element_section_power(transient_model,local,fractions,
            net_line_source_gradient=2*cell_area*beta*midpoints,net_line_source_curvature=2*cell_area*beta)
        fem["transient_probe_power_W"] = float(abs(recovered.numpy()-ref_power).max())
        fem["transient_nodal_temperature_K"] = float(abs(result.temperature.numpy()
            -np.concatenate(temperatures)-times[:,None]*speed).max())
        *_,heat,intervals = rod.solve_transient_stepped_rod(dx,cell_area,cell_k,1.,
            lambda t:fixed_t+t*ends,times,np.concatenate(means),source=np.concatenate(sources)+cell_speed,
            flux_order=2,contact_resistance=contacts,net_source_gradient=2*beta*midpoints,
            net_source_curvature=2*beta,method="sdirk2",return_interval_temperatures=True,return_step_heat=True)
        field,flux = rod.reconstruct_interval_rod_field(dx,cell_area,cell_k,intervals,fractions)
        fvm["transient_probe_temperature_K"] = float(abs(field-ref_field).max())
        fvm["transient_probe_power_W"] = float(abs(flux-ref_power).max())
        fvm["transient_contact_heat_J"] = float(abs(heat["contact"][:,n-1]-
            np.diff(times)*(interface+mean_time*power_rate)).max())
    return fem,fvm


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolve-source", action="store_true", help="resolve within-cell generation in FEM and FVM section powers")
    parser.add_argument("--transient",action="store_true",help="also verify moving contact temperatures and interval powers with graded net sources")
    args = parser.parse_args()
    resolve = args.resolve_source or args.transient
    rows = []
    previous = None
    for n in (4,8,16,32,64):
        fem,fvm = measure(n, resolve,args.transient)
        orders = {} if previous is None or resolve else {k:float(np.log2(previous[k]/v)) for k,v in fvm.items()}
        passed = max(fem.values()) < 1e-10 and all(np.isfinite(list(fvm.values())))
        if resolve:
            passed &= max(fvm.values()) < 1e-10
        elif previous is not None:
            passed &= min(orders.values()) > 1.8
        rows.append(dict(cells_per_material=n,FEM=fem,FVM=fvm,FVM_orders=orders,passed=bool(passed)))
        previous = fvm
    print(json.dumps(dict(resolved_source=resolve,transient=args.transient,sources=[provenance(p) for p in (__file__,thermal.__file__,rod.__file__)],
                          runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
