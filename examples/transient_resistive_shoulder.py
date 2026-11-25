"""Decay through a resistive overlap and an exposed storing shoulder."""
from dataclasses import replace
import argparse
import json

import numpy as np
import torch

from conduction_provenance import provenance
import two_layer_conduction
from tensorfem import thermal
from tensorfvm.cht import rod


def measure(n, expanding, steps_per_cell=2, *, time_steps=None, method="theta", fem_method="theta", first_step_study=False, interior_study=False):
    m, dx, decay = n//2, 1/n, 1.7
    k, c = np.array([2.,5.]), np.array([3.,1.5])
    area = np.array([.7,2.4]) if expanding else np.array([2.4,.7])
    h,cw,resistance = 1.8,.6,.24
    exposed = abs(np.diff(area)).item()
    q = np.array([1.,-.3])
    bridge = q[0] if expanding else q[1]
    left_trace = .8
    right_trace = left_trace-resistance*bridge/min(area)
    surface = right_trace if expanding else left_trace
    ambient = surface+((q[1]-q[0])/exposed-decay*cw*surface)/h
    wave = np.sqrt(decay*c/k)

    def profile(x,j):
        distance = x-.5
        return ((left_trace,right_trace)[j]*np.cos(wave[j]*distance)
                -q[j]*np.sin(wave[j]*distance)/(k[j]*area[j]*wave[j]))

    x = np.arange(n+1)*dx
    nodes = np.r_[profile(x[:m+1],0),profile(x[m:],1)]
    xc = (np.arange(n)+.5)*dx
    means = np.r_[profile(xc[:m],0),profile(xc[m:],1)]*np.sinc(wave.repeat(m)*dx/(2*np.pi))
    times = np.linspace(0.,1.,(steps_per_cell*n if time_steps is None else time_steps)+1)
    dt = np.diff(times)
    signal = np.exp(-decay*times)
    exact_mean = signal[:-1]*(-np.expm1(-decay*dt))/(dt*decay)

    contacts = np.zeros(n-1)
    contacts[m-1] = resistance
    def advance_fvm(grid, integrator):
        return rod.solve_transient_stepped_rod(np.full(n,dx),area.repeat(m),k.repeat(m),c.repeat(m),
                lambda t:nodes[[0,-1]]*np.exp(-decay*t),grid,means,flux_order=2,film_coefficient=h,
                ambient_temperature=lambda t:ambient*np.exp(-decay*t),shoulder_areal_heat_capacity=cw,
                initial_shoulder_temperature=surface,contact_resistance=contacts,return_traces=True,
                method=integrator,return_step_heat=True,return_interval_temperatures=True)

    if first_step_study:
        width = dt[0]
        target = q*exact_mean[0]
        def first_power(count, integrator):
            heat = advance_fvm(np.linspace(0.,width,count+1),integrator)[-2]
            return np.array([heat["right"][:,m-1].sum(),heat["left"][:,m].sum()])/width

        # Resolve the same discrete initial-value problem on the original
        # first interval. Extrapolate the integrated power, not its peak.
        ref = [first_power(count,"sdirk2") for count in (32,64,128)]
        extrapolated = ref[-1]+(ref[-1]-ref[-2])/3
        previous = ref[-2]+(ref[-2]-ref[-3])/3
        # Crank-Nicolson uses different stages; check the SDIRK2 estimate
        # against its own independently refined and extrapolated integral.
        cn64,cn128 = [first_power(count,"theta") for count in (64,128)]
        cn_reference = cn128+(cn128-cn64)/3
        uncertainty = np.maximum(abs(extrapolated-previous),abs(extrapolated-cn_reference))
        power = first_power(1,method)
        for side in (0,1):
            total = power[side]-target[side]
            spatial = extrapolated[side]-target[side]
            temporal = power[side]-extrapolated[side]
            yield dict(n=n,expanding=expanding,material=("left","right")[side],interval_s=float(width),
                       continuum_W=float(target[side]),refined_FVM_W=float(extrapolated[side]),
                       spatial_estimate_W=float(spatial),
                       temporal_estimate_W=float(temporal),total_error_W=float(total),
                       reference_disagreement_W=float(uncertainty[side]),
                       reference_resolved=bool(uncertainty[side]<max(1e-10,.01*min(abs(total),abs(spatial),abs(temporal)))))
        return
    fv,faces,_,_,(up,down),step_heat,fv_mean = advance_fvm(times,method)
    traces_fv = np.column_stack((fv_mean["trace_left"][:,m-1],fv_mean["trace_right"][:,m-1]))
    histories = [("FVM",traces_fv,fv,faces[:,m],up[:,m-1]-down[:,m-1],
                  np.column_stack((step_heat["right"][:,m-1],step_heat["left"][:,m])),
                  np.column_stack((step_heat["left"][:,0],-step_heat["right"][:,-1])))]

    model,_ = two_layer_conduction.slab_fem_model(k.repeat(m),nodes[0],nodes[-1],
                     resistance/min(area),None,torch.from_numpy(c.repeat(m)))
    side = m+int(expanding)
    shoulder = torch.tensor([side,side+n+2])
    model = replace(model,thickness=torch.from_numpy(area.repeat(m)),convection_edges=shoulder[None,:],
                    convection_area=exposed,convection_coefficient=h,ambient_temperature=0.,
                    wall_areal_heat_capacity=cw)
    load = model.nodes.new_zeros(len(model.nodes))
    load[shoulder] = ambient*h*exposed/2
    fixed = torch.from_numpy(nodes[[0,0,-1,-1]])
    result = thermal.solve_transient_thermal(model,torch.from_numpy(times),torch.from_numpy(np.tile(nodes,2)),
                 theta=.5,method=fem_method,sparse=True,nodal_heat_history=lambda t:load*torch.exp(-decay*t),
                 fixed_temperature_history=lambda t:fixed*torch.exp(-decay*t))
    t = result.temperature.numpy().reshape(len(times),2,n+2).mean(axis=1)
    cells = np.column_stack([.5*(s[:,1:]+s[:,:-1]) for s in (t[:,:m+1],t[:,m+1:])])
    local = thermal.thermal_element_nodal_heat_steps(model,result).numpy()
    mean_nodes = result.interval_temperature.numpy().reshape(len(dt),2,n+2).mean(axis=1)
    if interior_study:
        fractions = np.array([.13,.38,.5,.73,.91])
        points = x[:-1,None]+dx*fractions
        positions = points-.5
        ref_temperature = np.vstack((profile(points[:m],0),profile(points[m:],1)))
        # Differentiate each layer's cosine/sine solution in physical x.
        ka = (k*area).repeat(m)[:,None]
        w = wave.repeat(m)[:,None]
        ref_power = (ka*w*np.repeat([left_trace,right_trace],m)[:,None]*np.sin(w*positions)
                     +q.repeat(m)[:,None]*np.cos(w*positions))
        west = np.column_stack((mean_nodes[:,:m],mean_nodes[:,m+1:-1]))
        east = np.column_stack((mean_nodes[:,1:m+1],mean_nodes[:,m+2:]))
        for solver in ("FVM","FEM"):
            if solver=="FVM":
                field,power = rod.reconstruct_interval_rod_field(np.full(n,dx),area.repeat(m),
                    k.repeat(m),fv_mean,fractions)
            else:
                field = west[:,:,None]+(east-west)[:,:,None]*fractions
                power = -(east-west)[:,:,None]*ka/dx
            error = abs(power-exact_mean[:,None,None]*ref_power)
            row = dict(solver=solver,n=n,expanding=expanding,steps=len(dt),
                       interior_temperature_error_K=float(abs(field-exact_mean[:,None,None]*ref_temperature).max()),
                       interior_power_error_W=float(error.max()),midpoint_power_error_W=float(error[:,:,2].max()))
            if solver=="FEM":
                # Integrate the two edge nodal loads, both positive rightward.
                # A linear interpolation gives constant divergence in the element
                # and preserves its integrated heat balance.
                west_power = (local[:,:,0]+local[:,:,3])/dt[:,None]
                east_power = -(local[:,:,1]+local[:,:,2])/dt[:,None]
                recovered = west_power[:,:,None]+(east_power-west_power)[:,:,None]*fractions
                row["equilibrated_power_error_W"] = float(abs(recovered-exact_mean[:,None,None]*ref_power).max())
                residual = (dt[:,None]*(east_power-west_power)
                            +np.diff(cells,axis=0)*dx*(area*c).repeat(m))
                row["cell_balance_J"] = float(abs(residual).max())
            yield row
        return
    traces_fe = mean_nodes[:,m:m+2]
    histories.append(("FEM",traces_fe,cells,t[:,side],t[:,m]-t[:,m+1],
                      np.column_stack((-local[:,m-1,[1,2]].sum(axis=1),local[:,m,[0,3]].sum(axis=1))),
                      np.column_stack((local[:,0,[0,3]].sum(axis=1),local[:,-1,[1,2]].sum(axis=1)))))

    gamma = 1.-1./np.sqrt(2.) if fem_method=="sdirk2" else .5
    first_times = times[:-1]+gamma*dt if fem_method=="sdirk2" else times[:-1]
    fem_drive = (1-gamma)*np.exp(-decay*first_times)+gamma*signal[1:]
    for name, traces, cell, wall, jump, heat, end_heat in histories:
        exterior = (step_heat["ambient"][:,m-1] if name=="FVM" else
                    dt*h*exposed*(ambient*fem_drive-traces[:,int(expanding)]))
        wall_residual = cw*exposed*np.diff(wall)-heat[:,0]+heat[:,1]-exterior
        stored = cell.reshape(len(times),2,m).sum(axis=-1)*dx*c*area
        layer_residual = np.diff(stored,axis=0)-end_heat+heat*[1.,-1.]
        trace_heat = dt*min(area)/resistance*(traces[:,0]-traces[:,1])
        link_heat = trace_heat
        if name == "FVM":
            link_heat = step_heat["contact"][:,m-1]
        link_residual = link_heat-heat[:,0 if expanding else 1]
        trace_errors = abs(traces-exact_mean[:,None]*[left_trace,right_trace]).max(axis=0)
        signed = heat/dt[:,None]-exact_mean[:,None]*q
        step,side = np.unravel_index(abs(signed).argmax(),signed.shape)
        yield dict(peak_step=int(step+1),peak_material=("left","right")[side],
                   peak_interval_s=times[step:step+2].tolist(),peak_signed_W=float(signed[step,side]),
                   solver=name,n=n,steps=len(dt),expanding=expanding,temperature_error_K=float(abs(cell-signal[:,None]*means).max()),
                   mean_left_trace_error_K=float(trace_errors[0]),mean_right_trace_error_K=float(trace_errors[1]),
                   contact_jump_error_K=float(abs(jump-signal*(left_trace-right_trace)).max()),
                   shoulder_temperature_error_K=float(abs(wall-signal*surface).max()),
                   cumulative_material_heat_error_J=float(abs((heat-dt[:,None]*exact_mean[:,None]*q).cumsum(axis=0)).max()),
                   step_power_error_W=float(abs(signed[step,side])),
                   max_step_balance_J=float(max(abs(wall_residual).max(),abs(layer_residual).max(),abs(link_residual).max(),abs(trace_heat-link_heat).max())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps-per-cell",type=int,default=2,help="number of time steps per axial cell")
    parser.add_argument("--method",choices=("theta","sdirk2","sdirk2_start","rannacher"),default="theta",help="FVM time integrator")
    parser.add_argument("--fem-method",choices=("theta","sdirk2"),default="theta",help="FEM time integrator")
    parser.add_argument("--fixed-steps",type=int,help="hold time resolution fixed while refining the spatial grid")
    parser.add_argument("--fixed-cells",type=int,help="hold an even number of cells fixed while refining time")
    study = parser.add_mutually_exclusive_group()
    study.add_argument("--interior-study",action="store_true",help="check interval-mean interior fields at five local positions per cell")
    study.add_argument("--first-step-study",action="store_true",
                        help="estimate spatial and temporal errors in the two FVM material powers over the first interval")
    args = parser.parse_args()
    if args.steps_per_cell < 1:
        parser.error("steps per cell must be positive")
    if args.fixed_steps is not None and args.fixed_cells is not None:
        parser.error("select one fixed resolution")
    if args.fixed_steps is not None and args.fixed_steps < 1:
        parser.error("fixed steps must be positive")
    if args.fixed_cells is not None and (args.fixed_cells < 2 or args.fixed_cells%2):
        parser.error("fixed cells must be even and at least two")
    sensitivity = args.fixed_steps is not None or args.fixed_cells is not None
    rows = []
    for expanding in (True,False):
        previous = {}
        for size in (16,32,64,128,256,512):
            n = args.fixed_cells or size
            steps = args.fixed_steps or size*args.steps_per_cell
            for row in measure(n,expanding,time_steps=steps,method=args.method,fem_method=args.fem_method,
                               first_step_study=args.first_step_study,interior_study=args.interior_study):
                if args.first_step_study:
                    row["passed"] = row["reference_resolved"]
                    if args.fixed_steps is not None:
                        side = row["material"]
                        spatial = abs(row["spatial_estimate_W"])
                        if side in previous:
                            rate = float(np.log2(previous[side]/spatial)) if spatial else None
                            row["fixed_window_spatial_order"] = rate
                            row["passed"] &= rate is not None and rate>1.8
                        previous[side] = spatial
                    rows.append(row)
                    continue
                name = row["solver"]
                errors = {k:v for k,v in row.items() if "_error_" in k}
                rates = {} if name not in previous else {k:float(np.log2(previous[name][k]/v)) for k,v in errors.items()}
                row["orders"] = rates
                if args.interior_study:
                    # Linear FEM temperature gives constant axial power in each element.
                    row["passed"] = all(np.isfinite(v) for v in errors.values()) and row.get("cell_balance_J",0.)<1e-10
                    if not sensitivity:
                        row["passed"] &= all(v>(.9 if name=="FEM" and k=="interior_power_error_W" else 1.8)
                                            for k,v in rates.items())
                    rows.append(row)
                    previous[name] = errors
                    continue
                if not sensitivity and rates:
                    row["contact_mean_order_passed"] = min(rates["mean_left_trace_error_K"],
                                                           rates["mean_right_trace_error_K"])>1.8
                row["balance_passed"] = row["max_step_balance_J"]<1e-10
                if not sensitivity:
                    row["passed"] = row["balance_passed"] and all(v>1.8 for v in rates.values())
                rows.append(row)
                previous[name] = errors
    print(json.dumps(dict(runs=rows,interior_study=args.interior_study,first_step_study=args.first_step_study,fvm_method=args.method,fem_method=args.fem_method,fixed_cells=args.fixed_cells,fixed_steps=args.fixed_steps,sources=[provenance(f) for f in
                     (__file__,two_layer_conduction.__file__,rod.__file__,thermal.__file__)]),indent=2,allow_nan=False))
    raise SystemExit(0 if all((row["passed"] if "passed" in row else row["balance_passed"]) for row in rows) else 1)
