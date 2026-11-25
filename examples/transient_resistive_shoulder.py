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


def measure(n, expanding, steps_per_cell=2, *, time_steps=None, method="theta"):
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
    exact_mean = -np.diff(signal)/(dt*decay)
    def integrated(v):
        return dt[:,None]*(v[:-1]+v[1:])/2

    contacts = np.zeros(n-1)
    contacts[m-1] = resistance
    fv,faces,_,_,(up,down),step_heat = rod.solve_transient_stepped_rod(np.full(n,dx),area.repeat(m),
                        k.repeat(m),c.repeat(m),lambda t:nodes[[0,-1]]*np.exp(-decay*t),times,means,
                        flux_order=2,film_coefficient=h,ambient_temperature=lambda t:ambient*np.exp(-decay*t),
                        shoulder_areal_heat_capacity=cw,initial_shoulder_temperature=surface,
                        contact_resistance=contacts,return_traces=True,method=method,return_step_heat=True)
    histories = [("FVM",fv,faces[:,m],up[:,m-1]-down[:,m-1],
                  np.column_stack((step_heat["right"][:,m-1],step_heat["left"][:,m])),
                  np.column_stack((step_heat["left"][:,0],-step_heat["right"][:,-1])))]

    model,_ = two_layer_conduction.slab_fem_model(k.repeat(m),nodes[0],nodes[-1],
                     resistance/min(area),None,torch.from_numpy(c.repeat(m)))
    side = m+int(expanding)
    shoulder = torch.tensor([side,side+n+2])
    model = replace(model,thickness=torch.from_numpy(area.repeat(m)),convection_edges=shoulder[None,:],
                    convection_area=exposed,convection_coefficient=h,ambient_temperature=0.,
                    wall_areal_heat_capacity=cw)
    loads = torch.zeros((len(times),len(model.nodes)),dtype=torch.float64)
    loads[:,shoulder] = torch.from_numpy((signal*ambient*h*exposed/2)[:,None])
    result = thermal.solve_transient_thermal(model,torch.from_numpy(times),torch.from_numpy(np.tile(nodes,2)),
                 theta=.5,sparse=True,nodal_heat_history=loads,
                 fixed_temperature_history=torch.from_numpy(signal[:,None]*nodes[[0,0,-1,-1]]))
    t = result.temperature.numpy().reshape(len(times),2,n+2).mean(axis=1)
    cells = np.column_stack([.5*(s[:,1:]+s[:,:-1]) for s in (t[:,:m+1],t[:,m+1:])])
    local = thermal.thermal_element_nodal_heat_steps(model,result).numpy()
    histories.append(("FEM",cells,t[:,side],t[:,m]-t[:,m+1],
                      np.column_stack((-local[:,m-1,[1,2]].sum(axis=1),local[:,m,[0,3]].sum(axis=1))),
                      np.column_stack((local[:,0,[0,3]].sum(axis=1),local[:,-1,[1,2]].sum(axis=1)))))

    for name, cell, wall, jump, heat, end_heat in histories:
        exterior = integrated((h*exposed*(ambient*signal-wall))[:,None])[:,0]
        if name == "FVM":
            exterior = step_heat["ambient"][:,m-1]
        wall_residual = cw*exposed*np.diff(wall)-heat[:,0]+heat[:,1]-exterior
        stored = cell.reshape(len(times),2,m).sum(axis=-1)*dx*c*area
        layer_residual = np.diff(stored,axis=0)-end_heat+heat*[1.,-1.]
        link_heat = integrated((jump*min(area)/resistance)[:,None])[:,0]
        if name == "FVM":
            link_heat = step_heat["contact"][:,m-1]
        link_residual = link_heat-heat[:,0 if expanding else 1]
        signed = heat/dt[:,None]-exact_mean[:,None]*q
        step,side = np.unravel_index(abs(signed).argmax(),signed.shape)
        yield dict(peak_step=int(step+1),peak_material=("left","right")[side],
                   peak_interval_s=times[step:step+2].tolist(),peak_signed_W=float(signed[step,side]),
                   solver=name,n=n,steps=len(dt),expanding=expanding,temperature_error_K=float(abs(cell-signal[:,None]*means).max()),
                   contact_jump_error_K=float(abs(jump-signal*(left_trace-right_trace)).max()),
                   shoulder_temperature_error_K=float(abs(wall-signal*surface).max()),
                   step_power_error_W=float(abs(signed[step,side])),
                   max_step_balance_J=float(max(abs(wall_residual).max(),abs(layer_residual).max(),abs(link_residual).max())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps-per-cell",type=int,default=2,help="number of time steps per axial cell")
    parser.add_argument("--method",choices=("theta","sdirk2","rannacher"),default="theta",help="FVM time integrator")
    parser.add_argument("--fixed-steps",type=int,help="hold time resolution fixed while refining the spatial grid")
    parser.add_argument("--fixed-cells",type=int,help="hold an even number of cells fixed while refining time")
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
            for row in measure(n,expanding,time_steps=steps,method=args.method):
                name = row["solver"]
                errors = {k:v for k,v in row.items() if "_error_" in k}
                rates = {} if name not in previous else {k:np.log2(previous[name][k]/v) for k,v in errors.items()}
                row["orders"] = rates
                row["balance_passed"] = row["max_step_balance_J"]<1e-10
                if not sensitivity:
                    row["passed"] = row["balance_passed"] and all(v>1.8 for v in rates.values())
                rows.append(row)
                previous[name] = errors
    print(json.dumps(dict(runs=rows,fvm_method=args.method,fixed_cells=args.fixed_cells,fixed_steps=args.fixed_steps,sources=[provenance(f) for f in
                     (__file__,two_layer_conduction.__file__,rod.__file__,thermal.__file__)]),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row.get("passed",row["balance_passed"]) for row in rows) else 1)
