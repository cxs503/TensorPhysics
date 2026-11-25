"""Steady through-section power with two flux ends, a resistive contact and bulk exchange."""
import json

import numpy as np
import torch

from conduction_provenance import provenance
from tensorfem import thermal
from tensorfvm.cht import rod


def measure(n, bulk):
    resistance = .32  # total contact resistance in K/W
    areas,ks = [.8,1.4],[1.7,3.1]
    capacities = [1.6,.9]
    f = np.array([0.,.17,.46,.83,1.])
    g,w = np.polynomial.legendre.leggauss(8)
    shape = np.column_stack(((1-g)/2,(1+g)/2))
    nodes,tnodes,means,source,ambient,loads,temps,powers = [],[],[],[],[],[],[],[]
    start_temp,start_power = 1.2,.55
    for j,(length,a,k) in enumerate(zip((.4,.6),areas,ks)):
        freq,b = [(4.,.38),(6.,-.26)][j]
        h = [2.3,.9][j] if bulk else 0.
        x = length*np.linspace(0.,1.,n+1)**1.25
        dx = np.diff(x)
        def power(z):
            return start_power+b*(np.cos(freq*z)-1)
        def temp(z):
            return start_temp-((start_power-b)*z+b*np.sin(freq*z)/freq)/(k*a)
        probes = x[:-1,None]+dx[:,None]*f
        quad = x[:-1,None]+dx[:,None]*(1+g)/2
        air = .4+.3*np.sin(2*quad)
        net = -b*freq*np.sin(freq*quad)/a
        # Applied generation is net generation minus exchange. The FEM
        # right-hand side includes the reservoir load, integrated before assembly.
        applied = net-h*(air-temp(quad))
        loads.append(a*dx[:,None]/2*(((applied+h*air)*w)@shape))
        source.extend((applied @ w)/2)
        ambient.extend((air @ w)/2)
        means.extend((temp(quad) @ w)/2)
        nodes.append(x+(0. if j==0 else .4))
        tnodes.extend(temp(x))
        temps.extend(temp(probes))
        powers.extend(power(probes))
        end_temp,end_power = temp(length),power(length)
        start_temp,start_power = end_temp-resistance*end_power,end_power

    area,k,c = [np.repeat(v,n) for v in (areas,ks,capacities)]
    h = np.repeat([2.3,.9],n) if bulk else np.zeros(2*n)
    widths = np.concatenate([np.diff(x) for x in nodes])
    breaks = np.zeros(2*n-1)
    breaks[n-1] = resistance*min(area)
    contact_jump = tnodes[n]-tnodes[n+1]
    wall = [.55,-end_power]
    datum = None if bulk else np.average(means,weights=widths*c*area)
    cell,_,west,east,(hot,cold),(grad,bend) = rod.solve_steady_stepped_rod(widths,area,k,c,wall,
        source=source,mean_temperature=datum,end_types=("power","power"),return_traces=True,
        flux_order=2,contact_resistance=breaks,heat_loss_coefficient=h,bulk_ambient_temperature=ambient,
        source_derivative_breaks=[n],return_source_derivatives=True)
    field,p = rod.reconstruct_steady_rod_field(widths,area,k,cell,west,east,f,source_gradient=grad,source_curvature=bend)
    res = east-west-widths*area*(source+h*(ambient-cell))
    yield "FVM",dict(field_K=float(abs(field-temps).max()),power_W=float(abs(p-powers).max()),
                     jump_K=float(abs(hot[n-1]-cold[n-1]-contact_jump))),float(abs(res).max())

    # Use the known left-end temperature for the FEM datum in the floating case.
    xy = torch.from_numpy(np.concatenate(nodes)[:,None])
    starts = np.r_[np.arange(n),n+1+np.arange(n)]
    conn = np.column_stack((starts,starts+1))
    fixed = torch.tensor([] if bulk else [0],dtype=torch.long)
    rhs = np.zeros(len(xy))
    local_load = np.vstack(loads)
    np.add.at(rhs,conn,local_load)
    rhs[[0,-1]] += wall
    model = thermal.ThermalModel(xy,torch.from_numpy(conn),torch.from_numpy(k),1.,torch.from_numpy(c),
        fixed,torch.tensor(tnodes,dtype=xy.dtype)[fixed],thickness=torch.from_numpy(area),nodal_heat=torch.from_numpy(rhs),
        contact_pairs=torch.tensor([[n,n+1]]),contact_conductance=1/resistance,heat_loss_coefficient=torch.from_numpy(h))
    u = thermal.solve_steady_thermal(model,sparse=True)
    local,net = thermal.thermal_element_nodal_power(model,u,element_nodal_heat=torch.from_numpy(local_load),
                                               return_net_input=True)
    element = u.numpy()[conn]
    section = thermal.thermal_element_section_power(model,local,f,
        net_element_nodal_input=net).real.numpy()
    linear = thermal.thermal_element_section_power(model,local,f).real.numpy()
    field = element[:,:1]+np.diff(element,axis=1)*f
    res = section[:,-1]-section[:,0]+widths*area*h*element.mean(1)-local_load.sum(1)
    yield "FEM",dict(field_K=float(abs(field-temps).max()),power_W=float(abs(section-powers).max()),
                     linear_power_W=float(abs(linear-powers).max()),jump_K=float(abs(u[n]-u[n+1]-contact_jump))),float(abs(res).max())


if __name__=="__main__":
    rows = []
    for bulk in (False,True):
        previous = {}
        for n in (4,8,16,32,64):
            for solver,errors,balance in measure(n,bulk):
                orders = {k:float(np.log2(previous[solver][k]/v)) for k,v in errors.items()
                          if v>1e-10} if solver in previous else {}
                rows.append(dict(solver=solver,bulk_exchange=bulk,cells_per_material=n,errors=errors,
                                 orders=orders,balance_W=balance,passed=balance<1e-10 and
                                 all(v>(1.8 if solver=="FEM" else 3.5) for v in orders.values())))
                previous[solver] = errors
    print(json.dumps(dict(runs=rows,sources=[provenance(p) for p in (__file__,rod.__file__,thermal.__file__)]),
                     indent=2,allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
