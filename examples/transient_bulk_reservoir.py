"""Insulated slab exchanging heat with a spatially nonuniform, switched bulk reservoir."""
import json

import numpy as np
import torch

from conduction_provenance import provenance
from tensorfem import thermal
from tensorfvm.cht import rod
from tensorsph import heat_conduction_2d as sph


K, C, H = .8, 1.6, 4.
TIMES = np.array([0.,.045,.12,.21])
# Constant and cosine reservoir amplitudes at the two ends of each time interval.
DRIVES = np.array([[[1.,.8],[1.6,.2]], [[-.7,.9],[-.4,-.6]], [[.5,-.2],[.5,-.2]]])
INITIAL = np.array([.2,-.3])


def reference(n, particle_centers=False):
    faces = np.linspace(0.,1.,n+1)
    cosine = np.diff(np.sin(np.pi*faces))*n/np.pi
    temperature_cosine = np.cos(np.pi*(faces[:-1]+faces[1:])/2) if particle_centers else cosine
    modes = INITIAL.copy()
    temperatures = [modes[0]+modes[1]*temperature_cosine]
    sections,exchange = [],[]
    rate = (H+K*np.array([0.,np.pi**2]))/C
    for dt,(begin,end) in zip(np.diff(TIMES),DRIVES):
        slope = (end-begin)/dt
        decay = -np.expm1(-rate*dt)
        change = decay*(H*begin/(C*rate)-modes) + H*slope/(C*rate)*(dt-decay/rate)
        heat_input = .5*dt*(begin+end)
        integral = (H*heat_input/C-change)/rate
        modes += change
        temperatures.append(modes[0]+modes[1]*temperature_cosine)
        sections.append(K*np.pi*np.sin(np.pi*faces)*integral[1])
        reservoir_heat = H*(heat_input-integral)/n
        exchange.append(reservoir_heat[0]+reservoir_heat[1]*cosine)
    return np.array(temperatures),np.array(sections),np.array(exchange)


def cell_solver(n,particles):
    faces = np.linspace(0.,1.,n+1)
    cosine = np.diff(np.sin(np.pi*faces))*n/np.pi
    if particles:
        cosine = np.cos(np.pi*(faces[:-1]+faces[1:])/2)
    count = n
    times = np.r_[0.,np.concatenate([np.linspace(a,b,count+1)[1:] for a,b in zip(TIMES,TIMES[1:])])]
    def ambient(t):
        i = np.searchsorted(TIMES[1:-1],t,side="right")
        begin,end = DRIVES[i]
        amplitudes = begin+(end-begin)*(t-TIMES[i])/(TIMES[i+1]-TIMES[i])
        return amplitudes[0]+amplitudes[1]*cosine
    initial = INITIAL[0]+INITIAL[1]*cosine
    if particles:
        u,sections,exchange = sph.solve_transient_layered_slab(initial,times,1/n,(None,None),
            conductivity=np.full(n,K),volumetric_heat_capacity=C,method="sdirk2",heat_loss_coefficient=H,
            bulk_ambient_temperature=ambient,load_jump_times=TIMES[1:-1],return_x_heat=True,return_bulk_exchange=True)
    else:
        u,_,_,_,heat = rod.solve_transient_stepped_rod(np.diff(faces),1.,K,C,[0.,0.],times,
            initial,end_types=("power","power"),method="sdirk2",flux_order=2,
            heat_loss_coefficient=H,bulk_ambient_temperature=ambient,load_jump_times=TIMES[1:-1],return_step_heat=True)
        sections = np.column_stack((heat["left"][:,0],heat["right"]))
        exchange = heat["bulk_exchange"]
    def total(v):
        return np.asarray(v).reshape(len(DRIVES),count,-1).sum(axis=1)
    return np.asarray(u)[::count],total(sections),total(exchange)


def fem(n):
    x = torch.linspace(0.,1.,n+1,dtype=torch.float64)
    cosine = torch.cos(torch.pi*x)
    conn = torch.stack((torch.arange(n),torch.arange(1,n+1)),dim=1)
    model = thermal.ThermalModel(x[:,None],conn,K,1.,C,torch.empty(0,dtype=torch.long),x[:0],
                                 heat_loss_coefficient=H)
    capacity = thermal.assemble_thermal(model,sparse=True).capacity
    state = INITIAL[0]+INITIAL[1]*cosine
    fields,sections,exchange = [state[conn].mean(dim=1).numpy()],[],[]
    element_mass = x.new_tensor([[2.,1.],[1.,2.]])/(6*n)
    for dt,(begin,end) in zip(np.diff(TIMES),DRIVES):
        times = torch.linspace(0.,dt,n+1,dtype=x.dtype)
        values = torch.from_numpy(begin)+times[:,None]*torch.from_numpy(end-begin)/dt
        ambient = values[:,:1]+values[:,1:]*cosine
        power = (H/C)*(capacity @ ambient.T).T
        result = thermal.solve_transient_thermal(model,times,state,sparse=True,method="sdirk2",nodal_heat_history=power)
        element_input = H*(ambient[:,conn] @ element_mass)
        # For a linear drive the interval-mean ambient power is the endpoint average.
        element_heat = times.diff()[:,None,None]*(element_input[:-1]+element_input[1:])/2
        local = thermal.thermal_element_nodal_heat_steps(model,result,element_nodal_step_heat=element_heat).sum(dim=0)
        sections.append(torch.cat((local[:1,0],-local[:,1])).numpy())
        stored_loss = (H/n)*(times.diff() @ result.interval_temperature[:,conn].mean(dim=2))
        exchange.append((element_heat.sum(dim=(0,2))-stored_loss).numpy())
        state = result.temperature[-1]
        fields.append(state[conn].mean(dim=1).numpy())
    return np.asarray(fields),np.asarray(sections),np.asarray(exchange)


if __name__ == "__main__":
    rows,previous = [],{}
    for n in (8,16,32,64):
        for name,solve in (("FEM",fem),("FVM",lambda n: cell_solver(n,False)),
                           ("SPH",lambda n: cell_solver(n,True))):
            exact = reference(n,particle_centers=name=="SPH")
            u,cut,exchange = solve(n)
            # Report exchanged heat per unit cell volume so mesh refinement alone
            # does not improve the metric by shrinking the integration region.
            errors = dict(cell_temperature_K=float(abs(u-exact[0]).max()),
                          section_heat_J_m2=float(abs(cut-exact[1]).max()),
                          reservoir_heat_J_m3=float(n*abs(exchange-exact[2]).max()))
            residual = float(abs((C/n)*np.diff(u,axis=0)+np.diff(cut)-exchange).max())
            orders = {k:float(np.log2(previous[name][k]/v)) for k,v in errors.items()} if name in previous else {}
            rows.append(dict(solver=name,cells=n,temperature_sampling="center" if name=="SPH" else "cell_mean",errors=errors,orders=orders,max_cell_balance_J_m2=residual,
                             passed=residual<1e-11 and all(p>1.8 for p in orders.values())))
            previous[name] = errors
    print(json.dumps(dict(times=TIMES.tolist(),ambient_modes=DRIVES.tolist(),runs=rows,
                          sources=[provenance(p) for p in (__file__,thermal.__file__,rod.__file__,sph.__file__)]),
                     indent=2,allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
