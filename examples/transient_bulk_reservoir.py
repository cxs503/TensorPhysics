"""Slab exchanging heat with a spatially nonuniform, switched bulk reservoir."""
import argparse
import json

import numpy as np
import torch

from conduction_provenance import provenance
from tensordg import heat_1d
from tensorfem import thermal
from tensorfvm.cht import rod
from tensorsph import heat_conduction_2d as sph


K, C, H = .8, 1.6, 4.
TIMES = np.array([0.,.045,.12,.21])
# Constant and cosine reservoir amplitudes at the two ends of each time interval.
DRIVES = np.array([[[1.,.8],[1.6,.2]], [[-.7,.9],[-.4,-.6]], [[.5,-.2],[.5,-.2]]])
INITIAL = np.array([.2,-.3])


def extra_field(t,x,x2):
    return t*x+(t+.4)*x2


def extra_reservoir(t,x,x2):
    return extra_field(t,x,x2)+(C*(x+x2)-2*K*(t+.4))/H


def end_powers(t):
    return (-K*t,K*(3*t+.8))


def reference(n, particle_centers=False, driven=False):
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
    temperatures,sections,exchange = map(np.array,(temperatures,sections,exchange))
    if driven:
        mid = (faces[:-1]+faces[1:])/2
        mean_x2 = mid**2+1/(12*n*n)
        temperatures += extra_field(TIMES[:,None],mid,mid**2 if particle_centers else mean_x2)
        dt = np.diff(TIMES)[:,None]
        t = (TIMES[:-1]+TIMES[1:])[:,None]/2
        sections -= dt*K*(t+2*(t+.4)*faces)
        exchange += dt*(C*(mid+mean_x2)-2*K*(t+.4))/n
    return temperatures,sections,exchange


def cell_solver(n,particles,driven):
    faces = np.linspace(0.,1.,n+1)
    cosine = np.diff(np.sin(np.pi*faces))*n/np.pi
    if particles:
        cosine = np.cos(np.pi*(faces[:-1]+faces[1:])/2)
    mid = (faces[:-1]+faces[1:])/2
    x2 = mid**2 if particles else mid**2+1/(12*n*n)
    count = n
    times = np.r_[0.,np.concatenate([np.linspace(a,b,count+1)[1:] for a,b in zip(TIMES,TIMES[1:])])]
    def ambient(t):
        i = np.searchsorted(TIMES[1:-1],t,side="right")
        begin,end = DRIVES[i]
        amplitudes = begin+(end-begin)*(t-TIMES[i])/(TIMES[i+1]-TIMES[i])
        return amplitudes[0]+amplitudes[1]*cosine+(extra_reservoir(t,mid,x2) if driven else 0.)
    initial = INITIAL[0]+INITIAL[1]*cosine+(extra_field(0.,mid,x2) if driven else 0.)
    if particles:
        u,sections,exchange = sph.solve_transient_layered_slab(initial,times,1/n,(None,None),
            conductivity=np.full(n,K),volumetric_heat_capacity=C,method="sdirk2",heat_loss_coefficient=H,
            bulk_ambient_temperature=ambient,load_jump_times=TIMES[1:-1],
            x_wall_inward_flux=(lambda t: end_powers(t)[0],lambda t: end_powers(t)[1]) if driven else (0.,0.),
            return_x_heat=True,return_bulk_exchange=True)
    else:
        u,_,_,_,heat = rod.solve_transient_stepped_rod(np.diff(faces),1.,K,C,end_powers if driven else [0.,0.],times,
            initial,end_types=("power","power"),method="sdirk2",flux_order=2,
            heat_loss_coefficient=H,bulk_ambient_temperature=ambient,load_jump_times=TIMES[1:-1],return_step_heat=True)
        sections = np.column_stack((heat["left"][:,0],heat["right"]))
        exchange = heat["bulk_exchange"]
    def total(v):
        return np.asarray(v).reshape(len(DRIVES),count,-1).sum(axis=1)
    return np.asarray(u)[::count],total(sections),total(exchange)


def fem(n,driven):
    x = torch.linspace(0.,1.,n+1,dtype=torch.float64)
    cosine = torch.cos(torch.pi*x)
    conn = torch.stack((torch.arange(n),torch.arange(1,n+1)),dim=1)
    model = thermal.ThermalModel(x[:,None],conn,K,1.,C,torch.empty(0,dtype=torch.long),x[:0],
                                 heat_loss_coefficient=H)
    capacity = thermal.assemble_thermal(model,sparse=True).capacity
    state = INITIAL[0]+INITIAL[1]*cosine+(extra_field(0.,x,x*x) if driven else 0.)
    fields,sections,exchange = [state[conn].mean(dim=1).numpy()],[],[]
    element_mass = x.new_tensor([[2.,1.],[1.,2.]])/(6*n)
    for i,(dt,(begin,end)) in enumerate(zip(np.diff(TIMES),DRIVES)):
        times = torch.linspace(0.,dt,n+1,dtype=x.dtype)
        values = torch.from_numpy(begin)+times[:,None]*torch.from_numpy(end-begin)/dt
        ambient = values[:,:1]+values[:,1:]*cosine
        if driven:
            ambient += extra_reservoir(TIMES[i]+times[:,None],x,x*x)
        power = (H/C)*(capacity @ ambient.T).T
        if driven:
            power[:,[0,-1]] += torch.stack(end_powers(TIMES[i]+times),dim=1)
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


def dg(n,driven):
    def profile(amplitudes,t):
        return lambda x: (amplitudes[0]+amplitudes[1]*torch.cos(torch.pi*x)
                          +(extra_reservoir(t,x,x*x) if driven else 0.))
    pairs = [(profile(a,t0),profile(b,t1)) for (a,b),t0,t1 in zip(DRIVES,TIMES,TIMES[1:])]
    powers = [[end_powers(t0),end_powers(t1)] for t0,t1 in zip(TIMES,TIMES[1:])] if driven else None
    def initial(x):
        return INITIAL[0]+INITIAL[1]*torch.cos(torch.pi*x)+(extra_field(0.,x,x*x) if driven else 0.)
    _,states,heat,exchange = heat_1d.solve_heat_1d_bulk_history(TIMES,pairs,
        ncells=n,degree=2,capacity=C,conductivity=K,heat_loss_coefficient=H,
        initial_temperature=initial,steps_per_interval=n,inward_flux_intervals=powers)
    return states[:,:,0].numpy(),heat.numpy(),exchange.numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driven-ends",action="store_true",help="add independent ramped end powers and a quadratic temperature field")
    driven = parser.parse_args().driven_ends
    rows,previous = [],{}
    for n in (8,16,32,64):
        for name,solve in (("DG_P2",dg),("FEM",fem),("FVM",lambda n,d: cell_solver(n,False,d)),
                           ("SPH",lambda n,d: cell_solver(n,True,d))):
            exact = reference(n,particle_centers=name=="SPH",driven=driven)
            u,cut,exchange = solve(n,driven)
            # Report exchanged heat per unit cell volume so mesh refinement alone
            # does not improve the metric by shrinking the integration region.
            errors = dict(cell_temperature_K=float(abs(u-exact[0]).max()),
                          section_heat_J_m2=float(abs(cut-exact[1]).max()),
                          reservoir_heat_J_m3=float(n*abs(exchange-exact[2]).max()))
            residual = float(abs((C/n)*np.diff(u,axis=0)+np.diff(cut)-exchange).max())
            wall_error = float(abs(cut[:,[0,-1]]-exact[1][:,[0,-1]]).max())
            orders = {k:float(np.log2(previous[name][k]/v)) for k,v in errors.items()} if name in previous else {}
            rows.append(dict(solver=name,cells=n,temperature_sampling="center" if name=="SPH" else "cell_mean",errors=errors,orders=orders,max_cell_balance_J_m2=residual,
                             wall_heat_error_J_m2=wall_error,passed=wall_error<1e-11 and residual<1e-11 and all(p>1.8 for p in orders.values())))
            previous[name] = errors
    print(json.dumps(dict(driven_ends=driven,times=TIMES.tolist(),ambient_modes=DRIVES.tolist(),runs=rows,
                          sources=[provenance(p) for p in (__file__,heat_1d.__file__,thermal.__file__,rod.__file__,sph.__file__)]),
                     indent=2,allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
