"""Bulk and film exchange in a unit slab with two storing walls and moving reservoirs."""
import argparse
import json

import numpy as np
import torch
from conduction_provenance import provenance
from tensordg import heat_1d
from tensorfem import thermal
from tensorfvm.cht import rod
from tensorsph import heat_conduction_2d as sph


K,C,H = 1.4,2.,3.
WAVE,PEAK = 1.8,.38
RATE = (K*WAVE**2+H)/C
CW = np.array([.3,.8])
ENDS = np.array([0.,1.])
SIGN = np.array([1.,-1.])
FILM = K*WAVE*np.tan(WAVE*np.array([PEAK,1-PEAK]))+RATE*CW


def background(t,x):
    return .3+.2*x+t*(.4-.7*x)


def reservoir(t,x):
    return background(t,x)+C*(.4-.7*x)/H


def air(t):
    return background(t,ENDS)+(SIGN*(-K*(.2-.7*t))+CW*(.4-.7*ENDS))/FILM


def exact(times,n):
    faces = np.linspace(0.,1.,n+1)
    mid = .5*(faces[:-1]+faces[1:])
    mean_mode = np.cos(WAVE*(mid-PEAK))*np.sinc(WAVE/(2*np.pi*n))
    decay = np.exp(-RATE*times[:,None])
    u = background(times[:,None],mid)+decay*mean_mode
    walls = background(times[:,None],ENDS)+decay*np.cos(WAVE*(ENDS-PEAK))
    dt = np.diff(times)[:,None]
    t = .5*(times[:-1]+times[1:])[:,None]
    mode_integral = -np.diff(decay,axis=0)/RATE
    heat = dt*(-K*(.2-.7*t))+K*WAVE*np.sin(WAVE*(faces-PEAK))*mode_integral
    bulk = (dt*C*(.4-.7*mid)-H*mean_mode*mode_integral)/n
    external = heat[:,[0,-1]]*SIGN+CW*np.diff(walls,axis=0)
    return u,walls,heat,bulk,external


def dg(times,n):
    reservoirs = [[lambda x,t=t: reservoir(t,x) for t in (a,b)] for a,b in zip(times,times[1:])]
    film_history = [[air(a),air(b)] for a,b in zip(times,times[1:])]
    _,u,q,bulk,wall = heat_1d.solve_heat_1d_bulk_history(times,reservoirs,ncells=n,degree=2,
        initial_temperature=lambda x: background(0.,x)+torch.cos(WAVE*(x-PEAK)),conductivity=K,capacity=C,
        heat_loss_coefficient=H,steps_per_interval=1,film_coefficients=FILM,wall_areal_heat_capacity=CW,
        wall_ambient_intervals=film_history,return_wall_history=True)
    return tuple(v.numpy() for v in (u[:,:,0],wall["temperature"],q,bulk,wall["external_heat"]))


def fvm(times,n):
    mid = (np.arange(n)+.5)/n
    initial = background(0.,mid)+np.cos(WAVE*(mid-PEAK))*np.sinc(WAVE/(2*np.pi*n))
    cell,wall,_,_,heat = rod.solve_transient_stepped_rod(np.full(n,1/n),1.,K,C,air,times,initial,
        end_types=("robin","robin"),end_film_coefficient=FILM,end_areal_heat_capacity=CW,
        initial_end_temperature=background(0.,ENDS)+np.cos(WAVE*(ENDS-PEAK)),flux_order=2,method="sdirk2",
        heat_loss_coefficient=H,bulk_ambient_temperature=lambda t: reservoir(t,mid),return_step_heat=True)
    return cell,wall[:,[0,-1]],np.column_stack((heat["left"][:,0],heat["right"])),heat["bulk_exchange"],heat["end_input"]


def particles(times,n,method="sdirk2"):
    mid = (np.arange(n)+.5)/n
    films = tuple((FILM[i],lambda t,i=i: air(t)[i]) for i in range(2))
    states,heat,wall,bulk,external = sph.solve_transient_layered_slab(
        background(0.,mid)+np.cos(WAVE*(mid-PEAK)),times,1/n,(None,None),conductivity=np.full(n,K),
        volumetric_heat_capacity=C,heat_loss_coefficient=H,bulk_ambient_temperature=lambda t: reservoir(t,mid),
        method=method,x_wall_films=films,wall_areal_heat_capacity=CW,
        initial_wall_temperatures=background(0.,ENDS)+np.cos(WAVE*(ENDS-PEAK)),
        return_wall_temperatures=True,return_bulk_exchange=True,return_external_heat=True,return_x_heat=True)
    return tuple(v.numpy() for v in (states,wall,heat,bulk,external))


def fem(times,n):
    x = torch.linspace(0.,1.,n+1,dtype=torch.float64)
    conn = torch.stack((torch.arange(n),torch.arange(1,n+1)),dim=1)
    model = thermal.ThermalModel(x[:,None],conn,K,1.,C,conn.new_empty(0),x[:0],heat_loss_coefficient=H,
        convection_edges=conn.new_tensor([[0,0],[n,n]]),convection_coefficient=torch.from_numpy(FILM),
        wall_areal_heat_capacity=torch.from_numpy(CW),ambient_temperature=x.new_zeros(2))
    time = torch.from_numpy(times)
    ambient = reservoir(time[:,None],x)
    element_mass = x.new_tensor([[2.,1.],[1.,2.]])/(6*n)
    applied = H*(ambient[:,conn] @ element_mass)
    power = x.new_zeros((len(times),n+1))
    power[:,:-1] += applied[:,:,0]
    power[:,1:] += applied[:,:,1]
    fluid = torch.from_numpy(air(times[:,None]))
    power[:,[0,-1]] += fluid*torch.from_numpy(FILM)
    result = thermal.solve_transient_thermal(model,time,background(0.,x)+torch.cos(WAVE*(x-PEAK)),
                                             sparse=True,method="sdirk2",nodal_heat_history=power)
    dt = time.diff()[:,None]
    load_heat = dt[:,:,None]*(applied[:-1]+applied[1:])/2
    local = thermal.thermal_element_nodal_heat_steps(model,result,element_nodal_step_heat=load_heat)
    heat = torch.cat((local[:,:1,0],-local[:,:,1]),dim=1)
    bulk = load_heat.sum(2)-H/n*dt*result.interval_temperature[:,conn].mean(2)
    external = dt*torch.from_numpy(FILM)*((fluid[:-1]+fluid[1:])/2-result.interval_temperature[:,[0,-1]])
    return tuple(v.numpy() for v in (result.temperature[:,conn].mean(2),result.temperature[:,[0,-1]],heat,bulk,external))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exponential",action="store_true",help="also test SPH exponential integration on four fixed time intervals")
    methods = [("DG_P2",dg),("FEM_LINE2",fem),("FVM",fvm),("SPH",particles)]
    if parser.parse_args().exponential:
        methods.append(("SPH_exponential",lambda t,n: particles(t,n,method="exponential")))
    rows,previous = [],{}
    for n in (12,24,48,96):
        for name,solve in methods:
            steps = 4 if name=="SPH_exponential" else n
            times = np.linspace(0.,.3,steps+1)
            target = exact(times,n)
            u,wall,heat,bulk,external = solve(times,n)
            partition = {}
            if name=="SPH_exponential":
                uc,wc,qc,bc,ec = solve(times[[0,-1]],n)
                for key,difference in (("particle_K",uc[-1]-u[-1]),("wall_K",wc[-1]-wall[-1]),
                        ("section_J_m2",qc[0]-heat.sum(0)),("bulk_J_m2",bc[0]-bulk.sum(0)),
                        ("external_J_m2",ec[0]-external.sum(0))):
                    partition[key] = float(abs(difference).max())
            def cumulative_error(v,ref):
                return float(abs(np.cumsum(v-ref,axis=0)).max())
            ref_temperature = target[0]
            if name.startswith("SPH"):
                mid = (np.arange(n)+.5)/n
                ref_temperature = background(times[:,None],mid)+np.exp(-RATE*times[:,None])*np.cos(WAVE*(mid-PEAK))
            sample_key = "particle_temperature_K" if name.startswith("SPH") else "cell_mean_K"
            errors = dict(wall_temperature_K=float(abs(wall-target[1]).max()),
                          section_heat_J_m2=cumulative_error(heat,target[2]),bulk_heat_J_m3=n*cumulative_error(bulk,target[3]),
                          external_heat_J_m2=cumulative_error(external,target[4]))
            errors[sample_key] = float(abs(u-ref_temperature).max())
            balance = float(abs(C/n*np.diff(u,axis=0)+np.diff(heat)-bulk).max())
            surface_balance = float(abs(CW*np.diff(wall,axis=0)+heat[:,[0,-1]]*SIGN-external).max())
            orders = {k:float(np.log2(previous[name][k]/v)) for k,v in errors.items()} if name in previous else {}
            rows.append(dict(solver=name,temperature_sampling="center" if name.startswith("SPH") else "cell_mean",cells=n,steps=steps,errors=errors,orders=orders,cell_balance_J_m2=balance,
                             wall_balance_J_m2=surface_balance,**({"partition_error":partition} if partition else {}),
                             passed=max(balance,surface_balance,*partition.values())<1e-11
                             and all(v>1.7 for v in orders.values())))
            previous[name] = errors
    print(json.dumps(dict(runs=rows,sources=[provenance(p) for p in (__file__,heat_1d.__file__,thermal.__file__,rod.__file__,sph.__file__)]),
                     indent=2,allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
