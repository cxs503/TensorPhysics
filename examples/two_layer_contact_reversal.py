"""Free cooling of a two-material slab with a known contact-current reversal."""
import argparse
import inspect
import json
import math

import numpy as np
import torch
from scipy.optimize import brentq

from conduction_provenance import provenance
from two_layer_conduction import slab_fem_model
from tensordem.thermal import contact_chain
from tensorfem import thermal
from tensorfvm.cht import energy
from tensorsph import heat_conduction_2d as sph


K = (1.,3.)
C = (2.,.8)
FILMS = (2.,5.)
CONTACT = .2
REVERSAL = .035


def propagate(t,q,p,layer,distance):
    k,c = K[layer],C[layer]
    wave = p*math.sqrt(c/k)
    co,si = np.cos(wave*distance),np.sin(wave*distance)
    return t*co-q*si/(k*wave),q*co+k*wave*t*si


def modes(reversals):
    def traces(p):
        left,flux = propagate(1.,-FILMS[0],p,0,.5)
        return left-CONTACT*flux,flux
    def mismatch(p):
        t,q = propagate(*traces(p),p,1,.5)
        return q-FILMS[1]*t
    roots = []
    lo = 1e-7
    for hi in np.arange(.05,30.,.05):
        if mismatch(lo)*mismatch(hi)<0:
            roots.append(brentq(mismatch,lo,hi))
            if len(roots)==len(reversals)+1:
                break
        lo = hi
    # A sum of M exponentials with distinct real exponents has at most
    # M-1 zeros. Prescribe all of them by solving for the amplitudes.
    rates = np.square(roots)
    fluxes = np.array([traces(p)[1] for p in roots])
    constraints = np.exp(-np.outer(reversals,rates))*fluxes
    weights = np.r_[1.,np.linalg.solve(constraints[:,1:],-constraints[:,0])]
    if len(reversals)>1:
        weights /= np.max(abs(weights))
    return [(p,w,*traces(p)) for p,w in zip(roots,weights)]


def measure(n,reversals=(REVERSAL,)):
    dx,m = 1/n,n//2
    # Resolve the flux zero between interval midpoints on every mesh.
    time = np.linspace(0.,.1,4*n+1)
    dt = np.diff(time)
    k,c = np.repeat(K,m),np.repeat(C,m)
    edges = np.linspace(0.,1.,n+1)
    centers = (edges[1:]+edges[:-1])/2
    modes_data = modes(reversals)
    def exact(left,right,cell_average=False):
        columns = []
        for p,w,tr,q in modes_data:
            values = np.r_[propagate(1.,-FILMS[0],p,0,left)[0],propagate(tr,q,p,1,right-.5)[0]]
            if cell_average:
                values *= np.sinc(p*np.sqrt(c/k)*dx/(2*np.pi))
            columns.append((w*np.exp(-p*p*time[:,None]))*values)
        return sum(columns)
    def cumulative_heat(t):
        return sum(w*q*(-np.expm1(-p*p*t))/p**2 for p,w,_,q in modes_data)
    exact_heat = cumulative_heat(time)
    model,_ = slab_fem_model(k,0.,0.,CONTACT,FILMS,torch.from_numpy(c))
    nodes = (edges[:m+1],edges[m:])
    cells = (centers[:m],centers[m:])
    nodal_exact = exact(*nodes)
    cell_exact = exact(*cells)
    start = nodal_exact[0]
    u = thermal.solve_transient_thermal(model,torch.from_numpy(time),torch.from_numpy(np.tile(start,2)),
                                        theta=.5).temperature.reshape(len(time),2,-1).mean(dim=1).numpy()
    power = (u[:,m]-u[:,m+1])/CONTACT
    data = [("FEM",u,nodal_exact,.5*(power[1:]+power[:-1])*dt)]

    bc = {side:dict(type="robin",values=0.,film_coefficient=h) for side,h in zip(("west","east"),FILMS)}
    contacts = np.zeros(n-1)
    contacts[m-1] = CONTACT
    start = cell_exact[0]
    u = energy.solve_transient_conduction(np.full(n,dx),[1.],k,c,bc,time,start,
                                          contact_resistance_x=contacts[None,:])[:,0]
    power = (u[:,m-1]-u[:,m])/(CONTACT+.5*dx*sum(1./v for v in K))
    data.append(("FVM",u,cell_exact,.5*(power[1:]+power[:-1])*dt))
    states,heat = sph.solve_transient_layered_slab(start,time,dx,(None,None),conductivity=k,
            volumetric_heat_capacity=c,contact_resistance_x=contacts,x_wall_films=((FILMS[0],0.),(FILMS[1],0.)),
            return_x_heat=True)
    data.append(("SPH",states.numpy(),cell_exact,heat[:,m].numpy()))

    # Exact control-volume averages also give the layer heat used to
    # project nodal starting values conservatively on each material.
    average_exact = exact(*cells,cell_average=True)
    links,capacity = contact_chain.cell_centered_slab_network(
        np.full(n,dx),k,c,films=FILMS,contact_resistance=contacts)
    def network(name,links,capacity,initial,reference,cut):
        field = contact_chain.transient_contact_temperatures(
            links,capacity,time,np.r_[0.,initial,0.],t_left=0.,t_right=0.)[:,1:-1]
        power = (field[:,cut-1]-field[:,cut])/links[cut]
        data.append((name,field,reference,.5*(power[:-1]+power[1:])*dt))
    network("DEM_cell",links,capacity,average_exact[0],average_exact,m)
    links = np.r_[1/FILMS[0],dx/k[:m],CONTACT,dx/k[m:],1/FILMS[1]]
    mass = dx*np.repeat(C,m+1)
    mass[[0,m,m+1,-1]] *= .5
    initial = nodal_exact[0].reshape(2,m+1).copy()
    nodal_mass = mass.reshape(2,m+1)
    layer_heat = (dx*c*average_exact[0]).reshape(2,m).sum(axis=1)
    initial += ((layer_heat-(initial*nodal_mass).sum(axis=1))/nodal_mass.sum(axis=1))[:,None]
    network("DEM",links,np.r_[0.,mass,0.],initial.ravel(),nodal_exact,m+1)

    mid = .5*(time[1:]+time[:-1])
    rows = []
    for name,fields,reference,heat in data:
        avg = heat/dt
        switches = np.flatnonzero(np.signbit(avg[1:])!=np.signbit(avg[:-1]))
        crossings = [float(mid[i]-avg[i]*(mid[i+1]-mid[i])/(avg[i+1]-avg[i])) for i in switches]
        probes = .5*(np.r_[0.,reversals]+np.r_[reversals,time[-1]])
        exact_power = sum(w*q*np.exp(-p*p*probes) for p,w,_,q in modes_data)
        matched = len(crossings)==len(reversals)
        rows.append(dict(solver=name,cells=n,
                         **({"reversal_s":crossings[0] if matched else None} if len(reversals)==1 else
                            {"reversals_s":crossings}),
                         direction_matches=bool(matched and np.all(np.interp(probes,mid,avg)*exact_power>0)),
                         errors=dict(time_s=float(np.max(abs(np.array(crossings)-reversals))) if matched else None,
                         cumulative_contact_heat_J_m2=float(abs(heat.cumsum()-exact_heat[1:]).max()),
                         temperature_K=float(abs(fields-reference).max()))))
        if len(reversals)>1 and matched:
            between = float(np.diff(np.interp(crossings,time,np.r_[0.,heat.cumsum()]))[0])
            target = float(np.diff(cumulative_heat(np.array(reversals)))[0])
            rows[-1]["heat_between_reversals_J_m2"] = between
            rows[-1]["reference_heat_between_reversals_J_m2"] = target
            rows[-1]["errors"]["heat_between_reversals_J_m2"] = abs(between-target)
    return rows


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--double",action="store_true",help="prescribe two reversals using three decay modes")
    args = parser.parse_args()
    reversals = (.02,.065) if args.double else (REVERSAL,)
    rows,previous = [],{}
    meshes = (16,32,64,128,256) if args.double else (16,32,64,128)
    for n in meshes:
        for row in measure(n,reversals):
            errors = row["errors"]
            last = previous.get(row["solver"])
            orders = {k:math.log2(last[k]/v) for k,v in errors.items() if last and last.get(k) and v}
            passes = row["direction_matches"] and errors["time_s"] is not None
            if n==meshes[-1] and passes:
                passes = (errors["time_s"]<1e-5 and errors["temperature_K"]<2e-4
                          and errors["cumulative_contact_heat_J_m2"]<2e-5
                          and errors.get("heat_between_reversals_J_m2",0.)<2e-5
                          and all(v>1.7 for v in orders.values()))
            row.update(orders=orders,passed=bool(passes))
            rows.append(row)
            previous[row["solver"]] = errors
    sources = [provenance(f) for f in (__file__,inspect.getfile(slab_fem_model),thermal.__file__,energy.__file__,sph.__file__,contact_chain.__file__)]
    print(json.dumps(dict(**({"reference_reversals_s":reversals} if args.double else {"reference_reversal_s":REVERSAL}),sources=sources,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
