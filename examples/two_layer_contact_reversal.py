"""Free cooling of a two-material slab with a known contact-current reversal."""
import inspect
import json
import math

import numpy as np
import torch
from scipy.optimize import brentq

from conduction_provenance import provenance
from two_layer_conduction import slab_fem_model
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


def modes():
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
            if len(roots)==2:
                break
        lo = hi
    p1,p2 = roots
    q1,q2 = (traces(p)[1] for p in roots)
    # Choose the second modal amplitude so the sum of the contact currents
    # vanishes at REVERSAL. Distinct decay rates give a single zero.
    factor = -q1/q2*math.exp((p2*p2-p1*p1)*REVERSAL)
    return [(p,weight,*traces(p)) for p,weight in zip(roots,(1.,factor))]


def measure(n):
    dx,m = 1/n,n//2
    # Resolve the flux zero between interval midpoints on every mesh.
    time = np.linspace(0.,.1,4*n+1)
    dt = np.diff(time)
    k,c = np.repeat(K,m),np.repeat(C,m)
    edges = np.linspace(0.,1.,n+1)
    centers = (edges[1:]+edges[:-1])/2
    modes_data = modes()
    def exact(t,left,right):
        return sum(w*math.exp(-p*p*t)*np.r_[propagate(1.,-FILMS[0],p,0,left)[0],
                    propagate(tr,q,p,1,right-.5)[0]] for p,w,tr,q in modes_data)
    exact_heat = sum(w*q*(-np.expm1(-p*p*time))/p**2 for p,w,_,q in modes_data)
    model,_ = slab_fem_model(k,0.,0.,CONTACT,FILMS,torch.from_numpy(c))
    nodes = (edges[:m+1],edges[m:])
    cells = (centers[:m],centers[m:])
    start = exact(0.,*nodes)
    u = thermal.solve_transient_thermal(model,torch.from_numpy(time),torch.from_numpy(np.tile(start,2)),
                                        theta=.5).temperature.reshape(len(time),2,-1).mean(dim=1).numpy()
    power = (u[:,m]-u[:,m+1])/CONTACT
    data = [("FEM",u,nodes,.5*(power[1:]+power[:-1])*dt)]

    bc = {side:dict(type="robin",values=0.,film_coefficient=h) for side,h in zip(("west","east"),FILMS)}
    contacts = np.zeros(n-1)
    contacts[m-1] = CONTACT
    start = exact(0.,*cells)
    u = energy.solve_transient_conduction(np.full(n,dx),[1.],k,c,bc,time,start,
                                          contact_resistance_x=contacts[None,:])[:,0]
    power = (u[:,m-1]-u[:,m])/(CONTACT+.5*dx*sum(1./v for v in K))
    data.append(("FVM",u,cells,.5*(power[1:]+power[:-1])*dt))
    states,heat = sph.solve_transient_layered_slab(start,time,dx,(None,None),conductivity=k,
            volumetric_heat_capacity=c,contact_resistance_x=contacts,x_wall_films=((FILMS[0],0.),(FILMS[1],0.)),
            return_x_heat=True)
    data.append(("SPH",states.numpy(),cells,heat[:,m].numpy()))

    mid = .5*(time[1:]+time[:-1])
    rows = []
    for name,fields,points,heat in data:
        avg = heat/dt
        switches = np.flatnonzero(np.signbit(avg[1:])!=np.signbit(avg[:-1]))
        reversal = None
        if len(switches)==1:
            i = switches[0]
            reversal = float(mid[i]-avg[i]*(mid[i+1]-mid[i])/(avg[i+1]-avg[i]))
        rows.append(dict(solver=name,cells=n,reversal_s=reversal,
                         direction_matches=bool(avg[0]*exact_heat[1]>0 and avg[-1]*np.diff(exact_heat)[-1]>0),
                         errors=dict(time_s=abs(reversal-REVERSAL) if reversal is not None else None,
                         cumulative_contact_heat_J_m2=float(abs(heat.cumsum()-exact_heat[1:]).max()),
                         temperature_K=max(float(abs(field-exact(t,*points)).max()) for field,t in zip(fields,time)))))
    return rows


if __name__=="__main__":
    rows,previous = [],{}
    for n in (16,32,64,128):
        for row in measure(n):
            errors = row["errors"]
            last = previous.get(row["solver"])
            orders = {k:math.log2(last[k]/v) for k,v in errors.items() if last and last[k] and v}
            passes = row["direction_matches"] and row["reversal_s"] is not None
            if n==128 and passes:
                passes = (errors["time_s"]<1e-5 and errors["temperature_K"]<2e-4
                          and errors["cumulative_contact_heat_J_m2"]<2e-5
                          and all(v>1.7 for v in orders.values()))
            row.update(orders=orders,passed=bool(passes))
            rows.append(row)
            previous[row["solver"]] = errors
    sources = [provenance(f) for f in (__file__,inspect.getfile(slab_fem_model),thermal.__file__,energy.__file__,sph.__file__)]
    print(json.dumps(dict(reference_reversal_s=REVERSAL,sources=sources,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
