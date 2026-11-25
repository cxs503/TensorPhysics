"""Two independently phased surface heaters at a resistive, heat-storing rod contact."""
from dataclasses import replace
import json

import numpy as np
import torch

from conduction_provenance import provenance
from tensorfem import thermal
from tensorfvm.cht import rod
import two_layer_conduction


K = np.array([1.,2.])
C = np.array([4.,1.])
STORAGE = np.array([.56,.24])
RESISTANCE = .24
DRIVE = np.array([1.1+.4j,-.2+1.6j])


def reference(area,omega):
    # A rod held at zero temperature at its far end has temperature profile
    # sinh(w*(L-x))/sinh(w*L) times the driven surface temperature.
    w = np.sqrt(1j*omega*C/K)
    g = min(area)/RESISTANCE
    diagonal = area*K*w/np.tanh(.5*w)+1j*omega*STORAGE
    traces = np.linalg.solve(np.diag(diagonal)+g*np.array([[1.,-1.],[-1.,1.]]),DRIVE)
    contact = g*(traces[0]-traces[1])
    outward = area*K*w*traces/np.sinh(.5*w)
    return traces,contact,outward


def fvm(n,area,omega):
    face = np.zeros(2*n-1)
    face[n-1] = 1.
    _,_,left,right,(west,east) = rod.solve_harmonic_stepped_rod(
        np.full(2*n,.5/n),area.repeat(n),K.repeat(n),C.repeat(n),[0.,0.],omega,
        contact_resistance=face*RESISTANCE,interface_heat_capacity=face*STORAGE.sum(),
        interface_capacity_left_fraction=STORAGE[0]/STORAGE.sum(),trace_power=(face*DRIVE[0],face*DRIVE[1]),
        return_traces=True,flux_order=2)
    traces = np.array([west[n-1],east[n-1]])
    # Recover contact transfer independently from the left surface energy balance.
    contact = right[n-1]+DRIVE[0]-1j*omega*STORAGE[0]*traces[0]
    return traces,contact,np.array([-left[0],right[-1]])


def fem(n,area,omega):
    model,ends = two_layer_conduction.slab_fem_model(K.repeat(n),0.,0.,RESISTANCE/min(area),None,
                                                      torch.from_numpy(C.repeat(n)))
    stride = len(model.nodes)//2
    west = torch.tensor([n,n+stride])
    storage = torch.zeros_like(model.nodes[:,0])
    power = storage.to(torch.complex128)
    for side in (0,1):
        storage[west+side] = STORAGE[side]/2
        power[west+side] = DRIVE[side]/2
    model = replace(model,thickness=torch.from_numpy(area.repeat(n)),nodal_heat_capacity=storage,nodal_heat=power)
    result = thermal.solve_harmonic_thermal(model,omega,sparse=True)
    traces = result.temperature.reshape(2,stride)[:,n:n+2].mean(dim=0).numpy()
    return traces,(min(area)/RESISTANCE)*(traces[0]-traces[1]),-result.reaction_power[ends].sum(dim=1).numpy()


if __name__=="__main__":
    rows = []
    for area in (np.array([1.,2.]),np.array([2.,1.])):
        for omega in (1.,8.,32.):
            trace,q,wall = reference(area,omega)
            for solve in (fem,fvm):
                previous = {}
                for n in (8,16,32,64):
                    temp,contact,outward = solve(n,area,omega)
                    errors = dict(trace_K=float(max(abs(temp-trace))),contact_W=float(abs(contact-q)),
                                  end_W=float(max(abs(outward-wall))),
                                  contact_phase_deg=float(abs(np.angle(contact/q,deg=True))))
                    orders = {k:float(np.log2(previous[k]/v)) for k,v in errors.items()} if previous else {}
                    rows.append(dict(area_m2=area.tolist(),omega_rad_s=omega,solver=solve.__name__,
                                     cells_per_branch=n,errors=errors,orders=orders,
                                     passed=all(v>0 for v in orders.values()) and
                                     (n<64 or min(orders.values())>1.8)))
                    previous = errors
    print(json.dumps(dict(runs=rows,sources=[provenance(f) for f in
                     (__file__,rod.__file__,thermal.__file__,two_layer_conduction.__file__)]),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
