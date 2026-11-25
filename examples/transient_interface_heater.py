"""Heat partition from a storing, resistive heater between rods of unequal diffusivity."""
from dataclasses import replace
import json

import numpy as np
import torch

from conduction_provenance import provenance
from tensorfem import thermal
from tensorfvm.cht import rod
import two_layer_conduction


K = np.array([1.,2.])
CAPACITY = np.array([4.,1.])
STORAGE, RESISTANCE = .8,.24
POWER, DECAY = 3.,4.
SAMPLES = np.array([.0125,.025,.05,.1,.2,.4,.8])


def reference(area, degree):
    # Fixed Talbot contour. Each transform is from the continuum rod solution
    # T(x,s)=T(0,s)*sinh(w*(L-x))/sinh(w*L) measured from the sheet.
    theta = np.arange(1,degree)*np.pi/degree
    cot = 1/np.tan(theta)
    delta = .4*degree*np.r_[1.,theta*(cot+1j)]
    weights = np.exp(delta)*np.r_[.5,1+1j*(theta*(1+cot**2)-cot)]
    s = delta[None,:,None]/SAMPLES[:,None,None]
    wave = np.sqrt(s*CAPACITY/K)
    admittance = area*K*wave/np.tanh(.5*wave)
    # Place the heater on the larger section. The smaller branch sees the
    # contact resistance in series with its continuum thermal admittance.
    resistance = np.zeros(2)
    resistance[np.argmin(area)] = RESISTANCE/min(area)
    transmission = 1/(1+resistance*admittance)
    sheet = (POWER/(s+DECAY))/(s*STORAGE+(admittance*transmission).sum(axis=-1,keepdims=True))
    trace = transmission*sheet
    wall_heat = area*K*wave*trace/(s*np.sinh(.5*wave))
    transforms = np.concatenate((sheet,trace[...,:1]-trace[...,1:],wall_heat),axis=-1)
    return (.4/SAMPLES[:,None])*(weights[None,:,None]*transforms).sum(axis=1).real


def fvm(n,area,times):
    interface = np.zeros(2*n-1)
    interface[n-1] = 1.
    _,faces,_,_,(west,east),heat = rod.solve_transient_stepped_rod(
        np.full(2*n,.5/n),area.repeat(n),K.repeat(n),CAPACITY.repeat(n),[0.,0.],times,0.,
        interface_heat_capacity=STORAGE*interface,initial_shoulder_temperature=0.,
        contact_resistance=RESISTANCE*interface,interface_power=lambda t:POWER*np.exp(-DECAY*t)*interface,
        flux_order=2,method="sdirk2",return_traces=True,return_step_heat=True)
    wall = np.vstack((np.zeros(2),-heat["end_input"].cumsum(axis=0)))
    return np.column_stack((faces[:,n],west[:,n-1]-east[:,n-1],wall))


def fem(n,area,times):
    model,ends = two_layer_conduction.slab_fem_model(K.repeat(n),0.,0.,RESISTANCE/min(area),None,
                                torch.from_numpy(CAPACITY.repeat(n)))
    stride = len(model.nodes)//2
    nodes = torch.tensor([n,n+stride])+int(np.argmax(area))
    storage = torch.zeros_like(model.nodes[:,0])
    storage[nodes] = STORAGE/2
    model = replace(model,thickness=torch.from_numpy(area.repeat(n)),nodal_heat_capacity=storage)
    t = torch.from_numpy(times)
    load = storage.new_zeros((len(t),len(storage)))
    load[:,nodes] = POWER/2*torch.exp(-DECAY*t[:,None])
    result = thermal.solve_transient_thermal(model,t,torch.zeros_like(storage),theta=.5,sparse=True,
                                              nodal_heat_history=load)
    trace = result.temperature.reshape(len(t),2,stride).mean(dim=1)
    outward = -result.nodal_reaction_heat_steps()[:,ends].sum(dim=-1).cumsum(dim=0)
    wall = np.vstack((np.zeros(2),outward.numpy()))
    return np.column_stack((trace[:,nodes[0]],trace[:,n]-trace[:,n+1],wall))


if __name__=="__main__":
    rows = []
    for area in (np.array([1.,2.]),np.array([2.,1.])):
        exact = reference(area,32)
        inversion_change = float(abs(exact-reference(area,24)).max())
        for solve in (fem,fvm):
            previous = {}
            for n in (8,16,32,64):
                times = np.linspace(0.,.8,16*n+1)
                idx = np.searchsorted(times,SAMPLES)
                assert np.allclose(times[idx],SAMPLES,rtol=0.,atol=1e-14)
                diff = np.max(abs(solve(n,area,times)[idx]-exact),axis=0)
                errors = dict(zip(("sheet_K","signed_jump_K","west_end_J","east_end_J"),diff))
                orders = {k:np.log2(previous[k]/v) for k,v in errors.items()} if previous else {}
                passed = inversion_change<1e-10 and all(v>0 for v in orders.values())
                rows.append(dict(area_m2=area.tolist(),solver=solve.__name__,cells_per_branch=n,
                                 errors=errors,orders=orders,inversion_change=inversion_change,
                                 passed=bool(passed and (n<64 or min(orders.values())>1.8))))
                previous = errors
    print(json.dumps(dict(observation_times_s=SAMPLES.tolist(),runs=rows,sources=[provenance(f) for f in
                     (__file__,rod.__file__,thermal.__file__,two_layer_conduction.__file__)]),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
