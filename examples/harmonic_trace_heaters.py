"""Two independently phased surface heaters at a resistive, heat-storing rod contact."""
from dataclasses import replace
import argparse
import inspect
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


def dg(n,area,omega):
    from tensordg import solve_harmonic_diffusion_2d, poisson_face_powers_2d

    # Integrate the rod equation over its cross-section, using a unit-width strip.
    face = np.zeros((2*n-1,2))
    face[n-1] = 1.
    opts = dict(conductivity=np.tile((area*K).repeat(n)[:,None],(1,2)),
                bc=dict(top="neumann",bottom="neumann"),contact_resistance={"x":face*RESISTANCE/min(area)})
    _,_,field = solve_harmonic_diffusion_2d(omega,ncells=(2*n,2),degree=1,
                    capacity=np.tile((area*C).repeat(n)[:,None],(1,2)),
                    trace_power={"x":(face*DRIVE[0],face*DRIVE[1])},
                    trace_heat_capacity={"x":(face*STORAGE[0],face*STORAGE[1])},**opts)
    modes = field[:,:,:,0].mean(1).numpy()
    traces = np.array([modes[n-1].sum(),modes[n,0]-modes[n,1]])
    fx,_ = poisson_face_powers_2d(field,**opts)
    transfer = fx[n].sum().item()
    outward = np.array([-fx[0].sum().item(),fx[-1].sum().item()])
    bulk = .5*(area*C)*modes[:,0].reshape(2,n).mean(axis=1)
    residual = DRIVE-outward+np.array([-transfer,transfer])-1j*omega*(bulk+STORAGE*traces)
    return traces,transfer,outward,float(abs(residual).max())


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dg",action="store_true",help="include SIPG and its per-branch power balances")
    args = parser.parse_args()
    rows = []
    for area in (np.array([1.,2.]),np.array([2.,1.])):
        for omega in (1.,8.,32.):
            trace,q,wall = reference(area,omega)
            for solve in ((fem,fvm,dg) if args.dg else (fem,fvm)):
                previous = {}
                for n in (8,16,32,64):
                    temp,contact,outward,*balance = solve(n,area,omega)
                    errors = dict(trace_K=float(max(abs(temp-trace))),contact_W=float(abs(contact-q)),
                                  end_W=float(max(abs(outward-wall))),
                                  contact_phase_deg=float(abs(np.angle(contact/q,deg=True))))
                    orders = {k:float(np.log2(previous[k]/v)) for k,v in errors.items()} if previous else {}
                    rows.append(dict(area_m2=area.tolist(),omega_rad_s=omega,solver=solve.__name__,
                                     cells_per_branch=n,errors=errors,orders=orders,
                                     passed=all(v<1e-10 for v in balance) and all(v>0 for v in orders.values()) and
                                     (n<64 or min(orders.values())>1.8)))
                    if balance:
                        rows[-1]["max_branch_power_residual_W"] = balance[0]
                    previous = errors
    sources = [__file__,rod.__file__,thermal.__file__,two_layer_conduction.__file__]
    if args.dg:
        from tensordg import solve_harmonic_diffusion_2d, poisson_face_powers_2d
        sources.extend(inspect.getfile(f) for f in (solve_harmonic_diffusion_2d,poisson_face_powers_2d))
    print(json.dumps(dict(runs=rows,sources=[provenance(f) for f in sources]),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
