"""Closed two-material rod with a resistive contact and a storing shoulder."""
import json

import numpy as np

from conduction_provenance import provenance
from tensorfvm.cht import rod


def measure(n, warming):
    m,dx = n//2,1/n
    a,k,c = np.array([.7,2.4]),np.array([2.,5.]),np.array([3.,1.5])
    resistance, cw = .25,.4
    shoulder_mass = cw*(a[1]-a[0])
    alpha = .8 if warming else 0.
    # Insulated ends: P rises linearly to 1 W in the first section. The
    # shoulder absorbs C_s*alpha W, then P falls linearly to zero at the right.
    q0 = np.array([0.,1.-shoulder_mass*alpha])
    slope = np.array([2.,-2*q0[1]])
    source = c*alpha + slope/a
    def temperature(start,j,x):
        return start-(q0[j]*x+.5*slope[j]*x*x)/(k[j]*a[j])
    t_left = 2.
    contact_left = temperature(t_left,0,.5)
    surface = contact_left-resistance/a[0]
    x = np.arange(m)*dx
    # Exact cell average of the quadratic includes its curvature correction.
    means = np.concatenate([temperature(start,j,x+dx/2)-slope[j]*dx**2/(24*k[j]*a[j])
                            for j,start in enumerate((t_left,surface))])
    cm = dx*(a*c).repeat(m)
    total_capacity = cm.sum()+shoulder_mass
    gauge = (cm@means+shoulder_mass*surface)/total_capacity
    contacts = np.zeros(n-1)
    contacts[m-1] = resistance
    args = (np.full(n,dx),a.repeat(m),k.repeat(m),c.repeat(m),[0.,0.])
    opts = dict(source=source.repeat(m),shoulder_areal_heat_capacity=cw,contact_resistance=contacts,
                flux_order=2,return_traces=True,end_types=("power","power"))
    if warming:
        times = np.linspace(0.,.5,17)
        u,f,p,q,(hot,cold),heat = rod.solve_transient_stepped_rod(*args,times,means,
                        initial_shoulder_temperature=surface,return_step_heat=True,**opts)
        injected = heat["source"].sum(axis=1)+heat["left"][:,0]-heat["right"][:,-1]
        stored = u@cm+shoulder_mass*f[:,m]
        residual = abs(np.diff(stored)-injected).max()
    else:
        times = np.array([0.])
        cell,face,left,right,traces = rod.solve_steady_stepped_rod(*args,mean_temperature=gauge,**opts)
        u,f,p,q = (v[None,:] for v in (cell,face,left,right))
        hot,cold = (v[None,:] for v in traces)
        residual = abs(cm@cell+shoulder_mass*face[m]-gauge*total_capacity)
    p_left = np.concatenate([q0[j]+slope[j]*x for j in (0,1)])
    p_right = np.concatenate([q0[j]+slope[j]*(x+dx) for j in (0,1)])
    return dict(n=n,warming=warming,temperature_error_K=float(abs(u-means-alpha*times[:,None]).max()),
                shoulder_temperature_error_K=float(abs(f[:,m]-surface-alpha*times).max()),
                jump_error_K=float(abs(hot[:,m-1]-cold[:,m-1]-resistance/a[0]).max()),
                power_error_W=float(max(abs(p-p_left).max(),abs(q-p_right).max())),
                energy_residual_J=float(residual))


if __name__ == "__main__":
    rows = [measure(n,mode) for mode in (False,True) for n in (8,32,128,256)]
    for row in rows:
        row["passed"] = (row["temperature_error_K"]<1e-8 and row["shoulder_temperature_error_K"]<1e-8
                         and row["jump_error_K"]<1e-8 and row["power_error_W"]<1e-8
                         and row["energy_residual_J"]<1e-9)
    print(json.dumps(dict(runs=rows,sources=[provenance(f) for f in (__file__,rod.__file__)]),
                     indent=2,allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
