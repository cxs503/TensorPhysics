"""Closed two-material rod with a resistive contact and a storing shoulder."""
import argparse
import json

import numpy as np

from conduction_provenance import provenance
from tensorfvm.cht import rod


def measure(n, warming, dem=False):
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
    if dem:
        from tensordem.thermal import cell_centered_slab_network, affine_contact_interval
        sheet = np.zeros(n-1)
        sheet[m-1] = shoulder_mass/min(a)
        r,capacity = cell_centered_slab_network(np.full(n,dx),k.repeat(m),c.repeat(m),area=a.repeat(m),
                      contact_resistance=contacts,interface_heat_capacity=sheet,
                      left_resistance_fraction=1.,resolve_contacts=True)
        cells = np.r_[1:m+1,m+3:n+3]
        cold_node = m+2
        u0 = np.zeros_like(capacity)
        u0[cells],u0[cold_node] = means,surface
        powers = np.zeros_like(capacity)
        powers[cells] = source.repeat(m)*dx*a.repeat(m)
        dt = .5
        u,q,avg = affine_contact_interval(r,capacity,dt,u0,t_left=None,t_right=None,
                       nodal_power_start=powers,allow_massless=True,return_interval_temperature=True)
        expected_heat = (q0[0]+slope[0]*.5)*dt
        balance = capacity*(u-u0)+np.diff(np.r_[0.,q,0.])-dt*powers
        return dict(solver="DEM",n=n,warming=warming,
                    temperature_error_K=float(abs(u[cells]-means-alpha*dt).max()),
                    interval_temperature_error_K=float(abs(avg[cells]-means-alpha*dt/2).max()),
                    shoulder_temperature_error_K=float(abs(u[cold_node]-surface-alpha*dt)),
                    jump_error_K=float(abs(u[cold_node-1]-u[cold_node]-resistance/a[0])),
                    contact_heat_error_J=float(abs(q[cold_node-1]-expected_heat)),
                    energy_residual_J=float(abs(balance).max()))
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dem",action="store_true",help="also evolve an insulated DEM chain with a storing shoulder")
    args = parser.parse_args()
    rows = [measure(n,mode) for mode in (False,True) for n in (8,32,128,256)]
    for row in rows:
        row["passed"] = (row["temperature_error_K"]<1e-8 and row["shoulder_temperature_error_K"]<1e-8
                         and row["jump_error_K"]<1e-8 and row["power_error_W"]<1e-8
                         and row["energy_residual_J"]<1e-9)
    sources = [__file__,rod.__file__]
    if args.dem:
        from tensordem.thermal import contact_chain
        sources.append(contact_chain.__file__)
        for warming in (False,True):
            previous = {}
            for n in (16,32,64,128):
                row = measure(n,warming,dem=True)
                errors = {k:v for k,v in row.items() if "_error_" in k}
                orders = {k:float(np.log2(previous[k]/v)) for k,v in errors.items() if previous}
                row.update(orders=orders,passed=row["energy_residual_J"]<1e-11 and
                           all(v>1.8 for v in orders.values()) and
                           (n!=128 or max(errors.values())<1e-5))
                rows.append(row)
                previous = errors
    print(json.dumps(dict(runs=rows,sources=[provenance(f) for f in sources]),indent=2,allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
