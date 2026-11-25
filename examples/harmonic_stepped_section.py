"""Harmonic conduction along two joined sections of different area, with storing end films."""
from dataclasses import replace
import json

import numpy as np
import torch

from conduction_provenance import provenance
import two_layer_conduction
from tensorfem import thermal
from tensorfvm.cht import energy


def measure(n):
    omega, contact = 3., .18  # rad/s and total interface resistance in K/W
    areas = np.array([.7, 2.4])
    ks, caps = np.array([2., 5.]), np.array([3., 1.5])
    films, cw = np.array([1.8,4.]), np.array([.6,.9])
    waves = np.sqrt(1j*omega*caps/ks)

    def propagate(layer, x, t, p):
        z = waves[layer]*x
        g = ks[layer]*areas[layer]*waves[layer]
        return t*np.cosh(z)-p*np.sinh(z)/g, p*np.cosh(z)-g*t*np.sinh(z)

    # Total axial power is continuous; the discontinuity in flux density is P/A.
    t0, p0 = .8+.2j, 1.-.4j
    hot, cut = propagate(0,.5,t0,p0)
    cold = hot-contact*cut
    t1, p1 = propagate(1,.5,cold,cut)
    wall_t, wall_p = np.array([t0,t1]), np.array([p0,-p1])
    ambient = wall_t + (wall_p/areas+1j*omega*cw*wall_t)/films
    m, dx = n//2, 1/n
    local = np.arange(m+1)*dx
    nodal = np.r_[propagate(0,local,t0,p0)[0],propagate(1,local,cold,cut)[0]]
    def mean(layer,t,p):
        z = waves[layer]*dx/2
        return propagate(layer,local[:-1]+dx/2,t,p)[0]*np.sinh(z)/z
    avg = np.r_[mean(0,t0,p0),mean(1,cold,cut)]
    k,c,area = (v.repeat(m) for v in (ks,caps,areas))
    q4, _ = two_layer_conduction.slab_fem_model(k,0.,0.,contact,films,torch.from_numpy(c))
    q4 = replace(q4,thickness=torch.from_numpy(area),ambient_temperature=torch.from_numpy(ambient),
                 wall_areal_heat_capacity=torch.from_numpy(cw))
    line = replace(q4,nodes=q4.nodes[:n+2,:1],elements=q4.elements[:,:2],element_type="line2",
                   convection_edges=torch.tensor([[0,0],[n+1,n+1]]),
                   contact_pairs=torch.tensor([[m,m+1]]),contact_conductance=1/contact)
    for name,model in (("FEM-Q4",q4),("FEM-LINE2",line)):
        sol = thermal.solve_harmonic_thermal(model,omega,sparse=True)
        t = sol.temperature.numpy().reshape(-1,n+2).mean(axis=0)
        local_p = thermal.thermal_element_nodal_power(model,sol.temperature,omega).numpy()
        wall = local_p[[0,-1],[0,1]].copy()
        if name == "FEM-Q4":
            wall += local_p[[0,-1],[3,2]]
        interior = (t[m]-t[m+1])/contact
        cells = np.r_[(t[:m]+t[1:m+1])/2,(t[m+1:-1]+t[m+2:])/2]
        yield metrics(name,n,t-nodal,wall,wall_p,interior,cut,t[[0,-1]],ambient,areas,films,cw,
                      omega, c*area*dx*cells)
    # Unit-width FVM uses k*A and c*A so that its face powers correspond to watts.
    bc = {s:dict(type="robin",values=a,film_coefficient=h*A,wall_areal_heat_capacity=w*A)
          for s,a,h,w,A in zip(("west","east"),ambient,films,cw,areas)}
    r = np.zeros((1,n-1))
    r[0,m-1] = contact
    t,p,q = energy.solve_harmonic_conduction(np.full(n,dx),[1.],k*area,c*area,bc,omega,
                                        return_interior_power=True,contact_resistance_x=r)
    wall = np.array([p["west"][0],p["east"][0]])
    surface = t[0,[0,-1]]+wall*dx/(2*ks*areas)
    yield metrics("FVM",n,t[0]-avg,wall,wall_p,q["x"][0,m-1],cut,surface,ambient,areas,films,cw,
                  omega,c*area*dx*t[0])


def metrics(name,n,delta,wall,wall_ref,cut,cut_ref,t,ambient,area,h,cw,omega,stored):
    balance = 1j*omega*stored.reshape(2,-1).sum(axis=1)-wall+np.array([cut,-cut])
    return dict(solver=name,n=n,temperature_error_K=float(abs(delta).max()),
                wall_power_error_W=float(abs(wall-wall_ref).max()),contact_power_error_W=float(abs(cut-cut_ref)),
                wall_balance_W=float(abs(area*(h*(ambient-t)-1j*omega*cw*t)-wall).max()),
                layer_balance_W=float(abs(balance).max()))


if __name__ == "__main__":
    rows, previous = [], {}
    for n in (16,32,64,128,256):
        for row in measure(n):
            name = row["solver"]
            errors = {k:v for k,v in row.items() if "_error_" in k}
            orders = {} if name not in previous else {k:np.log2(previous[name][k]/v) for k,v in errors.items()}
            row.update(orders=orders,passed=all(p>1.85 for p in orders.values()) and
                       max(row["wall_balance_W"],row["layer_balance_W"])<1e-10)
            rows.append(row)
            previous[name] = errors
    print(json.dumps(dict(runs=rows,sources=[provenance(f) for f in
                     (__file__,two_layer_conduction.__file__,thermal.__file__,energy.__file__)]),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
