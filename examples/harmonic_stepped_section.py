"""Harmonic conduction along two joined sections of different area, with storing end films."""
from dataclasses import replace
import json

import numpy as np
import torch

from conduction_provenance import provenance
import two_layer_conduction
from tensorfem import thermal
from tensorfvm.cht import energy, rod


def propagate(x, t, p, wave, k_area):
    z = wave*x
    g = k_area*wave
    return t*np.cosh(z)-p*np.sinh(z)/g, p*np.cosh(z)-g*t*np.sinh(z)


def measure(n):
    omega, contact = 3., .18  # rad/s and total interface resistance in K/W
    areas = np.array([.7, 2.4])
    ks, caps = np.array([2., 5.]), np.array([3., 1.5])
    films, cw = np.array([1.8,4.]), np.array([.6,.9])
    waves = np.sqrt(1j*omega*caps/ks)

    # Total axial power is continuous; the discontinuity in flux density is P/A.
    t0, p0 = .8+.2j, 1.-.4j
    hot, cut = propagate(.5,t0,p0,waves[0],ks[0]*areas[0])
    cold = hot-contact*cut
    t1, p1 = propagate(.5,cold,cut,waves[1],ks[1]*areas[1])
    wall_t, wall_p = np.array([t0,t1]), np.array([p0,-p1])
    ambient = wall_t + (wall_p/areas+1j*omega*cw*wall_t)/films
    m, dx = n//2, 1/n
    local = np.arange(m+1)*dx
    nodal = np.r_[propagate(local,t0,p0,waves[0],ks[0]*areas[0])[0],propagate(local,cold,cut,waves[1],ks[1]*areas[1])[0]]
    def mean(layer,t,p):
        z = waves[layer]*dx/2
        return propagate(local[:-1]+dx/2,t,p,waves[layer],ks[layer]*areas[layer])[0]*np.sinh(z)/z
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


def shoulder(n):
    # Perfect thermal contact between the rod segments; only the excess area
    # of the larger cross section carries a film and surface heat capacity.
    m, omega, h, cw = n//2, 3., 1.8, .6
    area = np.array([.7,2.4])
    exposed = area[1]-area[0]
    k, c = np.array([2.,5.]), np.array([3.,1.5])
    wave = np.sqrt(1j*omega*c/k)
    center = .8+.2j
    left_power, right_power = 1.-.4j, -.3+.9j
    ambient = center + ((right_power-left_power)/exposed+1j*omega*cw*center)/h
    x = np.arange(m+1)/n
    left = propagate(x-.5,center,left_power,wave[0],k[0]*area[0])[0]
    right = propagate(x,center,right_power,wave[1],k[1]*area[1])[0]
    exact = np.r_[left,right[1:]]
    base, _ = two_layer_conduction.slab_fem_model(k.repeat(m),0.,0.,0.,None,torch.from_numpy(c.repeat(m)))
    q4 = replace(base,thickness=torch.from_numpy(area.repeat(m)), convection_area=exposed,
                   convection_edges=torch.tensor([[m,m+n+1]]),
                   convection_coefficient=h,ambient_temperature=ambient,wall_areal_heat_capacity=cw,
                   fixed_temperatures=torch.from_numpy(exact[[0,0,-1,-1]]))
    line = replace(q4,nodes=q4.nodes[:n+1,:1],elements=q4.elements[:,:2],element_type="line2",
                   fixed_nodes=torch.tensor([0,n]),fixed_temperatures=torch.from_numpy(exact[[0,-1]]),
                   convection_edges=torch.tensor([[m,m]]))
    for name,model in (("shoulder-Q4",q4),("shoulder-LINE2",line)):
        u = thermal.solve_harmonic_thermal(model,omega,sparse=True).temperature
        t = u.numpy().reshape(-1,n+1).mean(axis=0)
        powers = thermal.thermal_element_nodal_power(model,u,omega).numpy()
        before, after = -powers[m-1,1], powers[m,0]
        if name == "shoulder-Q4":
            before -= powers[m-1,2]
            after += powers[m,3]
        net_input = exposed*(h*(ambient-t[m])-1j*omega*cw*t[m])
        yield dict(solver=name,n=n,temperature_error_K=float(abs(t-exact).max()),
                   interface_power_error_W=float(max(abs(before-left_power),abs(after-right_power))),
                   shoulder_balance_W=float(abs(before+net_input-after)))

    cell, faces, incoming, outgoing = rod.solve_harmonic_stepped_rod(
        np.full(n,1/n),area.repeat(m),k.repeat(m),c.repeat(m),exact[[0,-1]],omega,
        film_coefficient=h,ambient_temperature=ambient,shoulder_areal_heat_capacity=cw)
    # Integrate the continuum axial phasor over each finite volume.
    midpoints = x[:-1]+.5/n
    exact_cell = np.concatenate([propagate(midpoints+offset,center,p,w,ka)[0]*np.sinh(w/(2*n))/(w/(2*n))
                                 for offset,p,w,ka in zip((-.5,0.),(left_power,right_power),wave,k*area)])
    before, after = outgoing[m-1], incoming[m]
    net_input = exposed*(h*(ambient-faces[m])-1j*omega*cw*faces[m])
    yield dict(solver="shoulder-FVM",n=n,temperature_error_K=float(abs(cell-exact_cell).max()),
               interface_power_error_W=float(max(abs(before-left_power),abs(after-right_power))),
               shoulder_balance_W=float(abs(before+net_input-after)),
               cell_balance_W=float(abs(1j*omega*(c*area).repeat(m)*cell/n-incoming+outgoing).max()))


if __name__ == "__main__":
    rows, previous = [], {}
    for n in (16,32,64,128,256):
        for row in (*measure(n), *shoulder(n)):
            name = row["solver"]
            errors = {k:v for k,v in row.items() if "_error_" in k}
            orders = {} if name not in previous else {k:np.log2(previous[name][k]/v) for k,v in errors.items()}
            row.update(orders=orders,passed=all(p>1.85 for p in orders.values()) and
                       max(v for k,v in row.items() if k.endswith("balance_W"))<1e-10)
            rows.append(row)
            previous[name] = errors
    print(json.dumps(dict(runs=rows,sources=[provenance(f) for f in
                     (__file__,two_layer_conduction.__file__,thermal.__file__,energy.__file__,rod.__file__)]),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
