"""FEM/DG interval heat and cell means under independently switched walls and volume heating."""
import json

import numpy as np
from numpy.polynomial import Polynomial
import torch
from tensorfem import thermal
from tensordg import heat_1d
from conduction_provenance import provenance


TIMES = np.array([0., .035, .06, .11, .14])
WALLS = np.array([[0.,0.], [.07,-.035], [.07,-.035], [-.03,.015], [-.03,.075]])
# Volumetric source density at x=0 and x=1, linearly interpolated in space.
SOURCES = np.array([[1.8,-.4], [0.,0.], [-.7,2.1], [.8,.8]])


def profile(pair, x):
    return pair[0]+(pair[1]-pair[0])*x


def reference(n, modes):
    edges = np.linspace(0.,1.,n+1)
    x = Polynomial([0.,1.])
    k = np.pi*np.arange(1,modes+1)
    sign = (-1.)**np.arange(1,modes+1)
    mean_sine = -n*np.diff(np.cos(edges[:,None]*k),axis=0)/k
    coefficients = np.zeros(modes)
    means = [np.zeros(n)]
    west_moment = 0.
    heats = []
    for i,dt in enumerate(np.diff(TIMES)):
        rates = (WALLS[i+1]-WALLS[i])/dt
        a, end = SOURCES[i]-rates
        b = end-a
        # Lift the linear wall profile. The stationary part H of the remainder
        # satisfies -H''=a+b*x with zero ends. Evolve the deviation in sines.
        stationary = 2*(a-sign*end)/k**3
        transient = np.exp(-k*k*dt)*(coefficients-stationary)
        coefficients = stationary+transient
        poly = profile(WALLS[i+1],x)+(a/2+b/6)*x-a*x**2/2-b*x**3/6
        means.append(n*np.diff(poly.integ()(edges))+mean_sine@transient)
        moment = (poly*(1-x)).integ()(1.)+np.sum(transient/k)
        q = profile(SOURCES[i],x)
        deposited = dt*np.diff(q.integ()(edges))
        # Test the continuum heat balance with 1-x to recover net west input.
        conduction = dt*np.mean(WALLS[i:i+2,0]-WALLS[i:i+2,1])
        west_heat = moment-west_moment+conduction-dt*((1-x)*q).integ()(1.)
        heats.append(west_heat+np.r_[0., np.cumsum(deposited-(means[-1]-means[-2])/n)])
        west_moment = moment
    return np.asarray(means),np.asarray(heats)


def fem(n):
    nodes = torch.linspace(0.,1.,n+1,dtype=torch.float64)
    elements = torch.stack((torch.arange(n),torch.arange(1,n+1)),dim=1)
    fixed = torch.tensor([0,n])
    state = torch.zeros_like(nodes)
    means, heats = [np.zeros(n)], []
    for i,dt in enumerate(np.diff(TIMES)):
        model = thermal.ThermalModel(nodes[:,None],elements,1.,1.,1.,fixed,state[fixed],
                                     heat_source=profile(SOURCES[i],nodes)[elements])
        times = torch.linspace(0.,dt,2*n+1,dtype=nodes.dtype)
        walls = torch.from_numpy(WALLS[i])+times[:,None]*torch.from_numpy(WALLS[i+1]-WALLS[i])/dt
        result = thermal.solve_transient_thermal(model,times,state,sparse=True,method="sdirk2",
                                                 fixed_temperature_history=walls)
        state = result.temperature[-1]
        means.append(state[elements].mean(dim=1).numpy())
        local = thermal.thermal_element_nodal_heat_steps(model,result).sum(dim=0)
        heats.append(torch.cat((local[:1,0],-local[:,1])).numpy())
    return np.asarray(means),np.asarray(heats)


def dg(n):
    sources = [lambda x,pair=q: profile(pair,x) for q in SOURCES]
    _,states,meta = heat_1d.solve_heat_1d_wall_history(TIMES,WALLS,n,2,backend="exact-time",
        initial_temperature=torch.zeros_like,interval_sources=sources,integrate_face_fluxes=True)
    return states[:,:,0].numpy(),np.asarray(meta["interval_face_heat"])


if __name__ == "__main__":
    rows = []
    previous = {}
    for n in (8,16,32,64):
        expected,heat = reference(n,64)
        check = reference(n,32)
        reference_change = max(np.max(abs(check[0]-expected)),np.max(abs(check[1]-heat)))
        mid = (np.arange(n)+.5)/n
        deposited = np.diff(TIMES)[:,None]*np.stack([profile(q,mid) for q in SOURCES])/n
        for name,solve in (("FEM_LINE2_SDIRK2",fem),("DG_P2_exponential",dg)):
            u,h = solve(n)
            errors = dict(cell_mean=float(np.max(abs(u-expected))),
                          section_heat=float(np.max(abs(h-heat))),
                          wall_heat=float(np.max(abs(h[:,[0,-1]]-heat[:,[0,-1]]))))
            orders = {k:float(np.log2(previous[name][k]/v)) for k,v in errors.items()} if name in previous else {}
            balance = float(np.max(abs(np.diff(u,axis=0)/n+np.diff(h)-deposited)))
            passed = reference_change<1e-12 and balance<1e-10 and all(o>1.7 for o in orders.values())
            rows.append(dict(solver=name,cells=n,errors=errors,orders=orders,max_cell_balance=balance,
                             reference_change=float(reference_change),passed=bool(passed)))
            previous[name] = errors
    print(json.dumps(dict(times=TIMES.tolist(),wall_temperatures=WALLS.tolist(),source_endpoints=SOURCES.tolist(),
                          runs=rows,sources=[provenance(p) for p in (__file__,thermal.__file__,heat_1d.__file__)]),
                     indent=2,allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
