"""Cross-solver interval heat and temperature under independently switched walls and volume heating."""
import json

import numpy as np
from numpy.polynomial import Polynomial
import torch
from tensorfem import thermal
from tensordg import heat_1d
from tensorfvm.cht import rod
from tensorsph import heat_conduction_2d as sph
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
    mid = (np.arange(n)+.5)/n
    points = [np.zeros(n)]
    point_sine = np.sin(mid[:,None]*k)
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
        points.append(poly(mid)+point_sine@transient)
        moment = (poly*(1-x)).integ()(1.)+np.sum(transient/k)
        q = profile(SOURCES[i],x)
        deposited = dt*np.diff(q.integ()(edges))
        # Test the continuum heat balance with 1-x to recover net west input.
        conduction = dt*np.mean(WALLS[i:i+2,0]-WALLS[i:i+2,1])
        west_heat = moment-west_moment+conduction-dt*((1-x)*q).integ()(1.)
        heats.append(west_heat+np.r_[0., np.cumsum(deposited-(means[-1]-means[-2])/n)])
        west_moment = moment
    return np.asarray(means),np.asarray(heats),np.asarray(points)


def fourier_section_heat(n, modes=4096):
    """Direct time integral of -dT/dx from the Fourier expansion."""
    edges = np.linspace(0.,1.,n+1)
    m = np.arange(1,modes+1)
    k = m*np.pi
    cosine = np.cos(edges[:,None]*k)
    previous = np.zeros(modes)
    heats = []
    for i,dt in enumerate(np.diff(TIMES)):
        a, end = SOURCES[i]-(WALLS[i+1]-WALLS[i])/dt
        b = end-a
        stationary = 2*(a-(-1.)**m*end)/k**3
        gradient = a/2+b/6-a*edges-b*edges**2/2
        wall_slope = np.mean(WALLS[i:i+2,1]-WALLS[i:i+2,0])
        increment = (previous-stationary)*np.expm1(-dt*k*k)
        heats.append(-dt*(gradient+wall_slope)+cosine@(increment/k))
        previous += increment
    return np.asarray(heats)


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


def cell_solver(n, method):
    mid = (np.arange(n)+.5)/n
    count = 2*n
    times = np.r_[TIMES[0],np.concatenate([np.linspace(a,b,count+1)[1:] for a,b in zip(TIMES,TIMES[1:])])]
    def source(t):
        i = np.searchsorted(TIMES[1:-1],t,side="right")
        return profile(SOURCES[i],mid)
    def wall(t):
        return tuple(np.interp(t,TIMES,g) for g in WALLS.T)
    if method=="FVM":
        u,_,_,_,heat = rod.solve_transient_stepped_rod(np.full(n,1/n),1.,1.,1.,wall,times,np.zeros(n),
            source=source,method="sdirk2",flux_order=2,return_step_heat=True,load_jump_times=TIMES[1:-1])
        sections = np.column_stack((heat["left"][:,0],heat["right"]))
    else:
        u,sections = sph.solve_transient_layered_slab(np.zeros(n),times,1/n,(lambda t: wall(t)[0],lambda t: wall(t)[1]),conductivity=np.ones(n),
            volumetric_heat_capacity=1.,heat_source=source,method="sdirk2",return_x_heat=True,
            load_jump_times=TIMES[1:-1])
    return np.asarray(u)[::count],np.asarray(sections).reshape(len(TIMES)-1,count,n+1).sum(axis=1)


if __name__ == "__main__":
    rows = []
    previous = {}
    for n in (8,16,32,64):
        means,heat,points = reference(n,64)
        check = reference(n,32)
        reference_change = max(np.max(abs(check[0]-means)),np.max(abs(check[1]-heat)),np.max(abs(check[2]-points)))
        flux_route_difference = float(np.max(abs(fourier_section_heat(n)-heat)))
        mid = (np.arange(n)+.5)/n
        deposited = np.diff(TIMES)[:,None]*np.stack([profile(q,mid) for q in SOURCES])/n
        for name,solve in (("FEM_LINE2_SDIRK2",fem),("DG_P2_exponential",dg),
                           ("FVM_SDIRK2",lambda n: cell_solver(n,"FVM")),
                           ("SPH_SDIRK2",lambda n: cell_solver(n,"SPH"))):
            u,h = solve(n)
            expected = points if name=="SPH_SDIRK2" else means
            errors = dict(temperature=float(np.max(abs(u-expected))),
                          section_heat=float(np.max(abs(h-heat))),
                          wall_heat=float(np.max(abs(h[:,[0,-1]]-heat[:,[0,-1]]))))
            orders = {k:float(np.log2(previous[name][k]/v)) for k,v in errors.items()} if name in previous else {}
            balance = float(np.max(abs(np.diff(u,axis=0)/n+np.diff(h)-deposited)))
            min_order = 3.6 if name=="DG_P2_exponential" else 1.7
            passed = (reference_change<1e-12 and flux_route_difference<1e-10 and balance<1e-10
                      and all(o>min_order for o in orders.values()))
            rows.append(dict(solver=name,cells=n,temperature_sampling="center" if name=="SPH_SDIRK2" else "cell_mean",errors=errors,orders=orders,max_cell_balance=balance,
                             reference_change=float(reference_change),flux_route_difference=flux_route_difference,passed=bool(passed)))
            previous[name] = errors
    print(json.dumps(dict(times=TIMES.tolist(),wall_temperatures=WALLS.tolist(),source_endpoints=SOURCES.tolist(),
                          runs=rows,sources=[provenance(p) for p in (__file__,thermal.__file__,heat_1d.__file__,rod.__file__,sph.__file__)]),
                     indent=2,allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
