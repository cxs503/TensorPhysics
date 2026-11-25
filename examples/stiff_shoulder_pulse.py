"""Time integration against exact semi-discrete evolution after two hot shoulders are released."""
import json

import numpy as np
from scipy.linalg import eigh

from conduction_provenance import provenance
from tensorfvm.cht import rod


def measure(steps,method):
    dx,area,k,c = map(np.array,([.3,.4,.3],[.5,2.,.8],[2.,3.,1.],[1.,1.,2.]))
    resistance,cw,h = map(np.array,([.05,.09],[.012,.024],[1.2,1.8]))
    exposed = abs(np.diff(area))
    g = 2*k*area/dx
    # Independent five-temperature circuit. Eliminate only the two massless
    # traces on the smaller sides by adding contact and half-cell resistances.
    links = np.array([1/(1/g[0]+resistance[0]/area[0]),g[1],g[1],
                      1/(resistance[1]/area[-1]+1/g[-1])])
    K = np.diag(np.r_[links,0.]+np.r_[0.,links])-np.diag(links,1)-np.diag(links,-1)
    K[0,0] += g[0]
    K[-1,-1] += g[-1]
    K[[1,3],[1,3]] += h*exposed
    mass = np.empty(5)
    mass[::2],mass[1::2] = c*area*dx,cw*exposed
    initial = np.array([0.,1.,0.,.4,0.])
    rates, modes = eigh(K,np.diag(mass))
    coeff = modes.T @ (mass*initial)
    time = np.linspace(0.,.1,steps+1)
    decay = np.exp(-time[:,None]*rates)
    exact = (decay*coeff) @ modes.T
    integral = (-np.expm1(-np.diff(time)[:,None]*rates)*decay[:-1]*coeff/rates) @ modes.T
    exact_contact = links[[0,3]]*(integral[:,[0,3]]-integral[:,[1,4]])
    samples = []
    def ends(t):
        samples.append(t)
        return [0.,0.]
    u,face,_,_,budget = rod.solve_transient_stepped_rod(dx,area,k,c,ends,time,0.,
                       initial_shoulder_temperature=initial[1::2],contact_resistance=resistance,
                       film_coefficient=h,shoulder_areal_heat_capacity=cw,method=method,return_step_heat=True)
    got = np.column_stack((u[:,0],face[:,1],u[:,1],face[:,2],u[:,2]))
    balance = (np.diff(got,axis=0) @ mass-budget["left"][:,0]+budget["right"][:,-1]
               -budget["ambient"].sum(axis=1))
    return dict(method=method,steps=steps,linear_solves=len(samples)-1,dt_times_fastest_rate=float(np.diff(time)[0]*rates[-1]),
                peak_temperature_error_K=float(abs(got-exact).max()),
                common_time_temperature_error_K=float(abs(got[::steps//4]-exact[::steps//4]).max()),minimum_temperature_K=float(got.min()),
                peak_step_contact_heat_error_J=float(abs(budget["contact"]-exact_contact).max()),
                cumulative_contact_heat_error_J=float(abs(np.cumsum(budget["contact"]-exact_contact,axis=0)).max()),
                max_energy_residual_J=float(abs(balance).max()))


if __name__ == "__main__":
    rows = []
    for method in ("theta","sdirk2","rannacher","positive_cn"):
        previous = None
        for steps in (4,8,16,32,64,128,256,512,1024,2048,4096):
            row = measure(steps,method)
            errors = {k:v for k,v in row.items() if "error" in k}
            rates = {} if previous is None else {k:np.log2(previous[k]/v) for k,v in errors.items()}
            row.update(orders=rates,passed=row["max_energy_residual_J"]<1e-12
                       and (method!="positive_cn" or row["minimum_temperature_K"]>=-1e-13)
                       and ((steps!=4096 and (steps!=512 or method=="rannacher")) or all(p>1.9 for p in rates.values())))
            rows.append(row)
            previous = errors
    print(json.dumps(dict(runs=rows,sources=[provenance(f) for f in (__file__,rod.__file__)]),
                     indent=2,allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
