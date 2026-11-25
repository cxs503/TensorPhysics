"""Decay of a continuum eigenmode between unequal convective end films."""
import json

import numpy as np

from conduction_provenance import provenance
from tensorfvm.cht import rod


def measure(n, method):
    length, area, k, capacity = 1., .7, 1.4, 2.
    wave, peak = 1.8, .38
    h = k*wave*np.tan(wave*np.array([peak,length-peak]))
    x = np.linspace(0.,length,n+1)
    dx = np.diff(x)
    # Integrate the cosine over each control volume to obtain exact cell means.
    shape = np.diff(np.sin(wave*(x-peak)))/(wave*dx)
    time = np.linspace(0.,.3,n+1)
    decay = np.exp(-time*k*wave**2/capacity)
    cell,faces,left,right,heat = rod.solve_transient_stepped_rod(
        dx,area,k,capacity,[0.,0.],time,shape,end_types=("robin","robin"),
        end_film_coefficient=h,method=method,return_step_heat=True)
    wall = faces[:,[0,-1]]
    film_input = -area*h*wall
    energy_error = np.diff(cell @ (capacity*area*dx))-heat["left"][:,0]+heat["right"][:,-1]
    return dict(n=n,method=method,
                cell_error_K=float(abs(cell-decay[:,None]*shape).max()),
                wall_error_K=float(abs(wall-decay[:,None]*np.cos(wave*(x[[0,-1]]-peak))).max()),
                film_residual_W=float(abs(np.column_stack((left[:,0],-right[:,-1]))-film_input).max()),
                energy_residual_J=float(abs(energy_error).max()))


if __name__ == "__main__":
    rows = []
    for method in ("theta","sdirk2","rannacher","positive_cn"):
        previous = None
        for n in (12,24,48,96):
            row = measure(n,method)
            errors = {key:value for key,value in row.items() if key.endswith("_error_K")}
            rates = {} if previous is None else {key:np.log2(previous[key]/v) for key,v in errors.items()}
            row.update(orders=rates,passed=all(v>1.8 for v in rates.values())
                       and row["film_residual_W"]<1e-10 and row["energy_residual_J"]<1e-10)
            rows.append(row)
            previous = errors
    print(json.dumps(dict(runs=rows,sources=[provenance(p) for p in (__file__,rod.__file__)]),
                     indent=2,allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
