"""Decay of a continuum eigenmode between unequal convective end films."""
import argparse
import json

import numpy as np

from conduction_provenance import provenance
from tensorfvm.cht import rod


def measure(n, method, end_storage=False):
    length, area, k, capacity = 1., .7, 1.4, 2.
    wave, peak = 1.8, .38
    h = k*wave*np.tan(wave*np.array([peak,length-peak]))
    wall_capacity = np.array([.3,.8]) if end_storage else np.zeros(2)
    # For this decaying eigenmode the exterior film must also remove the
    # heat released by the surface at rate (k*wave**2/capacity)*Cwall*T.
    h += (k*wave**2/capacity)*wall_capacity
    x = np.linspace(0.,length,n+1)
    dx = np.diff(x)
    # Integrate the cosine over each control volume to obtain exact cell means.
    shape = np.diff(np.sin(wave*(x-peak)))/(wave*dx)
    time = np.linspace(0.,.3,n+1)
    decay = np.exp(-time*k*wave**2/capacity)
    cell,faces,left,right,heat = rod.solve_transient_stepped_rod(
        dx,area,k,capacity,[0.,0.],time,shape,end_types=("robin","robin"),
        end_film_coefficient=h,end_areal_heat_capacity=wall_capacity,
        initial_end_temperature=np.cos(wave*(x[[0,-1]]-peak)),method=method,return_step_heat=True)
    wall = faces[:,[0,-1]]
    energy_error = np.diff(cell @ (capacity*area*dx))-heat["left"][:,0]+heat["right"][:,-1]
    stored = area*wall_capacity*np.diff(wall,axis=0)
    into_rod = np.column_stack((heat["left"][:,0],-heat["right"][:,-1]))
    wall_balance = stored+into_rod-heat["end_input"]
    # Integrate exterior film input from the analytic wall temperature.
    rate = k*wave**2/capacity
    exact_input = area*h*np.cos(wave*(x[[0,-1]]-peak))*np.diff(decay)[:,None]/rate
    row = dict(n=n,method=method,end_storage=end_storage,
                cumulative_input_error_J=float(abs(np.cumsum(heat["end_input"]-exact_input,axis=0)).max()),
                cell_error_K=float(abs(cell-decay[:,None]*shape).max()),
                wall_error_K=float(abs(wall-decay[:,None]*np.cos(wave*(x[[0,-1]]-peak))).max()),
                wall_residual_J=float(abs(wall_balance).max()),
                energy_residual_J=float(abs(energy_error).max()))
    if end_storage:
        row["final_cell_error_K"] = float(abs(cell[-1]-decay[-1]*shape).max())
    else:
        film_input = -area*h*wall
        row["film_residual_W"] = float(abs(np.column_stack((left[:,0],-right[:,-1]))-film_input).max())
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end-storage",action="store_true")
    args = parser.parse_args()
    rows = []
    for method in ("theta","sdirk2","rannacher","positive_cn"):
        previous = None
        for n in (12,24,48,96):
            row = measure(n,method,args.end_storage)
            errors = {key:value for key,value in row.items() if "_error_" in key}
            rates = {} if previous is None else {key:np.log2(previous[key]/v) for key,v in errors.items()}
            # A storing boundary introduces a short initial flux transient for
            # the first-order half-cell reconstruction. Track its maximum
            # temperature error separately from the final-time convergence.
            row.update(orders=rates,passed=all(v>(1.5 if args.end_storage and key=="cell_error_K" else 1.8)
                                               for key,v in rates.items())
                       and row.get("film_residual_W",0.)<1e-10
                       and row["wall_residual_J"]<1e-10 and row["energy_residual_J"]<1e-10)
            rows.append(row)
            previous = errors
    print(json.dumps(dict(runs=rows,sources=[provenance(p) for p in (__file__,rod.__file__)]),
                     indent=2,allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
