"""Verify each interval of reversing heat flow through an insulated slab's internal faces."""
import json
import math

import numpy as np

from conduction_provenance import provenance
from tensordg import poisson_2d_bc, transient_diffusion_2d


def measure(steps, method):
    nx, ny = 4, 6
    kx, ky, capacity = 1.8, .7, 2.1
    period = .4
    omega = 2*math.pi/period
    # Unequal intervals include steps straddling a flux reversal.
    fraction = np.linspace(0., 1., steps+1)
    times = period*(fraction+.12*np.sin(2*np.pi*fraction)/(2*np.pi))

    def shape(x, y):
        return x*x*(3-2*x) + .6*y*y*(3-2*y)

    def source(t, x, y):
        diffusion = kx*(6-12*x) + .6*ky*(6-12*y)
        return -capacity*omega*math.sin(omega*t)*shape(x,y) - math.cos(omega*t)*diffusion

    bc = dict.fromkeys(("left","right","top","bottom"), "neumann")
    _, _, states, means, deposited = transient_diffusion_2d.solve_transient_diffusion_2d(
        shape, times, ncells=(nx,ny), degree=3, capacity=capacity, conductivity=(kx,ky),
        bc=bc, source_history=source, return_history_heat=True, return_interval_means=True, method=method)
    stored = capacity*states[...,0,0].diff(dim=0)/(nx*ny)
    dt = np.diff(times)
    integral = np.diff(np.sin(omega*times))/omega
    # Each flux is constant along the face. Check every face in both directions.
    xf = np.arange(nx+1)/nx
    yf = np.arange(ny+1)/ny
    px = -6*kx*xf*(1-xf)/ny
    py = -3.6*ky*yf*(1-yf)/nx
    max_power_error = max_balance = 0.
    midplane = []
    for i, mean in enumerate(means):
        qx, qy = poisson_2d_bc.poisson_face_powers_2d(mean, bc=bc, conductivity=(kx,ky))
        factor = integral[i]/dt[i]
        max_power_error = max(max_power_error, float(np.max(abs(qx.numpy()-factor*px[:,None]))),
                              float(np.max(abs(qy.numpy()-factor*py[None,:]))))
        residual = stored[i] + dt[i]*(qx.diff(dim=0)+qy.diff(dim=1)) - deposited[i]
        max_balance = max(max_balance, float(residual.abs().max()))
        midplane.append(float(qx[nx//2].sum())*dt[i])
    exact_cut = -1.5*kx*integral
    numerical = np.asarray(midplane)
    return dict(steps=steps, method=method, max_interval_face_mean_power_error_W=max_power_error,
                max_interval_cell_balance_J=max_balance, midplane_net_heat_J=float(numerical.sum()),
                midplane_absolute_heat_J=float(abs(numerical).sum()),
                max_midplane_interval_error_J=float(max(abs(numerical-exact_cut))),
                max_cumulative_midplane_error_J=float(max(abs(np.cumsum(numerical-exact_cut)))))


if __name__ == "__main__":
    rows = []
    for method in ("theta", "sdirk2"):
        previous = None
        for steps in (8,16,32,64):
            row = measure(steps,method)
            error = row["max_interval_face_mean_power_error_W"]
            row["power_order"] = None if previous is None else math.log2(previous/error)
            row["passed"] = (row["max_interval_cell_balance_J"] < 1e-12
                             and (previous is None or row["power_order"] > 1.8)
                             and (steps < 64 or (error < 8e-4
                                  and row["max_cumulative_midplane_error_J"] < 5e-4
                                  and row["max_midplane_interval_error_J"] < 3.5e-5)))
            rows.append(row)
            previous = error
    print(json.dumps(dict(reference="T=cos(omega*t)*(3*x^2-2*x^3+.6*(3*y^2-2*y^3)), insulated boundaries, one period",
        reproduction="OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=../TensorDG python examples/dg_reversing_interval_heat.py",
        runs=rows,sources=[provenance(p) for p in (__file__,poisson_2d_bc.__file__,transient_diffusion_2d.__file__)]),
        indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
