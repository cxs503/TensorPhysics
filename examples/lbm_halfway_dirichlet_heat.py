"""Verify cell-centered D3Q7 wall heat against a decaying slab mode and a steady ramp."""
import json
import math

import numpy as np
import torch

from conduction_provenance import provenance
from tensorlbm import thermal_common as lbm


def measure(n, mode):
    dx, alpha, capacity = 1/n, .08, 1.7
    dt = dx**2/(8*alpha)
    x = (torch.arange(n, dtype=torch.float64)+.5)*dx
    b = 2*math.pi*mode
    rate = alpha*(math.pi**2+b*b)
    lateral = torch.cos(b*x)*np.sinc(mode/n)
    shape = (lateral[:,None]*torch.sin(math.pi*x)*np.sinc(.5/n)).unsqueeze(0)
    ramp = .4+.3*x
    field = ramp+shape
    zero = torch.zeros_like(field)
    g = lbm.thermal_equilibrium_3d(field,zero,zero,zero)
    max_flux = max_temp = max_balance = 0.
    for step in range(n*n//16):
        post = lbm.thermal_collide_bgk_3d(g,field,zero,zero,zero,1.)
        fx, fy, fz = lbm.thermal_link_fluxes_3d(post, dt=dt, spacing=dx, volumetric_heat_capacity=capacity, periodic="yz")
        g, (low,high) = lbm.thermal_temperature_wall_pair_3d(lbm.thermal_stream_3d(post), "x", (.4,.7),
            dt=dt,spacing=dx,volumetric_heat_capacity=capacity)
        updated = lbm.thermal_macroscopic_3d(g)
        decay = math.exp(-rate*dt*step)
        average = decay*(-math.expm1(-rate*dt))/(rate*dt)
        for wall, slope in ((low,.3),(high,-.3)):
            exact = -capacity*alpha*(slope+math.pi*average*lateral)
            max_flux = max(max_flux, float((wall-exact).abs().max()))
        div = torch.cat((low[...,None], fx, -high[...,None]), dim=2).diff(dim=2)
        div += fy-fy.roll(1,1)+fz-fz.roll(1,0)
        max_balance = max(max_balance, float((capacity*dx**3*(updated-field)+dt*dx**2*div).abs().max()))
        max_temp = max(max_temp,float((updated-ramp-decay*math.exp(-rate*dt)*shape).abs().max()))
        field = updated
    return dict(n=n, transverse_mode=mode, max_wall_flux_error_W_m2=max_flux,
                max_temperature_error_K=max_temp, max_cell_balance_J=max_balance)


if __name__ == "__main__":
    rows = []
    for mode in (0,1):
        previous = None
        for n in (8,16,32):
            row = measure(n, mode)
            errors = np.array([row["max_wall_flux_error_W_m2"], row["max_temperature_error_K"]])
            orders = [] if previous is None else np.log2(previous/errors).tolist()
            row.update(orders=orders, passed=bool(all(v > 1.7 for v in orders) and row["max_cell_balance_J"] < 1e-14))
            rows.append(row)
            previous = errors
    print(json.dumps(dict(reference="T=.4+.3*x+exp(-alpha*(pi^2+b^2)*t)*sin(pi*x)*cos(b*y); compare cell and time-averaged face quantities",
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorLBM/src python examples/lbm_halfway_dirichlet_heat.py",
        order_metrics=["max_wall_flux_error_W_m2","max_temperature_error_K"], runs=rows,
        sources=[provenance(p) for p in (__file__,lbm.__file__)]), indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
