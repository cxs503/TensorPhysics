"""Check half-way D3Q7 film temperatures and heat flux against a Robin slab eigenmode."""
import json
import math

import numpy as np
import torch

from conduction_provenance import provenance
from tensorlbm import thermal_common as lbm


def measure(n, tau, beta, amplitude):
    dx, alpha, capacity = 1/n, .08, 1.7
    k = alpha*capacity
    h = k*beta*math.tan(beta/2)
    ambient = (.9,.15)
    steady_flux = (ambient[0]-ambient[1])/(2/h+1/k)
    slope = -steady_flux/k
    dt = (tau-.5)*dx**2/(4*alpha)
    x = (torch.arange(n,dtype=torch.float64)+.5).reshape(1,1,n)*dx
    ramp = ambient[0]-steady_flux/h+slope*x
    factor = amplitude*np.sinc(beta*dx/(2*np.pi))
    mode = factor*torch.cos(beta*(x-.5))
    field = ramp+mode
    zero = torch.zeros_like(field)
    grad = torch.stack((slope-factor*beta*torch.sin(beta*(x-.5)),zero,zero))
    g = lbm.thermal_gradient_initialization_3d(field,zero,zero,zero,grad,tau_T=tau,spacing=dx)
    rate = alpha*beta**2
    err_t = err_q = err_wall = balance = 0.
    for step in range(round(.09765625/dt)):
        post = lbm.thermal_collide_bgk_3d(g,field,zero,zero,zero,tau)
        g, flux = lbm.thermal_mixed_wall_pair_3d(lbm.thermal_stream_3d(post),"x",("film","film"),ambient,
            film_coefficients=(h,h),dt=dt,spacing=dx,volumetric_heat_capacity=capacity)
        updated = lbm.thermal_macroscopic_3d(g)
        decay = math.exp(-rate*step*dt)
        mean_decay = decay*(-math.expm1(-rate*dt))/(rate*dt)
        for side, direction in enumerate((1,-1)):
            exact_wall = ambient[side]-direction*steady_flux/h + amplitude*mean_decay*math.cos(beta/2)
            wall = 4*(g[1+side,...,0 if side==0 else -1]+post[2-side,...,0 if side==0 else -1])
            err_wall = max(err_wall,abs(wall.item()-exact_wall))
            err_q = max(err_q,abs(flux[side].item()-h*(ambient[side]-exact_wall)))
        residual = capacity*dx*(updated-field).sum().item()-dt*(flux[0]+flux[1]).item()
        balance = max(balance,abs(residual))
        err_t = max(err_t,float((updated-ramp-mode*math.exp(-rate*(step+1)*dt)).abs().max()))
        field = updated
    return dict(n=n,tau=tau,beta=beta,amplitude_K=amplitude,film_W_m2K=h,temperature_error_K=err_t,
                wall_temperature_error_K=err_wall,inward_flux_error_W_m2=err_q,balance_J_m2=balance)


if __name__ == "__main__":
    rows = []
    metrics = ["temperature_error_K","wall_temperature_error_K","inward_flux_error_W_m2"]
    for tau in (.75,1.5):
        for beta in (1.,2.4):
            for amplitude in (0.,.6):
                previous = None
                for n in (8,16,32):
                    row = measure(n,tau,beta,amplitude)
                    errors = np.array([row[key] for key in metrics])
                    row["orders"] = [] if previous is None or not amplitude else np.log2(previous/errors).tolist()
                    passed = row["balance_J_m2"] < 1e-13
                    if amplitude:
                        passed &= all(order > 1.3 for order in row["orders"])
                        passed &= n < 32 or np.all(errors < [1.5e-4, 8e-4, 7e-4])
                    else:
                        passed &= max(errors) < 1e-12
                    row["passed"] = bool(passed)
                    rows.append(row)
                    previous = errors
    print(json.dumps(dict(reference="Series-resistance steady slab plus cos(beta*(x-.5))*exp(-alpha*beta^2*t); h=k*beta*tan(beta/2)",
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorLBM/src python examples/lbm_film_slab.py",
        order_metrics=metrics,runs=rows,sources=[provenance(p) for p in (__file__,lbm.__file__)]),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
