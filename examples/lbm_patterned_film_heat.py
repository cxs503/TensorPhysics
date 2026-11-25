"""Check both local Robin traces on a periodic strip with spatially varying film coefficients."""
import json
import math

import numpy as np
import torch

from conduction_provenance import provenance
from tensorlbm import thermal_common as lbm


def measure(n, tau, mode):
    dx, alpha, capacity = 1/n, .08, 1.7
    dt = (tau-.5)*dx**2/(4*alpha)
    beta, gamma = 2.4, 2*math.pi*mode
    rate = alpha*(beta**2+gamma**2)
    x = (torch.arange(n,dtype=torch.float64)+.5)*dx
    y = x[:,None]
    lateral_phase = gamma*y+.31
    lateral = torch.cos(lateral_phase)*np.sinc(mode/n)
    normal_phase = beta*x+.24
    normal = torch.sin(normal_phase)*np.sinc(beta*dx/(2*np.pi))
    shape = .6*lateral*normal
    ramp = .5+.2*x
    field = (ramp+shape).unsqueeze(0)
    zero = torch.zeros_like(field)
    gx = .2+.6*beta*lateral*torch.cos(normal_phase)*np.sinc(beta*dx/(2*np.pi))
    gy = -.6*gamma*torch.sin(lateral_phase)*np.sinc(mode/n)*normal
    grad = torch.stack((gx[None],gy[None],zero))
    g = lbm.thermal_gradient_initialization_3d(field,zero,zero,zero,grad,tau_T=tau,spacing=dx)
    # Each wall row has a constant positive h, sampled at its midpoint.
    h = (.75+.42*torch.cos(2*math.pi*x),1.2+.68*torch.sin(2*math.pi*x))
    h = tuple(v[None] for v in h)
    lateral_face = lateral.T
    err_t = err_wall = err_q = balance = 0.
    for step in range(round(.09765625/dt)):
        decay = math.exp(-step*rate*dt)
        mean = decay*(-math.expm1(-rate*dt))/(rate*dt)
        target_t, target_q = [], []
        for side, sign in ((0,-1),(1,1)):
            angle = beta*side+.24
            target_t.append(.5+.2*side+.6*mean*math.sin(angle)*lateral_face)
            target_q.append(sign*capacity*alpha*(.2+.6*mean*beta*math.cos(angle)*lateral_face))
        # Ambient input enforces the manufactured, face- and step-averaged Robin data.
        ambient = tuple(t+q/film for t,q,film in zip(target_t,target_q,h))
        post = lbm.thermal_collide_bgk_3d(g,field,zero,zero,zero,tau)
        fx,fy,_ = lbm.thermal_link_fluxes_3d(post,dt=dt,spacing=dx,volumetric_heat_capacity=capacity,periodic="yz")
        g,flux = lbm.thermal_mixed_wall_pair_3d(lbm.thermal_stream_3d(post),"x",("film","film"),
            ambient,film_coefficients=h,dt=dt,spacing=dx,volumetric_heat_capacity=capacity)
        updated = lbm.thermal_macroscopic_3d(g)
        for side,index in ((0,0),(1,-1)):
            wall = 4*(g[1+side,...,index]+post[2-side,...,index])
            err_wall = max(err_wall,float((wall-target_t[side]).abs().max()))
            err_q = max(err_q,float((flux[side]-target_q[side]).abs().max()))
        div = torch.cat((flux[0][...,None],fx,-flux[1][...,None]),dim=2).diff(dim=2)+fy-fy.roll(1,1)
        balance = max(balance,float((capacity*dx**3*(updated-field)+dt*dx**2*div).abs().max()))
        err_t = max(err_t,float((updated-ramp-shape*math.exp(-(step+1)*rate*dt)).abs().max()))
        field = updated
    return dict(n=n,tau=tau,transverse_mode=mode,temperature_error_K=err_t,
        wall_temperature_error_K=err_wall,wall_flux_error_W_m2=err_q,max_cell_balance_J=balance)


if __name__ == "__main__":
    rows = []
    metrics = ["temperature_error_K","wall_temperature_error_K","wall_flux_error_W_m2"]
    for tau in (.75,1.5):
        for mode in (1,2):
            previous = None
            for n in (8,16,32,64):
                row = measure(n,tau,mode)
                errors = np.array([row[k] for k in metrics])
                row["orders"] = [] if previous is None else np.log2(previous/errors).tolist()
                row["passed"] = bool(row["max_cell_balance_J"] < 1e-14
                    and all(order > 0 for order in row["orders"])
                    and (n < 64 or (min(row["orders"]) > 1.8
                                    and np.all(errors < [.0035, .0035, .0045]))))
                rows.append(row)
                previous = errors
    print(json.dumps(dict(reference="Linear background plus decaying sin(beta*x+.24)*cos(2*pi*m*y+.31); independent local Robin ambient data on both x walls",
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorLBM/src python examples/lbm_patterned_film_heat.py",
        order_metrics=metrics,runs=rows,sources=[provenance(p) for p in (__file__,lbm.__file__)]),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
