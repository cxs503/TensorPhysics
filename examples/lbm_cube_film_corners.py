"""Verify the heat budget of cells at the intersection of three convective walls."""
import json
import math

import numpy as np
import torch

from conduction_provenance import provenance
from tensorlbm import thermal_common as lbm


def measure(n, tau):
    dx, alpha, capacity = 1/n, .08, 1.7
    dt = (tau-.5)*dx**2/(4*alpha)
    beta = np.array([1.1, 1.75, 2.3])
    slope = np.array([.2, -.35, .16])
    h = capacity*alpha*beta*np.tan(beta/2)
    rate = alpha*(beta @ beta)
    coord = (torch.arange(n, dtype=torch.float64)+.5)*dx
    z,y,x = torch.meshgrid(coord, coord, coord, indexing="ij")
    xyz = (x,y,z)
    ramp = .4+sum(s*v for s,v in zip(slope,xyz))
    cosine = [torch.cos(b*(v-.5))*np.sinc(b*dx/(2*np.pi)) for b,v in zip(beta,xyz)]
    mode = .6*cosine[0]*cosine[1]*cosine[2]
    field = ramp+mode
    zero = torch.zeros_like(field)
    grad = torch.stack([s-mode*b*torch.tan(b*(v-.5)) for s,b,v in zip(slope,beta,xyz)])
    g = lbm.thermal_gradient_initialization_3d(field,zero,zero,zero,grad,tau_T=tau,spacing=dx)
    units = dict(dt=dt,spacing=dx,volumetric_heat_capacity=capacity)
    walls = []
    for a,b in enumerate(beta):
        dim = 2-a
        pair = []
        for side,index in ((0,0),(1,-1)):
            sign = 2*side-1
            wall_ramp = ramp.select(dim,index)+sign*.5*dx*slope[a]
            # The separated cosine mode satisfies homogeneous Robin data on each wall.
            ambient = wall_ramp+sign*capacity*alpha*slope[a]/h[a]
            wall_mode = mode.select(dim,index)*math.cos(b/2)/cosine[a].select(dim,index)
            pair.append((ambient, wall_ramp, wall_mode))
        walls.append(pair)
    corner = [(i,j,k) for i in (0,-1) for j in (0,-1) for k in (0,-1)]
    temp_error = flux_error = corner_balance = balance = sequence_error = 0.
    steps = round(.09765625/dt)
    for step in range(steps):
        post = lbm.thermal_collide_bgk_3d(g,field,zero,zero,zero,tau)
        interior = lbm.thermal_link_fluxes_3d(post,periodic=False,**units)
        streamed = lbm.thermal_stream_3d(post)
        g = streamed
        div = torch.zeros_like(field)
        average = math.exp(-rate*step*dt)*(-math.expm1(-rate*dt))/(rate*dt)
        for a,pair in enumerate(walls):
            dim = 2-a
            ambient = tuple(v[0] for v in pair)
            g,flux = lbm.thermal_mixed_wall_pair_3d(g,"xyz"[a],("film","film"),ambient,
                film_coefficients=(h[a],h[a]),**units)
            for side, (drive, base, surface) in enumerate(pair):
                target = h[a]*(drive-base-average*surface)
                flux_error = max(flux_error, float((flux[side]-target).abs().max()))
            faces = torch.cat((flux[0].unsqueeze(dim),interior[a],-flux[1].unsqueeze(dim)),dim)
            div += faces.diff(dim=dim)
        if step == 0:
            reverse = streamed
            for a in (2,1,0):
                reverse,_ = lbm.thermal_mixed_wall_pair_3d(reverse,"xyz"[a],("film","film"),
                    tuple(v[0] for v in walls[a]),film_coefficients=(h[a],h[a]),**units)
            sequence_error = float((g-reverse).abs().max())
        updated = lbm.thermal_macroscopic_3d(g)
        residual = capacity*dx**3*(updated-field)+dt*dx**2*div
        balance = max(balance,float(residual.abs().max()))
        corner_balance = max(corner_balance,max(abs(residual[idx].item()) for idx in corner))
        temp_error = max(temp_error,float((updated-ramp-mode*math.exp(-rate*(step+1)*dt)).abs().max()))
        field = updated
    return dict(n=n,tau=tau,steps=steps,temperature_error_K=temp_error,wall_flux_error_W_m2=flux_error,
                max_cell_balance_J=balance,max_corner_balance_J=corner_balance,axis_order_error_K=sequence_error)


if __name__ == "__main__":
    rows = []
    for tau in (.75,1.5):
        previous = None
        for n in (8,16,32):
            row = measure(n,tau)
            errors = np.array([row["temperature_error_K"],row["wall_flux_error_W_m2"]])
            row["orders"] = [] if previous is None else np.log2(previous/errors).tolist()
            row["passed"] = bool(row["max_cell_balance_J"] < 1e-14
                and row["max_corner_balance_J"] < 1e-14 and row["axis_order_error_K"] == 0.
                and all(order > 1. for order in row["orders"])
                and (n < 32 or np.all(errors < [1.5e-4, 3.5e-4])))
            rows.append(row)
            previous = errors
    print(json.dumps(dict(reference="Unit cube: affine temperature plus separated decaying cosine mode with h_i=k*beta_i*tan(beta_i/2)",
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorLBM/src python examples/lbm_cube_film_corners.py",
        order_metrics=["temperature_error_K","wall_flux_error_W_m2"],runs=rows,
        sources=[provenance(p) for p in (__file__,lbm.__file__)]),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
