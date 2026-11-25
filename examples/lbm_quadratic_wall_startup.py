"""Resolve the first-step Dirichlet flux defect for cell averages of quadratic temperature fields."""
import json

import numpy as np
import torch

from conduction_provenance import provenance
from tensorlbm import thermal_common as lbm


def measure(n, tau, curvature):
    dx, alpha, capacity = 1/n, .08, 1.7
    dt = (tau-.5)*dx**2/(4*alpha)
    coord = (torch.arange(n,dtype=torch.float64)+.5)*dx
    z,y,x = torch.meshgrid(coord,coord,coord,indexing="ij")
    xyz = (x,y,z)
    slopes = (.27,-.38,.15)
    laplacian = sum(curvature)
    field = .45+sum(s*v+.5*c*(v*v+dx**2/12) for s,c,v in zip(slopes,curvature,xyz))
    gradient = torch.stack([s+c*v for s,c,v in zip(slopes,curvature,xyz)])
    zero = torch.zeros_like(field)
    g = lbm.thermal_gradient_initialization_3d(field,zero,zero,zero,gradient,tau_T=tau,spacing=dx)
    post = lbm.thermal_collide_bgk_3d(g,field,zero,zero,zero,tau)
    g = lbm.thermal_stream_3d(post)
    units = dict(dt=dt,spacing=dx,volumetric_heat_capacity=capacity)
    gain = capacity*dx/(4*dt)
    diagnostics = []
    for axis,(c,s) in enumerate(zip(curvature,slopes)):
        dim = 2-axis
        wall_values = []
        for side, index in ((0,0),(1,-1)):
            center = xyz[axis].select(dim,index)
            wall = field.select(dim,index)+s*(side-center)+.5*c*(side**2-center**2-dx**2/12)
            wall_values.append(wall+.5*dt*alpha*laplacian)
        g, flux = lbm.thermal_temperature_wall_pair_3d(g,"xyz"[axis],wall_values,**units)
        # Normal cell average contributes H_nn*dx²/6. The postcollision
        # gradient correction samples the cell center, another (tau-1)*H_nn*dx²/2.
        # Averaging wall temperature over the step contributes alpha*dt*trace(H)/2.
        normal = gain*(alpha*dt/2 - dx**2*(1/6+(tau-1)/2))*c
        tangential = gain*alpha*dt*(laplacian-c)/2
        exact = [-capacity*alpha*s, capacity*alpha*(s+c)]
        errors = [value-target for value,target in zip(flux,exact)]
        diagnostics.append(dict(axis="xyz"[axis],normal_contribution_W_m2=normal,
            tangential_contribution_W_m2=tangential,
            mean_signed_wall_errors_W_m2=[float(err.mean()) for err in errors],
            peak_error_W_m2=max(float(err.abs().max()) for err in errors),
            prediction_residual_W_m2=max(float((err-normal-tangential).abs().max()) for err in errors)))
    return dict(n=n,tau=tau,diagonal_curvature_K_m2=curvature,axes=diagnostics,
                passed=all(v["prediction_residual_W_m2"] < 1e-12 for v in diagnostics))


if __name__ == "__main__":
    rows = [measure(n,tau,c) for tau in (.75,1.,1.5) for c in
            ((0.,0.,0.),(.8,0.,0.),(0.,-.45,.7),(.8,-.45,.7)) for n in (8,16,32)]
    print(json.dumps(dict(reference="T = affine(x) + sum(Hii*xi²)/2 + alpha*t*trace(H); exact cell averages, step-mean wall data and gradient-initialized BGK populations",
        reproduction="OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=../TensorLBM/src python examples/lbm_quadratic_wall_startup.py",
        runs=rows,sources=[provenance(p) for p in (__file__,lbm.__file__)]),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
