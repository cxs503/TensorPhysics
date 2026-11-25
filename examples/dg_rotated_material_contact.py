"""Verify the two one-sided temperature gradients and fluxes at a planar material contact."""
import json

import numpy as np
import torch
from numpy.polynomial import legendre

from conduction_provenance import provenance
from tensordg import poisson_2d_bc as dg


def measure(n, degree, axis, resistance, sparse):
    # In local coordinates (s,t), s is normal to the interface, t tangential.
    left = (1.3, 2.2, .48)
    right = (3.1, .95, -.61)
    def field(x, y):
        s, t = (x, y) if axis == "x" else (y, x)
        east = s > .5
        kn = torch.where(east, torch.full_like(s, right[0]), left[0])
        cross = torch.where(east, torch.full_like(s, right[2]), left[2])
        flux = .7 - 1.6*t
        jump = resistance*east.to(s.dtype)
        tangent = .35 + 1.6*jump
        slope = (-flux-cross*tangent)/kn
        return (1.1+.35*t-jump*flux+(s-.5)*slope, -2*cross*1.6/kn)

    material = torch.empty(n, n, 3, dtype=torch.float64)
    for block, diag in ((material[:n//2], left), (material[n//2:], right)):
        block[:] = torch.tensor(diag, dtype=material.dtype)
    faces = torch.zeros(n-1, n, dtype=material.dtype)
    faces[n//2-1] = resistance
    if axis == "y":
        material = material.transpose(0,1)[..., [1,0,2]]
        faces = faces.T
    def temperature(x,y):
        return field(x,y)[0]
    opts = dict(conductivity=material, contact_resistance={axis: faces},
                dirichlet_values=dict.fromkeys(("left","right","bottom","top"), temperature))
    x,y,u,_ = dg.solve_poisson_2d_bc(n, degree, forcing=lambda x,y: field(x,y)[1],
                                     sparse=sparse, **opts)
    qx,qy = dg.poisson_face_powers_2d(u, **opts)
    # At each face midpoint the affine analytic flux equals its face average.
    t = (np.arange(n)+.5)/n
    target = .7-1.6*t
    coeff = u.numpy() if axis=="x" else u.numpy().transpose(1,0,3,2)
    traces, gradients, fluxes = [], [], []
    for index, endpoint, tensor, jump in ((n//2-1, 1., left, 0.), (n//2, -1., right, resistance)):
        block = coeff[index].transpose(1,2,0)
        def midpoint(values):
            return legendre.legval2d(endpoint, 0., values)
        traces.append(midpoint(block))
        normal = 2*n*midpoint(legendre.legder(block, axis=0))
        tangent = 2*n*midpoint(legendre.legder(block, axis=1))
        kn, _, cross = tensor
        gradients.append(np.max(abs(normal-(-target-cross*(.35+1.6*jump))/kn)))
        fluxes.append(-kn*normal-cross*tangent)
    numerical = (qx[n//2] if axis=="x" else qy[:,n//2]).numpy()*n
    centers = field(x[:,None], y[None,:])[1]
    return dict(n=n, degree=degree, axis=axis, resistance_m2K_W=resistance, sparse=sparse,
                temperature_l2_K=dg.l2_error_2d(x,y,u,temperature)["absolute_l2"],
                normal_gradient_error_K_m=float(max(gradients)),
                contact_jump_error_K=float(max(abs(traces[0]-traces[1]-resistance*target))),
                one_sided_flux_error_W_m2=float(np.max(abs(np.array(fluxes)-target))),
                numerical_flux_error_W_m2=float(max(abs(numerical-target))),
                max_cell_balance_W=float((qx.diff(dim=0)+qy.diff(dim=1)-centers/n**2).abs().max()))


if __name__ == "__main__":
    rows = [measure(n,p,axis,r,sparse) for n in (4,8) for p in (1,2) for axis in ("x","y")
            for r in (0.,.27) for sparse in (False,True)]
    for row in rows:
        row["passed"] = max(v for k,v in row.items() if "error" in k or "balance" in k or "l2" in k) < 1e-9
    print(json.dumps(dict(reference="Piecewise bilinear field with normal flux .7-1.6*t and jump R*q at s=.5",
        reproduction="OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=../TensorDG python examples/dg_rotated_material_contact.py",
        sources=[provenance(p) for p in (__file__, dg.__file__)], runs=rows), indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
