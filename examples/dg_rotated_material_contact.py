"""Verify the two one-sided temperature gradients and fluxes at a planar material contact."""
import argparse
import json

import numpy as np
import torch

from conduction_provenance import provenance
from tensordg import poisson_2d_bc as dg


def measure(n, degree, axis, resistance, sparse, *, sinusoidal=False):
    # In local coordinates (s,t), s is normal to the interface, t tangential.
    left = (1.3, 2.2, .48)
    right = (3.1, .95, -.61)
    def field(x, y):
        s, t = (x, y) if axis == "x" else (y, x)
        east = s > .5
        kn = torch.where(east, torch.full_like(s, right[0]), left[0])
        cross = torch.where(east, torch.full_like(s, right[2]), left[2])
        jump = resistance*east.to(s.dtype)
        offset = s-.5
        if sinusoidal:
            kt = torch.where(east, torch.full_like(s, right[1]), left[1])
            b = 2*torch.pi
            flux = torch.sin(b*t)
            derivative = b*torch.cos(b*t)
            tangent = .35-jump*derivative
            slope = (-flux-cross*tangent)/kn
            slope_t = (-derivative-cross*jump*b**2*flux)/kn
            slope_tt = b**2*(flux-cross*jump*derivative)/kn
            source = -2*cross*slope_t-kt*(jump*b**2*flux+offset*slope_tt)
        else:
            flux = .7 - 1.6*t
            tangent = .35 + 1.6*jump
            slope = (-flux-cross*tangent)/kn
            source = -2*cross*1.6/kn
        return 1.1+.35*t-jump*flux+offset*slope, source

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
    # DG traces below sample midpoints; the numerical flux integrates the whole face.
    t = (np.arange(n)+.5)/n
    target = np.sin(2*np.pi*t) if sinusoidal else .7-1.6*t
    derivative = 2*np.pi*np.cos(2*np.pi*t) if sinusoidal else -1.6
    face_factor = np.sinc(1/n) if sinusoidal else 1.
    traces, normal, fluxes = (v[...,0].numpy() for v in
        dg.poisson_interface_traces_2d(u, axis, n//2, conductivity=material))
    gradients = []
    for side, tensor, jump in ((0, left, 0.), (1, right, resistance)):
        kn, _, cross = tensor
        gradients.append(np.max(abs(normal[side]-(-target-cross*(.35-derivative*jump))/kn)))
    numerical = (qx[n//2] if axis=="x" else qy[:,n//2]).numpy()*n
    # The source is affine in s and a single Fourier mode in t for the sinusoidal case.
    centers = field(x[:,None], y[None,:])[1]
    return dict(n=n, degree=degree, axis=axis, resistance_m2K_W=resistance, sparse=sparse,
                temperature_l2_K=dg.l2_error_2d(x,y,u,temperature)["absolute_l2"],
                normal_gradient_error_K_m=float(max(gradients)),
                contact_jump_error_K=float(max(abs(traces[0]-traces[1]-resistance*target))),
                one_sided_flux_error_W_m2=float(np.max(abs(np.array(fluxes)-target))),
                numerical_flux_error_W_m2=float(max(abs(numerical-target*face_factor))),
                max_cell_balance_W=float((qx.diff(dim=0)+qy.diff(dim=1)-centers*face_factor/n**2).abs().max()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sinusoidal", action="store_true", help="refine a sinusoidal interface flux")
    sinusoidal = parser.parse_args().sinusoidal
    command = "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=../TensorDG python examples/dg_rotated_material_contact.py"
    if sinusoidal:
        reference = "Piecewise affine in s, sinusoidal interface flux sin(2*pi*t) and jump R*q at s=.5"
        command += " --sinusoidal"
        rows = []
        metrics = ["temperature_l2_K", "normal_gradient_error_K_m", "contact_jump_error_K",
                   "one_sided_flux_error_W_m2", "numerical_flux_error_W_m2"]
        for p in (1,2):
            for axis in ("x","y"):
                for r in (0., .27):
                    previous = None
                    for n in (4,8,16,32):
                        row = measure(n,p,axis,r,True,sinusoidal=True)
                        errors = np.array([row[key] for key in metrics])
                        row["orders"] = [] if previous is None else np.log2(previous/errors).tolist()
                        passed = np.isfinite(errors).all() and row["max_cell_balance_W"] < 1e-10
                        if n == 32:
                            limits = ([5e-4, 4e-3, 1.2e-3, 5e-3, 2.5e-3] if p == 1 else
                                      [1e-5, 3.5e-4, 6e-6, 1e-3, 1.5e-5])
                            passed &= np.all(errors < limits) and min(row["orders"]) > 1.8
                        row["passed"] = bool(passed)
                        rows.append(row)
                        previous = errors
    else:
        reference = "Piecewise bilinear field with normal flux .7-1.6*t and jump R*q at s=.5"
        rows = [measure(n,p,axis,r,sparse) for n in (4,8) for p in (1,2) for axis in ("x","y")
                for r in (0.,.27) for sparse in (False,True)]
        for row in rows:
            row["passed"] = max(v for k,v in row.items() if "error" in k or "balance" in k or "l2" in k) < 1e-9
    print(json.dumps(dict(reference=reference, reproduction=command, **({"order_metrics":metrics} if sinusoidal else {}),
        sources=[provenance(p) for p in (__file__, dg.__file__)], runs=rows), indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
