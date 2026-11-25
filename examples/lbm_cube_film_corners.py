"""Verify the heat budget of cells at the intersection of three convective walls."""
import argparse
from itertools import product
import json
import math
import inspect

import numpy as np
import torch

from conduction_provenance import provenance
from tensorlbm import thermal_common as lbm
from lbm_directional_diffusion import diffusion_eigenmode


def measure(n, tau, kinds=("film", "film", "film"), eigen_start=False):
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
    if eigen_start:
        # Initialize each of the eight bulk plane waves in its slow discrete mode.
        # The boundary update and the continuum wall drives stay identical.
        ramp_grad = torch.stack([torch.full_like(field,s) for s in slope])
        g = lbm.thermal_gradient_initialization_3d(ramp,zero,zero,zero,ramp_grad,tau_T=tau,spacing=dx)
        amplitude = .6*np.prod(np.sinc(beta*dx/(2*np.pi)))/8
        for signs in product((-1,1), repeat=3):
            wave = beta*np.asarray(signs)
            _, vector = diffusion_eigenmode(wave*dx,tau)
            phase = torch.exp(1j*sum(k*(v-.5) for k,v in zip(wave,xyz)))
            g += amplitude*(torch.from_numpy(vector)[:,None,None,None]*phase).real
    initial_error = float((g.sum(dim=0)-field).abs().max())
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
    face_errors = np.zeros((3,2))
    first_three_error = later_error = 0.
    dirichlet_start = {}
    peak_step = peak_axis = peak_side = 0
    steps = round(.09765625/dt)
    for step in range(steps):
        post = lbm.thermal_collide_bgk_3d(g,field,zero,zero,zero,tau)
        interior = lbm.thermal_link_fluxes_3d(post,periodic=False,**units)
        streamed = lbm.thermal_stream_3d(post)
        g = streamed
        div = torch.zeros_like(field)
        average = math.exp(-rate*step*dt)*(-math.expm1(-rate*dt))/(rate*dt)
        wall_inputs = []
        for a,pair in enumerate(walls):
            dim = 2-a
            wall_t = tuple(base+average*surface for _,base,surface in pair)
            # The inward normal Fourier flux is sign*k*slope minus the mode contribution.
            target = tuple((2*side-1)*capacity*alpha*slope[a]-h[a]*average*surface
                           for side,(_,_,surface) in enumerate(pair))
            values = {"temperature": wall_t, "flux": target, "film": tuple(v[0] for v in pair)}[kinds[a]]
            wall_inputs.append(values)
            g,flux = lbm.thermal_mixed_wall_pair_3d(g,"xyz"[a],(kinds[a],)*2,values,
                film_coefficients=(h[a],h[a]),**units)
            for side in (0,1):
                error = float((flux[side]-target[side]).abs().max())
                if step == 0 and kinds[a] == "temperature" and not eigen_start:
                    # Apply the quadratic first-step defect to the smooth cosine mode.
                    wall_mode = pair[side][2]
                    normal_curvature = -beta[a]**2*wall_mode
                    laplacian = -(beta @ beta)*wall_mode
                    prediction = (capacity*dx/(4*dt))*(alpha*dt*laplacian/2
                        - dx**2*(1/6+(tau-1)/2)*normal_curvature)
                    remainder = float((flux[side]-target[side]-prediction).abs().max())
                    dirichlet_start[("lower","upper")[side]] = dict(raw_error_W_m2=error,
                        curvature_prediction_residual_W_m2=remainder)
                face_errors[a,side] = max(face_errors[a,side],error)
                if step < 3:
                    first_three_error = max(first_three_error,error)
                else:
                    later_error = max(later_error,error)
                if error > flux_error:
                    flux_error = error
                    peak_step, peak_axis, peak_side = step+1, a, side
            faces = torch.cat((flux[0].unsqueeze(dim),interior[a],-flux[1].unsqueeze(dim)),dim)
            div += faces.diff(dim=dim)
        if step == 0:
            reverse = streamed
            for a in (2,1,0):
                reverse,_ = lbm.thermal_mixed_wall_pair_3d(reverse,"xyz"[a],(kinds[a],)*2,
                    wall_inputs[a],film_coefficients=(h[a],h[a]),**units)
            sequence_error = float((g-reverse).abs().max())
        updated = lbm.thermal_macroscopic_3d(g)
        residual = capacity*dx**3*(updated-field)+dt*dx**2*div
        balance = max(balance,float(residual.abs().max()))
        corner_balance = max(corner_balance,max(abs(residual[idx].item()) for idx in corner))
        temp_error = max(temp_error,float((updated-ramp-mode*math.exp(-rate*(step+1)*dt)).abs().max()))
        field = updated
    return dict(first_step_dirichlet=dirichlet_start, initial_temperature_error_K=initial_error, first_three_flux_error_W_m2=first_three_error,
                subsequent_flux_error_W_m2=later_error if steps>3 else None, peak_flux_step=peak_step, peak_flux_axis="xyz"[peak_axis], peak_flux_side=("lower","upper")[peak_side],
                wall_types_xyz=kinds, lower_upper_flux_errors_xyz_W_m2=face_errors.tolist(), n=n,tau=tau,steps=steps,temperature_error_K=temp_error,wall_flux_error_W_m2=flux_error,
                max_cell_balance_J=balance,max_corner_balance_J=corner_balance,axis_order_error_K=sequence_error)


if __name__ == "__main__":
    rows = []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixed", action="store_true", help="cycle Dirichlet, Neumann and Robin conditions among the three axes")
    parser.add_argument("--compare-starts", action="store_true", help="compare gradient initialization with exact bulk diffusion eigenvectors")
    args = parser.parse_args()
    mixed = args.mixed
    cases = (("temperature","flux","film"),("film","temperature","flux"),("flux","film","temperature"))
    for kinds in (cases if mixed else (("film",)*3,)):
        for tau in (.75,1.5):
            previous = None
            for n in (8,16,32):
                row = measure(n,tau,kinds)
                errors = np.array([row["temperature_error_K"],row["wall_flux_error_W_m2"]])
                row["orders"] = [] if previous is None else np.log2(previous/errors).tolist()
                # Mixed Dirichlet faces can have a first-order startup flux error.
                limits = [6e-4, 2.5e-3] if mixed else [1.5e-4, 3.5e-4]
                order_floor = 0. if mixed else 1.
                row["passed"] = bool(row["max_cell_balance_J"] < 1e-14
                    and row["max_corner_balance_J"] < 1e-14 and row["axis_order_error_K"] == 0.
                    and all(order > order_floor for order in row["orders"])
                    and (n < 32 or np.all(errors < limits)))
                if mixed and n == 32:
                    row["passed"] &= all(face["curvature_prediction_residual_W_m2"] < 1e-4
                                         for face in row["first_step_dirichlet"].values())
                if args.compare_starts:
                    other = measure(n,tau,kinds,eigen_start=True)
                    row["bulk_eigenmode_start"] = other
                    row["passed"] &= (other["initial_temperature_error_K"] < 1e-13
                        and other["max_cell_balance_J"] < 1e-14 and other["axis_order_error_K"] == 0.
                        and (n < 32 or (other["temperature_error_K"] < 1e-3
                            and other["wall_flux_error_W_m2"] < .0035)))
                rows.append(row)
                previous = errors
    print(json.dumps(dict(reference="Unit cube: affine temperature plus separated decaying cosine mode with h_i=k*beta_i*tan(beta_i/2)",
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorLBM/src python examples/lbm_cube_film_corners.py"+(" --mixed" if mixed else "")+(" --compare-starts" if args.compare_starts else ""),
        order_metrics=["temperature_error_K","wall_flux_error_W_m2"],runs=rows,
        sources=[provenance(p) for p in (__file__,lbm.__file__)]+([provenance(inspect.getfile(diffusion_eigenmode))] if args.compare_starts else [])),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
