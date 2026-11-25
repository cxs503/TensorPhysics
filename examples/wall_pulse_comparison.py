"""Compare FEM, SIPG and SPH slab temperatures and wall heat after a finite Dirichlet pulse."""
from dataclasses import replace
import json
import math

import numpy as np
import torch

from conduction_provenance import provenance
from tensorfem import thermal, thermal_analytic
from tensorsph import heat_conduction_2d as sph
from tensordg import poisson_2d_bc, transient_diffusion_2d


def measure(n):
    start,stop,end = .013,.047,.1
    time = np.linspace(0.,end,n//2+1)
    mesh = thermal_analytic.structured_quad_mesh(n,2,dtype=torch.float64)
    nodes,conn = mesh.nodes,mesh.elements
    fixed = torch.arange(len(nodes)).reshape(3,n+1)[:,[0,-1]].T.flatten()
    model = thermal.ThermalModel(nodes,conn,1.,1.,1.,fixed,nodes.new_zeros(len(fixed)),element_type="q4")
    temp = nodes.new_zeros(len(nodes))
    wall_heat = nodes.new_zeros(2)
    impulse_totals = []
    fem_residual = 0.
    def stored(field):
        return field[...,conn].mean(dim=(-1,-2))
    for begin,finish,level in ((0.,start,0.),(start,stop,1.),(stop,end,0.)):
        target = nodes.new_tensor([level,0.]).repeat_interleave(3)
        if begin!=0:
            after,impulse = thermal.apply_thermal_temperature_jump(model,temp,target,sparse=True)
            impulse_heat = impulse[fixed].reshape(2,3).sum(dim=1)
            fem_residual = max(fem_residual,float(abs(stored(after-temp)-impulse_heat.sum())))
            wall_heat += impulse_heat
            impulse_totals.append(impulse_heat.tolist())
            temp = after
        model = replace(model,fixed_temperatures=target)
        grid = np.r_[begin,time[(begin<time)&(time<finish)],finish]
        first = [grid[0],grid[0]+(grid[1]-grid[0])/2,grid[1]]
        for subgrid,theta in ((first,1.),(grid[1:],.5)):
            if len(subgrid)<2:
                continue
            sol = thermal.solve_transient_thermal(model,nodes.new_tensor(subgrid),temp,theta=theta,sparse=True)
            heat = sol.nodal_reaction_heat_steps()[:,fixed].reshape(-1,2,3).sum(dim=2)
            fem_residual = max(fem_residual,float(abs(stored(sol.temperature).diff()-heat.sum(dim=1)).max()))
            wall_heat += heat.sum(dim=0)
            temp = sol.temperature[-1]

    sides = dict(left="dirichlet",right="dirichlet",bottom="neumann",top="neumann")
    _,_,states,means,loads = transient_diffusion_2d.solve_transient_diffusion_2d(
        0.,time,ncells=(n,2),degree=1,bc=sides,dirichlet_values={"left":1.},
        load_amplitude=lambda t:float(start<=t<stop),load_jump_times=(start,stop),rannacher=True,
        return_interval_means=True,return_load_means=True)
    dg_heat = torch.zeros_like(wall_heat)
    dg_residual = 0.
    for i,dt in enumerate(np.diff(time)):
        px,_ = poisson_2d_bc.poisson_face_powers_2d(means[i],bc=sides,dirichlet_values={"left":loads[i]})
        inward = dt*torch.stack((px[0].sum(),-px[-1].sum()))
        dg_heat += inward
        dg_residual = max(dg_residual,float(abs((states[i+1]-states[i])[...,0,0].mean()-inward.sum())))

    # The solution is uniform in y. A single periodic row represents the
    # homogeneous transverse mode; multiply section heat per area by unit area.
    particles,sph_sections = sph.solve_transient_slab_2d(nodes.new_zeros((1,n)),time,1/n,
                      (lambda t:float(start<=t<stop),0.),wall_jump_times=(start,stop),
                      rannacher=True,return_x_heat=True)
    sph_steps = sph_sections[:,[0,-1]]*nodes.new_tensor([1.,-1.])
    sph_heat = sph_steps.sum(dim=0)
    sph_residual = float(abs(particles.diff(dim=0).mean(dim=(1,2))-sph_steps.sum(dim=1)).max())
    sph_column_residual = float(abs(particles.diff(dim=0)[:,0,:]/n+sph_sections.diff(dim=1)).max())
    z,w = (torch.from_numpy(a) for a in np.polynomial.legendre.leggauss(8))
    xi,eta = torch.meshgrid(z,z,indexing="ij")
    shape = torch.stack(((1-xi)*(1-eta),(1+xi)*(1-eta),(1+xi)*(1+eta),(1-xi)*(1+eta)),dim=-1)/4
    fem = torch.einsum("xyi,qri->xyqr",temp[conn].reshape(2,n,4).transpose(0,1),shape)
    phi = torch.stack((torch.ones_like(z),z),dim=-1)
    dg = torch.einsum("xyij,qi,rj->xyqr",states[-1],phi,phi)
    x = ((torch.arange(n,dtype=z.dtype)+.5)[:,None]+z/2)/n
    mode = torch.arange(1,97,dtype=z.dtype)
    wave = math.pi*mode
    decay = torch.exp(-wave.square()*(end-stop))-torch.exp(-wave.square()*(end-start))
    exact = (torch.sin(x[...,None]*wave) @ (2*decay/wave))[:,None,:,None]
    # Superpose two delayed unit steps. Positive heat enters at either wall.
    ref_west = (stop-start)+float((2*decay/wave.square()).sum())
    ref_east = -(stop-start)-float((2*(-1.)**mode*decay/wave.square()).sum())
    def norm(field):
        return float(((field.square()*w[:,None]*w).sum()/(8*n)).sqrt())
    centers = (torch.arange(n,dtype=z.dtype)+.5)/n
    center_ref = torch.sin(centers[:,None]*wave) @ (2*decay/wave)
    fem_center = temp[conn].reshape(2,n,4).mean(dim=(0,2))
    dg_center = states[-1,:,:,0,0].mean(dim=1)
    def center_rms(field):
        return float((field-center_ref).square().mean().sqrt())
    # Interpolate particles between their centers and the known end values
    # for a physical L² comparison using the same quadrature as FEM and DG.
    knots = torch.cat((centers.new_zeros(1),centers,centers.new_ones(1)))
    values = torch.cat((centers.new_zeros(1),particles[-1,0],centers.new_zeros(1)))
    lo = torch.searchsorted(knots,x)-1
    interp = values[lo]+(x-knots[lo])*(values[lo+1]-values[lo])/(knots[lo+1]-knots[lo])
    sph_field = interp[:,None,:,None].expand_as(fem)
    return dict(n=n,sph_l2=norm(sph_field-exact),sph_center_rms=center_rms(particles[-1,0]),
                fem_center_rms=center_rms(fem_center),dg_center_rms=center_rms(dg_center),
                fem_sph_difference_l2=norm(fem-sph_field),dg_sph_difference_l2=norm(dg-sph_field),
                sph_wall_heat_J=sph_heat.tolist(),sph_residual_J=sph_residual,sph_column_residual_J=sph_column_residual,
                fem_l2=norm(fem-exact),dg_l2=norm(dg-exact),difference_l2=norm(fem-dg),
                exact_wall_heat_J=[ref_west,ref_east],fem_wall_heat_J=wall_heat.tolist(),dg_wall_heat_J=dg_heat.tolist(),
                fem_jump_heat_J=impulse_totals,fem_residual_J=fem_residual,dg_residual_J=dg_residual)


if __name__=="__main__":
    rows = []
    for n in (16,32,64,128):
        row = measure(n)
        ref = np.array(row["exact_wall_heat_J"])
        for name in ("fem","dg","sph"):
            row[name+"_max_wall_heat_error_J"] = float(abs(ref-row[name+"_wall_heat_J"]).max())
        row["passed"] = max(row["fem_residual_J"],row["dg_residual_J"],row["sph_residual_J"],row["sph_column_residual_J"])<1e-10 and (n<128 or
                    (max(row["fem_l2"],row["dg_l2"],row["sph_l2"])<3e-5 and
                     max(row[name+"_max_wall_heat_error_J"] for name in ("fem","dg","sph"))<8e-6))
        rows.append(row)
    sources = [provenance(p) for p in (__file__,thermal.__file__,thermal_analytic.__file__,
                                       sph.__file__,poisson_2d_bc.__file__,transient_diffusion_2d.__file__)]
    print(json.dumps(dict(runs=rows,sources=sources),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
