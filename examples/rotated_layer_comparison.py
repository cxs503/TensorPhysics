"""Compare Q4 FEM and bilinear SIPG on the same two-material square and physical quadrature."""
import json
import math

import numpy as np
import torch

from conduction_provenance import provenance
from tensorfem import thermal, thermal_analytic
from tensordg import poisson_2d_bc as dg
from tensordg import transient_diffusion_2d as transient_dg


def measure(n,transient):
    det,beta = 3.,1.5
    alpha = 2. if transient else math.sqrt(det)*beta
    rate = alpha**2+det*beta**2 if transient else 0.
    end = .0625
    def amplitude(t):
        return math.exp(-rate*t)

    def exact(x,y):
        # In r=integral(dx/Kxx), v=y-integral(Kxy/Kxx dx), the spatial
        # operator becomes (d_rr + det(K)*d_vv)/Kxx in either material.
        west,east = x.clamp(max=.5),(x-.5).clamp(min=0)
        a,b = alpha*(west+east/4)+.3,beta*(y+.7*west-.45*east)+.4
        axial = torch.sin(a) if transient else torch.cosh(a)
        return axial*torch.cos(b)

    mesh = thermal_analytic.structured_quad_mesh(n,n,dtype=torch.float64)
    nodes,conn = mesh.nodes,mesh.elements
    centers = nodes[conn].mean(dim=1)
    k = torch.where(centers[:,0]<.5,torch.ones_like(centers[:,0]),4.)
    tilt = torch.where(centers[:,0]<.5,torch.full_like(k,-.7),.45)
    cross,ky = k*tilt,det/k+k*tilt.square()
    tensor = torch.stack((k,cross,cross,ky),dim=-1).reshape(-1,2,2)
    fixed = torch.where(((nodes==0.) | (nodes==1.)).any(dim=1))[0]
    start = exact(*nodes.T)
    model = thermal.ThermalModel(nodes,conn,tensor,1.,1/k,fixed,start[fixed],element_type="q4")
    # FEM connectivity advances x first. DG arrays have leading (x,y) axes.
    material = torch.stack((k,ky,cross),dim=-1).reshape(n,n,3).transpose(0,1)
    walls = dict.fromkeys(("left","right","bottom","top"),exact)
    opts = dict(ncells=n,degree=1,conductivity=material,dirichlet_values=walls)
    if transient:
        time = torch.linspace(0.,end,n+1,dtype=nodes.dtype)
        fem = thermal.solve_transient_thermal(model,time,start,theta=.5,sparse=True,
                          fixed_temperature_history=torch.exp(-rate*time[:,None])*start[fixed])
        fem_field = fem.temperature[-1]
        _,_,states = transient_dg.solve_transient_diffusion_2d(exact,time,capacity=1/material[...,0],
                                                               load_amplitude=amplitude,**opts)
        coeff = states[-1]
    else:
        fem_field = thermal.solve_steady_thermal(model,sparse=True)
        _,_,coeff,_ = dg.solve_poisson_2d_bc(forcing=lambda x,y:0*x+0*y,sparse=True,**opts)

    # Interpolate both fields at the very same physical Gauss points; include
    # interior temperatures even for FEM elements with prescribed corner nodes.
    z,w = (torch.from_numpy(v) for v in np.polynomial.legendre.leggauss(8))
    xi,eta = torch.meshgrid(z,z,indexing="ij")
    shape = torch.stack(((1-xi)*(1-eta),(1+xi)*(1-eta),(1+xi)*(1+eta),(1-xi)*(1+eta)),dim=-1)/4
    f = torch.einsum("xyi,qri->xyqr",fem_field[conn].reshape(n,n,4).transpose(0,1),shape)
    phi = torch.stack((torch.ones_like(z),z),dim=-1)
    d = torch.einsum("xyij,qi,rj->xyqr",coeff,phi,phi)
    points = ((torch.arange(n,dtype=z.dtype)+.5)[:,None]+z/2)/n
    ref = amplitude(end)*exact(points[:,None,:,None],points[None,:,None,:])
    def norm(delta):
        return float(torch.sqrt((delta.square()*w[:,None]*w).sum()/(4*n*n)))

    # Compare rightward power per face area. FEM has traces on both sides.
    edge = thermal.thermal_element_edge_power(model,fem_field).reshape(n,n,4)
    fleft,fright = n*edge[:,n//2-1,1],-n*edge[:,n//2,3]
    qx,_ = dg.poisson_face_powers_2d(coeff,conductivity=material,
                       dirichlet_values=dict.fromkeys(walls,lambda x,y:amplitude(end)*exact(x,y)))
    slope = math.cos(alpha*.5+.3) if transient else math.sinh(alpha*.5+.3)
    true_flux = -amplitude(end)*alpha*slope*(torch.cos(beta*(points+.35)+.4) @ w)/2
    row = dict(n=n,transient=transient,fem_l2=norm(f-ref),dg_l2=norm(d-ref),difference_l2=norm(f-d),
                fem_flux_error=float(torch.maximum(abs(fleft-true_flux),abs(fright-true_flux)).max()),
                dg_flux_error=float(abs(n*qx[n//2]-true_flux).max()))

    if transient:
        # Unit depth, no volume source. Integral(C*T) equals stored heat;
        # the total inward boundary reaction over a step must equal its change.
        # The independent Gauss rule integrates smooth data inside each layer.
        capacity = 1/material[...,0]
        initial_exact = exact(points[:,None,:,None],points[None,:,None,:])
        exact_store = float((capacity[...,None,None]*initial_exact*w[:,None]*w).sum()/(4*n*n))
        exact_heat = (amplitude(end)-1)*exact_store
        fem_steps = fem.nodal_reaction_heat_steps()[:,fixed].sum(dim=1)
        fem_storage = (fem.temperature[:,conn].mean(dim=2)/k).sum(dim=1)/(n*n)
        dg_storage = (states[...,0,0]*capacity).sum(dim=(1,2))/(n*n)
        dg_steps = []
        for i,dt in enumerate(time.diff()):
            amp = (amplitude(float(time[i]))+amplitude(float(time[i+1])))/2
            px,py = dg.poisson_face_powers_2d((states[i]+states[i+1])/2,conductivity=material,
                             dirichlet_values=dict.fromkeys(walls,lambda x,y:amp*exact(x,y)))
            dg_steps.append(dt*(px[0].sum()-px[-1].sum()+py[:,0].sum()-py[:,-1].sum()))
        dg_steps = torch.stack(dg_steps)
        row.update(exact_inward_heat_J=exact_heat,fem_inward_heat_J=float(fem_steps.sum()),
                   dg_inward_heat_J=float(dg_steps.sum()),fem_heat_error=abs(float(fem_steps.sum())-exact_heat),
                   dg_heat_error=abs(float(dg_steps.sum())-exact_heat),
                   fem_balance_J=float((fem_steps-fem_storage.diff()).abs().max()),
                   dg_balance_J=float((dg_steps-dg_storage.diff()).abs().max()))
    return row


if __name__=="__main__":
    rows = []
    for transient in (False,True):
        previous = {}
        for n in (4,8,16,32):
            row = measure(n,transient)
            errors = {key:row[key] for key in ("fem_l2","dg_l2","difference_l2","fem_flux_error","dg_flux_error")}
            if transient:
                errors.update({k:row[k] for k in ("fem_heat_error","dg_heat_error")})
            rates = {key:math.log2(previous[key]/v) for key,v in errors.items()} if previous else {}
            row.update(orders=rates,passed=all(v>0 for v in rates.values()) and
                       (n<32 or (min(rates["fem_l2"],rates["dg_l2"])>1.8 and
                                 min(rates["dg_flux_error"],rates["fem_flux_error"])>.8)))
            if transient:
                row["passed"] &= max(row["fem_balance_J"],row["dg_balance_J"])<1e-10 and (
                    n<32 or min(rates["fem_heat_error"],rates["dg_heat_error"])>1.8)
            rows.append(row)
            previous = errors
    sources = [provenance(p) for p in (__file__,thermal.__file__,thermal_analytic.__file__,
                                       dg.__file__,transient_dg.__file__)]
    print(json.dumps(dict(runs=rows,sources=sources),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
