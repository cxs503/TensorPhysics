"""Oblique harmonic thermal wave in a slab with rotated principal conductivities."""
import json

import numpy as np
import torch

from conduction_provenance import provenance
from tensorfem import thermal, thermal_analytic
from tensordg import poisson_2d_bc, transient_diffusion_2d


OMEGA, CAPACITY = 3., 1.5


def material(angle):
    a = np.deg2rad(angle)
    rotation = np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    k = rotation @ np.diag([2.,.5]) @ rotation.T
    direction = np.array([1.,-.6])
    # T=exp(s·x) satisfies div(K grad T)=i omega c T.
    s = np.sqrt(1j*OMEGA*CAPACITY/(direction@k@direction))*direction
    return k,s


def dg_case(n,degree,k,s):
    nx,ny = n,3*n//2
    x,y = np.linspace(0,1,nx+1),np.linspace(0,1,ny+1)
    def trace(x,y):
        return torch.exp(s[0]*x+s[1]*y)
    data = dict(conductivity=(k[0,0],k[1,1],k[0,1]),
                dirichlet_values=dict.fromkeys(("left","right","bottom","top"),trace))
    _,_,coeff = transient_diffusion_2d.solve_harmonic_diffusion_2d(
        OMEGA,ncells=(nx,ny),degree=degree,capacity=CAPACITY,**data)
    fx,fy = (v.numpy() for v in poisson_2d_bc.poisson_face_powers_2d(coeff,**data))
    ex,ey = np.exp(s[0]*x),np.exp(s[1]*y)
    ax,ay = np.diff(ex)*nx/s[0],np.diff(ey)*ny/s[1]
    q = -k@s
    errors_x = abs(ny*fx-q[0]*np.outer(ex,ay))
    errors_y = abs(nx*fy-q[1]*np.outer(ax,ey))
    means = coeff[:,:,0,0].numpy()
    balance = 1j*OMEGA*CAPACITY*means/(nx*ny)+np.diff(fx,axis=0)+np.diff(fy,axis=1)
    return dict(solver=f"DG-P{degree}",temperature_error_K=float(abs(means-np.outer(ax,ay)).max()),
                side_wall_error_W_m2=float(errors_x[[0,-1]].max()),
                end_wall_error_W_m2=float(errors_y[:,[0,-1]].max()),
                interior_flux_error_W_m2=float(max(errors_x[1:-1].max(),errors_y[:,1:-1].max())),
                max_balance_residual_W=float(abs(balance).max()))


def fem_case(n,shear,k,s):
    nx,ny = n,3*n//2
    mesh = thermal_analytic.structured_quad_mesh(nx,ny,shear=shear,dtype=torch.float64)
    points = mesh.nodes.numpy()
    ids = np.arange(len(points)).reshape(ny+1,nx+1)
    sides = [ids[:,0],ids[:,-1],ids[0,:],ids[-1,:]]
    fixed = np.unique(np.concatenate(sides))
    exact = np.exp(points@s)
    model = thermal.ThermalModel(mesh.nodes,mesh.elements,torch.from_numpy(k[None]),1.,CAPACITY,
        torch.from_numpy(fixed),torch.from_numpy(exact[fixed]),element_type="q4")
    result = thermal.solve_harmonic_thermal(model,OMEGA,sparse=True)
    nodal = result.temperature.numpy()
    power = result.reaction_power.numpy()
    # Integrate the exact inward flux against the same boundary hats as
    # the FEM reaction. Each corner receives contributions from two sides.
    target = np.zeros_like(exact)
    area = np.zeros(len(points))
    unit,weights = np.polynomial.legendre.leggauss(8)
    shapes = np.column_stack(((1-unit)/2,(1+unit)/2))
    normals = [(-1,shear),(1,-shear),(0,-1),(0,1)]
    cells = np.arange(nx*ny).reshape(ny,nx)
    pairs = ((3,cells[:,0],cells[:,1]),(1,cells[:,-1],cells[:,-2]),
             (0,cells[0],cells[1]),(2,cells[-1],cells[-2]))
    raw = thermal.thermal_element_edge_power(model,result.temperature).numpy()
    raw_errors,edge_errors = [],[]
    for edge,normal,(local,outer,inner) in zip(sides,normals,pairs):
        conn = np.column_stack((edge[:-1],edge[1:]))
        ends = points[conn]
        length = np.linalg.norm(ends[0,1]-ends[0,0])
        normal = np.array(normal)/np.linalg.norm(normal)
        values = np.exp((ends@s) @ shapes.T)*(normal@k@s)
        outward = thermal.recover_q4_boundary_power(model,result.temperature,outer,inner,local)
        raw_errors.append(float(abs(raw[outer,local]/length + values@weights/2).max()))
        edge_errors.append(float(abs(outward.numpy()/length + values@weights/2).max()))
        np.add.at(target,conn,(length/2)*(values*weights) @ shapes)
        np.add.at(area,conn,length/2)
    norm_error = abs(power[fixed]-target[fixed])/area[fixed]
    def side_error(edges):
        idx = np.concatenate([edge[1:-1] for edge in edges])
        return float((abs(power[idx]-target[idx])/area[idx]).max())
    storage = 1j*OMEGA*CAPACITY*nodal[mesh.elements.numpy()].mean()
    return dict(solver="FEM-Q4",shear=shear,temperature_error_K=float(abs(nodal-exact).max()),
                side_wall_error_W_m2=side_error(sides[:2]),end_wall_error_W_m2=side_error(sides[2:]),
                raw_edge_error_W_m2=max(raw_errors),recovered_edge_error_W_m2=max(edge_errors),all_boundary_error_W_m2=float(norm_error.max()),
                max_balance_residual_W=float(abs(storage-power[fixed].sum())))


if __name__ == "__main__":
    rows = []
    for angle in (-30.,30.):
        k,s = material(angle)
        previous = {}
        for n in (4,8,16,32):
            for row in [dg_case(n,d,k,s) for d in (1,2)]+[fem_case(n,h,k,s) for h in (0.,.25)]:
                name = row["solver"],row.get("shear",0.)
                keys = [key for key in row if "_error_" in key]
                if name in previous:
                    row["orders"] = {key:float(np.log2(previous[name][key]/row[key])) for key in keys}
                row.update(nx=n,ny=3*n//2,angle_deg=angle)
                row["passed"] = row["max_balance_residual_W"] < 1e-10
                if n == 32:
                    orders = row["orders"]
                    wall_order = .85 if name[0] == "DG-P1" else 1.8
                    wall_limit = {"DG-P1":1., "DG-P2":.006, "FEM-Q4":.002}[name[0]]
                    row["passed"] &= (row["temperature_error_K"] < .002
                                      and orders["temperature_error_K"] > 1.8
                                      and max(row["side_wall_error_W_m2"],row["end_wall_error_W_m2"]) < wall_limit
                                      and min(orders["side_wall_error_W_m2"],orders["end_wall_error_W_m2"]) > wall_order)
                    if name[0] == "FEM-Q4":
                        row["passed"] &= (row["recovered_edge_error_W_m2"] < min(.015,row["raw_edge_error_W_m2"])
                                          and orders["recovered_edge_error_W_m2"] > 1.8)
                        row["passed"] &= row["all_boundary_error_W_m2"] < .04 and orders["all_boundary_error_W_m2"] > .85
                    else:
                        row["passed"] &= orders["interior_flux_error_W_m2"] > wall_order
                previous[name] = row
                rows.append(row)
    print(json.dumps(dict(runs=rows,sources=[provenance(f) for f in
        (__file__,thermal.__file__,thermal_analytic.__file__,transient_diffusion_2d.__file__,poisson_2d_bc.__file__)]),
        indent=2,allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
