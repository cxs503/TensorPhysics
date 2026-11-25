"""Q4 transient slab with volumetric heating, one prescribed wall flux and one wall temperature."""
import json
import math

import torch

from conduction_provenance import provenance
from tensorfem import thermal, thermal_analytic


def measure(n):
    mesh = thermal_analytic.structured_quad_mesh(n,n,dtype=torch.float64)
    x,y = mesh.nodes.T
    west = torch.where(x==0)[0]
    east = torch.where(x==1)[0]
    model = thermal.ThermalModel(mesh.nodes,mesh.elements,1.,1.,1.,west,x[west],element_type="q4")
    times = torch.linspace(0.,.15,n//2+1,dtype=x.dtype)
    t = times[:,None]
    # Unit volumetric heat capacity makes the capacity matrix the consistent
    # load operator for a volumetric source interpolated from nodal samples.
    mass = thermal.assemble_thermal(model,sparse=True).capacity
    mode = torch.cos(2*math.pi*y)*torch.sin(.5*math.pi*x)
    profile = .5+x-x.square()
    heating = profile+2*t+(2*t+4.25*math.pi**2*t.square())*mode
    power = (mass @ heating.T).T.contiguous()
    weights = torch.ones_like(x[east])/n
    weights[[0,-1]] *= .5
    power[:,east] -= t*weights
    result = thermal.solve_transient_thermal(model,times,torch.zeros_like(x),theta=.5,sparse=True,
                  fixed_temperature_history=(.5*t).expand(-1,len(west)),nodal_heat_history=power)
    flux = thermal.thermal_element_flux(model,result.temperature)
    xc,yc = mesh.nodes[mesh.elements].mean(dim=1).T
    qx = -t*(1-2*xc)-.5*math.pi*t.square()*torch.cos(2*math.pi*yc)*torch.cos(.5*math.pi*xc)
    qy = 2*math.pi*t.square()*torch.sin(2*math.pi*yc)*torch.sin(.5*math.pi*xc)
    # Average the analytic normal flux along each edge. The sinc factors
    # integrate the sinusoidal tangential dependence exactly on this square grid.
    dx = 1/n
    ky, kx = 2*math.pi, .5*math.pi
    cos_y = torch.cos(ky*yc)*math.sin(ky*dx/2)/(ky*dx/2)
    sin_x = torch.sin(kx*xc)*math.sin(kx*dx/2)/(kx*dx/2)
    def vertical(x_face):
        return -t*(1-2*x_face)-kx*t.square()*cos_y*torch.cos(kx*x_face)
    def horizontal(y_face):
        return ky*t.square()*torch.sin(ky*y_face)*sin_x
    outward = torch.stack((-horizontal(yc-dx/2),vertical(xc+dx/2),
                           horizontal(yc+dx/2),-vertical(xc-dx/2)),dim=-1)
    edge_power = thermal.thermal_element_edge_power(model,result.temperature)
    edge_error = float((edge_power-dx*outward).abs().max())
    # In a uniform material on a regular grid the two one-sided estimates
    # straddle each interior face. Orient them consistently before averaging.
    sides = edge_power.reshape(len(times),n,n,4)/dx
    normal = outward.reshape_as(sides)
    interior_x = .5*(sides[:,:,:-1,1]-sides[:,:,1:,3])
    interior_y = .5*(sides[:,:-1,:,2]-sides[:,1:,:,0])
    interior_error = max(float((interior_x-normal[:,:,:-1,1]).abs().max()),
                         float((interior_y-normal[:,:-1,:,2]).abs().max()))
    return dict(cells=n,edge_power_W=edge_error,edge_mean_flux_W_m2=edge_error/dx,
                averaged_interior_flux_W_m2=interior_error,temperature_K=float((result.temperature-t*profile-t.square()*mode).abs().max()),
                x_flux_W_m2=float((flux[...,0]-qx).abs().max()),y_flux_W_m2=float((flux[...,1]-qy).abs().max()))


if __name__=="__main__":
    rows = []
    for n in (8,16,32,64):
        row = measure(n)
        orders = {k:math.log2(rows[-1][k]/row[k]) for k in ("temperature_K","x_flux_W_m2","y_flux_W_m2")} if rows else {}
        if rows:
            orders.update({k:math.log2(rows[-1][k]/row[k]) for k in ("edge_power_W","edge_mean_flux_W_m2","averaged_interior_flux_W_m2")})
        # Q4 gradients sampled on faces converge at first order for this field.
        required_edge_order = .95 if n>=32 else 0.
        row.update(orders=orders,passed=all(v>(required_edge_order if k=="edge_mean_flux_W_m2" else 1.5)
                                            for k,v in orders.items()))
        rows.append(row)
    print(json.dumps(dict(runs=rows,sources=[provenance(f) for f in
                     (__file__,thermal.__file__,thermal_analytic.__file__)]),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
