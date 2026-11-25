"""Two anisotropic Q4 layers driven by an oblique harmonic wave across a resistive seam."""
import json

import numpy as np
import torch

from conduction_provenance import provenance
from rotated_harmonic_conduction import cubic_patches, material
from tensorfem import thermal, thermal_analytic

OMEGA, BETA = 3., 1.7
HOT, FLUX = 1.+.2j, .55-.3j


def exact_field(points,k,capacity,trace):
    # f(x) exp(i beta y) solves div(K grad T)=i omega c T. Remove the
    # first derivative by factoring out the phase induced by Kxy.
    x,y = points.T
    x = x-.5
    a = -1j*BETA*k[0,1]/k[0,0]
    gamma = np.sqrt((BETA**2*np.linalg.det(k)/k[0,0]+1j*OMEGA*capacity)/k[0,0])
    return np.exp(a*x+1j*BETA*y)*(trace*np.cosh(gamma*x)-FLUX*np.sinh(gamma*x)/(k[0,0]*gamma))


def measure(n,resistance,contact):
    ny = 2*n
    mesh = thermal_analytic.structured_quad_mesh(n,ny,lx=.5,dtype=torch.float64)
    conn = mesh.elements
    count = len(mesh.nodes)
    ids = np.arange(count).reshape(ny+1,n+1)
    seam = np.column_stack((ids[:,-1],ids[:,0]+count))
    area = np.full(ny+1,1/ny)
    area[[0,-1]] *= .5
    nodes, data, prescribed, tensors, capacities = [],[],[],[],[]
    for layer,(angle,capacity,trace) in enumerate(((-30.,1.5,HOT),(35.,2.2,HOT-resistance*FLUX))):
        k = material(angle)[0]
        coords = mesh.nodes.clone()
        x,y = coords.T
        shift = .04*torch.sin(2*torch.pi*x)*torch.sin(torch.pi*y)
        coords += torch.stack((shift+.5*layer,-.6*shift),dim=1)
        nodes.append(coords)
        data.append(exact_field(coords.numpy(),k,capacity,trace))
        prescribed.extend((np.r_[ids[0],ids[-1],ids[:,0 if layer==0 else -1]]+layer*count).tolist())
        tensors.append(np.repeat(k[None],len(conn),axis=0))
        capacities.extend([capacity]*len(conn))
    exact = np.concatenate(data)
    fixed = np.unique(prescribed)
    if contact == "integrated":
        pairs = np.stack((seam[:-1],seam[1:]),axis=2)
        options = dict(contact_edge_pairs=torch.from_numpy(pairs),contact_edge_conductance=1/(ny*resistance))
    else:
        options = dict(contact_pairs=torch.from_numpy(seam),contact_conductance=torch.from_numpy(area/resistance))
    model = thermal.ThermalModel(torch.cat(nodes),torch.cat((conn,conn+count)),
        torch.from_numpy(np.concatenate(tensors)),1.,torch.tensor(capacities,dtype=torch.float64),
        torch.from_numpy(fixed),torch.from_numpy(exact[fixed]),element_type="q4",
        **options)
    result = thermal.solve_harmonic_thermal(model,OMEGA,sparse=True)
    field = result.temperature
    cells = np.arange(len(conn)).reshape(ny,n)
    powers = []
    for edge,outer,inner,sign in ((1,cells[:,-1],cells[:,-2],1),
                                  (3,len(conn)+cells[:,0],len(conn)+cells[:,1],-1)):
        patch = cubic_patches(outer,inner)
        p = thermal.recover_q4_patch_boundary_power(model,field,patch,edge,degree=3)
        powers.append(sign*ny*p.numpy())
    phase = np.exp(1j*BETA*np.arange(ny+1)/ny)
    q_exact = FLUX*np.diff(phase)*ny/(1j*BETA)
    hot,cold = field.numpy()[seam].T
    jump = hot-cold
    q_jump = (jump[1:]+jump[:-1])/(2*resistance)
    weights = thermal.assemble_thermal(model,sparse=True).capacity @ torch.ones(len(exact),dtype=torch.float64)
    storage = (1j*OMEGA*weights*field).reshape(2,count).sum(1).numpy()
    external = np.array([result.reaction_power[fixed[fixed//count==layer]].sum().item() for layer in (0,1)])
    exchange = q_jump.mean()
    layer_balance = external+np.array([-exchange,exchange])-storage
    balance = (external-storage).sum()
    return dict(n=n,contact=contact,contact_resistance_m2K_W=resistance,
        hot_trace_error_K=float(abs(hot-HOT*phase).max()),
        cold_trace_error_K=float(abs(cold-(HOT-resistance*FLUX)*phase).max()),
        jump_error_K=float(abs(jump-resistance*FLUX*phase).max()),
        hot_flux_error_W_m2=float(abs(powers[0]-q_exact).max()),
        cold_flux_error_W_m2=float(abs(powers[1]-q_exact).max()),
        flux_disagreement_W_m2=float(abs(powers[0]-powers[1]).max()),
        jump_flux_error_W_m2=float(abs(q_jump-q_exact).max()),
        max_layer_balance_W=float(abs(layer_balance).max()),balance_residual_W=float(abs(balance)))


if __name__ == "__main__":
    rows = []
    for resistance in (.08,.3):
        for contact in ("nodal","integrated"):
            previous = None
            for n in (4,8,16,32):
                row = measure(n,resistance,contact)
                if previous:
                    row["orders"] = {k:float(np.log2(previous[k]/v)) for k,v in row.items()
                                     if "_error_" in k or k=="flux_disagreement_W_m2"}
                row["passed"] = max(row["max_layer_balance_W"],row["balance_residual_W"]) < 1e-10
                if n == 32:
                    row["passed"] &= (min(row["orders"].values()) > 1.8
                                      and max(row["hot_trace_error_K"],row["cold_trace_error_K"],row["jump_error_K"]) < 5e-5
                                      and max(row["hot_flux_error_W_m2"],row["cold_flux_error_W_m2"],
                                              row["jump_flux_error_W_m2"],row["flux_disagreement_W_m2"]) < 7e-4)
                previous = row
                rows.append(row)
    print(json.dumps(dict(runs=rows,sources=[provenance(f) for f in
        (__file__,cubic_patches.__code__.co_filename,thermal.__file__,thermal_analytic.__file__)]),
        indent=2,allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
