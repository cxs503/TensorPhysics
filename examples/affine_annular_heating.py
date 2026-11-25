"""Two-material annulus with Cartesian source gradients and a resistive contact."""
import json

import numpy as np
import torch

from conduction_provenance import provenance
from tensorbem import layered_annulus
from tensorfem import thermal, thermal_analytic


A, B, C = .3, .6, 1.
K = (2., .8)
GRADIENT = np.array([[12., -8.], [-4., 16.]])
RESISTANCE = .35
RATE = .7  # radial heat rate per radian and unit axial length


def reference(xy, layer):
    radius = np.linalg.norm(xy, axis=1)
    unit = xy/radius[:, None]
    # Both the amplitude and its derivative vanish at the contact. The
    # r and 1/r terms are harmonic in angular mode one, Δ(r³ cos θ)=8r cos θ.
    amplitude = (radius**2-B**2)**2/radius
    slope = 3*radius**2 - 2*B**2 - B**4/radius**2
    projection = unit @ GRADIENT[layer]
    temperature = -(RATE*np.log(radius/B) + amplitude*projection/8)/K[layer]
    if layer == 0:
        temperature += RESISTANCE*RATE/B
    radial_flux = RATE/radius + slope*projection/8
    flux = radial_flux[:, None]*unit + (amplitude/(8*radius))[:, None]*(
        GRADIENT[layer] - projection[:, None]*unit)
    return temperature, flux


def fem(n, nt):
    meshes = [thermal_analytic.annular_quad_mesh(n, nt, inner=lo, outer=hi, dtype=torch.float64)
              for lo,hi in ((A,B),(B,C))]
    count = len(meshes[0].nodes)
    nodes = torch.cat([mesh.nodes for mesh in meshes])
    elements = torch.cat([mesh.elements+i*count for i,mesh in enumerate(meshes)])
    exact = np.concatenate([reference(mesh.nodes.numpy(), i)[0] for i,mesh in enumerate(meshes)])
    source = torch.cat([(mesh.nodes @ torch.from_numpy(GRADIENT[i]))[mesh.elements]
                        for i,mesh in enumerate(meshes)])
    j = torch.arange(nt)
    fixed = torch.cat((j,2*count-nt+j))
    inner, outer = count-nt+j, count+j
    chord = 2*B*np.sin(np.pi/nt)
    model = thermal.ThermalModel(nodes, elements, torch.from_numpy(np.repeat(K,n*nt)), 1., 1.,
                                 fixed, torch.from_numpy(exact)[fixed], element_type="q4",
                                 heat_source=source, contact_pairs=torch.stack((inner,outer),dim=1),
                                 contact_conductance=chord/RESISTANCE)
    u = thermal.solve_steady_thermal(model,sparse=True)
    system = thermal.assemble_thermal(model,sparse=True)
    reaction = system.conductivity @ u - system.heat
    rates = np.array([reaction[j].sum(), -reaction[fixed[nt:]].sum()])
    return dict(temperature_K=float(np.max(abs(u.numpy()-exact))),
                contact_flux_W_m2=float(torch.max(abs((u[inner]-u[outer])/RESISTANCE-RATE/B))),
                wall_rate_W_m=float(np.max(abs(rates-2*np.pi*RATE))))


def bem(nt, form):
    theta = 2*np.pi*(np.arange(nt)+.5)/nt
    unit = np.column_stack((np.cos(theta),np.sin(theta)))
    walls = (reference(A*unit,0)[0],reference(C*unit,1)[0])
    solution = layered_annulus.solve_annulus_contact((A,B,C), K, walls, RESISTANCE, nt,
                                                      formulation=form, source_gradient=GRADIENT)
    errors = []
    for i,(radius,field) in enumerate(((.45,solution.inner_field),(.8,solution.outer_field))):
        angles = np.arange(37)*2*np.pi/37+.017
        xy = radius*np.column_stack((np.cos(angles),np.sin(angles)))
        t,q = reference(xy,i)
        errors.append((np.max(abs(field.temperature(xy)-t)),
                       np.linalg.norm(field.heat_flux(xy)-q,axis=1).max()))
    temperature,flux = np.max(errors,axis=0)
    return dict(temperature_K=temperature, interior_flux_W_m2=flux,
                contact_flux_W_m2=np.max(abs(solution.interface_flux-RATE/B)),
                wall_rate_W_m=np.max(abs(np.array(solution.heat_rates)[[0,2]]-2*np.pi*RATE)))


if __name__ == "__main__":
    rows = []
    previous = {}
    for n in (4,8,16,32):
        nt = 4*n
        for name,errors in (("FEM",fem(n,nt)),("BEM_polygonal",bem(nt,"polygonal")),
                            ("BEM_circular",bem(nt,"circular"))):
            orders = {}
            if name=="BEM_circular":
                passed = max(errors.values()) < 1e-10
            else:
                if name in previous:
                    orders = {k:np.log2(previous[name][k]/v) for k,v in errors.items()}
                passed = all(v>1.5 for v in orders.values())
            previous[name] = errors
            rows.append(dict(solver=name, panels=nt, errors=errors, orders=orders, passed=bool(passed)))
    print(json.dumps(dict(runs=rows, sources=[provenance(f) for f in
                     (__file__,thermal.__file__,thermal_analytic.__file__,layered_annulus.__file__)]),
                     indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
