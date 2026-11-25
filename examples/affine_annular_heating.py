"""Two-material annulus with Cartesian source gradients and a resistive contact."""
import argparse
import json

import numpy as np
import torch

from conduction_provenance import provenance
from tensorbem import layered_annulus
from tensorfem import thermal, thermal_analytic


A, B, C = .3, .6, 1.
K = (2., .8)
GRADIENT = np.array([[12., -8.], [-4., 16.]])
QUADRATIC = np.array([[12., -16., -12.], [-4., 32., 4.]])
RESISTANCE = .35
RATE = .7  # radial heat rate per radian and unit axial length


def reference(xy, layer, quadratic=False):
    radius = np.linalg.norm(xy, axis=1)
    unit = xy/radius[:, None]
    # Both the amplitude and its derivative vanish at the contact. The
    # r and 1/r terms are harmonic in angular mode one, Δ(r³ cos θ)=8r cos θ.
    amplitude = (radius**2-B**2)**2/radius
    slope = 3*radius**2 - 2*B**2 - B**4/radius**2
    projection = unit @ GRADIENT[layer]
    tangent_gradient = GRADIENT[layer] - projection[:, None]*unit
    divisor = 8
    if quadratic:
        # Angular mode two: Δ(r⁴ cos 2θ) = 12 r² cos 2θ. Add the two
        # homogeneous radial modes to cancel value and slope at the contact.
        amplitude = radius**4 - 1.5*B**2*radius**2 + .5*B**6/radius**2
        slope = 4*radius**3 - 3*B**2*radius - B**6/radius**3
        a, cross, c = QUADRATIC[layer]
        x, y = unit.T
        projection = a*x*x + cross*x*y + c*y*y
        tangent = np.column_stack((-y, x))
        angular_slope = cross*(x*x-y*y) + 2*(c-a)*x*y
        tangent_gradient = angular_slope[:, None]*tangent
        divisor = 12
    temperature = -(RATE*np.log(radius/B) + amplitude*projection/divisor)/K[layer]
    if layer == 0:
        temperature += RESISTANCE*RATE/B
    radial_flux = RATE/radius + slope*projection/divisor
    flux = radial_flux[:, None]*unit + (amplitude/(divisor*radius))[:, None]*tangent_gradient
    return temperature, flux


def fem(n, nt, quadratic=False):
    meshes = [thermal_analytic.annular_quad_mesh(n, nt, inner=lo, outer=hi, dtype=torch.float64)
              for lo,hi in ((A,B),(B,C))]
    count = len(meshes[0].nodes)
    nodes = torch.cat([mesh.nodes for mesh in meshes])
    elements = torch.cat([mesh.elements+i*count for i,mesh in enumerate(meshes)])
    exact = np.concatenate([reference(mesh.nodes.numpy(), i, quadratic)[0] for i,mesh in enumerate(meshes)])
    loads = []
    for i, mesh in enumerate(meshes):
        if quadratic:
            x, y = mesh.nodes.T
            a, cross, c = QUADRATIC[i]
            nodal_load = a*x*x + cross*x*y + c*y*y
        else:
            nodal_load = mesh.nodes @ torch.from_numpy(GRADIENT[i])
        loads.append(nodal_load[mesh.elements])
    source = torch.cat(loads)
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


def bem(nt, form, quadratic=False):
    theta = 2*np.pi*(np.arange(nt)+.5)/nt
    unit = np.column_stack((np.cos(theta),np.sin(theta)))
    walls = (reference(A*unit,0,quadratic)[0],reference(C*unit,1,quadratic)[0])
    load = dict(source_quadratic=QUADRATIC) if quadratic else dict(source_gradient=GRADIENT)
    solution = layered_annulus.solve_annulus_contact((A,B,C), K, walls, RESISTANCE, nt,
                                                      formulation=form, **load)
    errors = []
    for i,(radius,field) in enumerate(((.45,solution.inner_field),(.8,solution.outer_field))):
        angles = np.arange(37)*2*np.pi/37+.017
        xy = radius*np.column_stack((np.cos(angles),np.sin(angles)))
        t,q = reference(xy,i,quadratic)
        errors.append((np.max(abs(field.temperature(xy)-t)),
                       np.linalg.norm(field.heat_flux(xy)-q,axis=1).max()))
    temperature,flux = np.max(errors,axis=0)
    return dict(temperature_K=temperature, interior_flux_W_m2=flux,
                contact_flux_W_m2=np.max(abs(solution.interface_flux-RATE/B)),
                wall_rate_W_m=np.max(abs(np.array(solution.heat_rates)[[0,2]]-2*np.pi*RATE)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quadratic", action="store_true", help="use second angular harmonic volumetric heating")
    quadratic = parser.parse_args().quadratic
    rows = []
    previous = {}
    for n in (4,8,16,32):
        nt = 4*n
        for name,errors in (("FEM",fem(n,nt,quadratic)),("BEM_polygonal",bem(nt,"polygonal",quadratic)),
                            ("BEM_circular",bem(nt,"circular",quadratic))):
            orders = {}
            if name=="BEM_circular":
                passed = max(errors.values()) < 1e-10
            else:
                if name in previous:
                    orders = {k:np.log2(previous[name][k]/v) for k,v in errors.items()}
                passed = all(v>1.5 for v in orders.values())
            previous[name] = errors
            rows.append(dict(solver=name, panels=nt, errors=errors, orders=orders, passed=bool(passed)))
    print(json.dumps(dict(quadratic=quadratic, runs=rows, sources=[provenance(f) for f in
                     (__file__,thermal.__file__,thermal_analytic.__file__,layered_annulus.__file__)]),
                     indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
