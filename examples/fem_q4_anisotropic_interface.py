"""Two anisotropic Q4 materials with continuous normal heat flux on a skew interface."""
import argparse
import json

import numpy as np
import torch

from conduction_provenance import provenance
import q4_boundary_patches as patches
from tensorfem import thermal, thermal_analytic


def measure(n, shear, omega, sparse, curved=False):
    mesh = thermal_analytic.structured_quad_mesh(n, n, shear=shear, dtype=torch.float64)
    nodes, cells = mesh.nodes.numpy(), mesh.elements.numpy()
    points = nodes[cells]
    normal = np.array([1., -shear])
    normal_length = np.linalg.norm(normal)
    layer = ((points.mean(axis=1) @ normal) > .5).astype(int)
    tensors = np.array([[[2.1, .63], [.63, 1.2]], [[.85, -.31], [-.31, 3.6]]])
    capacity = np.array([1.1, 3.4])[layer]
    thickness, rate = .7, .27+.34j
    b, q0, q1 = -.24+.32j, .53-.17j, -.36+.29j
    # T=b*v+u*(a+d*v), u=x-shear*y-.5, v=y-.5. Solve for a,d so
    # -normal.K.grad(T)=q0+q1*v on both sides of u=0.
    kn = tensors @ normal
    a = (-q0-b*kn[:, 1])/(kn @ normal)
    d = -q1/(kn @ normal)
    u, v = points @ normal-.5, points[:,:,1]-.5
    temperature = b*v+u*(a[layer,None]+d[layer,None]*v)
    e = np.array([.37+.18j, -.29+.26j])
    f = np.array([-.23+.15j, .27+.11j])
    divergence = 2*(d*kn[:,1])[layer,None]
    correction = 0.
    if curved:
        # u²*(e+f*v) vanishes with its gradient on the interface. Its Q4
        # interpolant differs by a symmetric quadratic in u times a linear
        # function of v. Projecting that difference onto the nodal loads
        # subtracts (e+f*v)*h²/6 at both u endpoints.
        curvature = e[layer,None]+f[layer,None]*v
        temperature += u**2*curvature
        correction = -curvature/(6*n**2)
        divergence = divergence+2*curvature*(kn@normal)[layer,None]+4*u*(f*kn[:,1])[layer,None]
    temperature -= np.average((temperature+correction).mean(axis=1), weights=capacity)
    source = capacity[:,None]*(rate+1j*omega*(temperature+correction))-divergence

    def flux(xy, material):
        u, v = xy @ normal-.5, xy[:,1]-.5
        du, dv = a[material]+d[material]*v, b+d[material]*u
        if curved:
            du += 2*u*(e[material]+f[material]*v)
            dv += u**2*f[material]
        gradient = np.column_stack((du, dv-shear*du))
        return -np.einsum("cij,cj->ci", tensors[material], gradient)

    # Only exterior tractions enter the load vector. Do not prescribe an
    # interface flux; continuity follows from the assembled FEM equations.
    ids = np.arange(n*n).reshape(n,n)
    loads = np.zeros(len(nodes), dtype=complex)
    for edge, outer in enumerate((ids[0], ids[:,-1], ids[-1], ids[:,0])):
        i, j = cells[outer,edge], cells[outer,(edge+1)%4]
        tangent = nodes[j]-nodes[i]
        area_normal = thickness*np.column_stack((tangent[:,1], -tangent[:,0]))
        for h in ((1-3**-.5)/2, (1+3**-.5)/2):
            power = -np.sum(flux(nodes[i]+h*tangent, layer[outer])*area_normal, axis=1)/2
            np.add.at(loads, i, (1-h)*power)
            np.add.at(loads, j, h*power)
    model = thermal.ThermalModel(mesh.nodes, mesh.elements, torch.from_numpy(tensors[layer]),
        1., torch.from_numpy(capacity), torch.empty(0,dtype=torch.long), torch.empty(0,dtype=torch.float64),
        nodal_heat=torch.from_numpy(loads), heat_source=torch.from_numpy(source),
        thickness=thickness, element_type="q4")
    result = thermal.solve_harmonic_thermal(model, omega, sparse=sparse, return_mean_rate=True)
    edge_power = thermal.thermal_element_edge_power(model, result.temperature).numpy()
    left, right = ids[:,n//2-1], ids[:,n//2]
    face_area = thickness*normal_length/n
    left_flux, right_flux = edge_power[left,1]/face_area, -edge_power[right,3]/face_area
    reference = (q0+q1*((np.arange(n)+.5)/n-.5))/normal_length
    tangent_unit = np.array([shear, 1.])/normal_length
    normal_error = tangential_jump_error = 0.
    for eta in (-3**-.5, 3**-.5):
        h = (1+eta)/2
        xy = (1-h)*points[left,1]+h*points[left,2]
        q_left = thermal.thermal_element_flux(model, result.temperature, xi=1., eta=eta).numpy()[left]
        q_right = thermal.thermal_element_flux(model, result.temperature, xi=-1., eta=eta).numpy()[right]
        normal_ref = (q0+q1*(xy[:,1]-.5))/normal_length
        normal_error = max(normal_error, *(float(abs(q @ normal/normal_length-normal_ref).max())
                                          for q in (q_left, q_right)))
        jump = (flux(xy,layer[left])-flux(xy,layer[right])) @ tangent_unit
        tangential_jump_error = max(tangential_jump_error, float(abs((q_left-q_right) @ tangent_unit-jump).max()))
    local = thermal.thermal_element_nodal_power(model, result.temperature, omega,
                                               mean_heating_rate=result.mean_heating_rate).numpy()
    field = result.temperature.numpy()[cells]
    storage = capacity*(rate+1j*omega*(field if curved else temperature).mean(axis=1))
    balance = local.sum(axis=1)+thickness/n**2*(source.mean(axis=1)-storage)
    recovery = {}
    if curved:
        recovered = []
        for outer, inner, edge, sign in ((left, left-1, 1, 1), (right, right+1, 3, -1)):
            patch = patches.cubic_patches(outer, inner)
            if np.any(layer[patch] != layer[outer,None]):
                raise ValueError("recovery patch crosses the material interface")
            fit = thermal.recover_q4_patch_boundary_power(model, result.temperature, patch, edge, degree=3)
            recovered.append(sign*fit.numpy()/face_area)
        recovery = dict(recovered_left_flux_error_W_m2=float(abs(recovered[0]-reference).max()),
                        recovered_right_flux_error_W_m2=float(abs(recovered[1]-reference).max()),
                        recovered_flux_mismatch_W_m2=float(abs(recovered[0]-recovered[1]).max()))
    return dict(**recovery, cells_per_axis=n, shear=shear, omega_rad_s=omega, sparse=sparse,
                nodal_temperature_error_K=float(abs(field-temperature).max()),
                mean_rate_error_K_s=float(abs(result.mean_heating_rate.numpy()-rate).max()),
                interface_left_flux_error_W_m2=float(abs(left_flux-reference).max()),
                interface_right_flux_error_W_m2=float(abs(right_flux-reference).max()),
                interface_flux_mismatch_W_m2=float(abs(left_flux-right_flux).max()),
                interface_gauss_normal_error_W_m2=normal_error,
                interface_tangential_jump_error_W_m2=tangential_jump_error,
                cell_energy_residual_W=float(abs(balance).max()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curved", action="store_true", help="verify one-sided cubic recovery with normal curvature")
    curved = parser.parse_args().curved
    levels = (8,16,32) if curved else (4,8,16,32)
    rows = [measure(n,shear,omega,sparse,curved) for n in levels for shear in (0., .38)
            for omega in (0., 1e-12, 1.) for sparse in (False, True)]
    previous = {}
    for row in rows:
        if curved:
            metrics = ("nodal_temperature_error_K", "interface_left_flux_error_W_m2", "interface_right_flux_error_W_m2",
                       "recovered_left_flux_error_W_m2", "recovered_right_flux_error_W_m2")
            key = row["shear"], row["omega_rad_s"], row["sparse"]
            errors = np.array([row[m] for m in metrics])
            row["orders"] = {m:float(v) for m,v in zip(metrics,np.log2(previous[key]/errors))} if key in previous else {}
            row["passed"] = bool(row["mean_rate_error_K_s"] < 1e-10 and row["cell_energy_residual_W"] < 1e-12
                                 and np.all(errors[3:] < errors[1:3]) and all(p > 0 for p in row["orders"].values()))
            if row["cells_per_axis"] == 32:
                row["passed"] &= min(row["orders"].values()) > .9 and row["orders"][metrics[0]] > 1.7
            previous[key] = errors
        else:
            row["passed"] = all(v < 1e-10 for k,v in row.items() if "error_" in k or "mismatch_" in k or "residual_" in k)
    print(json.dumps(dict(reference=("Piecewise bilinear temperature plus u²*(e+f*v)" if curved else "Piecewise bilinear temperature")+" with aligned material interface, prescribed analytic normal flux and capacity-weighted zero mean",
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFEM/src python examples/fem_q4_anisotropic_interface.py"+(" --curved" if curved else ""),
        runs=rows, sources=[provenance(f) for f in (__file__,thermal.__file__,thermal_analytic.__file__, *((patches.__file__,) if curved else ()))]),
        indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
