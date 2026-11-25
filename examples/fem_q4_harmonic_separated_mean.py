"""Verify ultra-low-frequency anisotropic heat flux on a sheared Q4 mesh."""
import argparse
import json

import numpy as np
import torch

from conduction_provenance import provenance
import q4_boundary_patches as patches
from tensorfem import thermal, thermal_analytic


def boundary_metrics(model, temperature, exact, edge_power, target, areas, n):
    fields = torch.stack((temperature, torch.from_numpy(exact)))
    errors = {key: [] for key in ("raw", "center", "quadratic", "cubic", "cubic_exact_nodes")}
    weights = []
    for edge, outer, inner in patches.boundary_pairs(n, n):
        patch = patches.cubic_patches(outer, inner)
        area = areas[outer, edge]
        weights.extend(area)
        cubic = thermal.recover_q4_patch_boundary_power(model, fields, patch, edge, degree=3)
        powers = dict(raw=edge_power[outer, edge],
                      center=thermal.recover_q4_boundary_power(model, temperature, outer, inner, edge),
                      quadratic=thermal.recover_q4_patch_boundary_power(model, temperature, patch, edge),
                      cubic=cubic[0], cubic_exact_nodes=cubic[1])
        for name, power in powers.items():
            errors[name].extend(abs(power.numpy()-target[outer, edge])/area)
    return {name: dict(rms_W_m2=float(np.sqrt(np.average(np.square(values), weights=weights))),
                       max_W_m2=float(np.max(values))) for name, values in errors.items()}


def measure(n, omega, sparse, shear, curved=False, boundary_recovery=False):
    mesh = thermal_analytic.structured_quad_mesh(n, n, shear=shear, dtype=torch.float64)
    coords, conn = mesh.nodes.numpy(), mesh.elements.numpy()
    k = np.array([[2.1, .63], [.63, 1.2]])
    c, thickness, rate = 1.8, .7, .27+.34j
    a, b, d = .6+.15j, -.24+.32j, .37-.41j
    e, f = (.42+.18j, -.33+.27j) if curved else (0j, 0j)
    # u=x-shear*y and v=y are uniform on [0,1]^2. All three terms
    # have zero integral, as do the optional odd cubic terms.
    def local(points):
        return points[..., 0]-shear*points[..., 1]-.5, points[..., 1]-.5

    def flux(points):
        u, v = local(points)
        du, dv = a+d*v+3*e*u**2, b+d*u+3*f*v**2
        return -np.stack((du, dv-shear*du), axis=-1) @ k.T

    u, v = local(coords)
    exact = a*u+b*v+d*u*v+e*u**3+f*v**3
    divergence = 2*d*(k[0, 1]-shear*k[1, 1])
    divergence += 6*(e*u*(k[0, 0]-2*shear*k[0, 1]+shear**2*k[1, 1])+f*v*k[1, 1])
    generation = c*(rate+1j*omega*exact)-divergence
    points = coords[conn]
    tangent = np.roll(points, -1, axis=1)-points
    normals = np.stack((tangent[..., 1], -tangent[..., 0]), axis=-1)
    # Two Gauss points integrate quadratic edge flux times each linear
    # endpoint shape, including the shear contribution to the edge normal.
    first = np.zeros(normals.shape[:2], dtype=complex)
    second = np.zeros_like(first)
    for t in ((1-3**-.5)/2, (1+3**-.5)/2):
        inward = -thickness*np.sum(flux(points+t*tangent)*normals, axis=-1)
        first += inward*(1-t)/2
        second += inward*t/2
    exact_local = first+np.roll(second, 1, axis=1)
    loads = np.zeros(len(coords), dtype=complex)
    # Assemble only external edges. Interior opposite normals cancel.
    ids = np.arange(n*n).reshape(n, n)
    for edge, selected in enumerate((ids[0], ids[:, -1], ids[-1], ids[:, 0])):
        np.add.at(loads, conn[selected, edge], first[selected, edge])
        np.add.at(loads, conn[selected, (edge+1)%4], second[selected, edge])
    model = thermal.ThermalModel(mesh.nodes, mesh.elements, torch.from_numpy(k[None]), 1., c,
        torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float64),
        nodal_heat=torch.from_numpy(loads), heat_source=torch.from_numpy(generation[conn]),
        thickness=thickness, element_type="q4")
    result = thermal.solve_harmonic_thermal(model, omega, sparse=sparse, separate_mean=True)
    local_power = thermal.thermal_element_nodal_power(model, result.temperature, omega,
                                                       mean_temperature=result.mean_temperature)
    edge_power = thermal.thermal_element_edge_power(model, result.temperature)
    component_error = np.zeros(2)
    square_errors = np.zeros(3)
    for xi in (-3**-.5, 3**-.5):
        for eta in (-3**-.5, 3**-.5):
            shape = np.array([(1-xi)*(1-eta), (1+xi)*(1-eta), (1+xi)*(1+eta), (1-xi)*(1+eta)])/4
            samples = np.einsum("i,eij->ej", shape, points)
            numerical = thermal.thermal_element_flux(model, result.temperature, xi=xi, eta=eta).numpy()
            error = abs(numerical-flux(samples))
            component_error = np.maximum(component_error, error.max(axis=0))
            su, sv = local(samples)
            sample_t = a*su+b*sv+d*su*sv+e*su**3+f*sv**3
            error_t = abs(result.temperature.numpy()[conn] @ shape-sample_t)
            square_errors += np.r_[np.mean(error_t**2), np.mean(error**2, axis=0)]/4
    rms = np.sqrt(square_errors)
    row = dict(rms_temperature_error_K=rms[0], rms_x_flux_error_W_m2=rms[1], rms_y_flux_error_W_m2=rms[2],
                curved=curved, cells_per_axis=n, omega_rad_s=omega, sparse=sparse, shear=shear,
                mean_rate_error_K_s=float(max(abs(1j*omega*result.mean_temperature.numpy()-rate))),
                relative_temperature_error_K=float(max(abs(result.temperature.numpy()-exact))),
                x_flux_error_W_m2=float(component_error[0]), y_flux_error_W_m2=float(component_error[1]),
                inward_nodal_power_error_W=float(abs(local_power.numpy()-exact_local).max()),
                outward_edge_power_error_W=float(abs(edge_power.numpy()+first+second).max()))
    if boundary_recovery:
        row["boundary_flux"] = boundary_metrics(model, result.temperature, exact, edge_power,
                                                -first-second, thickness*np.linalg.norm(tangent, axis=-1), n)
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary-recovery", action="store_true", help="compare outward edge-averaged flux estimates")
    recovery = parser.parse_args().boundary_recovery
    rows = []
    metrics = ["rms_temperature_error_K", "rms_x_flux_error_W_m2", "rms_y_flux_error_W_m2"]
    for curved in (False, True):
        for omega in (1., 1e-6, 1e-12):
            for sparse in (False, True):
                for shear in (0., .38):
                    previous = previous_max = None
                    for n in ((4, 8, 16, 32) if curved else (4, 8, 16)):
                        row = measure(n, omega, sparse, shear, curved, recovery)
                        errors = np.array([row[k] for k in metrics])
                        if curved:
                            maxima = np.array([row[k] for k in ("relative_temperature_error_K", "x_flux_error_W_m2", "y_flux_error_W_m2")])
                            row["max_error_orders"] = np.log2(previous_max/maxima).tolist() if previous_max is not None else []
                            previous_max = maxima
                            orders = np.log2(previous/errors).tolist() if previous is not None else []
                            row.update(orders=orders, passed=bool(row["mean_rate_error_K_s"] < 1e-12 and
                                       (not orders or (orders[0] > 1.7 and min(orders[1:]) > .85))))
                        else:
                            row["passed"] = all(v < 1e-10 for k, v in row.items() if "_error_" in k)
                        if recovery:
                            boundary = row["boundary_flux"]
                            row["passed"] &= boundary["cubic_exact_nodes"]["max_W_m2"] < 1e-10
                            if curved and n == 32:
                                row["passed"] &= boundary["cubic"]["rms_W_m2"] < boundary["raw"]["rms_W_m2"]
                            if not curved:
                                row["passed"] &= max(method["max_W_m2"] for method in boundary.values()) < 1e-10
                        rows.append(row)
                        previous = errors
    print(json.dumps(dict(reference="Uniform temperature pole and a zero-mean bilinear field with optional cubic terms in sheared coordinates with off-diagonal conductivity",
                          reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFEM/src python examples/fem_q4_harmonic_separated_mean.py"+(" --boundary-recovery" if recovery else ""),
                          curved_case_note="Acceptance uses RMS at four Gauss points per cell. Pointwise maxima and their refinement orders are reported separately; the 4-to-8 refinement does not meet 1.7/0.85 temperature/flux maximum-error order targets.",
                          order_metrics=metrics, runs=rows, sources=[provenance(p) for p in (__file__, thermal.__file__, thermal_analytic.__file__, *((patches.__file__,) if recovery else ()))]),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
