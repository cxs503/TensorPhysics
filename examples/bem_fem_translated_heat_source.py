"""Verify a flux-driven sheared plate with a local quadratic heat source.

Run python examples/bem_fem_translated_heat_source.py with TensorBEM and
TensorFEM installed. Coordinates are metres, with unit out-of-plane depth.
"""
import json

import numpy as np
import torch

from conduction_provenance import provenance
from tensorbem import geometry, harmonic_conduction
from tensorfem import thermal, thermal_analytic


K, CV, OMEGA = 1.25, 1.5, 2.
SHEAR = .375
LINEAR = np.array([.25+.5j, -.75j])
PHASE = .75-.25j
CONSTANT = 1.+.125j


def temperature(points):
    x, y = points.T
    return CONSTANT + points @ LINEAR + PHASE*(x-SHEAR*y)*y


def gradient(points):
    x, y = points.T
    return LINEAR + PHASE*np.column_stack((y, x-2*SHEAR*y))


def measure(origin):
    mesh = thermal_analytic.structured_quad_mesh(8, 8, shear=SHEAR, dtype=torch.float64)
    local = mesh.nodes.numpy() + [.25, -.5]
    coords = local + origin
    grid = np.arange(len(coords)).reshape(9, 9)
    wall = np.r_[grid[0,:-1], grid[:-1,-1], grid[-1,:0:-1], grid[:0:-1,0]]
    boundary = geometry.PanelMesh(coords[wall])

    # Q = iωCV*T - KΔT. The FEM nodal source represents its affine part.
    storage = 1j*OMEGA*CV
    base = storage*CONSTANT + 2*K*SHEAR*PHASE
    linear = storage*LINEAR
    quadratic = storage*PHASE*np.array([0., 1., -SHEAR])
    samples = harmonic_conduction.harmonic_boundary_points(boundary, quadratic=True, origin=origin)
    inward = K*np.sum(gradient(samples)*np.repeat(boundary.normals, 3, axis=0), axis=1)
    bem = harmonic_conduction.solve_mixed_harmonic_conduction(
        boundary, "flux", inward, OMEGA, conductivity=K, volumetric_heat_capacity=CV,
        quadratic=True, polynomial_origin=origin, volumetric_source=base,
        source_gradient=linear, source_quadratic=quadratic)

    # The imposed inward flux varies linearly along each straight edge.
    next_wall = np.roll(wall, -1)
    q0 = K*np.sum(gradient(local[wall])*boundary.normals, axis=1)
    q1 = K*np.sum(gradient(local[next_wall])*boundary.normals, axis=1)
    load = np.zeros(len(coords), dtype=complex)
    load[wall] = boundary.lengths*(2*q0+q1)/6
    load[next_wall] += boundary.lengths*(q0+2*q1)/6
    model = thermal.ThermalModel(
        torch.from_numpy(coords), mesh.elements, K, 1., CV, torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.float64), element_type="q4", nodal_heat=torch.from_numpy(load),
        heat_source=torch.from_numpy((base+local @ linear)[mesh.elements.numpy()]),
        heat_source_quadratic=torch.from_numpy(quadratic), heat_source_origin=torch.from_numpy(origin))
    fem = thermal.solve_harmonic_thermal(model, OMEGA, sparse=True)
    probes = grid[1:-1,1:-1].ravel()
    exact_t = temperature(local[probes])
    # The mean of an affine gradient on a parallelogram is its centroid value.
    exact_q = -K*gradient(local[[grid[4,4]]])[0]
    rows = []
    for solver, temp, flux in (
        ("TensorBEM", bem.temperature(coords[probes]), bem.area_mean_heat_flux()),
        ("TensorFEM", fem.temperature.numpy()[probes], thermal.thermal_volume_mean_flux(model, fem.temperature).numpy()),
    ):
        t_error = float(np.max(abs(temp-exact_t)))
        q_error = float(np.max(abs(flux-exact_q)))
        rows.append(dict(solver=solver, origin_m=origin.tolist(), temperature_error_K=t_error,
                         mean_flux_error_W_m2=q_error, passed=max(t_error, q_error) < 1e-12))
    return rows


if __name__ == "__main__":
    runs = measure(np.zeros(2)) + measure(np.array([2.**43, -2.**43]))
    report = dict(command="python examples/bem_fem_translated_heat_source.py", runs=runs,
                  sources=[provenance(p) for p in (__file__, geometry.__file__, harmonic_conduction.__file__,
                                                    thermal.__file__, thermal_analytic.__file__)])
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in runs) else 1)
