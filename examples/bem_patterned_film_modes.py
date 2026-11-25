"""Harmonic conduction with spatially varying film and wall storage on polygon panels."""
import json

import numpy as np
from tensorbem import geometry, harmonic_conduction

import bem_disk_thermal_modes as modes
from conduction_provenance import provenance


def measure(n, omega, quadratic, mode):
    k, capacity = 1.7, 2.3
    wave = np.sqrt(1j*omega*capacity/k)
    mesh = geometry.regular_polygon(n)
    points = harmonic_conduction.harmonic_boundary_points(mesh, quadratic=quadratic)
    angle = np.arctan2(points[:, 1], points[:, 0])
    film = 2.+.65*np.cos(2*angle+.3)
    surface = .4+.22*np.sin(3*angle-.2)
    wall, flux = modes.reference(points, mode, wave, k)
    normals = np.repeat(mesh.normals, len(points)//n, axis=0)
    # Use the actual chord normals so the analytic field solves the PDE and
    # boundary condition on each polygon, including at off-midpoint samples.
    inward = -(flux*normals).sum(axis=1)
    storage = 1j*omega*surface*wall
    external = inward+storage
    ambient = wall+external/film
    result = harmonic_conduction.solve_mixed_harmonic_conduction(
        mesh, "film", ambient, omega, conductivity=k, volumetric_heat_capacity=capacity,
        film_coefficients=film, wall_areal_heat_capacity=surface, quadratic=quadratic)
    theta = 2*np.pi*np.arange(17)/17+.19
    probes = .65*np.column_stack((np.cos(theta), np.sin(theta)))
    ref_temp, ref_flux = modes.reference(probes, mode, wave, k)
    stored = 1j*omega*surface*result.wall_temperature
    return dict(panels=n, quadratic=quadratic, omega_rad_s=omega, angular_mode=mode,
                interior_temperature_error_K=float(abs(result.temperature(probes)-ref_temp).max()),
                interior_flux_error_W_m2=float(abs(result.heat_flux(probes)-ref_flux).max()),
                wall_temperature_error_K=float(abs(result.wall_temperature-wall).max()),
                material_flux_error_W_m2=float(abs(result.inward_flux-inward).max()),
                storage_rate_error_W_m2=float(abs(stored-storage).max()),
                external_flux_error_W_m2=float(abs(result.external_inward_flux-external).max()),
                wall_balance_residual_W_m2=float(abs(result.external_inward_flux-stored-result.inward_flux).max()))


if __name__ == "__main__":
    metrics = ["interior_temperature_error_K", "interior_flux_error_W_m2", "wall_temperature_error_K",
               "material_flux_error_W_m2", "storage_rate_error_W_m2", "external_flux_error_W_m2"]
    runs = []
    for quadratic in (False, True):
        for omega in (.5, 5.):
            for mode in (-1, 2):
                previous = None
                for n in (16, 32, 64):
                    row = measure(n, omega, quadratic, mode)
                    errors = np.array([row[key] for key in metrics])
                    orders = [] if previous is None else np.log2(previous/errors).tolist()
                    row.update(orders=orders, passed=bool(row["wall_balance_residual_W_m2"] < 1e-11
                               and (not orders or min(orders) > 0.)
                               and (n < 64 or min(orders) > (2.5 if quadratic else 1.8))))
                    runs.append(row)
                    previous = errors
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, order_metrics=metrics, runs=runs, sources=[provenance(path) for path in
                          (__file__, modes.__file__, harmonic_conduction.__file__, geometry.__file__)],
                          reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorBEM/src python examples/bem_patterned_film_modes.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
