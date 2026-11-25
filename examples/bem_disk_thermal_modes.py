"""Verify harmonic BEM fields with regular modified-Bessel modes on inscribed polygons."""
import json

import numpy as np
from scipy.special import iv, ivp

from conduction_provenance import provenance
from tensorbem import geometry, harmonic_conduction


def reference(points, mode, wave, conductivity):
    x, y = points.T
    radius = np.hypot(x, y)
    angle = np.arctan2(y, x)
    phase = np.exp(1j*mode*angle)/iv(mode, wave)
    t = iv(mode, wave*radius)*phase
    radial = wave*ivp(mode, wave*radius)*phase
    # For the tested modes 0 and 2 the Cartesian gradient vanishes at the origin.
    tangent = np.divide(1j*mode*t, radius, out=np.zeros_like(t), where=radius > 0)
    cs, sn = np.cos(angle), np.sin(angle)
    return t, -conductivity*np.column_stack((radial*cs-tangent*sn, radial*sn+tangent*cs))


def measure(n, mode, omega, quadratic):
    conductivity, capacity = 1.7, 2.3
    wave = np.sqrt(1j*omega*capacity/conductivity)
    mesh = geometry.regular_polygon(n)
    points = harmonic_conduction.harmonic_boundary_points(mesh, quadratic=quadratic)
    wall, flux = reference(points, mode, wave, conductivity)
    normals = np.repeat(mesh.normals, len(points)//n, axis=0)
    inward = -np.sum(flux*normals, axis=1)
    solution = harmonic_conduction.solve_harmonic_conduction(
        mesh, wall, omega, conductivity=conductivity, volumetric_heat_capacity=capacity,
        quadratic=quadratic)
    angles = np.arange(23)*2*np.pi/23+.17
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    probes = np.vstack(([0., 0.], .35*directions, .72*directions))
    exact_t, exact_q = reference(probes, mode, wave, conductivity)
    component = abs(solution.heat_flux(probes)-exact_q).max(axis=0)
    return dict(panels=n, mode=mode, omega_rad_s=omega, quadratic=quadratic,
                interior_temperature_error_K=float(abs(solution.temperature(probes)-exact_t).max()),
                interior_flux_x_error_W_m2=float(component[0]), interior_flux_y_error_W_m2=float(component[1]),
                inward_flux_error_W_m2=float(abs(solution.inward_flux-inward).max()))


if __name__ == "__main__":
    rows = []
    metrics = ["interior_temperature_error_K", "interior_flux_x_error_W_m2",
               "interior_flux_y_error_W_m2", "inward_flux_error_W_m2"]
    for quadratic in (False, True):
        for omega in (.5, 5.):
            for mode in (0, 2):
                previous = None
                for n in (16, 32, 64):
                    row = measure(n, mode, omega, quadratic)
                    errors = np.array([row[key] for key in metrics])
                    orders = [] if previous is None else np.log2(previous/errors).tolist()
                    passed = np.isfinite(errors).all() and all(p > 0 for p in orders)
                    if n == 64:
                        limits = (1e-8, 1e-7, 1e-7, 1e-4) if quadratic else (.001, .005, .005, .02)
                        passed = passed and min(orders) > (2.5 if quadratic else 1.8) and np.all(errors < limits)
                    row.update(orders=orders, passed=bool(passed))
                    previous = errors
                    rows.append(row)
    print(json.dumps(dict(reference="I_m(lambda*r) exp(i*m*theta)/I_m(lambda), lambda^2=i*omega*C/k; Dirichlet data imposed on the actual polygon",
                          reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorBEM/src python examples/bem_disk_thermal_modes.py",
                          order_metrics=metrics, runs=rows, sources=[provenance(__file__), provenance(harmonic_conduction.__file__), provenance(geometry.__file__)]),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
