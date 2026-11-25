"""Verify harmonic BEM fields with regular modified-Bessel modes on inscribed polygons."""
import argparse
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


def measure(n, mode, omega, quadratic, circle_data=False):
    conductivity, capacity = 1.7, 2.3
    wave = np.sqrt(1j*omega*capacity/conductivity)
    mesh = geometry.regular_polygon(n)
    points = harmonic_conduction.harmonic_boundary_points(mesh, quadratic=quadratic)
    wall, flux = reference(points, mode, wave, conductivity)
    normals = np.repeat(mesh.normals, len(points)//n, axis=0)
    extra = {}
    if circle_data:
        # Match chord and circle by polar angle. Compare their respective
        # inward normal fluxes at corresponding positions.
        circle = points/np.linalg.norm(points, axis=1)[:, None]
        circle_wall, flux = reference(circle, mode, wave, conductivity)
        extra["boundary_data_shift_K"] = float(abs(circle_wall-wall).max())
        wall, normals = circle_wall, circle
    inward = -np.sum(flux*normals, axis=1)
    solution = harmonic_conduction.solve_harmonic_conduction(
        mesh, wall, omega, conductivity=conductivity, volumetric_heat_capacity=capacity,
        quadratic=quadratic)
    angles = np.arange(23)*2*np.pi/23+.17
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    probes = np.vstack(([0., 0.], .35*directions, .72*directions))
    exact_t, exact_q = reference(probes, mode, wave, conductivity)
    component = abs(solution.heat_flux(probes)-exact_q).max(axis=0)
    if circle_data:
        # Project onto the driven angular mode before integrating so the
        # nonaxisymmetric signal survives integration over the perimeter.
        phase = np.exp(-1j*mode*np.arctan2(points[:, 1], points[:, 0]))
        weights = np.array([5., 8., 5.])/18 if quadratic else np.ones(1)
        modal_power = mesh.lengths @ ((solution.inward_flux*phase).reshape(n, -1) @ weights)
        exact_power = 2*np.pi*conductivity*wave*ivp(mode, wave)/iv(mode, wave)
        extra["modal_inward_power_error_W_m"] = float(abs(modal_power-exact_power))
    return dict(**extra, panels=n, mode=mode, omega_rad_s=omega, quadratic=quadratic,
                interior_temperature_error_K=float(abs(solution.temperature(probes)-exact_t).max()),
                interior_flux_x_error_W_m2=float(component[0]), interior_flux_y_error_W_m2=float(component[1]),
                inward_flux_error_W_m2=float(abs(solution.inward_flux-inward).max()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circle-data", action="store_true", help="use unit-circle boundary data on the inscribed polygon")
    circle_data = parser.parse_args().circle_data
    rows = []
    metrics = ["interior_temperature_error_K", "interior_flux_x_error_W_m2",
               "interior_flux_y_error_W_m2", "inward_flux_error_W_m2"]
    if circle_data:
        metrics.append("modal_inward_power_error_W_m")
    for quadratic in (False, True):
        for omega in (.5, 5.):
            for mode in (0, 2):
                previous = None
                for n in ((16, 32, 64, 128) if circle_data else (16, 32, 64)):
                    row = measure(n, mode, omega, quadratic, circle_data)
                    errors = np.array([row[key] for key in metrics])
                    orders = [] if previous is None else np.log2(previous/errors).tolist()
                    passed = np.isfinite(errors).all() and all(p > 0 for p in orders)
                    if n == 64 and not circle_data:
                        limits = (1e-8, 1e-7, 1e-7, 1e-4) if quadratic else (.001, .005, .005, .02)
                        passed = passed and min(orders) > (2.5 if quadratic else 1.8) and np.all(errors < limits)
                    if circle_data and n >= 64:
                        # Pointwise normals on a chord differ from radial normals
                        # by O(h) away from the midpoint of the panel.
                        passed = passed and min(orders[:3]+orders[4:]) > 1.8
                        passed = passed and orders[3] > (.85 if quadratic else 1.8)
                    row.update(orders=orders, passed=bool(passed))
                    previous = errors
                    rows.append(row)
    print(json.dumps(dict(reference="I_m(lambda*r) exp(i*m*theta)/I_m(lambda), lambda^2=i*omega*C/k; Dirichlet data evaluated "+("on the unit circle at matching polar angles" if circle_data else "on the actual polygon"),
                          reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorBEM/src python examples/bem_disk_thermal_modes.py"+(" --circle-data" if circle_data else ""),
                          order_metrics=metrics, runs=rows, sources=[provenance(__file__), provenance(harmonic_conduction.__file__), provenance(geometry.__file__)]),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
