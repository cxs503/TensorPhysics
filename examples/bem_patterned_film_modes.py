"""Harmonic conduction with spatially varying film and wall storage on polygon panels."""
import argparse
import json

import numpy as np
from tensorbem import geometry, harmonic_conduction

import bem_disk_thermal_modes as modes
from conduction_provenance import provenance


def window_powers(solution, omega, reference):
    mesh = solution.mesh
    step = 2*np.pi/mesh.panel_count
    nodes, weights = np.polynomial.legendre.leggauss(16)
    windows = []
    for begin, end in ((.19, 1.41), (5.83, 2*np.pi+.47)):
        segments = np.arange(int(np.floor(begin/step)), int(np.ceil(end/step)))
        mid = (segments+.5)*step
        panels = segments % mesh.panel_count
        # Intersect rays at the two window limits with the straight chords.
        def fraction(angle):
            return np.clip(.5+.5*np.tan(angle-mid)/np.tan(step/2), 0., 1.)
        lo = fraction(np.maximum(begin, segments*step))
        hi = fraction(np.minimum(end, (segments+1)*step))
        measured = np.array([solution.panel_inward_power(panels, lo, hi, component=kind).sum()
                             for kind in ("bulk", "storage", "external")])
        frac = (lo+hi)[:, None]/2+(hi-lo)[:, None]*nodes/2
        points = mesh.vertices[panels, None]+frac[:, :, None]*mesh.vectors[panels, None]
        wall, flux = reference(points.reshape(-1, 2))
        bulk = -(flux.reshape(points.shape)*mesh.normals[panels, None]).sum(axis=2)
        angle = np.arctan2(points[..., 1], points[..., 0])
        storage = 1j*omega*(.4+.22*np.sin(3*angle-.2))*wall.reshape(frac.shape)
        integrals = (mesh.lengths[panels]*(hi-lo)/2) @ (np.stack((bulk, storage), axis=-1)*weights[None, :, None]).sum(axis=1)
        exact = np.r_[integrals, integrals.sum()]
        windows.append(dict(angular_interval_rad=[begin, end], complex_power_errors_W_m=abs(measured-exact).tolist(),
                            phase_errors_rad=abs(np.angle(measured/exact)).tolist(),
                            reference_real_imag_W_m=np.column_stack((exact.real, exact.imag)).tolist(),
                            measured_real_imag_W_m=np.column_stack((measured.real, measured.imag)).tolist(),
                            balance_W_m=float(abs(measured[2]-measured[1]-measured[0]))))
    return windows


def measure(n, omega, quadratic, mode, integrate=False, cancel=False):
    k, capacity = 1.7, 2.3
    wave = np.sqrt(1j*omega*capacity/k)
    cancellation = {}
    coefficient = 0j
    other = 2 if mode == -1 else -1
    if cancel:
        probe = np.array([[.41, -.27]])
        first_flux = modes.reference(probe, mode, wave, k)[1][0, 0]
        second_flux = modes.reference(probe, other, wave, k)[1][0, 0]
        coefficient = -first_flux/second_flux

    def reference(points):
        temperature, flux = modes.reference(points, mode, wave, k)
        if cancel:
            other_temperature, other_flux = modes.reference(points, other, wave, k)
            temperature += coefficient*other_temperature
            flux += coefficient*other_flux
        return temperature, flux

    mesh = geometry.regular_polygon(n)
    points = harmonic_conduction.harmonic_boundary_points(mesh, quadratic=quadratic)
    angle = np.arctan2(points[:, 1], points[:, 0])
    film = 2.+.65*np.cos(2*angle+.3)
    surface = .4+.22*np.sin(3*angle-.2)
    wall, flux = reference(points)
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
    ref_temp, ref_flux = reference(probes)
    if cancel:
        remaining = result.heat_flux(probe)[0, 0]
        cancellation = dict(second_mode=other, coefficient_real_imag=[coefficient.real, coefficient.imag],
                            cancellation_probe_m=probe[0].tolist(), single_mode_x_flux_W_m2=float(abs(first_flux)),
                            residual_x_flux_W_m2=float(abs(remaining)),
                            relative_cancellation_residual=float(abs(remaining/first_flux)))
    stored = 1j*omega*surface*result.wall_temperature
    extra = dict(windows=window_powers(result, omega, reference)) if integrate else {}
    return dict(**extra, **cancellation, panels=n, quadratic=quadratic, omega_rad_s=omega, angular_mode=mode,
                interior_temperature_error_K=float(abs(result.temperature(probes)-ref_temp).max()),
                interior_flux_error_W_m2=float(abs(result.heat_flux(probes)-ref_flux).max()),
                wall_temperature_error_K=float(abs(result.wall_temperature-wall).max()),
                material_flux_error_W_m2=float(abs(result.inward_flux-inward).max()),
                storage_rate_error_W_m2=float(abs(stored-storage).max()),
                external_flux_error_W_m2=float(abs(result.external_inward_flux-external).max()),
                wall_balance_residual_W_m2=float(abs(result.external_inward_flux-stored-result.inward_flux).max()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", action="store_true", help="verify complex powers over two partial boundary windows")
    parser.add_argument("--cancel", action="store_true", help="combine angular modes to cancel x flux at an interior probe")
    args = parser.parse_args()
    integrate = args.windows
    metrics = ["interior_temperature_error_K", "interior_flux_error_W_m2", "wall_temperature_error_K",
               "material_flux_error_W_m2", "storage_rate_error_W_m2", "external_flux_error_W_m2"]
    if args.cancel:
        metrics.append("relative_cancellation_residual")
    runs = []
    for quadratic in (False, True):
        for omega in (.5, 5.):
            for mode in ((-1,) if args.cancel else (-1, 2)):
                previous = None
                for n in (16, 32, 64):
                    row = measure(n, omega, quadratic, mode, integrate, args.cancel)
                    errors = np.array([row[key] for key in metrics])
                    orders = [] if previous is None else np.log2(previous/errors).tolist()
                    row.update(orders=orders, passed=bool(row["wall_balance_residual_W_m2"] < 1e-11
                               and (not orders or min(orders) > 0.)
                               and (n < 64 or min(orders) > (2.5 if quadratic else 1.8))))
                    if args.cancel and n == 64:
                        row["passed"] &= row["relative_cancellation_residual"] < (1e-7 if quadratic else .005)
                    if integrate:
                        windows = row["windows"]
                        peak = max(max(w["complex_power_errors_W_m"]) for w in windows)
                        phase = max(max(w["phase_errors_rad"]) for w in windows)
                        row["passed"] &= all(w["balance_W_m"] < 1e-11 for w in windows)
                        if n == 64:
                            tolerance = 1e-5 if quadratic else .03
                            row["passed"] &= peak < tolerance and phase < tolerance
                    runs.append(row)
                    previous = errors
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, order_metrics=metrics, runs=runs, sources=[provenance(path) for path in
                          (__file__, modes.__file__, harmonic_conduction.__file__, geometry.__file__)],
                          reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorBEM/src python examples/bem_patterned_film_modes.py"
                                       + (" --windows" if integrate else "") + (" --cancel" if args.cancel else "")),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
