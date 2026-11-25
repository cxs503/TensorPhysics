"""Resolve the square-root singularity at a temperature/insulation junction."""
import json

import numpy as np
from scipy.special import iv

from conduction_provenance import provenance
from tensorbem import geometry, harmonic_conduction


def measure(n, quadratic, grading, omega):
    k, capacity = 1.7, 2.3
    wave = np.sqrt(1j * omega * capacity / k)
    t = np.arange(n) / n
    # Split the bottom at x=0 and grade both halves toward the junction.
    vertices = np.vstack([np.column_stack((-(1-t)**grading, 0*t)),
                          np.column_stack((t**grading, 0*t)),
                          np.column_stack((1+0*t, t)),
                          np.column_stack((1-2*t, 1+0*t)),
                          np.column_stack((0*t-1, 1-t))])
    mesh = geometry.PanelMesh(vertices)
    points = harmonic_conduction.harmonic_boundary_points(mesh, quadratic=quadratic)

    def temperature(xy):
        r = np.linalg.norm(xy, axis=1)
        theta = np.arctan2(xy[:, 1], xy[:, 0])
        return iv(.5, wave*r) * np.sin(theta/2)

    insulated = (points[:, 1] == 0) & (points[:, 0] < 0)
    result = harmonic_conduction.solve_mixed_harmonic_conduction(mesh,
        np.where(insulated, "flux", "temperature"), np.where(insulated, 0., temperature(points)),
        omega, conductivity=k, volumetric_heat_capacity=capacity, quadratic=quadratic)

    # On x>0, q_in=-k I_{1/2}(wave*x)/(2*x). Integrate using x=u².
    abscissa, weights = np.polynomial.legendre.leggauss(32)
    u = (abscissa + 1)/2
    power = -k * (weights/2) @ (iv(.5, wave*u*u)/u)
    load = result.panel_inward_power(np.arange(n, 2*n)).sum()
    # Mean wall temperature on the unit-length insulated interval.
    exact_mean = weights @ (u*iv(.5, wave*u*u))
    panel_weights = np.array([5., 8., 5.])/18 if quadratic else np.ones(1)
    wall = result.wall_temperature[insulated].reshape(n, -1)
    mean = mesh.lengths[:n] @ (wall @ panel_weights)
    probes = np.array([[-.6,.25], [-.12,.11], [.08,.09], [.55,.4], [.1,.7]])
    # The annular window approaches the junction while spanning more panels.
    # Exact integral of x^(-1/2) over [a,b] is 2*(sqrt(b)-sqrt(a)).
    panel_starts = mesh.vertices[n:2*n, 0]
    panel_lengths = mesh.lengths[n:2*n]
    a = .5 / np.sqrt(n)

    def window_coefficient(left):
        right = 2*left
        start = np.clip((left-panel_starts)/panel_lengths, 0., 1.)
        end = np.clip((right-panel_starts)/panel_lengths, 0., 1.)
        load = result.panel_inward_power(np.arange(n, 2*n), start, end).sum()
        return load / (2*(np.sqrt(right)-np.sqrt(left)))

    coefficient = window_coefficient(a)
    # The first correction to sqrt(x)*q is proportional to x².
    # Doubling the window multiplies this term by four.
    extrapolated = (4*coefficient - window_coefficient(2*a))/3
    exact_coefficient = -.5*k*np.sqrt(2*wave/np.pi)
    relative_error = abs(coefficient/exact_coefficient - 1)
    return dict(window_m=[a, 2*a], extrapolation_window_m=[2*a, 4*a],
                extrapolated_coefficient=[extrapolated.real, extrapolated.imag],
                extrapolated_relative_error=float(abs(extrapolated/exact_coefficient - 1)), flux_coefficient=[coefficient.real, coefficient.imag],
                exact_flux_coefficient=[exact_coefficient.real, exact_coefficient.imag],
                flux_coefficient_relative_error=float(relative_error), panels=5*n, quadratic=quadratic, grading=grading, omega_rad_s=omega,
                wall_power_error_W_m=float(abs(load-power)), insulated_mean_error_K=float(abs(mean-exact_mean)),
                interior_temperature_error_K=float(max(abs(result.temperature(probes)-temperature(probes)))))


if __name__ == "__main__":
    metrics = ["wall_power_error_W_m", "insulated_mean_error_K", "interior_temperature_error_K"]
    runs = []
    for omega in (.5, 3.):
        for quadratic in (False, True):
            for grading in (1, 2):
                previous = None
                for n in (4, 8, 16, 32):
                    row = measure(n, quadratic, grading, omega)
                    errors = np.array([row[key] for key in metrics])
                    row["orders"] = [] if previous is None else np.log2(previous/errors).tolist()
                    passed = np.isfinite(errors).all() and all(order > 0 for order in row["orders"])
                    if n == 32:
                        passed &= np.all(errors < [.012, .005, .007])
                        passed &= max(row["flux_coefficient_relative_error"], row["extrapolated_relative_error"]) < .025
                        if quadratic and grading == 2:
                            passed &= row["extrapolated_relative_error"] < 5e-4
                            passed &= min(row["orders"]) > 1.9 and np.max(errors) < 1e-4
                    row["passed"] = bool(passed)
                    runs.append(row)
                    previous = errors
    print(json.dumps(dict(reference="I_(1/2)(lambda*r) sin(theta/2), lambda^2=i omega C/k, rectangle [-1,1] x [0,1]",
        reproduction="OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=../TensorBEM/src python examples/bem_mixed_wall_junction.py",
        coefficient_units="W/m^(3/2)", complex_pair_order=["real", "imaginary"],
        order_metrics=metrics, runs=runs, sources=[provenance(p) for p in
        (__file__, geometry.__file__, harmonic_conduction.__file__)]), indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in runs) else 1)
