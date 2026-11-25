"""Verify harmonic BEM heat conduction against two independent planar thermal waves."""
import argparse
import json

import numpy as np

from conduction_provenance import provenance
from tensorbem import geometry, harmonic_conduction


def measure(n, omega, condition="temperature"):
    mesh = geometry.regular_polygon(n)
    k, capacity = 1.7, 2.3
    wave = np.sqrt(1j*omega*capacity/k)
    directions = np.array([[.6, .8], [-.8, .6]])
    amplitudes = np.array([.7-.2j, -.3+.4j])

    def modes(points):
        return amplitudes*np.exp(wave*(points @ directions.T-1))

    boundary = modes(mesh.centers)
    wall = boundary.sum(1)
    flux = k*wave*np.sum(boundary*(mesh.normals @ directions.T), axis=1)
    # Fixed arcs stay in place as the polygon is refined.
    kinds = (np.repeat(["temperature", "flux", "film", "film"], n//4)
             if condition == "mixed" else np.full(n, condition))
    film = 2.5+.4*mesh.centers[:, 0]
    values = np.select([kinds == "flux", kinds == "film"], [flux, wall+flux/film], default=wall)
    result = harmonic_conduction.solve_mixed_harmonic_conduction(
        mesh, kinds, values, omega, conductivity=k, volumetric_heat_capacity=capacity, film_coefficients=film)
    angles = np.arange(21)*2*np.pi/21+.137
    probes = np.concatenate([r*np.column_stack((np.cos(angles), np.sin(angles))) for r in (.15, .55, .83)])
    worst = np.argmax(abs(result.inward_flux-flux))
    return dict(worst_flux_panel=int(worst), worst_flux_condition=str(kinds[worst]), panels=n, omega_rad_s=omega, boundary=condition,
                wall_temperature_error_K=float(abs(result.wall_temperature-wall).max()), interior_error_K=float(abs(result.temperature(probes)-modes(probes).sum(1)).max()),
                boundary_flux_error_W_m2=float(abs(result.inward_flux-flux).max()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary", choices=("temperature", "flux", "film", "mixed"), default="temperature")
    args = parser.parse_args()
    rows = []
    for omega in (1., 6.):
        previous = None
        refinements = (16, 32, 64, 128, 256) if args.boundary == "mixed" else (16, 32, 64, 128)
        for n in refinements:
            row = measure(n, omega, args.boundary)
            metrics = ["interior_error_K"]
            if args.boundary != "flux":
                metrics.append("boundary_flux_error_W_m2")
            if args.boundary != "temperature":
                metrics.append("wall_temperature_error_K")
            errors = np.array([row[key] for key in metrics])
            row["order_metrics"] = metrics
            orders = np.log2(previous/errors).tolist() if previous is not None else []
            row.update(orders=orders, passed=bool(np.isfinite(errors).all() and all(p > 0 for p in orders)
                       and (n < 128 or min(orders) > 1.8)))
            rows.append(row)
            previous = errors
    print(json.dumps(dict(runs=rows, sources=[provenance(p) for p in
        (__file__, harmonic_conduction.__file__, geometry.__file__)]), indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
