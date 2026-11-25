"""Verify harmonic BEM heat conduction against two independent planar thermal waves."""
import argparse
import json

import numpy as np

from conduction_provenance import provenance
from tensorbem import geometry, harmonic_conduction


def measure(n, omega, condition="temperature", *, quadratic=False, graded=False, heated=False):
    if graded:
        local = np.arange(n//4)/(n//4)
        angle = (np.arange(4)[:, None]+(1-np.cos(np.pi*local))/2).ravel()*np.pi/2
        mesh = geometry.PanelMesh(np.column_stack((np.cos(angle), np.sin(angle))))
    else:
        mesh = geometry.regular_polygon(n)
    points = harmonic_conduction.harmonic_boundary_points(mesh, quadratic=quadratic)
    count = len(points)
    normals = np.repeat(mesh.normals, count//n, axis=0)
    k, capacity = 1.7, 2.3
    wave = np.sqrt(1j*omega*capacity/k)
    directions = np.array([[.6, .8], [-.8, .6]])
    amplitudes = np.array([.7-.2j, -.3+.4j])

    def modes(points):
        return amplitudes*np.exp(wave*(points @ directions.T-1))

    boundary = modes(points)
    base = -.25+.4j if heated else 0j
    source = 1j*omega*capacity*base
    wall = base+boundary.sum(1)
    flux = k*wave*np.sum(boundary*(normals @ directions.T), axis=1)
    # Fixed arcs stay in place as the polygon is refined.
    kinds = (np.repeat(["temperature", "flux", "film", "film"], count//4)
             if condition == "mixed" else np.full(count, condition))
    film = 2.5+.4*points[:, 0]
    values = np.select([kinds == "flux", kinds == "film"], [flux, wall+flux/film], default=wall)
    result = harmonic_conduction.solve_mixed_harmonic_conduction(
        mesh, kinds, values, omega, conductivity=k, volumetric_heat_capacity=capacity, film_coefficients=film, quadratic=quadratic,
        volumetric_source=source)
    angles = np.arange(21)*2*np.pi/21+.137
    probes = np.concatenate([r*np.column_stack((np.cos(angles), np.sin(angles))) for r in (.15, .55, .83)])
    extra = {}
    if heated:
        # Integrate the analytic storage over the actual polygon, using
        # triangles from the origin to each panel. No BEM flux enters storage.
        nodes, weights = np.polynomial.legendre.leggauss(10)
        u = (nodes+1)/2
        edge = mesh.vertices[:, None, :]+u[None, :, None]*mesh.vectors[:, None, :]
        xy = edge[:, :, None, :]*u[None, None, :, None]
        jac = mesh.vertices[:, 0]*mesh.ends[:, 1]-mesh.vertices[:, 1]*mesh.ends[:, 0]
        field = base+modes(xy.reshape(-1, 2)).sum(1).reshape(n, len(u), len(u))
        volume_temperature = np.einsum("p,i,j,pij->", jac, weights/2, weights*u/2, field)
        panel_weights = np.array([5., 8., 5.])/18 if quadratic else np.array([1.])
        inward_rate = mesh.lengths @ (result.inward_flux.reshape(n, -1) @ panel_weights)
        extra["energy_residual_W_m"] = float(abs(inward_rate+source*mesh.area-1j*omega*capacity*volume_temperature))
    worst = np.argmax(abs(result.inward_flux-flux))
    return dict(**extra, quadratic=quadratic, graded=graded, boundary_unknowns=count, worst_flux_panel=int(worst//(count//n)), worst_flux_condition=str(kinds[worst]), panels=n, omega_rad_s=omega, boundary=condition,
                wall_temperature_error_K=float(abs(result.wall_temperature-wall).max()), interior_error_K=float(abs(result.temperature(probes)-(base+modes(probes).sum(1))).max()),
                boundary_flux_error_W_m2=float(abs(result.inward_flux-flux).max()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary", choices=("temperature", "flux", "film", "mixed"), default="temperature")
    parser.add_argument("--quadratic", action="store_true", help="use discontinuous quadratic panel traces")
    parser.add_argument("--graded", action="store_true", help="cluster panel ends at quadrant boundaries")
    parser.add_argument("--heated", action="store_true", help="add uniform volumetric generation and check total heat balance")
    args = parser.parse_args()
    rows = []
    for omega in (1., 6.):
        previous = None
        refinements = (16, 32, 64, 128, 256) if args.boundary == "mixed" else (16, 32, 64, 128)
        for n in refinements:
            row = measure(n, omega, args.boundary, quadratic=args.quadratic, graded=args.graded, heated=args.heated)
            metrics = ["interior_error_K"]
            if args.boundary != "flux":
                metrics.append("boundary_flux_error_W_m2")
            if args.boundary != "temperature":
                metrics.append("wall_temperature_error_K")
            errors = np.array([row[key] for key in metrics])
            row["order_metrics"] = metrics
            orders = np.log2(previous/errors).tolist() if previous is not None else []
            row.update(orders=orders, passed=bool(np.isfinite(errors).all() and all(p > 0 for p in orders)
                       and (n < 128 or (min(orders) > 1.8 and row.get("energy_residual_W_m", 0.) < .005))))
            rows.append(row)
            previous = errors
    print(json.dumps(dict(runs=rows, sources=[provenance(p) for p in
        (__file__, harmonic_conduction.__file__, geometry.__file__)]), indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
