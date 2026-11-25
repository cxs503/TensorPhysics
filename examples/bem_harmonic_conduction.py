"""Verify harmonic BEM heat conduction against two independent planar thermal waves."""
import json

import numpy as np

from conduction_provenance import provenance
from tensorbem import geometry, harmonic_conduction


def measure(n, omega):
    mesh = geometry.regular_polygon(n)
    k, capacity = 1.7, 2.3
    wave = np.sqrt(1j*omega*capacity/k)
    directions = np.array([[.6, .8], [-.8, .6]])
    amplitudes = np.array([.7-.2j, -.3+.4j])

    def modes(points):
        return amplitudes*np.exp(wave*(points @ directions.T-1))

    boundary = modes(mesh.centers)
    result = harmonic_conduction.solve_harmonic_conduction(
        mesh, boundary.sum(1), omega, conductivity=k, volumetric_heat_capacity=capacity)
    flux = k*wave*np.sum(boundary*(mesh.normals @ directions.T), axis=1)
    angles = np.arange(21)*2*np.pi/21+.137
    probes = np.concatenate([r*np.column_stack((np.cos(angles), np.sin(angles))) for r in (.15, .55, .83)])
    return dict(panels=n, omega_rad_s=omega, interior_error_K=float(abs(result.temperature(probes)-modes(probes).sum(1)).max()),
                boundary_flux_error_W_m2=float(abs(result.inward_flux-flux).max()))


if __name__ == "__main__":
    rows = []
    for omega in (1., 6.):
        previous = None
        for n in (16, 32, 64, 128):
            row = measure(n, omega)
            errors = np.array([row["interior_error_K"], row["boundary_flux_error_W_m2"]])
            orders = np.log2(previous/errors).tolist() if previous is not None else []
            row.update(orders=orders, passed=bool(np.isfinite(errors).all() and all(p > 0 for p in orders)
                       and (n < 128 or min(orders) > 1.8)))
            rows.append(row)
            previous = errors
    print(json.dumps(dict(runs=rows, sources=[provenance(p) for p in
        (__file__, harmonic_conduction.__file__, geometry.__file__)]), indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
