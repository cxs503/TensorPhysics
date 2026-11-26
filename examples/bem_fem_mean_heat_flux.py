"""Compare area-mean Fourier flux on a rotated, flux-driven rectangle.

Run python examples/bem_fem_mean_heat_flux.py with TensorBEM and TensorFEM
installed. Add --distorted to warp the interior FEM nodes. All four walls receive inward flux from two harmonic thermal waves.
"""
import argparse
import json

import numpy as np
import torch

from conduction_provenance import provenance
from tensorbem import geometry, harmonic_conduction
from tensorfem import thermal, thermal_analytic


K, CAPACITY, OMEGA = 1.3, 1.8, 2.1
SIZE = np.array([1.4, .9])
ANGLE = .37
AXES = np.array([[np.cos(ANGLE), np.sin(ANGLE)], [-np.sin(ANGLE), np.cos(ANGLE)]])
RATES = np.sqrt(1j*OMEGA*CAPACITY/K)*np.array([[.6, -.8], [.8, .6]])
AMPLITUDES = np.array([1., -.25+.4j])


def gradient(points):
    return (np.exp(points @ RATES.T)*AMPLITUDES) @ RATES


def measure(n, *, distorted=False):
    mesh = thermal_analytic.structured_quad_mesh(n, n, dtype=torch.float64)
    uv = mesh.nodes.numpy().copy()
    if distorted:
        # A smooth interior displacement keeps all wall panels and loads fixed.
        interior = ((uv > 0) & (uv < 1)).all(axis=1)
        warp = np.sin(np.pi*uv[interior]).prod(axis=1)
        uv[interior] += warp[:, None]*np.array([.12, -.08])
    points = (uv*SIZE) @ AXES
    grid = np.arange(len(points)).reshape(n+1, n+1)
    wall = np.r_[grid[0, :-1], grid[:-1, -1], grid[-1, :0:-1], grid[:0:-1, 0]]
    boundary = geometry.PanelMesh(points[wall])
    samples = harmonic_conduction.harmonic_boundary_points(boundary, quadratic=True)
    inward = K*np.sum(gradient(samples)*boundary.normals.repeat(3, axis=0), axis=1)
    bem = harmonic_conduction.solve_mixed_harmonic_conduction(
        boundary, "flux", inward, OMEGA, conductivity=K, volumetric_heat_capacity=CAPACITY,
        quadratic=True)

    # Integrate each prescribed flux against the Q4 edge shape functions.
    x, w = np.polynomial.legendre.leggauss(6)
    s = (x+1)/2
    samples = boundary.vertices[:, None, :] + s[None, :, None]*boundary.vectors[:, None, :]
    flux = K*np.einsum("eqd,ed->eq", gradient(samples), boundary.normals)
    load = np.zeros(len(points), dtype=complex)
    load[wall] = boundary.lengths*(flux @ (w*(1-s)/2))
    load[np.roll(wall, -1)] += boundary.lengths*(flux @ (w*s/2))
    fem = thermal.ThermalModel(torch.from_numpy(points), mesh.elements, K, 1., CAPACITY,
                               torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float64),
                               nodal_heat=torch.from_numpy(load), element_type="q4")
    state = thermal.solve_harmonic_thermal(fem, OMEGA, sparse=True)
    fem_flux = thermal.thermal_volume_mean_flux(fem, state.temperature).numpy()
    return bem.area_mean_heat_flux(), fem_flux


def run(*, distorted=False):
    exponents = (RATES @ AXES.T)*SIZE
    modes = AMPLITUDES*np.prod(np.expm1(exponents)/exponents, axis=1)
    exact = -K*(modes @ RATES)
    rows = []
    previous = {}
    for n in (4, 8, 16, 32):
        for solver, flux in zip(("TensorBEM_quadratic", "TensorFEM_Q4"), measure(n, distorted=distorted)):
            errors = abs(flux-exact)
            passed = solver not in previous or np.all(previous[solver] > 3*errors)
            if n == 32:
                passed &= max(errors) < (1e-6 if solver == "TensorBEM_quadratic" else .001)
            rows.append(dict(solver=solver, segments_per_side=n, component_errors_W_m2=errors.tolist(),
                             passed=bool(passed)))
            previous[solver] = errors
    return dict(command="python examples/bem_fem_mean_heat_flux.py" + (" --distorted" if distorted else ""),
                fem_mesh="distorted" if distorted else "uniform",
                reference_flux_W_m2=[[v.real, v.imag] for v in exact], runs=rows,
                sources=[provenance(p) for p in (__file__, harmonic_conduction.__file__, geometry.__file__,
                                                  thermal.__file__, thermal_analytic.__file__)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distorted", action="store_true", help="warp the interior FEM nodes")
    report = run(distorted=parser.parse_args().distorted)
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in report["runs"]) else 1)
