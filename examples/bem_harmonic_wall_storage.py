"""Check the phasor response of a unit square slab with a storing film at x=1."""
import json

import numpy as np

from conduction_provenance import provenance
from tensorbem import geometry, harmonic_conduction


def measure(panels_per_side, omega, quadratic):
    k, bulk_capacity, wall_capacity, h = 1.3, 2., .8, 2.4
    ambient = .9-.3j
    corners = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    t = np.arange(panels_per_side)[:, None]/panels_per_side
    mesh = geometry.PanelMesh(np.concatenate([
        a+(b-a)*t for a, b in zip(corners, np.roll(corners, -1, axis=0))]))
    right = mesh.normals[:, 0] == 1
    result = harmonic_conduction.solve_mixed_harmonic_conduction(
        mesh, np.where(right, "film", "flux"), np.where(right, ambient, 0.), omega,
        conductivity=k, volumetric_heat_capacity=bulk_capacity, film_coefficients=h,
        wall_areal_heat_capacity=right*wall_capacity, quadratic=quadratic)

    # The insulated end imposes T(x)=A cosh(lambda x). At x=1 the
    # film supplies both bulk conduction and the thin wall's storage.
    wave = np.sqrt(1j*omega*bulk_capacity/k)
    amplitude = h*ambient/((h+1j*omega*wall_capacity)*np.cosh(wave)+k*wave*np.sinh(wave))
    wall = amplitude*np.cosh(wave)
    bulk_flux = k*wave*amplitude*np.sinh(wave)
    surface_rate = 1j*omega*wall_capacity*wall
    sample_right = np.repeat(right, 3 if quadratic else 1)
    temp = result.wall_temperature[sample_right]
    flux = result.inward_flux[sample_right]
    external = result.external_inward_flux[sample_right]
    x = np.array([.1, .35, .6, .9])
    probe = result.temperature(np.column_stack((x, np.full_like(x, .47))))
    return dict(panels=mesh.panel_count, omega_rad_s=omega, quadratic=quadratic,
                probe_error_K=float(max(abs(probe-amplitude*np.cosh(wave*x)))),
                wall_error_K=float(max(abs(temp-wall))),
                bulk_flux_error_W_m2=float(max(abs(flux-bulk_flux))),
                surface_storage_error_W_m2=float(max(abs((external-flux)-surface_rate))),
                external_flux_error_W_m2=float(max(abs(external-h*(ambient-wall)))))


if __name__ == "__main__":
    rows = []
    passed = True
    for quadratic in (False, True):
        for omega in (.5, 4.):
            levels = (8, 16, 32) if quadratic else (8, 16, 32, 64, 128)
            series = [measure(n, omega, quadratic) for n in levels]
            rows.extend(series)
            for metric in series[-1]:
                if "_error_" in metric:
                    errors = [row[metric] for row in series]
                    passed &= all(b < a for a, b in zip(errors, errors[1:])) and errors[-1] < .003
    print(json.dumps(dict(reference="T(x)=A cosh(lambda x), insulated at x=0, storing film at x=1",
                          reproduction="OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorBEM/src python examples/bem_harmonic_wall_storage.py",
                          solver=provenance(harmonic_conduction.__file__), benchmark=provenance(__file__),
                          runs=rows, passed=passed), indent=2))
    raise SystemExit(0 if passed else 1)
