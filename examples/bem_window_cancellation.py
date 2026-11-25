"""Cancel the integrated external heat load on one quadrant of an inscribed polygon."""
import json

import numpy as np
from tensorbem import geometry, harmonic_conduction

import bem_disk_thermal_modes as modes
import bem_patterned_film_modes as films
from conduction_provenance import provenance


def measure(n, omega, quadratic):
    mesh = geometry.regular_polygon(n)
    k, capacity = 1.7, 2.3
    wave = np.sqrt(1j*omega*capacity/k)
    panels = np.arange(n//4)
    nodes, weights = np.polynomial.legendre.leggauss(32)
    pts = mesh.vertices[panels, None]+((1+nodes)/2)[None, :, None]*mesh.vectors[panels, None]
    surface = .4+.22*np.sin(3*np.arctan2(pts[..., 1], pts[..., 0])-.2)
    ds = mesh.lengths[panels, None]*weights/2
    def external(mode):
        t, q = modes.reference(pts.reshape(-1, 2), mode, wave, k)
        return (-(q.reshape(pts.shape)*mesh.normals[panels, None]).sum(2)
                + 1j*omega*surface*t.reshape(surface.shape))

    first, second = external(-1), external(2)
    coefficient = -np.sum(ds*first)/np.sum(ds*second)
    target = first+coefficient*second

    def reference(points):
        t, q = modes.reference(points, -1, wave, k)
        u, v = modes.reference(points, 2, wave, k)
        return t+coefficient*u, q+coefficient*v

    result, *_ = films.solve_film(mesh, omega, quadratic, k, capacity, reference)
    loads = result.panel_inward_power(panels, component="external")
    exact = (ds*target).sum(axis=1)
    # Each half retains finite complex power even though their sum vanishes.
    half = len(panels)//2
    amounts = np.array([loads[:half].sum(), loads[half:].sum()])
    targets = np.array([exact[:half].sum(), exact[half:].sum()])
    scale = abs(targets).sum()
    return dict(panels=n, omega_rad_s=omega, quadratic=quadratic,
                coefficient_real_imag=[coefficient.real, coefficient.imag],
                absolute_reference_integral_W_m=float(np.sum(ds*abs(target))),
                reference_half_powers_W_m=np.column_stack((targets.real, targets.imag)).tolist(),
                measured_half_powers_W_m=np.column_stack((amounts.real, amounts.imag)).tolist(),
                reference_net_W_m=float(abs(exact.sum())), measured_net_W_m=float(abs(loads.sum())),
                relative_net_residual=float(abs(loads.sum())/scale),
                max_half_power_error_W_m=float(abs(amounts-targets).max()))


if __name__ == "__main__":
    runs = []
    for quadratic in (False, True):
        for omega in (.5, 5.):
            previous = None
            for n in (16, 32, 64):
                row = measure(n, omega, quadratic)
                errors = np.array([row["relative_net_residual"], row["max_half_power_error_W_m"]])
                orders = [] if previous is None else np.log2(previous/errors).tolist()
                row.update(orders=orders, passed=bool(row["reference_net_W_m"] < 1e-12
                           and row["absolute_reference_integral_W_m"] > .1
                           and (not orders or min(orders) > 0.)
                           and (n < 64 or min(orders) > (2.5 if quadratic else 1.8))))
                runs.append(row)
                previous = errors
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(p) for p in
                          (__file__, films.__file__, modes.__file__, harmonic_conduction.__file__, geometry.__file__)],
                          reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorBEM/src python examples/bem_window_cancellation.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
