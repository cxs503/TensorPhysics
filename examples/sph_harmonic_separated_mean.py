"""Verify through-flux in a three-layer slab undergoing a large uniform temperature oscillation."""
import json

import numpy as np

from conduction_provenance import provenance
from tensorsph import heat_conduction_2d as sph


def measure(n, omega, power):
    length = 1/3
    k = np.array([.9, 3.7, 1.8])
    c = np.array([1.6, .85, 2.1])
    surface = np.array([.3, .75])
    r = np.array([.32, .18])
    rate = .27+.34j
    dx = length/n
    t = 0j
    centers = []
    for i, cond in enumerate(k):
        centers.extend(t-power*dx*(np.arange(n)+.5)/cond)
        t -= power*length/cond
        if i < 2:
            t -= power*r[i]
    cap = np.repeat(c, n)
    centers = np.array(centers)
    walls = np.array([0j, t])
    # Midpoint quadrature integrates the piecewise linear profile exactly.
    shift = (dx*(cap @ centers)+surface @ walls)/(dx*cap.sum()+surface.sum())
    walls -= shift
    centers -= shift
    # T=rate/(i omega)+T_linear, so the source supplies the bulk
    # storage everywhere and both material-end fluxes are known exactly.
    load = cap*(rate+1j*omega*centers)
    applied = surface*(rate+1j*omega*walls)+np.array([power, -power])
    contact = np.zeros(3*n-1)
    contact[[n-1, 2*n-1]] = r
    field, wall, inward, cuts, mean = sph.solve_harmonic_layered_films(load, dx, (0., 0.), omega,
        film_coefficients=(0., 0.), conductivity=np.repeat(k, n), volumetric_heat_capacity=cap,
        contact_resistance_x=contact, wall_areal_heat_capacity=surface, wall_applied_flux=applied,
        return_x_flux=True, separate_mean=True, backend="banded")
    return dict(columns_per_layer=n, omega_rad_s=omega, through_flux_W_m2=[power.real, power.imag],
                mean_rate_error_K_s=float(abs(1j*omega*mean.item()-rate)),
                relative_temperature_error_K=float(max(max(abs(field.numpy()-centers)), max(abs(wall.numpy()-walls)))),
                cut_flux_error_W_m2=float(max(abs(cuts.numpy()-power))),
                inward_error_W_m2=float(max(abs(inward.numpy()-[power, -power]))))


if __name__ == "__main__":
    rows = [measure(n, omega, power) for n in (16, 32, 64) for omega in (1., 1e-4, 1e-8, 1e-12)
            for power in (0j, .35-.21j)]
    for row in rows:
        row["passed"] = max(row["mean_rate_error_K_s"], row["relative_temperature_error_K"],
                            row["cut_flux_error_W_m2"], row["inward_error_W_m2"]) < 1e-8
    print(json.dumps(dict(reference="Uniform complex pole plus an exact piecewise-linear zero-capacity-mean temperature and two resistive jumps",
                          reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src python examples/sph_harmonic_separated_mean.py",
                          runs=rows, sources=[provenance(f) for f in (__file__, sph.__file__)]), indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
