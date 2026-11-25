"""Verify the temperature gauge of a three-layer SPH slab with balanced Neumann data."""
import json

import numpy as np

from conduction_provenance import provenance
from tensorsph import heat_conduction_2d as sph


def measure(n, wall_capacity):
    length = 1/3
    dx = length/n
    k = np.array([.9, 3.7, 1.8])
    c = np.array([1.6, .85, 2.1])
    resistance = np.array([.32, .18])
    generation = np.array([1.2-.3j, -.8+.65j, .45+.2j])
    west_q = .35-.21j
    west_t = .61+.12j
    t, q = west_t, west_q
    stored = 0j
    centers, cuts, traces = [], [q], []
    for i in range(3):
        x = dx*(np.arange(n)+.5)
        centers.extend(t-(q*x+generation[i]*x*x/2)/k[i])
        cuts.extend(q+generation[i]*dx*np.arange(1, n+1))
        # Analytic integral of the quadratic temperature within this layer.
        stored += c[i]*length*(t-(q*length/2+generation[i]*length**2/6)/k[i])
        t -= (q*length+generation[i]*length**2/2)/k[i]
        q += generation[i]*length
        if i < 2:
            traces.extend([t, t-resistance[i]*q])
            t -= resistance[i]*q
    walls = np.array([west_t, t])
    target = (stored+wall_capacity @ walls)/(length*c.sum()+wall_capacity.sum())
    interfaces = np.array([n-1, 2*n-1])
    r = np.zeros(3*n-1)
    r[interfaces] = resistance
    cond = np.repeat(k, n)
    field, wall, inward, flux = sph.solve_harmonic_layered_films(
        np.repeat(generation, n), dx, (0., 0.), 0., conductivity=cond, volumetric_heat_capacity=np.repeat(c, n),
        contact_resistance_x=r, film_coefficients=(0., 0.), wall_applied_flux=(west_q, -q),
        wall_areal_heat_capacity=wall_capacity, backend="banded", return_x_flux=True, mean_temperature=target)
    left, right = sph.layered_slab_contact_temperatures(field, flux, dx, cond)
    computed_traces = np.column_stack((left.numpy()[interfaces], right.numpy()[interfaces])).ravel()
    mean = (dx*np.dot(np.repeat(c, n), field.numpy())+wall_capacity @ wall.numpy())/(length*c.sum()+wall_capacity.sum())
    return dict(columns_per_layer=n, wall_capacity_J_m2K=wall_capacity.tolist(),
                temperature_error_K=float(max(abs(field.numpy()-centers))),
                wall_error_K=float(max(abs(wall.numpy()-walls))),
                contact_trace_error_K=float(max(abs(computed_traces-traces))),
                flux_error_W_m2=float(max(abs(flux.numpy()-cuts))),
                inward_error_W_m2=float(max(abs(inward.numpy()-[west_q, -q]))),
                mean_residual_K=float(abs(mean-target)))


if __name__ == "__main__":
    rows = []
    metrics = ["temperature_error_K", "wall_error_K", "contact_trace_error_K"]
    for wall_capacity in (np.zeros(2), np.array([.28, .63])):
        previous = None
        for n in (8, 16, 32, 64):
            row = measure(n, wall_capacity)
            errors = np.array([row[k] for k in metrics])
            orders = np.log2(previous/errors).tolist() if previous is not None else []
            row.update(orders=orders, passed=bool(all(p > 1.8 for p in orders)
                       and (n < 64 or max(errors) < 5e-5)
                       and row["mean_residual_K"] < 1e-12 and max(row["inward_error_W_m2"], row["flux_error_W_m2"]) < 1e-9))
            rows.append(row)
            previous = errors
    print(json.dumps(dict(reference="Piecewise quadratic temperature, linear rightward flux, exact heat-capacity integrals and two resistive jumps",
                          reproduction="OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src python examples/sph_steady_flux_contact_mean.py",
                          sources=[provenance(f) for f in (__file__, sph.__file__)], order_metrics=metrics, runs=rows), indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
