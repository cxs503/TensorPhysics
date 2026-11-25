"""Verify independent applied flux at storing walls of a three-material particle slab."""
import json

import numpy as np
from scipy.linalg import expm

from conduction_provenance import provenance
from tensorsph import heat_conduction_2d as sph


def measure(n, omega, films):
    k, c = np.array([.9, 3.7, 1.8]), np.array([1.6, .85, 2.1])
    r = np.array([.32, .18])
    wall_c = np.array([.28, .63])
    applied = np.array([.8+.3j, -.4+.65j])
    ambient = np.array([-.2+.4j, .65+.12j])
    dx = 1/(3*n)
    matrices = [expm(np.array([[0., -1/cond], [-1j*omega*cap, 0.]])/3) for cond, cap in zip(k, c)]
    transfer = np.eye(2)
    for i, layer in enumerate(matrices):
        transfer = layer @ transfer
        if i < len(r):
            transfer = np.array([[1., -r[i]], [0., 1.]]) @ transfer
    y = films+1j*omega*wall_c
    west = np.linalg.solve(np.vstack(([y[0], 1.], [y[1], -1.] @ transfer)), applied+films*ambient)
    exact_wall = np.array([west[0], (transfer @ west)[0]])
    state = west
    contacts = []
    for i in range(2):
        t, q = matrices[i] @ state
        state = np.array([t-r[i]*q, q])
        contacts.append([t, state[0], q])
    exact_left, exact_right, exact_q = np.asarray(contacts).T

    cuts = np.array([n, 2*n])
    resistance = np.zeros(3*n-1)
    resistance[cuts-1] = r
    cond = np.repeat(k, n)
    field, wall, inward, flux = sph.solve_harmonic_layered_films(np.zeros(3*n), dx, ambient, omega,
        film_coefficients=films, conductivity=cond, volumetric_heat_capacity=np.repeat(c, n),
        contact_resistance_x=resistance, wall_areal_heat_capacity=wall_c, wall_applied_flux=applied,
        backend="banded", return_x_flux=True)
    left, right = sph.layered_slab_contact_temperatures(field, flux, dx, cond)
    left, right = left.numpy()[cuts-1], right.numpy()[cuts-1]
    q = flux.numpy()[cuts]
    wall = wall.numpy()
    boundaries = flux.numpy()[[0, n, 2*n, 3*n]]
    bulk_rate = 1j*omega*c*dx*field.numpy().reshape(3, n).sum(axis=1)
    return dict(columns_per_layer=n, omega_rad_s=omega, films_W_m2K=films.tolist(),
                wall_error_K=float(max(abs(wall-exact_wall))),
                contact_trace_error_K=float(max(max(abs(left-exact_left)), max(abs(right-exact_right)))),
                contact_flux_error_W_m2=float(max(abs(q-exact_q))),
                jump_error_K=float(max(abs(left-right-r*exact_q))),
                layer_balance_W_m2=float(max(abs(np.diff(boundaries)+bulk_rate))),
                wall_balance_W_m2=float(max(abs(inward.numpy()+1j*omega*wall_c*wall-applied-films*(ambient-wall)))))


if __name__ == "__main__":
    rows = []
    keys = ["wall_error_K", "contact_trace_error_K", "contact_flux_error_W_m2", "jump_error_K"]
    for films in (np.zeros(2), np.array([0., 2.4]), np.array([1.5, 2.4])):
        for omega in (.5, 4.):
            previous = None
            for n in (8, 16, 32, 64):
                row = measure(n, omega, films)
                errors = np.array([row[k] for k in keys])
                orders = np.log2(previous/errors).tolist() if previous is not None else []
                row.update(orders=orders, passed=bool(all(p > 1.8 for p in orders)
                           and (n < 64 or max(errors) < 1e-4)
                           and max(row["wall_balance_W_m2"], row["layer_balance_W_m2"]) < 1e-9))
                rows.append(row)
                previous = errors
    print(json.dumps(dict(reference="Three layer propagators and two temperature jumps with separately driven storing walls",
                          reproduction="OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src python examples/sph_three_layer_wall_flux.py",
                          order_metrics=keys, sources=[provenance(f) for f in (__file__, sph.__file__)],
                          runs=rows), indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
