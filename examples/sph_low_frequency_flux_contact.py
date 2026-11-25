"""Verify the zero-mean Taylor branch of a balanced flux-driven three-material slab."""
import argparse
import json

import numpy as np
from numpy.polynomial import Polynomial

from conduction_provenance import provenance
from tensorsph import heat_conduction_2d as sph


def measure(n, wall_capacity, unbalanced=False):
    length = 1/3
    dx = length/n
    k = np.array([.9, 3.7, 1.8])
    c = np.array([1.6, .85, 2.1])
    r = np.array([.32, .18])
    generation = np.array([1.2-.3j, -.8+.65j, .45+.2j])
    left_drive = .35-.21j
    drive = np.array([left_drive, -left_drive-length*generation.sum()])
    excess = .47-.38j if unbalanced else 0j
    drive[1] += excess
    exact_residue = excess/(length*c.sum()+wall_capacity.sum())
    contact = np.zeros(3*n-1)
    contact[[n-1, 2*n-1]] = r
    opts = dict(conductivity=np.repeat(k, n), volumetric_heat_capacity=np.repeat(c, n),
                film_coefficients=(0., 0.), contact_resistance_x=contact, wall_applied_flux=drive,
                wall_areal_heat_capacity=wall_capacity, backend="banded")
    args = (np.repeat(generation, n), dx, (0., 0.))
    result = sph.solve_low_frequency_layered_films(*args, order=2, return_pole=unbalanced, **opts)
    terms = result[1] if unbalanced else result
    previous = []
    reference_walls = np.zeros(2)
    metrics = {}
    if unbalanced:
        residue = result[0].item()
        metrics.update(residue_K_s=[residue.real, residue.imag], residue_error_K_s=float(abs(residue-exact_residue)))
    for m, (field, walls, inward) in enumerate(terms):
        # At order m>0, q'_m=-C*T_(m-1). Integrate first q then T
        # in each material, carrying the resistive temperature jumps.
        q = left_drive-wall_capacity[0]*exact_residue if m == 0 else -wall_capacity[0]*reference_walls[0]
        west_inward = q
        t = 0j
        fields = []
        for i in range(3):
            load = Polynomial([generation[i]-c[i]*exact_residue]) if m == 0 else -c[i]*previous[i]
            power = load.integ(k=q)
            polynomial = (-power/k[i]).integ(k=t)
            fields.append(polynomial)
            q, t = power(length), polynomial(length)
            if i < 2:
                t -= r[i]*q
        norm = sum(cap*p.integ()(length) for cap, p in zip(c, fields))+wall_capacity[1]*t
        shift = -norm/(length*c.sum()+wall_capacity.sum())
        previous = [p+shift for p in fields]
        reference_walls = np.array([shift, t+shift])
        exact = np.concatenate([p((np.arange(n)+.5)*dx) for p in previous])
        metrics[f"T{m}_error"] = float(max(np.max(abs(field.numpy()-exact)),
                                            np.max(abs(walls.numpy()-reference_walls))))
        metrics[f"q{m}_error"] = float(np.max(abs(inward.numpy()-[west_inward, -q])))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unbalanced", action="store_true", help="include the spatially uniform pole due to net harmonic heating")
    unbalanced = parser.parse_args().unbalanced
    rows = []
    metrics = ["T0_error", "T1_error", "T2_error"]
    for cw in (np.zeros(2), np.array([.28, .63])):
        previous = None
        for n in (8, 16, 32, 64):
            row = measure(n, cw, unbalanced)
            errors = np.array([row[k] for k in metrics])
            orders = np.log2(previous/errors).tolist() if previous is not None else []
            row.update(columns_per_layer=n, wall_capacity_J_m2K=cw.tolist(), orders=orders,
                       passed=bool(all(p > 1.8 for p in orders) and row["q0_error"] < 1e-9
                                   and row.get("residue_error_K_s", 0.) < 1e-13
                                   and (n < 64 or max(row[k] for k in metrics+["q1_error", "q2_error"]) < 2e-5)))
            rows.append(row)
            previous = errors
    print(json.dumps(dict(reference="Piecewise polynomial integration of all three coefficient equations, capacity-weighted mean zero at every order",
                          units="Tm: K s^m, qm: W s^m/m², coefficients of (i omega)^m",
                          order_metrics=metrics, runs=rows, sources=[provenance(f) for f in (__file__, sph.__file__)],
                          reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src python examples/sph_low_frequency_flux_contact.py"+(" --unbalanced" if unbalanced else "")),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
