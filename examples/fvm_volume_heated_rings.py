"""Manufactured quadratic temperature modes with independent heat sources in three annuli."""
import json

import numpy as np

from conduction_provenance import provenance
from tensorfvm.cht import annulus, storing_rings


def measure(n, mode, omega):
    edges = np.array([.4, .55, .9, 1.1])
    k = np.array([1., 2.5, .8])
    cv = np.array([.7, 2.3, 1.2])
    sheet_cv = np.array([.25, .8])
    sheet_k = np.array([.14, .35])
    resistance = np.array([.5, .9])
    fraction = np.array([.2, .7])
    radius = edges[1:-1]
    # T_i = A_i + B_i*r². Pick successive offsets to impose both contact drops.
    b = np.array([.4+.1j, -.3+.6j, .8-.2j])
    a = np.empty(3, dtype=complex)
    a[0] = .7-.3j
    incoming = -2*k[:-1]*b[:-1]*radius
    outgoing = -2*k[1:]*b[1:]*radius
    sheet = np.empty(2, dtype=complex)
    for i, r in enumerate(radius):
        sheet[i] = a[i]+b[i]*r*r - fraction[i]*resistance[i]*incoming[i]
        a[i+1] = sheet[i]-(1-fraction[i])*resistance[i]*outgoing[i]-b[i+1]*r*r
    admittance = 1j*omega*sheet_cv + sheet_k*(mode/radius)**2
    generation = outgoing-incoming + admittance*sheet
    mesh = np.r_[np.concatenate([np.geomspace(l, h, n+1)[:-1] for l, h in zip(edges, edges[1:])]), edges[-1]]
    lo, hi = mesh[:-1], mesh[1:]
    area = np.pi*(hi**2-lo**2)
    # Exact shell integrals of storage, azimuthal divergence, and radial divergence.
    mean_t = np.repeat(a, n) + np.repeat(b, n)*(lo**2+hi**2)/2
    stored = 1j*omega*np.repeat(cv, n)*mean_t*area
    lateral = 2*np.pi*mode**2*np.repeat(k, n)*(np.repeat(a, n)*np.log(hi/lo)
                                                + np.repeat(b, n)*(hi**2-lo**2)/2)
    density = (stored+lateral)/area - np.repeat(4*k*b, n)
    walls = a[[0,-1]]+b[[0,-1]]*edges[[0,-1]]**2
    expected_wall = -4*np.pi*(k*b)[[0,-1]]*edges[[0,-1]]**2
    temp, qi, qo, rates, storage, angular = storing_rings.solve_storing_rings(
        mesh, np.repeat(k, n), [n, 2*n], walls, omega, contact_resistances=resistance,
        sheet_capacities=sheet_cv, sheet_generation=generation, source=density,
        inner_resistance_fractions=fraction, sheet_conductivities=sheet_k, azimuthal_mode=mode,
        volumetric_heat_capacity=np.repeat(cv, n), return_wall_rates=True,
        return_bulk_storage=True, return_angular_conduction=True)
    reference = np.r_[sheet, incoming, outgoing, expected_wall]
    error = float(max(abs(np.r_[temp, qi, qo, rates]/reference-1)))
    storage_error = float(max(abs(storage/stored.reshape(3, n).sum(axis=1)-1)))
    angular_error = float(max(abs(angular/lateral.reshape(3, n).sum(axis=1)-1))) if mode else float(max(abs(angular)))
    sheet_net = 2*np.pi*radius @ (generation-admittance*temp)
    balance = abs(rates[0]-rates[1]+sheet_net+area@density-sum(storage)-sum(angular))
    return dict(cells_per_layer=n, mode=mode, omega=omega, max_relative_error=error,
                storage_relative_error=storage_error, angular_error=angular_error, modal_balance_W_m=float(balance))


if __name__ == "__main__":
    rows = []
    for mode in (0, 1, 3):
        for omega in (1., 8., 24.):
            previous = None
            for n in (4, 8, 16, 32):
                row = measure(n, mode, omega)
                error = row["max_relative_error"]
                order = None if previous is None else float(np.log2(previous/error))
                row.update(order=order, passed=row["modal_balance_W_m"] < 1e-9 and
                           (n < 32 or (max(error, row["storage_relative_error"], row["angular_error"]) < .001 and order > 1.8)))
                rows.append(row)
                previous = error
    passed = all(row["passed"] for row in rows)
    print(json.dumps(dict(passed=passed, runs=rows, sources=[provenance(p) for p in
          (__file__, annulus.__file__, storing_rings.__file__)],
          reproduction="OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFVM/src python examples/fvm_volume_heated_rings.py"), indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
