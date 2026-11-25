"""Check two storing sheets in a layered slab with pre-existing contact resistances."""
import json

import numpy as np
import torch
from tensorsph import heat_conduction_2d as heat

from conduction_provenance import provenance


def measure(n, method):
    dx = 1/n
    x = (np.arange(n)+.5)*dx
    k = np.where(x < .5, 1., 2.5)
    cv = np.where(x < .5, 1.5, .75)
    faces = np.array([n//4, n//2, 3*n//4])
    contacts = np.zeros(n-1)
    contacts[faces-1] = [.1, .3, .2]
    resistance = [.4, .7]
    capacity = [.25, .8]
    material = dict(conductivity=k, volumetric_heat_capacity=cv, contact_resistance_x=contacts)
    sheets = [faces[0], faces[2]+2]
    for face, r, c in zip(sheets, resistance, capacity):
        material = heat.two_column_sheet_materials(**material, dx=dx, face=face,
                                                   resistance=r, areal_heat_capacity=c, sheet_conductivity=1/dx)
    # Independent unit-area series solution on the original physical mesh.
    q = 1./(sum(dx/k)+sum(contacts)+sum(resistance))
    upstream = np.cumsum(dx/k)-dx/(2*k)+np.r_[0., np.cumsum(contacts)]
    for face, r in zip(faces[::2], resistance):
        upstream[face:] += r
    field, cuts = heat.solve_harmonic_layered_slab(torch.zeros(n+4), dx, (1., 0.), 0.,
                                                  **material, return_x_flux=True)
    bulk = np.ones(n+4, dtype=bool)
    for face in sheets:
        bulk[face:face+2] = False
    temperature_error = float(abs(field.real.numpy()[bulk]-(1-q*upstream)).max())
    flux_error = float(abs(cuts.real.numpy()/q-1).max())
    total_c = dx*float(material["volumetric_heat_capacity"].sum())
    capacity_error = abs(total_c-(dx*cv.sum()+sum(capacity)))
    # Quench both ends to zero. Integrated left-minus-right flux must equal
    # each sheet's energy change, including heat crossing two-column pairs.
    states, transfers = heat.solve_transient_layered_slab(
        field.real, np.linspace(0., .12, 25), dx, (0., 0.), **material,
        method=method, return_x_heat=True)
    errors = []
    for face, c in zip(sheets, capacity):
        delta = c*states[:, face:face+2].mean(dim=1).diff()
        errors.append(float(abs(delta-transfers[:, face]+transfers[:, face+2]).max()))
    total_energy = states @ (dx*material["volumetric_heat_capacity"])
    balance = float(abs(total_energy.diff()-transfers[:, 0]+transfers[:, -1]).max())
    return dict(bulk_columns=n, method=method, max_bulk_temperature_error_K=temperature_error,
                max_relative_cut_flux_error=flux_error, capacity_error_J_per_m2_K=capacity_error,
                sheet_balance_errors_J_per_m2=errors, total_balance_error_J_per_m2=balance,
                passed=max(temperature_error, flux_error, capacity_error, *errors, balance) < 1e-11)


if __name__ == "__main__":
    runs = [measure(n, method) for n in (16, 32, 64) for method in ("theta", "sdirk2")]
    passed = all(r["passed"] for r in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(p) for p in (__file__, heat.__file__)],
                          reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src python examples/sph_layered_storing_contacts.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
