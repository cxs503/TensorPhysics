"""Frequency response of material temperatures and jumps at storing annular contacts."""
import argparse
import json

import numpy as np

import bem_harmonic_storing_rings as geometry
import fvm_harmonic_storing_rings as analytic
from conduction_provenance import provenance
from tensorfvm.cht import annulus, storing_rings


def measure(films, fractions, capacities=(.7, 2.3, 1.2)):
    edges = np.array(geometry.RADII)[[0, 1, 3, 4]]
    sheet_k = (.14, .35)
    ambient = np.array([.7-.2j, -.3+.5j])
    heat = np.array([.2j, -.3+.4j])
    rows = []
    for mode in (0, 1, 2, 3):
        for omega in (1., 8., 24.):
            _, exact_storage, exact = analytic.bulk_reference(edges, omega, mode, ambient, heat, capacities, fractions,
                sheet_k, films=films, return_bulk_storage=True, return_contact_temperatures=True)
            exact_jump = exact[:, 0]-exact[:, 1]
            previous = None
            for n in (4, 8, 16, 32):
                mesh = np.r_[np.concatenate([np.geomspace(a,b,n+1)[:-1] for a,b in zip(edges, edges[1:])]), edges[-1]]
                sheet, incoming, outgoing, storage, traces = storing_rings.solve_storing_rings(
                    mesh, np.repeat(geometry.K,n), [n, 2*n], ambient, omega, azimuthal_mode=mode,
                    volumetric_heat_capacity=np.repeat(capacities,n), contact_resistances=geometry.CONTACT,
                    inner_resistance_fractions=fractions, sheet_capacities=geometry.CAPACITY, sheet_generation=heat,
                    sheet_conductivities=sheet_k, film_coefficients=films, return_bulk_storage=True, return_contact_temperatures=True)
                jump_ratio = (traces[:, 0]-traces[:, 1])/exact_jump
                errors = np.array([max(abs(traces/exact-1).ravel()), max(abs(jump_ratio-1))])
                order = np.log2(previous/errors) if previous is not None and (mode or any(capacities)) else None
                balance = max(abs(incoming-outgoing+heat -
                    (1j*omega*np.asarray(geometry.CAPACITY)+sheet_k*(mode/edges[1:-1])**2)*sheet))
                if mode == 0 and not any(capacities):
                    resolved = max(errors) < 1e-10
                else:
                    resolved = n < 32 or (max(errors) < .001 and min(order) > 1.8)
                passed = balance < 1e-10 and resolved
                storing = np.asarray(capacities) > 0
                storage_error = float(max(abs(storage[storing]/exact_storage[storing]-1), default=0.))
                passed &= np.all(storage[~storing] == 0) and (n < 32 or storage_error < .001)
                rows.append(dict(storage_relative_error=storage_error, mode=mode, omega=omega, cells_per_layer=n,
                    reference_jump_K=[[z.real, z.imag] for z in exact_jump],
                    jump_amplitude_ratio=abs(jump_ratio).tolist(), jump_phase_error_deg=np.angle(jump_ratio, deg=True).tolist(),
                    trace_relative_error=errors[0], jump_relative_error=errors[1],
                    orders=None if order is None else order.tolist(), sheet_balance_W_m2=balance, passed=bool(passed)))
                previous = errors
    return dict(film_coefficients_W_m2_K=films, inner_resistance_fractions=fractions,
                fixed_boundary_data_K=[[z.real,z.imag] for z in ambient],
                sheet_generation_W_m2=[[z.real,z.imag] for z in heat], runs=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacities", type=float, nargs=3, default=(.7, 2.3, 1.2))
    capacities = parser.parse_args().capacities
    cases = [measure(films, split, capacities) for films in ((7., None), (None, 3.), (7., 3.))
             for split in ((.2, .7), (0., 1.), (1., 0.))]
    passed = all(row["passed"] for case in cases for row in case["runs"])
    print(json.dumps(dict(passed=bool(passed), capacities_J_m3_K=capacities, cases=cases, phasors="exp(i omega t), real and imaginary parts",
        sources=[provenance(p) for p in (__file__, analytic.__file__, geometry.__file__, storing_rings.__file__, annulus.__file__)],
        reproduction="OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFVM/src python examples/fvm_storing_contact_phase.py --capacities "
        + " ".join(map(str, capacities))),
        allow_nan=False, indent=2))
    raise SystemExit(0 if passed else 1)
