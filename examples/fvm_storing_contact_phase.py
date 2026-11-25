"""Frequency response of material temperatures and jumps at storing annular contacts."""
import json

import numpy as np

import bem_harmonic_storing_rings as geometry
import fvm_harmonic_storing_rings as analytic
from conduction_provenance import provenance
from tensorfvm.cht import annulus, storing_rings


def measure(films, fractions):
    edges = np.array(geometry.RADII)[[0, 1, 3, 4]]
    capacities = (.7, 2.3, 1.2)
    sheet_k = (.14, .35)
    ambient = np.array([.7-.2j, -.3+.5j])
    heat = np.array([.2j, -.3+.4j])
    rows = []
    for mode in (0, 1, 2, 3):
        for omega in (1., 8., 24.):
            _, exact = analytic.bulk_reference(edges, omega, mode, ambient, heat, capacities, fractions,
                sheet_k, films=films, return_contact_temperatures=True)
            exact_jump = exact[:, 0]-exact[:, 1]
            previous = None
            for n in (4, 8, 16, 32):
                mesh = np.r_[np.concatenate([np.geomspace(a,b,n+1)[:-1] for a,b in zip(edges, edges[1:])]), edges[-1]]
                sheet, incoming, outgoing, traces = storing_rings.solve_storing_rings(
                    mesh, np.repeat(geometry.K,n), [n, 2*n], ambient, omega, azimuthal_mode=mode,
                    volumetric_heat_capacity=np.repeat(capacities,n), contact_resistances=geometry.CONTACT,
                    inner_resistance_fractions=fractions, sheet_capacities=geometry.CAPACITY, sheet_generation=heat,
                    sheet_conductivities=sheet_k, film_coefficients=films, return_contact_temperatures=True)
                jump_ratio = (traces[:, 0]-traces[:, 1])/exact_jump
                errors = np.array([max(abs(traces/exact-1).ravel()), max(abs(jump_ratio-1))])
                order = np.log2(previous/errors) if previous is not None else None
                balance = max(abs(incoming-outgoing+heat -
                    (1j*omega*np.asarray(geometry.CAPACITY)+sheet_k*(mode/edges[1:-1])**2)*sheet))
                passed = balance < 1e-10 and (n < 32 or (max(errors) < .001 and min(order) > 1.8))
                rows.append(dict(mode=mode, omega=omega, cells_per_layer=n,
                    reference_jump_K=[[z.real, z.imag] for z in exact_jump],
                    jump_amplitude_ratio=abs(jump_ratio).tolist(), jump_phase_error_deg=np.angle(jump_ratio, deg=True).tolist(),
                    trace_relative_error=errors[0], jump_relative_error=errors[1],
                    orders=None if order is None else order.tolist(), sheet_balance_W_m2=balance, passed=bool(passed)))
                previous = errors
    return dict(film_coefficients_W_m2_K=films, inner_resistance_fractions=fractions,
                fixed_boundary_data_K=[[z.real,z.imag] for z in ambient],
                sheet_generation_W_m2=[[z.real,z.imag] for z in heat], runs=rows)


if __name__ == "__main__":
    cases = [measure(films, split) for films in ((7., None), (None, 3.), (7., 3.))
             for split in ((.2, .7), (0., 1.), (1., 0.))]
    passed = all(row["passed"] for case in cases for row in case["runs"])
    print(json.dumps(dict(passed=passed, cases=cases, phasors="exp(i omega t), real and imaginary parts",
        sources=[provenance(p) for p in (__file__, analytic.__file__, geometry.__file__, storing_rings.__file__, annulus.__file__)],
        reproduction="OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFVM/src python examples/fvm_storing_contact_phase.py"),
        allow_nan=False, indent=2))
    raise SystemExit(0 if passed else 1)
