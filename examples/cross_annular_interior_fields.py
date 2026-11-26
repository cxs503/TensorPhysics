"""Compare circular BEM and radial FVM directly at shell centroids and internal faces."""
import json

import numpy as np

import cross_volume_heated_rings as case
from conduction_provenance import provenance


def max_error(first, second):
    return float(np.max(np.abs(first-second)))


def measure(mode):
    bem = case.harmonic_rings.assemble_ring_response(
        case.EDGES, case.K, case.CONTACT, 32, mode=mode,
        inner_resistance_fractions=case.FRACTIONS, include_volume_sources=True)
    rows = []
    previous = {}
    for n in (4, 8, 16, 32, 64):
        segments = [np.geomspace(a, b, n+1) for a, b in zip(case.EDGES, case.EDGES[1:])]
        mesh = np.r_[np.concatenate([r[:-1] for r in segments]), case.EDGES[-1]]
        lo, hi = mesh[:-1], mesh[1:]
        centers = (2/3)*(hi**2+lo*hi+lo**2)/(hi+lo)
        faces = np.concatenate([r[1:-1] for r in segments])
        samples = np.r_[centers, faces]
        density = (np.repeat(case.VOLUME, n)*2*np.diff(mesh**(mode+2)) /
                   ((mode+2)*case.EDGES[-1]**mode*np.diff(mesh**2)))
        for omega in (1., 8., 24.):
            boundary = bem.solve(omega, case.WALLS, case.CAPACITY, sheet_generation=case.SHEETS,
                                 volume_generation=case.VOLUME, sample_radii=samples)
            _, _, _, fv_t, layer_rates = case.storing_rings.solve_storing_rings(
                mesh, np.repeat(case.K, n), [n, 2*n], case.WALLS, omega, contact_resistances=case.CONTACT,
                sheet_capacities=case.CAPACITY, sheet_generation=case.SHEETS, source=density,
                inner_resistance_fractions=case.FRACTIONS, azimuthal_mode=mode,
                return_cell_temperatures=True, return_layer_face_rates=True)
            fv_q = np.concatenate([rate[1:-1] for rate in layer_rates])/(2*np.pi*faces)
            bem_t = boundary.sample_temperature[:len(centers)]
            bem_q = boundary.sample_radial_flux[len(centers):]
            _, ref = case.continuum(mode, omega, sample_radii=samples)
            exact_t, exact_q = ref[:len(centers), 0], ref[len(centers):, 1]
            bem_errors = [max_error(bem_t, exact_t), max_error(bem_q, exact_q)]
            fv_errors = np.array([max_error(fv_t, exact_t), max_error(fv_q, exact_q)])
            differences = [max_error(bem_t, fv_t), max_error(bem_q, fv_q)]
            orders = np.log2(previous[omega]/fv_errors) if mode and omega in previous else None
            previous[omega] = fv_errors
            passed = max(bem_errors) < 1e-10
            if mode == 0:
                passed &= max(fv_errors) < 1e-9
            elif n == 64:
                passed &= min(orders) > 1.8 and max(fv_errors) < .001 and max(differences) < .001
            rows.append(dict(mode=mode, omega_rad_s=omega, cells_per_layer=n,
                             temperature_samples=len(centers), flux_samples=len(faces),
                             bem_errors=bem_errors, fvm_errors=fv_errors.tolist(), solver_difference=differences,
                             fvm_orders=None if orders is None else orders.tolist(), passed=bool(passed)))
    return rows


if __name__ == "__main__":
    rows = [row for mode in (0, 1, 2) for row in measure(mode)]
    passed = all(row["passed"] for row in rows)
    print(json.dumps(dict(passed=passed, runs=rows, error_columns=["temperature_K", "radial_flux_W_m2"],
                          sources=[provenance(p) for p in (__file__, case.__file__, case.harmonic_rings.__file__,
                                   case.layered_annulus.__file__, case.storing_rings.__file__, case.annulus.__file__)],
                          reproduction="OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorBEM/src:../TensorFVM/src python examples/cross_annular_interior_fields.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
