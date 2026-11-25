"""Check two directly heated FVM rings against radial and angular continuum solutions."""
import json

import numpy as np

import bem_harmonic_storing_rings as reference
from conduction_provenance import provenance
from tensorfvm.cht import annulus, storing_rings


def measure():
    edges = np.array(reference.RADII)[[0, 1, 3, 4]]
    walls = np.zeros(2)
    generation = np.array([1., .6j])
    rows = []
    for mode in range(4):
        for omega in (1., 8., 24.):
            expected = (reference.angular_reference(omega, mode, walls, generation) if mode
                        else reference.continuum(omega, walls, generation))
            previous = None
            for n in (4, 8, 16, 32):
                mesh = np.r_[np.concatenate([np.geomspace(a, b, n+1)[:-1]
                                             for a, b in zip(edges, edges[1:])]), edges[-1]]
                fields = storing_rings.solve_storing_rings(
                    mesh, np.repeat(reference.K, n), [n, 2*n], walls, omega,
                    contact_resistances=reference.CONTACT, sheet_capacities=reference.CAPACITY,
                    sheet_generation=generation, azimuthal_mode=mode)
                temp, incoming, outgoing = fields
                balance = max(abs(incoming-outgoing+generation-1j*omega*np.asarray(reference.CAPACITY)*temp))
                error = float(abs(np.column_stack(fields)/expected-1).max())
                order = np.log2(previous/error) if previous is not None and mode else None
                passed = balance < 1e-10 and (error < 1e-10 if mode == 0 else
                                               n < 32 or (error < .001 and order > 1.8))
                rows.append(dict(mode=mode, omega=omega, cells_per_layer=n, max_relative_error=error,
                                 order=order, sheet_balance_W_m2=float(balance), passed=bool(passed)))
                previous = error
    return rows


if __name__ == "__main__":
    runs = measure()
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(p) for p in
                          (__file__, reference.__file__, annulus.__file__, storing_rings.__file__)],
                          reproduction="OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFVM/src python examples/fvm_harmonic_storing_rings.py"),
                     allow_nan=False, indent=2))
    raise SystemExit(0 if passed else 1)
