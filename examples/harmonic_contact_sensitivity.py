"""Check the derivative of both end temperatures with respect to total contact resistance."""
import json

import numpy as np

import harmonic_contact_impedance as impedance
from conduction_provenance import provenance
from tensorfem import thermal
from tensorfvm.cht import rod


def phasors(pairs):
    values = np.asarray(pairs)
    return values[..., 0]+1j*values[..., 1]


def measure(n, omega, solver):
    resistance = .32
    base = impedance.measure(n, omega, solver, resistance=resistance, sensitivity=True)
    derivative = phasors(base["derivative"])
    exact = phasors(base["exact_derivative"])
    differences = []
    step = .005  # K/W
    for h in (step, step/2):
        matrices = [phasors(impedance.measure(n, omega, solver, resistance=resistance+offset)["impedance_K_W"])
                    for offset in (-h, h)]
        differences.append((matrices[1]-matrices[0])/(2*h))
    extrapolated = (4*differences[1]-differences[0])/3
    return dict(solver=solver, cells_per_material=n, omega_rad_s=omega,
                derivative=base["derivative"], exact_derivative=base["exact_derivative"],
                derivative_error=float(abs(derivative-exact).max()),
                magnitude_error=float(abs(abs(derivative)-abs(exact)).max()),
                phase_error_rad=float(abs(np.angle(derivative*exact.conj())).max()),
                perturbation_discrepancy=float(abs(extrapolated-derivative).max()))


if __name__ == "__main__":
    rows = []
    for solver in ("FVM", "FEM"):
        for omega in (.5, 4., 12.):
            previous = None
            for n in (4, 8, 16, 32, 64):
                row = measure(n, omega, solver)
                errors = np.array([row[k] for k in ("derivative_error", "magnitude_error", "phase_error_rad")])
                orders = np.log2(previous/errors).tolist() if previous is not None else []
                row.update(orders=orders, passed=bool(all(p > 1.8 for p in orders)
                           and row["perturbation_discrepancy"] < 1e-8
                           and (n < 64 or (errors[0] < 2e-5 and errors[2] < 2e-4))))
                rows.append(row)
                previous = errors
    print(json.dumps(dict(reference="Analytic derivative of the continuum transfer matrix at R=0.32 K/W",
                          discretization_check="Complex-symmetric contact current product versus Richardson-extrapolated centered port differences with steps 0.005 and 0.0025 K/W",
                          reproduction="OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFVM/src:../TensorFEM/src python examples/harmonic_contact_sensitivity.py",
                          sources=[provenance(f) for f in (__file__, impedance.__file__, rod.__file__, thermal.__file__)],
                          runs=rows), indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
