"""Check the derivative of both end temperatures with respect to total contact resistance."""
import argparse
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


def finite_changes(n, omega, solver):
    reference_r = .32
    base = impedance.measure(n, omega, solver, resistance=reference_r, contact_response=True)
    z = phasors(base["impedance_K_W"])
    currents = phasors(base["contact_currents"])
    traces = phasors(base["contact_traces_K"])
    # On the undriven half the leftward power is -q, the rightward power +q.
    # The two half-rod impedances and R form a series path across the contact.
    series = reference_r-traces[1, 0]/currents[1]+traces[0, 1]/currents[0]
    for resistance in (.032, .16, .64, 3.2):
        delta = resistance-reference_r
        factor = 1+delta/series
        predicted = z+(delta/factor)*np.outer(currents, currents)
        predicted_current = currents/factor
        direct = impedance.measure(n, omega, solver, resistance=resistance, contact_response=True)
        direct_current = phasors(direct["contact_currents"])
        yield dict(solver=solver, cells_per_material=n, omega_rad_s=omega, resistance_K_W=resistance,
                   impedance_error_K_W=float(abs(predicted-phasors(direct["exact_impedance_K_W"])).max()),
                   contact_power_error_W=float(abs(predicted_current-phasors(direct["exact_contact_power_W"])).max()),
                   direct_impedance_difference_K_W=float(abs(predicted-phasors(direct["impedance_K_W"])).max()),
                   direct_contact_current_difference_W=float(abs(predicted_current-direct_current).max()),
                   predicted_impedance_K_W=np.stack((predicted.real, predicted.imag), axis=-1).tolist(),
                   predicted_contact_power_W=np.stack((predicted_current.real, predicted_current.imag), axis=-1).tolist())


def verify_finite_changes():
    rows = []
    for solver in ("FVM", "FEM"):
        for omega in (.5, 4., 12.):
            previous = {}
            for n in (4, 8, 16, 32, 64):
                for row in finite_changes(n, omega, solver):
                    r = row["resistance_K_W"]
                    errors = np.array([row["impedance_error_K_W"], row["contact_power_error_W"]])
                    orders = np.log2(previous[r]/errors).tolist() if r in previous else []
                    previous[r] = errors
                    row.update(orders=orders, passed=bool(all(order > 1.8 for order in orders)
                               and (n < 64 or max(errors) < 2e-5)
                               and row["direct_impedance_difference_K_W"] < 1e-10
                               and row["direct_contact_current_difference_W"] < 1e-10))
                    rows.append(row)
    print(json.dumps(dict(baseline_resistance_K_W=.32, order_metrics=["impedance_error_K_W", "contact_power_error_W"],
                          baseline_drives="Unit inward power at each end in turn",
                          reference="Fresh continuum layer transfer matrix and independent FEM/FVM solves at each target resistance",
                          reproduction="OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFVM/src:../TensorFEM/src python examples/harmonic_contact_sensitivity.py --finite-changes",
                          sources=[provenance(f) for f in (__file__, impedance.__file__, rod.__file__, thermal.__file__)],
                          runs=rows), indent=2, allow_nan=False))
    return all(row["passed"] for row in rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finite-changes", action="store_true", help="verify finite rank-one updates of contact resistance")
    if parser.parse_args().finite_changes:
        raise SystemExit(0 if verify_finite_changes() else 1)
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
