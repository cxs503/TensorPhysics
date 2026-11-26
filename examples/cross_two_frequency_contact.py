"""Resolve the first two interface heat-flux harmonics from graded-step heat integrals."""
import json

import numpy as np

import cross_periodic_contact_reversals as periodic
from conduction_provenance import provenance


def study(solver, method):
    frequencies = 2*np.pi*np.array([1., 2.])
    rows = []
    previous = None
    for steps in (32, 64, 128, 256):
        s = np.linspace(0., 1., steps+1)
        times = s+.4*s*(1-s)
        heat, resistance = periodic.solve_contact_heat(times, method, solver, second_harmonic=True)
        exact = np.array([1+.15j, .18-.25j])/resistance
        measured = periodic.fit_flux_phasors(times, heat, frequencies)
        primitive = .2*times/resistance + np.real(
            np.expm1(1j*times[:, None]*frequencies) @ (exact/(1j*frequencies)))
        control = periodic.fit_flux_phasors(times, np.diff(primitive), frequencies)
        amplitude_errors = abs(abs(measured)-abs(exact))
        phase_errors = abs(np.angle(measured/exact))
        errors = np.stack((amplitude_errors, phase_errors))
        orders = None if previous is None else np.log2(previous/errors)
        control_error = float(max(abs(control-exact)))
        passed = control_error < 1e-12
        if steps == 256:
            passed &= np.min(orders) > 1.8 and max(amplitude_errors) < 5e-4 and max(phase_errors) < 1e-3
        rows.append(dict(steps=steps, solver=solver, method=method, amplitudes_W_m2=abs(measured).tolist(),
                         phases_rad=np.angle(measured).tolist(), amplitude_errors_W_m2=amplitude_errors.tolist(),
                         phase_errors_rad=phase_errors.tolist(), amplitude_phase_orders=None if orders is None else orders.tolist(),
                         exact_fit_error_W_m2=control_error, passed=bool(passed)))
        previous = errors
    return rows


if __name__ == "__main__":
    rows = [r for solver in ("fvm", "sph") for method in ("theta", "sdirk2") for r in study(solver, method)]
    passed = all(r["passed"] for r in rows)
    print(json.dumps(dict(passed=passed, runs=rows, frequencies_Hz=[1, 2],
        sources=[provenance(p) for p in (__file__, periodic.__file__, periodic.case.__file__,
                                         periodic.rod.__file__, periodic.sph.__file__)],
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFVM/src:../TensorSPH/src python examples/cross_two_frequency_contact.py"),
        indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
