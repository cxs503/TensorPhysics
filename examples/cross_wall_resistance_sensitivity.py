"""Verify the logarithmic resistance sensitivity of transmitted harmonic heat flux."""
import json

import numpy as np

import cross_wall_reciprocity as reciprocal
from conduction_provenance import provenance


def study(solver, resistance, position=.25):
    _, target = reciprocal.wall_transfer(position, resistance, with_sensitivity=True)
    delta = 1e-4
    lower, upper = resistance*np.exp([-delta, delta])
    control = np.log(reciprocal.wall_transfer(position, upper)/reciprocal.wall_transfer(position, lower))/(2*delta)
    rows = []
    previous = None
    for n in (16, 32, 64, 128):
        samples = [reciprocal.slab.measure(n, solver, interface_position=position, contact_resistance=r)
                   for r in (lower, upper)]
        lo, hi = map(reciprocal.cold_phasor, samples)
        derivative = np.log(hi/lo)/(2*delta)
        errors = np.array([abs(derivative.real-target.real), abs(derivative.imag-target.imag)])
        orders = None if previous is None else np.log2(previous/errors)
        passed = max(abs(control-target)) < 1e-8 and np.all(derivative.real < 0)
        if n == 128:
            passed &= errors.max() < .001 and orders.min() > 1.8
        rows.append(dict(solver=solver, columns=n, steps=4*n, resistance_m2_K_W=resistance,
                         log_amplitude_derivative=derivative.real.tolist(), phase_derivative_rad=derivative.imag.tolist(),
                         exact_log_amplitude_derivative=target.real.tolist(), exact_phase_derivative_rad=target.imag.tolist(),
                         amplitude_phase_errors=errors.tolist(), amplitude_phase_orders=None if orders is None else orders.tolist(),
                         finite_difference_control=float(max(abs(control-target))), passed=bool(passed)))
        previous = errors
    return rows


if __name__ == "__main__":
    rows = [row for solver in ("fvm", "sph") for resistance in (.1, .6) for row in study(solver, resistance)]
    passed = all(row["passed"] for row in rows)
    slab = reciprocal.slab
    print(json.dumps(dict(passed=passed, runs=rows, frequencies_Hz=[1, 2], interface_position_m=.25,
        differentiation_variable="natural logarithm of contact resistance", log_resistance_half_step=1e-4,
        sources=[provenance(p) for p in (__file__, reciprocal.__file__, slab.__file__, slab.periodic.__file__,
            slab.continuum.__file__, slab.periodic.rod.__file__, slab.periodic.sph.__file__)],
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFVM/src:../TensorSPH/src python examples/cross_wall_resistance_sensitivity.py"),
        indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
