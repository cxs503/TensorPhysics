"""Recover a real contact resistance from simulated complex cold-wall heat flux."""
import json

import numpy as np

import cross_wall_reciprocity as reciprocal
from conduction_provenance import provenance


def estimates(gain, position):
    # Fit 1/G - Z_bulk = B*R using a real R. Keep estimates unconstrained
    # so that a negative numerical bias is visible in the perfect-contact case.
    bulk, weight = reciprocal.impedance_terms(position)
    projection = np.real(weight.conj()*(1/gain-bulk))
    norm = abs(weight)**2
    joint = projection.sum(axis=-1, keepdims=True)/norm.sum()
    return np.concatenate((projection/norm, joint), axis=-1)


def study(solver, target, position=.25):
    rows = []
    previous = None
    for n in (16, 32, 64, 128):
        run = reciprocal.slab.measure(n, solver, interface_position=position, contact_resistance=target)
        fitted = estimates(reciprocal.cold_phasor(run), position)
        control = estimates(reciprocal.cold_phasor(run, "exact_"), position)
        errors = abs(fitted-target)
        orders = None if previous is None else np.log2(previous/errors)
        passed = max(abs(control-target)) < 1e-12 and np.isfinite(errors).all()
        if n == 128:
            passed &= max(errors) < 5e-4 and min(orders) > 1.8
        rows.append(dict(solver=solver, columns=n, steps=4*n, target_m2_K_W=target,
                         estimates_m2_K_W=fitted.tolist(), absolute_errors_m2_K_W=errors.tolist(),
                         continuum_control_max_error_m2_K_W=float(max(abs(control-target))),
                         orders=None if orders is None else orders.tolist(), passed=bool(passed)))
        previous = errors
    return rows


if __name__ == "__main__":
    runs = [r for solver in ("fvm", "sph") for target in (0., .1, .6) for r in study(solver, target)]
    passed = all(r["passed"] for r in runs)
    slab = reciprocal.slab
    print(json.dumps(dict(passed=passed, runs=runs, estimate_columns=["1 Hz", "2 Hz", "joint"],
        interface_position_m=.25, sources=[provenance(p) for p in (__file__, reciprocal.__file__, slab.__file__,
            slab.periodic.__file__, slab.continuum.__file__, slab.periodic.rod.__file__, slab.periodic.sph.__file__)],
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFVM/src:../TensorSPH/src python examples/cross_wall_resistance_inversion.py"),
        indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
