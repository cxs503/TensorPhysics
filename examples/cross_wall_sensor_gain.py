"""Infer contact resistance, a positive sensor gain and timing delay from two flux tones."""
import json

import numpy as np

import cross_wall_clock_offset as clock
from conduction_provenance import provenance


def recover(measured, position):
    ref = clock.inversion.reciprocal
    z, b = ref.impedance_terms(position)
    # Eliminating sensor gain and delay gives |Z1|² - |G2/G1|² |Z2|² = 0.
    ratio = abs(measured[1]/measured[0])**2
    terms = np.array([abs(b)**2, 2*(z*b.conj()).real, abs(z)**2])
    roots = np.roots(terms[:, 0]-ratio*terms[:, 1])
    positive = [float(r.real) for r in roots if np.isreal(r) and r.real >= 0]
    if len(positive) != 1:
        raise ValueError("two magnitudes do not identify a unique nonnegative contact resistance")
    resistance = positive[0]
    residual = measured/ref.wall_transfer(position, resistance)
    return resistance, float(np.mean(abs(residual))), clock.shared_delay(residual)


def study(solver, target, position=.25):
    ref = clock.inversion.reciprocal
    rows = []
    for n in (32, 64, 128):
        run = ref.slab.measure(n, solver, interface_position=position, contact_resistance=target)
        signal, exact = (ref.cold_phasor(run, prefix) for prefix in ("", "exact_"))
        for scale in (.85, 1.15):
            for delay in (-.38, .012):
                multiplier = scale*np.exp(-2j*np.pi*np.array([1., 2.])*delay)
                truth = np.array([target, scale, delay])
                estimates = recover(multiplier*signal, position)
                errors = abs(estimates-truth)
                control = abs(recover(multiplier*exact, position)-truth)
                passed = max(control)<1e-11
                if n == 128:
                    passed &= errors[0]<.002 and errors[1]<.002 and errors[2]<1e-4
                rows.append(dict(solver=solver, columns=n, steps=4*n, resistance_m2_K_W=target,
                                 sensor_gain=scale, delay_s=delay, estimates=estimates,
                                 absolute_errors=errors.tolist(), continuum_control_errors=control.tolist(),
                                 passed=bool(passed)))
    return rows


if __name__ == "__main__":
    rows = [r for solver in ("fvm", "sph") for target in (.1, .6) for r in study(solver, target)]
    passed = all(r["passed"] for r in rows)
    ref = clock.inversion.reciprocal
    slab = ref.slab
    print(json.dumps(dict(passed=passed, runs=rows, interface_position_m=.25, frequencies_Hz=[1, 2],
        estimate_columns=["resistance_m2_K_W", "sensor_gain", "delay_s"],
        sources=[provenance(p) for p in (__file__, clock.__file__, clock.inversion.__file__, ref.__file__,
            slab.__file__, slab.periodic.__file__, slab.continuum.__file__, slab.periodic.rod.__file__, slab.periodic.sph.__file__)],
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFVM/src:../TensorSPH/src python examples/cross_wall_sensor_gain.py"),
        indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
