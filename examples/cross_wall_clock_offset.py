"""Recover resistance and a shared timing delay from two cold-wall flux phasors."""
import json

import numpy as np

import cross_wall_resistance_inversion as inversion
from conduction_provenance import provenance


def recover(measured, position):
    # Delay leaves |G| unchanged. Solve |Z_bulk + B*R|² = |1/G|²
    # on the branch through R=0 where the magnitude of impedance increases.
    ref = inversion.reciprocal
    bulk, weight = ref.impedance_terms(position)
    a = abs(weight)**2
    b = np.real(bulk*weight.conj())
    c = abs(bulk)**2-abs(1/measured)**2
    per_tone = -c/(b+np.sqrt(b*b-a*c))
    resistance = per_tone.mean()
    ratios = measured/ref.wall_transfer(position, resistance)
    return float(resistance), shared_delay(ratios)


def shared_delay(ratios):
    omega = 2*np.pi*np.array([1., 2.])
    delays = -np.angle(ratios)/omega
    # The 1 Hz phase fixes the common delay modulo one second. Choose
    # the 2 Hz branch nearest that value before fitting the phase slope.
    delays[1] += .5*np.rint((delays[0]-delays[1])/.5)
    delay = np.average(delays, weights=omega**2)
    return float(delay)


def study(solver, target, position=.25):
    ref = inversion.reciprocal
    rows = []
    for n in (32, 64, 128):
        run = ref.slab.measure(n, solver, interface_position=position, contact_resistance=target)
        gains = ref.cold_phasor(run)
        exact = ref.cold_phasor(run, "exact_")
        for delay in (-.38, -.012, 0., .012, .38):
            shift = np.exp(-2j*np.pi*np.array([1., 2.])*delay)
            r, t = recover(gains*shift, position)
            control_r, control_t = recover(exact*shift, position)
            uncorrected = inversion.estimates(gains*shift, position)[-1]
            passed = max(abs(control_r-target), abs(control_t-delay)) < 1e-12
            if n == 128:
                passed &= abs(r-target) < 1e-4 and abs(t-delay) < 2e-5
            rows.append(dict(solver=solver, columns=n, steps=4*n, target_m2_K_W=target,
                             delay_s=delay, inferred_delay_s=t, inferred_resistance_m2_K_W=r,
                             uncorrected_resistance_m2_K_W=uncorrected,
                             resistance_error_m2_K_W=abs(r-target), delay_error_s=abs(t-delay),
                             control_resistance_error_m2_K_W=abs(control_r-target),
                             control_delay_error_s=abs(control_t-delay), passed=bool(passed)))
    return rows


if __name__ == "__main__":
    runs = [row for solver in ("fvm", "sph") for r in (.1, .6) for row in study(solver, r)]
    passed = all(row["passed"] for row in runs)
    ref = inversion.reciprocal
    slab = ref.slab
    print(json.dumps(dict(passed=passed, runs=runs, frequencies_Hz=[1, 2], interface_position_m=.25,
        delay_convention="measured gain = gain * exp(-i omega delay), delay in (-0.5,0.5) s",
        sources=[provenance(p) for p in (__file__, inversion.__file__, ref.__file__, slab.__file__,
        slab.periodic.__file__, slab.continuum.__file__, slab.periodic.rod.__file__, slab.periodic.sph.__file__)],
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFVM/src:../TensorSPH/src python examples/cross_wall_clock_offset.py"),
        indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
