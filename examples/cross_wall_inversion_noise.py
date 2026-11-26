"""Check contact-resistance uncertainty against closed-form multiplicative phasor noise moments."""
import json

import numpy as np

import cross_wall_resistance_inversion as inv
from conduction_provenance import provenance


def study(solver, target, position, noise):
    ref = inv.reciprocal
    run = ref.slab.measure(128, solver, contact_resistance=target, interface_position=position)
    gain = ref.cold_phasor(run)
    _, b = ref.impedance_terms(position)
    coeff = 1/(b*gain)
    weights = abs(b)**2/sum(abs(b)**2)
    baseline = inv.estimates(gain, position)
    # If the measured phasor is G exp(A+iP), the single-tone estimate
    # shifts by Re(C*(exp(-A-iP)-1)), C=1/(B*G).
    amplitude_sd, phase_sd = .005, .01
    a2, p2 = amplitude_sd**2, phase_sd**2
    mean_factor = np.exp((a2-p2)/2)
    var_real = (np.exp(2*a2)*(1+np.exp(-2*p2))/2 - mean_factor**2)
    var_imag = np.exp(2*a2)*(-np.expm1(-2*p2))/2
    rows = []
    for shared in (False, True):
        draws = noise[:, :1] if shared else noise
        measured = gain*np.exp(amplitude_sd*draws[..., 0]+1j*phase_sd*draws[..., 1])
        fitted = inv.estimates(measured, position)
        covariance = var_real*np.outer(coeff.real, coeff.real) + var_imag*np.outer(coeff.imag, coeff.imag)
        if not shared:
            covariance = np.diag(np.diag(covariance))
        expected_sd = np.sqrt(np.r_[covariance.diagonal(), weights @ covariance @ weights])
        shifts = coeff.real*np.expm1((a2-p2)/2)
        expected_mean = baseline + np.r_[shifts, weights @ shifts]
        actual_sd = fitted.std(axis=0, ddof=1)
        mean_z = abs(fitted.mean(axis=0)-expected_mean)/(expected_sd/np.sqrt(len(noise)))
        sd_error = abs(actual_sd/expected_sd-1)
        rows.append(dict(solver=solver, target_m2_K_W=target, shared_frequency_noise=shared,
                         noiseless_estimates_m2_K_W=baseline.tolist(), mean_m2_K_W=fitted.mean(axis=0).tolist(),
                         expected_mean_m2_K_W=expected_mean.tolist(), sd_m2_K_W=actual_sd.tolist(),
                         expected_sd_m2_K_W=expected_sd.tolist(), negative_fraction=(fitted < 0).mean(axis=0).tolist(),
                         max_mean_error_in_standard_errors=float(max(mean_z)), max_relative_sd_error=float(max(sd_error)),
                         passed=bool(max(mean_z)<5 and max(sd_error)<.015)))
    return rows


if __name__ == "__main__":
    seed, count = 261126, 100000
    noise = np.random.default_rng(seed).standard_normal((count, 2, 2))
    rows = [r for solver in ("fvm", "sph") for target in (0., .1, .6) for r in study(solver, target, .25, noise)]
    passed = all(r["passed"] for r in rows)
    ref = inv.reciprocal
    slab = ref.slab
    print(json.dumps(dict(passed=passed, runs=rows, draws=count, seed=seed, log_amplitude_sd=.005, phase_sd_rad=.01,
        columns=["1 Hz", "2 Hz", "joint"], sources=[provenance(p) for p in (__file__, inv.__file__, ref.__file__,
        slab.__file__, slab.periodic.__file__, slab.continuum.__file__, slab.periodic.rod.__file__, slab.periodic.sph.__file__)],
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFVM/src:../TensorSPH/src python examples/cross_wall_inversion_noise.py"),
        indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
