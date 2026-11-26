"""Check contact-resistance uncertainty against closed-form multiplicative phasor noise moments."""
import argparse
import json

import numpy as np

import cross_wall_resistance_inversion as inv
from conduction_provenance import provenance


def study(solver, target, position, noise, *, unequal=False):
    ref = inv.reciprocal
    run = ref.slab.measure(128, solver, contact_resistance=target, interface_position=position)
    gain = ref.cold_phasor(run)
    _, b = ref.impedance_terms(position)
    coeff = 1/(b*gain)
    weights = abs(b)**2/sum(abs(b)**2)
    baseline = inv.estimates(gain, position)
    # If the measured phasor is G exp(A+iP), the single-tone estimate
    # shifts by Re(C*(exp(-A-iP)-1)), C=1/(B*G).
    amplitude_sd = np.array([.005, .015 if unequal else .005])
    phase_sd = np.array([.01, .02 if unequal else .01])
    mean_exponent = (amplitude_sd**2-phase_sd**2)/2
    mean_factors = np.exp(mean_exponent)
    rows = []
    for shared in (False, True):
        draws = noise[:, :1] if shared else noise
        measured = gain*np.exp(amplitude_sd*draws[..., 0]+1j*phase_sd*draws[..., 1])
        fitted = inv.estimates(measured, position)
        correlation = np.ones((2, 2)) if shared else np.eye(2)
        cov_a = correlation*np.outer(amplitude_sd, amplitude_sd)
        cov_p = correlation*np.outer(phase_sd, phase_sd)
        # Gaussian exponential moments for Re and Im of exp(-A-iP).
        factors = np.outer(mean_factors, mean_factors)
        cov_real = factors*(np.expm1(cov_a)*np.cosh(cov_p) + 2*np.sinh(cov_p/2)**2)
        cov_imag = factors*np.exp(cov_a)*np.sinh(cov_p)
        covariance = cov_real*np.outer(coeff.real, coeff.real) + cov_imag*np.outer(coeff.imag, coeff.imag)
        expected_sd = np.sqrt(np.r_[covariance.diagonal(), weights @ covariance @ weights])
        shifts = coeff.real*np.expm1(mean_exponent)
        expected_mean = baseline + np.r_[shifts, weights @ shifts]
        actual_sd = fitted.std(axis=0, ddof=1)
        mean_z = abs(fitted.mean(axis=0)-expected_mean)/(expected_sd/np.sqrt(len(noise)))
        sd_error = abs(actual_sd/expected_sd-1)
        # Minimum variance among real linear combinations whose weights sum to one.
        precision_sum = np.linalg.solve(covariance, np.ones(2))
        optimal = precision_sum/precision_sum.sum()
        combined = fitted[:, :2] @ optimal
        predicted = np.sqrt(optimal @ covariance @ optimal)
        observed = combined.std(ddof=1)
        optimal_sd_error = abs(observed/predicted-1)
        expected_weighted_mean = optimal @ expected_mean[:2]
        optimal_mean_z = abs(combined.mean()-expected_weighted_mean)/(predicted/np.sqrt(len(noise)))
        rows.append(dict(optimal_weights=optimal.tolist(), optimal_sd_m2_K_W=float(observed),
                         optimal_expected_sd_m2_K_W=float(predicted), optimal_mean_m2_K_W=float(combined.mean()),
                         optimal_expected_mean_m2_K_W=float(expected_weighted_mean),
                         optimal_relative_sd_error=float(optimal_sd_error), optimal_mean_z=float(optimal_mean_z),
                         solver=solver, target_m2_K_W=target, shared_frequency_noise=shared,
                         noiseless_estimates_m2_K_W=baseline.tolist(), mean_m2_K_W=fitted.mean(axis=0).tolist(),
                         expected_mean_m2_K_W=expected_mean.tolist(), sd_m2_K_W=actual_sd.tolist(),
                         expected_sd_m2_K_W=expected_sd.tolist(), negative_fraction=(fitted < 0).mean(axis=0).tolist(),
                         max_mean_error_in_standard_errors=float(max(mean_z)), max_relative_sd_error=float(max(sd_error)),
                         passed=bool(max(mean_z)<5 and optimal_mean_z<5 and max(sd_error)<.015 and optimal_sd_error<.015
                                     and predicted <= min(expected_sd)*(1+1e-12))))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unequal", action="store_true", help="use larger noise at 2 Hz")
    unequal = parser.parse_args().unequal
    seed, count = 261126, 100000
    noise = np.random.default_rng(seed).standard_normal((count, 2, 2))
    rows = [r for solver in ("fvm", "sph") for target in (0., .1, .6) for r in study(solver, target, .25, noise, unequal=unequal)]
    passed = all(r["passed"] for r in rows)
    ref = inv.reciprocal
    slab = ref.slab
    print(json.dumps(dict(passed=passed, runs=rows, draws=count, seed=seed, log_amplitude_sd=[.005, .015 if unequal else .005], phase_sd_rad=[.01, .02 if unequal else .01],
        columns=["1 Hz", "2 Hz", "joint"], sources=[provenance(p) for p in (__file__, inv.__file__, ref.__file__,
        slab.__file__, slab.periodic.__file__, slab.continuum.__file__, slab.periodic.rod.__file__, slab.periodic.sph.__file__)],
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFVM/src:../TensorSPH/src python examples/cross_wall_inversion_noise.py" + (" --unequal" if unequal else "")),
        indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
