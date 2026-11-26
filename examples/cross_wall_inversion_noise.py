"""Check contact-resistance uncertainty against closed-form multiplicative phasor noise moments."""
import argparse
import json

import numpy as np

import cross_wall_resistance_inversion as inv
from conduction_provenance import provenance


def calibration_study(calibration, validation, covariance, optimal_sd, target):
    """Estimate weights on separate calibration batches, then score on held-out readings."""
    # Only the first two columns are the individual frequency estimates.
    held_out = validation[:, :2]
    holdout_cov = np.cov(held_out.T)
    holdout_mean = held_out.mean(axis=0)
    results = []
    for count in (8, 32, 128):
        batch = calibration[:, :count, :2]
        centered = batch - batch.mean(axis=1, keepdims=True)
        # Var(y + w*(x-y)) is minimized at -Cov(y,x-y)/Var(x-y).
        difference = centered[:, :, 0] - centered[:, :, 1]
        w = -np.sum(centered[:, :, 1]*difference, axis=1)/np.sum(difference**2, axis=1)
        weights = np.column_stack((w, 1-w))
        def sd(cov):
            return np.sqrt(np.sum((weights @ cov)*weights, axis=1))
        observed, predicted = sd(holdout_cov), sd(covariance)
        bias = weights @ holdout_mean-target
        rmse = np.sqrt(observed**2*(len(held_out)-1)/len(held_out)+bias**2)
        ratios = observed/optimal_sd
        agreement = float(np.max(abs(observed/predicted-1)))
        results.append(dict(calibration_readings=count, batches=len(batch),
                            median_first_frequency_weight=float(np.median(w)),
                            median_heldout_sd_m2_K_W=float(np.median(observed)),
                            median_heldout_rmse_m2_K_W=float(np.median(rmse)),
                            median_sd_over_oracle=float(np.median(ratios)),
                            p95_sd_over_oracle=float(np.quantile(ratios, .95)),
                            max_relative_sd_prediction_error=agreement,
                            passed=bool(agreement < .02 and (count < 128 or np.quantile(ratios, .95) < 1.05))))
    return results


def study(solver, target, position, noise, *, unequal=False, calibration_noise=None):
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
        if calibration_noise is not None:
            draws = calibration_noise[:, :, :1] if shared else calibration_noise
            calibration_gain = gain*np.exp(amplitude_sd*draws[..., 0]+1j*phase_sd*draws[..., 1])
            trials = calibration_study(inv.estimates(calibration_gain, position), fitted,
                                       covariance, predicted, target)
            rows[-1]["calibration_trials"] = trials
            rows[-1]["passed"] &= all(t["passed"] for t in trials)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unequal", action="store_true", help="use larger noise at 2 Hz")
    parser.add_argument("--calibration", action="store_true", help="fit weights on independent calibration batches")
    args = parser.parse_args()
    unequal = args.unequal
    seed, count = 261126, 100000
    noise = np.random.default_rng(seed).standard_normal((count, 2, 2))
    calibration_seed = 261127
    calibration_noise = (np.random.default_rng(calibration_seed).standard_normal((256, 128, 2, 2))
                         if args.calibration else None)
    rows = [r for solver in ("fvm", "sph") for target in (0., .1, .6) for r in study(solver, target, .25, noise, unequal=unequal, calibration_noise=calibration_noise)]
    passed = all(r["passed"] for r in rows)
    ref = inv.reciprocal
    slab = ref.slab
    print(json.dumps(dict(passed=passed, runs=rows, draws=count, seed=seed, log_amplitude_sd=[.005, .015 if unequal else .005], phase_sd_rad=[.01, .02 if unequal else .01],
        calibration_seed=calibration_seed if args.calibration else None, columns=["1 Hz", "2 Hz", "joint"], sources=[provenance(p) for p in (__file__, inv.__file__, ref.__file__,
        slab.__file__, slab.periodic.__file__, slab.continuum.__file__, slab.periodic.rod.__file__, slab.periodic.sph.__file__)],
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFVM/src:../TensorSPH/src python examples/cross_wall_inversion_noise.py" + (" --unequal" if unequal else "") + (" --calibration" if args.calibration else "")),
        indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
