"""Compare BGK and directionally tuned TRT with identical slab pulse, grid, and step size."""
import json
import math

import lbm_two_mode_pulse as slab
from conduction_provenance import provenance


if __name__ == "__main__":
    odd = .5 + math.sqrt(3)/6
    even = .5 + math.sqrt(3)/3
    observables = ("max_temperature_error_over_peak", "max_cumulative_wall_heat_error_over_peak",
                   "reversal_error_over_interval")
    runs = []
    for width in (.05, .5):
        previous = {}
        for n in (16, 32, 64):
            for collision, even_time in (("BGK", None), ("TRT", even)):
                result = slab.measure(n, width, odd, even_time)
                errors = {name: result[name] for name in observables}
                for k, fitted in enumerate(result["measured_decay_rates"], start=1):
                    errors[f"mode_{k}_relative_rate_error"] = abs(fitted/(k*math.pi)**2-1)
                orders = {} if collision not in previous else {
                    key: math.log2(previous[collision][key]/value) for key, value in errors.items()}
                residual = max(result["relative_heat_error"], result["max_balance_over_peak"],
                               result["max_projection_residual_over_peak"])
                passed = residual < 1e-12 and result["max_relative_rate_mismatch"] < 1e-8
                if n == 64:
                    rate_target = 3.8 if collision == "TRT" else 1.7
                    passed &= (min(orders[key] for key in observables) > 1.7 and
                               min(orders[f"mode_{k}_relative_rate_error"] for k in (1, 2)) > rate_target)
                runs.append(dict(collision=collision, cells=n, width_over_interval=width,
                                 dt=result["dt"], end_time=result["end_time"], errors=errors, orders=orders,
                                 max_balance_over_peak=result["max_balance_over_peak"],
                                 relative_heat_error=result["relative_heat_error"],
                                 max_relative_rate_mismatch=result["max_relative_rate_mismatch"],
                                 measured_reversal_time=result["reversal_time"],
                                 exact_reversal_time=result["exact_reversal_time"], passed=passed))
                previous[collision] = errors
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, tau_odd=odd, trt_tau_even=even, diffusivity=1., runs=runs,
                          sources=[provenance(f) for f in (__file__, slab.__file__, slab.lbm.__file__, slab.pulse.__file__)],
                          reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorLBM/src python examples/lbm_trt_slab_comparison.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
