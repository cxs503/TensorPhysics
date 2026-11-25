"""Spatial refinement after second-order Richardson extrapolation in time for a short pulse."""
import argparse
import json
import math

import numpy as np

import slab_pulse_time_refinement as pulse
from conduction_provenance import provenance


def exact_profile(n, width):
    # Dimensionless time is pi²*t. Integrate the continuum exponential on
    # each smooth piece to avoid sampling across the switch discontinuities.
    edges = np.arange(19)/16
    start, stop = 1.2/16, (1.2+width)/16
    peak = -math.expm1(start-stop)
    points, weights = np.polynomial.legendre.leggauss(12)
    means = []
    for a, b in zip(edges, edges[1:]):
        area = 0.
        knots = [a] + [t for t in (start, stop) if a < t < b] + [b]
        for left, right in zip(knots, knots[1:]):
            t = (left+right)/2 + (right-left)*points/2
            on_time = np.clip(t-start, 0, stop-start)
            factor = -np.expm1(-on_time)*np.exp(-np.maximum(t-stop, 0))
            area += (right-left)/2*(weights @ factor)
        means.append(area/((b-a)*peak))
    return np.outer(means, np.sin(np.pi*(np.arange(n)+0.5)/n))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dg-only", action="store_true", help="check the spatial order of exact-time DG center temperatures")
    args = parser.parse_args()
    runs = []
    for width in (.05,.5):
        if args.dg_only:
            previous_dg_error = None
            for n in (16,32,64):
                _, centers, _ = pulse.dg.measure(n,2,width,return_profiles=True)
                error = float(abs(centers-exact_profile(n,width)).max())
                order = None if previous_dg_error is None else math.log2(previous_dg_error/error)
                runs.append(dict(solver="dg", method="exact-time", cells=n, width_over_interval=width,
                                 center_error_over_peak=error, spatial_order=order,
                                 passed=n < 64 or (error < 1e-6 and order > 3.5)))
                previous_dg_error = error
            continue
        for method in ("theta","sdirk2"):
            previous = {}
            for n in (16,32,64):
                exact = exact_profile(n,width)
                coarse, middle, fine = [pulse.profiles(n,method,width,r) for r in (2,4,8)]
                for name in fine:
                    extrapolated = (4*fine[name]-middle[name])/3
                    earlier = (4*middle[name]-coarse[name])/3
                    error = float(abs(extrapolated-exact).max())
                    change = float(abs(extrapolated-earlier).max())
                    order = None if name not in previous else math.log2(previous[name]/error)
                    runs.append(dict(solver=name, method=method, cells=n, width_over_interval=width,
                                     fine_error_over_peak=float(abs(fine[name]-exact).max()),
                                     extrapolated_error_over_peak=error, extrapolant_change_over_peak=change,
                                     spatial_order=order, passed=change < 0.01*error
                                     and (n < 64 or (error < 0.0003 and order > 1.9))))
                    previous[name] = error
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, subdivisions=[] if args.dg_only else [2,4,8], runs=runs,
                          sources=[provenance(path) for path in (__file__,pulse.__file__,pulse.thermal.__file__,
                                                                 pulse.rod.__file__,pulse.heat.__file__,pulse.dg.__file__,
                                                                 pulse.dg.heat_1d.__file__)],
                          reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFEM/src:../TensorFVM/src:../TensorSPH/src:../TensorDG python examples/slab_pulse_space_refinement.py" + (" --dg-only" if args.dg_only else "")),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
