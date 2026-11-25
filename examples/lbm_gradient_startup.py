"""Measure first-step and full-history flux accuracy away from BGK tau=1."""
import argparse
import json

import numpy as np

from conduction_provenance import provenance
import lbm_periodic_advective_flux as benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--periodic-gradient", action="store_true", help="compute centered gradients from the periodic temperature samples")
    group.add_argument("--wall-gradient", action="store_true", help="estimate gradients from cell samples beside the two x walls")
    args = parser.parse_args()
    sampled = args.periodic_gradient or args.wall_gradient
    rows = []
    for tau in (.75, 1.5):
        for walls in ((args.wall_gradient,) if sampled else (False, True)):
            previous = None
            for n in (8, 16, 32):
                row = benchmark.measure(n, 1, walls, tau=tau, gradient_start=not sampled, periodic_gradient=sampled)
                base = benchmark.measure(n, 1, walls, tau=tau, first_step_only=True)
                corrected = np.array(row["first_step_flux_errors_xyz_W_m2"])
                initial = np.array(base["first_step_flux_errors_xyz_W_m2"])
                errors = np.r_[row["temperature_error_K"], row["total_flux_errors_xyz_W_m2"]]
                orders = [] if previous is None else np.log2(previous/errors).tolist()
                row.update(tau=tau, gradient_source=("mixed-boundary cell samples" if walls else "periodic cell samples") if sampled else "analytic", periodic_axes="yz" if walls else "xyz",
                           equilibrium_first_step_errors_xyz_W_m2=initial.tolist(),
                           temperature_and_total_flux_orders=orders,
                           passed=bool(np.all(corrected < initial) and row["max_cell_energy_residual_J"] < 1e-13
                                       and (not orders or min(orders) > 0) and (n < 32 or min(orders) > 1.5)))
                if args.wall_gradient:
                    near_wall = row["first_step_near_wall_x_flux_error_W_m2"]
                    baseline = base["first_step_near_wall_x_flux_error_W_m2"]
                    row["equilibrium_first_step_near_wall_x_flux_error_W_m2"] = baseline
                    row["passed"] &= near_wall < baseline and (n < 32 or near_wall < .003)
                rows.append(row)
                previous = errors
    print(json.dumps(dict(reference="Exact space and time integrals of a moving Fourier mode; equal physical diffusivity and end time across grids and relaxation times",
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorLBM/src python examples/lbm_gradient_startup.py"+(" --wall-gradient" if args.wall_gradient else " --periodic-gradient" if sampled else ""),
        runs=rows, sources=[provenance(f) for f in (__file__, benchmark.__file__, benchmark.lbm.__file__)]),
        indent=2, allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
