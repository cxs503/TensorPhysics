"""Measure first-step and full-history flux accuracy away from BGK tau=1."""
import json

import numpy as np

from conduction_provenance import provenance
import lbm_periodic_advective_flux as benchmark


if __name__ == "__main__":
    rows = []
    for tau in (.75, 1.5):
        for walls in (False, True):
            previous = None
            for n in (8, 16, 32):
                row = benchmark.measure(n, 1, walls, tau=tau, gradient_start=True)
                base = benchmark.measure(n, 1, walls, tau=tau, first_step_only=True)
                corrected = np.array(row["first_step_flux_errors_xyz_W_m2"])
                initial = np.array(base["first_step_flux_errors_xyz_W_m2"])
                errors = np.r_[row["temperature_error_K"], row["total_flux_errors_xyz_W_m2"]]
                orders = [] if previous is None else np.log2(previous/errors).tolist()
                row.update(tau=tau, periodic_axes="yz" if walls else "xyz",
                           equilibrium_first_step_errors_xyz_W_m2=initial.tolist(),
                           temperature_and_total_flux_orders=orders,
                           passed=bool(np.all(corrected < initial) and row["max_cell_energy_residual_J"] < 1e-13
                                       and (not orders or min(orders) > 0) and (n < 32 or min(orders) > 1.5)))
                rows.append(row)
                previous = errors
    print(json.dumps(dict(reference="Exact space and time integrals of a moving Fourier mode; equal physical diffusivity and end time across grids and relaxation times",
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorLBM/src python examples/lbm_gradient_startup.py",
        runs=rows, sources=[provenance(f) for f in (__file__, benchmark.__file__, benchmark.lbm.__file__)]),
        indent=2, allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
