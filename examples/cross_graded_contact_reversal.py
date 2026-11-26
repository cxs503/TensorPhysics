"""Locate wall-driven heat reversals on nested time grids graded in opposite directions."""
import json

import numpy as np
from tensorfvm.cht import rod
from tensorsph import heat_conduction_2d as sph

import sph_reversing_contact_walls as case
from conduction_provenance import provenance


def study(solver, method, grading):
    rows = []
    previous = None
    for steps in (16, 32, 64, 128):
        row = case.measure(40, steps, method, solver, grading=grading)
        error = row["contact_reversal_error_s"]
        bounds = row["crossing_interval_s"]
        passed = (row["reversed_flow"] and bounds[0] <= row["contact_reversal_s"] <= bounds[1]
                  and row["reconstruction_only_error_s"] < 1e-11)
        order = None
        if method == "exponential_quadratic":
            passed &= error < 1e-10
        elif previous is not None:
            order = float(np.log2(previous/error))
            if steps == 128:
                passed &= order > 1.8 and error < 1e-5
        rows.append(dict(solver=solver, method=method, grading=grading, steps=steps,
                         crossing_interval_s=bounds, reversal_s=row["contact_reversal_s"],
                         error_s=error, order=order, reconstruction_only_error_s=row["reconstruction_only_error_s"],
                         passed=bool(passed)))
        previous = error
    return rows


if __name__ == "__main__":
    runs = [row for solver, methods in (("fvm", ("theta", "sdirk2")),
                                        ("sph", ("theta", "sdirk2", "exponential_quadratic")))
            for method in methods for grading in (-.6, .6) for row in study(solver, method, grading)]
    passed = all(r["passed"] for r in runs)
    print(json.dumps(dict(passed=passed, runs=runs, exact_reversal_s=(np.sqrt(13.)-1)/4,
        time_grid="t = s + grading*s*(1-s), s uniformly spaced on [0,1] seconds",
        sources=[provenance(p) for p in (__file__, case.__file__, rod.__file__, sph.__file__)],
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFVM/src:../TensorSPH/src python examples/cross_graded_contact_reversal.py"),
        indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
