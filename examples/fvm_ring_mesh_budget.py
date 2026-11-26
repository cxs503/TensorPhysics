"""Compare radial mesh allocations at fixed total cell count on the heated-ring benchmark."""
import json

import numpy as np

import fvm_volume_heated_rings as benchmark
from conduction_provenance import provenance


def compare():
    rows = []
    # Six blocks give ten positive integer allocations across the three layers.
    for inner in range(1, 5):
        for middle in range(1, 6-inner):
            ratios = (inner, middle, 6-inner-middle)
            previous = None
            for block in (8, 16, 32):
                cases = [benchmark.measure(block, mode, omega, ratios)
                         for mode in (0, 1, 3) for omega in (1., 8., 24.)]
                counts = cases[0]["layer_cell_counts"]
                temperature = np.max([c["layer_temperature_errors_K"] for c in cases], axis=0)
                power = np.max([c["layer_face_errors_W_m"] for c in cases], axis=0)
                errors = np.array([temperature, power])
                orders = None if previous is None else np.log2(previous/errors)
                residual = max(max(c["modal_balance_W_m"], c["sheet_balance_max_W_m"],
                                   c["cell_balance_max_W_m"]) for c in cases)
                passed = sum(counts) == 6*block and residual < 1e-9
                if block == 32:
                    passed &= bool(np.min(orders) > 1.8 and max(temperature) < .001 and max(power) < .01)
                rows.append(dict(total_cells=sum(counts), layer_cells=counts, cases=len(cases),
                                 max_temperature_errors_K=temperature.tolist(), max_face_rate_errors_W_m=power.tolist(),
                                 temperature_orders=orders[0].tolist() if orders is not None else None,
                                 face_rate_orders=orders[1].tolist() if orders is not None else None,
                                 max_balance_residual_W_m=residual, passed=passed))
                previous = errors
    return rows


if __name__ == "__main__":
    rows = compare()
    passed = all(r["passed"] for r in rows)
    print(json.dumps(dict(passed=passed, runs=rows, sources=[provenance(p) for p in (
        __file__, benchmark.__file__, benchmark.annulus.__file__, benchmark.storing_rings.__file__)],
        reproduction="OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFVM/src python examples/fvm_ring_mesh_budget.py"),
        indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
