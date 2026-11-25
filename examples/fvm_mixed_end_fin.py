"""Verify absolute-temperature fin solutions with fixed, power and film ends."""
import argparse
import json

import numpy as np
from tensorfvm.cht import rod

from conduction_provenance import provenance


def measure(n, types):
    k, exchange, ambient = 1.7, 4., 305.
    m = np.sqrt(exchange/k)
    h = 1/n
    x = (np.arange(n)+0.5)*h
    # Exact T = ambient + 2*cosh(mx) - sinh(mx); Q = -k*dT/dx.
    excess = 2*np.cosh(m*x) - np.sinh(m*x)
    z = m*h/2
    exact_mean = ambient + excess*np.sinh(z)/z
    ends = np.array([0., 1.])
    exact_faces = ambient + 2*np.cosh(m*ends) - np.sinh(m*ends)
    exact_current = -k*m*(2*np.sinh(m*ends)-np.cosh(m*ends))
    values = exact_faces.copy()
    film = 3.0
    for i, kind in enumerate(types):
        inward = (1, -1)[i]*exact_current[i]
        if kind == "power":
            values[i] = inward
        elif kind == "robin":
            values[i] += inward/film
    cells, faces, left, right = rod.solve_steady_stepped_rod(
        np.full(n, h), 1., k, 1., values, end_types=types, end_film_coefficient=film,
        flux_order=2, heat_loss_coefficient=exchange, bulk_ambient_temperature=ambient,
        source_derivative_breaks=[],
    )
    loss = exchange*h*(cells-ambient).sum()
    return dict(cells=n, ends=types, cell_mean_error_K=float(abs(cells-exact_mean).max()),
                wall_temperature_error_K=float(abs(faces[[0,-1]]-exact_faces).max()),
                wall_power_error_W=float(abs(np.array([left[0], right[-1]])-exact_current).max()),
                balance_error_W=float(abs(left[0]-right[-1]-loss)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--free-ends", action="store_true", help="use only power and film end conditions")
    args = parser.parse_args()
    cases = ([(left, right) for left in ("power", "robin") for right in ("power", "robin")]
             if args.free_ends else [("power", "temperature"), ("robin", "temperature"),
                                     ("temperature", "power"), ("temperature", "robin")])
    runs = []
    for types in cases:
        for n in (32, 64, 128, 256):
            r = measure(n, types)
            r["passed"] = r["balance_error_W"] < 1e-8 and (n < 256 or max(
                r["cell_mean_error_K"], r["wall_temperature_error_K"], r["wall_power_error_W"]) < 1e-7)
            runs.append(r)
    passed = all(r["passed"] for r in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(__file__), provenance(rod.__file__)],
                          reproduction="PYTHONPATH=../TensorFVM/src python examples/fvm_mixed_end_fin.py" + (" --free-ends" if args.free_ends else "")),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
