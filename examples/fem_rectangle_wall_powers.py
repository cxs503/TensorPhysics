"""Check one-sided Q4 wall powers for T=x+2y+xy on a 2 m by 1 m rectangle."""
import json

import torch

from conduction_provenance import provenance
from tensorfem import thermal, thermal_analytic


def measure(nx, ny, graded, sparse):
    mesh = thermal_analytic.structured_quad_mesh(nx, ny, lx=2., dtype=torch.float64)
    nodes = mesh.nodes.clone()
    if graded:
        nodes[:, 0] = 2 * (nodes[:, 0] / 2) ** 1.35
        nodes[:, 1] = nodes[:, 1] ** 1.2
    x, y = nodes.T
    reference = x + 2 * y + x * y
    boundary = torch.where((x == 0) | (x == 2) | (y == 0) | (y == 1))[0]
    model = thermal.ThermalModel(nodes, mesh.elements, 1., 1., 1., boundary,
                                 reference[boundary], element_type="q4")
    temperature = thermal.solve_steady_thermal(model, sparse=sparse)
    edges = thermal.thermal_element_edge_power(model, temperature).reshape(ny, nx, 4)
    # Integrate the linear exact flux using each edge midpoint and physical length.
    xs = x.reshape(ny + 1, nx + 1)[0]
    ys = y.reshape(ny + 1, nx + 1)[:, 0]
    vertical = ys.diff() * (1 + (ys[1:] + ys[:-1]) / 2)
    horizontal = xs.diff() * (2 + (xs[1:] + xs[:-1]) / 2)
    faces = {"left": (edges[:, 0, 3], vertical, 1.5),
             "right": (edges[:, -1, 1], -vertical, -1.5),
             "bottom": (edges[0, :, 0], horizontal, 6.),
             "top": (edges[-1, :, 2], -horizontal, -6.)}
    totals = {face: float(values.sum()) for face, (values, _, _) in faces.items()}
    segment_error = max(float((values - exact).abs().max()) for values, exact, _ in faces.values())
    wall_error = max(abs(totals[face] - exact) for face, (_, _, exact) in faces.items())
    balance_error = abs(sum(totals.values()))
    temperature_error = float((temperature - reference).abs().max())
    return dict(nx=nx, ny=ny, graded=graded, sparse=sparse, outward_power_W=totals,
                max_segment_error_W=segment_error, max_wall_error_W=wall_error,
                net_power_error_W=balance_error, max_temperature_error_K=temperature_error,
                passed=max(segment_error, wall_error, balance_error, temperature_error) < 1e-10)


if __name__ == "__main__":
    runs = [measure(nx, ny, graded, sparse) for nx, ny in ((3, 2), (8, 5), (24, 16))
            for graded in (False, True) for sparse in (False, True)]
    print(json.dumps(dict(reference="Steady T=x+2y+xy, unit conductivity and thickness, zero source",
                          reproduction="OMP_NUM_THREADS=1 PYTHONPATH=../TensorFEM/src python examples/fem_rectangle_wall_powers.py",
                          runs=runs, sources=[provenance(path) for path in
                          (__file__, thermal.__file__, thermal_analytic.__file__)]), indent=2, allow_nan=False))
    raise SystemExit(0 if all(run["passed"] for run in runs) else 1)
