"""Verify closed-loop Q4 reaction flux on a square with a smooth wall-normal trace."""
import json

import numpy as np
import torch

from conduction_provenance import provenance
from tensorfem import thermal, thermal_analytic


def measure(n, omega, grading, sparse):
    mesh = thermal_analytic.structured_quad_mesh(n, n, dtype=torch.float64)
    nodes = mesh.nodes.numpy().copy()
    # Keep endpoints fixed; vary x and y spacing independently.
    nodes += grading*np.sin(2*np.pi*nodes)*np.array([1., -.6])/(2*np.pi)
    x, y = nodes.T
    xx, yy = x*(1-x), y*(1-y)
    a, b = .31-.27j, .85+.43j
    k, c, depth = 1.7, 2.3, .65
    exact = a*((x-.5)**2+(y-.5)**2)+b*xx*yy
    source = 1j*omega*c*exact-k*(4*a-2*b*(xx+yy))
    grid = np.arange(len(nodes)).reshape(n+1,n+1)
    wall = np.r_[grid[0,:-1], grid[:-1,-1], grid[-1,:0:-1], grid[:0:-1,0]]
    model = thermal.ThermalModel(torch.from_numpy(nodes), mesh.elements, k, 1., c,
        torch.from_numpy(wall), torch.from_numpy(exact[wall]),
        heat_source=torch.from_numpy(source[mesh.elements.numpy()]), thickness=depth, element_type="q4")
    result = thermal.solve_harmonic_thermal(model, omega, sparse=sparse)
    powers = -result.reaction_power[wall]
    starts = nodes[wall]
    ends = np.roll(starts, -1, axis=0)
    areas = torch.from_numpy(depth*np.linalg.norm(ends-starts,axis=1))
    fit = thermal.recover_linear_edge_flux(powers, areas, closed=True)
    segment = ((fit+fit.roll(-1))/2).numpy()
    # On each wall q_out = -k*(a-b*t*(1-t)), t is x or y along the edge.
    horizontal = np.abs(ends[:,0]-starts[:,0]) > .5*np.linalg.norm(ends-starts, axis=1)
    t0, t1 = (np.where(horizontal, xy[:,0], xy[:,1]) for xy in (starts, ends))
    mean_parabola = (t0+t1)/2-(t0**2+t0*t1+t1**2)/3
    target = -k*(a-b*mean_parabola)
    total = areas.numpy()@segment
    shift = n//2+1
    rotated = thermal.recover_linear_edge_flux(powers.roll(shift), areas.roll(shift), closed=True).roll(-shift)
    return dict(cells_per_axis=n, omega_rad_s=omega, grading=grading, sparse=sparse,
                nodal_temperature_error_K=float(abs(result.temperature.numpy()-exact).max()),
                segment_max_flux_error_W_m2=float(abs(segment-target).max()),
                segment_rms_flux_error_W_m2=float(np.sqrt(np.average(abs(segment-target)**2, weights=areas.numpy()))),
                total_outward_power_error_W=float(abs(total+4*k*depth*(a-b/6))),
                projected_power_residual_W=float(abs(total-powers.sum().item())),
                seam_rotation_error_W_m2=float(abs(rotated-fit).max()))


if __name__ == "__main__":
    rows = []
    metrics = ("segment_max_flux_error_W_m2", "segment_rms_flux_error_W_m2", "total_outward_power_error_W")
    for omega in (.5, 4.):
        for grading in (0., .35):
            for sparse in (False, True):
                previous = None
                for n in (4, 8, 16, 32):
                    row = measure(n, omega, grading, sparse)
                    errors = np.array([row[m] for m in metrics])
                    orders = [] if previous is None else np.log2(previous/errors).tolist()
                    row.update(orders=orders, passed=bool(row["projected_power_residual_W"] < 1e-12
                               and row["seam_rotation_error_W_m2"] < 1e-12
                               and all(v > 1.7 for v in orders)
                               and (n != 32 or row["segment_max_flux_error_W_m2"] < 5e-4)))
                    previous = errors
                    rows.append(row)
    print(json.dumps(dict(reference="Q4 unit square with biquadratic manufactured temperature, continuous normal flux at all four corners, and exact segment and perimeter integrals",
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFEM/src python examples/fem_closed_wall_flux.py",
        order_metrics=metrics, runs=rows, sources=[provenance(p) for p in (__file__,thermal.__file__,thermal_analytic.__file__)]),
        indent=2, allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
