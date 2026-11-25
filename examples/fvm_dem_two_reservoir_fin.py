"""Compare cell-mean FVM and nodal DEM temperatures against the same two-reservoir fin."""
import json

import numpy as np
from tensorfvm.cht import rod

import dem_lattice_two_reservoir_fin as dem
from conduction_provenance import provenance


def finite_volume(n, split):
    # Insert the reservoir boundary so each FVM cell exchanges with exactly
    # one reservoir. DEM uses its uniform lattice and partial particle areas.
    edges = np.unique(np.append(np.linspace(0, 1, n+1), split))
    widths = np.diff(edges)
    centers = (edges[:-1] + edges[1:])/2
    hot = centers < split
    reservoirs = np.array([321., 283.])
    films = np.array([4., 7.])
    walls = np.array([305., 303.])
    axial = 1.7
    ambient = np.where(hot, *reservoirs)
    coefficient = np.where(hot, *films)
    # Only temperature differences drive this linear problem. Center the
    # unknowns to reduce cancellation when extracting the two end powers.
    baseline = walls.mean()
    cell, _, left, right = rod.solve_steady_stepped_rod(
        widths, 1., axial, 1., walls-baseline, flux_order=2, heat_loss_coefficient=coefficient,
        bulk_ambient_temperature=ambient-baseline, source_derivative_breaks=[int(hot.sum())],
    )
    cell += baseline
    midpoint, powers = dem.reference(centers, split, axial, films, reservoirs, walls)
    # Integrate the exact hyperbolic profile over the cell rather than using
    # its midpoint value as the reference for a finite-volume average.
    z = np.sqrt(coefficient/axial)*widths/2
    mean = ambient + (midpoint-ambient)*np.sinh(z)/z
    loss = widths*coefficient*(cell-ambient)
    values = np.array([loss[hot].sum(), loss[~hot].sum()])
    return dict(cells=len(cell), mean_error_K=float(np.max(abs(cell-mean))),
                hot_outward_W=float(values[0]), cold_outward_W=float(values[1]),
                hot_error_W=float(abs(values[0]-powers[0])), cold_error_W=float(abs(values[1]-powers[1])),
                balance_error_W=float(abs(left[0]-right[-1]-values.sum())))


if __name__ == "__main__":
    runs = []
    for split in (0.5, 0.37):
        for n in (16, 32, 64, 128, 256):
            fv = finite_volume(n, split)
            particles = dem.measure(n, split)
            passed = all(result["hot_outward_W"] < 0 < result["cold_outward_W"]
                         and result["balance_error_W"] < 1e-8 for result in (fv, particles))
            if n == 256:
                passed = passed and fv["mean_error_K"] < 2e-4 and particles["field_error_K"] < 2e-4
                passed = passed and all(max(result["hot_error_W"], result["cold_error_W"]) < 1e-3
                                        for result in (fv, particles))
            runs.append(dict(intervals=n, split=split, fvm=fv, dem=particles, passed=passed))
    passed = all(run["passed"] for run in runs)
    print(json.dumps(dict(passed=passed, runs=runs,
                          sources=[provenance(path) for path in (__file__, dem.__file__, rod.__file__, dem.contact_chain.__file__)],
                          reproduction="PYTHONPATH=../TensorDEM/src:../TensorFVM/src python examples/fvm_dem_two_reservoir_fin.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
