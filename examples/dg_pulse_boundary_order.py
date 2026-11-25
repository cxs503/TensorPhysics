"""Locate DG center errors and isolate the contribution of the Dirichlet penalty."""
import json
import math

import numpy as np
import torch
from scipy.linalg import eigh
from tensordg import heat_1d

import dg_subinterval_heat_pulse as dg
from conduction_provenance import provenance


def eigen_means(n, width, wall_multiplier):
    rate = math.pi**2
    system = heat_1d.assemble_sipg_1d(n, 2, source=lambda x: rate*torch.sin(math.pi*x))
    stiffness = system["diffusion"].numpy().copy()
    # Control experiment: modify just the two boundary trace penalty blocks.
    # Interior faces, mass, weak derivative terms and zero Dirichlet data stay fixed.
    for indices, trace in ((slice(0,3), [1,-1,1]), (slice(-3,None), [1,1,1])):
        stiffness[indices, indices] += (wall_multiplier-1)*20*9*n*np.outer(trace, trace)
    rates, vectors = eigh(stiffness, system["mass_matrix"].numpy())
    forcing = vectors.T @ system["source_load"].numpy().ravel()
    times = np.arange(19)[:, None]/(16*rate)
    start, stop = 1.2/(16*rate), (1.2+width)/(16*rate)
    on = np.clip(times-start, 0, stop-start)
    peak = -math.expm1(-rate*(stop-start))

    def integrated_response(lam):
        response = -np.expm1(-lam*on)*np.exp(-lam*np.maximum(times-stop, 0))/lam
        return (np.diff(on,axis=0)-np.diff(response,axis=0))/(lam*np.diff(times,axis=0))

    modal = ((integrated_response(rates)*forcing) @ vectors.T).reshape(18,n,3)
    centers = modal @ np.array([1.,0.,-.5])/peak
    continuum = rate*integrated_response(rate)*np.sin(math.pi*(np.arange(n)+.5)/n)/peak
    return centers, continuum


if __name__ == "__main__":
    runs = []
    for width in (.05,.5):
        previous = {}
        for n in (16,32,64,128):
            _, native, _ = dg.measure(n,2,width,return_profiles=True)
            for multiplier in (1,2):
                values, exact = eigen_means(n,width,multiplier)
                err = abs(values-exact)
                interval, cell = np.unravel_index(err.argmax(), err.shape)
                peak = float(err.max())
                order = None if multiplier not in previous else math.log2(previous[multiplier]/peak)
                previous[multiplier] = peak
                runs.append(dict(cells=n, width_over_interval=width, boundary_penalty_multiplier=multiplier,
                                 max_center_error_over_peak=peak, worst_interval=int(interval), worst_cell=int(cell),
                                 central_cell_error_over_peak=float(err[:, n//2].max()), order=order,
                                 difference_from_production_over_peak=float(abs(native-values).max())))
    for row in runs:
        if row["boundary_penalty_multiplier"] == 1:
            row["passed"] = (row["difference_from_production_over_peak"] < 5e-10
                             and (row["cells"] < 64 or row["worst_cell"] in (0, row["cells"]-1)))
        else:
            row["passed"] = row["order"] is None or row["order"] > 3.5
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(p) for p in (__file__,dg.__file__,heat_1d.__file__)],
                          reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorDG python examples/dg_pulse_boundary_order.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
