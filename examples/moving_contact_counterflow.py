"""Integrate directional contact power as a temperature-jump zero traverses the interface.

On a unit square contact, prescribe jump y-t K and conductance 1+y W/(m² K)
in SI coordinates for one second. Integrating in time first gives directional
heat densities (1+y)y²/2 and (1+y)(1-y)²/2 J/m².
"""
import json

import numpy as np

import spatial_contact_counterflow as spatial
from conduction_provenance import provenance


def run():
    reference = np.array([7/24, 5/24])
    rows = []
    for solver, power in (("TensorFEM_Q4", spatial.fem_power), ("TensorDG_P1", spatial.dg_power)):
        previous = None
        for n in (4, 8, 16, 32):
            heat = np.zeros((3, n))
            # Between edge crossings each face power is at most cubic in time.
            for i in range(n):
                for node in (-1/np.sqrt(3), 1/np.sqrt(3)):
                    time = (i + (1+node)/2)/n
                    heat += np.asarray(power(n, time))/(2*n)
            totals = heat[:2].sum(axis=1)
            errors = abs(totals-reference)
            edges = np.linspace(0., 1., n+1)
            forward = np.diff(edges**3/6 + edges**4/8)
            reverse = np.diff(edges/2 - edges**2/4 - edges**3/6 + edges**4/8)
            local_error = float(np.max(abs(heat[:2]-[forward, reverse])))
            balance = float(np.max(abs(heat[0]-heat[1]-heat[2])))
            passed = balance < 1e-13
            if solver == "TensorFEM_Q4":
                passed &= local_error < 1e-13 and max(errors) < 1e-13
            else:
                # DG represents conductance by its midpoint value on each face.
                conductance = 1 + (edges[1:] + edges[:-1])/2
                discrete_reference = conductance * np.array([
                    np.diff(edges**3)/6, -np.diff((1-edges)**3)/6])
                passed &= np.max(abs(heat[:2]-discrete_reference)) < 1e-13
                passed &= previous is None or np.all(previous > 3.9*errors)
                passed &= n < 32 or max(errors) < 5e-5
            rows.append(dict(solver=solver, faces=n, forward_reverse_J=totals.tolist(),
                             errors_J=errors.tolist(), max_face_error_J=local_error,
                             max_face_balance_J=balance, passed=bool(passed)))
            previous = errors
    return dict(command="python examples/moving_contact_counterflow.py",
                reference_forward_reverse_J=reference.tolist(), runs=rows,
                sources=[provenance(p) for p in (__file__, spatial.__file__,
                         spatial.thermal.__file__, spatial.dg.__file__)])


if __name__ == "__main__":
    report = run()
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in report["runs"]) else 1)
