"""Run python examples/spatial_contact_counterflow.py with FEM and DG installed.

Verify instantaneous power on a unit-area contact with a prescribed temperature
jump y-5/9 K and conductance 1+y W/(m² K). The net continuum power vanishes.
"""
import json

import numpy as np
import torch
from tensorfem import thermal
from tensordg import poisson_2d_bc as dg

from conduction_provenance import provenance


ROOT = 5/9


def fem_power(n):
    y = torch.linspace(0., 1., n+1, dtype=torch.float64)
    columns = torch.stack([torch.stack((y*0+x, y), dim=1) for x in (0., .5, .5, 1.)])
    nodes = columns.reshape(-1, 2)
    ids = torch.arange(len(nodes)).reshape(4, n+1)
    elements = torch.cat([torch.stack((ids[i,:-1], ids[i+1,:-1], ids[i+1,1:], ids[i,1:]), dim=1)
                          for i in (0, 2)])
    pairs = ids[1:3].unfold(1, 2, 1).transpose(0, 1)
    temperature = torch.cat((y-ROOT, y-ROOT, 0*y, 0*y))
    # TensorFEM takes area-scaled conductances (W/K) at the edge endpoints.
    edge_conductance = (1+y).unfold(0, 2, 1)*y.diff()[:, None]
    model = thermal.ThermalModel(nodes, elements, 1., 1., 1., torch.arange(len(nodes)), temperature,
                                 element_type="q4", contact_edge_pairs=pairs,
                                 contact_edge_conductance=edge_conductance)
    forward, reverse = thermal.thermal_contact_edge_directional_power(model, temperature)
    net = thermal.thermal_contact_edge_power(model, temperature)
    return forward.numpy(), reverse.numpy(), net.numpy()


def dg_power(n):
    mid = (torch.arange(n, dtype=torch.float64)+.5)/n
    field = mid.new_zeros((2, n, 2, 2))
    field[0, :, 0, 0] = mid-ROOT
    field[0, :, 0, 1] = .5/n
    contacts = {"x": 1/(1+mid[None, :])}
    power = dg.directional_contact_face_powers_2d(field, contacts)["x"]
    net, _ = dg.poisson_face_powers_2d(field, contact_resistance=contacts)
    return power["forward"].numpy(), power["reverse"].numpy(), net[1].numpy()


def run():
    def primitive(y):
        return y**3/3+(1-ROOT)*y**2/2-ROOT*y

    rows = []
    for name, evaluate in (("TensorFEM_Q4", fem_power), ("TensorDG_P1", dg_power)):
        previous = None
        for n in (4, 8, 16, 32, 64):
            edges = np.linspace(0., 1., n+1)
            split = np.clip(ROOT, edges[:-1], edges[1:])
            pos = primitive(edges[1:])-primitive(split)
            neg = primitive(edges[:-1])-primitive(split)
            forward, reverse, net = evaluate(n)
            totals = np.array([forward.sum(), reverse.sum()])
            errors = abs(totals-[pos.sum(), neg.sum()])
            local_error = max(max(abs(forward-pos)), max(abs(reverse-neg)))
            balance = max(abs(forward-reverse-net))
            passed = balance < 1e-13 and np.count_nonzero((forward > 0) & (reverse > 0)) == 1
            if name == "TensorFEM_Q4":
                passed &= local_error < 1e-13 and max(errors) < 1e-13
            else:
                passed &= previous is None or np.all(previous > 3*errors)
                passed &= n < 64 or max(errors) < 2e-5
            rows.append(dict(solver=name, faces=n, forward_reverse_W=totals.tolist(),
                             total_errors_W=errors.tolist(), max_face_error_W=float(local_error),
                             net_power_W=float(net.sum()), max_face_balance_W=float(balance), passed=bool(passed)))
            previous = errors
    return dict(reference_forward_reverse_W=[primitive(1)-primitive(ROOT), -primitive(ROOT)],
                reference_net_W=0., zero_position_m=ROOT, runs=rows,
                sources=[provenance(p) for p in (__file__, thermal.__file__, dg.__file__)])


if __name__ == "__main__":
    report = run()
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in report["runs"]) else 1)
