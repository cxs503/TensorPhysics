"""FEM/FVM transient rod with two spatial heating modes at distinct frequencies."""
import json

import numpy as np
import torch

from conduction_provenance import provenance
from tensorfem import thermal
from tensorfvm.cht import energy


def measure(n):
    h, k, capacity, omega = 1/n, 2., 3., 15.
    time = np.arange(4*n+1)*(2*np.pi/(omega*4*n))
    dt = np.diff(time)
    edges = np.arange(n+1)*h
    centers = (edges[1:]+edges[:-1])/2
    load = np.zeros((len(time), n+1))
    source = np.zeros((len(time), n))
    exact = np.zeros_like(source)
    node_initial = np.zeros(n+1)
    end_heat = np.zeros((len(dt), 2))
    modes = []
    for m, strength in ((1, 1.), (2, .7+.3j)):
        b, w = m*np.pi, m*omega
        phase = np.exp(1j*w*time)
        forcing = strength*phase
        gain = strength/(k*b*b+1j*w*capacity)
        # Source integrations against a LINE2 hat and over an FVM cell.
        hat = h*np.sinc(m*h/2)**2*np.sin(b*edges)
        hat[[0,-1]] = (1-np.sinc(m*h))/b*np.array([1, (-1)**(m+1)])
        avg = np.sinc(m*h/2)*np.sin(b*centers)
        load += forcing.real[:, None]*hat
        source += forcing.real[:, None]*avg
        exact += (gain*phase).real[:, None]*avg
        node_initial += gain.real*np.sin(b*edges)
        # q=-k*T_x; inward powers have opposite normal conventions at the two ends.
        end_heat += (np.diff(phase)*gain/(1j*w)).real[:, None]*k*b*np.array([-1, (-1)**m])
        modes.append((w, strength, avg))
    nodes = torch.from_numpy(edges[:, None])
    ids = torch.arange(n)
    fixed = torch.tensor([0,n])
    model = thermal.ThermalModel(nodes, torch.stack((ids, ids+1), 1), k, 1., capacity,
                                 fixed, nodes.new_zeros(2))
    fem = thermal.solve_transient_thermal(model, torch.from_numpy(time), torch.from_numpy(node_initial),
                .5, sparse=True, nodal_heat_history=torch.from_numpy(load))
    fem_cells = .5*(fem.temperature[:, :-1]+fem.temperature[:, 1:]).numpy()
    fem_heat = fem.nodal_reaction_heat_steps()[:, fixed].numpy()

    def heating(t):
        return sum((amp*np.exp(1j*w*t)).real*avg for w, amp, avg in modes)

    bc = {side: {"type":"dirichlet", "values":0.} for side in ("west","east")}
    history, powers = energy.solve_transient_conduction(np.full(n, h), [1.], k, capacity, bc,
                    time, exact[0], source=heating, return_wall_power=True)
    wall = np.column_stack((powers["west"][:,0], powers["east"][:,0]))
    fvm_heat = .5*dt[:,None]*(wall[:-1]+wall[1:])
    generation = .5*dt*h*(source[1:]+source[:-1]).sum(axis=1)
    reports = []
    for name, cells, heat in (("FEM", fem_cells, fem_heat), ("FVM", history[:,0], fvm_heat)):
        stored = capacity*np.diff(cells.mean(axis=1))
        balance = abs(stored-heat.sum(axis=1)-generation).max()
        # Divide by interval duration to compare errors on different time grids.
        reports.append(dict(solver=name, cells=n, steps=len(dt),
                     max_cell_temperature_error_K=float(abs(cells-exact).max()),
                     max_interval_mean_power_error_W=float(abs((heat-end_heat)/dt[:,None]).max()),
                     max_step_balance_error_J=float(balance)))
    return reports


def main():
    previous = {}
    runs = []
    for n in (8, 16, 32, 64, 128):
        for report in measure(n):
            name = report["solver"]
            errors = np.array([report["max_cell_temperature_error_K"], report["max_interval_mean_power_error_W"]])
            orders = np.log2(previous[name]/errors) if name in previous else None
            passed = np.isfinite(errors).all() and report["max_step_balance_error_J"] < 1e-12
            if orders is not None:
                passed &= min(orders) > 1.75
            report.update(orders=None if orders is None else orders.tolist(), passed=bool(passed))
            runs.append(report)
            previous[name] = errors
    print(json.dumps(dict(runs=runs, sources=[provenance(f) for f in
                     (__file__, energy.__file__, thermal.__file__)]), indent=2, allow_nan=False))
    return 0 if all(row["passed"] for row in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
