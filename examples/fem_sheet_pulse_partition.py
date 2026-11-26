"""Verify wall heat partition and exit-time moments for two capacitive FEM contacts."""
import json

import numpy as np
import torch
from tensorfem import thermal

from conduction_provenance import provenance
import sph_sheet_pulse_partition as pulse


def measure(n, heater, method):
    dx = 1/n
    _, _, k, cv, faces, contacts, sheet_r, capacities = pulse.slab.layered_materials(n)
    resistances = contacts[faces-1].copy()
    resistances[::2] += sheet_r
    # Duplicate the bulk traces at every contact. A storing sheet has an
    # independent node between two half resistances at the same location.
    x, elements, links, conductance, sheets = [0.], [], [], [], []
    for cell in range(n):
        node = len(x)-1
        x.append((cell+1)*dx)
        elements.append((node, node+1))
        for j in np.flatnonzero(faces == cell+1):
            segments = 1 if j == 1 else 2
            for _ in range(segments):
                node = len(x)-1
                links.append((node, node+1))
                conductance.append(segments/resistances[j])
                x.append(x[-1])
            if segments == 2:
                sheets.append(len(x)-2)
    coords = torch.tensor(x, dtype=torch.float64)
    sheet_mass = torch.zeros_like(coords)
    sheet_mass[sheets] = coords.new_tensor(capacities)
    model = thermal.ThermalModel(coords[:, None], torch.tensor(elements), torch.from_numpy(k),
        1., torch.from_numpy(cv), torch.tensor([0, len(x)-1]), coords.new_zeros(2),
        contact_pairs=torch.tensor(links), contact_conductance=coords.new_tensor(conductance),
        nodal_heat_capacity=sheet_mass)
    stop = .0625
    times = np.r_[np.linspace(0., stop, n+1), stop+np.geomspace(.016/n, 32., 5*n+1)]
    injected = np.diff(np.minimum(times, stop))
    loads = coords.new_zeros((len(injected), len(x)))
    loads[:, sheets[heater]] = torch.from_numpy(injected)
    result = thermal.solve_transient_thermal(model, torch.from_numpy(times), torch.zeros_like(coords),
        method=method, theta=.5, sparse=True, nodal_step_heat=loads)
    outward = -result.nodal_reaction_heat_steps().numpy()[:, [0, -1]]
    total = outward.sum(axis=0)
    mid = (times[:-1]+times[1:])/2
    centroid = mid@outward/total
    spread = np.sqrt((((mid[:, None]-centroid)**2)*outward).sum(axis=0)/total)
    total_r = (dx/k).sum()+resistances.sum()
    j = 2*heater
    west_r = (dx/k[:faces[j]]).sum()+resistances[:j].sum()+resistances[j]/2
    expected = stop*np.array([1-west_r/total_r, west_r/total_r])
    ref_mean, ref_std = pulse.exit_time_spread(dx, k, cv, faces, resistances, capacities, heater, stop)
    sheet_errors = []
    mean_t = result.interval_temperature.numpy()
    for i, s in enumerate(sheets):
        incoming = np.diff(times)*(mean_t[:, s-1]+mean_t[:, s+1]-2*mean_t[:, s])*2/resistances[2*i]
        stored = capacities[i]*np.diff(result.temperature[:, s].numpy())
        sheet_errors.append(abs(stored-incoming-(injected if i == heater else 0.)).max())
    np.testing.assert_allclose(total, expected, rtol=0., atol=1e-10)
    np.testing.assert_array_less(sheet_errors, 1e-10)
    if n == 64:
        np.testing.assert_allclose(centroid, ref_mean, rtol=0., atol=1e-4)
        np.testing.assert_allclose(spread, ref_std, rtol=0., atol=3e-4)
    return dict(reference_wall_heats_J_m2=expected.tolist(), reference_mean_s=ref_mean.tolist(),
                reference_std_s=ref_std.tolist(), elements=n, heated_sheet=heater, method=method, wall_heats_J_m2=total.tolist(),
                wall_mean_exit_times_s=centroid.tolist(), wall_exit_time_std_s=spread.tolist(),
                mean_error_s=float(abs(centroid-ref_mean).max()), std_error_s=float(abs(spread-ref_std).max()),
                wall_heat_error_J_m2=float(abs(total-expected).max()), sheet_balance_J_m2=float(max(sheet_errors)))


if __name__ == "__main__":
    runs = [measure(n, heater, method) for heater in (0, 1) for method in ("theta", "sdirk2")
            for n in (16, 32, 64)]
    print(json.dumps(dict(runs=runs, sources=[provenance(p) for p in
                          (__file__, thermal.__file__, pulse.__file__, pulse.slab.__file__)],
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFEM/src:../TensorSPH/src python examples/fem_sheet_pulse_partition.py"),
        indent=2, allow_nan=False))
