"""Discharge an internal sheet heat pulse through two isothermal walls."""
import json

import numpy as np
import torch

from conduction_provenance import provenance
import sph_layered_storing_contacts as slab


def measure(n, heater, method):
    dx = 1/n
    material, sheets, k, _, faces, contacts, sheet_r, capacities = slab.layered_materials(n)
    p = sheets[heater]
    # Apply 1 W/m² to the two particles representing one storing sheet.
    load = torch.zeros(n+4, dtype=torch.float64)
    load[p:p+2] = .5/dx
    stop = .0625
    times = np.r_[np.linspace(0., stop, 33), stop+np.geomspace(.001, 32., 161)]
    pulse_end = times.searchsorted(stop)
    state, heat = slab.heat.solve_transient_layered_slab(
        torch.zeros_like(load), times, dx, (0., 0.), **material, method=method,
        heat_source=lambda t: load if t < stop else 0., load_jump_times=[stop], return_x_heat=True)
    # Integrating the linear heat equation from zero initial to zero final
    # temperature gives a steady resistance divider for the injected energy.
    combined_r = contacts[faces-1].copy()
    combined_r[::2] += sheet_r
    j = 2*heater
    west_r = (dx/k[:faces[j]]).sum()+combined_r[:j].sum()+combined_r[j]/2
    total_r = (dx/k).sum()+combined_r.sum()
    expected = stop*np.array([1-west_r/total_r, west_r/total_r])
    wall_heat = np.array([-heat[:, 0].sum().item(), heat[:, -1].sum().item()])
    residuals, release = [], []
    for i, (position, capacity) in enumerate(zip(sheets, capacities)):
        stored = capacity*state[:, position:position+2].mean(dim=1).numpy()
        increments = heat[:, position].numpy()-heat[:, position+2].numpy()
        if i == heater:
            increments = increments+np.diff(np.minimum(times, stop))
        residuals.append(float(abs(np.diff(stored)-increments).max()))
        release.append(float(heat[pulse_end:, position+2].sum()-heat[pulse_end:, position].sum()))
        np.testing.assert_allclose(release[-1], stored[pulse_end]-stored[-1], atol=1e-10, rtol=0.)
    remaining = float((state[-1].abs()*material["volumetric_heat_capacity"]).sum()*dx)
    assert release[heater] > 0 and remaining < 1e-10
    assert max(residuals) < 1e-10
    np.testing.assert_allclose(wall_heat, expected, atol=1e-9, rtol=0.)
    return dict(bulk_columns=n, heated_sheet=heater, method=method, pulse_end_s=stop, final_time_s=times[-1], injected_J_m2=stop,
                outward_wall_heats_J_m2=wall_heat.tolist(), reference_J_m2=expected.tolist(),
                max_partition_error_J_m2=float(abs(wall_heat-expected).max()),
                post_pulse_sheet_release_J_m2=release, remaining_energy_bound_J_m2=remaining,
                max_sheet_balance_error_J_m2=max(residuals))


if __name__ == "__main__":
    runs = [measure(n, heater, method) for n in (16, 32, 64)
            for heater in (0, 1) for method in ("theta", "sdirk2")]
    print(json.dumps(dict(runs=runs, sources=[provenance(p) for p in
                          (__file__, slab.__file__, slab.heat.__file__)],
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src python examples/sph_sheet_pulse_partition.py"),
        indent=2, allow_nan=False))
