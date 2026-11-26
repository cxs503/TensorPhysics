"""Discharge an internal sheet heat pulse through two isothermal walls."""
import json

import numpy as np
import torch

from conduction_provenance import provenance
import sph_layered_storing_contacts as slab


def mean_exit_times(dx, k, cv, contacts, faces, combined_r, capacities, heater_r, total_r, pulse_end):
    """First wall heat moments per unit injected energy, divided by each exit fraction."""
    # In resistance coordinates the impulse temperature integral is the
    # Dirichlet Green function. Each element of its stored-energy integral
    # exits the two walls in the steady resistance-divider proportions.
    def weighted_green(r):
        green = np.minimum(r, heater_r)*(total_r-np.maximum(r, heater_r))/total_r
        return green[:, None]*np.column_stack((1-r/total_r, r/total_r))

    jumps = np.zeros_like(contacts)
    jumps[faces-1] = combined_r
    dr = dx/k
    starts = np.cumsum(dr)-dr+np.r_[0., np.cumsum(jumps)]
    # Two Gauss points exactly integrate the quadratic product in each cell.
    offset = .5/np.sqrt(3)
    bulk = sum(dx*cv @ weighted_green(starts+f*dr)/2 for f in (.5-offset, .5+offset))
    sheet_r = starts[faces[::2]]-combined_r[::2]/2
    moments = bulk+np.asarray(capacities) @ weighted_green(sheet_r)
    return pulse_end/2+moments/np.array([1-heater_r/total_r, heater_r/total_r])


def measure(n, heater, method):
    dx = 1/n
    material, sheets, k, cv, faces, contacts, sheet_r, capacities = slab.layered_materials(n)
    p = sheets[heater]
    # Apply 1 W/m² to the two particles representing one storing sheet.
    load = torch.zeros(n+4, dtype=torch.float64)
    load[p:p+2] = .5/dx
    stop = .0625
    times = np.r_[np.linspace(0., stop, n+1), stop+np.geomspace(.016/n, 32., 5*n+1)]
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
    wall_increments = np.column_stack((-heat[:, 0].numpy(), heat[:, -1].numpy()))
    centroid = ((times[1:]+times[:-1])/2) @ wall_increments/wall_heat
    expected_time = mean_exit_times(dx, k, cv, contacts, faces, combined_r, capacities, west_r, total_r, stop)
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
                wall_mean_exit_times_s=centroid.tolist(), reference_mean_exit_times_s=expected_time.tolist(),
                max_mean_exit_time_error_s=float(abs(centroid-expected_time).max()),
                max_partition_error_J_m2=float(abs(wall_heat-expected).max()),
                post_pulse_sheet_release_J_m2=release, remaining_energy_bound_J_m2=remaining,
                max_sheet_balance_error_J_m2=max(residuals))


if __name__ == "__main__":
    runs = []
    for heater in (0, 1):
        for method in ("theta", "sdirk2"):
            previous = None
            for n in (16, 32, 64):
                row = measure(n, heater, method)
                error = row["max_mean_exit_time_error_s"]
                if previous is not None:
                    row["mean_exit_time_order"] = float(np.log2(previous/error))
                    assert row["mean_exit_time_order"] > 1.5
                if n == 64:
                    assert error < 8e-5
                runs.append(row)
                previous = error
    print(json.dumps(dict(runs=runs, sources=[provenance(p) for p in
                          (__file__, slab.__file__, slab.heat.__file__)],
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src python examples/sph_sheet_pulse_partition.py"),
        indent=2, allow_nan=False))
