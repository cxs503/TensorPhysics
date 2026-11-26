"""Verify wall heat partition and exit-time moments for two capacitive contacts using FEM, FVM or a DEM thermal contact chain."""
import argparse
import json

import numpy as np
import torch

from conduction_provenance import provenance
import sph_sheet_pulse_partition as pulse


def fem_history(n, heater, method, faces, k, cv, resistances, capacities, times, injected):
    from tensorfem import thermal

    dx = 1/n
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
    loads = coords.new_zeros((len(injected), len(x)))
    loads[:, sheets[heater]] = torch.from_numpy(injected)
    result = thermal.solve_transient_thermal(model, torch.from_numpy(times), torch.zeros_like(coords),
        method=method, theta=.5, sparse=True, nodal_step_heat=loads)
    outward = -result.nodal_reaction_heat_steps().numpy()[:, [0, -1]]
    sheet_errors = []
    mean_t = result.interval_temperature.numpy()
    for i, s in enumerate(sheets):
        incoming = np.diff(times)*(mean_t[:, s-1]+mean_t[:, s+1]-2*mean_t[:, s])*2/resistances[2*i]
        stored = capacities[i]*np.diff(result.temperature[:, s].numpy())
        sheet_errors.append(abs(stored-incoming-(injected if i == heater else 0.)).max())
    return outward, sheet_errors


def fvm_history(n, heater, method, faces, k, cv, resistances, capacities, times, injected):
    from tensorfvm.cht import rod

    # Each extra volume stores C and contributes R/2 on either side of
    # its center in the two-point flux scheme. The bookkeeping width is dx.
    dx = 1/n
    sheets = faces[::2]+[0, 1]
    conductivity = np.insert(k, faces[::2], dx/resistances[::2])
    density = np.insert(cv, faces[::2], np.asarray(capacities)/dx)
    contacts = np.zeros(n+1)
    contacts[faces[1]] = resistances[1]
    source = np.zeros(n+2)
    source[sheets[heater]] = 1/dx
    stop = times[np.count_nonzero(injected)]
    state, _, _, _, ledger = rod.solve_transient_stepped_rod(
        np.full(n+2, dx), 1., conductivity, density, (0., 0.), times, np.zeros(n+2),
        method=method, flux_order=1, contact_resistance=contacts, return_step_heat=True,
        source=lambda t: source if t < stop else 0., load_jump_times=[stop])
    stored = np.diff(state[:, sheets], axis=0)*capacities
    incoming = ledger["left"][:, sheets]-ledger["right"][:, sheets]
    incoming[:, heater] += injected
    return -ledger["end_input"], abs(stored-incoming).max(axis=0)


def dem_history(n, heater, faces, k, cv, resistances, capacities, times, injected):
    from tensordem.thermal import contact_chain

    contacts = np.zeros(n-1)
    contacts[faces-1] = resistances
    sheets = faces[::2]
    sheet_capacity = np.zeros_like(contacts)
    sheet_capacity[sheets-1] = capacities
    links, capacity = contact_chain.cell_centered_slab_network(
        np.full(n, 1/n), k, cv, contact_resistance=contacts, interface_heat_capacity=sheet_capacity)
    # One west boundary node plus one extra node per preceding storing contact.
    nodes = sheets+1+np.arange(len(sheets))
    np.testing.assert_array_equal(capacity[nodes], capacities)
    state = np.zeros_like(capacity)
    power = np.zeros_like(capacity)
    outward, residuals = [], []
    wall_moments = np.zeros((2, 2))
    for start, duration, energy in zip(times[:-1], np.diff(times), injected):
        power[nodes[heater]] = energy/duration
        end, heat, local = contact_chain.constant_contact_interval(
            links, capacity, duration, state, nodal_power=power, t_left=0., t_right=0., return_heat_moments=True)
        outward.append([-heat[0], heat[-1]])
        weighted = local[:, [0, -1]]*[-1., 1.]
        wall_moments[0] += weighted[0]+start*np.asarray(outward[-1])
        wall_moments[1] += weighted[1]+2*start*weighted[0]+start**2*np.asarray(outward[-1])
        residuals.append(capacity[nodes]*(end[nodes]-state[nodes])
                         -heat[nodes-1]+heat[nodes]-duration*power[nodes])
        state = end
    return np.asarray(outward), np.abs(residuals).max(axis=0), wall_moments


def measure(n, heater, method, *, fvm=False, dem=False):
    dx = 1/n
    _, _, k, cv, faces, contacts, sheet_r, capacities = pulse.slab.layered_materials(n)
    resistances = contacts[faces-1].copy()
    resistances[::2] += sheet_r
    stop = .0625
    times = np.r_[np.linspace(0., stop, n+1), stop+np.geomspace(.016/n, 32., 5*n+1)]
    if dem:
        times = np.array([0., stop, times[-1]])
    injected = np.diff(np.minimum(times, stop))
    if dem:
        outward, sheet_errors, moments = dem_history(n, heater, faces, k, cv, resistances, capacities, times, injected)
    else:
        history = fvm_history if fvm else fem_history
        outward, sheet_errors = history(n, heater, method, faces, k, cv, resistances, capacities, times, injected)
    total = outward.sum(axis=0)
    if dem:
        centroid = moments[0]/total
        spread = np.sqrt(moments[1]/total-centroid**2)
        fine = np.r_[np.linspace(0., stop, 9), stop+np.geomspace(.001, 32., 21)]
        _, _, check = dem_history(n, heater, faces, k, cv, resistances, capacities, fine,
                                  np.diff(np.minimum(fine, stop)))
        np.testing.assert_allclose(moments, check, atol=1e-10, rtol=0.)
    else:
        mid = (times[:-1]+times[1:])/2
        centroid = mid@outward/total
        spread = np.sqrt((((mid[:, None]-centroid)**2)*outward).sum(axis=0)/total)
    total_r = (dx/k).sum()+resistances.sum()
    j = 2*heater
    west_r = (dx/k[:faces[j]]).sum()+resistances[:j].sum()+resistances[j]/2
    expected = stop*np.array([1-west_r/total_r, west_r/total_r])
    ref_mean, ref_std = pulse.exit_time_spread(dx, k, cv, faces, resistances, capacities, heater, stop)
    np.testing.assert_allclose(total, expected, rtol=0., atol=1e-10)
    np.testing.assert_array_less(sheet_errors, 1e-10)
    if n == 64:
        np.testing.assert_allclose(centroid, ref_mean, rtol=0., atol=1e-4)
        np.testing.assert_allclose(spread, ref_std, rtol=0., atol=3e-4)
    return dict(reference_wall_heats_J_m2=expected.tolist(), reference_mean_s=ref_mean.tolist(),
                reference_std_s=ref_std.tolist(), output_intervals=len(injected), elements=n, heated_sheet=heater, method=method, wall_heats_J_m2=total.tolist(),
                wall_mean_exit_times_s=centroid.tolist(), wall_exit_time_std_s=spread.tolist(),
                mean_error_s=float(abs(centroid-ref_mean).max()), std_error_s=float(abs(spread-ref_std).max()),
                wall_heat_error_J_m2=float(abs(total-expected).max()), sheet_balance_J_m2=float(max(sheet_errors)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fvm", action="store_true", help="use the finite-volume rod with lumped sheet cells")
    group.add_argument("--dem", action="store_true", help="use the lumped thermal contact chain")
    args = parser.parse_args()
    fvm, dem = args.fvm, args.dem
    if fvm:
        from tensorfvm.cht import rod
        solver = rod
        pythonpath = "../TensorFVM/src:../TensorSPH/src"
    elif dem:
        from tensordem.thermal import contact_chain
        solver = contact_chain
        pythonpath = "../TensorDEM/src:../TensorSPH/src"
    else:
        from tensorfem import thermal
        solver = thermal
        pythonpath = "../TensorFEM/src:../TensorSPH/src"
    methods = ("constant_interval",) if dem else ("theta", "sdirk2")
    runs = [measure(n, heater, method, fvm=fvm, dem=dem) for heater in (0, 1) for method in methods
            for n in (16, 32, 64)]
    print(json.dumps(dict(solver="DEM" if dem else "FVM" if fvm else "FEM", runs=runs, sources=[provenance(p) for p in
                          (__file__, solver.__file__, pulse.__file__, pulse.slab.__file__)],
        reproduction=f"OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH={pythonpath} python examples/fem_sheet_pulse_partition.py"+(" --dem" if dem else " --fvm" if fvm else "")),
        indent=2, allow_nan=False))
