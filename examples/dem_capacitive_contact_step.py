"""Finite-layer contact transient with a heat-storing sheet between two contact resistances."""
import argparse
import json

import mpmath as mp
import numpy as np

from conduction_provenance import provenance

LEFT_R, RIGHT_R = .04, .12
SHEET_CAPACITY = .18


def reference(times, initial_sheet, method):
    # Unit area. Positive fluxes point from the hot layer towards the cold one.
    with mp.workdps(32):
        def transform(s, channel):
            a, b = mp.sqrt(4*s), mp.sqrt(2*s/3)
            left = LEFT_R + mp.tanh(.4*a)/a
            right = RIGHT_R + mp.tanh(.6*b)/(3*b)
            sheet = (1/(s*left) + SHEET_CAPACITY*initial_sheet)/(SHEET_CAPACITY*s + 1/left + 1/right)
            if channel == 0:
                return (1/s-sheet)/(s*left)
            if channel == 1:
                return sheet/(s*right)
            return sheet

        values = np.array([[float(mp.invertlaplace(lambda s: transform(s, channel), t, method=method))
                            for channel in range(3)] for t in times[1:]])
    heat = np.diff(np.vstack(([0., 0.], values[:, :2])), axis=0)
    return heat, values[:, 2]


def measure(n, times, initial_sheet, exact_heat, exact_temperature):
    from tensordem.thermal import contact_chain

    left = 2*n//5
    contacts = np.zeros(n-1)
    contacts[left-1] = LEFT_R + RIGHT_R
    storage = np.zeros_like(contacts)
    storage[left-1] = SHEET_CAPACITY
    hot = np.arange(n) < left
    links, capacity = contact_chain.cell_centered_slab_network(
        np.full(n, 1./n), np.where(hot, 1., 3.), np.where(hot, 4., 2.),
        contact_resistance=contacts, interface_heat_capacity=storage,
        left_resistance_fraction=LEFT_R/(LEFT_R + RIGHT_R), resolve_contacts=True)
    # Ordered nodes: west reservoir, left cells, left trace, sheet, right trace, right cells, east reservoir.
    sheet = left+2
    state = np.zeros_like(capacity)
    state[:sheet] = 1.
    state[sheet] = initial_sheet
    temperatures, heats = [], []
    for dt in np.diff(times):
        state, heat = contact_chain.affine_contact_interval(
            links, capacity, dt, state, t_left=(1., 1.), t_right=(0., 0.), allow_massless=True)
        heats.append(heat[sheet-1:sheet+1])
        temperatures.append(state[sheet])
    return summarize(n, initial_sheet, heats, temperatures, exact_heat, exact_temperature)


def measure_fem(n, times, initial_sheet, exact_heat, exact_temperature):
    import torch
    from tensorfem import thermal

    left = 2*n//5
    grid = torch.linspace(0., 1., n+1, dtype=torch.float64)
    # Distinct coincident nodes represent the two material traces and the sheet.
    nodes = torch.cat((grid[:left+1], grid[left:], grid[left:left+1]))
    start = torch.cat((torch.arange(left), torch.arange(left+1, n+1)))
    sheet = n+2
    lump = torch.zeros_like(nodes)
    lump[sheet] = SHEET_CAPACITY
    model = thermal.ThermalModel(
        nodes[:,None], torch.stack((start, start+1), dim=1),
        grid.new_tensor([1.]*left + [3.]*(n-left)), 1., grid.new_tensor([4.]*left + [2.]*(n-left)),
        torch.tensor([0, n+1]), grid.new_tensor([1., 0.]), nodal_heat_capacity=lump,
        contact_pairs=torch.tensor([[left, sheet], [sheet, left+1]]),
        contact_conductance=1./grid.new_tensor([LEFT_R, RIGHT_R]))
    initial = torch.zeros_like(nodes)
    initial[:left+1] = 1.
    initial[sheet] = initial_sheet
    count = n//5
    time = np.r_[times[0], np.concatenate([np.linspace(a,b,count+1)[1:] for a,b in zip(times[:-1], times[1:])])]
    result = thermal.solve_transient_thermal(model, torch.from_numpy(time), initial,
                                             sparse=True, method="sdirk2")
    local = thermal.thermal_element_nodal_heat_steps(model, result).numpy()
    heats = np.column_stack((-local[:,left-1,1], local[:,left,0])).reshape(-1,count,2).sum(axis=1)
    temperatures = result.temperature[count::count,sheet].numpy()
    row = summarize(n, initial_sheet, heats, temperatures, exact_heat, exact_temperature)
    trace = result.interval_temperature[:, [left, sheet, left+1]].numpy()
    link_power = -np.diff(trace, axis=1)/np.array([LEFT_R, RIGHT_R])
    link_heat = (link_power*np.diff(time)[:,None]).reshape(-1,count,2).sum(axis=1)
    row["max_element_link_heat_discrepancy_J_m2"] = float(abs(heats-link_heat).max())
    row["substeps_per_interval"] = count
    return row


def sheet_cell_arrays(n, initial_sheet, width_scale):
    left = 2*n//5
    # One well-mixed control volume represents the sheet. Its numerical width
    # cancels from its capacitance and both half-cell resistances. The extra
    # series resistance on the right gives the specified asymmetric contact.
    width = width_scale/n
    half_resistance = min(LEFT_R, RIGHT_R)
    widths = np.r_[np.full(left, 1./n), width, np.full(n-left, 1./n)]
    conductivity = np.r_[np.ones(left), width/(2*half_resistance), np.full(n-left, 3.)]
    capacity = np.r_[np.full(left, 4.), SHEET_CAPACITY/width, np.full(n-left, 2.)]
    contacts = np.zeros(n)
    contacts[left-1:left+1] = np.array([LEFT_R, RIGHT_R]) - half_resistance
    initial = np.r_[np.ones(left), initial_sheet, np.zeros(n-left)]
    return left, widths, conductivity, capacity, contacts, initial


def measure_sph(n, times, initial_sheet, exact_heat, exact_temperature):
    from tensorsph import heat_conduction_2d as sph

    left, widths, k, capacity, contacts, initial = sheet_cell_arrays(n, initial_sheet, 1.)
    # The compact kernel reaches adjacent columns only, so every heat path
    # from the left material to the right passes through the sheet column.
    state, heat = sph.solve_transient_layered_slab(
        initial, times, 1./n, (1., 0.), conductivity=k, volumetric_heat_capacity=capacity,
        contact_resistance_x=contacts, h_factor=.9, method="exponential", return_x_heat=True)
    state, heat = state.numpy(), heat.numpy()
    row = summarize(n, initial_sheet, heat[:,left:left+2], state[1:,left], exact_heat, exact_temperature)
    stored = np.diff(state, axis=0)*(capacity*widths)
    material_storage = np.column_stack((stored[:,:left].sum(axis=1), stored[:,left+1:].sum(axis=1)))
    material_net_heat = heat[:,[0, left+1]] - heat[:,[left, -1]]
    row["max_material_balance_residual_J_m2"] = float(abs(material_storage-material_net_heat).max())
    return row


def measure_fvm(n, times, initial_sheet, exact_heat, exact_temperature, *, width_scale=1.):
    from tensorfvm.cht import rod

    left, widths, conductivity, capacity, contacts, initial = sheet_cell_arrays(n, initial_sheet, width_scale)
    count = n//5
    time = np.r_[times[0], np.concatenate([np.linspace(a,b,count+1)[1:] for a,b in zip(times[:-1], times[1:])])]
    field, _, _, _, ledger = rod.solve_transient_stepped_rod(
        widths, 1., conductivity, capacity, (1., 0.), time, initial,
        contact_resistance=contacts, flux_order=1, method="sdirk2", return_step_heat=True)
    # Read each side from its adjoining material volume, independently of the sheet storage.
    material_heat = np.column_stack((ledger["right"][:,left-1], ledger["left"][:,left+1]))
    sheet_heat = np.column_stack((ledger["left"][:,left], ledger["right"][:,left]))
    heats = material_heat.reshape(-1,count,2).sum(axis=1)
    discrepancy = abs((material_heat-sheet_heat).reshape(-1,count,2).sum(axis=1)).max()
    row = summarize(n, initial_sheet, heats, field[count::count,left], exact_heat, exact_temperature)
    row["max_shared_face_heat_discrepancy_J_m2"] = float(discrepancy)
    row["substeps_per_interval"] = count
    return row


def summarize(n, initial_sheet, heats, temperatures, exact_heat, exact_temperature):
    heats = np.array(heats)
    stored = SHEET_CAPACITY*np.diff(np.r_[initial_sheet, temperatures])
    residual = stored - heats[:, 0] + heats[:, 1]
    return dict(cells=n, max_left_right_heat_errors_J_m2=abs(heats-exact_heat).max(axis=0).tolist(),
                max_sheet_temperature_error_K=float(abs(temperatures-exact_temperature).max()),
                max_sheet_balance_residual_J_m2=float(abs(residual).max()),
                interval_left_right_heat_J_m2=heats.tolist(), sheet_endpoint_temperature_K=np.asarray(temperatures).tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=("DEM", "FEM", "FVM", "SPH"), default="DEM")
    solver = parser.parse_args().solver
    if solver == "FEM":
        from tensorfem import thermal as implementation
        solve = measure_fem
    elif solver == "FVM":
        from tensorfvm.cht import rod as implementation
        solve = measure_fvm
    elif solver == "SPH":
        from tensorsph import heat_conduction_2d as implementation
        solve = measure_sph
    else:
        from tensordem.thermal import contact_chain as implementation
        solve = measure

    times = np.r_[0., .00025*2.**np.arange(12)]
    studies = []
    for initial in (0., 1.):
        heat, temperature = reference(times, initial, "dehoog")
        audit_heat, audit_temperature = reference(times, initial, "talbot")
        rows = []
        previous = None
        for n in (40, 80, 160, 320):
            row = solve(n, times, initial, heat, temperature)
            errors = np.array(row["max_left_right_heat_errors_J_m2"] + [row["max_sheet_temperature_error_K"]])
            orders = [] if previous is None else np.log2(previous/errors).tolist()
            row.update(orders=orders, passed=bool(row["max_sheet_balance_residual_J_m2"] < 1e-10
                       and (not orders or min(orders) > 0.) and (n != 320 or
                           (min(orders) > 1.85 and max(errors[:2]) < 5e-5 and errors[2] < 5e-4))))
            if solver == "SPH":
                row["passed"] &= row["max_material_balance_residual_J_m2"] < 1e-10
            if solver == "FEM":
                row["passed"] &= row["max_element_link_heat_discrepancy_J_m2"] < 1e-10
            if solver == "FVM":
                row["passed"] &= row["max_shared_face_heat_discrepancy_J_m2"] < 1e-10
                if n == 80:
                    control = measure_fvm(n, times, initial, heat, temperature, width_scale=4.)
                    for key, label in (("interval_left_right_heat_J_m2", "width_control_heat_difference_J_m2"),
                                       ("sheet_endpoint_temperature_K", "width_control_temperature_difference_K")):
                        row[label] = float(abs(np.asarray(row[key])-control[key]).max())
                        row["passed"] &= row[label] < 1e-10
            rows.append(row)
            previous = errors
        q_diff = float(abs(heat-audit_heat).max())
        t_diff = float(abs(temperature-audit_temperature).max())
        reference_balance = float(abs(SHEET_CAPACITY*np.diff(np.r_[initial, temperature])
                                      - heat[:, 0] + heat[:, 1]).max())
        studies.append(dict(reference_sheet_balance_residual_J_m2=reference_balance, initial_sheet_temperature_K=initial, reference_interval_left_right_heat_J_m2=heat.tolist(),
                            reference_sheet_endpoint_temperature_K=temperature.tolist(), reference_heat_discrepancy_J_m2=q_diff,
                            reference_temperature_discrepancy_K=t_diff, runs=rows,
                            passed=q_diff < 1e-12 and t_diff < 1e-12 and reference_balance < 1e-12 and all(row["passed"] for row in rows)))
    passed = all(study["passed"] for study in studies)
    print(json.dumps(dict(solver=solver, times_s=times.tolist(), resistance_m2_K_W=[LEFT_R, RIGHT_R],
        sheet_capacity_J_m2_K=SHEET_CAPACITY, studies=studies, passed=passed,
        reproduction=f"OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../Tensor{solver}/src python examples/dem_capacitive_contact_step.py --solver {solver}",
        sources=[provenance(__file__), provenance(implementation.__file__)]), indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
