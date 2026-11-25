"""Verify tiny-frequency rod powers with bulk, end and split-contact storage."""
import argparse
import json

import numpy as np

from conduction_provenance import provenance


def measure(n, omega, power, fem=False, drifting=False):
    lengths = np.array([.37, .63])
    areas = np.array([.8, 1.4])
    k, c = np.array([.9, 3.7]), np.array([1.6, .85])
    widths = np.repeat(lengths/n, n)
    rate = .27+.34j
    drop_resistance = .32  # K/W over the contact area
    resistance = np.zeros(2*n-1)
    resistance[n-1] = drop_resistance*min(areas)
    # The larger (right) section owns the exposed shoulder.
    sheet, fraction, shoulder_capacity = .73, .38, .46
    trace_mass = np.array([sheet*fraction, sheet*(1-fraction)+shoulder_capacity*np.diff(areas)[0]])
    end_capacity = np.array([.3, .75])
    end_mass = areas*end_capacity
    west_contact = -power*lengths[0]/(k[0]*areas[0])
    east_contact = west_contact-drop_resistance*power
    faces_by_layer = [t-power*np.linspace(0., length, n+1)/(cond*area)
                      for t, length, cond, area in zip((0., east_contact), lengths, k, areas)]
    cells = np.concatenate([(f[:-1]+f[1:])/2 for f in faces_by_layer])
    section_power = [np.full(n+1, power), np.full(n+1, power)]
    slopes = np.zeros(2, dtype=complex)
    if drifting:
        slopes = np.array([.45-.12j, -.31+.22j])  # dP/dx, W/m
        # Use the contact as origin in both layers. The integrated Fourier
        # law gives T and the layer-specific quadratic cell means exactly.
        coordinates = (np.linspace(-lengths[0], 0., n+1), np.linspace(0., lengths[1], n+1))
        faces_by_layer, means = [], []
        for x, base, slope, ka in zip(coordinates, (west_contact, east_contact), slopes, k*areas):
            face = base-(power*x+.5*slope*x**2)/ka
            faces_by_layer.append(face)
            means.append((face[:-1]+face[1:])/2+slope*np.diff(x)**2/(12*ka))
        cells = np.concatenate(means)
        section_power = [power+slope*x for slope,x in zip(slopes, coordinates)]
    ends = np.array([faces_by_layer[0][0], faces_by_layer[-1][-1]])
    traces = np.array([west_contact, east_contact])
    mass = widths*np.repeat(areas*c, n)
    shift = (mass @ cells+end_mass @ ends+trace_mass @ traces)/(mass.sum()+end_mass.sum()+trace_mass.sum())
    cells, ends, traces = cells-shift, ends-shift, traces-shift
    end_load = end_mass*(rate+1j*omega*ends)+np.array([section_power[0][0], -section_power[-1][-1]])
    sheet_profile = np.zeros_like(resistance)
    sheet_profile[n-1] = sheet
    trace_loads = np.zeros((2, len(resistance)), dtype=complex)
    trace_loads[:, n-1] = trace_mass*(rate+1j*omega*traces)
    if fem:
        import torch
        from tensorfem import thermal

        coords = np.r_[np.linspace(0., lengths[0], n+1), np.linspace(lengths[0], sum(lengths), n+1)]
        starts = np.r_[np.arange(n), n+1+np.arange(n)]
        elements = np.column_stack((starts, starts+1))
        point_load = np.zeros(len(coords), dtype=complex)
        point_load[[0, -1, n, n+1]] = np.r_[end_load, trace_loads[:, n-1]]
        point_capacity = np.zeros(len(coords))
        point_capacity[[0, -1, n, n+1]] = np.r_[end_mass, trace_mass]
        exact_nodes = np.concatenate(faces_by_layer)-shift
        source = np.repeat(c, n)[:, None]*(rate+1j*omega*exact_nodes[elements])
        if drifting:
            # The quadratic field differs from its linear interpolant by a
            # symmetric bubble. Each endpoint hat integrates half its total.
            correction = cells-exact_nodes[elements].mean(axis=1)
            source += np.repeat(slopes/areas,n)[:, None]+1j*omega*np.repeat(c,n)[:, None]*correction[:, None]
        model = thermal.ThermalModel(torch.from_numpy(coords[:, None]), torch.from_numpy(elements),
            torch.from_numpy(np.repeat(k, n)), 1., torch.from_numpy(np.repeat(c, n)),
            torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float64),
            thickness=torch.from_numpy(np.repeat(areas, n)), heat_source=torch.from_numpy(source),
            nodal_heat=torch.from_numpy(point_load), nodal_heat_capacity=torch.from_numpy(point_capacity),
            contact_pairs=torch.tensor([[n, n+1]]), contact_conductance=1/drop_resistance)
        state = thermal.solve_harmonic_thermal(model, omega, sparse=True, separate_mean=not drifting,
                                               return_mean_rate=drifting)
        local = thermal.thermal_element_nodal_power(model, state.temperature, omega,
                    mean_temperature=state.mean_temperature, mean_heating_rate=state.mean_heating_rate).numpy()
        temperature = state.temperature.numpy()
        computed_rate = state.mean_heating_rate.numpy() if drifting else 1j*omega*state.mean_temperature.numpy()
        left = np.concatenate([p[:-1] for p in section_power])
        right = np.concatenate([p[1:] for p in section_power])
        diagnostics = {}
        if drifting:
            computed_traces = temperature[[n, n+1]]
            crossing = (computed_traces[0]-computed_traces[1])/drop_resistance
            inputs = trace_loads[:, n-1]+[-local[n-1,1]-crossing, crossing-local[n,0]]
            stored = trace_mass*(computed_rate[[n,n+1]]+1j*omega*computed_traces)
            diagnostics = dict(contact_power_error_W=float(abs(crossing-power)),
                               contact_storage_residual_W=float(max(abs(inputs-stored))))
        return dict(**diagnostics, cells_per_layer=n, omega_rad_s=omega, through_power_W=[power.real, power.imag],
                    mean_rate_error_K_s=float(max(abs(computed_rate-rate))),
                    relative_temperature_error_K=float(max(abs(temperature-exact_nodes))),
                    section_power_error_W=float(np.max(abs(local-np.column_stack((left,-right))))))
    from tensorfvm.cht import rod
    field, faces, west, east, sides, mean = rod.solve_harmonic_stepped_rod(
        widths, np.repeat(areas, n), np.repeat(k, n), np.repeat(c, n), end_load, omega,
        source=np.repeat(c, n)*(rate+1j*omega*cells)+np.repeat(slopes/areas,n), end_types=("power", "power"),
        end_areal_heat_capacity=end_capacity, shoulder_areal_heat_capacity=shoulder_capacity,
        interface_heat_capacity=sheet_profile, interface_capacity_left_fraction=fraction,
        contact_resistance=resistance, trace_power=trace_loads, flux_order=2,
        return_traces=True, separate_mean=not drifting, return_mean_rate=drifting)
    diagnostics = {}
    if drifting:
        computed_traces = np.asarray(sides)[:, n-1]
        contact_power = (computed_traces[0]-computed_traces[1])/drop_resistance
        trace_input = trace_loads[:, n-1]+[east[n-1]-contact_power, contact_power-west[n]]
        stored = trace_mass*(mean+1j*omega*computed_traces)
        diagnostics = dict(contact_power_error_W=float(abs(contact_power-power)),
                           contact_storage_residual_W=float(max(abs(trace_input-stored))))
    return dict(**diagnostics, cells_per_layer=n, omega_rad_s=omega, through_power_W=[power.real, power.imag],
                mean_rate_error_K_s=float(abs((mean if drifting else 1j*omega*mean)-rate)),
                relative_temperature_error_K=float(max(max(abs(field-cells)), max(abs(faces[[0,-1]]-ends)),
                                                       max(abs(np.asarray(sides)[:, n-1]-traces)))),
                section_power_error_W=float(max(max(abs(west-np.concatenate([p[:-1] for p in section_power]))),
                                                max(abs(east-np.concatenate([p[1:] for p in section_power]))))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drifting", action="store_true", help="verify zero and finite frequencies with varying section powers")
    parser.add_argument("--fem", action="store_true", help="verify the sparse consistent-mass LINE2 solver")
    args = parser.parse_args()
    fem, drifting = args.fem, args.drifting
    if fem:
        from tensorfem import thermal as module
        path, option = "../TensorFEM/src", " --fem"+(" --drifting" if drifting else "")
    else:
        from tensorfvm.cht import rod as module
        path, option = "../TensorFVM/src", " --drifting" if drifting else ""
    frequencies = (0., 1e-12, 1e-4, 1.) if drifting else (1., 1e-4, 1e-8, 1e-12)
    rows = [measure(n, omega, power, fem, drifting) for n in (16, 32, 64) for omega in frequencies
            for power in (0j, .35-.21j)]
    previous = {}
    for row in rows:
        if fem and drifting:
            # LINE2 represents the manufactured quadratic only approximately.
            # Check spatial convergence; tiny-frequency power errors can already
            # be at roundoff, while storage balance remains a discrete identity.
            key = row["omega_rad_s"], tuple(row["through_power_W"])
            errors = np.array([row["relative_temperature_error_K"], row["section_power_error_W"],
                               row["contact_power_error_W"]])
            orders = np.log2(previous[key]/errors) if key in previous else None
            row["temperature_section_contact_orders"] = orders.tolist() if orders is not None else []
            row["passed"] = max(row["mean_rate_error_K_s"], row["contact_storage_residual_W"]) < 1e-10
            if orders is not None:
                row["passed"] &= bool(np.all((orders > 1.9) | (errors < 1e-10)))
            if row["cells_per_layer"] == 64:
                row["passed"] &= bool(max(errors) < 1e-6)
            previous[key] = errors
            continue
        row["passed"] = max(row["mean_rate_error_K_s"], row["relative_temperature_error_K"],
                            row["section_power_error_W"]) < 1e-9
        if drifting:
            row["passed"] &= max(row["contact_power_error_W"], row["contact_storage_residual_W"]) < 1e-9
    print(json.dumps(dict(reference=("Common mean heating and piecewise quadratic temperature with varying axial power and a resistive jump" if drifting else "Uniform temperature pole plus piecewise linear temperature with constant axial power and a resistive jump"),
                          reproduction=f"OPENBLAS_NUM_THREADS=1 PYTHONPATH={path} python examples/fvm_harmonic_separated_mean.py{option}",
                          runs=rows, sources=[provenance(f) for f in (__file__, module.__file__)]), indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
