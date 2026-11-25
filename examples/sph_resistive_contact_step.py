"""Transient contact heat between unequal finite layers with a resistive interface."""
import argparse
import json

import mpmath as mp
import numpy as np

from conduction_provenance import provenance


def interval_reference(times, resistance, method, include_traces=False, surface_capacities=(0., 0.), return_trace_changes=False):
    """Invert integrals of cut fluxes and optional left/right interface temperatures."""
    if return_trace_changes and not include_traces:
        raise ValueError("Trace changes require include_traces=True")
    with mp.workdps(32):
        def transform(s, cut):
            roots = (mp.sqrt(4*s), mp.sqrt(2*s/3))
            lengths = (mp.mpf("0.4"), mp.mpf("0.6"))
            arguments = [length*root for length, root in zip(lengths, roots)]
            left = mp.tanh(arguments[0])/roots[0]
            right = mp.tanh(arguments[1])/(3*roots[1])
            # A trace initially at its adjoining material temperature stores
            # heat in parallel with the finite layer's thermal admittance.
            factors = (1+s*surface_capacities[0]*left, 1+s*surface_capacities[1]*right)
            left, right = left/factors[0], right/factors[1]
            impedance = resistance + left + right
            # A unit initial jump drives the flux. Integration introduces
            # a second factor 1/s. Each fixed outer wall sees a sech transfer.
            heat = 1 / (s*s*impedance)
            # Time integrals of the two material-side interface temperatures.
            if cut in (3, 5):
                trace = 1/s**2 - left*heat
                return trace if cut == 3 else s*trace
            if cut in (4, 6):
                trace = right*heat
                return trace if cut == 4 else s*trace
            if cut == 1:
                return heat
            wall = 0 if cut == 0 else 1
            return heat / (mp.cosh(arguments[wall])*factors[wall])

        channels = 7 if return_trace_changes else (5 if include_traces else 3)
        cumulative = [[float(mp.invertlaplace(lambda s: transform(s, cut), t, method=method))
                       for cut in range(channels)] for t in times[1:]]
    initial = np.zeros(channels)
    if return_trace_changes:
        initial[5] = 1.
    return np.diff(np.vstack((initial, cumulative)), axis=0)


def measure(n, times, resistance, reference, solver="SPH", surface_capacities=(0., 0.)):
    left = 2*n//5
    conductivity = np.repeat([1., 3.], [left, n-left])
    capacity = np.repeat([4., 2.], [left, n-left])
    initial = np.repeat([1., 0.], [left, n-left])
    contacts = np.zeros(n-1)
    contacts[left-1] = resistance
    trace_metrics = {}
    surface_storage = 0.
    if solver == "SPH":
        if any(surface_capacities):
            raise ValueError("independent trace capacities require the FVM rod")
        from tensorsph import heat_conduction_2d as sph

        temperature, heat = sph.solve_transient_layered_slab(
            initial, times, 1./n, (1., 0.), conductivity=conductivity,
            volumetric_heat_capacity=capacity, contact_resistance_x=contacts,
            method="exponential", h_factor=.9, return_x_heat=True)
        measured = heat.numpy()[:, [0, left, n]]
        temperature = temperature.numpy()
    else:
        from tensorfvm.cht import rod

        # Refine each interval along with the spatial mesh to limit SDIRK2 error.
        count = n // 5
        subtime = np.r_[times[0], np.concatenate([np.linspace(a, b, count+1)[1:]
                        for a, b in zip(times[:-1], times[1:])])]
        face = left-1
        surface = np.zeros(n-1)
        total = sum(surface_capacities)
        surface[face] = total
        temperature, _, _, _, endpoints, heat, means = rod.solve_transient_stepped_rod(
            np.full(n, 1./n), 1., conductivity, capacity, (1., 0.), subtime, initial,
            interface_heat_capacity=surface, interface_capacity_left_fraction=surface_capacities[0]/total if total else .5,
            initial_trace_temperatures=(1., 0.) if total else (None, None), return_traces=True,
            contact_resistance=contacts, method="sdirk2", return_step_heat=True, return_interval_temperatures=True)
        cuts = np.column_stack((heat["left"][:, 0], heat["contact"][:, left-1], heat["right"][:, -1]))
        measured = cuts.reshape(len(times)-1, count, 3).sum(axis=1)
        temperature = temperature[::count]
        trace = np.column_stack((means["trace_left"][:, left-1], means["trace_right"][:, left-1]))
        dt = np.diff(times)
        integrated = (trace*np.diff(subtime)[:, None]).reshape(len(dt), count, 2).sum(axis=1)
        trace = integrated / dt[:, None]
        exact_trace = reference[:, 3:5] / dt[:, None]
        jump = trace[:, 0] - trace[:, 1]
        trace_metrics = dict(first_interval_left_right_K=trace[0].tolist(),
                             max_left_right_mean_error_K=np.max(abs(trace-exact_trace), axis=0).tolist(),
                             max_contact_law_residual_K=float(max(abs(jump-resistance*measured[:, 1]/dt))))

        if total:
            endpoint_trace = np.column_stack([side[::count, face] for side in endpoints])
            surface_storage = np.diff(endpoint_trace, axis=0)*surface_capacities
            material_heat = np.column_stack((heat["right"][:, face], heat["left"][:, left]))
            material_heat = material_heat.reshape(len(dt), count, 2).sum(axis=1)
            net = np.column_stack((material_heat[:, 0]-measured[:, 1], measured[:, 1]-material_heat[:, 1]))
            trace_metrics.update(interval_trace_storage_J_m2=surface_storage.tolist(),
                                 interval_material_heat_J_m2=material_heat.tolist(),
                                 interval_mean_trace_K=trace.tolist(),
                                 max_trace_balance_residual_J_m2=abs(surface_storage-net).max(0).tolist(),
                                 max_trace_storage_errors_J_m2=abs(surface_storage-reference[:, 5:7]*surface_capacities).max(0).tolist())

    storage = np.diff(temperature, axis=0)*capacity/n
    layer_storage = np.column_stack((storage[:, :left].sum(axis=1), storage[:, left:].sum(axis=1)))
    errors = np.max(abs(measured-reference[:, :3]), axis=0)
    return dict(**trace_metrics, solver=solver, cells=n, interval_heat_J_m2=measured.tolist(), max_west_interface_east_interval_errors_J_m2=errors.tolist(),
                max_layer_balance_error_J_m2=float(abs(layer_storage + surface_storage + np.diff(measured, axis=1)).max()),
                first_interval_mean_interface_flux_W_m2=float(measured[0, 1]/times[1]))


if __name__ == "__main__":
    from tensorsph import heat_conduction_2d as sph

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-fvm", action="store_true", help="also run the SDIRK2 finite-volume rod")
    with_fvm = parser.parse_args().with_fvm
    times = np.r_[0., .00025*2.**np.arange(12)]
    studies = []
    for resistance in (.05, .3):
        reference = interval_reference(times, resistance, "talbot", with_fvm)
        audit = interval_reference(times, resistance, "dehoog", with_fvm)
        discrepancy = float(abs(reference[:, :3]-audit[:, :3]).max())
        rows = []
        for solver in (("SPH", "FVM") if with_fvm else ("SPH",)):
            previous = None
            previous_trace = None
            for n in (40, 80, 160, 320):
                row = measure(n, times, resistance, reference, solver)
                errors = np.array(row["max_west_interface_east_interval_errors_J_m2"])
                orders = [] if previous is None else np.log2(previous/errors).tolist()
                row.update(west_interface_east_orders=orders,
                           passed=bool(row["max_layer_balance_error_J_m2"] < 1e-10
                                       and (not orders or min(orders) > 0.)
                                       and (n != 320 or (min(orders) > 1.85 and max(errors) < 5e-5))))
                if solver == "FVM":
                    counterpart = next(r for r in rows if r["solver"] == "SPH" and r["cells"] == n)
                    interface_heat = np.asarray(row["interval_heat_J_m2"])[:, 1]
                    sph_heat = np.asarray(counterpart["interval_heat_J_m2"])[:, 1]
                    difference = float(np.max(abs(interface_heat - sph_heat)))
                    row["max_interface_difference_from_sph_J_m2"] = difference
                    row["passed"] = row["passed"] and (n != 320 or difference < 1e-7)
                    trace_errors = np.array(row["max_left_right_mean_error_K"])
                    trace_orders = [] if previous_trace is None else np.log2(previous_trace/trace_errors).tolist()
                    row["left_right_trace_orders"] = trace_orders
                    row["passed"] = bool(row["passed"] and (row["max_contact_law_residual_K"] < 1e-11
                                      and (not trace_orders or min(trace_orders) > 0.)
                                      and (n != 320 or (min(trace_orders) > 1.85 and max(trace_errors) < .004))))
                    previous_trace = trace_errors
                    row["substeps_per_interval"] = n // 5
                rows.append(row)
                previous = errors
        trace_summary = {}
        trace_discrepancy = 0.
        if with_fvm:
            exact_traces = reference[:, 3:] / np.diff(times)[:, None]
            trace_discrepancy = float(np.max(abs(reference[:, 3:]-audit[:, 3:]) / np.diff(times)[:, None]))
            trace_summary = dict(reference_interval_left_right_K=exact_traces.tolist(),
                                 trace_reference_discrepancy_K=trace_discrepancy)
        studies.append(dict(**trace_summary, contact_resistance_m2_K_W=resistance, reference_discrepancy_J_m2=discrepancy,
                            initial_interface_flux_W_m2=1./resistance,
                            reference_first_interval_mean_flux_W_m2=reference[0, 1]/times[1],
                            reference_interval_heat_J_m2=reference[:, :3].tolist(), runs=rows,
                            passed=discrepancy < 1e-12 and trace_discrepancy < 1e-12
                                   and all(row["passed"] for row in rows)))
    passed = all(study["passed"] for study in studies)
    sources = [provenance(__file__), provenance(sph.__file__)]
    paths, flags = "../TensorSPH/src", ""
    if with_fvm:
        from tensorfvm.cht import rod

        sources.append(provenance(rod.__file__))
        paths += ":../TensorFVM/src"
        flags = " --with-fvm"
    print(json.dumps(dict(reference="Finite-layer Laplace impedances in series; Talbot and de Hoog inversions",
        cut_order=["west", "interface", "east"], times_s=times.tolist(), studies=studies, passed=passed,
        reproduction=f"OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH={paths} python examples/sph_resistive_contact_step.py{flags}",
        sources=sources), indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
