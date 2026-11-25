"""Transient contact heat between unequal finite layers with a resistive interface."""
import argparse
import json

import mpmath as mp
import numpy as np

from conduction_provenance import provenance
from tensorsph import heat_conduction_2d as sph


def interval_reference(times, resistance, method, include_traces=False):
    """Invert integrals of cut fluxes and optional left/right interface temperatures."""
    with mp.workdps(32):
        def transform(s, cut):
            roots = (mp.sqrt(4*s), mp.sqrt(2*s/3))
            lengths = (mp.mpf("0.4"), mp.mpf("0.6"))
            arguments = [length*root for length, root in zip(lengths, roots)]
            left = mp.tanh(arguments[0])/roots[0]
            right = mp.tanh(arguments[1])/(3*roots[1])
            impedance = resistance + left + right
            # A unit initial jump drives the flux. Integration introduces
            # a second factor 1/s. Each fixed outer wall sees a sech transfer.
            heat = 1 / (s*s*impedance)
            # Time integrals of the two material-side interface temperatures.
            if cut == 3:
                return 1/s**2 - left*heat
            if cut == 4:
                return right*heat
            if cut == 1:
                return heat
            wall_argument = arguments[0] if cut == 0 else arguments[1]
            return heat / mp.cosh(wall_argument)

        cumulative = [[float(mp.invertlaplace(lambda s: transform(s, cut), t, method=method))
                       for cut in range(5 if include_traces else 3)] for t in times[1:]]
    return np.diff(np.vstack((np.zeros_like(cumulative[0]), cumulative)), axis=0)


def measure(n, times, resistance, reference, solver="SPH"):
    left = 2*n//5
    conductivity = np.repeat([1., 3.], [left, n-left])
    capacity = np.repeat([4., 2.], [left, n-left])
    initial = np.repeat([1., 0.], [left, n-left])
    contacts = np.zeros(n-1)
    contacts[left-1] = resistance
    trace_metrics = {}
    if solver == "SPH":
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
        temperature, _, _, _, heat, means = rod.solve_transient_stepped_rod(
            np.full(n, 1./n), 1., conductivity, capacity, (1., 0.), subtime, initial,
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

    storage = np.diff(temperature, axis=0)*capacity/n
    layer_storage = np.column_stack((storage[:, :left].sum(axis=1), storage[:, left:].sum(axis=1)))
    errors = np.max(abs(measured-reference[:, :3]), axis=0)
    return dict(**trace_metrics, solver=solver, cells=n, interval_heat_J_m2=measured.tolist(), max_west_interface_east_interval_errors_J_m2=errors.tolist(),
                max_layer_balance_error_J_m2=float(abs(layer_storage + np.diff(measured, axis=1)).max()),
                first_interval_mean_interface_flux_W_m2=float(measured[0, 1]/times[1]))


if __name__ == "__main__":
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
