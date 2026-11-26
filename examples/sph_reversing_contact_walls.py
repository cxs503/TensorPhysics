"""Verify reversing conductive heat flow with independently driven Dirichlet walls."""
import argparse
import json

import numpy as np

from conduction_provenance import provenance


def contact_reversal_time(times, heat):
    """Locate a single east-to-west crossing using three interval heat integrals."""
    mean_flux = heat/np.diff(times)
    crossings = np.flatnonzero((mean_flux[:-1] > 0) & (mean_flux[1:] <= 0))
    if len(crossings) != 1 or len(heat) < 3:
        raise ValueError("expected a single positive-to-negative crossing and at least three intervals")
    return decreasing_root(times, mean_flux, crossings[0])


def decreasing_root(times, mean_flux, i):
    """Fit a decreasing zero across the adjacent means at i and i+1."""
    start = max(0, i-1)
    # Fit the interval averages of 1, u and u² in local time coordinates.
    # This preserves each measured heat even when a step straddles the zero.
    origin = times[i+1]
    scale = times[i+1]-times[i]
    u = (times[start:start+4]-origin)/scale
    a, b = u[:-1], u[1:]
    moments = np.column_stack((np.ones(3), (a+b)/2, (a*a+a*b+b*b)/3))
    coefficients = np.linalg.solve(moments, mean_flux[start:start+3])
    roots = np.polynomial.polynomial.polyroots(coefficients)
    candidates = [r.real for r in roots if np.isreal(r) and u[0] <= r.real <= u[-1]
                  and coefficients[1]+2*coefficients[2]*r.real < 0]
    if len(candidates) != 1:
        raise ValueError("no unique decreasing zero in the local reconstruction")
    return float(origin+scale*candidates[0])


def slab_geometry(n):
    cut = 2*n//5
    k = np.repeat([1., 3.], [cut, n-cut])
    capacity = np.repeat([4., 2.], [cut, n-cut])
    resistance = np.zeros(n-1)
    resistance[cut-1] = .3
    # Accumulate thermal resistance from the west wall to each particle.
    halves = .5/(n*k)
    path = np.r_[0., np.cumsum(2*halves[:-1]+resistance)] + halves
    total = path[-1]+halves[-1]
    weight = path/total
    return cut, k, capacity, resistance, total, weight


def measure(n, steps, method, solver="sph", *, grading=0.):
    cut, k, capacity, resistance, total, weight = slab_geometry(n)
    # A linear temperature in resistance coordinates gives the same flux
    # across both materials and the finite jump. Generation supplies storage.
    def west(t):
        return 1.-2*t*t
    def east(t):
        return -.5+t
    def source(t):
        return capacity*((1-weight)*(-4*t)+weight)
    times = np.linspace(0, 1., steps+1)
    # A smooth mapping preserves nested report times under step doubling.
    times += grading*times*(1-times)
    exact = (1-weight)*west(times[:, None])+weight*east(times[:, None])
    diagnostics = {}
    if solver == "fvm":
        from tensorfvm.cht import rod

        field, _, _, _, traces, ledger, means = rod.solve_transient_stepped_rod(
            np.full(n, 1/n), 1., k, capacity, lambda t: (west(t), east(t)), times, exact[0],
            source=source, contact_resistance=resistance, method=method, flux_order=2, return_step_heat=True,
            return_traces=True, return_interval_temperatures=True)
        cuts = np.column_stack((ledger["left"][:, 0], ledger["contact"][:, cut-1], ledger["right"][:, -1]))
        face = cut-1
        left_path = cut/(n*k[0])
        weights = np.array([left_path, left_path+resistance[face]])/total
        endpoint = np.column_stack([side[:, face] for side in traces])
        expected = west(times[:, None])*(1-weights) + east(times[:, None])*weights
        # Analytical interval means of the quadratic and linear wall histories.
        a, b = times[:-1, None], times[1:, None]
        west_mean = 1-2*(a*a+a*b+b*b)/3
        east_mean = -.5+(a+b)/2
        exact_mean = west_mean*(1-weights)+east_mean*weights
        trace_mean = np.column_stack([means[key][:, face] for key in ("trace_left", "trace_right")])
        mean_flux = cuts[:, 1]/np.diff(times)
        exact_flux = ((west_mean-east_mean)/total).ravel()
        jump = trace_mean[:, 0]-trace_mean[:, 1]
        diagnostics.update(
            endpoint_trace_errors_left_right_K=np.max(abs(endpoint-expected), axis=0).tolist(),
            interval_trace_errors_left_right_K=np.max(abs(trace_mean-exact_mean), axis=0).tolist(),
            interval_jump_error_K=float(max(abs(jump-(exact_mean[:, 0]-exact_mean[:, 1])))),
            interval_flux_error_W_m2=float(max(abs(mean_flux-exact_flux))),
            interval_contact_law_residual_K=float(max(abs(jump-resistance[face]*mean_flux))))
        balance = np.diff(field, axis=0)*capacity/n + ledger["right"]-ledger["left"]-ledger["source"]
        diagnostics["cell_balance_J_m2"] = float(np.max(abs(balance)))
    else:
        from tensorsph import heat_conduction_2d as sph

        particles, heats = sph.solve_transient_layered_slab(
            exact[0], times, 1/n, (west, east), conductivity=k, volumetric_heat_capacity=capacity,
            heat_source=source, contact_resistance_x=resistance, h_factor=.9,
            method=method, return_x_heat=True)
        field = particles.numpy()
        cuts = heats.numpy()[:, [0, cut, n]]
    # Positive heat points east. Integrate q=(west-east)/total analytically.
    primitive = (1.5*times-times**2/2-2*times**3/3)/total
    cumulative = np.cumsum(cuts, axis=0)
    step_error = np.max(abs(cuts-np.diff(primitive)[:, None]))
    t_error = float(np.max(abs(field-exact)))
    h_errors = np.max(abs(cumulative-primitive[1:, None]), axis=0)
    exact_sign = np.sign(np.diff(primitive))[:, None]
    reversed_flow = bool(np.all(np.sign(cuts) == exact_sign))
    reversal = contact_reversal_time(times, cuts[:, 1])
    exact_reversal = (np.sqrt(13.)-1)/4
    i = np.searchsorted(times, exact_reversal)-1
    return dict(crossing_interval_s=times[i:i+2].tolist(),
                reconstruction_only_error_s=abs(contact_reversal_time(times, np.diff(primitive))-exact_reversal),
                contact_reversal_s=reversal, exact_reversal_s=exact_reversal,
                contact_reversal_error_s=abs(reversal-exact_reversal), **diagnostics, solver=solver, columns=n, steps=steps, method=method, temperature_error_K=t_error,
                west_contact_east_heat_errors_J_m2=h_errors.tolist(), max_step_heat_error_J_m2=float(step_error),
                reversed_flow=reversed_flow)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=("sph", "fvm"), default="sph")
    solver = parser.parse_args().solver
    if solver == "fvm":
        from tensorfvm.cht import rod as implementation
        methods = ("theta", "sdirk2")
        repo = "TensorFVM"
    else:
        from tensorsph import heat_conduction_2d as implementation
        methods = ("theta", "sdirk2", "exponential_quadratic")
        repo = "TensorSPH"
    rows = []
    for method in methods:
        previous = None
        previous_interface = None
        previous_reversal = None
        for steps in (8, 16, 32, 64):
            row = measure(40, steps, method, solver)
            errors = np.r_[row["temperature_error_K"], row["west_contact_east_heat_errors_J_m2"]]
            orders = None
            passed = row["reversed_flow"] and row.get("cell_balance_J_m2", 0.) < 1e-10
            if method == "exponential_quadratic":
                passed &= max(errors) < 1e-10 and row["max_step_heat_error_J_m2"] < 1e-10
            else:
                if method == "theta":
                    # The trapezoidal rule integrates the linear storage load
                    # exactly. Its endpoint temperature error is roundoff-sized.
                    passed &= errors[0] < 1e-10
                if previous is not None:
                    heat_orders = np.log2(previous[1:]/errors[1:]).tolist()
                    t_order = np.log2(previous[0]/errors[0]) if method == "sdirk2" else None
                    orders = [t_order, *heat_orders]
                    if steps == 64:
                        passed &= max(errors) < 1e-4 and min(heat_orders) > 1.8
                        passed &= row["max_step_heat_error_J_m2"] < 1e-5
                        passed &= t_order is None or t_order > 1.8
            if solver == "fvm":
                interface_errors = np.r_[row["interval_trace_errors_left_right_K"],
                                         row["interval_jump_error_K"], row["interval_flux_error_W_m2"]]
                passed &= row["interval_contact_law_residual_K"] < 1e-10
                endpoint_limit = 1e-10 if method == "theta" else 2e-5
                if previous_interface is not None:
                    row["interval_orders_left_right_jump_flux"] = np.log2(previous_interface/interface_errors).tolist()
                if steps == 64:
                    passed &= max(row["endpoint_trace_errors_left_right_K"]) < endpoint_limit
                    passed &= max(interface_errors) < 1e-4 and min(row["interval_orders_left_right_jump_flux"]) > 1.8
                previous_interface = interface_errors
            reversal_error = row["contact_reversal_error_s"]
            if method == "exponential_quadratic":
                passed &= reversal_error < 1e-10
            elif previous_reversal is not None:
                row["reversal_order"] = np.log2(previous_reversal/reversal_error)
                if steps == 64:
                    passed &= row["reversal_order"] > 1.8 and reversal_error < 1e-4
            previous_reversal = reversal_error
            row.update(orders=orders, passed=bool(passed))
            rows.append(row)
            previous = errors
    passed = all(row["passed"] for row in rows)
    print(json.dumps(dict(passed=passed, runs=rows, sources=[provenance(p) for p in (__file__, implementation.__file__)],
        reproduction=f"OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../{repo}/src python examples/sph_reversing_contact_walls.py --solver {solver}"),
        indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
