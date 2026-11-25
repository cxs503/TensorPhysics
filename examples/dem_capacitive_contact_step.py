"""Finite-layer contact transient with a heat-storing sheet between two contact resistances."""
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
    heats = np.array(heats)
    stored = SHEET_CAPACITY*np.diff(np.r_[initial_sheet, temperatures])
    residual = stored - heats[:, 0] + heats[:, 1]
    return dict(cells=n, max_left_right_heat_errors_J_m2=abs(heats-exact_heat).max(axis=0).tolist(),
                max_sheet_temperature_error_K=float(abs(temperatures-exact_temperature).max()),
                max_sheet_balance_residual_J_m2=float(abs(residual).max()),
                interval_left_right_heat_J_m2=heats.tolist(), sheet_endpoint_temperature_K=temperatures)


if __name__ == "__main__":
    from tensordem.thermal import contact_chain

    times = np.r_[0., .00025*2.**np.arange(12)]
    studies = []
    for initial in (0., 1.):
        heat, temperature = reference(times, initial, "dehoog")
        audit_heat, audit_temperature = reference(times, initial, "talbot")
        rows = []
        previous = None
        for n in (40, 80, 160, 320):
            row = measure(n, times, initial, heat, temperature)
            errors = np.array(row["max_left_right_heat_errors_J_m2"] + [row["max_sheet_temperature_error_K"]])
            orders = [] if previous is None else np.log2(previous/errors).tolist()
            row.update(orders=orders, passed=bool(row["max_sheet_balance_residual_J_m2"] < 1e-10
                       and (not orders or min(orders) > 0.) and (n != 320 or
                           (min(orders) > 1.85 and max(errors[:2]) < 5e-5 and errors[2] < 5e-4))))
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
    print(json.dumps(dict(times_s=times.tolist(), resistance_m2_K_W=[LEFT_R, RIGHT_R],
        sheet_capacity_J_m2_K=SHEET_CAPACITY, studies=studies, passed=passed,
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorDEM/src python examples/dem_capacitive_contact_step.py",
        sources=[provenance(__file__), provenance(contact_chain.__file__)]), indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
